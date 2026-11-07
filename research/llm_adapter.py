"""LLM API adapter of the ReAct controller (P2 integration; no paid call is enabled by default).

An API LLM spec names the provider, the model and the call parameters, e.g.
  {"kind": "api", "provider": "anthropic", "model": "claude-opus-5-5", "max_tokens": 4096,
   "effort": "low", "request_timeout_s": 60, "max_retries": 2, "retry_backoff_s": 2.0}
  {"kind": "api", "provider": "openai", "model": "gpt-6.1-terra", "max_tokens": 4096, "effort": "low",
   "price_per_mtok": [<input>, <output>, <cache write>, <cached input>]}
  {"kind": "api", "provider": "fake", "model": "fake-1", "fake_mode": "max_llc"}
APILLM(messages, tools) sends one request per attempt through the provider's sender and returns
the reply text, which ReActPolicy parses and validates (only a legal filter order can be
committed). Errors are classified by the sender: a TransientError (rate limit, overload,
timeout, dropped connection, 5xx) is retried up to max_retries times with exponential backoff,
any other error fails the call at once (fallback:llm_error). The whole call, retries included,
is bounded by ReActPolicy's llm_timeout, after which the worker is killed with the call in it;
for a paid provider validate_spec requires llm_timeout to cover every attempt (call_bound).

Every attempt is recorded (attempts: outcome reply/retried/failed, masked error, HTTP status and
request id when given, latency, billed): 'reported' when the provider returned usage, 'none' when
the provider rejected it unprocessed (rate limit, overload, a 4xx), 'unknown' when it may have
been processed and billed without a usage (timeout, connection error, 5xx, a reply without
usage). For a paid provider `exchange` holds the provider's own request (payload, payload_sha256:
the same for every attempt of a call) from before the first attempt, so a failed call keeps it
too, and the request id and response (id, model, status, output or content with any reasoning,
usage) of the reply; ReActPolicy writes usage, attempts and exchange into the transcript and
ReplayLLM restores them, so a check replays the wall without the provider. spend() turns a
decision's transcripts into reported tokens, estimated tokens (the fake provider's), cost and
the count of unknown usages; the client stops every later API decision after an unknown one and
before a token or cost budget is exceeded, and APILLM reserves, before each attempt, the most
that attempt can cost (see reserve: every byte of the whole request as an input token, the full
max_tokens output) from what the client says is left. These budgets are this program's own
guard of what a run spends, computed from the provider's reported usage and our price table: not
a billing limit of the provider (set a project budget at the provider for that).

APILLM is built by make_policy, i.e. in the controller worker for live walls: the main process
only validates the spec (identity, enabled), so the API client and the API key exist only in the
worker. The key is read there from the provider's fixed environment variable KEY_ENV (a spec
cannot name another source; never from the spec, the trace or a transcript) and registered in
controllers.SECRETS before the client is built, so error_text masks it (and any sk- key pattern)
before it shortens a text, and the worker masks everything it sends. Providers:
- 'fake': a deterministic local stand-in (fake_mode '<MockLLM mode>[+transient@N|+fatal@N]':
  MockLLM's scripted replies, token counts estimated from text lengths, optionally a failure of
  the first attempt of step N's first call); never paid.
- 'anthropic': the Messages API through the official anthropic SDK (imported in the worker
  only; the SDK client's own retries are off so every attempt is ours), tool use with
  tool_choice auto, one tool call per reply and strict tool schemas; the thinking blocks of a
  decision's earlier replies are sent back unchanged. Exactly the requested model.
- 'openai': the Responses API through the official openai SDK (imported in the worker only, its
  own retries off), function tools with tool_choice auto, no parallel tool calls and strict
  schemas, store false (nothing kept at the provider) and encrypted reasoning always included: a
  decision's earlier output items (reasoning with its encrypted content, messages, function
  calls) are sent back unchanged and each tool feedback is the function_call_output of its call;
  a reasoning item without encrypted content fails the call (its context would be lost).
  temperature only for models that take it; the reply must come from the requested model or a
  dated snapshot of it (<model>-YYYY-MM-DD); usage is normalized to USAGE_KEYS (cached and
  cache-written input apart, reasoning tokens kept as part of the output).
Both paid providers: a reply with more than one tool call fails the call (one per reply is the
protocol; none of them is executed), its usage counted and its exchange kept.
Both paid providers: enabled() refuses them unless STONEPACKING_ALLOW_PAID_API=1 is set for an
approved test, the key variable is set and the model has a price (PRICES or the spec's
price_per_mtok: USD per million input, output, cache-write and cache-read tokens);
validate_spec also requires max_total_tokens and max_cost_usd.
"""
import hashlib
import json
import os
import re
import time

from controllers import SECRETS, LLMFailure, MockLLM, error_text

COMMON = {'max_tokens': 1024, 'max_retries': 2, 'retry_backoff_s': 1.0}
PAID_DEFAULTS = {'temperature': None, 'effort': None, 'request_timeout_s': 60.0, 'price_per_mtok': None}
PROVIDER_DEFAULTS = {'fake': {'temperature': 0.0, 'fake_mode': None}, 'anthropic': PAID_DEFAULTS, 'openai': PAID_DEFAULTS}
KEY_ENV = {'anthropic': 'ANTHROPIC_API_KEY', 'openai': 'OPENAI_API_KEY'}  # fixed: a spec cannot name another source
PAID_GATE = 'STONEPACKING_ALLOW_PAID_API'
# the provider's documented values (openai: those of gpt-6.1-terra, a subset of the SDK's ReasoningEffort)
EFFORTS = {'anthropic': (None, 'low', 'medium', 'high', 'xhigh', 'max'),
           'openai': (None, 'none', 'low', 'medium', 'high', 'xhigh', 'max')}
INPUT_OVERHEAD = 1000  # tokens of the provider's own framing of a request, reserved on top of its bytes
USAGE_KEYS = ('input_tokens', 'output_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens')
# USD per million tokens (input, output, cache write 5 min, cache read): the claude-api reference
# table cached 2026-10-06; check the provider's price page before a paid run.
PRICES = {'claude-opus-5-5': (4.00, 20.00, 5.00, 0.20), 'claude-sonnet-5-5': (2.00, 10.00, 2.50, 0.20)}


class ProviderError(Exception):
    """A provider failure the sender classified; billed: 'none' or 'unknown' (see the docstring)."""

    def __init__(self, text, billed='none', status_code=None, request_id=None):
        super().__init__(text)
        self.billed, self.status_code, self.request_id = billed, status_code, request_id


class TransientError(ProviderError):
    """A provider failure worth retrying (rate limit, overload, timeout, dropped connection, 5xx)."""


def full(llm):
    """An API LLM spec with its provider's defaults; bad specs raise. An unknown provider gets the
    common defaults only (enabled() refuses it)."""
    if not isinstance(llm, dict) or llm.get('kind') != 'api' or not isinstance(llm.get('provider'), str):
        raise ValueError(f'an API LLM spec is {{"kind": "api", "provider": ..., "model": ...}}, got {llm!r}')
    extra = PROVIDER_DEFAULTS.get(llm['provider'], {})
    keys = {'kind', 'provider', 'model'} | set(COMMON) | set(extra)
    if set(llm) - keys:
        raise ValueError(f"unknown {llm['provider']} LLM keys {sorted(set(llm) - keys)}")
    out = {**COMMON, **extra, **llm}
    number = lambda v, lo, hi: type(v) in (int, float) and lo <= v <= hi
    checks = {'model': isinstance(out.get('model'), str) and out['model'] != '',
              'max_tokens': type(out['max_tokens']) is int and 0 < out['max_tokens'] <= 128_000,
              'max_retries': type(out['max_retries']) is int and 0 <= out['max_retries'] <= 5,
              'retry_backoff_s': number(out['retry_backoff_s'], 0, 60)}
    if llm['provider'] == 'fake':
        checks.update(temperature=number(out['temperature'], 0, 2), fake_mode=out['fake_mode'] is None or isinstance(out['fake_mode'], str))
    if llm['provider'] in ('anthropic', 'openai'):
        checks.update(effort=out['effort'] in EFFORTS[llm['provider']], request_timeout_s=number(out['request_timeout_s'], 1, 600),
                      price_per_mtok=out['price_per_mtok'] is None or (isinstance(out['price_per_mtok'], list) and len(out['price_per_mtok']) == 4
                                                                       and all(number(v, 0, 1000) for v in out['price_per_mtok'])),
                      # current Claude models reject sampling parameters with a 400; OpenAI's take 0..2 where a model allows it
                      temperature=out['temperature'] is None or (llm['provider'] == 'openai' and number(out['temperature'], 0, 2)))
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise ValueError(f'bad API LLM spec fields {bad}: {llm!r}')
    return out


def identity(llm):
    """(model, params) of the requests this LLM's transcripts record."""
    spec = full(llm)
    params = {'temperature': spec.get('temperature'), 'max_tokens': spec['max_tokens']}
    return spec['model'], {**params, **({'effort': spec['effort']} if spec.get('effort') else {})}


def paid(llm):
    return full(llm)['provider'] != 'fake'


def price(llm):
    """USD per million tokens (input, output, cache write, cache read); None if unknown."""
    spec = full(llm)
    if spec['provider'] == 'fake':
        return (0.0, 0.0, 0.0, 0.0)
    return tuple(spec['price_per_mtok']) if spec.get('price_per_mtok') else PRICES.get(spec['model'])


def call_bound(llm):
    """Seconds one call may take with every retry (request timeouts and backoffs)."""
    spec = full(llm)
    tries = spec['max_retries'] + 1
    return tries * spec.get('request_timeout_s', 0) + sum(spec['retry_backoff_s'] * 2 ** k for k in range(tries - 1))


def enabled(llm):
    """Raises unless a live wall may call this provider: the fake one, or a paid one with the gate
    of an approved API test set, its key variable set (not read here) and a price for the model."""
    spec = full(llm)
    if spec['provider'] not in PROVIDERS:
        raise NotImplementedError(f"provider {spec['provider']!r} is not implemented")
    if paid(llm):
        if os.environ.get(PAID_GATE) != '1':
            raise NotImplementedError(f'paid API calls are off: an approved API test sets {PAID_GATE}=1')
        if KEY_ENV[spec['provider']] not in os.environ:
            raise ValueError(f"the API key variable {KEY_ENV[spec['provider']]} is not set")
        if price(llm) is None:
            raise ValueError(f"no price for model {spec['model']!r}: its cost could not be budgeted")


def usage_ok(usage):
    return isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0 for k in USAGE_KEYS[:2]) and \
        all(type(usage.get(k, 0)) is int and usage.get(k, 0) >= 0 for k in USAGE_KEYS[2:])


def cost(prices, usage):
    return sum(usage.get(k, 0) * p for k, p in zip(USAGE_KEYS, prices)) / 1e6


def spend(llm, transcripts):
    """What a decision's transcripts cost: {'tokens' (reported), 'estimated_tokens', 'cost_usd',
    'unknown' (attempts or calls whose usage is unknown: never counted as free)}."""
    prices, out = price(llm), {'tokens': 0, 'estimated_tokens': 0, 'cost_usd': 0.0, 'unknown': 0}
    for t in transcripts if isinstance(transcripts, list) else []:
        response = t.get('response') if isinstance(t, dict) and isinstance(t.get('response'), dict) else {}
        attempts = response.get('attempts') if isinstance(response.get('attempts'), list) else []
        out['unknown'] += sum(isinstance(a, dict) and a.get('billed') == 'unknown' for a in attempts)
        out['unknown'] += isinstance(t, dict) and t.get('status') == 'timeout'  # abandoned in flight
        usage = response.get('usage')
        if usage_ok(usage):
            out['estimated_tokens' if usage.get('estimated') else 'tokens'] += sum(usage.get(k, 0) for k in USAGE_KEYS)
            if prices is None:
                out['unknown'] += 1
            else:
                out['cost_usd'] += cost(prices, usage)
        elif not isinstance(t, dict) or t.get('status') == 'ok':
            out['unknown'] += 1
    return out


class FakeSender:
    """The 'fake' provider: MockLLM's reply to the payload as JSON text, usage estimated from text
    lengths; fake_mode '<mode>+transient@N' or '<mode>+fatal@N' (mode max_llc by default) also
    fails the first attempt of step N's first call, unprocessed (billed 'none')."""

    def __init__(self, spec):
        mode, _, inject = (spec.get('fake_mode') or 'max_llc').partition('+')
        failure, _, step = inject.partition('@')
        if inject and failure not in ('transient', 'fatal'):
            raise ValueError(f'fake_mode injects transient@N or fatal@N, got {inject!r}')
        self.failure, self.step = (failure, int(step)) if inject else (None, None)
        self.mock = MockLLM(mode)

    def __call__(self, payload, attempt):
        messages = payload['messages']
        if self.failure and attempt == 0 and len(messages) == 2 and json.loads(messages[1]['content'])['step'] == self.step:
            raise (TransientError if self.failure == 'transient' else ProviderError)(f'injected {self.failure} failure')
        text = json.dumps(self.mock(messages, payload['tools']))
        return {'text': text, 'usage': {'input_tokens': len(json.dumps(messages)) // 4, 'output_tokens': len(text) // 4, 'estimated': True}}


def classified(sdk, error):
    """The ProviderError of an SDK exception (the anthropic and openai SDKs share these names), with
    its masked text, HTTP status and request id: retried or not, billed or not (see the docstring)."""
    text, status = error_text(error), getattr(error, 'status_code', None)
    meta = {'status_code': status if type(status) is int else None, 'request_id': getattr(error, 'request_id', None)}
    if isinstance(error, sdk.RateLimitError):  # an exhausted quota is no rate limit: retrying cannot help
        return ProviderError(text, **meta) if getattr(error, 'code', None) == 'insufficient_quota' else TransientError(text, 'none', **meta)
    if isinstance(error, sdk.APIStatusError):
        if status == 529:
            return TransientError(text, 'none', **meta)  # overloaded: not processed
        if type(status) is not int or status >= 500 or status in (408, 409):
            return TransientError(text, 'unknown', **meta)
        return ProviderError(text, 'none', **meta)
    if isinstance(error, sdk.APIConnectionError):  # timeouts included: the request may have been processed
        return TransientError(text, 'unknown', **meta)
    return ProviderError(text, 'unknown', **meta)  # anything else: not retried, never counted as free


def api_key(provider):
    """The key of a paid provider, from its fixed environment variable (KEY_ENV), registered for
    masking before any client sees it."""
    secret = os.environ[KEY_ENV[provider]]
    SECRETS.add(secret)
    return secret


def tool_calls(problem, calls):
    """problem, or the protocol's own: one tool call per reply (more are rejected, never executed)."""
    return problem or (f'{len(calls)} tool calls in one reply: the protocol allows one' if len(calls) > 1 else None)


class AnthropicSender:
    """The 'anthropic' provider (see the module docstring). sdk: the anthropic module, or a stand-in
    with the same names (tests); the client is built here, in the worker, with the key of KEY_ENV."""

    def __init__(self, spec, sdk=None):
        if sdk is None:
            import anthropic as sdk
        self.sdk = sdk
        self.client = sdk.Anthropic(api_key=api_key('anthropic'), max_retries=0, timeout=spec['request_timeout_s'])
        self.turns = []  # the content of this decision's replies, sent back unchanged

    def request(self, payload):
        """The Messages API request of ReActPolicy's messages: its system message, user turns, each
        assistant turn as the provider's own reply content and each tool feedback as the
        tool_result of that reply's tool call (a text turn if the reply called no tool)."""
        messages = payload['messages']
        if len(messages) == 2:
            self.turns = []
        out, call, assistants = [], None, 0
        for m in messages[1:]:
            if m['role'] == 'assistant':
                content = self.turns[assistants] if assistants < len(self.turns) else [{'type': 'text', 'text': m['content']}]
                assistants += 1
                out.append({'role': 'assistant', 'content': content})
                call = next((b['id'] for b in content if b.get('type') == 'tool_use'), None)
            elif m['role'] == 'tool' and call is not None:
                out.append({'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': call, 'content': m['content']}]})
                call = None
            else:
                out.append({'role': 'user', 'content': m['content']})
        body = {'model': payload['model'], 'max_tokens': payload['max_tokens'], 'system': messages[0]['content'],
                'messages': out, 'tool_choice': {'type': 'auto', 'disable_parallel_tool_use': True},
                'tools': [{'name': t['name'], 'description': t['description'], 'input_schema': t['parameters'], 'strict': True}
                          for t in payload['tools']]}
        if payload.get('effort'):
            body['output_config'] = {'effort': payload['effort']}
        return body

    def __call__(self, body, attempt):
        try:
            response = self.client.messages.create(**body)
        except Exception as error:
            raise classified(self.sdk, error)
        data = response.to_dict()
        usage = data.get('usage') if isinstance(data.get('usage'), dict) else {}
        usage = {k: usage[k] for k in USAGE_KEYS if type(usage.get(k)) is int} or None
        content = data.get('content') if isinstance(data.get('content'), list) else []
        uses = [b for b in content if isinstance(b, dict) and b.get('type') == 'tool_use']
        problem = None
        if data.get('model') != body['model']:
            problem = f"served by model {data.get('model')!r}, not {body['model']!r}"
        elif data.get('stop_reason') == 'refusal':
            problem = f"refusal ({(data.get('stop_details') or {}).get('category')})"
        elif data.get('stop_reason') == 'max_tokens':
            problem = 'the reply stopped at max_tokens'
        self.turns.append(content)
        thought = ' '.join(b['text'] for b in content if isinstance(b, dict) and b.get('type') == 'text' and isinstance(b.get('text'), str))
        reply = {'thought': thought or None, **({'tool': uses[0].get('name'), 'arguments': uses[0].get('input')} if uses else {})}
        return {'text': json.dumps(reply), 'usage': usage, 'problem': tool_calls(problem, uses),
                'request_id': getattr(response, '_request_id', None),
                'response': {k: data.get(k) for k in ('id', 'model', 'stop_reason', 'stop_details', 'content', 'usage')}}


class OpenAISender:
    """The 'openai' provider (see the module docstring). sdk: the openai module, or a stand-in with
    the same names (tests); the client is built here, in the worker, with the key of KEY_ENV."""

    def __init__(self, spec, sdk=None):
        if sdk is None:
            import openai as sdk
        self.sdk = sdk
        self.client = sdk.OpenAI(api_key=api_key('openai'), max_retries=0, timeout=spec['request_timeout_s'])
        self.turns = []  # the output items of this decision's replies, sent back unchanged

    def request(self, payload):
        """The Responses API request of ReActPolicy's messages: the system message as instructions,
        user turns, each assistant turn as the provider's own output items (reasoning items with
        their encrypted content: nothing is stored at the provider) and each tool feedback as the
        function_call_output of that reply's call (a user turn if the reply called no function)."""
        messages = payload['messages']
        if len(messages) == 2:
            self.turns = []
        items, call, assistants = [], None, 0
        for m in messages[1:]:
            if m['role'] == 'assistant':
                output = self.turns[assistants] if assistants < len(self.turns) else [{'role': 'assistant', 'content': m['content']}]
                assistants += 1
                items += output
                call = next((i['call_id'] for i in output if i.get('type') == 'function_call'), None)
            elif m['role'] == 'tool' and call is not None:
                items.append({'type': 'function_call_output', 'call_id': call, 'output': m['content']})
                call = None
            else:
                items.append({'role': 'user', 'content': m['content']})
        body = {'model': payload['model'], 'instructions': messages[0]['content'], 'input': items,
                'max_output_tokens': payload['max_tokens'], 'tool_choice': 'auto', 'parallel_tool_calls': False, 'store': False,
                'include': ['reasoning.encrypted_content'],  # with or without an effort: the model may reason either way
                'tools': [{'type': 'function', 'name': t['name'], 'description': t['description'], 'parameters': t['parameters'],
                           'strict': True} for t in payload['tools']]}
        if payload.get('effort'):
            body['reasoning'] = {'effort': payload['effort']}
        if payload.get('temperature') is not None:
            body['temperature'] = payload['temperature']
        return body

    def __call__(self, body, attempt):
        try:
            response = self.client.responses.create(**body)
        except Exception as error:
            raise classified(self.sdk, error)
        data = response.to_dict()
        raw = data.get('usage') if isinstance(data.get('usage'), dict) else {}
        details = raw.get('input_tokens_details') or {}
        cached, written = details.get('cached_tokens') or 0, details.get('cache_write_tokens') or 0
        usage = None  # input_tokens counts cached and cache-written input too: each is priced apart
        if all(type(v) is int for v in (raw.get('input_tokens'), raw.get('output_tokens'), cached, written)) and \
                raw['input_tokens'] >= cached + written:
            usage = {'input_tokens': raw['input_tokens'] - cached - written, 'output_tokens': raw['output_tokens'],
                     'cache_creation_input_tokens': written, 'cache_read_input_tokens': cached,
                     'reasoning_tokens': (raw.get('output_tokens_details') or {}).get('reasoning_tokens')}
        output = [i for i in data.get('output') or [] if isinstance(i, dict)]
        texts = [c for i in output if i.get('type') == 'message' for c in i.get('content') or [] if isinstance(c, dict)]
        uses = [i for i in output if i.get('type') == 'function_call']
        model, problem = str(data.get('model')), None
        if not re.fullmatch(re.escape(body['model']) + r'(-\d{4}-\d{2}-\d{2})?', model):  # the model or a dated snapshot of it
            problem = f"served by model {model!r}, not {body['model']!r}"
        elif data.get('status') != 'completed':
            problem = f"response {data.get('status')}: {data.get('incomplete_details') or data.get('error')}"
        elif any(c.get('type') == 'refusal' for c in texts):
            problem = 'refusal'
        elif any(i.get('type') == 'reasoning' and not i.get('encrypted_content') for i in output):
            problem = 'a reasoning item without encrypted content: its context could not be sent back'
        self.turns.append(output)
        thought = ' '.join(c['text'] for c in texts if c.get('type') == 'output_text' and isinstance(c.get('text'), str))
        args = None
        if uses:
            try:
                args = json.loads(uses[0].get('arguments'))
            except (TypeError, ValueError):
                args = uses[0].get('arguments')  # not an object: rejected by reply_problem
        reply = {'thought': thought or None, **({'tool': uses[0].get('name'), 'arguments': args} if uses else {})}
        return {'text': json.dumps(reply), 'usage': usage, 'problem': tool_calls(problem, uses),
                'request_id': getattr(response, '_request_id', None),
                'response': {k: data.get(k) for k in ('id', 'model', 'status', 'incomplete_details', 'error', 'output', 'usage')}}


PROVIDERS = {'fake': FakeSender, 'anthropic': AnthropicSender, 'openai': OpenAISender}


def sha256(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


class APILLM:
    """The LLM callable of ReActPolicy for an API spec; see the module docstring. sender replaces
    the provider's (tests)."""

    def __init__(self, llm, sender=None):
        spec = full(llm)
        (self.model, self.params), self.prices = identity(llm), price(llm)
        self.max_retries, self.backoff = spec['max_retries'], spec['retry_backoff_s']
        if sender is None:
            enabled(llm)
            sender = PROVIDERS[spec['provider']](spec)
        self.send, self.usage, self.attempts, self.exchange = sender, None, None, None
        self.begin(None)

    def begin(self, budget):
        """A new decision with what is left of the wall's budget ({'tokens_left', 'usd_left'} or None)."""
        self.budget, self.spent_tokens, self.spent_usd = budget or {}, 0, 0.0

    def reserve(self, body):
        """(tokens, usd) an attempt may cost at most: every UTF-8 byte of the whole request it sends
        is counted as an input token (a byte-level tokenizer makes at most one token of each byte;
        INPUT_OVERHEAD covers the provider's framing), priced at the dearer of input and cache write,
        plus its full max_tokens output (reasoning included); raises if what is left cannot pay that."""
        tokens_in = len(json.dumps(body, ensure_ascii=False).encode()) + INPUT_OVERHEAD
        tokens, usd = tokens_in + self.params['max_tokens'], 0.0
        if self.prices is not None:
            usd = (tokens_in * max(self.prices[0], self.prices[2]) + self.params['max_tokens'] * self.prices[1]) / 1e6
        left_tokens, left_usd = self.budget.get('tokens_left'), self.budget.get('usd_left')
        if left_tokens is not None and self.spent_tokens + tokens > left_tokens:
            raise LLMFailure('error', f'token budget: an attempt may use {tokens} tokens, {left_tokens - self.spent_tokens} left')
        if left_usd is not None and self.spent_usd + usd > left_usd:
            raise LLMFailure('error', f'cost budget: an attempt may cost {usd:.4f} USD, {left_usd - self.spent_usd:.4f} left')
        return tokens, usd

    def record(self, attempt, start, outcome, billed, reserved, error=None, request_id=None):
        """One attempt: its outcome ('reply', 'retried' or 'failed'), masked error, HTTP status and
        request id when the provider gave them, latency and billing; spends what it cost (unknown:
        the most it could have)."""
        meta = {'status_code': getattr(error, 'status_code', None), 'request_id': request_id or getattr(error, 'request_id', None)}
        self.attempts.append({'attempt': attempt, 'outcome': outcome, 'error': None if error is None else error_text(error),
                              'latency_s': round(time.monotonic() - start, 3), 'billed': billed,
                              **{k: v for k, v in meta.items() if v is not None}})
        if billed == 'unknown':
            self.spent_tokens, self.spent_usd = self.spent_tokens + reserved[0], self.spent_usd + reserved[1]
        elif billed == 'reported':
            self.spent_tokens += sum(self.usage.get(k, 0) for k in USAGE_KEYS)
            self.spent_usd += 0.0 if self.prices is None else cost(self.prices, self.usage)

    def __call__(self, messages, tools):
        self.usage, self.attempts, self.exchange = None, [], None
        payload = {'model': self.model, **self.params, 'messages': messages, 'tools': tools}
        request = self.send.request if hasattr(type(self.send), 'request') else None  # a provider's own request
        body = request(payload) if request else payload
        if request:  # the provider's request, recorded whether or not an attempt gets a reply
            self.exchange = {'payload': body, 'payload_sha256': sha256(body), 'request_id': None, 'response': None}
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(self.backoff * 2 ** (attempt - 1))
            reserved, start = self.reserve(body), time.monotonic()
            try:
                out = self.send(body, attempt)
            except TransientError as error:
                self.record(attempt, start, 'retried', error.billed, reserved, error)
                continue
            except Exception as error:  # not worth a retry: the call fails (fallback:llm_error)
                self.record(attempt, start, 'failed', getattr(error, 'billed', 'unknown'), reserved, error)
                raise
            out = out if isinstance(out, dict) else {}
            self.usage = out.get('usage')
            if request:
                self.exchange.update(request_id=out.get('request_id'), response=out.get('response'))
            self.record(attempt, start, 'reply', 'reported' if usage_ok(self.usage) else 'unknown', reserved,
                        request_id=out.get('request_id'))
            if out.get('problem'):
                raise LLMFailure('error', out['problem'])
            if not isinstance(out.get('text'), str):
                raise LLMFailure('error', 'the provider reply has no text')
            return out['text']
        raise LLMFailure('error', f"no reply after {self.max_retries + 1} attempts: {self.attempts[-1]['error']}")
