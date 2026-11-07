"""LLM API adapter of the ReAct controller (P2 integration; no real provider is enabled yet).

An API LLM spec names the provider, the model and the call parameters, e.g.
  {"kind": "api", "provider": "fake", "model": "fake-1", "temperature": 0.0, "max_tokens": 1024,
   "max_retries": 2, "retry_backoff_s": 1.0, "fake_mode": "max_llc"}
APILLM(messages, tools) sends one request per attempt through the provider's sender and returns
the reply text, which ReActPolicy parses and validates (only a legal filter order can be
committed). A TransientError (rate limit, overload, connection reset) is retried up to
max_retries times with exponential backoff; any other error fails the call at once. The whole
call, retries included, is bounded by ReActPolicy's llm_timeout, after which the worker is killed
with the call in it. After a call `usage` holds the provider's token counts and `attempts` one
entry per attempt (error, latency); ReActPolicy writes both into the transcript and ReplayLLM
restores them, so a check replays the wall without the provider. The token budget of a wall is
the client's (max_total_tokens in the react spec), counted from these usages.

APILLM is built by make_policy, i.e. in the controller worker for live walls: the main process
only validates the spec (identity, enabled). Only the 'fake' provider exists: a deterministic
local stand-in (fake_mode '<MockLLM mode>[+transient@N|+fatal@N]': MockLLM's scripted replies,
token counts from text lengths, optionally a failure of the first attempt of step N's first call). A real provider
is a sender `send(payload, attempt) -> {'text': str, 'usage': {...}}` added to PROVIDERS; until a
real API test is approved, enabled() refuses every provider but 'fake'.
"""
import json
import time

from controllers import LLMFailure, MockLLM, error_text

DEFAULTS = {'temperature': 0.0, 'max_tokens': 1024, 'max_retries': 2, 'retry_backoff_s': 1.0}
KEYS = {'kind', 'provider', 'model', 'fake_mode'} | set(DEFAULTS)


class TransientError(Exception):
    """A provider failure worth retrying (rate limit, overload, dropped connection)."""


def full(llm):
    """An API LLM spec with its defaults; bad specs raise."""
    if not isinstance(llm, dict) or llm.get('kind') != 'api' or set(llm) - KEYS:
        raise ValueError(f'an API LLM spec has the keys {sorted(KEYS)}, got {llm!r}')
    out = {**DEFAULTS, **llm}
    checks = {'provider': isinstance(out.get('provider'), str), 'model': isinstance(out.get('model'), str) and out['model'] != '',
              'temperature': type(out['temperature']) in (int, float) and 0 <= out['temperature'] <= 2,
              'max_tokens': type(out['max_tokens']) is int and out['max_tokens'] > 0,
              'max_retries': type(out['max_retries']) is int and 0 <= out['max_retries'] <= 5,
              'retry_backoff_s': type(out['retry_backoff_s']) in (int, float) and 0 <= out['retry_backoff_s'] <= 60,
              'fake_mode': out.get('fake_mode') is None or (out['provider'] == 'fake' and isinstance(out['fake_mode'], str))}
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise ValueError(f'bad API LLM spec fields {bad}: {llm!r}')
    return out


def identity(llm):
    """(model, params) of the requests this LLM's transcripts record."""
    spec = full(llm)
    return spec['model'], {'temperature': spec['temperature'], 'max_tokens': spec['max_tokens']}


def enabled(llm):
    """Raises unless the provider may be called by a live wall: only the local fake for now."""
    if full(llm)['provider'] not in PROVIDERS:
        raise NotImplementedError(f"provider {llm['provider']!r} is not enabled: real API calls need an approved API test")


class FakeSender:
    """The 'fake' provider: MockLLM's reply to the payload as JSON text, usage from text lengths;
    fake_mode '<mode>+transient@N' or '<mode>+fatal@N' (mode max_llc by default) also fails the
    first attempt of step N's first call."""

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
            raise (TransientError if self.failure == 'transient' else RuntimeError)(f'injected {self.failure} failure')
        text = json.dumps(self.mock(messages, payload['tools']))
        return {'text': text, 'usage': {'input_tokens': len(json.dumps(messages)) // 4, 'output_tokens': len(text) // 4}}


PROVIDERS = {'fake': FakeSender}


class APILLM:
    """The LLM callable of ReActPolicy for an API spec; see the module docstring. sender replaces
    the provider's (tests)."""

    def __init__(self, llm, sender=None):
        spec = full(llm)
        self.model, self.params = identity(llm)
        self.max_retries, self.backoff = spec['max_retries'], spec['retry_backoff_s']
        if sender is None:
            enabled(llm)
            sender = PROVIDERS[spec['provider']](spec)
        self.send, self.usage, self.attempts = sender, None, None

    def __call__(self, messages, tools):
        self.usage, self.attempts = None, []
        payload = {'model': self.model, **self.params, 'messages': messages, 'tools': tools}
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(self.backoff * 2 ** (attempt - 1))
            start = time.monotonic()
            try:
                out = self.send(payload, attempt)
            except TransientError as error:
                self.attempts.append({'attempt': attempt, 'error': error_text(error), 'latency_s': round(time.monotonic() - start, 3)})
                continue
            except Exception as error:  # not worth a retry: the call fails (fallback:llm_error)
                self.attempts.append({'attempt': attempt, 'error': error_text(error), 'latency_s': round(time.monotonic() - start, 3)})
                raise
            self.attempts.append({'attempt': attempt, 'error': None, 'latency_s': round(time.monotonic() - start, 3)})
            if not (isinstance(out, dict) and isinstance(out.get('text'), str)):
                raise LLMFailure('error', 'the provider reply has no text')
            self.usage = out.get('usage')
            return out['text']
        raise LLMFailure('error', f"no reply after {self.max_retries + 1} attempts: {self.attempts[-1]['error']}")
