"""Dynamic filter-order controllers for the paper policy (P2 architecture, mock LLM only).

Usage: python -B research/controllers.py offline <campaign id> <out dir> [--arm paper_3-0-2-1-4]
       python -B research/controllers.py check <run>

build_wall calls a controller before every selection (selection.controlled_policy) with a
read-only copy of the base policy and detached copies of that selection's inputs; it returns the
policy for this one selection: the base (Paper) order or another permutation. Every decision
point starts from the run's base policy (receipt filter_policy), never from an earlier choice.
Candidate generation, image convolution, the LLC > 0 gate and the stability analysis are not
touched; a controller can only pick what some filter order selects (an outcome class).

The controller is split in two. ControllerClient (main process, trusted) builds the observation
and the tool table of the 120 orders, validates the order string it gets back, runs select()
and writes the record. The policy (Null, Random, Matched-Random, Rule, ReAct with its LLM) only
maps a JSON request to a JSON reply; live walls run it in a worker process
(controller_worker.ProcessTransport), offline evaluation and check() in this one (InlineTransport).

Information boundary. A controller sees the current wall, the sampled stones' candidates and
filter scores, how many stones remain, and what it saw at its own earlier calls. It does not see
the RNG state, stone ids (stones are labelled S1, S2, ... in order of first sampling), the
unsampled inventory, future samples, the downstream wall or the final evaluation (lambda).
Geometry of an outcome (wall_state after pasting the candidate's pixels at its pose) is a
pre-stabilization proxy and exists only live, where the stone pixels are at hand.

Every call is one line of <result>/trace/controller.jsonl: event, decision key (step, phase,
sample_trial, side), observation, ReAct steps (status, thought, tool, arguments, feedback) and the
final order with the candidate it selects. check() re-executes the controller named in the
receipt on the traced inputs and compares every record.
"""
import argparse
import copy
import hashlib
import json
import pickle
import re
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from failure_analysis import FILTERS, wall_state  # noqa: E402

from check_case import chain_problems  # noqa: E402
from Stonepacker2D.stack.selection import ALL_ORDERS, PAPER_ORDER, detached_inputs, footprint, load_trace, published_events, select  # noqa: E402

ORDER_TEXT = re.compile(r'[0-4](-[0-4]){4}')


def key(order):
    return '-'.join(map(str, order))


def parse_order(value):
    """'3-0-2-1-4' or a list/tuple of five ints -> tuple; None for anything else (floats and
    numeric strings are not coerced) or if it is not a permutation of 0..4."""
    if isinstance(value, str):
        order = tuple(int(c) for c in value.split('-')) if ORDER_TEXT.fullmatch(value) else None
    elif isinstance(value, (list, tuple)) and all(type(c) is int for c in value):
        order = tuple(value)
    else:
        order = None
    return order if order is not None and sorted(order) == [0, 1, 2, 3, 4] else None


def context_from_trace(meta, a):
    """The selection inputs of a trace event as a controller receives them live (detached, no stone pixels)."""
    return detached_inputs({'step': meta['step'], 'side': meta['side'], 'phase': meta['phase'],
                            'sample_trial': meta['sample_trial'], 'wall': a['wall'], 'all_bricks': None,
                            'sampled_ids': a['sampled_ids'].tolist(), 'inventory_ids': a['inventory_ids'].tolist(),
                            'x': a['x'], 'y': a['y'], 'distance': a['distance'],
                            'features_raw': a['features_raw'], 'weights': a['weights']})


def plain(v):
    """Copy with NaN/inf -> None, so observations are plain JSON and compare equal after a round trip."""
    if isinstance(v, dict):
        return {k: plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    return None if isinstance(v, float) and not np.isfinite(v) else v


def placement_geometry(wall, pixels):
    """Change of wall_state when pixels (bool, wall-sized) are filled; no physics, no stabilization."""
    after = wall.copy()
    after[pixels & (after == 0)] = 1
    before, now = wall_state(wall), wall_state(after)
    return plain({f'd_{k}': round(float(now[k] - before[k]), 4) for k in before} | {'stone_px': int(pixels.sum())})


class Observer:
    """Builds the controller-visible observation and keeps the controller's memory of earlier calls."""

    def __init__(self, geometry=None):
        self.labels, self.stones, self.recent = {}, {}, []
        self.geometry = geometry  # LiveGeometry when re-executing a live wall from its trace

    def label(self, sid):
        return self.labels.setdefault(int(sid), f'S{len(self.labels) + 1}')

    def __call__(self, policy, ctx):
        raw, weights, tol = np.asarray(ctx['features_raw']), ctx['weights'], policy['tol']
        sampled = [int(s) for s in ctx['sampled_ids']]
        n = len(sampled)
        for s in sampled:
            self.label(s)
        stone = np.array([sampled[i % n] for i in range(len(raw))])
        rotation = np.arange(len(raw)) // n * 90
        feasible = (raw[:, 3] > 0) & np.isfinite(raw[:, :5]).all(axis=1)
        base = tuple(policy['order'] or PAPER_ORDER)
        by_index, selections = {}, {order: select(raw, weights, 'paper', order, tol) for order in ALL_ORDERS}
        for order, s in selections.items():
            if s.index is not None:
                by_index.setdefault(int(s.index), []).append(order)
        paper = selections[base].index
        paper = None if paper is None else int(paper)
        classes = sorted(by_index, key=lambda i: (i != paper, -len(by_index[i]), i))
        remaining = {int(s) for s in ctx['inventory_ids']} | set(sampled)
        outcomes = []
        for k, i in enumerate(classes):
            geometry = None
            if ctx['all_bricks'] is not None:
                moved = ctx['all_bricks'][i].transformed([ctx['x'][i], ctx['y'][i]])
                geometry = None if moved is None else placement_geometry(np.asarray(ctx['wall']), moved.matrix != 0)
            outcomes.append({'id': k, 'stone': self.label(stone[i]), 'rotation': int(rotation[i]),
                             'x': int(ctx['x'][i]), 'y': int(ctx['y'][i]), 'n_orders': len(by_index[i]),
                             'example_order': key(min(by_index[i])), 'is_paper': i == paper,
                             'scores': {FILTERS[c]: round(float(raw[i, c]), 6) for c in range(5)},
                             'geometry': geometry})
        reachable = {o['stone'] for o in outcomes}
        stones_now = [{'stone': self.label(s), 'feasible_rotations': int(feasible[stone == s].sum()),
                       'best_llc': round(float(raw[stone == s, 3].max()), 6), 'reachable': self.label(s) in reachable}
                      for s in sampled]
        now = {'feasible_share': round(float(feasible.mean()), 4),
               'mean_feasible_llc': round(float(raw[feasible, 3].mean()), 6) if feasible.any() else None}
        obs = {'step': int(ctx['step']), 'phase': ctx['phase'], 'sample_trial': int(ctx['sample_trial']),
               'side': ctx['side'], 'n_remaining': len(remaining), 'base_order': key(base), 'frame': 'evaluation',
               'wall': wall_state(np.asarray(ctx['wall'])), 'now': now, 'sampled': stones_now,
               'memory': {'stones': {self.label(s): self.stones[self.label(s)] for s in sorted(remaining)
                                     if s in self.labels and self.label(s) in self.stones},
                          'recent': list(self.recent)},
               'outcomes': outcomes}
        obs = plain(obs)  # also detaches the memory entries updated below
        if self.geometry is not None:
            self.geometry.fill(obs)
        for s in obs['sampled']:  # memory is updated after the observation, so it only holds the past
            m = self.stones.setdefault(s['stone'], {'times_sampled': 0, 'min_feasible_rotations': 4})
            m.update(times_sampled=m['times_sampled'] + 1, last_step=obs['step'],
                     last_feasible_rotations=s['feasible_rotations'], last_best_llc=s['best_llc'],
                     min_feasible_rotations=min(m['min_feasible_rotations'], s['feasible_rotations']))
        self.recent = (self.recent + [{'step': obs['step'], **obs['now']}])[-3:]
        info = {'classes': classes, 'by_index': by_index, 'paper': paper, 'base': base, 'stone_ids': stone, 'tol': tol,
                'selections': selections, 'n': n}
        return obs, info


def decision_key(record):
    """Identifies one selection across walls: probe and place of a sample_twice step share step and trial."""
    return (record['step'], record['phase'], record['sample_trial'], record['side'])


def tool_table(info, label):
    """What each of the 120 orders selects and why (the explain_order answer), keyed in ALL_ORDERS
    order: the legal order space of every policy. Built here, where the scores are; a policy gets
    it as JSON and never sees the score matrix or the stones."""
    n, stone, classes = info['n'], info['stone_ids'], info['classes']
    name = lambda i: f'{label(stone[i])}/{i // n * 90}'
    return {key(o): {'branch': s.branch, 'selects_outcome': None if s.index is None else classes.index(int(s.index)),
                     'stages': [{'after': 'LLC>0 gate' if k == 0 else FILTERS[o[k - 1]], 'survivors': [name(int(i)) for i in st]}
                                for k, st in enumerate(s.stages)]}
            for o, s in info['selections'].items()}


SPEC_DEFAULTS = {'null': {}, 'random': {}, 'rule': {'max_remaining': 6, 'llc_drop': 0.2},
                 'react': {'llm': None, 'tool_feedback': True, 'max_tool_calls': 4, 'max_decisions': 25,
                           'llm_timeout': 60.0, 'max_llm_failures': 3}}
SPEC_KEYS = {'null': set(), 'random': {'seed', 'steps'}, 'rule': set(SPEC_DEFAULTS['rule']), 'react': set(SPEC_DEFAULTS['react'])}
WORKER_KEYS = {'react': {'tool_feedback', 'max_tool_calls', 'llm_timeout'}}  # the rest of a react spec is the client's
SOURCES = {'null': {'null'}, 'random': {'random', 'matched_random', 'matched_random:not_matched'},
           'rule': {'rule:scarcity', 'rule:llc_drop', 'rule:none'},
           'react': {'llm', 'fallback:llm_timeout', 'fallback:llm_error', 'fallback:no_valid_commit'}}
LLM_FAILURES = {'fallback:llm_timeout', 'fallback:llm_error'}
REPLY_KEYS = {'order', 'source', 'prompt_sha256', 'react', 'transcripts', 'restart'}


def full_spec(spec):
    """A controller spec with its defaults, as records and receipts name it; unknown keys raise."""
    if not isinstance(spec, dict) or spec.get('kind') not in SPEC_KEYS:
        raise ValueError(f'controller spec needs a kind in {sorted(SPEC_KEYS)}, got {spec!r}')
    params = {k: v for k, v in spec.items() if k != 'kind'}
    if set(params) - SPEC_KEYS[spec['kind']]:
        raise ValueError(f"unknown {spec['kind']} controller keys {sorted(set(params) - SPEC_KEYS[spec['kind']])}")
    if spec['kind'] == 'random' and isinstance(params.get('steps'), (list, tuple)):
        params['steps'] = [list(s) if isinstance(s, (list, tuple)) else s for s in params['steps']]
    return {'kind': spec['kind'], **SPEC_DEFAULTS[spec['kind']], **params}


def reply_problem_of(reply, kind, base):
    """Why a policy's reply is not one of `kind`'s answers, None if it is: an object with known
    fields, its order an 'a-b-c-d-e' string, its source one of the kind's, the base order if a fallback."""
    if not isinstance(reply, dict) or set(reply) - REPLY_KEYS or not {'order', 'source'} <= set(reply):
        return 'the reply is not an object with an order and a source'
    if not isinstance(reply['order'], str) or parse_order(reply['order']) is None:
        return "the reply's order is not an 'a-b-c-d-e' permutation of 0..4"
    if not isinstance(reply['source'], str) or reply['source'] not in SOURCES[kind]:
        return f'source is not one of the {kind} policy'
    if reply['source'].startswith('fallback:') and reply['order'] != base:
        return 'a fallback must keep the base order'
    return None


class ControllerClient:
    """The controller build_wall calls (selection.controlled_policy), in the main process.

    Per selection: the Observer's observation; at a decision point a request (observation, base
    order, tool table of the 120 orders) through the transport to the policy, which answers with an
    order string; the order is validated here, select() gives the candidate, the record is written.
    A transport failure (fallback:worker_timeout / worker_error / worker_protocol_error) or a reply
    that is not one of the policy's answers (worker_protocol_error) keeps the base order. For react
    the client holds the wall's counters: request failures (LLM and worker) and LLM decisions;
    after max_llm_failures failures every later decision point is skip:llm_unavailable, after
    max_decisions skip:wall_budget, and neither sends a request.
    """

    def __init__(self, spec, transport, sink=None):
        self.spec_, self.transport = full_spec(spec), transport
        self.observer, self.sink, self.calls = Observer(), sink or write_record, 0
        self.request_failures = self.llm_decisions = 0

    def spec(self):
        return self.spec_

    def ask(self, obs, info):
        spec, base = self.spec_, info['base']
        if spec['kind'] == 'react':
            if self.request_failures >= spec['max_llm_failures']:
                return base, {'source': 'skip:llm_unavailable'}
            if self.llm_decisions >= spec['max_decisions']:
                return base, {'source': 'skip:wall_budget'}
            self.llm_decisions += 1
        tools = tool_table(info, self.observer.label)
        digest = {'tool_table_sha256': hashlib.sha256(json.dumps(tools, sort_keys=True).encode()).hexdigest()}
        status, reply = self.transport.decide({'observation': obs, 'base_order': key(base), 'orders': tools})
        if status == 'ok' and reply_problem_of(reply, spec['kind'], key(base)):
            status, reply = 'worker_protocol_error', reply_problem_of(reply, spec['kind'], key(base))
        if status != 'ok':
            self.request_failures += 1
            return base, {'source': f'fallback:{status}', 'error': reply, **digest}
        self.request_failures += reply['source'] in LLM_FAILURES
        return parse_order(reply['order']), {**{k: v for k, v in reply.items() if k not in ('order', 'restart')}, **digest}

    def __call__(self, policy, ctx):
        obs, info = self.observer(policy, ctx)
        decision_point = len(info['classes']) >= 2
        order, detail = self.ask(obs, info) if decision_point else (info['base'], {'source': 'not_a_decision_point'})
        index = info['selections'][order].index
        index = None if index is None else int(index)
        self.sink({'event': self.calls, 'step': obs['step'], 'phase': obs['phase'], 'sample_trial': obs['sample_trial'],
                   'side': obs['side'], 'controller': self.spec(), 'decision_point': decision_point,
                   'n_outcomes': len(info['classes']), 'observation': obs, **detail,
                   'final': {'order': key(order), 'index': index, 'switched': index != info['paper'],
                             'outcome': info['classes'].index(index) if index in info['classes'] else None,
                             'stone_id': None if index is None else int(info['stone_ids'][index])}})
        self.calls += 1
        return policy if order == info['base'] else {**policy, 'order': order}


class InlineTransport:
    """The policy in this process, on JSON copies of request and reply: offline evaluation, check()
    and tests. Failures and restarts mirror ProcessTransport (any failure or a reply asking for a
    restart rebuilds the policy), but nothing is isolated: an abandoned LLM call keeps running."""

    def __init__(self, make):
        self.make, self.policy = make, None

    def decide(self, request):
        self.policy = self.policy or self.make()
        try:
            reply = json.loads(json.dumps(self.policy.decide(json.loads(json.dumps(request))), allow_nan=False))
        except Exception as error:  # the worker would report it the same way
            self.policy = None
            return 'worker_error', error_text(error)
        if not isinstance(reply, dict) or reply.get('restart'):
            self.policy = None
        return 'ok', reply

    def close(self):
        self.policy = None


def error_text(error):
    """'Type: message' of an exception, at most 300 characters, without trusting its __str__."""
    try:
        text = str(error)
    except Exception:
        text = '(unprintable)'
    return f'{type(error).__name__}: {text}'[:300]


class NullPolicy:
    """Always the base order."""

    def decide(self, request):
        return {'order': request['base_order'], 'source': 'null'}


class RandomPolicy:
    """Random Adaptive: a uniformly drawn order of the 120 at every decision point.

    Matched-Random (steps given): only at the decision keys [step, phase, sample_trial, side]
    where the paired LLM wall's pick differed from Paper's, a uniformly drawn non-Paper outcome
    class (reached by its smallest order); elsewhere the base order. Draws come from its own
    generator seeded by a non-negative int, never the global RNG.
    """

    def __init__(self, seed, steps=None):
        if type(seed) is not int or seed < 0:
            raise ValueError(f'random controller needs a non-negative int seed, got {seed!r}')
        if steps is not None:
            steps = [list(s) if isinstance(s, (list, tuple)) else s for s in steps]
            if not all(isinstance(s, list) and len(s) == 4 and type(s[0]) is int and isinstance(s[1], str) and type(s[2]) is int
                       and isinstance(s[3], str) for s in steps) or len({tuple(s) for s in steps}) != len(steps):
                raise ValueError('steps must be distinct [step, phase, sample_trial, side] keys')
        self.rng, self.steps = np.random.default_rng(seed), None if steps is None else {tuple(s) for s in steps}

    def decide(self, request):
        obs, orders = request['observation'], list(request['orders'])
        if self.steps is None:
            return {'order': orders[int(self.rng.integers(len(orders)))], 'source': 'random'}
        if decision_key(obs) not in self.steps:
            return {'order': request['base_order'], 'source': 'matched_random:not_matched'}
        others = [o for o in obs['outcomes'] if not o['is_paper']]
        return {'order': others[int(self.rng.integers(len(others)))]['example_order'], 'source': 'matched_random'}


class RulePolicy:
    """Pre-defined rules (draft parameters, to be frozen before any test seed is built).

    scarcity: with <= max_remaining stones left, a sampled stone with <= 1 feasible rotation that
    some order selects is placed now (its outcome with the highest LLC).
    llc_drop: the mean LLC of feasible candidates fell by >= llc_drop against the mean of the
    previous three calls: the outcome with the highest LLC. Otherwise the base order.
    """

    def __init__(self, max_remaining=6, llc_drop=0.2):
        self.max_remaining, self.llc_drop = max_remaining, llc_drop

    def decide(self, request):
        obs, base = request['observation'], request['base_order']
        best = lambda outs: max(outs, key=lambda o: (o['scores']['LLC'], -o['id']))
        scarce = {s['stone'] for s in obs['sampled'] if s['feasible_rotations'] <= 1 and s['reachable']}
        if obs['n_remaining'] <= self.max_remaining and scarce:
            pick, source = best([o for o in obs['outcomes'] if o['stone'] in scarce]), 'rule:scarcity'
        else:
            past = [r['mean_feasible_llc'] for r in obs['memory']['recent'] if r['mean_feasible_llc'] is not None]
            current = obs['now']['mean_feasible_llc']
            if not past or current is None or current > (1 - self.llc_drop) * np.mean(past):
                return {'order': base, 'source': 'rule:none'}
            pick, source = best(obs['outcomes']), 'rule:llc_drop'
        return {'order': base if pick['is_paper'] else pick['example_order'], 'source': source}


SYSTEM = ("You choose the filter order of Wang's progressive above-mean cascade for one stone placement. "
          "Candidates, positions and the LLC > 0 gate are fixed; you can only pick an order (a permutation of "
          "0 NHR, 1 GDS, 2 IWR, 3 LLC, 4 LFR) or keep the base order. Geometry values are pre-stabilization "
          "proxies. Finish with commit_order.")
COMMIT = {'name': 'commit_order', 'description': "Final action: an order 'a-b-c-d-e' or 'keep_paper'.",
          'parameters': {'type': 'object', 'additionalProperties': False, 'required': ['order', 'rationale'],
                         'properties': {'order': {'type': 'string'}, 'rationale': {'type': 'string'}}}}
TOOLS = [{'name': 'list_order_outcomes', 'description': 'Distinct candidates the 120 orders select, with scores and geometry.',
          'parameters': {'type': 'object', 'additionalProperties': False, 'required': [], 'properties': {}}},
         {'name': 'explain_order', 'description': 'Survivors of each cascade stage for one order.',
          'parameters': {'type': 'object', 'additionalProperties': False, 'required': ['order'],
                         'properties': {'order': {'type': 'string'}}}},
         COMMIT]


def reply_problem(reply, tools):
    """Why an LLM reply is not a valid action under the tool schemas, None if it is. Nothing is
    coerced: arguments must be strings, orders 'a-b-c-d-e' permutations, rationale non-empty."""
    if not isinstance(reply, dict):
        return f'reply must be an object, got {type(reply).__name__}'
    if set(reply) - {'thought', 'tool', 'arguments'}:
        return f"unexpected reply fields {sorted(map(repr, set(reply) - {'thought', 'tool', 'arguments'}))}"
    if not isinstance(reply.get('thought'), (str, type(None))):
        return 'thought must be a string'
    if not isinstance(reply.get('tool'), str):
        return f"tool must be a tool name, got {type(reply.get('tool')).__name__}"
    schema = {t['name']: t['parameters'] for t in tools}.get(reply['tool'])
    if schema is None:
        return f"unknown tool {reply['tool']!r}"
    args = reply.get('arguments', {})
    if not isinstance(args, dict):
        return f'arguments must be an object, got {type(args).__name__}'
    missing, extra = set(schema['required']) - set(args), set(args) - set(schema['properties'])
    if missing or extra:
        return f'missing arguments {sorted(missing)}, unexpected arguments {sorted(map(repr, extra))}'
    if any(not isinstance(v, str) for v in args.values()):
        return 'arguments must be strings'
    if 'order' in args and not (reply['tool'] == 'commit_order' and args['order'] == 'keep_paper') \
            and parse_order(args['order']) is None:
        return f"order must be 'a-b-c-d-e', a permutation of 0..4, got {args['order']!r}"
    if reply['tool'] == 'commit_order' and not args['rationale'].strip():
        return 'rationale must not be empty'
    return None


MAX_REPLY_CHARS = 100_000


def safe_json(value, depth=32):
    """A plain JSON copy of an LLM reply for the record and the next message: dicts with str keys,
    lists, str, int, finite float, bool and None; anything else, or nested deeper than `depth`,
    becomes {'__type__': its type name}. No __repr__, __str__ or __eq__ of the reply is called."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return value if np.isfinite(value) else {'__type__': 'float'}
    if depth and type(value) is dict and all(type(k) is str for k in value):
        return {k: safe_json(v, depth - 1) for k, v in value.items()}
    if depth and type(value) in (list, tuple):
        return [safe_json(v, depth - 1) for v in value]
    return {'__type__': type(value).__name__}


def parse_reply(reply):
    """What a policy decides on: the safe_json copy of a reply object, or of the JSON in a reply
    text; text that is no JSON and copies over MAX_REPLY_CHARS become a {'__type__': ...} marker."""
    if isinstance(reply, str):
        try:
            reply = json.loads(reply)
        except (ValueError, RecursionError):
            return {'__type__': 'unparseable text'}
    parsed = safe_json(reply)
    return parsed if len(json.dumps(parsed)) <= MAX_REPLY_CHARS else {'__type__': 'oversized reply'}


def request_sha256(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()


class LLMFailure(Exception):
    """An LLM call that failed with status 'timeout' or 'error' and this text (raised by a client
    with its own deadline, and by ReplayLLM to repeat a recorded failure without waiting)."""

    def __init__(self, status, text):
        super().__init__(text)
        self.status, self.text = status, text


class ReActPolicy:
    """LLM-driven ReAct. llm(messages, tools) -> a reply object {'thought', 'tool', 'arguments'} or
    its JSON text; optional llm.model and llm.params name the model in the transcripts.

    The first message is the observation without outcome classes; with tool_feedback the LLM gets
    them by calling list_order_outcomes / explain_order (answered from the request's tool table).
    tool_feedback=False is the ablation: commit_order is the only tool. The policy decides on the
    parsed copy of each reply (parse_reply), the copy its transcript keeps, so a replay of the
    transcripts reproduces every decision.

    Every LLM call leaves a transcript: call, request {messages, tools, model, params}, its
    request_sha256, response {raw_text, parsed, usage, error}, status and latency_s.

    Fallback rules (the decision then keeps the base order and its record says why):
    - a reply that breaks the tool schema is rejected with the reason as feedback and counts as
      one of the max_tool_calls actions; no valid commit within them: fallback:no_valid_commit;
    - an LLM call that raises or gives no reply within llm_timeout seconds: fallback:llm_error /
      fallback:llm_timeout at once, no retry. A timed-out call cannot be stopped from its thread,
      so the reply asks for a restart: ProcessTransport kills the worker with the call in it.
    The wall budget and the failure limit are the client's (ControllerClient).
    """

    def __init__(self, llm, tool_feedback=True, max_tool_calls=4, llm_timeout=60.0):
        self.llm, self.max_tool_calls, self.llm_timeout = llm, max_tool_calls, llm_timeout
        self.tools = TOOLS if tool_feedback else [COMMIT]
        self.prompt_sha256 = hashlib.sha256(json.dumps([SYSTEM, self.tools]).encode()).hexdigest()

    def call_llm(self, messages, transcripts):
        """('ok', parsed reply) or ('timeout' | 'error', reason); appends the call's transcript."""
        request = {'messages': copy.deepcopy(messages), 'tools': copy.deepcopy(self.tools),
                   'model': safe_json(getattr(self.llm, 'model', None)), 'params': safe_json(getattr(self.llm, 'params', {}))}
        box, args = {}, (copy.deepcopy(messages), copy.deepcopy(self.tools))

        def call():
            try:
                box['reply'] = self.llm(*args)
            except LLMFailure as failure:
                box['failure'] = (failure.status, str(failure.text)[:300])
            except Exception as error:  # any client failure is a failed call, never a decision
                box['failure'] = ('error', error_text(error))
        start = time.monotonic()
        thread = threading.Thread(target=call, name='llm-call', daemon=True)
        thread.start()
        thread.join(self.llm_timeout)
        if thread.is_alive():
            status, out = 'timeout', f'no reply within {self.llm_timeout} s'
        elif 'reply' in box:
            status, out = 'ok', box['reply']
        else:
            status, out = box.get('failure', ('error', 'the call ended without a reply'))
        ok = status == 'ok'
        parsed = parse_reply(out) if ok else None
        transcripts.append({'call': len(transcripts), 'request': request, 'request_sha256': request_sha256(request),
                            'response': {'raw_text': out[:MAX_REPLY_CHARS] if ok and isinstance(out, str) else None,
                                         'parsed': parsed, 'usage': safe_json(getattr(self.llm, 'usage', None)),
                                         'error': None if ok else out},
                            'status': status, 'latency_s': round(time.monotonic() - start, 3)})
        return status, parsed if ok else out

    def tool(self, name, args, request):
        """(feedback, committed order or None) for a reply that passed reply_problem."""
        obs = request['observation']
        if name == 'list_order_outcomes':
            return {'outcomes': obs['outcomes'],
                    'unreachable_sampled': [s['stone'] for s in obs['sampled'] if not s['reachable']]}, None
        order = request['base_order'] if args['order'] == 'keep_paper' else key(parse_order(args['order']))
        if name == 'commit_order':
            return {'ok': True, 'order': order}, order
        return request['orders'][order], None

    def decide(self, request):
        obs, base = request['observation'], request['base_order']
        view = {k: v for k, v in obs.items() if k != 'outcomes'}  # outcomes only through list_order_outcomes
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': json.dumps(view)}]
        steps, transcripts = [], []
        meta = {'prompt_sha256': self.prompt_sha256, 'transcripts': transcripts}
        for _ in range(self.max_tool_calls):
            status, reply = self.call_llm(messages, transcripts)
            if status != 'ok':
                steps.append({'status': status, 'error': reply})
                return {'order': base, 'source': f'fallback:llm_{status}', **meta, 'react': steps,
                        **({'restart': True} if status == 'timeout' else {})}
            try:
                problem = reply_problem(reply, self.tools)
            except Exception as error:  # the validator must never decide by crashing
                problem = f'unreadable reply: {error_text(error)}'
            said = reply if isinstance(reply, dict) else {'reply': reply}
            if problem is None:
                feedback, order = self.tool(reply['tool'], reply.get('arguments', {}), request)
            else:
                feedback, order = {'error': problem}, None
            steps.append({**said, 'status': 'ok' if problem is None else 'rejected', 'feedback': feedback})
            messages += [{'role': 'assistant', 'content': json.dumps(said)},
                         {'role': 'tool', 'name': str(said.get('tool')), 'content': json.dumps(feedback)}]
            if order is not None:
                return {'order': order, 'source': 'llm', **meta, 'react': steps}
        return {'order': base, 'source': 'fallback:no_valid_commit', **meta, 'react': steps}


class ReplayLLM:
    """The LLM of a check: plays back the transcripts of a wall's records, never an API. A call is
    found by its decision key (from the observation message) and its position in the decision; the
    request the policy sends must hash to the recorded request_sha256, and recorded timeouts and
    errors recur at once with their text. Anything else is a divergence: listed in `diverged` and
    the call fails."""

    def __init__(self, records):
        records = [r for r in records if isinstance(r, dict) and isinstance(r.get('transcripts'), list)]
        self.recorded = {decision_key(r): r['transcripts'] for r in records}
        first = next((t['request'] for r in records for t in r['transcripts']), {})
        self.model, self.params, self.diverged = first.get('model'), first.get('params', {}), []

    def __call__(self, messages, tools):
        key_ = decision_key(json.loads(messages[1]['content']))
        call, done = (len(messages) - 2) // 2, self.recorded.get(key_, [])
        t = done[call] if call < len(done) else None
        request = {'messages': messages, 'tools': tools, 'model': self.model, 'params': self.params}
        if t is None or t.get('request_sha256') != request_sha256(request) or request_sha256(t.get('request')) != t['request_sha256']:
            self.diverged.append(f"{key_} call {call}: {'not recorded' if t is None else 'request differs from the recording'}")
            raise LLMFailure('error', 'replay diverged')
        if t['status'] != 'ok':
            raise LLMFailure(t['status'], t['response']['error'])
        raw = t['response']['raw_text']
        return raw if raw is not None else t['response']['parsed']


class ReplayedFailures(InlineTransport):
    """Inline transport for checking an LLM wall: the decisions whose worker failed live fail again
    with the recorded status and text; the others run the policy."""

    def __init__(self, make, records):
        super().__init__(make)
        self.failed = {decision_key(r): (r['source'][len('fallback:'):], r.get('error'))
                       for r in records if isinstance(r, dict) and str(r.get('source')).startswith('fallback:worker_')}

    def decide(self, request):
        failed = self.failed.get(decision_key(request['observation']))
        if failed:
            self.policy = None
            return failed
        return super().decide(request)


class MockLLM:
    """Scripted stand-in for the tool flow (no API): 'max_llc' lists outcomes, explains the
    highest-LLC one and commits it; 'keep' commits keep_paper; 'blind' commits 0-1-2-3-4 without
    looking; 'invalid' commits a non-permutation; 'stone' tries to name a stone. Failure modes,
    max_llc elsewhere: 'hang@N' never answers and 'crash@N' ends its process (exit code 3) at the
    first call of the decisions of step N (read from the observation, so a restart does not move it)."""

    def __init__(self, mode):
        self.mode, self.model, self.params = mode, f'mock:{mode}', {}

    def __call__(self, messages, tools):
        failure, _, step = self.mode.partition('@')
        if step and len(messages) == 2 and json.loads(messages[1]['content'])['step'] == int(step):
            if failure == 'hang':
                threading.Event().wait()
            if failure == 'crash':
                import os
                os._exit(3)
        last = json.loads(messages[-1]['content']) if messages[-1]['role'] == 'tool' else None
        commit = lambda order, why: {'thought': why, 'tool': 'commit_order', 'arguments': {'order': order, 'rationale': why}}
        if self.mode == 'keep':
            return commit('keep_paper', 'keep the base order')
        if self.mode == 'blind':
            return commit('0-1-2-3-4', 'no feedback: fixed guess')
        if self.mode == 'invalid':
            return commit('0-0-1-2-3', 'malformed order')
        if self.mode == 'stone':
            return {'thought': 'force a stone', 'tool': 'commit_order',
                    'arguments': {'order': '3-0-2-1-4', 'rationale': 'force', 'stone': 'S1'}}
        if last is None:
            return {'thought': 'compare what the orders select', 'tool': 'list_order_outcomes', 'arguments': {}}
        if 'outcomes' in last:
            self.best = max(last['outcomes'], key=lambda o: (o['scores']['LLC'], -o['id']))
            return {'thought': f"outcome {self.best['id']} has the highest LLC", 'tool': 'explain_order',
                    'arguments': {'order': self.best['example_order']}}
        return commit(self.best['example_order'], f"place outcome {self.best['id']}")


def make_policy(spec):
    """The decision logic of a controller spec (what runs behind the transport); bad specs raise."""
    spec = full_spec(spec)
    params = {k: v for k, v in spec.items() if k != 'kind'}
    if spec['kind'] == 'null':
        return NullPolicy()
    if spec['kind'] == 'random':
        return RandomPolicy(params.get('seed'), params.get('steps'))
    if spec['kind'] == 'rule':
        return RulePolicy(**params)
    llm = params['llm']
    if not isinstance(llm, dict) or llm.get('kind') != 'mock' or set(llm) != {'kind', 'mode'}:
        raise NotImplementedError(f'only {{"kind": "mock", "mode": ...}} exists before P2, got {llm!r}')
    return ReActPolicy(MockLLM(llm['mode']), **{k: params[k] for k in WORKER_KEYS['react']})


def make_controller(spec, sink=None, transport='process'):
    """Controller from a JSON-able spec, e.g. {'kind': 'random', 'seed': 7}; bad specs raise here,
    before any wall is built. transport 'process' (live walls) runs the policy in a worker
    process (controller_worker.ProcessTransport); 'inline' runs it in this process."""
    make_policy(spec)
    if transport == 'inline':
        return ControllerClient(spec, InlineTransport(lambda: make_policy(spec)), sink)
    import controller_worker
    return ControllerClient(spec, controller_worker.ProcessTransport(full_spec(spec)), sink)


def write_record(record):
    from Stonepacker2D.utils.constant import get_dir
    path = Path(get_dir()) / 'trace' / 'controller.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, default=lambda v: v.item()) + '\n')


def trace_events(result):
    for path in published_events(Path(result) / 'trace'):
        yield load_trace(path)


def run_of(result):
    result = Path(result).resolve()
    return next(p for p in (result, *result.parents) if (p / 'receipt.json').exists())


def base_policy(run):
    """The policy a run was launched with (receipt filter_policy); every decision point starts from it."""
    fp = json.loads((Path(run) / 'receipt.json').read_text(encoding='utf-8'))['filter_policy']
    if fp['policy'] != 'paper' or fp.get('switch'):
        raise ValueError(f'controllers need a paper-policy run without --switch, got {fp}')
    return {'policy': 'paper', 'order': None if fp['order'] is None else tuple(fp['order']), 'tol': fp['tol']}


def rerun(result, spec, base, transport='inline', geometry=None):
    """The records a controller writes when it sees the traced selections of one result directory;
    transport: 'inline', 'process' or a transport object (check() replays LLM walls with one);
    geometry: a LiveGeometry giving the observations the live outcome geometry."""
    records = []
    controller = make_controller(spec, records.append, transport) if isinstance(transport, str) else \
        ControllerClient(spec, transport, records.append)
    controller.observer.geometry = geometry
    try:
        for meta, a in trace_events(result):
            controller(types.MappingProxyType(dict(base)), context_from_trace(meta, a))
    finally:
        controller.transport.close()
    return json.loads(json.dumps(records, default=lambda v: v.item()))


def replay(result, spec):
    """Evaluate a controller offline on a wall built with the base policy alone (the wall keeps its
    recorded path whatever the controller chooses). Walls with interventions are refused: their
    traced orders are applied, not base, and their states carry the earlier controller's actions."""
    run = run_of(result)
    if json.loads((run / 'receipt.json').read_text(encoding='utf-8')).get('controller') is not None:
        raise ValueError(f'{run} was built with a controller; replay needs a base-policy wall')
    base = base_policy(run)
    if any(meta['policy'] != json.loads(json.dumps(base)) for meta, _ in trace_events(result)):
        raise ValueError(f'{result} has selections with a policy other than the base {base}')
    return rerun(result, spec, base)


def comparable(v):
    """A record without what a re-execution cannot reproduce: outcome geometry (it needs the stone
    pixels) and LLM call latencies."""
    if isinstance(v, dict):
        return {k: None if k in ('geometry', 'latency_s') else comparable(x) for k, x in v.items()}
    if isinstance(v, list):
        return [comparable(x) for x in v]
    return v


GEOMETRY_KEYS = {f'd_{k}' for k in wall_state(np.zeros((6, 6)))} | {'stone_px'}


def geometry_ok(g):
    """The schema placement_geometry writes: its keys, each an int, a finite float or None (NaN)."""
    return isinstance(g, dict) and set(g) == GEOMETRY_KEYS and all(
        v is None or (type(v) in (int, float) and np.isfinite(v)) for v in g.values())


def pose_of(outcome):
    """An outcome's candidate: stone label, rotation and evaluation-frame position."""
    return tuple(outcome.get(k) for k in ('stone', 'rotation', 'x', 'y')) if isinstance(outcome, dict) else None


class LiveGeometry:
    """Outcome geometry as a wall's live records hold it, for re-executing the wall from its trace,
    which keeps no stone pixels (Transcript Replay: the policy must see what it saw live). Found by
    decision key and candidate pose (pose_of), never by position or count. fill() lists in
    `problems` a selection without a live record, outcome poses that differ from the re-executed
    ones (missing, extra or duplicated candidates) and geometry that is missing or malformed. This
    only carries the live values over; whether they are right is validate_geometry's question."""

    def __init__(self, records):
        self.by_key, self.problems = {}, []
        for r in records:
            if not (isinstance(r, dict) and isinstance(r.get('observation'), dict)):
                continue
            key, outcomes = decision_key(r), r['observation'].get('outcomes')
            if key in self.by_key:
                self.problems.append(f'two live records for {key}')
                continue
            poses = [pose_of(o) for o in outcomes] if isinstance(outcomes, list) else []
            if len(set(poses)) != len(poses) or None in poses:
                self.problems.append(f'{key}: live outcomes without a pose or with a pose twice')
            self.by_key[key] = {p: o.get('geometry') for p, o in zip(poses, outcomes or []) if p is not None}

    def fill(self, obs):
        key = decision_key(obs)
        recorded = self.by_key.get(key)
        if recorded is None:
            self.problems.append(f'{key}: no live record to take the geometry from')
            return
        if {pose_of(o) for o in obs['outcomes']} != set(recorded):
            self.problems.append(f'{key}: the live outcomes are not the candidates the trace re-executes')
        for o in obs['outcomes']:
            g = recorded.get(pose_of(o))
            if not geometry_ok(g):
                self.problems.append(f'{key}: outcome {o["id"]} {pose_of(o)} has no valid live geometry')
            o['geometry'] = g


def validate_geometry(result, records, events):
    """Geometry Validation, independent of the records: every outcome geometry that can be
    recomputed from the stone pixels base.pkl holds must equal the record's. A placed rock gives its
    stone's world-frame pixel crop; a candidate of that stone whose traced footprint hash equals one
    of the crop's 8 rotations and mirrorings has exactly those pixels, which are pasted at the
    candidate's evaluation-frame (x, y) on the traced wall (placement_geometry). Stones whose rock
    stabilization turned or moved off the pixel grid, stones never placed and a missing base.pkl
    leave outcomes unverifiable; they are counted, not passed."""
    crops, path = {}, Path(result) / 'base.pkl'
    if path.exists():
        with path.open('rb') as f:
            base = pickle.load(f)
        for rock in base.placed_rocks:
            m = np.asarray(rock.matrix) != 0
            if m.any():
                rows, cols = np.nonzero(m)
                crops[int(rock.id)] = m[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    labels = {}  # stone labels as the Observer gives them: S1, S2, ... in order of first sampling
    for _, a in events:
        for s in a['sampled_ids'].tolist():
            labels.setdefault(int(s), f'S{len(labels) + 1}')
    ids = {v: k for k, v in labels.items()}
    by_event, problems, total, verified = {m['event']: (m, a) for m, a in events}, [], 0, 0
    for r in records:
        if not (isinstance(r, dict) and r.get('event') in by_event and isinstance(r.get('observation'), dict)):
            continue
        meta, a = by_event[r['event']]
        for o in r['observation'].get('outcomes') or []:
            total += 1
            stone, rotation, x, y = pose_of(o)
            match = np.nonzero((a['stone_id'] == ids.get(stone, -1)) & (a['rotation'] == rotation) & (a['x'] == x) & (a['y'] == y))[0]
            if len(match) != 1:
                problems.append(f"event {r['event']}: outcome {pose_of(o)} is not one candidate of the trace")
                continue
            crop, world = crops.get(ids[stone]), None
            for k in range(8 if crop is not None else 0):
                t = np.rot90(crop, k % 4)
                t = np.fliplr(t) if k >= 4 else t
                if footprint(t, 'left') == a['footprint'][match[0]]:
                    world = t
                    break
            if world is None:
                continue
            wall = np.asarray(a['wall'])
            shape = world if meta['side'] == 'left' else np.fliplr(world)
            piece = shape[:wall.shape[0] - y, :wall.shape[1] - x]
            pixels = np.zeros(wall.shape, bool)
            pixels[y:y + piece.shape[0], x:x + piece.shape[1]] = piece
            if placement_geometry(wall, pixels) != o.get('geometry'):
                problems.append(f"event {r['event']}: geometry of outcome {pose_of(o)} differs from its recomputation from base.pkl")
            else:
                verified += 1
    return {'outcomes': total, 'verified': verified, 'mismatched': len(problems), 'unverifiable': total - verified - len(problems),
            'problems': problems}


def check(result, records=None, spec=None, stopped=None):
    """Controller records of a built wall against its receipt and trace. The controller named in
    the receipt (spec only for tests on walls without one) is re-executed on the traced inputs from
    the receipt's base policy; every record must equal its re-execution (geometry excluded: it
    needs stone pixels), the traced policy must be the base with the re-executed order, replaying
    select with it must give the traced pick, decision keys must be unique and the trace itself
    must be complete and consistent (check_case.chain_problems). A stopped run (stopped; by
    default read from the receipt) may hold one record more than events: the controller is called
    before its event is written. Counts come from the re-execution, not from the records.
    Records holding live outcome geometry are re-executed with it (LiveGeometry: Transcript
    Replay) and that geometry is recomputed where base.pkl allows (validate_geometry: Geometry
    Validation); report['geometry'] counts both."""
    result, run = Path(result), run_of(result)
    receipt = json.loads((run / 'receipt.json').read_text(encoding='utf-8'))
    spec = receipt.get('controller') if spec is None else spec
    stopped = receipt.get('process_status') == 'STOPPED_AFTER_PLACEMENTS' if stopped is None else stopped
    if records is None:
        path = result / 'trace' / 'controller.jsonl'
        records = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    if spec is None:
        return {'ok': False, 'problems': ['the receipt names no controller'], 'records': len(records),
                'decision_points': None, 'switches': None}
    base = base_policy(run)
    llm, transport = None, 'inline'
    if full_spec(spec)['kind'] == 'react':  # the LLM answers from the records' transcripts, never an API
        llm, worker = ReplayLLM(records), {k: full_spec(spec)[k] for k in WORKER_KEYS['react']}
        transport = ReplayedFailures(lambda: ReActPolicy(llm, **worker), records)
    live = any(isinstance(r, dict) and isinstance(r.get('observation'), dict) and isinstance(r['observation'].get('outcomes'), list)
               and any(isinstance(o, dict) and o.get('geometry') is not None for o in r['observation']['outcomes']) for r in records)
    geometry = LiveGeometry(records) if live else None
    expected, events = rerun(result, spec, base, transport, geometry), list(trace_events(result))
    problems = chain_problems(result, events, stopped)
    if not len(events) <= len(records) <= len(events) + int(stopped):
        problems.append(f'{len(records)} records for {len(events)} selections')
    keys = [decision_key(x) for x in expected]
    if len(set(keys)) != len(keys):
        problems.append('decision keys (step, phase, sample_trial, side) are not unique')
    rest = lambda p: {k: v for k, v in p.items() if k != 'order'}
    # records written before P2 carry no tool_table_sha256: compared without it and reported as legacy
    legacy = bool(records) and not any(isinstance(r, dict) and 'tool_table_sha256' in r for r in records)
    if legacy:
        expected = [{k: v for k, v in x.items() if k != 'tool_table_sha256'} for x in expected]
    for r, x, (meta, a) in zip(records, expected, events):
        where = f"event {meta['event']} (step {meta['step']} {meta['phase']} trial {meta['sample_trial']} {meta['side']})"
        if (x['event'], *decision_key(x)) != (meta['event'], meta['step'], meta['phase'], meta['sample_trial'], meta['side']):
            problems.append(f'{where}: controller call {x["event"]} is for {decision_key(x)}')
        if rest(meta['policy']) != rest(json.loads(json.dumps(base))) or \
                list(meta['policy']['order'] or PAPER_ORDER) != list(parse_order(x['final']['order'])):
            problems.append(f"{where}: traced policy {meta['policy']}, base {base}, controller order {x['final']['order']}")
        index = select(a['features_raw'], a['weights'], 'paper', parse_order(x['final']['order']), base['tol']).index
        if (-1 if index is None else int(index)) != int(a['selected']) or x['final']['index'] != (None if index is None else int(index)):
            problems.append(f"{where}: replay selects {index}, traced {int(a['selected'])}, controller {x['final']['index']}")
        if comparable(r) != comparable(x):
            problems.append(f'{where}: record differs from the controller re-executed on the trace')
    for r in records[len(events):len(events) + 1]:
        problems += [f'trailing record: {p}' for p in trailing_problems(r, spec, events)]
    problems += [f'replay diverged at {d}' for d in (llm.diverged[:5] if llm else [])]
    failed = [r['event'] for r in records if isinstance(r, dict) and str(r.get('source')).startswith('fallback:worker_')]
    if failed and full_spec(spec)['kind'] != 'react':
        problems.append(f'the worker of this deterministic controller failed at events {failed[:5]}: '
                        'its decisions are not the controller ones, the wall is no valid result')
    validation = None
    if geometry is not None:
        problems += [f'live geometry: {p}' for p in geometry.problems[:5]]
        validation = validate_geometry(result, records, events)
        problems += validation.pop('problems')[:5]
    report = {'ok': not problems, 'problems': problems, 'records': len(records),
              'geometry': {'source': 'live records' if live else 'none recorded',
                           'replay_problems': None if geometry is None else len(geometry.problems), 'validation': validation},
              'decision_points': sum(x['decision_point'] for x in expected),
              'switches': sum(x['final']['switched'] for x in expected)}
    if len(records) > len(events):
        report['unverified_trailing_records'] = len(records) - len(events)
    if legacy:
        report['legacy_records'] = True
    return report


RECORD_KEYS = {'event', 'step', 'phase', 'sample_trial', 'side', 'controller', 'decision_point', 'n_outcomes',
               'observation', 'source', 'final'}


def trailing_problems(record, spec, events):
    """The record of a selection whose event a stopped run never wrote cannot be re-executed; it must
    still be the next call of the receipt's controller: the next event number, a well-formed record
    of that spec whose observation is for its own decision key and, if the last traced attempt has
    its left side only, that attempt's right side."""
    if not isinstance(record, dict) or not RECORD_KEYS <= set(record):
        return ['not a controller record']
    problems, obs, final = [], record['observation'], record['final']
    if record['event'] != len(events) or record['controller'] != spec:
        problems.append(f"event {record['event']} of {record['controller']}, expected {len(events)} of {spec}")
    if not isinstance(obs, dict) or decision_key(record) != tuple(obs.get(k) for k in ('step', 'phase', 'sample_trial', 'side')):
        problems.append(f'its observation is not for {decision_key(record)}')
    if not isinstance(final, dict) or parse_order(final.get('order')) is None:
        problems.append('its final order is not a permutation of 0..4')
    if events:
        meta = events[-1][0]
        if meta['construction_order'] == 'two_sides' and meta['side'] == 'left' and \
                decision_key(record) != (meta['step'], meta['phase'], meta['sample_trial'], 'right'):
            problems.append(f"it is not the right side of the attempt of event {meta['event']}")
    return problems


OFFLINE = [{'kind': 'null'}, {'kind': 'random', 'seed': 1}, {'kind': 'rule'},
           {'kind': 'react', 'llm': {'kind': 'mock', 'mode': 'max_llc'}},
           {'kind': 'react', 'llm': {'kind': 'mock', 'mode': 'blind'}, 'tool_feedback': False},
           {'kind': 'react', 'llm': {'kind': 'mock', 'mode': 'invalid'}}]


def offline(manifest, out):
    import pandas as pd
    import run_case
    rows = []
    with (out / 'records.jsonl').open('w', encoding='utf-8') as f:
        for w in manifest.itertuples():
            result = next((run_case.RUNS / w.run).rglob('trace/commits.jsonl')).parent.parent
            llm_switches = None
            for spec in OFFLINE + [{'kind': 'random', 'seed': 1, 'matched': True}]:
                if spec.get('matched'):  # paired with the max_llc mock's switches on this wall
                    spec = {'kind': 'random', 'seed': 1, 'steps': llm_switches}
                records = replay(result, spec)
                if spec.get('llm', {}).get('mode') == 'max_llc':
                    llm_switches = [list(decision_key(r)) for r in records if r['final']['switched']]
                for r in records:
                    f.write(json.dumps({'seed': w.seed, **r}) + '\n')
                    rows.append({'seed': w.seed, 'controller': json.dumps(spec, sort_keys=True)[:90],
                                 'decision_point': r['decision_point'], 'switched': r['final']['switched'],
                                 'source': r.get('source'), 'tool_calls': len(r.get('react', [])),
                                 'n_outcomes': r['n_outcomes']})
    df = pd.DataFrame(rows)
    df.to_csv(out / 'decisions.csv', index=False)
    return df


def main(argv=None):
    import pandas as pd
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    off = sub.add_parser('offline')
    off.add_argument('id')
    off.add_argument('out', type=Path)
    off.add_argument('--arm', default='paper_3-0-2-1-4')
    chk = sub.add_parser('check')
    chk.add_argument('run', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'check':
        import check_case
        reports = [check(r) for r in check_case.result_dirs(args.run)]
        print(json.dumps(reports, indent=2))
        return 0 if reports and all(r['ok'] for r in reports) else 1
    import campaign
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = pd.read_csv(campaign.CAMPAIGNS / args.id / 'manifest.csv')
    df = offline(manifest[(manifest.arm == args.arm) & (manifest.status == 'PROCESS_PASS')], args.out)
    pd.set_option('display.width', 250)
    print(df.groupby('controller').agg(calls=('switched', 'size'), decision_points=('decision_point', 'sum'),
                                       switches=('switched', 'sum'), tool_calls=('tool_calls', 'sum')).to_string())
    print(df.groupby(['controller', 'source']).size().to_string())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
