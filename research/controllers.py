"""Dynamic filter-order controllers for the paper policy (P0/P1: no real LLM).

Usage: python -B research/controllers.py offline <campaign id> <out dir> [--arm paper_3-0-2-1-4]
       python -B research/controllers.py check <run>

build_wall calls a controller before every selection (selection.controlled_policy) with a
read-only copy of the base policy and detached copies of that selection's inputs; it returns the
policy for this one selection: the base (Paper) order or another permutation. Every decision
point starts from the run's base policy (receipt filter_policy), never from an earlier choice.
Candidate generation, image convolution, the LLC > 0 gate and the stability analysis are not
touched; a controller can only pick what some filter order selects (an outcome class).

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
import re
import sys
import threading
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from failure_analysis import FILTERS, wall_state  # noqa: E402

from Stonepacker2D.stack.selection import ALL_ORDERS, PAPER_ORDER, detached_inputs, load_trace, select, to_world  # noqa: E402

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

    def __init__(self):
        self.labels, self.stones, self.recent = {}, {}, []

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
        by_index = {}
        for order in ALL_ORDERS:
            index = select(raw, weights, 'paper', order, tol).index
            if index is not None:
                by_index.setdefault(int(index), []).append(order)
        paper = select(raw, weights, 'paper', base, tol).index
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
        for s in obs['sampled']:  # memory is updated after the observation, so it only holds the past
            m = self.stones.setdefault(s['stone'], {'times_sampled': 0, 'min_feasible_rotations': 4})
            m.update(times_sampled=m['times_sampled'] + 1, last_step=obs['step'],
                     last_feasible_rotations=s['feasible_rotations'], last_best_llc=s['best_llc'],
                     min_feasible_rotations=min(m['min_feasible_rotations'], s['feasible_rotations']))
        self.recent = (self.recent + [{'step': obs['step'], **obs['now']}])[-3:]
        info = {'classes': classes, 'by_index': by_index, 'paper': paper, 'base': base, 'stone_ids': stone, 'tol': tol}
        return obs, info


def decision_key(record):
    """Identifies one selection across walls: probe and place of a sample_twice step share step and trial."""
    return (record['step'], record['phase'], record['sample_trial'], record['side'])


class Controller:
    """Null controller: always the base order. Subclasses override decide()."""
    kind = 'null'

    def __init__(self, sink=None, **params):
        self.params, self.observer, self.sink, self.calls = params, Observer(), sink or write_record, 0

    def spec(self):
        return {'kind': self.kind, **self.params}

    def decide(self, obs, info, ctx):
        return info['base'], {'source': 'null'}

    def __call__(self, policy, ctx):
        obs, info = self.observer(policy, ctx)
        decision_point = len(info['classes']) >= 2
        order, detail = self.decide(obs, info, ctx) if decision_point else (info['base'], {'source': 'not_a_decision_point'})
        index = select(np.asarray(ctx['features_raw']), ctx['weights'], 'paper', order, policy['tol']).index
        index = None if index is None else int(index)
        self.sink({'event': self.calls, 'step': obs['step'], 'phase': obs['phase'], 'sample_trial': obs['sample_trial'],
                   'side': obs['side'], 'controller': self.spec(), 'decision_point': decision_point,
                   'n_outcomes': len(info['classes']), 'observation': obs, **detail,
                   'final': {'order': key(order), 'index': index, 'switched': index != info['paper'],
                             'outcome': info['classes'].index(index) if index in info['classes'] else None,
                             'stone_id': None if index is None else int(info['stone_ids'][index])}})
        self.calls += 1
        return policy if order == info['base'] else {**policy, 'order': order}


class RandomController(Controller):
    """Random Adaptive: a uniformly drawn order of the 120 at every decision point.

    Matched-Random (steps given): only at the decision keys [step, phase, sample_trial, side]
    where the paired LLM wall's pick differed from Paper's, a uniformly drawn non-Paper outcome
    class (reached by its smallest order); elsewhere the base order. Draws come from its own
    generator seeded by a non-negative int, never the global RNG.
    """
    kind = 'random'

    def __init__(self, seed, steps=None, sink=None):
        if type(seed) is not int or seed < 0:
            raise ValueError(f'random controller needs a non-negative int seed, got {seed!r}')
        if steps is not None:
            steps = [list(s) for s in steps]
            if not all(len(s) == 4 and type(s[0]) is int and isinstance(s[1], str) and type(s[2]) is int
                       and isinstance(s[3], str) for s in steps) or len({tuple(s) for s in steps}) != len(steps):
                raise ValueError('steps must be distinct [step, phase, sample_trial, side] keys')
        super().__init__(sink, seed=seed, **({} if steps is None else {'steps': steps}))
        self.rng, self.steps = np.random.default_rng(seed), None if steps is None else {tuple(s) for s in steps}

    def decide(self, obs, info, ctx):
        if self.steps is None:
            return ALL_ORDERS[int(self.rng.integers(len(ALL_ORDERS)))], {'source': 'random'}
        if decision_key(obs) not in self.steps:
            return info['base'], {'source': 'matched_random:not_matched'}
        others = [i for i in info['classes'] if i != info['paper']]
        pick = others[int(self.rng.integers(len(others)))]
        return min(info['by_index'][pick]), {'source': 'matched_random'}


class RuleController(Controller):
    """Pre-defined rules (draft parameters, to be frozen before any test seed is built).

    scarcity: with <= max_remaining stones left, a sampled stone with <= 1 feasible rotation that
    some order selects is placed now (its outcome with the highest LLC).
    llc_drop: the mean LLC of feasible candidates fell by >= llc_drop against the mean of the
    previous three calls: the outcome with the highest LLC. Otherwise the base order.
    """
    kind = 'rule'

    def __init__(self, max_remaining=6, llc_drop=0.2, sink=None):
        super().__init__(sink, max_remaining=max_remaining, llc_drop=llc_drop)

    def decide(self, obs, info, ctx):
        best = lambda outs: max(outs, key=lambda o: (o['scores']['LLC'], -o['id']))
        scarce = {s['stone'] for s in obs['sampled'] if s['feasible_rotations'] <= 1 and s['reachable']}
        if obs['n_remaining'] <= self.params['max_remaining'] and scarce:
            pick, source = best([o for o in obs['outcomes'] if o['stone'] in scarce]), 'rule:scarcity'
        else:
            past = [r['mean_feasible_llc'] for r in obs['memory']['recent'] if r['mean_feasible_llc'] is not None]
            current = obs['now']['mean_feasible_llc']
            if not past or current is None or current > (1 - self.params['llc_drop']) * np.mean(past):
                return info['base'], {'source': 'rule:none'}
            pick, source = best(obs['outcomes']), 'rule:llc_drop'
        if pick['is_paper']:
            return info['base'], {'source': source}
        return parse_order(pick['example_order']), {'source': source}


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


def loggable(reply):
    """A reply as a JSON object for the record and the next message; {'reply': repr} if it is not one."""
    try:
        said = json.loads(json.dumps(reply, default=repr))
    except (TypeError, ValueError):  # non-string keys of mixed types, cycles
        said = None
    return said if isinstance(said, dict) and all(isinstance(k, str) for k in reply) else {'reply': repr(reply)}


class ReActController(Controller):
    """LLM-driven ReAct. llm(messages, tools) -> {'thought', 'tool', 'arguments'}.

    The first message is the observation without outcome classes; with tool_feedback the LLM gets
    them by calling list_order_outcomes / explain_order. tool_feedback=False is the ablation:
    commit_order is the only tool.

    Fallback rules (the decision then uses the base order and its record says why):
    - a reply that breaks the tool schema is rejected with the reason as feedback and counts as
      one of the max_tool_calls actions; no valid commit within them: fallback:no_valid_commit;
    - an LLM call that raises or gives no reply within llm_timeout seconds: fallback:llm_error /
      fallback:llm_timeout at once, no retry; while a timed-out call is still running no new call
      starts: fallback:llm_busy; after max_llm_failures such failures in a wall every later
      decision point: fallback:llm_unavailable;
    - more than max_decisions LLM decisions in a wall: fallback:wall_budget.
    Calls run in a daemon thread, so a timed-out call holds up neither the wall nor process exit,
    but it is abandoned, not killed: a real client must also set its own request deadline.
    """
    kind = 'react'

    def __init__(self, llm, llm_spec=None, tool_feedback=True, max_tool_calls=4, max_decisions=25,
                 llm_timeout=60.0, max_llm_failures=3, sink=None):
        super().__init__(sink, llm=llm_spec, tool_feedback=tool_feedback, max_tool_calls=max_tool_calls,
                         max_decisions=max_decisions, llm_timeout=llm_timeout, max_llm_failures=max_llm_failures)
        self.llm, self.decisions, self.failures, self.pending = llm, 0, 0, None
        self.tools = TOOLS if tool_feedback else [COMMIT]
        self.prompt_sha256 = hashlib.sha256(json.dumps([SYSTEM, self.tools]).encode()).hexdigest()

    def call_llm(self, messages):
        """('ok', reply) or ('timeout' | 'error' | 'busy', reason)."""
        if self.pending is not None and self.pending.is_alive():
            return 'busy', 'an earlier timed-out call is still running'
        box, args = {}, (copy.deepcopy(messages), copy.deepcopy(self.tools))

        def call():
            try:
                box['reply'] = self.llm(*args)
            except Exception as error:  # any client failure is a failed call, never a decision
                box['error'] = f'{type(error).__name__}: {error}'
        thread = threading.Thread(target=call, name='llm-call', daemon=True)
        thread.start()
        thread.join(self.params['llm_timeout'])
        if thread.is_alive():
            self.pending = thread
            return 'timeout', f"no reply within {self.params['llm_timeout']} s"
        if 'reply' not in box:
            return 'error', box.get('error', 'the call ended without a reply')
        return 'ok', box['reply']

    def tool(self, name, args, obs, info, ctx):
        """(feedback, committed order or None) for a reply that passed reply_problem."""
        if name == 'list_order_outcomes':
            return {'outcomes': obs['outcomes'],
                    'unreachable_sampled': [s['stone'] for s in obs['sampled'] if not s['reachable']]}, None
        order = info['base'] if args['order'] == 'keep_paper' else parse_order(args['order'])
        if name == 'commit_order':
            return {'ok': True, 'order': key(order)}, order
        s = select(np.asarray(ctx['features_raw']), ctx['weights'], 'paper', order, info['tol'])
        label = lambda i: f"{self.observer.label(info['stone_ids'][i])}/{i // len(ctx['sampled_ids']) * 90}"
        return {'branch': s.branch, 'selects_outcome': info['classes'].index(int(s.index)) if s.index is not None else None,
                'stages': [{'after': 'LLC>0 gate' if k == 0 else FILTERS[order[k - 1]], 'survivors': [label(int(i)) for i in st]}
                           for k, st in enumerate(s.stages)]}, None

    def decide(self, obs, info, ctx):
        if self.failures >= self.params['max_llm_failures']:
            return info['base'], {'source': 'fallback:llm_unavailable'}
        if self.decisions >= self.params['max_decisions']:
            return info['base'], {'source': 'fallback:wall_budget'}
        self.decisions += 1
        view = {k: v for k, v in obs.items() if k != 'outcomes'}  # outcomes only through list_order_outcomes
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': json.dumps(view)}]
        steps, meta = [], {'prompt_sha256': self.prompt_sha256}
        for _ in range(self.params['max_tool_calls']):
            status, reply = self.call_llm(messages)
            if status != 'ok':
                self.failures += 1
                steps.append({'status': status, 'error': reply})
                return info['base'], {'source': f'fallback:llm_{status}', **meta, 'react': steps}
            try:
                problem = reply_problem(reply, self.tools)
            except Exception as error:  # the validator must never decide by crashing
                problem = f'unreadable reply: {type(error).__name__}: {error}'
            said = loggable(reply)
            if problem is None:
                feedback, order = self.tool(reply['tool'], reply.get('arguments', {}), obs, info, ctx)
            else:
                feedback, order = {'error': problem}, None
            steps.append({**said, 'status': 'ok' if problem is None else 'rejected', 'feedback': feedback})
            messages += [{'role': 'assistant', 'content': json.dumps(said)},
                         {'role': 'tool', 'name': str(said.get('tool')), 'content': json.dumps(feedback)}]
            if order is not None:
                return order, {'source': 'llm', **meta, 'react': steps}
        return info['base'], {'source': 'fallback:no_valid_commit', **meta, 'react': steps}


class MockLLM:
    """Scripted stand-in for the tool flow (no API): 'max_llc' lists outcomes, explains the
    highest-LLC one and commits it; 'keep' commits keep_paper; 'blind' commits 0-1-2-3-4 without
    looking; 'invalid' commits a non-permutation; 'stone' tries to name a stone."""

    def __init__(self, mode):
        self.mode = mode

    def __call__(self, messages, tools):
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


SPEC_KEYS = {'null': set(), 'random': {'seed', 'steps'}, 'rule': {'max_remaining', 'llc_drop'},
             'react': {'llm', 'tool_feedback', 'max_tool_calls', 'max_decisions', 'llm_timeout', 'max_llm_failures'}}


def make_controller(spec, sink=None):
    """Controller from a JSON-able spec, e.g. {'kind': 'random', 'seed': 7}; unknown keys raise."""
    if not isinstance(spec, dict) or spec.get('kind') not in SPEC_KEYS:
        raise ValueError(f'controller spec needs a kind in {sorted(SPEC_KEYS)}, got {spec!r}')
    params = {k: v for k, v in spec.items() if k != 'kind'}
    if set(params) - SPEC_KEYS[spec['kind']]:
        raise ValueError(f"unknown {spec['kind']} controller keys {sorted(set(params) - SPEC_KEYS[spec['kind']])}")
    if spec['kind'] == 'null':
        return Controller(sink)
    if spec['kind'] == 'random':
        return RandomController(params.get('seed'), params.get('steps'), sink)
    if spec['kind'] == 'rule':
        return RuleController(sink=sink, **params)
    llm = params.pop('llm', None)
    if not isinstance(llm, dict) or llm.get('kind') != 'mock' or set(llm) != {'kind', 'mode'}:
        raise NotImplementedError(f'only {{"kind": "mock", "mode": ...}} exists before P2, got {llm!r}')
    return ReActController(MockLLM(llm['mode']), llm, sink=sink, **params)


def write_record(record):
    from Stonepacker2D.utils.constant import get_dir
    path = Path(get_dir()) / 'trace' / 'controller.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, default=lambda v: v.item()) + '\n')


def trace_events(result):
    for path in sorted((Path(result) / 'trace').glob('e*.npz')):
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


def rerun(result, spec, base):
    """The records a controller writes when it sees the traced selections of one result directory."""
    records = []
    controller = make_controller(spec, records.append)
    for meta, a in trace_events(result):
        controller(types.MappingProxyType(dict(base)), context_from_trace(meta, a))
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


def strip_geometry(v):
    if isinstance(v, dict):
        return {k: None if k == 'geometry' else strip_geometry(x) for k, x in v.items()}
    if isinstance(v, list):
        return [strip_geometry(x) for x in v]
    return v


def trace_problems(result, events, stopped=False):
    """The trace a controller check relies on, independent of the controller: events numbered 0..n-1
    and each part of a commit in commits.jsonl (a stopped run may end in events of an unfinished
    attempt), candidate i is sampled stone i % n at rotation i // n * 90, the sampled and remaining
    stones are the inventory left by earlier placements, world poses follow to_world, and each
    placed commit names the pose of the candidate it points at."""
    problems, numbers = [], [m['event'] for m, _ in events]
    if numbers != list(range(len(events))):
        problems.append(f'trace events are not numbered 0..{len(events) - 1}')
    path = Path(result) / 'trace' / 'commits.jsonl'
    commits = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    if not path.exists() and not stopped:
        problems.append('no trace/commits.jsonl')
    referenced = {s['event'] for c in commits for s in c['sides'].values()}
    if referenced - set(numbers):
        problems.append(f'commits reference missing events {sorted(referenced - set(numbers))[:5]}')
    loose = set(numbers) - referenced
    if loose and not (stopped and min(loose) > max(referenced, default=-1)):
        problems.append(f'events {sorted(loose)[:5]} belong to no commit')
    by_event = dict(zip(numbers, events))
    placed = {max(s['event'] for s in c['sides'].values()): c for c in commits if c['outcome'] == 'placed'}
    start, gone = None, set()
    for meta, a in events:
        where, n, i = f"event {meta['event']}", len(a['sampled_ids']), np.arange(len(a['stone_id']))
        sampled, remaining = set(a['sampled_ids'].tolist()), set(a['inventory_ids'].tolist())
        start = sampled | remaining if start is None else start
        if not n or not ((a['stone_id'] == a['sampled_ids'][i % n]).all() and (a['rotation'] == i // n * 90).all()):
            problems.append(f'{where}: candidate stones and rotations do not follow the sampled stones')
        if sampled & remaining or sampled | remaining != start - gone:
            problems.append(f'{where}: sampled and remaining stones are not the inventory left by earlier placements')
        x_world, rotation_world = to_world(meta['side'], a['x'], a['rotation'], a['width'], meta['world_width'])
        if not (np.array_equal(x_world, a['x_world']) and np.array_equal(rotation_world, a['rotation_world'])):
            problems.append(f'{where}: world poses do not follow the evaluation-frame poses')
        if meta['event'] in placed:
            gone.add(placed[meta['event']]['stone_id'])
    for c in placed.values():
        side = c['sides'][c['chosen_side']]
        if side['event'] in by_event:
            a, k = by_event[side['event']][1], side['index']
            pose = (int(a['stone_id'][k]), int(a['x_world'][k]), int(a['y'][k]), int(a['rotation_world'][k]))
            if pose != (c['stone_id'], c['x_world'], c['y'], c['rotation_world']):
                problems.append(f"step {c['step']}: commit pose differs from its candidate {pose}")
    return problems


def check(result, records=None, spec=None, stopped=None):
    """Controller records of a built wall against its receipt and trace. The controller named in
    the receipt (spec only for tests on walls without one) is re-executed on the traced inputs from
    the receipt's base policy; every record must equal its re-execution (geometry excluded: it
    needs stone pixels), the traced policy must be the base with the re-executed order, replaying
    select with it must give the traced pick, decision keys must be unique and the trace itself
    must be complete and consistent (trace_problems). A run stopped by the launcher (stopped; by
    default read from the receipt) may hold one record more than events: the controller is called
    before its event is written. Counts come from the re-execution, not from the records."""
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
    expected, events = rerun(result, spec, base), list(trace_events(result))
    problems = trace_problems(result, events, stopped)
    if not len(events) <= len(records) <= len(events) + int(stopped):
        problems.append(f'{len(records)} records for {len(events)} selections')
    keys = [decision_key(x) for x in expected]
    if len(set(keys)) != len(keys):
        problems.append('decision keys (step, phase, sample_trial, side) are not unique')
    rest = lambda p: {k: v for k, v in p.items() if k != 'order'}
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
        if strip_geometry(r) != strip_geometry(x):
            problems.append(f'{where}: record differs from the controller re-executed on the trace')
    report = {'ok': not problems, 'problems': problems, 'records': len(records),
              'decision_points': sum(x['decision_point'] for x in expected),
              'switches': sum(x['final']['switched'] for x in expected)}
    if len(records) > len(events):
        report['unverified_trailing_records'] = len(records) - len(events)
    return report


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
