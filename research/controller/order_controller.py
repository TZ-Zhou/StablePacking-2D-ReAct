"""Dynamic filter-order controllers for the paper policy (P2 architecture, mock LLM only).

Usage: python -B -m research.controller.order_controller offline <campaign id> <out dir> [--arm paper_3-0-2-1-4]
       python -B -m research.controller.order_controller check <run>

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
final order with the candidate it selects; live records also keep each outcome's candidate pixels
(outcome_pixels). Every request the client sends is one line of trace/transport.jsonl, the
client's own ledger (request hash, transport status, reply hash), next to the worker lifecycle.
check() re-executes the controller named in the receipt on the traced inputs and compares every
record and ledger line; it reports replay consistency (ok), Geometry Validation and provenance
(ledger, run manifest) separately, and only all three complete give the verdict PASS.
"""
import argparse
import base64
import copy
import hashlib
import json
import os
import pickle
import re
import threading
import time
import types
from pathlib import Path

import numpy as np

from research.analysis.unplaced_stones import FILTERS, wall_state

from research.walls.check_run import chain_problems
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
    """The selection inputs of a trace event as a controller receives them live (detached, no stone
    pixels), plus the candidates' traced bounding-box width and height (read by observation version 2
    only; live, it takes them from the bricks)."""
    return detached_inputs({'step': meta['step'], 'side': meta['side'], 'phase': meta['phase'],
                            'sample_trial': meta['sample_trial'], 'wall': a['wall'], 'all_bricks': None,
                            'sampled_ids': a['sampled_ids'].tolist(), 'inventory_ids': a['inventory_ids'].tolist(),
                            'x': a['x'], 'y': a['y'], 'distance': a['distance'],
                            'features_raw': a['features_raw'], 'weights': a['weights'],
                            'width': a['width'], 'height': a['height']})


def plain(v):
    """Copy with NaN/inf -> None, so observations are plain JSON and compare equal after a round trip."""
    if isinstance(v, dict):
        return {k: plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    return None if isinstance(v, float) and not np.isfinite(v) else v


def encode_pixels(mask):
    """A candidate's pixels as a record keeps them: origin (x, y) and shape of their bounding box
    and the box packed into base64; None without pixels."""
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return None
    crop = mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    return {'origin': [int(cols.min()), int(rows.min())], 'shape': [int(n) for n in crop.shape],
            'bits': base64.b64encode(np.packbits(crop)).decode()}


def decode_pixels(p, shape):
    """The wall-sized mask of encode_pixels' output, None if it is malformed, off the wall or not
    canonical (its box not the tight box of its pixels, padding bits set, other base64): then its
    origin is where its pixels are."""
    try:
        (x, y), (h, w) = p['origin'], p['shape']
        bits = np.unpackbits(np.frombuffer(base64.b64decode(p['bits'], validate=True), np.uint8))
        if not (set(p) == {'origin', 'shape', 'bits'} and all(type(v) is int for v in (x, y, h, w)) and h > 0 and w > 0
                and len(bits) == -(-h * w // 8) * 8 and 0 <= y <= shape[0] - h and 0 <= x <= shape[1] - w):
            return None
        mask = np.zeros(shape, bool)
        mask[y:y + h, x:x + w] = bits[:h * w].reshape(h, w).astype(bool)
        return mask if encode_pixels(mask) == p else None
    except Exception:  # a malformed record is a failed check, never a crash
        return None


def placement_geometry(wall, pixels):
    """Change of wall_state when pixels (bool, wall-sized) are filled; no physics, no stabilization."""
    after = wall.copy()
    after[pixels & (after == 0)] = 1
    before, now = wall_state(wall), wall_state(after)
    return plain({f'd_{k}': round(float(now[k] - before[k]), 4) for k in before} | {'stone_px': int(pixels.sum())})


GEOMETRY_NOTES = {
    'pixels': "outcome.geometry is the change of wall_state after pasting the candidate's own pixels at its pose: "
              "a pre-stabilization proxy without physics. outcome.shape is the candidate's bounding box.",
    'summary': "Only outcome.shape (bounding-box width and height of the candidate in its rotation) is known; there are "
               "no stone pixels, so outcome.geometry is null. It is not the wall after placement.",
    'none': 'No candidate pixels or sizes are available: outcome.geometry and outcome.shape are null.'}


def candidate_sizes(ctx):
    """(widths, heights) of every candidate's bounding box: from the bricks live, from the trace's
    width/height arrays offline (context_from_trace; the same brick attributes); None without either."""
    if ctx['all_bricks'] is not None:
        return [int(b.width) for b in ctx['all_bricks']], [int(b.height) for b in ctx['all_bricks']]
    if ctx.get('width') is not None and ctx.get('height') is not None:
        return [int(w) for w in ctx['width']], [int(h) for h in ctx['height']]
    return None


class Observer:
    """Builds the controller-visible observation and keeps the controller's memory of earlier calls.

    version 1 is the observation of every wall built before observation_version existed. version 2
    adds, from the current selection and the controller's own earlier calls only: the filter
    weights, the wall size, each outcome's bounding box (shape), which geometry exists
    (geometry_available: pixels live, summary offline, none) with its note, the remaining stones
    (sizes of those already sampled, the count never seen), up to 10 earlier calls (history) and
    the change since the last one (trend). Every stone-keyed list or mapping is in label order, so
    the serialized observation is the same under any renaming of the stone ids."""

    def __init__(self, geometry=None, version=1):
        self.labels, self.stones, self.recent = {}, {}, []
        self.geometry = geometry  # LiveGeometry when re-executing a live wall from its trace
        self.version, self.sizes, self.history = version, {}, []

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
        outcomes, pixels = [], None if ctx['all_bricks'] is None else []
        for k, i in enumerate(classes):
            geometry = None
            if ctx['all_bricks'] is not None:
                moved = ctx['all_bricks'][i].transformed([ctx['x'][i], ctx['y'][i]])
                geometry = None if moved is None else placement_geometry(np.asarray(ctx['wall']), moved.matrix != 0)
                pixels.append(None if moved is None else encode_pixels(np.asarray(moved.matrix) != 0))
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
        if self.version == 2:
            obs = self.version_2(obs, ctx, classes, sampled, remaining, feasible, raw)
        obs = plain(obs)  # also detaches the memory entries updated below
        if self.geometry is not None:
            self.geometry.fill(obs, classes)
        for s in obs['sampled']:  # memory is updated after the observation, so it only holds the past
            m = self.stones.setdefault(s['stone'], {'times_sampled': 0, 'min_feasible_rotations': 4})
            m.update(times_sampled=m['times_sampled'] + 1, last_step=obs['step'],
                     last_feasible_rotations=s['feasible_rotations'], last_best_llc=s['best_llc'],
                     min_feasible_rotations=min(m['min_feasible_rotations'], s['feasible_rotations']))
        self.recent = (self.recent + [{'step': obs['step'], **obs['now']}])[-3:]
        if self.version == 2:
            self.history = (self.history + [{'step': obs['step'], 'side': obs['side'], **obs['now'],
                                             'n_outcomes': len(classes)}])[-10:]
        info = {'classes': classes, 'by_index': by_index, 'paper': paper, 'base': base, 'stone_ids': stone, 'tol': tol,
                'selections': selections, 'n': n, 'pixels': pixels}
        return obs, info

    def version_2(self, obs, ctx, classes, sampled, remaining, feasible, raw):
        """obs with the version 2 fields; everything is from this selection or earlier calls."""
        sizes, n = candidate_sizes(ctx), len(sampled)
        live = ctx['all_bricks'] is not None or self.geometry is not None  # self.geometry: a live wall re-executed
        mode = 'pixels' if live else 'summary' if sizes else 'none'
        if sizes:
            for j, s in enumerate(sampled):  # rotation 0 rows: a stone's own size as it was sampled
                self.sizes[self.label(s)] = {'width': sizes[0][j], 'height': sizes[1][j]}
        for o, i in zip(obs['outcomes'], classes):
            o['shape'] = {'width': sizes[0][i], 'height': sizes[1][i]} if sizes else None
        now = {**obs['now'], 'n_candidates': len(raw), 'n_feasible': int(feasible.sum()),
               'max_feasible_llc': round(float(raw[feasible, 3].max()), 6) if feasible.any() else None}
        last = self.history[-1] if self.history else None
        delta = lambda k: None if last is None or now[k] is None or last[k] is None else round(now[k] - last[k], 6)
        seen = sorted(({'stone': self.label(s), **self.sizes[self.label(s)]} for s in remaining
                       if s in self.labels and self.label(s) in self.sizes), key=lambda e: int(e['stone'][1:]))  # label order, not stone ids
        wall = np.asarray(ctx['wall'])
        return {**obs, 'observation_version': 2, 'now': now,
                'filter_weights': {FILTERS[c]: float(ctx['weights'][c]) for c in range(5)},
                'wall_size': {'width': int(wall.shape[1]), 'height': int(wall.shape[0])},
                'geometry_available': mode, 'geometry_note': GEOMETRY_NOTES[mode],
                'remaining': {'n': len(remaining), 'n_sampled_before_or_now': len(seen),
                              'n_never_sampled': len(remaining) - len(seen), 'sampled_sizes': seen},
                'memory': {'stones': dict(sorted(obs['memory']['stones'].items(), key=lambda kv: int(kv[0][1:]))),
                           'recent': list(self.history)},  # label order (version 1 keeps its stone-id order)
                'trend': {'since_step': None if last is None else last['step'],
                          'feasible_share': delta('feasible_share'), 'mean_feasible_llc': delta('mean_feasible_llc')}}


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
                           'llm_timeout': 60.0, 'max_llm_failures': 3},
                 'pick': {'llm': None, 'max_tool_calls': 3, 'max_decisions': 10, 'llm_timeout': 60.0, 'max_llm_failures': 3}}
# max_total_tokens, max_cost_usd, observation_version, first_step (react, optional): only in specs that
# set them, so earlier walls keep their spec; observation_version 1 (the default) or 2 (Observer,
# SYSTEM_V2, TOOLS_V2); first_step: decision points of earlier steps are skip:before_first_step
SPEC_KEYS = {'null': set(), 'random': {'seed', 'steps'}, 'rule': set(SPEC_DEFAULTS['rule']),
             'react': set(SPEC_DEFAULTS['react']) | {'max_total_tokens', 'max_cost_usd', 'observation_version', 'first_step'},
             'pick': set(SPEC_DEFAULTS['pick']) | {'max_total_tokens', 'max_cost_usd'}}
# the rest of an LLM spec is the client's
WORKER_KEYS = {'react': {'tool_feedback', 'max_tool_calls', 'llm_timeout'}, 'pick': {'max_tool_calls', 'llm_timeout'}}
LLM_KINDS = ('react', 'pick')  # pick: research/controller/pick.py, a direct choice of an endgame candidate
SOURCES = {'null': {'null'}, 'random': {'random', 'matched_random', 'matched_random:not_matched'},
           'rule': {'rule:scarcity', 'rule:llc_drop', 'rule:none'},
           'react': {'llm', 'fallback:llm_timeout', 'fallback:llm_error', 'fallback:no_valid_commit'}}
SOURCES['pick'] = SOURCES['react']
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


def llm_identity(llm):
    """(model, params) an LLM spec puts into the requests of its transcripts; bad specs raise.
    A mock is 'mock:<mode>' without parameters; an API LLM is llm_api's."""
    if isinstance(llm, dict) and llm.get('kind') == 'mock' and set(llm) == {'kind', 'mode'} and isinstance(llm['mode'], str):
        return f"mock:{llm['mode']}", {}
    if isinstance(llm, dict) and llm.get('kind') == 'api':
        from research.controller import llm_api
        return llm_api.identity(llm)
    raise NotImplementedError(f'an LLM spec is {{"kind": "mock", "mode": ...}} or {{"kind": "api", ...}}, got {llm!r}')


def validate_spec(spec):
    """The full spec if a live wall may run it; raises otherwise. A Null, Random or Rule policy is
    built to validate its parameters (it has no side effects); a ReAct policy and its LLM client
    are not: they exist only in the worker (make_policy runs there)."""
    spec = full_spec(spec)
    if spec['kind'] not in LLM_KINDS:
        make_policy(spec)
    else:
        llm_identity(spec['llm'])
        if spec.get('observation_version', 1) not in (1, 2) or type(spec.get('observation_version', 1)) is not int:
            raise ValueError(f"observation_version is 1 or 2, got {spec['observation_version']!r}")
        if 'first_step' in spec and not (type(spec['first_step']) is int and spec['first_step'] >= 1):
            raise ValueError(f"first_step is a step number >= 1, got {spec['first_step']!r}")
        budget, usd = spec.get('max_total_tokens'), spec.get('max_cost_usd')
        if not (budget is None or (type(budget) is int and budget > 0)):
            raise ValueError(f'max_total_tokens is None or a positive int, got {budget!r}')
        if not (usd is None or (type(usd) in (int, float) and 0 < usd < float('inf'))):
            raise ValueError(f'max_cost_usd is None or a positive number, got {usd!r}')
        if spec['llm']['kind'] == 'api':
            from research.controller import llm_api
            llm_api.enabled(spec['llm'])
            if llm_api.paid(spec['llm']):
                if budget is None or usd is None:
                    raise ValueError('a paid provider needs max_total_tokens and max_cost_usd')
                if llm_api.call_bound(spec['llm']) > spec['llm_timeout']:
                    raise ValueError(f"llm_timeout {spec['llm_timeout']} s is shorter than the provider's retries "
                                     f"({llm_api.call_bound(spec['llm'])} s): an abandoned call leaves its usage unknown")
            elif usd is not None and llm_api.price(spec['llm']) is None:
                raise ValueError('max_cost_usd needs a price for the model')
    return spec


def observation_version(spec):
    """The observation and prompt version of a full spec: 1 unless a react spec sets 2."""
    return spec.get('observation_version', 1) if spec['kind'] == 'react' else 1


def tokens_of(reply):
    """Tokens the LLM calls of a reply report (usage input_tokens + output_tokens; 0 without usage)."""
    total = 0
    for t in reply.get('transcripts') if isinstance(reply.get('transcripts'), list) else []:
        usage = t.get('response', {}).get('usage') if isinstance(t, dict) and isinstance(t.get('response'), dict) else None
        if isinstance(usage, dict):
            total += sum(v for k, v in usage.items() if k in ('input_tokens', 'output_tokens') and type(v) is int and v >= 0)
    return total


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
    that is not one of the policy's answers (worker_protocol_error) keeps the base order; a
    rejected reply also ends the worker, so no policy state that produced it survives. For react
    the client holds the wall's counters: request failures (LLM and worker), LLM decisions and the
    tokens the replies report; after max_llm_failures failures every later decision point is
    skip:llm_unavailable, after max_decisions skip:wall_budget, after max_total_tokens
    skip:token_budget, after max_cost_usd skip:cost_budget, before a spec's first_step
    skip:before_first_step (no decision of the budget is used), and none of them sends a request.
    For an API LLM the spend is llm_api.spend's: reported and estimated tokens (both count)
    and their cost; an attempt whose usage is unknown (a call abandoned or failed after it may have
    been sent, a worker failure of the decision, a reply without usage) is never counted as free:
    every later decision point is skip:usage_unknown. With a budget in the spec each request
    carries what is left of it, which the API LLM reserves before every attempt. Each request
    leaves a ledger line (ledger: trace/transport.jsonl by default) with what this process sent
    and received, written before the transport cleans up after a failure; the transport's worker
    lifecycle goes to the same ledger.
    """

    def __init__(self, spec, transport, sink=None, ledger=None):
        self.spec_, self.transport = full_spec(spec), transport
        self.observer = Observer(version=observation_version(self.spec_))
        self.sink, self.ledger, self.calls = sink or write_record, ledger or write_ledger, 0
        self.request_failures = self.llm_decisions = self.tokens = self.unknown_usage = 0
        self.cost_usd, self.api = 0.0, self.spec_['kind'] in LLM_KINDS and self.spec_['llm'].get('kind') == 'api'
        if hasattr(transport, 'log'):
            transport.log = self.ledger

    def spec(self):
        return self.spec_

    def request_of(self, obs, info):
        """(request, record fields) of a decision point: the observation, the base order and the tool table."""
        tools = tool_table(info, self.observer.label)
        digest = {'tool_table_sha256': hashlib.sha256(json.dumps(tools, sort_keys=True).encode()).hexdigest()}
        return {'observation': obs, 'base_order': key(info['base']), 'orders': tools}, digest

    def keep(self, info):
        """The decision of a skip or a fallback: the base order."""
        return info['base']

    def reply_problem(self, reply, base):
        return reply_problem_of(reply, self.spec_['kind'], key(base))

    def decision(self, reply):
        return parse_order(reply['order']), {k: v for k, v in reply.items() if k not in ('order', 'restart')}

    def ask(self, obs, info):
        spec, base, keep = self.spec_, info['base'], self.keep(info)
        if spec['kind'] in LLM_KINDS:
            if spec.get('first_step') is not None and obs['step'] < spec['first_step']:
                return keep, {'source': 'skip:before_first_step'}
            if self.request_failures >= spec['max_llm_failures']:
                return keep, {'source': 'skip:llm_unavailable'}
            if self.llm_decisions >= spec['max_decisions']:
                return keep, {'source': 'skip:wall_budget'}
            if self.api and self.unknown_usage:
                return keep, {'source': 'skip:usage_unknown'}
            if spec.get('max_total_tokens') is not None and self.tokens >= spec['max_total_tokens']:
                return keep, {'source': 'skip:token_budget'}
            if spec.get('max_cost_usd') is not None and self.cost_usd >= spec['max_cost_usd']:
                return keep, {'source': 'skip:cost_budget'}
            self.llm_decisions += 1
        request, digest = self.request_of(obs, info)
        if spec['kind'] in LLM_KINDS and (spec.get('max_total_tokens') is not None or spec.get('max_cost_usd') is not None):
            request['budget'] = {'tokens_left': None if spec.get('max_total_tokens') is None else spec['max_total_tokens'] - self.tokens,
                                 'usd_left': None if spec.get('max_cost_usd') is None else round(spec['max_cost_usd'] - self.cost_usd, 6)}

        def settle(status, reply):  # before the transport cleans up: the ledger line comes first
            problem = self.reply_problem(reply, base) if status == 'ok' else None
            if problem:  # the transport then ends the worker that produced it, its policy state with it
                status, reply = 'worker_protocol_error', problem
            self.ledger({'type': 'request', 'event': self.calls, **{k: obs[k] for k in ('step', 'phase', 'sample_trial', 'side')},
                         'request_sha256': request_sha256(request), 'status': status,
                         'reply_sha256': request_sha256(comparable(reply)) if status == 'ok' else None,
                         'restart': status == 'ok' and reply.get('restart') is True, 'error': None if status == 'ok' else reply,
                         'worker_pid': getattr(self.transport, 'pid', None)})
            return status, reply
        status, reply = self.transport.decide(request, settle)
        if status != 'ok':
            self.request_failures += 1
            self.unknown_usage += self.api  # an API call of the decision may have been sent
            return keep, {'source': f'fallback:{status}', 'error': reply, **digest}
        self.request_failures += reply['source'] in LLM_FAILURES
        if self.api:
            from research.controller import llm_api
            spent = llm_api.spend(spec['llm'], reply.get('transcripts'))
            self.tokens += spent['tokens'] + spent['estimated_tokens']
            self.cost_usd += spent['cost_usd']
            self.unknown_usage += spent['unknown']
        else:
            self.tokens += tokens_of(reply)
        choice, fields = self.decision(reply)
        return choice, {**fields, **digest}

    def warm_worker(self):
        """Start a policy worker without an LLM before the first stone; react keeps starting at its first request."""
        if self.spec_["kind"] not in LLM_KINDS and hasattr(self.transport, "warm"):
            self.transport.warm()

    def __call__(self, policy, ctx):
        if self.calls == 0:
            self.warm_worker()
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
                             'stone_id': None if index is None else int(info['stone_ids'][index])},
                   **({} if info['pixels'] is None else {'outcome_pixels': info['pixels']})})
        self.calls += 1
        return policy if order == info['base'] else {**policy, 'order': order}


class InlineTransport:
    """The policy in this process, on JSON copies of request and reply: offline evaluation, check()
    and tests. Failures and restarts mirror ProcessTransport (any failure or a reply asking for a
    restart rebuilds the policy), but nothing is isolated: an abandoned LLM call keeps running."""

    def __init__(self, make):
        self.make, self.policy = make, None

    def decide(self, request, settle=None):
        self.policy = self.policy or self.make()
        try:
            status, reply = 'ok', json.loads(json.dumps(self.policy.decide(json.loads(json.dumps(request))), allow_nan=False))
        except Exception as error:  # the worker would report it the same way
            status, reply = 'worker_error', error_text(error)
        if settle is not None:
            status, reply = settle(status, reply)
        if status != 'ok' or not isinstance(reply, dict) or reply.get('restart'):
            self.policy = None
        return status, reply

    def close(self, reason='close'):
        self.policy = None


SECRETS = set()  # the API keys this process read (llm_api.api_key registers them): never written anywhere
KEY_PATTERN = re.compile(r'(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_*-]{8,}')  # a key, or a provider's partly starred echo of one


def masked(text):
    """text without any registered key or key-shaped string; masked before anything shortens it."""
    for secret in SECRETS:
        text = text.replace(secret, '***') if secret else text
    return KEY_PATTERN.sub('sk-***', text)


def error_text(error):
    """'Type: message' of an exception, masked, then at most 300 characters, without trusting its __str__."""
    try:
        text = str(error)
    except Exception:
        text = '(unprintable)'
    return masked(f'{type(error).__name__}: {text}')[:300]


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

# Version 2: the filters as Stonepacker2D/stack/place_stone.py computes them (place_one_stone: LFR from
# local_void.get_void_to_left before future_base.add_stone; NHR, GDS, IWR, LLC after it, on the
# stabilized rock's pixels and the attributes rotate_brick leaves unturned), the cascade of
# selection._paper and the observation fields of Observer.version_2.
SYSTEM_V2 = (
    "You choose the filter order of Wang's hierarchical filtering for one stone placement. The candidates "
    "(one per sampled stone and rotation, each at the position Wang's image convolution found), their scores and "
    "the LLC > 0 gate are fixed; you only choose the order of the five filters (a permutation 'a-b-c-d-e' of "
    "0 NHR, 1 GDS, 2 IWR, 3 LLC, 4 LFR) or keep the base order (keep_paper). You cannot pick a stone or a position.\n"
    "Filters (raw scores; for every filter a higher score is better). Rows count upward from the ground; "
    "coordinates are in the evaluation frame: mirrored when building from the right, so 'left' is the side the "
    "course is built from. Wall pixels are the stones and the ground; the frame around the wall (pixel value 255) is "
    "never a wall pixel for LFR, GDS or LLC, but its sides are edges for IWR's touch test. Wang evaluates each "
    "candidate in two states. LFR: the candidate at the pose convolution found, on the current wall. NHR, GDS, IWR and "
    "LLC: a copy of the current wall to which add_stone added the candidate and stabilized it (the 'rotation' method "
    "turns the stone by a small angle, or, if no angle works, adds wedges beside it as separate pieces). Turning "
    "replaces only the stone's pixels: its stored center, width and height stay those of the unturned candidate, and "
    "so do the corner points and centers the wall stored for each earlier stone when it was added, before its own "
    "stabilization. Below, 'pixels' are the stabilized stone's pixels and 'stored' marks those unturned values.\n"
    "- 0 NHR course_height: -|top row of the pixels - top row of the neighbour| / stored height. The neighbour: among "
    "the earlier stones' stored top-right corners whose column is left of the stone's stored center, the nearest "
    "(Euclidean) to the top-left corner of the pixels (leftmost column, top row). With no such corner: -|median "
    "stored height of the sampled stones - stored height| / stored height. 0 = level with its left neighbour.\n"
    "- 1 GDS global_density: B = the box from the lowest to the highest row of the current wall's wall pixels (ground "
    "included, frame excluded) and from their leftmost column to the right edge of the matrix (rows 0-2 across the "
    "matrix on a wall without wall pixels); P = the pixels; score = (|P & B| / |P | B|) / (|P| / |B|), set to 0 when "
    "it is 0.7 or less. Higher = the stone lies inside the wall's current band.\n"
    "- 2 IWR interlocking: 'touching' a side = at the candidate pose, Wang's convolution finds a vertical edge of the "
    "current wall matrix (pixel values differing between neighbouring columns over at least 3 rows: a stone side, a "
    "joint between stones or a side of the frame) within 1 px of that side of the candidate's bounding box, along the "
    "upper half of the side (from its middle row to its top row; one row lower for an even height). Stones below = "
    "earlier stones whose stored center row is lower than the bottom row of the pixels. Touching both sides: 0.5. "
    "Touching only the left: d1 = the distance from the bottom-right corner of the pixels (rightmost column, bottom "
    "row) to the nearest stored top-left corner of a stone below, divided by that stone's stored span (its stored "
    "top-right column minus its stored top-left column). Touching only the right: d2 = the same from the bottom-left "
    "corner of the pixels to the nearest stored top-right corner. Touching neither: (d1 + d2) / 2. d1 and d2 are 0.5 "
    "when no stone lies below. Higher = its vertical joint is farther from the joints below.\n"
    "- 3 LLC kinematic: the kinematic limit load (tilting table) of the copy of the wall after add_stone (its wall "
    "pixels and its stones: the stabilized stone and any wedges included), divided by the friction coefficient. Only "
    "candidates with LLC > 0 and five finite scores are feasible.\n"
    "- 4 LFR local_void: -(L + D) / (matrix width x stone height + matrix height x stone width), from the candidate's "
    "own pixels at its pose, sizes in pixels (the matrix includes the frame). L = for each row of the stone, the empty "
    "pixels between its leftmost pixel and the rightmost wall pixel in the columns left of its center column (or the "
    "matrix's left edge), summed; D = for each column of the stone, the empty pixels between its lowest pixel and the "
    "highest wall pixel in the rows below its center row (or the matrix's bottom edge), summed; frame pixels count as "
    "empty. 0 = flush on the left and on the bottom.\n"
    "Cascade: start from the feasible candidates; for each filter in the chosen order take the weighted scores w (raw "
    "x filter_weights) of the candidates still kept and their mean m, and keep those with w >= m - tol x max(1, |m|), "
    "tol being the base policy's tolerance (1e-9: float rounding only). With finite w and m the best candidate always "
    "passes. If a stage keeps none (only when the arithmetic is not finite, e.g. the mean overflows), the pick is the "
    "candidate of that stage's input with the largest sum of the weighted scores of the filters applied so far, this "
    "one included, and the later filters are not used. Otherwise the final pick is the largest sum of all five "
    "weighted scores among the survivors. The mean is taken over the current survivors, so the order decides which "
    "candidates are compared with which; the base order is 3-0-2-1-4.\n"
    "Data: geometry_available says what geometry exists. pixels: outcome.geometry is the change of the wall after "
    "pasting the candidate's own unstabilized pixels at its pose, without physics: a proxy computed for you, not a "
    "state any filter uses. summary: only bounding-box sizes, never the wall after placement. none: neither. Never "
    "cite a geometry value that is null. memory, history and trend are your own earlier calls; nothing about future "
    "samples or the final wall is known.\n"
    "Before committing, compare the main candidates on all five filters: in the rationale state each one's main "
    "advantages and costs, not one score alone. Finish with commit_order.")
COMMIT_V2 = {'name': 'commit_order', 'description': "Final action: an order 'a-b-c-d-e' or 'keep_paper'.",
             'parameters': {'type': 'object', 'additionalProperties': False, 'required': ['order', 'rationale'],
                            'properties': {'order': {'type': 'string'},
                                           'rationale': {'type': 'string', 'description':
                                                         'The main advantages and costs of the compared candidates '
                                                         'on the five filters, and why this order.'}}}}
TOOLS_V2 = [{'name': 'list_order_outcomes',
             'description': 'Distinct candidates the 120 orders select: pose, the five raw scores, how many orders select '
                            'each (n_orders, example_order), its bounding box (shape) and geometry as geometry_available says.',
             'parameters': {'type': 'object', 'additionalProperties': False, 'required': [], 'properties': {}}},
            {'name': 'explain_order', 'description': 'Survivors of each cascade stage for one order.',
             'parameters': {'type': 'object', 'additionalProperties': False, 'required': ['order'],
                            'properties': {'order': {'type': 'string'}}}},
            COMMIT_V2]
PROMPTS = {1: (SYSTEM, TOOLS, COMMIT), 2: (SYSTEM_V2, TOOLS_V2, COMMIT_V2)}


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
    request_sha256, response {raw_text, parsed, usage, error, and attempts for an LLM that retries},
    status and latency_s. usage and attempts are what the LLM object holds after the call.

    Fallback rules (the decision then keeps the base order and its record says why):
    - a reply that breaks the tool schema is rejected with the reason as feedback and counts as
      one of the max_tool_calls actions; no valid commit within them: fallback:no_valid_commit;
    - an LLM call that raises or gives no reply within llm_timeout seconds: fallback:llm_error /
      fallback:llm_timeout at once, no retry. A timed-out call cannot be stopped from its thread,
      so the reply asks for a restart: ProcessTransport kills the worker with the call in it.
    The wall budget and the failure limit are the client's (ControllerClient).
    """

    def __init__(self, llm, tool_feedback=True, max_tool_calls=4, llm_timeout=60.0, observation_version=1):
        self.llm, self.max_tool_calls, self.llm_timeout = llm, max_tool_calls, llm_timeout
        self.system, tools, commit = PROMPTS[observation_version]
        self.tools = tools if tool_feedback else [commit]
        self.prompt_sha256 = hashlib.sha256(json.dumps([self.system, self.tools]).encode()).hexdigest()

    def call_llm(self, messages, transcripts):
        """('ok', parsed reply) or ('timeout' | 'error', reason); appends the call's transcript."""
        request = {'messages': copy.deepcopy(messages), 'tools': copy.deepcopy(self.tools),
                   'model': safe_json(getattr(self.llm, 'model', None)), 'params': safe_json(getattr(self.llm, 'params', {}))}
        box, args = {}, (copy.deepcopy(messages), copy.deepcopy(self.tools))

        def call():
            try:
                box['reply'] = self.llm(*args)
            except LLMFailure as failure:
                box['failure'] = (failure.status, masked(str(failure.text))[:300])
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
        response = {'raw_text': out[:MAX_REPLY_CHARS] if ok and isinstance(out, str) else None,
                    'parsed': parsed, 'usage': safe_json(getattr(self.llm, 'usage', None)), 'error': None if ok else out}
        if getattr(self.llm, 'attempts', None) is not None:
            response['attempts'] = safe_json(self.llm.attempts)
        if getattr(self.llm, 'exchange', None) is not None:  # the provider's own request and response
            response['exchange'] = safe_json(self.llm.exchange)
        transcripts.append({'call': len(transcripts), 'request': request, 'request_sha256': request_sha256(request),
                            'response': response, 'status': status, 'latency_s': round(time.monotonic() - start, 3)})
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

    def view(self, obs):
        """The observation of the first message: outcomes only through list_order_outcomes."""
        return {k: v for k, v in obs.items() if k != 'outcomes'}

    def answer(self, request, choice):
        """The reply's decision: the committed order, the base order for a fallback (choice None)."""
        return {'order': request['base_order'] if choice is None else choice}

    def problem(self, reply, request):
        return reply_problem(reply, self.tools)

    def decide(self, request):
        view = self.view(request['observation'])
        messages = [{'role': 'system', 'content': self.system}, {'role': 'user', 'content': json.dumps(view)}]
        steps, transcripts = [], []
        meta = {'prompt_sha256': self.prompt_sha256, 'transcripts': transcripts}
        if callable(getattr(self.llm, 'begin', None)):  # an API LLM reserves this decision's budget per attempt
            self.llm.begin(request.get('budget'))
        for _ in range(self.max_tool_calls):
            status, reply = self.call_llm(messages, transcripts)
            if status != 'ok':
                steps.append({'status': status, 'error': reply})
                return {**self.answer(request, None), 'source': f'fallback:llm_{status}', **meta, 'react': steps,
                        **({'restart': True} if status == 'timeout' else {})}
            try:
                problem = self.problem(reply, request)
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
                return {**self.answer(request, order), 'source': 'llm', **meta, 'react': steps}
        return {**self.answer(request, None), 'source': 'fallback:no_valid_commit', **meta, 'react': steps}


class ReplayLLM:
    """The LLM of a check: plays back the transcripts of a wall's records, never an API. A call is
    found by its decision key (from the observation message) and its position in the decision; the
    request the policy sends, with the model and params the receipt's LLM spec gives (never the
    records'), must hash to the recorded request_sha256, and recorded timeouts and errors recur at
    once with their text. usage, attempts and the provider exchange are restored per call, as the
    live LLM held them.
    Anything else is a divergence: listed in `diverged` and the call fails."""

    def __init__(self, records, model, params):
        records = [r for r in records if isinstance(r, dict) and isinstance(r.get('transcripts'), list)]
        self.recorded = {decision_key(r): r['transcripts'] for r in records}
        self.model, self.params, self.diverged = model, params, []
        self.usage = self.attempts = self.exchange = None

    def __call__(self, messages, tools):
        key_ = decision_key(json.loads(messages[1]['content']))
        call, done = (len(messages) - 2) // 2, self.recorded.get(key_, [])
        t = done[call] if call < len(done) else None
        request = {'messages': messages, 'tools': tools, 'model': self.model, 'params': self.params}
        self.usage = self.attempts = self.exchange = None
        if not isinstance(t, dict) or t.get('request_sha256') != request_sha256(request) or \
                request_sha256(t.get('request')) != t['request_sha256'] or not isinstance(t.get('response'), dict):
            self.diverged.append(f"{key_} call {call}: {'not recorded' if t is None else 'request differs from the recording'}")
            raise LLMFailure('error', 'replay diverged')
        self.usage, self.attempts, self.exchange = (t['response'].get(k) for k in ('usage', 'attempts', 'exchange'))
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

    def decide(self, request, settle=None):
        failed = self.failed.get(decision_key(request['observation']))
        if failed:
            self.policy = None
            return failed if settle is None else settle(*failed)
        return super().decide(request, settle)


class MockLLM:
    """Scripted stand-in for the tool flow (no API): 'max_llc' lists outcomes, explains the
    highest-LLC one and commits it; 'keep' commits keep_paper; 'blind' commits 0-1-2-3-4 without
    looking; 'invalid' commits a non-permutation; 'stone' tries to name a stone. Failure modes,
    max_llc elsewhere: 'hang@N' never answers and 'crash@N' ends its process (exit code 3) at the
    first call of the decisions of step N (read from the observation, so a restart does not move it)."""

    def __init__(self, mode):
        self.mode, self.model, self.params = mode, f'mock:{mode}', {}

    def __call__(self, messages, tools):
        if self.mode.startswith('pick_'):  # a direct pick (research/controller/pick.py)
            from research.controller import pick
            return pick.mock_reply(self.mode, messages)
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
    llm_identity(llm)
    if llm['kind'] == 'api':
        from research.controller import llm_api
        llm = llm_api.APILLM(llm)  # its client is built here, in the process that calls it
    else:
        llm = MockLLM(llm['mode'])
    if spec['kind'] == 'pick':
        from research.controller import pick
        return pick.PickPolicy(llm, **{k: params[k] for k in WORKER_KEYS['pick']})
    return ReActPolicy(llm, **{k: params[k] for k in WORKER_KEYS['react']}, observation_version=observation_version(spec))


def make_controller(spec, sink=None, transport='process', ledger=None, worker_venv=False):
    """Controller from a JSON-able spec, e.g. {'kind': 'random', 'seed': 7}; bad specs raise here,
    before any wall is built (validate_spec: no policy or LLM client is built in this process).
    transport 'process' (live walls) runs the policy in a worker process
    (controller_worker.ProcessTransport); 'inline' runs it in this process. worker_venv: the worker
    runs in the registered venv (controller_worker.worker_python, checked here), with only the
    environment worker_env gives it; the API key variables then leave this process's environment,
    so nothing this process starts later (Wang's Pool) inherits them."""
    if isinstance(spec, dict) and spec.get('kind') == 'endgame':  # a direct pick in the endgame, no transport
        from research.controller import endgame
        return endgame.EndgamePicker(spec, sink)
    spec = validate_spec(spec)
    client = ControllerClient
    if spec['kind'] == 'pick':
        from research.controller.pick import PickClient as client
    if transport == 'inline':
        return client(spec, InlineTransport(lambda: make_policy(spec)), sink, ledger)
    from research.controller import worker as controller_worker
    python = controller_worker.worker_python() if worker_venv else None
    client = client(spec, controller_worker.ProcessTransport(spec, python=python), sink, ledger)
    if worker_venv:
        from research.controller import llm_api
        for k in llm_api.KEY_ENV.values():
            os.environ.pop(k, None)
    return client


def append_trace(name, entry):
    """Append a JSON line to trace/<name> of the wall being built; nothing outside a traced wall
    (run_wall always traces controller runs; a lost line shows as a missing record or ledger)."""
    from Stonepacker2D.utils.constant import get_dir, get_trace
    if get_trace() is None:
        return
    path = Path(get_dir()) / 'trace' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(entry, default=lambda v: v.item()) + '\n')


def write_record(record):
    append_trace('controller.jsonl', record)


def write_ledger(entry):
    append_trace('transport.jsonl', entry)


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


def refuse_paid(spec, what):
    """Raises if spec's LLM is a paid API: offline work only replays recorded replies (check())."""
    spec = full_spec(spec)
    if spec['kind'] in LLM_KINDS and spec['llm']['kind'] == 'api':
        from research.controller import llm_api
        if llm_api.paid(spec['llm']):
            raise ValueError(f"{what} never calls a paid API ({spec['llm']['provider']}); check() replays a wall's recorded replies")


def rerun(result, spec, base, transport='inline', geometry=None, ledger=None):
    """The records a controller writes when it sees the traced selections of one result directory;
    transport: 'inline', 'process' or a transport object (check() replays LLM walls with one);
    a paid API spec is refused unless the transport is given (it then decides what is called);
    geometry: a LiveGeometry giving the observations the live outcome geometry; ledger: a list
    that receives the ledger lines (discarded by default)."""
    if isinstance(transport, str):
        refuse_paid(spec, 'rerun')
    records, ledger = [], [] if ledger is None else ledger
    controller = make_controller(spec, records.append, transport, ledger.append) if isinstance(transport, str) else \
        ControllerClient(spec, transport, records.append, ledger.append)
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
    traced orders are applied, not base, and their states carry the earlier controller's actions.
    A paid API spec is refused: an offline evaluation never calls a real API."""
    refuse_paid(spec, 'replay')
    run = run_of(result)
    if json.loads((run / 'receipt.json').read_text(encoding='utf-8')).get('controller') is not None:
        raise ValueError(f'{run} was built with a controller; replay needs a base-policy wall')
    base = base_policy(run)
    if any(meta['policy'] != json.loads(json.dumps(base)) for meta, _ in trace_events(result)):
        raise ValueError(f'{result} has selections with a policy other than the base {base}')
    return rerun(result, spec, base)


def comparable(v):
    """A record without what a re-execution from the trace cannot reproduce: the candidate pixels
    and outcome geometry (live only; LiveGeometry checks them) and LLM call latencies."""
    if isinstance(v, dict):
        return {k: None if k in ('geometry', 'latency_s') else comparable(x) for k, x in v.items() if k != 'outcome_pixels'}
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
    """The live outcome geometry when a wall is re-executed from its trace, which keeps no stone
    pixels (Transcript Replay: the policy must see what it saw live), and its Geometry Validation.
    Outcomes are found by decision key and candidate pose (pose_of), never by position or count.

    An outcome whose record keeps its candidate pixels (outcome_pixels) gets its geometry
    recomputed from them, not copied: the pixels must lie at the candidate's (x, y) and hash to the
    footprint Wang's trace holds for that candidate (they are its pixels), and the recorded
    geometry must equal the recomputation (verified). Without pixels (records written before
    outcome_pixels) the recorded geometry is carried over unverified, for validate_geometry to try
    on base.pkl. `problems` lists a selection without a live record, outcome sets that differ from
    the re-executed candidates, missing or malformed geometry or pixels, and every mismatch."""

    def __init__(self, records, events):
        self.by_key, self.problems, self.carried, self.verified, self.outcomes = {}, [], [], 0, 0
        self.traced = {(m['step'], m['phase'], m['sample_trial'], m['side']): a for m, a in events}
        for r in records:
            if not (isinstance(r, dict) and isinstance(r.get('observation'), dict)):
                continue
            key, outcomes = decision_key(r), r['observation'].get('outcomes')
            if key in self.by_key:
                self.problems.append(f'two live records for {key}')
                continue
            outcomes = outcomes if isinstance(outcomes, list) else []
            poses = [pose_of(o) for o in outcomes]
            if len(set(poses)) != len(poses) or None in poses:
                self.problems.append(f'{key}: live outcomes without a pose or with a pose twice')
            pixels = r.get('outcome_pixels', [None] * len(outcomes))
            if not (isinstance(pixels, list) and len(pixels) == len(outcomes)):
                self.problems.append(f'{key}: outcome_pixels do not list one entry per outcome')
                pixels = [None] * len(outcomes)
            self.by_key[key] = {p: (o.get('geometry'), px) for p, o, px in zip(poses, outcomes, pixels) if p is not None}

    def fill(self, obs, classes):
        key = decision_key(obs)
        recorded = self.by_key.get(key)
        if recorded is None:
            self.problems.append(f'{key}: no live record to take the geometry from')
            return
        if {pose_of(o) for o in obs['outcomes']} != set(recorded):
            self.problems.append(f'{key}: the live outcomes are not the candidates the trace re-executes')
        a = self.traced[key]
        for o, i in zip(obs['outcomes'], classes):
            self.outcomes += 1
            g, pixels = recorded.get(pose_of(o), (None, None))
            o['geometry'] = g
            if not geometry_ok(g):
                self.problems.append(f'{key}: outcome {o["id"]} {pose_of(o)} has no valid live geometry')
            elif pixels is None:
                self.carried.append((key, pose_of(o), g))
            else:
                wall = np.asarray(a['wall'])
                mask = decode_pixels(pixels, wall.shape) if isinstance(pixels, dict) else None
                if mask is None or pixels['origin'] != [o['x'], o['y']] or footprint(mask, obs['side']) != a['footprint'][i]:
                    self.problems.append(f'{key}: outcome {o["id"]} {pose_of(o)}: its pixels are not the traced candidate')
                    continue
                o['geometry'] = placement_geometry(wall, mask)
                if o['geometry'] != g:
                    self.problems.append(f'{key}: outcome {o["id"]} {pose_of(o)}: recorded geometry differs from its recomputation')
                else:
                    self.verified += 1


def validate_geometry(result, carried, events):
    """Geometry Validation of outcomes whose records keep no pixels (carried: (decision key, pose,
    geometry) from LiveGeometry), from the stone pixels base.pkl holds. A placed rock gives its
    stone's world-frame pixel crop; a candidate of that stone whose traced footprint hash equals one
    of the crop's 8 rotations and mirrorings has exactly those pixels, which are pasted at the
    candidate's evaluation-frame (x, y) on the traced wall (placement_geometry). Stones whose rock
    stabilization turned or moved off the pixel grid, stones never placed and a missing base.pkl
    leave outcomes unverifiable; they are counted, never passed."""
    crops, path = {}, Path(result) / 'base.pkl'
    if carried and path.exists():
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
    traced = {(m['step'], m['phase'], m['sample_trial'], m['side']): (m, a) for m, a in events}
    problems, verified = [], 0
    for key_, (stone, rotation, x, y), g in carried:
        meta, a = traced[key_]
        match = np.nonzero((a['stone_id'] == ids.get(stone, -1)) & (a['rotation'] == rotation) & (a['x'] == x) & (a['y'] == y))[0]
        if len(match) != 1:
            problems.append(f'{key_}: outcome {(stone, rotation, x, y)} is not one candidate of the trace')
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
        if placement_geometry(wall, pixels) != g:
            problems.append(f'{key_}: geometry of outcome {(stone, rotation, x, y)} differs from its recomputation from base.pkl')
        else:
            verified += 1
    return verified, problems


def ledger_problems(result, expected, events, stopped):
    """The live ledger (trace/transport.jsonl, written by the main process) against the ledger of
    the re-execution: the same request hashes, transport statuses, reply hashes, restarts and
    errors in the same order (worker pids aside); a stopped run may hold one request more, for the
    selection whose event it never wrote. None if the wall has no ledger (written before it)."""
    path = Path(result) / 'trace' / 'transport.jsonl'
    if not path.exists():
        return None
    try:
        live = [e for e in map(json.loads, path.read_text(encoding='utf-8').splitlines()) if isinstance(e, dict) and e.get('type') == 'request']
    except ValueError:
        return ['the ledger is not JSON lines']
    strip = lambda e: {k: v for k, v in e.items() if k != 'worker_pid'}
    got, want = [strip(e) for e in live], [strip(e) for e in expected]
    extra = got[len(want):]
    if got[:len(want)] != want:
        bad = next(i for i, (g, w) in enumerate(zip(got + [None] * len(want), want)) if g != w)
        return [f'ledger line {bad} differs from the request the re-execution sends ({want[bad]["step"]} {want[bad]["side"]})']
    if extra and not (stopped and len(extra) == 1 and extra[0].get('event') == len(events)):
        return [f'the ledger holds {len(extra)} requests the trace does not re-execute']
    return []


def worker_problems(entries, stopped):
    """The worker lifecycle of a live run (ledger lines in the order written: start, the requests
    it answered, close): every worker closed was confirmed gone after the reap timeout (a close
    with alive true or without a pid is a worker that may still run), every failed request (and
    every reply asking for a restart) is followed by closing the worker it names, no request
    names a worker closed before it, and every worker started was closed (a stopped run's last
    worker excepted: the launcher kills the process tree and verifies it). Returns (problems, counts)."""
    problems, open_, closed, pending = [], [], set(), {}
    starts = [e.get('pid') for e in entries if e.get('type') == 'worker' and e.get('action') == 'start']
    for e in entries:
        if e.get('type') == 'worker' and e.get('action') == 'start':
            open_.append(e.get('pid'))
        elif e.get('type') == 'worker' and e.get('action') == 'close':
            closed.add(e.get('pid'))
            open_ = [p for p in open_ if p != e.get('pid')]
            pending.pop(e.get('pid'), None)
            if e.get('alive') is not False:
                problems.append(f"worker {e.get('pid')} was not confirmed ended when it was closed ({e.get('reason')})")
        elif e.get('type') == 'request':
            pid = e.get('worker_pid')
            if pid is not None and pid in closed:
                problems.append(f"the request at event {e.get('event')} names worker {pid}, closed before it")
            if (e.get('status') != 'ok' or e.get('restart')) and pid is not None:
                pending[pid] = e.get('event')
    last = starts[-1] if stopped and starts else None
    problems += [f'the worker {p} of the failed request at event {ev} was never closed' for p, ev in pending.items() if p != last]
    left = open_[:-1] if stopped else open_
    problems += [f'worker {p} was started and never closed' for p in left]
    return problems, {'workers_started': len(starts), 'workers_closed': len(closed)}


def manifest_problems(result, run, receipt):
    """The run manifest (receipt 'manifest': sha256 of every trace and result file, written by the
    launcher when the case process ended) against the files now: None without a manifest."""
    from research.walls import check_run
    manifest = receipt.get('manifest')
    if manifest is None:
        return None
    files = {p.relative_to(run).as_posix(): p for p in check_run.artifact_files(result)}
    mine = {k for k in manifest if k.startswith(Path(result).relative_to(run).as_posix() + '/')}
    problems = [f'{k}: missing' for k in sorted(mine - set(files))] + [f'{k}: not in the manifest' for k in sorted(set(files) - mine)]
    problems += [f'{k}: changed since the run' for k in sorted(mine & set(files)) if check_run.sha256_file(files[k]) != manifest[k]]
    return problems


def check(result, records=None, spec=None, stopped=None):
    """Controller records of a built wall against its receipt and trace. The controller named in
    the receipt (spec only for tests on walls without one) is re-executed on the traced inputs from
    the receipt's base policy; every record must equal its re-execution, the traced policy must be
    the base with the re-executed order, replaying select with it must give the traced pick,
    decision keys must be unique and the trace itself must be complete and consistent
    (check_run.chain_problems). A stopped run (stopped; by default read from the receipt) may hold
    one record more than events: the controller is called before its event is written. Counts come
    from the re-execution, not from the records.

    The report keeps three questions apart:
    - ok: replay consistency. The records, the LLM transcripts (ReplayLLM, the receipt's model and
      params) and the ledger, if any, are what the controller produces on the trace, and nothing
      recorded contradicts a recomputation (geometry, ledger, manifest);
    - geometry: Geometry Validation of the live outcome geometry (LiveGeometry from the candidate
      pixels, validate_geometry from base.pkl for records without them): verified, unverifiable,
      mismatched; complete only if every outcome is verified;
    - provenance: the main process's ledger and the launcher's run manifest exist and match.
    verdict: PASS if ok with complete geometry and provenance; PARTIAL if ok but something could
    not be verified (listed in `incomplete`); FAIL otherwise."""
    result = Path(result).resolve()  # run_of resolves; the manifest keys are relative to the run
    run = run_of(result)
    receipt = json.loads((run / 'receipt.json').read_text(encoding='utf-8'))
    spec = receipt.get('controller') if spec is None else spec
    stopped = receipt.get('process_status') == 'STOPPED_AFTER_PLACEMENTS' if stopped is None else stopped
    if records is None:
        path = result / 'trace' / 'controller.jsonl'
        records = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    if spec is None:
        return {'ok': False, 'verdict': 'FAIL', 'problems': ['the receipt names no controller'], 'records': len(records),
                'decision_points': None, 'switches': None}
    if isinstance(spec, dict) and spec.get('kind') == 'endgame':
        from research.controller import endgame
        return endgame.check(result, records, spec, stopped)
    if isinstance(spec, dict) and spec.get('kind') == 'pick':
        from research.controller import pick
        return pick.check(result, records, spec, stopped)
    base, full, events = base_policy(run), full_spec(spec), list(trace_events(result))
    llm, transport = None, 'inline'
    if full['kind'] == 'react':  # the LLM answers from the records' transcripts, never an API
        llm, worker = ReplayLLM(records, *llm_identity(full['llm'])), {k: full[k] for k in WORKER_KEYS['react']}
        transport = ReplayedFailures(lambda: ReActPolicy(llm, **worker, observation_version=observation_version(full)), records)
    live = any(isinstance(r, dict) and isinstance(r.get('observation'), dict) and isinstance(r['observation'].get('outcomes'), list)
               and any(isinstance(o, dict) and o.get('geometry') is not None for o in r['observation']['outcomes']) for r in records)
    geometry, ledger = LiveGeometry(records, events) if live else None, []
    expected = rerun(result, spec, base, transport, geometry, ledger)
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
    if failed and full['kind'] != 'react':
        problems.append(f'the worker of this deterministic controller failed at events {failed[:5]}: '
                        'its decisions are not the controller ones, the wall is no valid result')
    incomplete, report_geometry = [], {'source': 'live records' if live else 'none recorded'}
    if geometry is not None:
        problems += [f'live geometry: {p}' for p in geometry.problems[:5]]
        from_base, base_problems = validate_geometry(result, geometry.carried, events)
        problems += base_problems[:5]
        unverifiable = len(geometry.carried) - from_base - len(base_problems)
        report_geometry.update(outcomes=geometry.outcomes, verified_from_pixels=geometry.verified, verified_from_base_pkl=from_base,
                               unverifiable=unverifiable, mismatched=len(geometry.problems) + len(base_problems),
                               complete=unverifiable == 0 and not geometry.problems and not base_problems)
        if unverifiable:
            incomplete.append(f'{unverifiable} of {geometry.outcomes} outcome geometries not verifiable (no candidate pixels recorded)')
    elif records:
        incomplete.append('no live outcome geometry')
    found = ledger_problems(result, ledger, events, stopped)
    problems += [f'ledger: {p}' for p in found or []]
    manifest = manifest_problems(result, run, receipt)
    problems += [f'manifest: {p}' for p in (manifest or [])[:5]]
    provenance = {'ledger': 'missing' if found is None else 'ok' if not found else 'mismatch',
                  'manifest': 'missing' if manifest is None else 'ok' if not manifest else 'mismatch'}
    incomplete += [f'no {name}' for name, v in (('transport ledger', provenance['ledger']), ('run manifest', provenance['manifest']))
                   if v == 'missing']
    if legacy:
        incomplete.append('records without tool_table_sha256 (written before P2)')
    report = {'ok': not problems, 'verdict': 'FAIL' if problems else 'PARTIAL' if incomplete else 'PASS',
              'problems': problems, 'incomplete': incomplete, 'records': len(records),
              'geometry': report_geometry, 'provenance': provenance,
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
    of that spec (with its defaults, as records hold it) whose observation is for its own decision
    key and, if the last traced attempt has its left side only, that attempt's right side."""
    if not isinstance(record, dict) or not RECORD_KEYS <= set(record):
        return ['not a controller record']
    problems, obs, final = [], record['observation'], record['final']
    if record['event'] != len(events) or record['controller'] != full_spec(spec):
        problems.append(f"event {record['event']} of {record['controller']}, expected {len(events)} of {full_spec(spec)}")
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
    from research.storage.case_id import wall_dir
    rows = []
    with (out / 'records.jsonl').open('w', encoding='utf-8') as f:
        for w in manifest.itertuples():
            result = next(wall_dir(w.run).rglob('trace/commits.jsonl')).parent.parent
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
        from research.walls import check_run
        reports = [check(r) for r in check_run.result_dirs(args.run)]
        print(json.dumps(reports, indent=2))
        return 0 if reports and all(r['ok'] for r in reports) else 1
    from research.walls import campaign
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = pd.read_csv(campaign.manifest_path(args.id))
    df = offline(manifest[(manifest.arm == args.arm) & (manifest.status == 'PROCESS_PASS')], args.out)
    pd.set_option('display.width', 250)
    print(df.groupby('controller').agg(calls=('switched', 'size'), decision_points=('decision_point', 'sum'),
                                       switches=('switched', 'sum'), tool_calls=('tool_calls', 'sum')).to_string())
    print(df.groupby(['controller', 'source']).size().to_string())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
