"""Counterfactual branches of a live GPT wall, rebuilt without any API; research/runner/gpt_counterfactual.py builds them.

A branch is run_wall --policy paper --order 3,0,2,1,4 with GPT's recorded orders as --switch
STEP=order (the order traced at each step of the GPT wall). gpt-001 made one selection per step,
so a step switch replays it. Walls (C_j: GPT's orders at steps <= j, Paper after):
- V: every GPT order. It must rebuild the GPT wall exactly (check_run.compare_runs: decisions,
  filter, transformation, commits, base.pkl matrices, every trace event's meta and arrays), and
  every branch waits for it;
- for k in STEPS: G_k = GPT's orders at steps <= k; P_k = at steps < k, so the Paper order at k.
  Both continue with the Paper order and start from GPT's state before k. P6 has no switch: it is
  the dev Paper wall of the seed, reused only if it passes the P check.
check of a branch against the GPT wall: PROCESS_PASS, check_run.check_run ok, every attempt
before k (G: through k) byte-identical (commit line, trace meta but run_id, trace arrays); for P at
k, the same candidates (all arrays but 'selected'), the selection is Wang's select with the Paper
order on them, and it is not GPT's; the traced order of every step is the planned one.
Controller-only files of the GPT wall (controller.jsonl, transport.jsonl, its receipt) are format,
not construction, and are not compared. No controller and no API key reach a branch.

Each new wall is a case (research/cases/<case id>/) that the study lists as {case_id, branch}, with the GPT wall as
its parent. A plan already in the research/runs archive (branches/gpt-001-cf, where its walls were first built) is
only read: check and summarize work on it, run refuses it.
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from research.paths import LEGACY_RUNS
from research.storage import case_id as cid
from research.storage import studies
from research.walls import campaign, campaign_quality, case_definition, check_run, wall_metrics
from Stonepacker2D.stack.selection import PAPER_ORDER, load_trace, select

PLAN_ID = 'gpt-001-cf'
SOURCE = 'gpt-walls/gpt-001'
PAPER_WALL = 'campaigns/dev-2seeds/thangavelu-noise80/paper/seed66794912'
STEPS = (6, 8, 17)
TIMEOUT = 2400
LEGACY_PLAN = LEGACY_RUNS / 'branches' / PLAN_ID / 'plan.json'
KEY_VARS = ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'STONEPACKING_ALLOW_PAID_API')


def result_of(wall):
    return next(cid.wall_dir(wall).glob('result_*'))


def commits(result):
    return [json.loads(l) for l in (Path(result) / 'trace' / 'commits.jsonl').read_text(encoding='utf-8').splitlines()]


def event(result, c):
    return load_trace(Path(result) / 'trace' / f"e{c['sides'][c['chosen_side']]['event']:05d}.npz")


def recorded_orders(result):
    """{step: order} of the GPT wall's trace where it is not the Paper order; refuses a step with
    more than one selection (a step switch could not replay it)."""
    seen, orders = set(), {}
    for c in commits(result):
        if c['step'] in seen or len(c['sides']) != 1:
            raise ValueError(f"step {c['step']}: more than one selection, a step switch cannot replay it")
        seen.add(c['step'])
        order = tuple(event(result, c)[0]['policy']['order'])
        if order != tuple(PAPER_ORDER):
            orders[c['step']] = order
    return orders


def branches(orders):
    out = {'V': {'switch': orders, 'arm': 'V', 'k': None}}
    for k in STEPS:
        if k not in orders:
            raise ValueError(f'step {k}: GPT kept the Paper order there')
        out[f'G{k}'] = {'switch': {s: o for s, o in orders.items() if s <= k}, 'arm': 'G', 'k': k}
        out[f'P{k}'] = {'switch': {s: o for s, o in orders.items() if s < k}, 'arm': 'P', 'k': k}
    return out


def make_plan():
    orders = recorded_orders(result_of(SOURCE))
    walls = branches(orders)
    seed = json.loads((cid.wall_dir(SOURCE) / 'receipt.json').read_text(encoding='utf-8'))['seed']
    plan = {'id': PLAN_ID, 'created': datetime.now().isoformat(timespec='seconds'), 'source': SOURCE, 'seed': seed,
            'case': 'thangavelu-noise80', 'timeout_seconds': TIMEOUT, 'steps': list(STEPS), 'code_sha256': case_definition.code_fingerprint(),
            'recorded_orders': {str(s): list(o) for s, o in orders.items()},
            'walls': {name: {'arm': w['arm'], 'k': w['k'], 'switch': {str(s): list(o) for s, o in w['switch'].items()},
                             'run': PAPER_WALL if name == 'P6' and not w['switch'] else None}
                      for name, w in walls.items()},
            'build_order': ['V'] + [n for k in STEPS for n in (f'G{k}', f'P{k}') if not (n == 'P6' and not walls[n]['switch'])],
            'continuation': 'paper order after k for both arms', 'api': 'none'}
    return plan


def load_plan():
    """The plan: the design of its study, or plan.json of the plan in the research/runs archive."""
    if campaign.is_study(PLAN_ID):
        plan = studies.load(PLAN_ID)['design']
    else:
        plan = json.loads(LEGACY_PLAN.read_text(encoding='utf-8'))
    for w in plan['walls'].values():
        w['switch'] = {int(s): tuple(o) for s, o in w['switch'].items()}
    return plan


def wall_ref(plan, name):
    """A wall of the plan, for case_id.wall_dir: its 'run' (the reused Paper wall, or a wall of a plan in the
    research/runs archive), else the case its study lists (None while there is none)."""
    wall = plan['walls'][name]
    if wall['run'] is not None:
        return wall['run']
    found = studies.find(plan['id'], branch=name) if campaign.is_study(plan['id']) else []
    return found[-1]['case_id'] if found else None


def built(plan, name):
    ref = wall_ref(plan, name)
    return ref is not None and (cid.wall_dir(ref) / 'receipt.json').exists()


def same_attempt(a, b, ra, rb, select_too=True):
    """Problems between two attempts: commit line, trace meta but run_id, trace arrays."""
    problems = []
    (ma, xa), (mb, xb) = event(ra, a), event(rb, b)
    if select_too and a != b:
        problems.append('commit differs')
    skip = {'run_id'} | (set() if select_too else {'policy', 'stages', 'branch'})
    if {k: v for k, v in ma.items() if k not in skip} != {k: v for k, v in mb.items() if k not in skip}:
        problems.append('trace meta differs')
    keys = [k for k in xa if select_too or k != 'selected']
    if sorted(xa) != sorted(xb) or any(not np.array_equal(xa[k], xb[k], equal_nan=xa[k].dtype.kind == 'f') for k in keys):
        problems.append('trace arrays differ')
    return problems


def check_branch(source, branch, wall):
    """Construction of a branch (result dir) against the GPT wall (result dir); see the module doc."""
    arm, k, switch = wall['arm'], wall['k'], wall['switch']
    cs, cb = commits(source), commits(branch)
    problems = []
    if arm == 'V':
        problems += check_run.compare_runs(Path(source).parent, Path(branch).parent)
    else:
        last = k if arm == 'G' else k - 1
        prefix = [(a, b) for a, b in zip(cs, cb) if a['step'] <= last]
        if len(prefix) != sum(c['step'] <= last for c in cs):
            problems.append('the branch has fewer attempts than the GPT wall before k')
        for a, b in prefix:
            if a['step'] != b['step']:
                problems.append(f"attempt {a['step']}: step {b['step']} in the branch")
            problems += [f"step {a['step']}: {p}" for p in same_attempt(a, b, source, branch)]
        if arm == 'P':
            at = [(a, b) for a, b in zip(cs, cb) if a['step'] == k]
            if len(at) != 1:
                problems.append(f'step {k}: not one attempt in both walls')
            else:
                a, b = at[0]
                problems += [f'step {k}: {p}' for p in same_attempt(a, b, source, branch, select_too=False)]
                mb, xb = event(branch, b)
                paper = select(xb['features_raw'], xb['weights'], 'paper', PAPER_ORDER, mb['policy']['tol']).index
                if paper != int(xb['selected']):
                    problems.append(f'step {k}: the selection is not the Paper order on these candidates')
                if int(xb['selected']) == int(event(source, a)[1]['selected']):
                    problems.append(f'step {k}: the Paper candidate is GPT\'s: nothing to compare')
    for c in cb:
        want = tuple(switch.get(c['step'], PAPER_ORDER))
        if tuple(event(branch, c)[0]['policy']['order']) != want:
            problems.append(f"step {c['step']}: traced order is not the planned {want}")
    return problems


def check_wall(plan, name):
    wall, ref = plan['walls'][name], wall_ref(plan, name)
    run = cid.wall_dir(ref)
    receipt = json.loads((run / 'receipt.json').read_text(encoding='utf-8'))
    problems = [] if receipt['status'] == 'PROCESS_PASS' else [f"status {receipt['status']}"]
    if receipt.get('controller') is not None:
        problems.append('the branch ran a controller')
    problems += check_run.check_run(run)['problems']
    problems += check_branch(result_of(plan['source']), result_of(ref), wall)
    return problems


def branch_argv(plan, wall):
    """The run_wall arguments of one wall: the Paper order with GPT's recorded orders switched in."""
    switches = [a for s, o in sorted(wall['switch'].items()) for a in ('--switch', f"{s}={','.join(map(str, o))}")]
    return [plan['case'], '--parent', plan['source'], '--policy', 'paper', '--order', '3,0,2,1,4',
            '--seed', str(plan['seed']), '--timeout', str(plan['timeout_seconds'])] + switches


def summarize(plan):
    rows = []
    for name, wall in plan['walls'].items():
        if not built(plan, name):
            continue
        ref = wall_ref(plan, name)
        run = cid.wall_dir(ref)
        row, _ = campaign_quality.evaluate(run)
        lam = wall_metrics.mechanism_lambdas(result_of(ref))
        receipt = json.loads((run / 'receipt.json').read_text(encoding='utf-8'))
        rows.append({'wall': name, 'arm': wall['arm'], 'k': wall['k'], 'status': row['status'],
                     **{m: row.get(m) for m in ('F_SF', 'F_AH', 'F_AV', 'F_LR', 'placed_ratio', 'n_placed', 'n_input')},
                     'lambda_left': lam['left'], 'lambda_right': lam['right'], 'elapsed_seconds': receipt.get('elapsed_seconds'),
                     'check_problems': len(check_wall(plan, name))})
    by = {r['wall']: r for r in rows}
    deltas = {k: {m: by[f'G{k}'][m] - by[f'P{k}'][m] for m in ('F_SF', 'F_AH', 'F_AV', 'F_LR', 'placed_ratio')}
              for k in plan['steps'] if f'G{k}' in by and f'P{k}' in by and None not in
              [by[w][m] for w in (f'G{k}', f'P{k}') for m in ('F_SF', 'F_AH', 'F_AV', 'F_LR')]}
    out = {'rows': rows, 'g_minus_p': deltas}
    if campaign.is_study(plan['id']):  # a plan of the research/runs archive is only read
        (studies.path_of(plan['id']).parent / 'summary.json').write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    return out
