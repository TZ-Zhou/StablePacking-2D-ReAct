"""Counterfactual branches of a live GPT wall, rebuilt without any API.

  python -B research/gpt_counterfactual.py init                  # research/runs/branches/gpt-001-cf/plan.json, once
  python -B research/gpt_counterfactual.py run [--only V]        # build the walls not built yet, check each, stop at a failure
  python -B research/gpt_counterfactual.py check                 # check every built wall (and the reused P6)
  python -B research/gpt_counterfactual.py summarize             # wang-quality-v1 of every wall, G - P per step

A branch is run_case --policy paper --order 3,0,2,1,4 with GPT's recorded orders as --switch
STEP=order (the order traced at each step of the GPT wall). gpt-001 made one selection per step,
so a step switch replays it. Walls (C_j: GPT's orders at steps <= j, Paper after):
- V: every GPT order. It must rebuild the GPT wall exactly (check_case.compare_runs: decisions,
  filter, transformation, commits, base.pkl matrices, every trace event's meta and arrays), and
  every branch waits for it;
- for k in STEPS: G_k = GPT's orders at steps <= k; P_k = at steps < k, so the Paper order at k.
  Both continue with the Paper order and start from GPT's state before k. P6 has no switch: it is
  the dev Paper wall of the seed, reused only if it passes the P check.
check of a branch against the GPT wall: PROCESS_PASS, check_case.check_run ok, every attempt
before k (G: through k) byte-identical (commit line, trace meta but run_id, trace arrays); for P at
k, the same candidates (all arrays but 'selected'), the selection is Wang's select with the Paper
order on them, and it is not GPT's; the traced order of every step is the planned one.
Controller-only files of the GPT wall (controller.jsonl, transport.jsonl, its receipt) are format,
not construction, and are not compared. No controller and no API key reach a branch.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_case  # noqa: E402
import run_case  # noqa: E402

from Stonepacker2D.stack.selection import PAPER_ORDER, load_trace, select  # noqa: E402

PLAN_ID = 'gpt-001-cf'
SOURCE = 'gpt-walls/gpt-001'
PAPER_WALL = 'campaigns/dev-2seeds/thangavelu-noise80/paper/seed66794912'
STEPS = (6, 8, 17)
TIMEOUT = 2400
PLANS = run_case.RUNS / 'branches'
KEY_VARS = ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'STONEPACKING_ALLOW_PAID_API')


def result_of(run):
    return next((run_case.RUNS / run).glob('result_*'))


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
    seed = json.loads((run_case.RUNS / SOURCE / 'receipt.json').read_text(encoding='utf-8'))['seed']
    plan = {'id': PLAN_ID, 'created': datetime.now().isoformat(timespec='seconds'), 'source': SOURCE, 'seed': seed,
            'case': 'thangavelu-noise80', 'timeout_seconds': TIMEOUT, 'steps': list(STEPS), 'code_sha256': run_case.code_fingerprint(),
            'recorded_orders': {str(s): list(o) for s, o in orders.items()},
            'walls': {name: {'arm': w['arm'], 'k': w['k'], 'switch': {str(s): list(o) for s, o in w['switch'].items()},
                             'run': PAPER_WALL if name == 'P6' and not w['switch'] else f'branches/{PLAN_ID}/{name}'}
                      for name, w in walls.items()},
            'build_order': ['V'] + [n for k in STEPS for n in (f'G{k}', f'P{k}') if not (n == 'P6' and not walls[n]['switch'])],
            'continuation': 'paper order after k for both arms', 'api': 'none'}
    return plan


def plan_path():
    return PLANS / PLAN_ID / 'plan.json'


def load_plan():
    plan = json.loads(plan_path().read_text(encoding='utf-8'))
    for w in plan['walls'].values():
        w['switch'] = {int(s): tuple(o) for s, o in w['switch'].items()}
    return plan


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
        problems += check_case.compare_runs(Path(source).parent, Path(branch).parent)
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
    wall = plan['walls'][name]
    run = run_case.RUNS / wall['run']
    receipt = json.loads((run / 'receipt.json').read_text(encoding='utf-8'))
    problems = [] if receipt['status'] == 'PROCESS_PASS' else [f"status {receipt['status']}"]
    if receipt.get('controller') is not None:
        problems.append('the branch ran a controller')
    problems += check_case.check_run(run)['problems']
    problems += check_branch(result_of(plan['source']), result_of(wall['run']), wall)
    return problems


def run_plan(plan, only=None, launch=run_case.main):
    if run_case.code_fingerprint() != plan['code_sha256']:
        raise SystemExit('executed code differs from the code registered in plan.json')
    if any(os.environ.get(k) for k in KEY_VARS):
        raise SystemExit(f'unset {", ".join(KEY_VARS)}: a branch needs no key and no gate')
    log = PLANS / PLAN_ID / 'runs.jsonl'
    for name in plan['build_order']:
        if only and name not in only:
            continue
        if name != 'V':
            v = check_wall(plan, 'V') if (run_case.RUNS / plan['walls']['V']['run'] / 'receipt.json').exists() else ['V not built']
            if v:
                raise SystemExit(f'V has not passed: {v[:5]}; no branch is built')
            if 'P6' in plan['walls'] and plan['walls']['P6']['run'] == PAPER_WALL and (p6 := check_wall(plan, 'P6')):
                raise SystemExit(f'the reused P6 fails its check: {p6[:5]}')
        wall = plan['walls'][name]
        run = run_case.RUNS / wall['run']
        if run.exists():
            print(f'skip {name}: already built')
            continue
        argv = [plan['case'], wall['run'], '--policy', 'paper', '--order', '3,0,2,1,4', '--seed', str(plan['seed']),
                '--timeout', str(plan['timeout_seconds'])] + [a for s, o in sorted(wall['switch'].items())
                                                              for a in ('--switch', f"{s}={','.join(map(str, o))}")]
        started = datetime.now()
        launch(argv)
        problems = check_wall(plan, name)
        with log.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'wall': name, 'run': wall['run'], 'started': started.isoformat(timespec='seconds'),
                                'finished': datetime.now().isoformat(timespec='seconds'), 'problems': problems}) + '\n')
        print(f'{name}:', 'OK' if not problems else problems[:10])
        if problems:
            raise SystemExit(f'{name} failed its check: stopped, nothing else is built')


def summarize(plan):
    import metrics
    import quality_batch
    rows = []
    for name, wall in plan['walls'].items():
        run = run_case.RUNS / wall['run']
        if not (run / 'receipt.json').exists():
            continue
        row, _ = quality_batch.evaluate(run)
        lam = metrics.mechanism_lambdas(result_of(wall['run']))
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
    (PLANS / PLAN_ID / 'summary.json').write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('command', choices=['init', 'run', 'check', 'summarize'])
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args(argv)
    if a.command == 'init':
        if plan_path().exists():
            raise SystemExit(f'{plan_path()} exists: a plan is registered once')
        plan = make_plan()
        plan_path().parent.mkdir(parents=True)
        plan_path().write_text(json.dumps(plan, indent=1) + '\n', encoding='utf-8')
        print(json.dumps({k: plan[k] for k in ('recorded_orders', 'walls', 'build_order')}, indent=1))
        return 0
    plan = load_plan()
    if a.command == 'run':
        run_plan(plan, a.only)
        return 0
    if a.command == 'check':
        bad = {}
        for name, wall in plan['walls'].items():
            if (run_case.RUNS / wall['run'] / 'receipt.json').exists():
                bad[name] = check_wall(plan, name)
                print(name, 'OK' if not bad[name] else bad[name][:10])
        return 1 if any(bad.values()) else 0
    print(json.dumps(summarize(plan), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
