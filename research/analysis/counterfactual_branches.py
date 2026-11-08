"""One-step filter-order branches of a recorded paper-order wall.

  python -B -m research.analysis.counterfactual_branches options <run> <target stone> [...]
  python -B -m research.analysis.counterfactual_branches init <plan id> --wall <run>:<seed>:<target> [...] [--timeout 2400]
  python -B -m research.analysis.counterfactual_branches run <plan id> [--max-runs N]
  python -B -m research.analysis.counterfactual_branches summarize <plan id>
(<run> is a recorded wall: a case id of research/cases or a path in the research/runs archive.)

options (offline): for every construction step of the recorded wall that was decided in a
single sampling attempt, replays all 120 paper-cascade orders (selection.select, Algorithm 2
with the LLC > 0 gate) on the recorded candidate matrix. Per step:
- picks_target: orders whose pick is a target stone (only feasible candidates can be picked);
- neutral: orders other than the recorded one that pick the recorded candidate (identity checks:
  the switch is active but nothing may change);
- control: orders that pick a non-target stone other than the recorded one.
A branch is 'run_wall.py --policy paper --order <recorded> --seed <seed> --switch STEP=<order>':
the same seed and code re-execute every earlier decision, so the candidate matrix at STEP is
the recorded one and the branch's choice there is the offline one; later steps use the
recorded order again but see a different wall and inventory (and the sampling draws that
follow from them). Nothing forces a stone or relaxes the LLC gate; a target is reachable only
if some order selects it.
init registers the plan once (never overwritten) as a study (research/storage/studies.py, kind
'branches') whose design is the plan: at every step where some order picks a target, one 'target'
branch per distinct target pose and one 'control' branch per distinct non-target pose other than the
recorded one, each with the lexicographically first order reaching that pose (orders reaching the
same pose build the same wall). run builds them one at a time, each a new case (run_wall.run
--parent <recorded wall>) that the study lists as {case_id, branch}; a branch the study already
lists is never rebuilt. After each wall it requires PROCESS_PASS, check_run ok and check_branch ok
(identical prefix, same candidates at the switched step, the planned pose chosen, recorded order
everywhere else), logs the result to runs.jsonl beside study.json and stops at the first failure.
summarize writes branches.csv beside study.json: each branch next to its original wall.
Plans of the research/runs archive (branches/<plan id>/plan.json, their walls beside it) are read
by load_plan and branch_wall like the studies, never built into or written.
"""
import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from research.paths import LEGACY_RUNS
from research.storage import case_id as cid
from research.storage import studies
from research.walls import campaign
from research.walls import run_wall

from Stonepacker2D.stack.selection import ALL_ORDERS, load_trace, select  # noqa: E402


def key(order):
    return '-'.join(map(str, order))


def step_options(result, targets):
    """One row per (single-attempt step, order)."""
    result = Path(result)
    commits = [json.loads(l) for l in (result / 'trace' / 'commits.jsonl').read_text(encoding='utf-8').splitlines()]
    attempts = pd.Series([c['step'] for c in commits]).value_counts()
    rows = []
    for c in commits:
        if attempts[c['step']] != 1 or c['outcome'] != 'placed':
            continue
        side = c['sides'][c['chosen_side']]
        meta, a = load_trace(result / 'trace' / f"e{side['event']:05d}.npz")
        p = meta['policy']
        assert p['policy'] == 'paper', 'branches are defined for paper-policy walls'
        recorded = int(a['selected'])
        for order in ALL_ORDERS:
            i = select(a['features_raw'], a['weights'], 'paper', order, p['tol']).index
            rows.append({'step': c['step'], 'event': side['event'], 'n_remaining': len(a['inventory_ids']) + len(a['sampled_ids']),
                         'recorded_order': key(p['order']), 'order': key(order), 'stone': int(a['stone_id'][i]),
                         'rotation': int(a['rotation_world'][i]), 'x': int(a['x_world'][i]), 'y': int(a['y'][i]),
                         'same_as_recorded': i == recorded, 'target': int(a['stone_id'][i]) in targets,
                         'targets_sampled': ' '.join(str(t) for t in targets if t in a['sampled_ids'])})
    return pd.DataFrame(rows)


def summarize(options):
    out = []
    for step, g in options.groupby('step'):
        rec = g[g.order == g.recorded_order.iloc[0]].iloc[0]
        others = g[g.order != g.recorded_order]
        tgt = others[others.target]
        out.append({'step': step, 'n_remaining': rec.n_remaining, 'recorded_stone': rec.stone,
                    'targets_sampled': rec.targets_sampled, 'n_picks_target': len(tgt),
                    'target_poses': len(tgt.groupby(['stone', 'rotation', 'x', 'y'])),
                    'n_neutral': int(others.same_as_recorded.sum()),
                    'n_control': int((~others.same_as_recorded & ~others.target).sum()),
                    'control_poses': len(others[~others.same_as_recorded & ~others.target].groupby(['stone', 'rotation', 'x', 'y'])),
                    'picks_target': ' '.join(tgt.order)})
    return pd.DataFrame(out)


def check_branch(original, branch, switch):
    """Branch result dir vs the recorded one. switch: {step: order tuple} as given to run_wall.
    Every attempt before the first switched step must be byte-identical (commits line, trace
    arrays, trace meta except run_id); at a switched step the trace meta policy must carry the
    switched order and the choice must be the offline replay's; every other step must carry
    the recorded order."""
    import numpy as np
    original, branch = Path(original), Path(branch)
    lines = lambda r: (r / 'trace' / 'commits.jsonl').read_text(encoding='utf-8').splitlines()
    lo, lb = lines(original), lines(branch)
    first = min(switch)
    problems = []
    n_prefix = 0
    for a, b in zip(lo, lb):
        ca, cb = json.loads(a), json.loads(b)
        at_switch = ca['step'] >= first   # first switched attempt: same state and candidates, own selection
        if not at_switch:
            n_prefix += 1
            if a != b:
                problems.append(f"step {ca['step']}: commit differs")
        ea, eb = ca['sides'][ca['chosen_side']]['event'], cb['sides'][cb['chosen_side']]['event']
        (ma, aa), (mb, ab) = load_trace(original / 'trace' / f'e{ea:05d}.npz'), load_trace(branch / 'trace' / f'e{eb:05d}.npz')
        skip = {'run_id'} | ({'policy', 'stages', 'branch'} if at_switch else set())
        if {k: v for k, v in ma.items() if k not in skip} != {k: v for k, v in mb.items() if k not in skip}:
            problems.append(f"step {ca['step']}: trace meta differs")
        keys = [k for k in aa if not (at_switch and k == 'selected')]
        if sorted(aa) != sorted(ab) or any(not np.array_equal(aa[k], ab[k], equal_nan=aa[k].dtype.kind == 'f') for k in keys):
            problems.append(f"step {ca['step']}: trace arrays differ")
        if at_switch:
            break
    recorded = tuple(load_trace(original / 'trace' / 'e00000.npz')[0]['policy']['order'])
    choices = {}
    for line in lb:
        c = json.loads(line)
        meta, a = load_trace(branch / 'trace' / f"e{c['sides'][c['chosen_side']]['event']:05d}.npz")
        want = tuple(switch.get(c['step'], recorded))
        if tuple(meta['policy']['order']) != want:
            problems.append(f"step {c['step']}: order {meta['policy']['order']} instead of {want}")
        if c['step'] in switch and c['outcome'] == 'placed':
            choices[c['step']] = (c['stone_id'], c['rotation_world'], c['x_world'], c['y'])
            replay = select(a['features_raw'], a['weights'], 'paper', switch[c['step']], meta['policy']['tol']).index
            if replay != int(a['selected']):
                problems.append(f"step {c['step']}: live choice differs from the offline replay")
    return {'ok': not problems, 'problems': problems, 'identical_prefix_attempts': n_prefix, 'switched_choices': choices}


def legacy_plan_dir(plan_id):
    """The folder of a plan built in the research/runs archive (read only)."""
    return LEGACY_RUNS / 'branches' / plan_id


def load_plan(plan_id):
    """A plan: the design of its study, or plan.json of a plan in the research/runs archive."""
    if campaign.is_study(plan_id):
        return studies.load(plan_id)['design']
    return json.loads((legacy_plan_dir(plan_id) / 'plan.json').read_text(encoding='utf-8'))


def register(plan):
    """Create the study of a new plan; an id the research/runs archive already holds is refused."""
    if legacy_plan_dir(plan['id']).exists():
        raise FileExistsError(f"{plan['id']} is a plan of the research/runs archive: choose another id")
    return studies.create(plan['id'], 'branches', plan)


def branch_wall(plan, b):
    """The wall of branch b, for case_id.wall_dir: the case its study lists (None while there is none), or
    for a plan of the research/runs archive its folder beside plan.json."""
    if campaign.is_study(plan['id']):
        found = studies.find(plan['id'], branch=b['name'])
        return found[-1]['case_id'] if found else None
    return f"branches/{plan['id']}/{b['name']}"


def make_plan(plan_id, walls, timeout):
    """walls: [(run, seed, target stone)], run a case id or a research/runs path."""
    branches = []
    for run, seed, target in walls:
        result = next(cid.wall_dir(run).rglob('trace/commits.jsonl')).parent.parent
        options = step_options(result, {target})
        for step in sorted(options[options.target].step.unique()):
            g = options[(options.step == step) & (options.order != options.recorded_order)]
            for kind, sub in (('target', g[g.target]), ('control', g[~g.target & ~g.same_as_recorded])):
                for pose, h in sub.groupby(['stone', 'rotation', 'x', 'y']):
                    order = min(h.order)
                    branches.append({'name': f"{Path(run).name}-step{step}-{kind}-stone{pose[0]}-{order}", 'run': run,
                                     'seed': seed, 'target': target, 'step': int(step), 'kind': kind,
                                     'n_remaining': int(h.n_remaining.iloc[0]), 'pose': [int(v) for v in pose],
                                     'order': order, 'orders_reaching_pose': len(h),
                                     'recorded_order': h.recorded_order.iloc[0]})
    git = lambda *a: subprocess.run(['git', *a], cwd=run_wall.REPO, capture_output=True, text=True).stdout.strip()
    return {'id': plan_id, 'case': 'thangavelu-noise80', 'timeout_seconds': timeout, 'branches': branches,
            'git_head': git('rev-parse', 'HEAD'), 'git_status': git('status', '--porcelain').splitlines(),
            'code_sha256': run_wall.code_fingerprint()}


def run_plan(plan, max_runs=None, launch=run_wall.run):
    """True when every branch built so far passed; stops at the first failure."""
    from research.walls import check_run
    campaign.require_study(plan['id'])
    if run_wall.code_fingerprint() != plan['code_sha256']:
        raise RuntimeError('executed code differs from the code registered in the plan')
    started = 0
    for b in plan['branches']:
        if campaign.run_status(branch_wall(plan, b)) is not None:
            continue
        if max_runs is not None and started >= max_runs:
            break
        argv = [plan['case'], '--parent', b['run'], '--policy', 'paper',
                '--order', b['recorded_order'].replace('-', ','), '--seed', str(b['seed']),
                '--timeout', str(plan['timeout_seconds']), '--switch', f"{b['step']}={b['order'].replace('-', ',')}"]
        receipt = campaign.launch_case(plan['id'], argv, {'branch': b['name']}, launch)
        started += 1
        record = {'branch': b['name'], 'case_id': receipt['case_id'], 'status': receipt['status'],
                  'check_run_ok': False, 'check_branch_ok': False}
        if receipt['status'] == 'PROCESS_PASS':
            original, branch = (check_run.result_dirs(cid.wall_dir(r))[0] for r in (b['run'], receipt['case_id']))
            run_check = check_run.check_run(cid.wall_dir(receipt['case_id']))
            bc = check_branch(original, branch, {b['step']: tuple(int(c) for c in b['order'].split('-'))})
            chosen = bc['switched_choices'].get(b['step'])
            if chosen is None or list(chosen) != b['pose']:
                bc['ok'] = False
                bc['problems'].append(f"step {b['step']}: chose {chosen}, planned {b['pose']}")
            record.update(check_run_ok=run_check['ok'], check_branch_ok=bc['ok'],
                          problems=run_check['problems'][:5] + bc['problems'][:5],
                          identical_prefix_attempts=bc['identical_prefix_attempts'])
        campaign.log_run(plan['id'], record)
        if not (record['check_run_ok'] and record['check_branch_ok']):
            print(f"{b['name']}: {record}; plan stopped for inspection")
            return False
    return True


def summarize_plan(plan):
    """Write branches.csv beside the plan's study.json; returns its rows."""
    from research.walls import check_run
    from research.walls import wall_metrics
    campaign.require_study(plan['id'])
    rows = []
    for b in plan['branches']:
        wall = branch_wall(plan, b)
        if campaign.run_status(wall) in (None, 'NO_RECEIPT'):
            continue
        row = {k: b[k] for k in ('name', 'run', 'seed', 'target', 'step', 'n_remaining', 'kind', 'order')}
        row.update(wall=wall, pose=' '.join(map(str, b['pose'])))
        after = {}
        for tag, run in (('orig', b['run']), ('branch', wall)):
            result = check_run.result_dirs(cid.wall_dir(run))[0]
            m = wall_metrics.wall_metrics(cid.wall_dir(run), result)
            for k in ('status', 'n_placed', 'wall_density', 'wall_height', 'lambda_min', 'stone_filling', 'termination'):
                row[f'{k}_{tag}'] = m[k]
            commits = [json.loads(l) for l in (result / 'trace' / 'commits.jsonl').read_text(encoding='utf-8').splitlines()]
            placed = [c['stone_id'] for c in commits if c['outcome'] == 'placed']
            after[tag] = [c['stone_id'] for c in commits if c['outcome'] == 'placed' and c['step'] >= b['step']]
            last = commits[-1]
            row[f'unplaced_{tag}'] = ' '.join(map(str, last['sampled_ids'])) if last['outcome'] == 'no_placement' else ''
            row[f'target_position_{tag}'] = placed.index(b['target']) + 1 if b['target'] in placed else None
            row[f'placed_from_step_{tag}'] = ' '.join(map(str, after[tag]))
        row['same_stone_same_position_after_step'] = sum(x == y for x, y in zip(after['orig'][1:], after['branch'][1:]))
        for k in ('n_placed', 'wall_density', 'wall_height', 'lambda_min'):
            row[f'diff_{k}'] = row[f'{k}_branch'] - row[f'{k}_orig']
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(studies.path_of(plan['id']).parent / 'branches.csv', index=False)
    return df


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    opt = sub.add_parser('options')
    opt.add_argument('run')
    opt.add_argument('targets', nargs='+', type=int)
    init = sub.add_parser('init')
    init.add_argument('id')
    init.add_argument('--wall', action='append', required=True, metavar='RUN:SEED:TARGET')
    init.add_argument('--timeout', type=int, default=2400)
    run = sub.add_parser('run')
    run.add_argument('id')
    run.add_argument('--max-runs', type=int)
    summ = sub.add_parser('summarize')
    summ.add_argument('id')
    args = parser.parse_args(argv)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 80)
    if args.command == 'options':
        result = next(cid.wall_dir(args.run).rglob('trace/commits.jsonl')).parent.parent
        print(summarize(step_options(result, set(args.targets))).to_string(index=False))
        return 0
    if args.command == 'init':
        walls = [(run, int(seed), int(target)) for run, seed, target in (w.rsplit(':', 2) for w in args.wall)]
        plan = make_plan(args.id, walls, args.timeout)
        register(plan)
        print(pd.DataFrame(plan['branches']).drop(columns=['name', 'run']).to_string(index=False))
        return 0
    plan = load_plan(args.id)
    if args.command == 'run':
        return 0 if run_plan(plan, args.max_runs) else 1
    print(summarize_plan(plan).to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
