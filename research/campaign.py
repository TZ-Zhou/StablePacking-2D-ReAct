"""Fixed filter-order campaigns: pre-registered arms x seeds, run one wall at a time.

  python -B research/campaign.py init <id> --case thangavelu-noise80 --arms code_original \
      paper=3,0,2,1,4 alt=2,4,3,1,0 random --n-seeds 3 [--master-seed 20261106] [--timeout 1200]
      [--extends <parent id>] [--note TEXT]
  python -B research/campaign.py run <id> [--dry-run] [--max-runs N]
  python -B research/campaign.py summarize <id>

init writes research/runs/campaigns/<id>/campaign.json once (never overwritten): the seed list
(20 first, the rest drawn from --master-seed), arms (code_original, or the paper cascade with a
fixed order; 'random' draws one order not used by another arm from the same master RNG), the
git head and the SHA256 of every executed source file. run refuses to start if that code
changed, so all arms share convolution, candidates, scores and stability analysis and differ
only in the filter priority. Each wall is its own process with seed_everything(seed) and only
the first wall of a batch case. Runs already present are never rerun or overwritten.
--extends reuses the parent's runs for every (arm, seed) it shares; it requires the same case,
timeout and executed code, and parent seeds that are a prefix of the new seed list (same
master seed). Give the parent's random order explicitly (random=a,b,c,d,e) to reuse it.
Stopping rules: FAILED stops the campaign; TERMINATION_FAILED raises; TIMEOUT and
INCOMPLETE_TRACE are recorded and the campaign continues.
summarize (re)writes the derived manifest.csv and steps.csv from the run directories.
"""
import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_case  # noqa: E402

from Stonepacker2D.stack.selection import ALL_ORDERS, CODE_ORIGINAL_ORDER, PAPER_ORDER  # noqa: E402

CAMPAIGNS = run_case.RUNS / 'campaigns'
SUCCESS = ('PROCESS_PASS',)


def parse_arm(text):
    if text == 'code_original':
        return {'name': 'code_original', 'policy': 'code_original', 'order': None}
    if text == 'random':
        return {'name': 'random', 'policy': 'paper', 'order': None}
    name, order = text.split('=')
    order = [int(c) for c in order.split(',')]
    if sorted(order) != [0, 1, 2, 3, 4]:
        raise ValueError(f'{text}: order must be a permutation of 0..4')
    return {'name': f"{name}_{'-'.join(map(str, order))}", 'policy': 'paper', 'order': order}


def make_campaign(campaign_id, case, arm_texts, n_seeds, master_seed, timeout, extends=None, notes=()):
    if case not in run_case.CASES:
        raise ValueError(f'unknown case {case}')
    rng = np.random.default_rng(master_seed)
    seeds = [20]
    while len(seeds) < n_seeds:
        s = int(rng.integers(1, 2 ** 31 - 1))
        if s not in seeds:
            seeds.append(s)
    arms = [parse_arm(t) for t in arm_texts]
    used = [tuple(a['order']) for a in arms if a['order']]
    for arm in arms:
        if arm['name'] == 'random':
            pool = [o for o in ALL_ORDERS if o not in used]
            arm['order'] = list(pool[int(rng.integers(len(pool)))])
            arm['name'] = f"random_{'-'.join(map(str, arm['order']))}"
            used.append(tuple(arm['order']))
    names = [a['name'] for a in arms]
    if len(set(names)) != len(names):
        raise ValueError(f'duplicate arms {names}')
    reused = []
    if extends:
        parent = load(extends)
        if (parent['case'], parent['timeout_seconds'], parent['code_sha256']) != (case, timeout, run_case.code_fingerprint()):
            raise ValueError(f'{extends}: case, timeout or executed code differs; its runs cannot be reused')
        if seeds[:len(parent['seeds'])] != parent['seeds']:
            raise ValueError(f"{extends}: seeds {parent['seeds']} are not a prefix of {seeds}")
        for seed, arm, run in planned_runs(parent):
            same = [a for a in arms if (a['name'], a['policy'], a['order']) == (arm['name'], arm['policy'], arm['order'])]
            if same:
                reused.append({'seed': seed, 'arm': arm['name'], 'run': run})
    git = lambda *a: subprocess.run(['git', *a], cwd=run_case.REPO, capture_output=True, text=True).stdout.strip()
    return {
        'id': campaign_id, 'created': datetime.now().isoformat(timespec='seconds'), 'case': case,
        'arms': arms, 'seeds': seeds, 'master_seed': master_seed, 'timeout_seconds': timeout,
        'seed_rule': 'seeds[0] = 20 (original default); others = default_rng(master_seed).integers(1, 2**31-1); '
                     'random arm order drawn afterwards from the same RNG among orders unused by other arms',
        'python_hash_seed': 20, 'paper_order': list(PAPER_ORDER), 'code_original_order': list(CODE_ORIGINAL_ORDER),
        'git_head': git('rev-parse', 'HEAD'), 'git_status': git('status', '--porcelain').splitlines(),
        'code_sha256': run_case.code_fingerprint(), 'extends': extends, 'reused_runs': reused, 'notes': list(notes),
    }


def planned_runs(campaign):
    """Seed-major order, so a stopped campaign has every arm on the same seeds. Runs reused from
    the parent campaign keep their directory there."""
    reused = {(r['seed'], r['arm']): r['run'] for r in campaign.get('reused_runs', [])}
    return [(seed, arm, reused.get((seed, arm['name']), f"campaigns/{campaign['id']}/{campaign['case']}/{arm['name']}/seed{seed}"))
            for seed in campaign['seeds'] for arm in campaign['arms']]


def run_status(name):
    receipt = run_case.RUNS / name / 'receipt.json'
    if not receipt.exists():
        return None if not (run_case.RUNS / name).exists() else 'NO_RECEIPT'
    return json.loads(receipt.read_text(encoding='utf-8'))['status']


def run_campaign(campaign, dry_run=False, max_runs=None, launch=run_case.main):
    if run_case.code_fingerprint() != campaign['code_sha256']:
        raise RuntimeError('executed code differs from the code registered in campaign.json')
    log = CAMPAIGNS / campaign['id'] / 'runs.jsonl'
    started = []
    for seed, arm, name in planned_runs(campaign):
        status = run_status(name)
        if status is not None:
            print(f'skip {name}: already {status}')
            continue
        if max_runs is not None and len(started) >= max_runs:
            break
        argv = [campaign['case'], name, '--policy', arm['policy'], '--seed', str(seed),
                '--timeout', str(campaign['timeout_seconds'])]
        if arm['order']:
            argv += ['--order', ','.join(map(str, arm['order']))]
        if dry_run:
            print('would run', ' '.join(argv))
            started.append(name)
            continue
        launch(argv)  # TERMINATION_FAILED raises and stops everything
        status = run_status(name)
        started.append(name)
        with log.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'run': name, 'seed': seed, 'arm': arm['name'], 'status': status,
                                'finished': datetime.now().isoformat(timespec='seconds')}) + '\n')
        if status == 'FAILED':
            print(f'{name} FAILED: campaign stopped for inspection')
            break
    return started


def summarize(campaign):
    import metrics
    from check_case import check_run, result_dirs
    root = CAMPAIGNS / campaign['id']
    manifest, steps, by_key = [], [], {}
    names = {(seed, arm['name']): name for seed, arm, name in planned_runs(campaign)}
    for seed, arm, name in planned_runs(campaign):
        run = run_case.RUNS / name
        row = {'campaign': campaign['id'], 'case': campaign['case'], 'arm': arm['name'], 'policy': arm['policy'],
               'order': '' if arm['order'] is None else '-'.join(map(str, arm['order'])), 'seed': seed,
               'run': name, 'status': run_status(name) or 'NOT_RUN'}
        results = result_dirs(run) if run.exists() else []
        if row['status'] in SUCCESS and len(results) == 1:
            check = check_run(run)
            rows = metrics.step_rows(results[0])
            by_key[(seed, arm['name'])] = rows
            row.update(metrics.wall_metrics(run, results[0]), checks_ok=check['ok'], n_check_problems=len(check['problems']))
            steps += [{'arm': arm['name'], 'seed': seed, **r} for r in rows]
        manifest.append(row)
    for row in manifest:  # decision divergence and identical first candidate set vs code_original, same seed
        ref = by_key.get((row['seed'], 'code_original'))
        mine = by_key.get((row['seed'], row['arm']))
        if ref is not None and mine is not None:
            row['first_divergence_step_vs_code_original'] = metrics.first_divergence(mine, ref)
            row['first_candidates_same_as_code_original'] = same_first_candidates(
                names[(row['seed'], row['arm'])], names[(row['seed'], 'code_original')])
    write_rows(root / 'manifest.csv', manifest)
    write_rows(root / 'steps.csv', steps)
    return manifest


def same_first_candidates(run, reference):
    """First filter event of the wall: identical raw score matrix and poses across arms."""
    from Stonepacker2D.stack.selection import load_trace
    def first(name):
        return load_trace(next((run_case.RUNS / name).rglob('trace/e00000.npz')))[1]
    a, b = first(run), first(reference)
    keys = ['features_raw', 'stone_id', 'rotation', 'x', 'y', 'wall', 'footprint']
    return all(np.array_equal(a[k], b[k]) for k in keys)


def write_rows(path, rows):
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load(campaign_id):
    return json.loads((CAMPAIGNS / campaign_id / 'campaign.json').read_text(encoding='utf-8'))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init')
    init.add_argument('id')
    init.add_argument('--case', required=True)
    init.add_argument('--arms', nargs='+', required=True)
    init.add_argument('--n-seeds', type=int, required=True)
    init.add_argument('--master-seed', type=int, default=20261106)
    init.add_argument('--timeout', type=int, default=1200)
    init.add_argument('--extends', help='reuse runs of this campaign for arms and seeds it shares')
    init.add_argument('--note', action='append', default=[], help='recorded in campaign.json, e.g. why an arm was chosen')
    run = sub.add_parser('run')
    run.add_argument('id')
    run.add_argument('--dry-run', action='store_true')
    run.add_argument('--max-runs', type=int)
    summ = sub.add_parser('summarize')
    summ.add_argument('id')
    args = parser.parse_args(argv)
    if args.command == 'init':
        campaign = make_campaign(args.id, args.case, args.arms, args.n_seeds, args.master_seed, args.timeout,
                                 args.extends, args.note)
        root = CAMPAIGNS / args.id
        root.mkdir(parents=True, exist_ok=False)
        (root / 'campaign.json').write_text(json.dumps(campaign, indent=2), encoding='utf-8')
        print(json.dumps({k: campaign[k] for k in ('id', 'case', 'arms', 'seeds')}, indent=2))
    elif args.command == 'run':
        run_campaign(load(args.id), args.dry_run, args.max_runs)
    else:
        for row in summarize(load(args.id)):
            print({k: row.get(k) for k in ('arm', 'seed', 'status', 'n_placed', 'stone_filling', 'lambda_min',
                                           'first_divergence_step_vs_code_original')})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
