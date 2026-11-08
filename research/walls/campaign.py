"""Fixed filter-order campaigns: pre-registered arms x seeds, run one wall at a time.

  python -B -m research.walls.campaign init <id> --case thangavelu-noise80 --arms code_original \
      paper=3,0,2,1,4 alt=2,4,3,1,0 random --n-seeds 3 [--master-seed 20261106] [--timeout 1200]
      [--extends <parent id>] [--note TEXT]
  python -B -m research.walls.campaign run <id> [--dry-run] [--max-runs N]
  python -B -m research.walls.campaign summarize <id>

A campaign is a study (research/storage/studies.py, kind 'campaign'). init writes its
research/studies/<id>/study.json once (never overwritten); the design there holds the seed list (20
first, the rest drawn from --master-seed), the arms (code_original, or the paper cascade with a fixed
order; 'random' draws one order not used by another arm from the same master RNG), the git head and
the SHA256 of every executed source file. run refuses to start if that code changed, so all arms
share convolution, candidates, scores and stability analysis and differ only in the filter priority.
Each wall is one run_wall.run (a new research/cases/<case id>) with seed_everything(seed) and only the
first wall of a batch case; the study lists it as {case_id, arm, seed} and runs.jsonl beside
study.json logs it. A cell that has a wall is never rerun or overwritten.
--extends reuses the parent's walls for every (arm, seed) it shares; it requires the same case,
timeout and executed code, and parent seeds that are a prefix of the new seed list (same master
seed). Give the parent's random order explicitly (random=a,b,c,d,e) to reuse it. The parent may be
a campaign of the research/runs archive (campaigns/<id>/campaign.json, walls at
campaigns/<id>/<case>/<arm>/seed<s>): its walls are read where they are, never copied.
Stopping rules, from the wall's receipt status: FAILED stops the campaign; TERMINATION_FAILED
raises; TIMEOUT and INCOMPLETE_TRACE are recorded and the campaign continues.
summarize (re)writes the derived manifest.csv and steps.csv beside study.json; their run column is a
case id or a research/runs path, which case_id.wall_dir resolves. Archived campaigns are only read:
load, planned_runs, run_status and manifest_path serve both kinds.
"""
import argparse
import csv
import json
import subprocess
from datetime import datetime

import numpy as np

from research.paths import LEGACY_RUNS
from research.storage import case_id as cid
from research.storage import studies
from research.walls import run_wall

from Stonepacker2D.stack.selection import ALL_ORDERS, CODE_ORIGINAL_ORDER, PAPER_ORDER  # noqa: E402

SUCCESS = ('PROCESS_PASS',)


def legacy_dir(campaign_id):
    """The folder of a campaign built in the research/runs archive (read only)."""
    return LEGACY_RUNS / 'campaigns' / campaign_id


def is_study(study_id):
    """True when research/studies holds this id (a campaign, dev walls, a branch plan, a lookahead study)."""
    return studies.STUDY_ID.fullmatch(study_id) is not None and studies.path_of(study_id).exists()


def folder(campaign_id):
    """Where a campaign's tables are: beside its study.json, or its folder in the research/runs archive."""
    return studies.path_of(campaign_id).parent if is_study(campaign_id) else legacy_dir(campaign_id)


def manifest_path(campaign_id):
    """manifest.csv of a campaign (summarize): beside its study.json, or in the research/runs archive."""
    return folder(campaign_id) / 'manifest.csv'


def load(campaign_id):
    """The design of a campaign: its study's, or campaign.json of a campaign in the research/runs archive."""
    if is_study(campaign_id):
        return studies.load(campaign_id)['design']
    return json.loads((legacy_dir(campaign_id) / 'campaign.json').read_text(encoding='utf-8'))


def register(design, kind='campaign'):
    """Create the study of a new campaign; an id the research/runs archive already holds is refused."""
    if legacy_dir(design['id']).exists():
        raise FileExistsError(f"{design['id']} is a campaign of the research/runs archive: choose another id")
    return studies.create(design['id'], kind, design)


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
    if case not in run_wall.CASES:
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
        if (parent['case'], parent['timeout_seconds'], parent['code_sha256']) != (case, timeout, run_wall.code_fingerprint()):
            raise ValueError(f'{extends}: case, timeout or executed code differs; its runs cannot be reused')
        if seeds[:len(parent['seeds'])] != parent['seeds']:
            raise ValueError(f"{extends}: seeds {parent['seeds']} are not a prefix of {seeds}")
        for seed, arm, wall in planned_runs(parent):
            same = [a for a in arms if (a['name'], a['policy'], a['order']) == (arm['name'], arm['policy'], arm['order'])]
            if same and run_status(wall) is not None:
                reused.append({'seed': seed, 'arm': arm['name'], 'run': wall})
    git = lambda *a: subprocess.run(['git', *a], cwd=run_wall.REPO, capture_output=True, text=True).stdout.strip()
    return {
        'id': campaign_id, 'created': datetime.now().isoformat(timespec='seconds'), 'case': case,
        'arms': arms, 'seeds': seeds, 'master_seed': master_seed, 'timeout_seconds': timeout,
        'seed_rule': 'seeds[0] = 20 (original default); others = default_rng(master_seed).integers(1, 2**31-1); '
                     'random arm order drawn afterwards from the same RNG among orders unused by other arms',
        'python_hash_seed': 20, 'paper_order': list(PAPER_ORDER), 'code_original_order': list(CODE_ORIGINAL_ORDER),
        'git_head': git('rev-parse', 'HEAD'), 'git_status': git('status', '--porcelain').splitlines(),
        'code_sha256': run_wall.code_fingerprint(), 'extends': extends, 'reused_runs': reused, 'notes': list(notes),
    }


def planned_runs(campaign):
    """(seed, arm, wall) in seed-major order, so a stopped campaign has every arm on the same seeds. wall is
    what case_id.wall_dir resolves: the case id the campaign's study lists for the cell, else a wall reused
    from the parent campaign; for a campaign of the research/runs archive its path there. None while a cell
    of a study has no wall."""
    walls = {(r['seed'], r['arm']): r['run'] for r in campaign.get('reused_runs', [])}
    if is_study(campaign['id']):
        walls.update({(e['seed'], e['arm']): e['case_id'] for e in studies.load(campaign['id'])['cases']})
    else:
        nested = {(s, a['name']): f"campaigns/{campaign['id']}/{campaign['case']}/{a['name']}/seed{s}"
                  for s in campaign['seeds'] for a in campaign['arms']}
        walls = {**nested, **walls}
    return [(seed, arm, walls.get((seed, arm['name']))) for seed in campaign['seeds'] for arm in campaign['arms']]


def run_status(wall):
    """The status in a wall's receipt.json; None when there is no wall, NO_RECEIPT for a folder without one."""
    if wall is None:
        return None
    path = cid.wall_dir(wall)
    if not (path / 'receipt.json').exists():
        return 'NO_RECEIPT' if path.exists() else None
    return json.loads((path / 'receipt.json').read_text(encoding='utf-8'))['status']


def launch_case(study_id, argv, keys, launch=run_wall.run):
    """Build one wall of a study with `launch` (run_wall.run: a new case folder) and list it there as
    {'case_id', **keys}; returns its receipt. TERMINATION_FAILED lists the wall, then raises and stops the
    whole batch."""
    try:
        receipt = launch(argv)
    except run_wall.TerminationFailed as error:
        studies.add_case(study_id, {'case_id': error.receipt['case_id'], **keys})
        raise
    studies.add_case(study_id, {'case_id': receipt['case_id'], **keys})
    return receipt


def log_run(study_id, record):
    """Append one record to runs.jsonl beside the study's study.json."""
    with (studies.path_of(study_id).parent / 'runs.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')


def require_study(study_id):
    """Walls are built, and tables written, only for a study: the research/runs archive is only read."""
    if not is_study(study_id):
        raise RuntimeError(f'{study_id} is not in research/studies: what the research/runs archive holds is only read')


def run_campaign(campaign, dry_run=False, max_runs=None, launch=run_wall.run):
    require_study(campaign['id'])
    if run_wall.code_fingerprint() != campaign['code_sha256']:
        raise RuntimeError('executed code differs from the code registered in the study')
    started = []
    for seed, arm, wall in planned_runs(campaign):
        status = run_status(wall)
        if status is not None:
            print(f'skip {wall}: already {status}')
            continue
        if max_runs is not None and len(started) >= max_runs:
            break
        argv = [campaign['case'], '--policy', arm['policy'], '--seed', str(seed),
                '--timeout', str(campaign['timeout_seconds'])]
        if arm['order']:
            argv += ['--order', ','.join(map(str, arm['order']))]
        if dry_run:
            print('would run', ' '.join(argv))
            started.append(f"{arm['name']} seed {seed}")
            continue
        receipt = launch_case(campaign['id'], argv, {'arm': arm['name'], 'seed': seed}, launch)
        started.append(receipt['case_id'])
        log_run(campaign['id'], {'run': receipt['case_id'], 'seed': seed, 'arm': arm['name'],
                                 'status': receipt['status'], 'finished': datetime.now().isoformat(timespec='seconds')})
        if receipt['status'] == 'FAILED':
            print(f"{receipt['case_id']} FAILED: campaign stopped for inspection")
            break
    return started


def summarize(campaign):
    """(Re)write manifest.csv and steps.csv beside the campaign's study.json; returns the manifest rows."""
    from research.walls import wall_metrics
    from research.walls.check_run import check_run, result_dirs
    require_study(campaign['id'])
    manifest, steps, by_key = [], [], {}
    planned = planned_runs(campaign)
    walls = {(seed, arm['name']): wall for seed, arm, wall in planned}
    for seed, arm, wall in planned:
        run = None if wall is None else cid.wall_dir(wall)
        row = {'campaign': campaign['id'], 'case': campaign['case'], 'arm': arm['name'], 'policy': arm['policy'],
               'order': '' if arm['order'] is None else '-'.join(map(str, arm['order'])), 'seed': seed,
               'run': wall or '', 'status': run_status(wall) or 'NOT_RUN'}
        results = result_dirs(run) if run is not None and run.exists() else []
        if row['status'] in SUCCESS and len(results) == 1:
            check = check_run(run)
            rows = wall_metrics.step_rows(results[0])
            by_key[(seed, arm['name'])] = rows
            row.update(wall_metrics.wall_metrics(run, results[0]), checks_ok=check['ok'], n_check_problems=len(check['problems']))
            steps += [{'arm': arm['name'], 'seed': seed, **r} for r in rows]
        manifest.append(row)
    for row in manifest:  # decision divergence and identical first candidate set vs code_original, same seed
        ref = by_key.get((row['seed'], 'code_original'))
        mine = by_key.get((row['seed'], row['arm']))
        if ref is not None and mine is not None:
            row['first_divergence_step_vs_code_original'] = wall_metrics.first_divergence(mine, ref)
            row['first_candidates_same_as_code_original'] = same_first_candidates(
                walls[(row['seed'], row['arm'])], walls[(row['seed'], 'code_original')])
    write_rows(manifest_path(campaign['id']), manifest)
    write_rows(folder(campaign['id']) / 'steps.csv', steps)
    return manifest


def same_first_candidates(run, reference):
    """First filter event of the wall: identical raw score matrix and poses across arms."""
    from Stonepacker2D.stack.selection import load_trace
    def first(wall):
        return load_trace(next(cid.wall_dir(wall).rglob('trace/e00000.npz')))[1]
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
    init.add_argument('--note', action='append', default=[], help='recorded in the design, e.g. why an arm was chosen')
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
        register(campaign)
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
