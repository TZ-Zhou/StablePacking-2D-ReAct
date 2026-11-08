"""Development walls before any GPT wall: on 2 development seeds (not the formal campaign's), the
Paper wall and the same wall under the non-LLM controllers and the fake-provider ReAct controller,
each through the registered venv worker (run_wall --controller --worker-venv), one wall at a time.

  python -B -m research.llm_walls.dev_walls init      # research/runs/campaigns/dev-2seeds/campaign.json, once
  python -B -m research.llm_walls.dev_walls run       # builds the walls not built yet, then check
  python -B -m research.llm_walls.dev_walls check     # check every wall, Null == Paper, writes manifest.csv
  python -B -m research.walls.campaign_quality dev-2seeds --out research/quality/dev-2seeds   # Wang's metrics

The seeds are drawn by a fixed rule (SEED_RULE) and the arms fixed here before any wall is built.
check: check_run.check_run of every wall, order_controller.check of every controller wall (PASS), and
the Null wall identical to the Paper wall of its seed (check_run.compare_runs: decisions, filter,
commits, base.pkl matrices, every trace event); it writes manifest.csv (campaign_quality's input).
Pre-registered for the GPT walls that follow: a GPT wall with any decision point not decided by
the LLM (a skip or a fallback) is 'GPT incomplete' and stays out of any quality comparison.
"""
import argparse
import json
import sys
from datetime import datetime

import numpy as np

from research.walls import campaign as camp
from research.walls import check_run
from research.controller import order_controller
from research.walls import run_wall

ID = 'dev-2seeds'
MASTER_SEED = 20261108
SEED_RULE = ('default_rng(20261108).integers(1, 2**31-1) drawn until 2 seeds not in the formal campaign '
             'thangavelu-5arms-10seeds (and distinct)')
FAKE_REACT = {'kind': 'react', 'llm': {'kind': 'api', 'provider': 'fake', 'model': 'fake-1', 'fake_mode': 'max_llc'},
              'max_tool_calls': 3, 'max_decisions': 20, 'max_llm_failures': 1, 'observation_version': 2}
ARMS = [{'name': 'paper', 'policy': 'paper', 'order': [3, 0, 2, 1, 4], 'controller': None},
        {'name': 'null', 'policy': 'paper', 'order': [3, 0, 2, 1, 4], 'controller': {'kind': 'null'}},
        {'name': 'rule', 'policy': 'paper', 'order': [3, 0, 2, 1, 4], 'controller': {'kind': 'rule', 'max_remaining': 6, 'llc_drop': 0.2}},
        {'name': 'random', 'policy': 'paper', 'order': [3, 0, 2, 1, 4], 'controller': {'kind': 'random', 'seed': MASTER_SEED}},
        {'name': 'fake_react', 'policy': 'paper', 'order': [3, 0, 2, 1, 4], 'controller': FAKE_REACT}]
TIMEOUT = 2400


def seeds():
    formal = set(camp.load('thangavelu-5arms-10seeds')['seeds'])
    rng, out = np.random.default_rng(MASTER_SEED), []
    while len(out) < 2:
        s = int(rng.integers(1, 2 ** 31 - 1))
        if s not in formal and s not in out:
            out.append(s)
    return out


def init():
    path = camp.CAMPAIGNS / ID / 'campaign.json'
    if path.exists():
        raise SystemExit(f'{path} exists: a campaign is registered once')
    for arm in ARMS:
        if arm['controller']:
            order_controller.validate_spec(arm['controller'])
    data = {**camp.make_campaign(ID, 'thangavelu-noise80', ['paper=3,0,2,1,4'], 1, MASTER_SEED, TIMEOUT),
            'arms': ARMS, 'seeds': seeds(), 'seed_rule': SEED_RULE, 'worker_venv': run_wall.worker_venv_identity(),
            'notes': ['development walls; not the formal campaign; no metric of them chooses anything of the GPT walls',
                      'GPT walls: a wall with a skip or a fallback at any decision point is GPT incomplete, kept but not compared']}
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data, indent=1) + '\n', encoding='utf-8')
    return data


def argv_of(c, seed, arm, name):
    argv = [c['case'], name, '--policy', arm['policy'], '--order', ','.join(map(str, arm['order'])),
            '--seed', str(seed), '--timeout', str(c['timeout_seconds'])]
    return argv + (['--controller', json.dumps(arm['controller']), '--worker-venv'] if arm['controller'] else [])


def run(launch=run_wall.main):
    c = camp.load(ID)
    if run_wall.code_fingerprint() != c['code_sha256']:
        raise SystemExit('executed code differs from the code registered in campaign.json')
    for seed, arm, name in camp.planned_runs(c):
        status = camp.run_status(name)
        if status is not None:
            print(f'skip {name}: already {status}')
            continue
        launch(argv_of(c, seed, arm, name))
        status = camp.run_status(name)
        with (camp.CAMPAIGNS / ID / 'runs.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps({'run': name, 'seed': seed, 'arm': arm['name'], 'status': status,
                                'finished': datetime.now().isoformat(timespec='seconds')}) + '\n')
        if status != 'PROCESS_PASS':
            raise SystemExit(f'{name} {status}: stopped for inspection')
    return check()


def check():
    c = camp.load(ID)
    rows, problems = [], []
    runs = {(seed, arm['name']): name for seed, arm, name in camp.planned_runs(c)}
    for seed, arm, name in camp.planned_runs(c):
        run = run_wall.RUNS / name
        row = {'campaign': ID, 'arm': arm['name'], 'policy': arm['policy'], 'order': '-'.join(map(str, arm['order'])),
               'seed': seed, 'run': name, 'status': camp.run_status(name) or 'NOT_RUN'}
        if row['status'] == 'PROCESS_PASS':
            found = check_run.check_run(run)['problems']
            if arm['controller']:
                report = order_controller.check(check_run.result_dirs(run)[0])
                found += [f"controller check {report['verdict']}: {p}" for p in report['problems'] or report.get('incomplete') or []] \
                    if report['verdict'] != 'PASS' else []
                row.update(decision_points=report['decision_points'], switches=report['switches'], controller_check=report['verdict'])
            row['check_problems'] = len(found)
            problems += [f'{name}: {p}' for p in found]
        else:
            problems.append(f"{name}: {row['status']}")
        rows.append(row)
    for seed in c['seeds']:
        a, b = (run_wall.RUNS / runs[(seed, k)] for k in ('paper', 'null'))
        if a.exists() and b.exists():
            diffs = check_run.compare_runs(a, b)
            problems += [f'seed {seed} null vs paper: {d}' for d in diffs]
            print(f'seed {seed}: null vs paper', 'IDENTICAL' if not diffs else diffs[:5])
    camp.write_rows(camp.CAMPAIGNS / ID / 'manifest.csv', rows)
    print(json.dumps({'walls': len(rows), 'problems': problems}, indent=1))
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('command', choices=['init', 'run', 'check'])
    a = ap.parse_args(argv)
    if a.command == 'init':
        print(json.dumps(init(), indent=1))
        return 0
    return 1 if (run() if a.command == 'run' else check()) else 0


if __name__ == '__main__':
    sys.exit(main())
