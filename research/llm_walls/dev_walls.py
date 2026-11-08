"""Development walls before any GPT wall: on 2 development seeds (not the formal campaign's), the
Paper wall and the same wall under the non-LLM controllers and the fake-provider ReAct controller,
each through the registered venv worker (run_wall --controller --worker-venv), one wall at a time.
research/runner/dev_walls.py builds them (usage there); then

  python -B -m research.walls.campaign_quality <id> --out research/quality/<id>   # Wang's metrics

The dev walls are a study (research/storage/studies.py, kind 'dev') whose design has a campaign's
shape (campaign.make_campaign with ARMS and the seeds here), so campaign.py reads it; each wall is a
case built by run_wall.run, listed as {case_id, arm, seed}. The seeds are drawn by a fixed rule
(SEED_RULE) and the arms fixed here before any wall is built.
check: check_run.check_run of every wall, order_controller.check of every controller wall (PASS), and
the Null wall identical to the Paper wall of its seed (check_run.compare_runs: decisions, filter,
commits, base.pkl matrices, every trace event); it writes manifest.csv beside study.json
(campaign_quality's input). dev-2seeds, check's default id, is in the research/runs archive
(campaigns/dev-2seeds): check reads its walls there and writes nothing.
Pre-registered for the GPT walls that follow: a GPT wall with any decision point not decided by
the LLM (a skip or a fallback) is 'GPT incomplete' and stays out of any quality comparison.
"""
import json

import numpy as np

from research.storage import case_id as cid
from research.walls import campaign as camp
from research.walls import case_definition
from research.walls import check_run
from research.controller import order_controller

ID = 'dev-2seeds'  # the dev walls of the research/runs archive
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


def init(study_id):
    """Register the dev walls as the study `study_id` (kind 'dev'), once."""
    for arm in ARMS:
        if arm['controller']:
            order_controller.validate_spec(arm['controller'])
    data = {**camp.make_campaign(study_id, 'thangavelu-noise80', ['paper=3,0,2,1,4'], 1, MASTER_SEED, TIMEOUT),
            'arms': ARMS, 'seeds': seeds(), 'seed_rule': SEED_RULE, 'worker_venv': case_definition.worker_venv_identity(),
            'notes': ['development walls; not the formal campaign; no metric of them chooses anything of the GPT walls',
                      'GPT walls: a wall with a skip or a fallback at any decision point is GPT incomplete, kept but not compared']}
    camp.register(data, kind='dev')
    return data


def argv_of(c, seed, arm):
    argv = [c['case'], '--policy', arm['policy'], '--order', ','.join(map(str, arm['order'])),
            '--seed', str(seed), '--timeout', str(c['timeout_seconds'])]
    return argv + (['--controller', json.dumps(arm['controller']), '--worker-venv'] if arm['controller'] else [])


def check(study_id=ID):
    c = camp.load(study_id)
    rows, problems = [], []
    planned = camp.planned_runs(c)
    walls = {(seed, arm['name']): wall for seed, arm, wall in planned}
    for seed, arm, wall in planned:
        name = wall or f"{arm['name']} seed {seed}"
        row = {'campaign': study_id, 'arm': arm['name'], 'policy': arm['policy'],
               'order': '-'.join(map(str, arm['order'])), 'seed': seed, 'run': wall or '',
               'status': camp.run_status(wall) or 'NOT_RUN'}
        if row['status'] == 'PROCESS_PASS':
            run = cid.wall_dir(wall)
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
        a, b = (walls[(seed, k)] for k in ('paper', 'null'))
        if camp.run_status(a) is not None and camp.run_status(b) is not None:
            diffs = check_run.compare_runs(cid.wall_dir(a), cid.wall_dir(b))
            problems += [f'seed {seed} null vs paper: {d}' for d in diffs]
            print(f'seed {seed}: null vs paper', 'IDENTICAL' if not diffs else diffs[:5])
    if camp.is_study(study_id):
        camp.write_rows(camp.manifest_path(study_id), rows)
    else:
        print(f'{study_id} is in the research/runs archive: its manifest.csv there is kept')
    print(json.dumps({'walls': len(rows), 'problems': problems}, indent=1))
    return problems
