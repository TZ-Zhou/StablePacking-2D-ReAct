"""Checks for run_wall --seed, campaign.py, wall_metrics.py and check_run.py.

Run from the repository root: python -B -m unittest discover -s research/tests
Tests marked 'existing run' use traced runs under research/runs (git-ignored) and are skipped
when those directories are absent.
"""
import difflib
import itertools
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from research.paths import LEGACY_RUNS  # noqa: E402
from research.storage import case_id  # noqa: E402
from research.storage import studies  # noqa: E402
from research.walls import campaign  # noqa: E402
from research.walls import check_run  # noqa: E402
from research.walls import wall_metrics  # noqa: E402
from research.walls import run_wall  # noqa: E402

THANGAVELU = REPO / 'research' / 'runs' / 'thangavelu-noise80-code-original-full-01'
PR_RIGHT = REPO / 'research' / 'runs' / 'validation-pr-paper-right-01'
MINUTES = itertools.count()


def sandbox(test, *more):
    """Point research/runs, research/cases and research/studies at a temporary folder for `test` and return
    it; each of `more` is a further module whose LEGACY_RUNS is patched."""
    tmp = tempfile.TemporaryDirectory()
    test.addCleanup(tmp.cleanup)
    root = Path(tmp.name)
    targets = [(case_id, 'CASES_DIR', root / 'cases'), (studies, 'STUDIES_DIR', root / 'studies')]
    targets += [(module, 'LEGACY_RUNS', root / 'runs') for module in (case_id, campaign, *more)]
    for module, name, path in targets:
        patcher = mock.patch.object(module, name, path)
        patcher.start()
        test.addCleanup(patcher.stop)
    return root


def fake_wall(argv, status):
    """What run_wall.run leaves: a new case folder in the patched research/cases with its receipt, which it
    returns. One minute per wall, so that a test never runs out of same-minute suffixes."""
    start = datetime(2026, 11, 8, 12, 0) + timedelta(minutes=next(MINUTES))
    wall, out = case_id.allocate(argv[0], start=start, hostname='GIAAIL-STRIX')
    receipt = {'case_id': wall, 'case': argv[0], 'status': status,
               'parent': argv[argv.index('--parent') + 1] if '--parent' in argv else None}
    (out / 'receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
    return receipt


class SeedStaging(unittest.TestCase):
    def test_only_the_seed_call_changes(self):
        for case, (script, _) in run_wall.CASES.items():
            source = (REPO / script).read_text(encoding='utf-8')
            policy = {'filter': {'policy': 'paper', 'order': (3, 0, 2, 1, 4)}, 'run_id': 'x'}
            default, seeded = run_wall.stage(source, policy), run_wall.stage(source, policy, seed=1234)
            changed = [l for l in difflib.unified_diff(default.splitlines(), seeded.splitlines(), lineterm='', n=0)
                       if l[:1] in '+-' and not l.startswith(('+++', '---'))]
            self.assertEqual(changed, ['-    seed_everything()', '+    seed_everything(1234)'], case)


class DifferingFiles(unittest.TestCase):
    def test_lists_changed_and_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            for run, extra in (('a', b'1'), ('b', b'2')):
                r = Path(tmp) / run / 'result_x_wall0'
                (r / 'img').mkdir(parents=True)
                (r / 'decisions.txt').write_bytes(b'same')
                (r / 'img' / 'time.txt').write_bytes(extra)
            (Path(tmp) / 'b' / 'result_x_wall0' / 'only_b.txt').write_bytes(b'')
            self.assertEqual(check_run.differing_files(Path(tmp) / 'a', Path(tmp) / 'b'), ['only_b.txt', 'img/time.txt'])


class BranchRunner(unittest.TestCase):
    BRANCH = {'name': 'b1', 'run': 'orig', 'seed': 7, 'target': 13, 'step': 10, 'kind': 'target',
              'pose': [13, 0, 1, 2], 'order': '0-1-2-3-4', 'recorded_order': '3-0-2-1-4'}

    def test_stops_at_first_failure_and_refuses_changed_code(self):
        from research.analysis import counterfactual_branches
        sandbox(self, counterfactual_branches)
        plan = {'id': 'p', 'case': 'thangavelu-noise80', 'timeout_seconds': 2400,
                'code_sha256': run_wall.code_fingerprint(), 'branches': [self.BRANCH, {**self.BRANCH, 'name': 'b2'}]}
        counterfactual_branches.register(plan)
        calls = []

        def launch(argv):
            calls.append(argv)
            return fake_wall(argv, 'TIMEOUT')
        self.assertFalse(counterfactual_branches.run_plan(plan, launch=launch))
        self.assertEqual(len(calls), 1)                              # stopped after the failed wall
        self.assertEqual(calls[0][calls[0].index('--parent') + 1], 'orig')
        self.assertEqual(calls[0][calls[0].index('--switch') + 1], '10=0,1,2,3,4')
        self.assertEqual(calls[0][calls[0].index('--order') + 1], '3,0,2,1,4')
        calls.clear()
        self.assertFalse(counterfactual_branches.run_plan(plan, launch=launch))  # b1 kept, b2 runs (and fails) next
        self.assertEqual(len(calls), 1)
        self.assertEqual([e['branch'] for e in studies.load('p')['cases']], ['b1', 'b2'])
        second = studies.find('p', branch='b2')[0]['case_id']
        self.assertEqual(counterfactual_branches.branch_wall(plan, plan['branches'][1]), second)
        self.assertEqual(len((studies.path_of('p').parent / 'runs.jsonl').read_text(encoding='utf-8').splitlines()), 2)
        with self.assertRaises(RuntimeError):
            counterfactual_branches.run_plan({**plan, 'code_sha256': {}}, launch=launch)

    def test_a_plan_of_the_archive_is_read_and_never_built_into(self):
        from research.analysis import counterfactual_branches as branches
        root = sandbox(self, branches)
        plan = {'id': 'old', 'case': 'thangavelu-noise80', 'timeout_seconds': 2400,
                'code_sha256': run_wall.code_fingerprint(), 'branches': [self.BRANCH]}
        (root / 'runs' / 'branches' / 'old').mkdir(parents=True)
        (root / 'runs' / 'branches' / 'old' / 'plan.json').write_text(json.dumps(plan), encoding='utf-8')
        self.assertEqual(branches.load_plan('old'), plan)
        self.assertEqual(branches.branch_wall(plan, self.BRANCH), 'branches/old/b1')
        with self.assertRaises(FileExistsError):
            branches.register(plan)
        with self.assertRaises(RuntimeError):
            branches.run_plan(plan, launch=lambda argv: self.fail('a wall was launched'))
        with self.assertRaises(RuntimeError):
            branches.summarize_plan(plan)


class OrderSwitch(unittest.TestCase):
    def tearDown(self):
        from Stonepacker2D.utils import constant
        constant.set_filter_policy()

    def test_switch_applies_only_at_its_step(self):
        from Stonepacker2D.utils import constant
        constant.set_filter_policy('paper', (3, 0, 2, 1, 4), 1e-9, switch={'14': [2, 4, 3, 1, 0]})
        base = constant.get_filter_policy()
        self.assertEqual(constant.get_filter_policy(14), {'policy': 'paper', 'order': (2, 4, 3, 1, 0), 'tol': 1e-9})
        self.assertIs(constant.get_filter_policy(13), base)
        self.assertIs(constant.get_filter_policy(None), base)
        constant.set_filter_policy('paper', (3, 0, 2, 1, 4))            # no switch: every step the same object
        self.assertIs(constant.get_filter_policy(14), constant.get_filter_policy())

    def test_parse_and_stage(self):
        self.assertEqual(run_wall.parse_switch(['14=2,4,3,1,0', '20=0,1,2,3,4']), {14: (2, 4, 3, 1, 0), 20: (0, 1, 2, 3, 4)})
        for bad in (['14=2,4,3,1,1'], ['14=2,4,3,1,0', '14=0,1,2,3,4']):
            with self.assertRaises(ValueError):
                run_wall.parse_switch(bad)
        source = (REPO / run_wall.CASES['thangavelu-noise80'][0]).read_text(encoding='utf-8')
        policy = {'policy': 'paper', 'order': (3, 0, 2, 1, 4), 'tol': 1e-9}
        plain = run_wall.stage(source, {'filter': policy, 'run_id': 'x'}, seed=5)
        branch = run_wall.stage(source, {'filter': {**policy, 'switch': {14: (2, 4, 3, 1, 0)}}, 'run_id': 'x'}, seed=5)
        changed = [l for l in difflib.unified_diff(plain.splitlines(), branch.splitlines(), lineterm='', n=0)
                   if l[:1] in '+-' and not l.startswith(('+++', '---'))]
        self.assertEqual(len(changed), 2)
        self.assertIn("'switch': {14: (2, 4, 3, 1, 0)}", changed[1])


def registered(**kw):
    args = dict(campaign_id='c', case='thangavelu-noise80', arm_texts=['code_original', 'paper=3,0,2,1,4',
                'alt=2,4,3,1,0', 'random'], n_seeds=3, master_seed=20261106, timeout=1200)
    args.update(kw)
    return campaign.make_campaign(**args)


class CampaignDesign(unittest.TestCase):
    def test_seeds_and_random_order_are_reproducible(self):
        a, b = registered(), registered()
        self.assertEqual((a['seeds'], a['arms']), (b['seeds'], b['arms']))
        self.assertEqual(a['seeds'][0], 20)
        self.assertEqual(len(set(a['seeds'])), 3)
        orders = [tuple(arm['order']) for arm in a['arms'] if arm['order']]
        self.assertEqual(len(set(orders)), 3)                      # random differs from paper and alt
        self.assertEqual([arm['policy'] for arm in a['arms']], ['code_original', 'paper', 'paper', 'paper'])
        self.assertNotEqual(registered(master_seed=1)['seeds'], a['seeds'])

    def test_rejects_bad_arms(self):
        with self.assertRaises(ValueError):
            registered(arm_texts=['x=1,2,3,4,4'])
        with self.assertRaises(ValueError):
            registered(arm_texts=['paper=3,0,2,1,4', 'paper=3,0,2,1,4'])
        with self.assertRaises(ValueError):
            registered(case='nope')


class CampaignRunner(unittest.TestCase):
    def setUp(self):
        self.root = sandbox(self)
        self.c = registered()
        campaign.register(self.c)
        self.calls = []

    def launcher(self, statuses):
        def launch(argv):
            self.calls.append(argv)
            status = statuses.get(len(self.calls), 'PROCESS_PASS')
            if status == 'TERMINATION_FAILED':
                raise run_wall.TerminationFailed(fake_wall(argv, status))
            return fake_wall(argv, status)
        return launch

    def extended_arms(self):
        random = next(a['order'] for a in self.c['arms'] if a['name'].startswith('random'))
        return ['code_original', 'paper=3,0,2,1,4', 'alt=2,4,3,1,0', 'random=' + ','.join(map(str, random)),
                'mainstream=4,1,0,3,2']

    def archived_campaign(self, name='old'):
        """A campaign in the temporary archive: campaign.json, manifest.csv, a passed wall per cell at its legacy path."""
        design = registered(campaign_id=name)
        folder = campaign.legacy_dir(name)
        folder.mkdir(parents=True)
        (folder / 'campaign.json').write_text(json.dumps(design), encoding='utf-8')
        (folder / 'manifest.csv').write_text('arm,seed,run\n', encoding='utf-8')
        for _, _, wall in campaign.planned_runs(design):
            case_id.wall_dir(wall).mkdir(parents=True)
            receipt = json.dumps({'status': 'PROCESS_PASS'})
            (case_id.wall_dir(wall) / 'receipt.json').write_text(receipt, encoding='utf-8')
        return design

    def archive_listing(self):
        return sorted(p.relative_to(self.root).as_posix() for p in (self.root / 'runs').rglob('*'))

    def test_runs_seed_major_with_the_registered_settings(self):
        campaign.run_campaign(self.c, launch=self.launcher({}))
        self.assertEqual(len(self.calls), 12)
        first = self.calls[:4]
        self.assertTrue(all(a[a.index('--seed') + 1] == '20' for a in first))
        self.assertEqual(first[0][:3], ['thangavelu-noise80', '--policy', 'code_original'])
        self.assertNotIn('--order', first[0])
        self.assertEqual(first[1][first[1].index('--order') + 1], '3,0,2,1,4')
        self.assertEqual(first[1][first[1].index('--timeout') + 1], '1200')
        entries = studies.load('c')['cases']
        self.assertEqual([(e['arm'], e['seed']) for e in entries[:2]], [('code_original', 20), ('paper_3-0-2-1-4', 20)])
        self.assertTrue(all(case_id.is_case_id(wall) for _, _, wall in campaign.planned_runs(self.c)))
        log = (studies.path_of('c').parent / 'runs.jsonl').read_text(encoding='utf-8').splitlines()
        self.assertEqual(len(log), 12)

    def test_resume_skips_existing_runs_and_never_overwrites(self):
        campaign.run_campaign(self.c, max_runs=2, launch=self.launcher({2: 'TIMEOUT'}))
        self.calls.clear()
        campaign.run_campaign(self.c, launch=self.launcher({}))
        self.assertEqual(len(self.calls), 10)          # the TIMEOUT run is kept, not rerun
        self.assertEqual(sum(a[a.index('--seed') + 1] == '20' for a in self.calls), 2)  # of 4 arms, 2 were built
        entries = studies.load('c')['cases']
        self.assertEqual(len(entries), 12)
        self.assertEqual(len({(e['arm'], e['seed']) for e in entries}), 12)
        kept = studies.find('c', arm='paper_3-0-2-1-4', seed=20)[0]['case_id']
        self.assertEqual(campaign.run_status(kept), 'TIMEOUT')

    def test_failed_stops_and_termination_failure_raises(self):
        campaign.run_campaign(self.c, launch=self.launcher({3: 'FAILED'}))
        self.assertEqual(len(self.calls), 3)
        self.calls.clear()                              # resume: next new run fails to stop
        listed = len(studies.load(self.c['id'])['cases'])
        with self.assertRaises(run_wall.TerminationFailed):
            campaign.run_campaign(self.c, launch=self.launcher({1: 'TERMINATION_FAILED'}))
        self.assertEqual(len(studies.load(self.c['id'])['cases']), listed + 1)  # a resume sees that wall's status

    def test_refuses_changed_code(self):
        changed = dict(self.c, code_sha256={'Stonepacker2D/stack/selection.py': 'old'})
        with self.assertRaises(RuntimeError):
            campaign.run_campaign(changed, launch=self.launcher({}))
        self.assertEqual(self.calls, [])

    def test_extension_reuses_parent_runs(self):
        campaign.run_campaign(self.c, launch=self.launcher({}))
        arms = self.extended_arms()
        child = registered(campaign_id='d', arm_texts=arms, n_seeds=5, extends='c')
        self.assertEqual(child['seeds'][:3], self.c['seeds'])
        self.assertEqual(len(child['reused_runs']), 12)
        campaign.register(child)
        self.calls.clear()
        campaign.run_campaign(child, launch=self.launcher({}))
        self.assertEqual(len(self.calls), 25 - 12)
        self.assertEqual(len(studies.load('d')['cases']), 13)          # only its own new walls are listed
        parents = {e['case_id'] for e in studies.load('c')['cases']}
        walls = [wall for _, _, wall in campaign.planned_runs(child)]
        self.assertEqual(sum(wall in parents for wall in walls), 12)
        self.assertEqual(len(set(walls)), 25)
        with self.assertRaises(ValueError):
            registered(campaign_id='e', arm_texts=arms, n_seeds=5, extends='c', timeout=2400)
        with self.assertRaises(ValueError):
            registered(campaign_id='e', arm_texts=arms, n_seeds=5, extends='c', master_seed=1)

    def test_extension_of_a_campaign_of_the_archive(self):
        self.archived_campaign()
        before = self.archive_listing()
        child = registered(campaign_id='d', arm_texts=self.extended_arms(), n_seeds=5, extends='old')
        self.assertEqual(len(child['reused_runs']), 12)
        campaign.register(child)
        campaign.run_campaign(child, launch=self.launcher({}))
        self.assertEqual(len(self.calls), 25 - 12)
        walls = [wall for _, _, wall in campaign.planned_runs(child)]
        self.assertEqual(sum(wall.startswith('campaigns/old/thangavelu-noise80/') for wall in walls), 12)
        self.assertEqual(sum(case_id.is_case_id(wall) for wall in walls), 13)
        archived = next(wall for wall in walls if wall.startswith('campaigns/'))
        self.assertEqual(campaign.run_status(archived), 'PROCESS_PASS')
        self.assertEqual(self.archive_listing(), before)               # read where they are, nothing copied or added

    def test_a_campaign_of_the_archive_is_only_read(self):
        old = self.archived_campaign()
        self.assertEqual(campaign.load('old'), old)
        with self.assertRaises(RuntimeError):
            campaign.run_campaign(old, launch=self.launcher({}))
        with self.assertRaises(RuntimeError):
            campaign.summarize(old)
        with self.assertRaises(FileExistsError):
            campaign.register(registered(campaign_id='old'))
        self.assertEqual(self.calls, [])

    def test_manifest_path_of_a_study_and_of_the_archive(self):
        self.assertEqual(campaign.manifest_path('c'), self.root / 'studies' / 'c' / 'manifest.csv')
        self.assertEqual(campaign.manifest_path('old'), self.root / 'runs' / 'campaigns' / 'old' / 'manifest.csv')

    def test_summarize_writes_beside_the_study_with_case_ids(self):
        campaign.run_campaign(self.c, max_runs=1, launch=self.launcher({}))
        rows = campaign.summarize(self.c)
        self.assertEqual(len(rows), 12)
        self.assertTrue(case_id.is_case_id(rows[0]['run']))
        self.assertEqual(rows[0]['status'], 'PROCESS_PASS')
        self.assertEqual((rows[1]['run'], rows[1]['status']), ('', 'NOT_RUN'))
        self.assertTrue(campaign.manifest_path('c').exists())
        self.assertTrue((studies.path_of('c').parent / 'steps.csv').exists())

    def test_dry_run_launches_nothing(self):
        planned = campaign.run_campaign(self.c, dry_run=True, launch=self.launcher({}))
        self.assertEqual((len(planned), self.calls), (12, []))
        self.assertEqual(studies.load('c')['cases'], [])


class MetricHelpers(unittest.TestCase):
    def test_first_divergence(self):
        row = lambda step, stone, x: {'step': step, 'outcome': 'placed', 'stone_id': stone, 'rotation_world': 0,
                                      'x_world': x, 'y': 2}
        ref = [row(1, 5, 1), row(2, 6, 9), row(3, 7, 20)]
        self.assertIsNone(wall_metrics.first_divergence(ref, ref))
        self.assertEqual(wall_metrics.first_divergence([row(1, 5, 1), row(2, 6, 10), row(3, 7, 20)], ref), 2)
        self.assertEqual(wall_metrics.first_divergence(ref[:2], ref), 3)

    def test_mechanism_lambdas(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'img').mkdir()
            (Path(tmp) / 'img' / 'left0.3225_mechanism.png').write_bytes(b'')
            (Path(tmp) / 'img' / 'right0.5_mechanism.png').write_bytes(b'')
            self.assertEqual(wall_metrics.mechanism_lambdas(tmp), {'left': 0.3225, 'right': 0.5})


class OrderAgreement(unittest.TestCase):
    def test_agreement_and_prefix(self):
        import pandas as pd
        from research.analysis import order_agreement
        choice = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [1, 2, 3, 5], 'c': [9, 9, 3, 4]})
        a = order_agreement.agreement(choice)
        self.assertEqual((a.at['a', 'b'], a.at['a', 'c'], a.at['b', 'c'], a.at['a', 'a']), (0.75, 0.5, 0.25, 1.0))
        self.assertEqual([order_agreement.shared_prefix('3-0-2-1-4', o) for o in ('0-3-2-1-4', '3-0-1-2-4', '3-0-2-4-1')],
                         [0, 2, 3])


class ArmComparison(unittest.TestCase):
    def manifest(self):
        import pandas as pd
        base = {20: 0.50, 21: 0.60, 22: 0.55, 23: 0.65}            # seed effect
        rows = []
        for seed, v in base.items():
            rows += [{'arm': 'code_original', 'seed': seed, 'wall_density': v, 'n_placed': 18},
                     {'arm': 'b', 'seed': seed, 'wall_density': v + 0.10, 'n_placed': 18},        # constant +0.10
                     {'arm': 'c', 'seed': seed, 'wall_density': v - 0.01 * (seed % 2), 'n_placed': 17 + seed % 2}]
        m = pd.DataFrame(rows).assign(n_input=18)
        return m.assign(wall_height=1 / m.wall_density, lambda_min=0.1, stone_filling=m.wall_density / 2)

    def test_paired_shares_and_ties(self):
        from research.analysis import arm_comparison
        out = arm_comparison.compare(self.manifest(), n_boot=200)
        p = out['wall_density']['paired'].set_index('arm')
        self.assertAlmostEqual(p.at['b', 'mean_diff'], 0.10)
        self.assertAlmostEqual(p.at['b', 'ci_low'], 0.10)               # no spread in a constant difference
        self.assertEqual((p.at['b', 'better_in'], p.at['b', 'worse_in']), (4, 0))
        self.assertEqual((p.at['c', 'better_in'], p.at['c', 'worse_in']), (0, 2))
        self.assertEqual(out['wall_height']['paired'].set_index('arm').at['b', 'better_in'], 4)  # lower is better
        s = out['wall_density']['shares']
        self.assertAlmostEqual(s['arm'] + s['seed'] + s['residual'], 1.0)
        self.assertEqual(out['wall_density']['best_per_seed'], {'b': 4})
        self.assertEqual(out['complete']['per_arm'].at['c', 'mean'], 0.5)
        self.assertEqual(out['complete']['best_per_seed'], {'tie': 4})  # never credited to the first arm
        self.assertTrue(np.isnan(out['lambda_min']['shares']['arm']))   # constant metric


class FailureAnalysis(unittest.TestCase):
    def test_wall_state(self):
        from research.analysis import unplaced_stones
        wall = np.zeros((10, 6))
        wall[:2], wall[-1], wall[:, [0, -1]] = 1, 255, 255     # ground, top boundary, side columns
        wall[2:5, 1] = wall[2:4, 2] = 7                       # columns 1-2 built 3 and 2 px high
        wall[4, 3] = 8                                        # column 3: overhang above a 2 px void
        s = unplaced_stones.wall_state(wall)
        self.assertEqual((s['h_max'], s['h_min'], s['h_range']), (3, 0, 3))
        self.assertEqual(s['void_under_skyline_px'], 2)
        self.assertAlmostEqual(s['fill_under_hmax'], 6 / 12)

    def test_eliminated_by(self):
        from research.analysis.unplaced_stones import eliminated_by
        order = (3, 0, 2, 1, 4)                               # paper order: gate, LLC, NHR, IWR, ...
        stages = [[0, 1, 2, 3], [0, 1, 2], [0, 1], [0]]
        self.assertEqual([eliminated_by(r, stages, order, 0) for r in ([0], [1], [2], [3], [4, 5])],
                         ['selected', 'IWR', 'NHR', 'LLC', 'infeasible'])
        fallback = [[0, 1, 2, 3], [0, 1, 2], [0, 1]]          # IWR stage emptied: argmax over [0, 1]
        self.assertEqual(eliminated_by([1], fallback, order, 0), 'final_argmax')


FAILED_PAPER = REPO / 'research' / 'runs' / 'campaigns' / 'pilot-thangavelu-4arms' / 'thangavelu-noise80' / 'paper_3-0-2-1-4' / 'seed490861164'


@unittest.skipUnless(FAILED_PAPER.exists(), 'existing run directories not present')
class FailureOnRecordedWall(unittest.TestCase):
    def test_unplaced_stone_has_positions_but_no_stable_one(self):
        import pandas as pd
        from research.analysis import unplaced_stones
        a, s = unplaced_stones.attempt_rows(check_run.result_dirs(FAILED_PAPER)[0])
        a, s = pd.DataFrame(a), pd.DataFrame(s)
        summary = unplaced_stones.wall_summary(a, s, hard=(8, 9, 13))
        self.assertEqual((summary['unplaced'], summary['n_no_placement']), ('13', 6))
        last = s[s.event == a.event.iloc[-1]].iloc[0]
        self.assertEqual((last.stone, last.geom_rotations, last.feasible_rotations, last.eliminated_by), (13, 4, 0, 'infeasible'))
        self.assertEqual((a.outcome == 'placed').sum(), 17)
        self.assertEqual(set(s[(s.stone == 13) & (a.set_index('event').loc[s.event, 'outcome'].to_numpy() == 'placed')]
                             .eliminated_by) - {'LLC', 'NHR', 'IWR', 'final_argmax'}, set())


FAILED_PAPER_2 = FAILED_PAPER.parent / 'seed184947496'


@unittest.skipUnless(FAILED_PAPER_2.exists(), 'existing run directories not present')
class CounterfactualPlanning(unittest.TestCase):
    def test_orders_that_pick_the_target_and_branch_check(self):
        from research.analysis import counterfactual_branches
        result = check_run.result_dirs(FAILED_PAPER_2)[0]
        options = counterfactual_branches.step_options(result, {13})
        recorded = options[options.order == options.recorded_order]
        self.assertEqual(len(options), 120 * len(recorded))
        self.assertTrue(recorded.same_as_recorded.all())
        s = counterfactual_branches.summarize(options).set_index('step')
        self.assertEqual(s.n_picks_target[s.n_picks_target > 0].to_dict(), {10: 5})
        self.assertEqual((s.n_picks_target + s.n_neutral + s.n_control == 119).all(), True)
        plan = counterfactual_branches.make_plan('p', [(str(FAILED_PAPER_2.relative_to(LEGACY_RUNS)), 184947496, 13)], 2400)
        got = [(b['kind'], b['step'], b['pose'], b['order']) for b in plan['branches']]
        self.assertEqual(got, [('target', 10, [13, 180, 204, 58], '3-2-0-1-4'), ('control', 10, [19, 0, 207, 57], '0-3-1-4-2'),
                               ('control', 10, [19, 90, 26, 43], '3-2-0-4-1'), ('control', 10, [19, 180, 204, 58], '3-4-0-1-2')])
        same = counterfactual_branches.check_branch(result, result, {10: (3, 0, 2, 1, 4)})
        self.assertEqual((same['ok'], same['identical_prefix_attempts']), (True, 9))
        self.assertFalse(counterfactual_branches.check_branch(result, result, {10: (0, 1, 2, 3, 4)})['ok'])


@unittest.skipUnless(THANGAVELU.exists() and PR_RIGHT.exists(), 'existing run directories not present')
class ExistingRuns(unittest.TestCase):
    def test_checks_pass_on_recorded_walls(self):
        for run in (THANGAVELU, PR_RIGHT):
            report = check_run.check_run(run)
            self.assertTrue(report['ok'], report['problems'][:3])
        stats = check_run.check_run(PR_RIGHT)['results'][0]['stats']
        self.assertEqual(stats['right/unrotated/same'], 2)       # real right-side placements
        self.assertGreater(stats['right candidates checked'], 0)

    def test_compare_identical(self):
        self.assertEqual(check_run.compare_runs(THANGAVELU, THANGAVELU), [])

    def test_wall_metrics_relations(self):
        result = check_run.result_dirs(THANGAVELU)[0]
        m = wall_metrics.wall_metrics(THANGAVELU, result)
        self.assertEqual((m['n_input'], m['n_placed'], m['termination']), (18, 18, 'inventory_empty'))
        self.assertLess(m['stone_filling'], m['fill_wang_code'])  # ground and boundaries count only in the latter
        self.assertGreater(m['wall_density'], m['stone_filling'])  # built part only
        self.assertEqual(m['lambda_min'], min(m['lambda_left'], m['lambda_right']))
        self.assertAlmostEqual(m['lambda_wang_code'], round(m['lambda_min'], 1))
        rows = wall_metrics.step_rows(result)
        self.assertEqual(len(rows), 18)
        self.assertTrue(all(r['placement_id'] and r['n_stones'] >= 1 for r in rows))


if __name__ == '__main__':
    unittest.main()
