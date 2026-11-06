"""Checks for run_case --seed, campaign.py, metrics.py and check_case.py.

Run from the repository root: python -B -m unittest discover -s research/tests
Tests marked 'existing run' use traced runs under research/runs (git-ignored) and are skipped
when those directories are absent.
"""
import difflib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'research'))
import campaign  # noqa: E402
import check_case  # noqa: E402
import metrics  # noqa: E402
import run_case  # noqa: E402

THANGAVELU = REPO / 'research' / 'runs' / 'thangavelu-noise80-code-original-full-01'
PR_RIGHT = REPO / 'research' / 'runs' / 'validation-pr-paper-right-01'


class SeedStaging(unittest.TestCase):
    def test_only_the_seed_call_changes(self):
        for case, (script, _) in run_case.CASES.items():
            source = (REPO / script).read_text(encoding='utf-8')
            policy = {'filter': {'policy': 'paper', 'order': (3, 0, 2, 1, 4)}, 'run_id': 'x'}
            default, seeded = run_case.stage(source, policy), run_case.stage(source, policy, seed=1234)
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
            self.assertEqual(check_case.differing_files(Path(tmp) / 'a', Path(tmp) / 'b'), ['only_b.txt', 'img/time.txt'])


class BranchRunner(unittest.TestCase):
    def test_stops_at_first_failure_and_refuses_changed_code(self):
        import counterfactual
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(run_case, 'RUNS', Path(tmp)), \
                mock.patch.object(counterfactual, 'PLANS', Path(tmp) / 'branches'):
            branch = {'name': 'b1', 'run': 'orig', 'seed': 7, 'target': 13, 'step': 10, 'kind': 'target',
                      'pose': [13, 0, 1, 2], 'order': '0-1-2-3-4', 'recorded_order': '3-0-2-1-4'}
            plan = {'id': 'p', 'case': 'thangavelu-noise80', 'timeout_seconds': 2400, 'code_sha256': run_case.code_fingerprint(),
                    'branches': [branch, {**branch, 'name': 'b2'}]}
            (Path(tmp) / 'branches' / 'p').mkdir(parents=True)
            calls = []

            def launch(argv):
                calls.append(argv)
                out = run_case.RUNS / argv[1]
                out.mkdir(parents=True)
                (out / 'receipt.json').write_text(json.dumps({'status': 'TIMEOUT'}))
            self.assertFalse(counterfactual.run_plan(plan, launch=launch))
            self.assertEqual(len(calls), 1)                              # stopped after the failed wall
            self.assertEqual(calls[0][calls[0].index('--switch') + 1], '10=0,1,2,3,4')
            self.assertEqual(calls[0][calls[0].index('--order') + 1], '3,0,2,1,4')
            calls.clear()
            self.assertFalse(counterfactual.run_plan(plan, launch=launch))  # b1 kept, b2 runs (and fails) next
            self.assertEqual([c[1] for c in calls], ['branches/p/b2'])
            with self.assertRaises(RuntimeError):
                counterfactual.run_plan({**plan, 'code_sha256': {}}, launch=launch)


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
        self.assertEqual(run_case.parse_switch(['14=2,4,3,1,0', '20=0,1,2,3,4']), {14: (2, 4, 3, 1, 0), 20: (0, 1, 2, 3, 4)})
        for bad in (['14=2,4,3,1,1'], ['14=2,4,3,1,0', '14=0,1,2,3,4']):
            with self.assertRaises(ValueError):
                run_case.parse_switch(bad)
        source = (REPO / run_case.CASES['thangavelu-noise80'][0]).read_text(encoding='utf-8')
        policy = {'policy': 'paper', 'order': (3, 0, 2, 1, 4), 'tol': 1e-9}
        plain = run_case.stage(source, {'filter': policy, 'run_id': 'x'}, seed=5)
        branch = run_case.stage(source, {'filter': {**policy, 'switch': {14: (2, 4, 3, 1, 0)}}, 'run_id': 'x'}, seed=5)
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
        self.tmp = tempfile.TemporaryDirectory()
        runs = Path(self.tmp.name)
        self.patches = [mock.patch.object(run_case, 'RUNS', runs), mock.patch.object(campaign, 'CAMPAIGNS', runs / 'campaigns')]
        for p in self.patches:
            p.start()
        self.c = registered()
        (runs / 'campaigns' / 'c').mkdir(parents=True)
        self.calls = []

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def launcher(self, statuses):
        def launch(argv):
            self.calls.append(argv)
            out = run_case.RUNS / argv[1]
            out.mkdir(parents=True)
            status = statuses.get(len(self.calls), 'PROCESS_PASS')
            if status == 'TERMINATION_FAILED':
                raise RuntimeError('not verified stopped')
            (out / 'receipt.json').write_text(json.dumps({'status': status}))
        return launch

    def test_runs_seed_major_with_the_registered_settings(self):
        campaign.run_campaign(self.c, launch=self.launcher({}))
        self.assertEqual(len(self.calls), 12)
        first = self.calls[:4]
        self.assertTrue(all(a[a.index('--seed') + 1] == '20' for a in first))
        self.assertEqual(first[0][:4], ['thangavelu-noise80', 'campaigns/c/thangavelu-noise80/code_original/seed20',
                                        '--policy', 'code_original'])
        self.assertNotIn('--order', first[0])
        self.assertEqual(first[1][first[1].index('--order') + 1], '3,0,2,1,4')
        self.assertEqual(first[1][first[1].index('--timeout') + 1], '1200')
        log = (campaign.CAMPAIGNS / 'c' / 'runs.jsonl').read_text().splitlines()
        self.assertEqual(len(log), 12)

    def test_resume_skips_existing_runs_and_never_overwrites(self):
        campaign.run_campaign(self.c, max_runs=2, launch=self.launcher({2: 'TIMEOUT'}))
        self.calls.clear()
        campaign.run_campaign(self.c, launch=self.launcher({}))
        self.assertEqual(len(self.calls), 10)          # the TIMEOUT run is kept, not rerun
        self.assertNotIn('campaigns/c/thangavelu-noise80/code_original/seed20', [a[1] for a in self.calls])

    def test_failed_stops_and_termination_failure_raises(self):
        campaign.run_campaign(self.c, launch=self.launcher({3: 'FAILED'}))
        self.assertEqual(len(self.calls), 3)
        self.calls.clear()                              # resume: next new run fails to stop
        with self.assertRaises(RuntimeError):
            campaign.run_campaign(self.c, launch=self.launcher({1: 'TERMINATION_FAILED'}))

    def test_refuses_changed_code(self):
        changed = dict(self.c, code_sha256={'Stonepacker2D/stack/selection.py': 'old'})
        with self.assertRaises(RuntimeError):
            campaign.run_campaign(changed, launch=self.launcher({}))
        self.assertEqual(self.calls, [])

    def test_extension_reuses_parent_runs(self):
        (campaign.CAMPAIGNS / 'c' / 'campaign.json').write_text(json.dumps(self.c))
        campaign.run_campaign(self.c, launch=self.launcher({}))
        random = next(a['order'] for a in self.c['arms'] if a['name'].startswith('random'))
        arms = ['code_original', 'paper=3,0,2,1,4', 'alt=2,4,3,1,0', 'random=' + ','.join(map(str, random)),
                'mainstream=4,1,0,3,2']
        child = registered(campaign_id='d', arm_texts=arms, n_seeds=5, extends='c')
        self.assertEqual(child['seeds'][:3], self.c['seeds'])
        self.assertEqual(len(child['reused_runs']), 12)
        (campaign.CAMPAIGNS / 'd').mkdir()
        self.calls.clear()
        campaign.run_campaign(child, launch=self.launcher({}))
        self.assertEqual(len(self.calls), 25 - 12)
        self.assertTrue(all(a[1].startswith('campaigns/d/') for a in self.calls))
        self.assertEqual(sum(n.startswith('campaigns/c/') for _, _, n in campaign.planned_runs(child)), 12)
        with self.assertRaises(ValueError):
            registered(campaign_id='e', arm_texts=arms, n_seeds=5, extends='c', timeout=2400)
        with self.assertRaises(ValueError):
            registered(campaign_id='e', arm_texts=arms, n_seeds=5, extends='c', master_seed=1)

    def test_dry_run_launches_nothing(self):
        planned = campaign.run_campaign(self.c, dry_run=True, launch=self.launcher({}))
        self.assertEqual((len(planned), self.calls), (12, []))


class MetricHelpers(unittest.TestCase):
    def test_first_divergence(self):
        row = lambda step, stone, x: {'step': step, 'outcome': 'placed', 'stone_id': stone, 'rotation_world': 0,
                                      'x_world': x, 'y': 2}
        ref = [row(1, 5, 1), row(2, 6, 9), row(3, 7, 20)]
        self.assertIsNone(metrics.first_divergence(ref, ref))
        self.assertEqual(metrics.first_divergence([row(1, 5, 1), row(2, 6, 10), row(3, 7, 20)], ref), 2)
        self.assertEqual(metrics.first_divergence(ref[:2], ref), 3)

    def test_mechanism_lambdas(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'img').mkdir()
            (Path(tmp) / 'img' / 'left0.3225_mechanism.png').write_bytes(b'')
            (Path(tmp) / 'img' / 'right0.5_mechanism.png').write_bytes(b'')
            self.assertEqual(metrics.mechanism_lambdas(tmp), {'left': 0.3225, 'right': 0.5})


class OrderAgreement(unittest.TestCase):
    def test_agreement_and_prefix(self):
        import pandas as pd
        import order_agreement
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
        import arm_comparison
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
        import failure_analysis
        wall = np.zeros((10, 6))
        wall[:2], wall[-1], wall[:, [0, -1]] = 1, 255, 255     # ground, top boundary, side columns
        wall[2:5, 1] = wall[2:4, 2] = 7                       # columns 1-2 built 3 and 2 px high
        wall[4, 3] = 8                                        # column 3: overhang above a 2 px void
        s = failure_analysis.wall_state(wall)
        self.assertEqual((s['h_max'], s['h_min'], s['h_range']), (3, 0, 3))
        self.assertEqual(s['void_under_skyline_px'], 2)
        self.assertAlmostEqual(s['fill_under_hmax'], 6 / 12)

    def test_eliminated_by(self):
        from failure_analysis import eliminated_by
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
        import failure_analysis
        a, s = failure_analysis.attempt_rows(check_case.result_dirs(FAILED_PAPER)[0])
        a, s = pd.DataFrame(a), pd.DataFrame(s)
        summary = failure_analysis.wall_summary(a, s, hard=(8, 9, 13))
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
        import counterfactual
        result = check_case.result_dirs(FAILED_PAPER_2)[0]
        options = counterfactual.step_options(result, {13})
        recorded = options[options.order == options.recorded_order]
        self.assertEqual(len(options), 120 * len(recorded))
        self.assertTrue(recorded.same_as_recorded.all())
        s = counterfactual.summarize(options).set_index('step')
        self.assertEqual(s.n_picks_target[s.n_picks_target > 0].to_dict(), {10: 5})
        self.assertEqual((s.n_picks_target + s.n_neutral + s.n_control == 119).all(), True)
        plan = counterfactual.make_plan('p', [(str(FAILED_PAPER_2.relative_to(run_case.RUNS)), 184947496, 13)], 2400)
        got = [(b['kind'], b['step'], b['pose'], b['order']) for b in plan['branches']]
        self.assertEqual(got, [('target', 10, [13, 180, 204, 58], '3-2-0-1-4'), ('control', 10, [19, 0, 207, 57], '0-3-1-4-2'),
                               ('control', 10, [19, 90, 26, 43], '3-2-0-4-1'), ('control', 10, [19, 180, 204, 58], '3-4-0-1-2')])
        same = counterfactual.check_branch(result, result, {10: (3, 0, 2, 1, 4)})
        self.assertEqual((same['ok'], same['identical_prefix_attempts']), (True, 9))
        self.assertFalse(counterfactual.check_branch(result, result, {10: (0, 1, 2, 3, 4)})['ok'])


@unittest.skipUnless(THANGAVELU.exists() and PR_RIGHT.exists(), 'existing run directories not present')
class ExistingRuns(unittest.TestCase):
    def test_checks_pass_on_recorded_walls(self):
        for run in (THANGAVELU, PR_RIGHT):
            report = check_case.check_run(run)
            self.assertTrue(report['ok'], report['problems'][:3])
        stats = check_case.check_run(PR_RIGHT)['results'][0]['stats']
        self.assertEqual(stats['right/unrotated/same'], 2)       # real right-side placements
        self.assertGreater(stats['right candidates checked'], 0)

    def test_compare_identical(self):
        self.assertEqual(check_case.compare_runs(THANGAVELU, THANGAVELU), [])

    def test_wall_metrics_relations(self):
        result = check_case.result_dirs(THANGAVELU)[0]
        m = metrics.wall_metrics(THANGAVELU, result)
        self.assertEqual((m['n_input'], m['n_placed'], m['termination']), (18, 18, 'inventory_empty'))
        self.assertLess(m['stone_filling'], m['fill_wang_code'])  # ground and boundaries count only in the latter
        self.assertGreater(m['wall_density'], m['stone_filling'])  # built part only
        self.assertEqual(m['lambda_min'], min(m['lambda_left'], m['lambda_right']))
        self.assertAlmostEqual(m['lambda_wang_code'], round(m['lambda_min'], 1))
        rows = metrics.step_rows(result)
        self.assertEqual(len(rows), 18)
        self.assertTrue(all(r['placement_id'] and r['n_stones'] >= 1 for r in rows))


if __name__ == '__main__':
    unittest.main()
