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
