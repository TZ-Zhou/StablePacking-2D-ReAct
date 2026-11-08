"""research/llm_walls/gpt_counterfactual.py on the recorded walls (no wall is built): the plan's switches,
the branch check on identities and on walls that must fail it, and the run refusals."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO)]
from research.llm_walls import gpt_counterfactual as cf  # noqa: E402
from research.paths import LEGACY_RUNS  # noqa: E402
from research.storage import studies  # noqa: E402
from research.runner import gpt_counterfactual as cf_runner  # noqa: E402
from research.walls import case_definition  # noqa: E402

PRESENT = (LEGACY_RUNS / cf.SOURCE).exists() and (LEGACY_RUNS / cf.PAPER_WALL).exists()
GPT_STEPS = [6, 7, 8, 9, 10, 11, 14, 17, 20]


@unittest.skipUnless(PRESENT, 'gpt-001 or the dev Paper wall not present')
class Plan(unittest.TestCase):
    def setUp(self):
        self.gpt, self.paper = cf.result_of(cf.SOURCE), cf.result_of(cf.PAPER_WALL)
        self.plan = cf.make_plan()
        for w in self.plan['walls'].values():
            w['switch'] = {int(s): tuple(o) for s, o in w['switch'].items()}

    def test_the_switches_are_gpts_recorded_orders(self):
        orders = cf.recorded_orders(self.gpt)
        self.assertEqual(sorted(orders), GPT_STEPS)
        self.assertEqual(orders[6], (0, 1, 3, 4, 2))
        self.assertEqual(cf.recorded_orders(self.paper), {})  # the Paper wall: no switch
        w = self.plan['walls']
        self.assertEqual({n: sorted(x['switch']) for n, x in w.items()},
                         {'V': GPT_STEPS, 'G6': [6], 'P6': [], 'G8': [6, 7, 8], 'P8': [6, 7],
                          'G17': [6, 7, 8, 9, 10, 11, 14, 17], 'P17': [6, 7, 8, 9, 10, 11, 14]})
        self.assertEqual(self.plan['build_order'], ['V', 'G6', 'G8', 'P8', 'G17', 'P17'])  # 6 walls, P6 reused
        self.assertEqual(w['P6']['run'], cf.PAPER_WALL)

    def test_identities_pass(self):
        self.assertEqual(cf.check_branch(self.gpt, self.gpt, self.plan['walls']['V']), [])
        self.assertEqual(cf.check_branch(self.gpt, self.paper, self.plan['walls']['P6']), [])  # the reuse of P6

    def test_walls_that_are_not_the_branch_fail(self):
        w = self.plan['walls']
        self.assertTrue(cf.check_branch(self.gpt, self.paper, w['V']))    # the Paper wall is not GPT's
        self.assertTrue(cf.check_branch(self.gpt, self.paper, w['G6']))   # Paper did not take GPT's step 6
        self.assertTrue(cf.check_branch(self.gpt, self.paper, w['P8']))   # its steps 6-7 are not GPT's
        self.assertTrue(cf.check_branch(self.gpt, self.gpt, w['P8']))     # GPT took its own candidate at 8
        self.assertTrue(any('traced order' in p for p in cf.check_branch(self.gpt, self.gpt, w['G8'])))  # orders after 8

    def test_run_refusals(self):
        launch = mock.Mock()
        with mock.patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-x'}), self.assertRaises(SystemExit) as refused:
            cf_runner.run_plan(self.plan, launch=launch)
        self.assertIn('no key', str(refused.exception))
        env = {k: v for k, v in os.environ.items() if k not in cf.KEY_VARS}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(SystemExit):
                cf_runner.run_plan({**self.plan, 'code_sha256': {}}, launch=launch)
            missing = {**self.plan, 'walls': {**self.plan['walls'], 'V': {**self.plan['walls']['V'], 'run': 'branches/never/V'}}}
            with self.assertRaises(SystemExit) as refused:
                cf_runner.run_plan(missing, only=['G6'], launch=launch)
            self.assertIn('V has not passed', str(refused.exception))
        launch.assert_not_called()


class Build(unittest.TestCase):
    """How run builds a wall (launch and the wall check are mocked; no wall is built)."""
    CASE = '20261108-1254-TH_T80-pc1'

    def plan(self):
        wall = {'arm': 'V', 'k': None, 'switch': {6: (0, 1, 3, 4, 2)}, 'run': None}
        return {'id': 'cf-test', 'source': cf.SOURCE, 'seed': 7, 'case': 'thangavelu-noise80', 'timeout_seconds': 60,
                'code_sha256': case_definition.code_fingerprint(), 'walls': {'V': wall}, 'build_order': ['V']}

    def setUp(self):
        env = {k: v for k, v in os.environ.items() if k not in cf.KEY_VARS}
        for patcher in (mock.patch.dict(os.environ, env, clear=True), mock.patch.object(cf, 'check_wall', return_value=[])):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.launch = mock.Mock(return_value={'case_id': self.CASE, 'status': 'PROCESS_PASS'})

    def test_a_wall_is_a_case_of_the_study_with_the_gpt_wall_as_parent(self):
        plan = self.plan()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(studies, 'STUDIES_DIR', Path(tmp)):
            studies.create(plan['id'], 'gpt-counterfactual', plan)
            self.assertIsNone(cf.wall_ref(plan, 'V'))
            cf_runner.run_plan(plan, launch=self.launch)
            self.assertEqual(cf.wall_ref(plan, 'V'), self.CASE)
            self.assertEqual(studies.find(plan['id'], branch='V')[0]['case_id'], self.CASE)
        argv = self.launch.call_args.args[0]
        self.assertEqual(argv[:3], ['thangavelu-noise80', '--parent', cf.SOURCE])
        self.assertEqual(argv[-2:], ['--switch', '6=0,1,3,4,2'])

    def test_a_plan_without_a_study_builds_nothing(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(studies, 'STUDIES_DIR', Path(tmp)):
            with self.assertRaises(RuntimeError):
                cf_runner.run_plan(self.plan(), launch=self.launch)
        self.launch.assert_not_called()

    def test_a_wall_of_the_archive_keeps_its_path(self):
        plan = self.plan()
        plan['walls']['V']['run'] = 'branches/gpt-001-cf/V'
        self.assertEqual(cf.wall_ref(plan, 'V'), 'branches/gpt-001-cf/V')


if __name__ == '__main__':
    unittest.main()
