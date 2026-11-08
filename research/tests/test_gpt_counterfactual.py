"""research/gpt_counterfactual.py on the recorded walls (no wall is built): the plan's switches,
the branch check on identities and on walls that must fail it, and the run refusals."""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'research')]
import gpt_counterfactual as cf  # noqa: E402

PRESENT = (cf.run_case.RUNS / cf.SOURCE).exists() and (cf.run_case.RUNS / cf.PAPER_WALL).exists()
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
            cf.run_plan(self.plan, launch=launch)
        self.assertIn('no key', str(refused.exception))
        env = {k: v for k, v in os.environ.items() if k not in cf.KEY_VARS}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(SystemExit):
                cf.run_plan({**self.plan, 'code_sha256': {}}, launch=launch)
            missing = {**self.plan, 'walls': {**self.plan['walls'], 'V': {**self.plan['walls']['V'], 'run': 'branches/never/V'}}}
            with self.assertRaises(SystemExit) as refused:
                cf.run_plan(missing, only=['G6'], launch=launch)
            self.assertIn('V has not passed', str(refused.exception))
        launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
