"""research/analysis/llm_endgame_lookahead.py: the rule of docs/plans/llm-endgame-lookahead.md."""

import unittest

import pandas as pd

from research.analysis import llm_endgame_lookahead as la


class Rule(unittest.TestCase):
    def table(self, llm, n=18, cascade_n=18, safe=0.2, safe_n=18):
        return pd.DataFrame({"llm_mean_z": llm, "cascade_mean_z": 0.0, "low_top_mean_z": 0.2, "random_median": -0.3,
                             "llm_n_placed": n, "cascade_n_placed": cascade_n, "complete": True, "switches": 2,
                             "llm_decisions": 7, "low_top_safe_mean_z": safe, "low_top_safe_n_placed": safe_n})

    def test_rule(self):
        r = la.reading
        self.assertEqual(r(self.table([0.5] * 9)), "go")
        self.assertEqual(r(self.table([-0.5] * 5 + [0.5] * 4)), "stop direct pick")  # step 3's stop comes first
        # (d): 1 losing state against the cascade's 0 (above the stop's 3, so no stop)
        self.assertEqual(r(self.table([0.5] * 9, n=[17] + [18] * 8)), "no endgame contribution shown")
        self.assertEqual(r(self.table([0.5] * 9, n=[17] + [18] * 8, cascade_n=[17] + [18] * 8)), "go")
        # (e): not below low_top_safe in 5 states, with no fewer stones
        self.assertEqual(r(self.table([0.5] * 9, safe=[0.6] * 5 + [0.4] * 4)), "no endgame contribution shown")
        self.assertEqual(r(self.table([0.5] * 9, safe_n=[18] * 4 + [19] * 5)), "no endgame contribution shown")
        self.assertEqual(r(self.table([0.5] * 9, safe=[0.6] * 4 + [0.5] * 5)), "go")  # a tie counts


if __name__ == "__main__":
    unittest.main()
