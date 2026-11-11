"""The post-hoc description of the pick-4 repeats against walls without spread (research/analysis/
llm_repeats_against.py)."""
import math
import unittest

import pandas as pd

from research.analysis import llm_repeats_against as against


class Against(unittest.TestCase):
    def test_a_difference_is_read_in_standard_errors_of_the_within_block_spread(self):
        zero = dict.fromkeys(against.COLS[1:-1], 0.0)
        walls = pd.DataFrame({"seed": [1, 1, 2, 2], "scene": "TH_T80", "round": [1, 2, 1, 2], "complete": True,
                              "mean_z": [1.0, 3.0, 0.0, 0.0], "n_placed": [18, 18, 18, 17], **zero})
        other = pd.DataFrame({"seed": [1, 2], "scene": "TH_T80", "complete": True, "mean_z": [1.0, 0.0],
                              "n_placed": [18, 18], **zero})
        d = against.differences(walls, other)
        self.assertEqual((list(d.mean_z), list(d.n_placed)), ([0.0, 2.0, 0.0, 0.0], [0, 0, 0, -1]))
        self.assertEqual(list(d.outcome), ["tie", "win", "tie", "loss"])
        row = against.summary(d).set_index("quantity").loc["mean_z"]  # block means 1 and 0, variances 2 and 0
        self.assertEqual((row.mean_diff, row.sd_within, row.se_within, row.in_se_within, row.said),
                         (0.5, 1.0, 0.5, 1.0, "inside the spread"))
        self.assertEqual((round(row.sd_blocks, 12), round(row.in_se_blocks, 12)), (round(math.sqrt(0.5), 12), 1.0))
        flat = against.summary(d).set_index("quantity").loc["diff_z_F_LR"]
        self.assertEqual((flat.said, math.isnan(flat.in_se_within)), ("no spread", True))
        block = against.per_block(d).set_index("seed").loc[1]
        self.assertEqual((block.win, block.tie, block.loss, block.mean_z, block.mean_z_in_se), (1, 1, 0, 1.0, 2 ** 0.5))
        self.assertEqual(against.in_se(1.0, 0.5), (2.0, "outside the spread"))


if __name__ == "__main__":
    unittest.main()
