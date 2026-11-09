"""research/analysis/b3_ceiling.py on a synthetic block: signs, Paper's position and the block check."""

import tempfile
import unittest
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

from research.analysis import b3_ceiling


def block(seed=1):
    orders = ["".join(map(str, p)) for p in permutations(range(5))]
    rank = np.arange(len(orders), dtype=float)
    return pd.DataFrame(dict(wall_type="TH_T80", seed=seed, filter_policy=orders, unique_id=orders,
                             F_SF=rank, F_AH=rank, F_AV=rank, F_LR=rank, wall_height=rank))


class Ceiling(unittest.TestCase):
    def test_signs_follow_the_better_direction(self):
        rows = {r["metric"]: r for r in b3_ceiling.block_rows(block())}
        paper = block().filter_policy.tolist().index(b3_ceiling.PAPER)
        self.assertEqual(rows["F_SF"]["best_order"], "43210")  # largest value
        self.assertEqual(rows["F_AH"]["best_order"], "01234")  # smallest value
        self.assertAlmostEqual(rows["F_SF"]["paper_pct_beaten"], (119 - paper) / 120)
        self.assertAlmostEqual(rows["F_AH"]["paper_pct_beaten"], paper / 120)
        for r in rows.values():
            self.assertGreater(r["best_vs_paper_sd"], 0)
            self.assertAlmostEqual(r["best_vs_median_sd"], r["best_vs_paper_sd"] + r["paper_vs_median_sd"])

    def test_a_block_must_hold_every_order_once(self):
        with self.assertRaises(ValueError):
            b3_ceiling.block_rows(block().iloc[1:])

    def test_blocks_are_seed_x_scene_across_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            for seed in (1, 2):
                Path(tmp, str(seed)).mkdir()
                block(seed).to_csv(Path(tmp, str(seed), "walls.csv"), index=False)
            table = b3_ceiling.ceiling_table([Path(tmp, "1"), Path(tmp, "2")])
        self.assertEqual(sorted(table.groupby(["scene", "seed"]).size()), [5, 5])

if __name__ == "__main__":
    unittest.main()
