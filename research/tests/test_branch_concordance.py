"""research/analysis/branch_concordance.py on a hand-made block: tree, labels, concordance, test and Holm."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from research.analysis import branch_concordance as bc


def block(folder, seed, flip=False):
    """Three walls: a and b share the first placement and branch at the second; c branches at the first.
    Wall quality F_SF a > b > c; H2b follows it (or the reverse with flip)."""
    walls = pd.DataFrame(dict(wall=["a", "b", "c"], wall_type="TH_T80", seed=seed, filter_policy=["1", "2", "3"],
                              status="PROCESS_PASS", unique_id=[f"{seed}a", f"{seed}b", f"{seed}c"],
                              F_SF=[3.0, 2.0, 1.0], F_AH=[1.0, 1.0, 1.0], F_AV=[np.nan, 1.0, 2.0], F_LR=1.0))
    pose = {"a": [(1, 0), (2, 0)], "b": [(1, 0), (3, 0)], "c": [(4, 0), (2, 0)]}
    steps = pd.DataFrame([dict(wall=w, event=e, outcome="placed", stone=s, rot=r, x=0, y=0, c_GDS=float(s),
                               c_NHR=0.0, c_IWR=0.0, c_LLC=1.0, c_LFR=0.0)
                          for w in "abc" for e, (s, r) in enumerate(pose[w])])
    good = {"a": 3.0, "b": 2.0, "c": 1.0}
    first = {"a": 2.5, "b": 2.5, "c": 1.0}  # one value per child: a and b share theirs
    h = pd.DataFrame([dict(wall=w, event=e, H2b=(-1 if flip else 1) * (good[w] if e else first[w]), H0_unlevel=0.0)
                      for w in "abc" for e in range(2)])
    folder.mkdir()
    Path(f"{folder}-h").mkdir()
    walls.to_csv(folder / "walls.csv", index=False)
    steps.to_csv(folder / "steps.csv", index=False)
    h.to_csv(Path(f"{folder}-h") / "h.csv", index=False)
    return folder


class Concordance(unittest.TestCase):
    def test_tree_labels_and_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            folders = [block(Path(tmp, "s1"), 1), block(Path(tmp, "s2"), 2, flip=True)]
            table, blocks = bc.concordance(folders, ["H2b"], ["c_GDS", "H0_unlevel"])
        self.assertEqual(blocks.branch_points.tolist(), [2, 2])  # the root and the node after (1, 0)
        row = table.set_index(["score", "metric"])
        # F_SF: root pair {a,b} (mean 2.5) vs {c} (1), node pair a (3) vs b (2); both blocks: 2 comparable each
        self.assertEqual((row.loc[("H2b", "F_SF"), "comparable"], row.loc[("H2b", "F_SF"), "concordant"]), (4, 2))
        self.assertEqual(row.loc[("H2b", "F_SF"), "statistic"], 0)  # +2 and -2 cancel
        self.assertEqual(row.loc[("H2b", "F_AH"), "comparable"], 0)  # equal labels never compare
        # F_AV: a has none, so the root compares {b} (z of 1) with {c} (z of 2) and the node a with b is skipped
        self.assertEqual(row.loc[("H2b", "F_AV"), "comparable"], 2)
        self.assertEqual(row.loc[("H0_unlevel", "F_SF"), "comparable"], 0)  # equal scores never compare
        self.assertTrue(np.isnan(row.loc[("c_GDS", "F_SF"), "p"]))  # baselines are not tested

    def test_a_score_with_two_values_is_left_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = block(Path(tmp, "s1"), 1)
            steps = pd.read_csv(folder / "steps.csv")
            steps.loc[(steps.wall == "b") & (steps.event == 0), "c_GDS"] = 9.0  # a and b share this child
            steps.to_csv(folder / "steps.csv", index=False)
            table, blocks = bc.concordance([folder], ["c_GDS", "c_SUM"], [])
        self.assertEqual(blocks.n_score_mismatch.tolist(), [1])
        row = table.set_index(["score", "metric"])
        # the root pair {a,b} vs {c} is gone; a (c_GDS 2) vs b (3) is left, and b is worse on F_SF
        self.assertEqual((row.loc[("c_GDS", "F_SF"), "comparable"], row.loc[("c_GDS", "F_SF"), "concordant"]), (1, 0))
        self.assertEqual(row.loc[("c_SUM", "F_SF"), "comparable"], 1)

    def test_sign_flip_and_holm(self):
        self.assertAlmostEqual(bc.sign_flip_p(np.zeros(5)), 1.0)
        self.assertLess(bc.sign_flip_p(np.full(15, 3.0)), 0.001)  # 2 of 32768 patterns reach |45|
        np.testing.assert_allclose(bc.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


if __name__ == "__main__":
    unittest.main()
