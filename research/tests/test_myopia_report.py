"""research/analysis/myopia_report.py on small hand-made inputs: concordance, branch points, variance shares,
within-seed demeaning, the divergence search and the other pure statistics."""

import unittest

import numpy as np
import pandas as pd

from research.analysis import myopia_report as report


def point(d_llc, f_lr=0.0, f_height=0.0):
    return {"d_LLC": d_llc, "f_F_LR": f_lr, "f_wall_height": f_height}


def scored(step_scores, pose):
    """A placement row with the given (LLC, NHR, IWR, GDS, LFR) scores."""
    return pd.Series({**dict(zip(report.POSE_KEYS, pose)), **{f"c_{s}": v for s, v in zip(report.SCORES, step_scores)}})


def finals(f_lr, height):
    return pd.Series({"F_LR": f_lr, "F_AV": 0.0, "F_AH": 0.0, "wall_height": height, "n_placed": 18})


class Concordance(unittest.TestCase):
    def test_ties_excluded(self):
        points = pd.DataFrame([point(0.5, 0.1), point(1e-12, 0.1), point(0.5, 0.0), point(-0.5, -0.1)])
        self.assertEqual(report.concordance(points, "LLC", "F_LR"), (2, 2))  # the tied score and the tied final are out

    def test_discordant_is_counted_in_n_only(self):
        points = pd.DataFrame([point(0.5, 0.1), point(0.5, -0.1), point(-0.5, 0.1)])
        self.assertEqual(report.concordance(points, "LLC", "F_LR"), (1, 3))

    def test_lower_is_better_for_height(self):
        taller = pd.DataFrame([point(0.5, f_height=10.0)])
        shorter = pd.DataFrame([point(0.5, f_height=-10.0)])
        self.assertEqual(report.concordance(taller, "LLC", "wall_height"), (0, 1))
        self.assertEqual(report.concordance(shorter, "LLC", "wall_height"), (1, 1))

    def test_tied_range_uses_only_llc_tied_points(self):
        row = {f"d_{s}": 0.0 for s in report.SCORES}
        row.update({f"f_{m}": 1.0 * report.BETTER[m] for m in report.FINALS})
        points = pd.DataFrame([{**row, "d_NHR": 1.0}, {**row, "d_NHR": -1.0}, {**row, "d_LLC": 1.0, "d_NHR": -1.0}])
        self.assertEqual(report.tied_agreement_range(points), (2, 0.5, 0.5))


class Divergence(unittest.TestCase):
    def test_first_divergence(self):
        a = np.array([[1, 0], [2, 0], [3, 0]])
        self.assertEqual(report.first_divergence(a, np.array([[1, 0], [2, 1], [3, 0]])), 1)
        self.assertIsNone(report.first_divergence(a, a[:2]))  # a prefix never differs

    def test_oriented_pair_is_mirror_invariant(self):
        step_a, step_b = scored([1.0, 0, 0, 0, 0], (5, 0, 1, 2)), scored([0.5, 0, 0, 0, 0], (2, 0, 1, 2))
        forward = report.oriented_pair(7, 3, step_a, step_b, finals(0.4, 200), finals(0.1, 180))
        mirror = report.oriented_pair(7, 3, step_b, step_a, finals(0.1, 180), finals(0.4, 200))
        self.assertEqual(forward, mirror)
        self.assertEqual(forward["pick_a"], (2, 0, 1, 2))
        self.assertAlmostEqual(forward["d_LLC"], -0.5)
        self.assertAlmostEqual(forward["f_F_LR"], -0.3)
        self.assertAlmostEqual(forward["d_sum"], -0.5)


class BranchPoints(unittest.TestCase):
    def pair(self, t, picks, f_lr):
        row = {f"d_{s}": 0.25 for s in report.IMMEDIATE}
        return {
            **row,
            **{f"f_{m}": 0.0 for m in report.FINALS},
            "seed": 1,
            "t": t,
            "pick_a": picks[0],
            "pick_b": picks[1],
            "f_F_LR": f_lr,
        }

    def test_groups_by_seed_step_and_picks(self):
        picks = ((1, 0, 0, 0), (2, 0, 0, 0))
        pairs = pd.DataFrame(
            [
                self.pair(3, picks, 0.2),
                self.pair(3, picks, 0.4),
                self.pair(4, picks, 0.1),
                self.pair(3, ((1, 0, 0, 0), (9, 0, 0, 0)), 0.7),
            ]
        )
        points = report.branch_points(pairs).sort_values(["t", "pick_b"])
        self.assertEqual(points.n_pairs.tolist(), [2, 1, 1])
        self.assertAlmostEqual(points.f_F_LR.iloc[0], 0.3)
        self.assertEqual(points.t.tolist(), [3, 3, 4])


class VarianceShares(unittest.TestCase):
    def test_sum_to_one(self):
        table = pd.DataFrame({"arm": list("aabb"), "seed": [1, 2, 1, 2], "F_LR": [1.0, 2.0, 4.0, 9.0]})
        self.assertAlmostEqual(sum(report.variance_shares(table, "F_LR")), 1.0)

    def test_pure_arm_and_pure_seed(self):
        arm_only = pd.DataFrame({"arm": list("aabb"), "seed": [1, 2, 1, 2], "F_LR": [1.0, 1.0, 3.0, 3.0]})
        seed_only = arm_only.assign(F_LR=[1.0, 3.0, 1.0, 3.0])
        self.assertEqual(report.variance_shares(arm_only, "F_LR"), (1.0, 0.0, 0.0))
        self.assertEqual(report.variance_shares(seed_only, "F_LR"), (0.0, 1.0, 0.0))


class WithinSeed(unittest.TestCase):
    def test_demean_is_per_seed(self):
        table = pd.DataFrame({"seed": [1, 1, 2, 2], "x": [1.0, 3.0, 10.0, 30.0], "y": [5.0, 5.0, 6.0, 8.0]})
        out = report.demean_within_seed(table, ["x"])
        self.assertEqual(out.x.tolist(), [-1.0, 1.0, -10.0, 10.0])
        self.assertEqual(out.y.tolist(), table.y.tolist())  # only the named columns change


class Statistics(unittest.TestCase):
    def test_pareto_treats_f_ah_as_lower_better(self):
        quality = pd.DataFrame(
            {"F_SF": [1.0, 1.0, 0.5], "F_AH": [1.0, 2.0, 0.5], "F_AV": [1.0, 1.0, 0.5], "F_LR": [1.0, 1.0, 0.5]}
        )
        self.assertEqual(report.pareto_mask(quality).tolist(), [True, False, True])

    def test_enumeration_poses(self):
        """Two steps: 1 node at 3 stones left, then one node per feasible pose at 2 stones left."""
        self.assertEqual(report.enumeration_poses(np.array([5.0, 2.0]), np.array([3.0, 2.0])), 3 * 4 + 5 * 2 * 4)

    def test_final_skyline(self):
        """Rows 0-1 are ground; a rock two rows high and a thin one placed last on top of it."""
        low = np.zeros((10, 6), bool)
        low[2:4, 1:5] = True
        high = np.zeros((10, 6), bool)
        high[4:6, 1:3] = True
        sky = report.final_skyline((10, 6), [low, high])
        self.assertEqual(sky["peak"], 4)
        self.assertEqual(sky["ideal"], 3.0)
        self.assertEqual(sky["unlevel"], 1.0)
        self.assertEqual(sky["top_from_end"], 0)

    def test_timed_pose_ignores_rows_without_kinematics(self):
        table = np.array([[0, 0, 0, 1.0, 0, 0.0], [0, 0, 0, 1.0, 0, 2.0]])
        self.assertEqual(report.timed_pose_seconds(table), 3.0)


if __name__ == "__main__":
    unittest.main()
