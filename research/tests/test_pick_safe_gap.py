"""The step-by-step comparison of the LLM and pick_safe (research/analysis/pick_safe_gap.py)."""
import unittest

import numpy as np

from research.analysis import pick_safe_gap as gap


class PickSafeGap(unittest.TestCase):
    def test_a_decision_is_classed_against_the_rule_in_the_same_state(self):
        raw = np.zeros((5, 6))
        raw[:, 3] = [0.9, 0.7, 0.7, 0.5, 0.0]  # LLC; the last row is not feasible
        after = np.array([[0, 2], [1, 2], [3, 1], [2, 2], [-1, -1]])  # row 0 leaves a stone without a pose
        self.assertEqual(gap.rule(raw, after)[0].tolist(), [1, 2])  # the highest LLC among the safe rows
        self.assertEqual([gap.classify(raw, after, k)["kind"] for k in range(4)],
                         ["unsafe", "same", "same_llc", "lower_llc"])
        self.assertAlmostEqual(gap.classify(raw, after, 3)["llc_diff"], -0.2)
        self.assertTrue(gap.classify(raw, after, 3)["higher_safe"])
        none_safe = np.zeros((5, 2), dtype=int)
        self.assertEqual(gap.rule(raw, none_safe)[0].tolist(), [0])  # no safe row: the highest LLC of all
        self.assertEqual(gap.classify(raw, none_safe, 1)["kind"], "lower_llc")

    def test_the_sentence_about_a_candidate_is_found_by_pose_then_by_stone(self):
        text = ("S7 at 180° is balanced. S8 at 0° or 90° has top LLC but worse NHR; S8 at 180° has a larger void. "
                "S12 (2) offers better IWR.")
        s8 = {"id": 5, "stone": "S8", "rotation": 90}
        self.assertEqual(gap.said(text, s8), ("pose", "S8 at 0° or 90° has top LLC but worse NHR;"))
        self.assertEqual(gap.said(text, {**s8, "rotation": 270})[0], "stone")
        self.assertEqual(gap.said(text, {"id": 2, "stone": "S1", "rotation": 0}),
                         ("pose", "S12 (2) offers better IWR."))
        self.assertEqual(gap.said(text, {"id": 9, "stone": "S1", "rotation": 0}), ("none", ""))
        self.assertEqual(gap.tags("worse NHR and a larger void"), "height+void")


if __name__ == "__main__":
    unittest.main()
