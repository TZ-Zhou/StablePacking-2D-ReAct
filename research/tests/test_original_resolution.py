"""The recheck at the original resolution (research/analysis/original_resolution.py)."""
import unittest

import pandas as pd

from research.analysis import original_resolution as recheck


class OriginalResolution(unittest.TestCase):
    def test_a_conclusion_holds_only_when_both_renderings_meet_its_rule(self):
        in_se = {"LLM above Wang": (3.0, 2.0), "LLM level with pick_safe": (1.0, -2.0),
                 "pick-5 lowers F_AV": (-1.0, -1.9), "pick-5 raises F_LR": (2.2, 2.2)}
        reading = pd.DataFrame([
            {"rendering": r, "comparison": comparison, "scope": "all", "quantity": quantity, "mean_diff": x / 10,
             "se_within": 0.1, "in_se_within": x, "se_blocks": 0.2, "in_se_blocks": x / 2, "blocks_positive": 5}
            for name, comparison, quantity, _ in recheck.CONCLUSIONS for r, x in zip("AB", in_se[name])])
        c = recheck.conclusions(reading)
        self.assertEqual(c.drop_duplicates("conclusion").verdict.tolist(),
                         ["holds", "depends on the rendering", "does not hold", "holds"])
        self.assertEqual(c.meets.tolist(), [True, True, True, False, False, False, True, True])

    @unittest.skipUnless((recheck.FOLDER / "walls.csv").exists(), "the walls are not measured")
    def test_the_recorded_grid_values_give_the_published_tables(self):
        self.assertLess(recheck.off_published(recheck.scored(recheck.measured(), "G")), recheck.TOLERANCE)


if __name__ == "__main__":
    unittest.main()
