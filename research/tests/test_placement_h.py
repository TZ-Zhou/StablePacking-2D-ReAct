"""research/analysis/placement_h.py: one placement's state on hand-made walls (a wedge under the stone, the
remaining stones), and on S1's recorded walls when this checkout has them."""

import unittest

import numpy as np

from research.analysis import placement_h as ph
from research.analysis import surface_fit as sf
from research.paths import BATCH_RUNS


def wall_with(blocks, shape=(30, 40)):
    """Frame columns and top row 255, ground rows 0-1 = 1, then each (label, r0, r1, c0, c1) block."""
    w = np.zeros(shape)
    w[:2] = 1
    w[:, 0] = w[:, -1] = w[-1] = 255
    for value, r0, r1, c0, c1 in blocks:
        w[r0:r1, c0:c1] = value
    return w


class PlacementState(unittest.TestCase):
    def setUp(self):
        self.masks = {7: np.ones((4, 6), bool), 8: np.ones((3, 3), bool), 9: np.ones((2, 5), bool)}
        self.poses = {stone: sf.stone_poses(mask) for stone, mask in self.masks.items()}
        self.arrays = {"selected": np.array(0), "stone_id": np.array([7]), "sampled_ids": np.array([7, 8]),
                       "inventory_ids": np.array([9])}

    def test_the_largest_new_label_is_the_stone_and_a_wedge_fills_part_of_the_void(self):
        before = wall_with([])
        # stone 7 lands 3 rows up on columns 5-10; a 1 px wedge under its column 5 (inner column 4)
        after = wall_with([(2, 5, 9, 5, 11), (3, 2, 5, 5, 6)])
        state = ph.placement_state(before, after, self.arrays, self.poses, self.masks)
        self.assertEqual((state["stone"], state["n_remaining"], state["iou"]), (7, 2, 1.0))
        self.assertEqual(state["H0_unlevel"], -float(7 - sf.skyline(after).mean()))
        without_wedge = ph.placement_state(before, wall_with([(2, 5, 9, 5, 11)]), self.arrays, self.poses, self.masks)
        # 3 rows of void under each of the stone's 6 columns; the wedge fills one column, the skyline is the same
        self.assertEqual(state["H2b"] - without_wedge["H2b"], 3)

    def test_no_new_label_is_an_error(self):
        with self.assertRaises(ValueError):
            ph.placement_state(wall_with([]), wall_with([]), self.arrays, self.poses, self.masks)


S1 = BATCH_RUNS / "factorial-s313-AL_IR"


@unittest.skipUnless((S1 / "plan.csv").exists(), "needs S1's AL_IR walls")
class RecordedWalls(unittest.TestCase):
    def test_rebuilt_stones_are_the_scene_inventory_and_match_the_placed_stones(self):
        row, record = next(r for r in ph.case_records(S1) if r[1]["status"] == "PROCESS_PASS")
        case = S1 / "cases" / record["case_id"]
        masks = ph.scene_stones(case)
        steps = list(ph.placements(case))
        first = steps[0][4]
        self.assertEqual(set(masks), set(first["sampled_ids"].tolist()) | set(first["inventory_ids"].tolist()))
        table, _ = ph.h_table(S1, {row["case"]})
        self.assertEqual(len(table), len(steps))
        self.assertGreaterEqual(table["iou"].median(), 0.99)
        self.assertFalse(table[["H2b", "H3b", "H4a", "H0_unlevel"]].isna().any().any())


if __name__ == "__main__":
    unittest.main()
