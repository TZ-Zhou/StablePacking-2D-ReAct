"""research/analysis/surface_fit.py and the pure parts of surface_fit_check.py on small skylines with known answers.

Run from the repo root: python -m pytest research/tests/test_surface_fit.py -q
"""

import math
import unittest

import numpy as np
import pandas as pd

from research.analysis import surface_fit as sf
from research.analysis import surface_fit_check as check

POCKETS = np.array([10] * 20 + [30] * 20 + [0] * 5 + [30] * 20 + [50] * 10 + [10] * 25)  # a 5 px slit, depth 30
SLIT_AREA = 30 * 5
WIDE = 12  # wider than the slit
NARROW = 4  # narrower than the slit
STAIRS = 10 * np.arange(6)  # every drop touches only the highest column, right of the centre: no supported seat


def bar(rows, cols):
    """The pose of a full rectangle."""
    return sf.make_pose(np.ones((rows, cols), bool))


def wall_with(blocks, shape=(40, 60)):
    """A wall matrix: frame columns and top row 255, ground rows 0-1 = 1, then each (r0, r1, c0, c1)
    block as brick 2, 3, ..."""
    w = np.zeros(shape)
    w[:2] = 1
    w[:, 0] = w[:, -1] = w[-1] = 255
    for k, (r0, r1, c0, c1) in enumerate(blocks, 2):
        w[r0:r1, c0:c1] = k
    return w


def event(wall, **fields):
    """The trace arrays surface_fit_check reads from one event: candidate 0, stone 7, unless fields say otherwise."""
    arrays = {"selected": 0, "stone_id": (7,), "sampled_ids": (), "inventory_ids": (), **fields}
    return {"wall": wall, **{key: np.array(value, int) for key, value in arrays.items()}}


class Profiles(unittest.TestCase):
    def test_skyline_is_the_highest_occupied_pixel(self):
        w = wall_with([(2, 12, 1, 30), (2, 7, 30, 40), (10, 14, 45, 50)])  # the last block overhangs an empty gap
        s = sf.skyline(w)
        self.assertEqual(len(s), 58)
        self.assertEqual(s.tolist(), [10] * 29 + [5] * 10 + [0] * 5 + [12] * 5 + [0] * 9)

    def test_make_pose(self):
        pose = sf.make_pose(np.array([[1, 1, 1], [1, 0, 0]], bool))  # row 0 is the lowest
        self.assertEqual(pose.bottom.tolist(), [0, 0, 0])
        self.assertEqual(pose.top.tolist(), [2, 1, 1])
        self.assertEqual(pose.center, 0.75)
        self.assertEqual(pose.area, 4)

    def test_make_pose_with_an_undercut(self):
        pose = sf.make_pose(np.array([[1, 0], [1, 1]], bool))
        self.assertEqual(pose.bottom.tolist(), [0, 1])
        self.assertEqual(pose.top.tolist(), [2, 2])

    def test_stone_poses_are_the_four_rotations(self):
        poses = sf.stone_poses(np.ones((2, 3), bool))
        self.assertEqual([len(pose.bottom) for pose in poses], [3, 2, 3, 2])
        self.assertEqual({pose.area for pose in poses}, {6})


class Place(unittest.TestCase):
    def test_flat_seat_seals_nothing(self):
        new, sealed = sf.place(POCKETS, 22, 30, bar(10, 6))
        self.assertEqual(sealed, 0)
        self.assertEqual(new[22:28].tolist(), [40] * 6)

    def test_seals_the_void_it_bridges(self):
        new, sealed = sf.place(np.array([5, 0, 5]), 0, 5, bar(3, 3))
        self.assertEqual(sealed, 5)
        self.assertEqual(new.tolist(), [8, 8, 8])

    def test_input_skyline_is_not_changed(self):
        s = POCKETS.copy()
        sf.place(s, 22, 30, bar(10, 6))
        self.assertTrue(np.array_equal(s, POCKETS))


class SurfaceTerms(unittest.TestCase):
    def test_identity_without_a_peak_rise(self):
        """G_after - G_before = W*rise - sum(s_after - s_before), with rise = 0."""
        new, _ = sf.place(POCKETS, 22, 30, bar(10, 6))
        before, _ = sf.surface_terms(POCKETS, WIDE)
        after, _ = sf.surface_terms(new, WIDE)
        self.assertEqual(new.max(), POCKETS.max())
        self.assertEqual(after - before, -int((new - POCKETS).sum()))

    def test_identity_with_a_peak_rise(self):
        new, _ = sf.place(POCKETS, 66, 50, bar(40, 6))
        before, _ = sf.surface_terms(POCKETS, WIDE)
        after, _ = sf.surface_terms(new, WIDE)
        rise = int(new.max() - POCKETS.max())
        self.assertGreater(rise, 0)
        self.assertEqual(after - before, len(POCKETS) * rise - int((new - POCKETS).sum()))

    def test_closing_marks_a_slit_narrower_than_the_width_dead(self):
        _, dead = sf.surface_terms(POCKETS, WIDE)
        self.assertGreaterEqual(dead, SLIT_AREA)

    def test_a_narrower_width_leaves_the_slit_open(self):
        _, dead = sf.surface_terms(POCKETS, NARROW)
        self.assertLess(dead, SLIT_AREA)

    def test_envelope_counts_open_area_dead_pockets_and_sealed_void(self):
        open_area, dead = sf.surface_terms(POCKETS, WIDE)
        self.assertEqual(sf.envelope_h(POCKETS, 7, WIDE), -(open_area + dead + 7))
        self.assertEqual(sf.envelope_h(POCKETS, 7, WIDE, dead_weight=0.0), -open_area)


class FluidLevel(unittest.TestCase):
    def test_flat(self):
        self.assertEqual(sf.fluid_level(np.zeros(10, int), 100), 10.0)

    def test_stays_inside_the_low_columns(self):
        self.assertEqual(sf.fluid_level(np.array([0, 0, 10, 10]), 10), 5.0)

    def test_rises_above_every_column(self):
        self.assertEqual(sf.fluid_level(np.array([0, 0, 10, 10]), 60), 20.0)


class Seats(unittest.TestCase):
    def test_small_skyline(self):
        rest, void, top, supported = sf.seats(np.array([5, 5, 0, 5, 5, 5]), bar(3, 3))
        self.assertEqual(rest.tolist(), [5, 5, 5, 5])
        self.assertEqual(void.tolist(), [5, 5, 5, 0])
        self.assertEqual(top.tolist(), [8, 8, 8, 8])
        self.assertEqual(supported.tolist(), [True, True, True, True])

    def test_contact_on_one_side_of_the_centre_is_not_supported(self):
        self.assertFalse(sf.seats(STAIRS, bar(3, 3))[3].any())

    def test_contact_tolerance_counts_a_shallow_gap(self):
        depth = sf.CONTACT_TOLERANCE
        self.assertTrue(sf.seats(np.array([0, depth, depth + depth]), bar(3, 3))[3].all())
        self.assertFalse(sf.seats(np.array([0, depth + 1, 2 * depth + 2]), bar(3, 3))[3].any())


class Fit(unittest.TestCase):
    def test_stone_fit_on_a_flat_skyline(self):
        self.assertEqual(sf.stone_fit(np.zeros(6, int), [bar(3, 3)]), (4, 0, 3))

    def test_a_bad_seat_is_not_counted(self):
        count, void, low_top = sf.stone_fit(np.array([5, 0, 5]), [bar(3, 3)])  # void 5 > GOOD_VOID_RATIO * area
        self.assertEqual((count, void, low_top), (0, 5, 8))

    def test_fit_scores_with_nothing_remaining(self):
        s = np.array([3, 9, 4])
        self.assertEqual(sf.fit_scores(s, []), {"worst_top": -9.0, "scarcity": 0.0, "void": 0.0})

    def test_fit_scores_on_a_flat_skyline(self):
        scores = sf.fit_scores(np.zeros(6, int), [[bar(3, 3)]])
        self.assertEqual(scores["worst_top"], -3.0)
        self.assertAlmostEqual(scores["scarcity"], -1 / 5)
        self.assertEqual(scores["void"], 0.0)

    def test_stone_without_a_supported_seat(self):
        """It counts as no good seat, its own area as void, and a top above the current peak."""
        self.assertEqual(sf.stone_fit(STAIRS, [bar(3, 3)]), (0, math.inf, math.inf))
        scores = sf.fit_scores(STAIRS, [[bar(3, 3)]])
        self.assertEqual(scores, {"worst_top": -53.0, "scarcity": -1.0, "void": -9.0})


class Rollout(unittest.TestCase):
    LEDGE_THEN_STAIRS = np.array([0, 0, 0, 10, 20, 30, 40, 50])  # a 3 px ledge; nothing wider has a supported seat

    def test_best_seat_prefers_the_low_flat_pocket(self):
        pose = bar(3, 3)
        self.assertEqual(sf.best_seat(np.array([5, 5, 0, 0, 0, 5, 5]), [pose]), (3.0, 2, 0, pose))

    def test_best_seat_is_none_without_a_supported_seat(self):
        self.assertIsNone(sf.best_seat(STAIRS, [bar(3, 3)]))

    def test_rollout_counts_an_unplaced_stone(self):
        result = sf.rollout(self.LEDGE_THEN_STAIRS, [[bar(3, 3)], [bar(5, 5)]])
        self.assertEqual(result, (50, 1, 0))

    def test_rollout_with_every_stone_unplaced(self):
        self.assertEqual(sf.rollout(STAIRS, [[bar(3, 3)], [bar(2, 3)]]), (50, 2, 0))

    def test_bridging_a_pocket_adds_to_sealed_void(self):
        self.assertEqual(sf.rollout(np.array([5, 0, 5]), [[bar(3, 3)]]), (8, 0, 5))
        self.assertEqual(sf.rollout(np.array([5, 5, 5]), [[bar(3, 3)]]), (8, 0, 0))

    def test_best_fit_rollout_counts_what_is_left(self):
        result = sf.rollout_best_fit(self.LEDGE_THEN_STAIRS, [[bar(3, 3)], [bar(5, 5)]])
        self.assertEqual(result, (50, 1, 0))

    def test_best_fit_rollout_places_everything_that_fits(self):
        self.assertEqual(sf.rollout_best_fit(np.zeros(6, int), [[bar(3, 3)], [bar(2, 2)]]), (3, 0, 0))

    def test_rollouts_leave_the_input_skyline_alone(self):
        s = np.zeros(6, int)
        sf.rollout(s, [[bar(3, 3)]])
        sf.rollout_best_fit(s, [[bar(3, 3)]])
        self.assertEqual(s.tolist(), [0] * 6)


class Landing(unittest.TestCase):
    def test_exact_landing(self):
        self.assertEqual(
            check.compare_landing(np.array([5, 5, 5]), bar(2, 3), 0, 5),
            {"diff": 0, "sealed_ratio": 0.0, "supported": True},
        )

    def test_landing_under_an_overhang_is_lower_than_the_drop(self):
        landing = check.compare_landing(np.array([20, 20, 20]), bar(2, 3), 0, 0)
        self.assertEqual(landing["diff"], -20)
        self.assertEqual(landing["sealed_ratio"], 0.0)

    def test_void_over_area(self):
        landing = check.compare_landing(np.array([5, 0, 5]), bar(3, 3), 0, 5)
        self.assertAlmostEqual(landing["sealed_ratio"], 5 / 9)
        self.assertTrue(landing["supported"])

    def test_contact_on_one_side_of_the_centre_is_not_supported(self):
        self.assertFalse(check.compare_landing(np.array([0, 10, 20]), bar(3, 3), 0, 20)["supported"])

    def test_no_contact_is_not_supported(self):
        landing = check.compare_landing(np.zeros(3, int), bar(3, 3), 0, 30)
        self.assertEqual(landing, {"diff": 30, "sealed_ratio": 10.0, "supported": False})


class Fidelity(unittest.TestCase):
    def test_crop(self):
        mask = np.zeros((6, 6), bool)
        mask[1:3, 2:4] = True
        cropped, row, col = check.crop(mask)
        self.assertEqual(cropped.tolist(), [[True, True], [True, True]])
        self.assertEqual((row, col), (1, 2))

    def test_new_label(self):
        before = np.array([0, 1, 2])
        self.assertEqual(check.new_label(before, np.array([0, 1, 2, 5])), 5)
        self.assertIsNone(check.new_label(before, np.array([0, 1, 2, 5, 6])))
        self.assertIsNone(check.new_label(before, before))

    def test_selected_row(self):
        self.assertEqual(check.selected_row(event(None, selected=3)), 3)
        self.assertEqual(check.selected_row(event(None, selected=[])), -1)

    def test_placement_records_skip_what_is_not_one_placed_brick(self):
        empty, one = wall_with([]), wall_with([(2, 5, 3, 6)])
        placed = event(empty)
        no_choice = event(empty, selected=-1)
        self.assertEqual(check.placement_records([no_choice, event(one)]), [])
        self.assertEqual(check.placement_records([placed, event(empty)]), [])
        self.assertEqual(check.placement_records([placed, event(wall_with([(2, 5, 3, 6), (2, 5, 9, 12)]))]), [])
        ((stone, mask, row, col, before),) = check.placement_records([placed, event(one)])
        self.assertEqual((stone, mask.shape, row, col), (7, (3, 3), 2, 3))
        self.assertIs(before, empty)

    def test_fidelity_table_on_a_stone_dropped_on_the_ground(self):
        events = [event(wall_with([])), event(wall_with([(2, 5, 3, 6)]))]
        table, masks = check.fidelity_table([("a", events)])
        self.assertEqual(table.to_dict("records"), [{"wall": "a", "diff": 0, "sealed_ratio": 0.0, "supported": True}])
        self.assertEqual(masks[7].shape, (3, 3))

    def test_fidelity_table_keeps_the_first_mask_of_a_stone(self):
        first = [event(wall_with([])), event(wall_with([(2, 5, 3, 6)]))]
        second = [event(wall_with([])), event(wall_with([(2, 4, 3, 9)]))]
        _, masks = check.fidelity_table([("a", first), ("b", second)])
        self.assertEqual(masks[7].shape, (3, 3))

    def test_summary(self):
        table = pd.DataFrame(
            {"diff": [0, 0, 0, -10], "sealed_ratio": [0.0, 0.2, 0.4, 0.6], "supported": [True, True, True, False]}
        )
        summary = check.fidelity_summary(table)
        self.assertEqual(summary["placements"], 4)
        self.assertEqual(summary["equal"], 0.75)
        self.assertEqual(summary["lower"], 0.25)
        self.assertEqual(summary["diff_min"], -10)
        self.assertAlmostEqual(summary["diff_p5"], -8.5)
        self.assertEqual(summary["supported"], 0.75)
        self.assertAlmostEqual(summary["void_median"], 0.3)
        self.assertAlmostEqual(summary["void_p90"], 0.54)


class Timing(unittest.TestCase):
    def test_timed_runs_fn_repeat_times(self):
        calls = []
        self.assertGreaterEqual(check.timed(lambda: calls.append(1), 5), 0.0)
        self.assertEqual(len(calls), 5)

    def test_remaining_stones_exclude_the_chosen_one(self):
        decided = event(None, selected=1, stone_id=(4, 7), sampled_ids=(4, 7), inventory_ids=(2,))
        self.assertEqual(check.remaining_stones(decided), [2, 4])

    def test_timing_wall(self):
        unique = pd.DataFrame(
            {
                "wall": [
                    f"branches/x/{check.TIMING_WALL_SEED}",
                    "campaigns/x/seed1",
                    f"campaigns/y/{check.TIMING_WALL_SEED}",
                ],
                "result": ["a", "b", "c"],
            }
        )
        self.assertEqual(check.timing_wall(unique).result, "c")

    def test_timing_table_has_one_row_per_step(self):
        masks = {1: np.ones((3, 3), bool), 2: np.ones((2, 4), bool)}
        skipped = event(wall_with([]), selected=-1)
        decided = event(wall_with([]), stone_id=(9,), sampled_ids=(1,), inventory_ids=(2,))
        table = check.timing_table([skipped, decided], masks, steps=(1,))
        self.assertEqual(table["stones_left"].tolist(), [2])
        self.assertTrue((table.drop(columns="stones_left").to_numpy() >= 0).all())


class HScores(unittest.TestCase):
    def test_reference_width_is_the_quarter_quantile_of_narrow_sides(self):
        remaining = [sf.stone_poses(np.ones((r, c), bool)) for r, c in [(2, 9), (4, 9), (6, 9), (8, 9), (10, 9)]]
        self.assertEqual(sf.reference_width(remaining), 4)
        self.assertEqual(sf.reference_width([]), 1)

    def test_a_flat_top_beats_a_slit_on_h2b(self):
        stones = [sf.stone_poses(np.ones((10, 10), bool))]
        flat, slit = np.full(30, 20), np.array([20] * 12 + [0] * 6 + [20] * 12)
        self.assertGreater(sf.h_scores(flat, 0, stones)["H2b"], sf.h_scores(slit, 0, stones)["H2b"])
        self.assertEqual(sf.h_scores(flat, 0, stones)["H0_unlevel"], 0.0)

    def test_h4a_puts_unplaced_stones_before_height_and_height_before_void(self):
        self.assertGreater(sf.h4a_score(0, 10**5 - 1, 10**7 - 1), sf.h4a_score(1, 0, 0))
        self.assertGreater(sf.h4a_score(0, 5, 10**7 - 1), sf.h4a_score(0, 6, 0))
        self.assertGreater(sf.h4a_score(0, 5, 3), sf.h4a_score(0, 5, 4))

    def test_no_stone_left(self):
        scores = sf.h_scores(np.array([5, 5, 9]), 3, [])
        self.assertEqual((scores["H3b"], scores["H2b"]), (0.0, -(4 + 4 + 3)))


if __name__ == "__main__":
    unittest.main()
