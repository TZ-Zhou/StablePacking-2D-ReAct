"""research/analysis/myopia_tables.py on hand-made inputs: step-state statistics of a small wall, one step row from a
fake trace event, the choice of the traced result directory, and the deduplication of walls."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from research.analysis import myopia_tables


def wall_with(blocks, shape=(20, 8)):
    """A wall matrix: frame columns and top row 255, ground rows 0-1 = 1, then each (r0, r1, c0, c1)
    block as brick 2, 3, ..."""
    wall = np.zeros(shape)
    wall[:2] = 1
    wall[:, 0] = wall[:, -1] = wall[-1] = 255
    for k, (r0, r1, c0, c1) in enumerate(blocks, 2):
        wall[r0:r1, c0:c1] = k
    return wall


class StepState(unittest.TestCase):
    def test_empty_wall_is_flat(self):
        state = myopia_tables.step_state(wall_with([]))
        self.assertEqual((state["h_max"], state["h_min"], state["void_under"], state["n_jumps"]), (0, 0, 0, 0))
        self.assertEqual((state["trapped_ratio"], state["unlevel"], state["roughness"]), (0.0, 0.0, 0.0))

    def test_heights_voids_and_levelling(self):
        """Columns of heights 3, 3, 1, 3, 0, 0; the fourth holds one floating pixel, so two pixels below it are void."""
        state = myopia_tables.step_state(wall_with([(2, 5, 1, 3), (2, 3, 3, 4), (4, 5, 4, 5)]))
        heights = np.array([3, 3, 1, 3, 0, 0])
        self.assertEqual((state["h_max"], state["h_min"], state["h_range"]), (3, 0, 3))
        self.assertAlmostEqual(state["h_mean"], heights.mean())
        self.assertAlmostEqual(state["h_std"], heights.std())
        self.assertAlmostEqual(state["roughness"], (0 + 2 + 2 + 3 + 0) / 5)
        self.assertEqual(state["void_under"], 2)
        self.assertAlmostEqual(state["trapped_ratio"], 2 / 10)
        self.assertAlmostEqual(state["unlevel"], (0 + 0 + 2 + 0 + 3 + 3) / (6 * 3))

    def test_only_steps_above_the_threshold_are_jumps(self):
        """Heights 9, 0, 8, 8, 8, 8: the step of 9 counts, the step of 8 does not."""
        state = myopia_tables.step_state(wall_with([(2, 11, 1, 2), (2, 10, 3, 7)]))
        self.assertEqual(state["h_max"], 9)
        self.assertEqual(state["n_jumps"], 1)


def trace_arrays(raw, selected):
    """The fields of a trace event that step_row reads, for two or three candidates."""
    count = len(raw)
    return dict(
        features_raw=np.array(raw, float),
        weights=np.ones(5),
        selected=np.array(selected),
        stone_id=np.arange(7, 7 + count),
        sampled_ids=np.array([7, 8]),
        inventory_ids=np.array([20, 21, 22]),
        wall=wall_with([]),
        rotation_world=np.full(count, 90),
        x_world=np.arange(count) + 40,
        y=np.full(count, 2),
        width=np.full(count, 30),
        height=np.full(count, 12),
    )


META = {"policy": {"tol": 1e-9}, "branch": "full"}
COMMIT = {"step": 4, "outcome": "placed", "sample_trial": 0, "sides": {"left": {"event": 12}}, "chosen_side": "left"}


class StepRow(unittest.TestCase):
    def test_counts_and_chosen_scores(self):
        """Row 2 has a non-finite score, so only rows 0 and 1 are feasible, and row 0 wins under every order."""
        raw = [[1, 1, 1, 1, 1], [0, 0, 0, 0.5, 0], [np.nan, 1, 1, 1, 1]]
        row = myopia_tables.step_row("w", COMMIT, META, trace_arrays(raw, 0))
        self.assertEqual((row["wall"], row["event"], row["step"], row["termination"]), ("w", 12, 4, None))
        self.assertEqual((row["n_candidates"], row["n_feasible"], row["n_feasible_stones"]), (3, 2, 2))
        self.assertEqual((row["n_sampled"], row["n_remaining"]), (2, 5))
        self.assertEqual((row["n_reach"], row["n_reach_stones"], row["n_orders_diff_from_paper"]), (1, 1, 0))
        self.assertTrue(row["chosen_is_paper_pick"])
        self.assertEqual(row["n_feasible_llc_levels"], 2)
        self.assertEqual(
            (row["stone"], row["rot"], row["x"], row["y"], row["width"], row["height_px"]), (7, 90, 40, 2, 30, 12)
        )
        self.assertEqual((row["c_LLC"], row["mean_LLC"], row["max_LLC"], row["pct_LLC"]), (1.0, 0.75, 1.0, 1.0))
        self.assertEqual(row["llc_tie_share"], 0.5)

    def test_orders_that_disagree(self):
        """Whichever of columns 0 and 1 is filtered first wins; the paper order filters column 0 first."""
        raw = [[1, 0, 0.5, 1, 0.5], [0, 1, 0.5, 1, 0.5]]
        row = myopia_tables.step_row("w", COMMIT, META, trace_arrays(raw, 1))
        self.assertEqual((row["n_reach"], row["n_reach_stones"], row["n_orders_diff_from_paper"]), (2, 2, 60))
        self.assertFalse(row["chosen_is_paper_pick"])
        self.assertEqual(row["pct_NHR"], 0.5)

    def test_no_selection_has_no_pose(self):
        row = myopia_tables.step_row("w", COMMIT, META, trace_arrays([[0, 0, 0, 0, 0]], -1))
        self.assertEqual(row["n_feasible"], 0)
        self.assertEqual(row["n_reach"], 0)
        self.assertTrue(row["chosen_is_paper_pick"])  # neither the recording nor the paper order selected anything
        self.assertNotIn("stone", row)

    def test_chosen_event_falls_back_to_the_first_side(self):
        sides = {"left": {"event": 3}, "right": {"event": 5}}
        self.assertEqual(myopia_tables.chosen_event({"sides": sides, "chosen_side": "right"}), 5)
        self.assertEqual(myopia_tables.chosen_event({"sides": sides, "chosen_side": None}), 3)
        self.assertIsNone(myopia_tables.chosen_event({"sides": {"left": {}}, "chosen_side": "left"}))


def make_result(run, name, traced):
    """A result directory under run, with an empty commits.jsonl if traced."""
    trace = run / name / "trace"
    trace.mkdir(parents=True)
    if traced:
        (trace / "commits.jsonl").write_text("", encoding="utf-8")


class TracedResult(unittest.TestCase):
    def test_last_result_with_a_trace(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            make_result(run, "result_1", traced=True)
            make_result(run, "result_2", traced=True)
            make_result(run, "result_3", traced=False)
            self.assertEqual(myopia_tables.traced_result(run).name, "result_2")

    def test_no_trace(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(myopia_tables.traced_result(Path(folder)))


def walls_table(rows):
    """walls.csv columns unique_walls reads, from (name, seed) pairs of finished walls with a quality value."""
    return pd.DataFrame(
        [dict(wall=name, seed=seed, status="PROCESS_PASS", F_LR=0.2, result=f"/r/{name}") for name, seed in rows]
    )


def steps_table(sequences):
    """steps.csv columns unique_walls reads: every (stone, rot, x, y) of {wall: placements} as a placed step."""
    return pd.DataFrame(
        [
            dict(wall=wall, outcome="placed", stone=s, rot=r, x=x, y=y)
            for wall, poses in sequences.items()
            for s, r, x, y in poses
        ]
    )


FIRST = [(1, 0, 10, 2), (2, 90, 40, 2)]
SECOND = [(2, 90, 40, 2), (1, 0, 10, 2)]


class UniqueWalls(unittest.TestCase):
    def test_identical_sequences_of_one_seed_keep_the_first_by_name(self):
        unique = myopia_tables.unique_walls(walls_table([("b", 7), ("a", 7)]), steps_table({"a": FIRST, "b": FIRST}))
        self.assertEqual(unique.to_dict("records"), [{"wall": "a", "result": "/r/a"}])

    def test_identical_sequences_of_different_seeds_are_kept(self):
        unique = myopia_tables.unique_walls(walls_table([("a", 7), ("b", 8)]), steps_table({"a": FIRST, "b": FIRST}))
        self.assertEqual(unique["wall"].tolist(), ["a", "b"])

    def test_placement_order_is_part_of_the_signature(self):
        unique = myopia_tables.unique_walls(walls_table([("a", 7), ("b", 7)]), steps_table({"a": FIRST, "b": SECOND}))
        self.assertEqual(unique["wall"].tolist(), ["a", "b"])

    def test_only_placed_steps_count(self):
        walls = walls_table([("a", 7), ("b", 7)])
        steps = pd.concat(
            [
                steps_table({"a": FIRST, "b": FIRST}),
                pd.DataFrame([dict(wall="b", outcome="no_placement", stone=9, rot=0, x=0, y=0)]),
            ]
        )
        self.assertEqual(myopia_tables.unique_walls(walls, steps)["wall"].tolist(), ["a"])

    def test_unfinished_and_unmeasured_walls_are_left_out(self):
        walls = walls_table([("a", 7), ("b", 8), ("c", 9)])
        walls.loc[0, "status"] = "PROCESS_FAIL"
        walls.loc[1, "F_LR"] = np.nan
        steps = steps_table({"a": FIRST, "b": FIRST, "c": FIRST})
        self.assertEqual(myopia_tables.unique_walls(walls, steps)["wall"].tolist(), ["c"])


class WallRow(unittest.TestCase):
    def test_columns_and_group(self):
        entry = dict(
            path="branches/x/y",
            seed=7,
            policy="paper",
            order="3-0-2-1-4",
            switch="",
            controller="",
            llm="",
            status="PROCESS_PASS",
            placements_limit=0,
        )
        row = myopia_tables.wall_row(entry, Path("/r"), 23, {"n_input": 18})
        self.assertEqual(
            (row["wall"], row["group"], row["n_steps"], row["n_input"]), ("branches/x/y", "branches", 23, 18)
        )
        self.assertEqual(list(row)[:3], ["wall", "group", "seed"])


if __name__ == "__main__":
    unittest.main()
