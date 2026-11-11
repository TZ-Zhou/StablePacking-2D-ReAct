"""research/controller/endgame.py: when it acts, what each mode picks, and its batch_plan level."""

import unittest

import numpy as np
import pandas as pd

from research.analysis import endgame_arena
from research.controller import endgame
from research.walls import batch_plan

# rows: NHR, GDS, IWR, LLC, LFR; row 2 is infeasible (LLC 0)
RAW = np.array([[0.1, 0.5, 0.2, 1.0, 0.0],
                [0.3, 0.4, 0.1, 2.0, 0.1],
                [0.9, 0.9, 0.9, 0.0, 0.9],
                [0.2, 0.6, 0.3, 1.5, 0.2]])
Y, HEIGHT = [10, 4, 0, 3], [5, 5, 5, 9]


# the check_after table of the 4 rows over 2 stones: row 1 leaves a stone at 0 (unsafe), row 2 is infeasible
AFTER = np.array([[-1, 2], [0, -1], [-1, -1], [1, -1]])


def context(inventory, after=AFTER):
    return {"step": 15, "phase": "single", "sample_trial": 0, "side": "left", "features_raw": RAW,
            "weights": np.ones(5), "inventory_ids": inventory, "y": Y, "height": HEIGHT, "after": after}


class Picker(unittest.TestCase):
    def picks(self, spec, inventory=(), after=AFTER):
        records = []
        policy = endgame.EndgamePicker(spec, records.append)({"policy": "paper", "order": (3, 0, 2, 1, 4), "tol": 1e-9},
                                                              context(list(inventory), after))
        return policy.get("pick"), records[0]

    def test_rules(self):
        self.assertEqual(self.picks({"kind": "endgame", "mode": "llc"})[0], 1)
        self.assertEqual(self.picks({"kind": "endgame", "mode": "borda"})[0], 3)  # ranks 2+3+3+2+3 of the 3 feasible
        self.assertEqual(self.picks({"kind": "endgame", "mode": "low_top"})[0], 1)  # tops 15, 9, 12

    def test_random_is_feasible_and_seeded(self):
        picks = {self.picks({"kind": "endgame", "mode": "random", "seed": s})[0] for s in range(30)}
        self.assertEqual(picks, {0, 1, 3})
        self.assertEqual(self.picks({"kind": "endgame", "mode": "random", "seed": 4})[0],
                         self.picks({"kind": "endgame", "mode": "random", "seed": 4})[0])

    def test_safe_modes_keep_to_the_safe_rows_and_fall_back_when_none_is(self):
        pick, record = self.picks({"kind": "endgame", "mode": "low_top_safe"})
        self.assertEqual((pick, record["n_safe"]), (3, 2))  # low_top's row 1 is unsafe; tops of rows 0, 3: 15, 12
        picks = {self.picks({"kind": "endgame", "mode": "random_safe", "seed": s})[0] for s in range(30)}
        self.assertEqual(picks, {0, 3})
        unsafe = np.array([[0, 2], [0, -1], [-1, -1], [1, 0]])
        self.assertEqual(self.picks({"kind": "endgame", "mode": "low_top_safe"}, after=unsafe)[0], 1)  # as low_top
        self.assertTrue(endgame.EndgamePicker({"kind": "endgame", "mode": "random_safe", "seed": 1}).lookahead)
        self.assertFalse(endgame.EndgamePicker({"kind": "endgame", "mode": "low_top"}).lookahead)
        self.assertNotIn("n_safe", self.picks({"kind": "endgame", "mode": "low_top"})[1])  # earlier records unchanged

    def test_a_check_after_row_must_be_what_the_wall_did_next(self):
        raw = np.vstack([RAW, [[0.0] * 5] * 4])  # 2 stones x 4 rotations; stone 41 at rows 0, 2, 4, 6
        nxt = {"sampled_ids": np.array([41]), "features_raw": RAW, "selected": np.array(0)}  # 41 then: rows 0, 1, 3
        told = np.full((8, 2), -1)
        told[1] = [3, -1]  # row 1 places stone 17 and tells 3 feasible poses of 41
        first = {"sampled_ids": np.array([41, 17]), "features_raw": raw, "selected": np.array(1), "after": told,
                 "inventory_ids": np.array([])}
        events = [({"event": 0, "step": 15}, first), ({"event": 1, "step": 16}, nxt)]
        self.assertEqual(endgame.lookahead_problems(events), [])
        told[1] = [2, -1]
        self.assertEqual(len(endgame.lookahead_problems(events)), 1)

    def test_no_pick_while_stones_are_drawn_or_in_mode_none(self):
        pick, record = self.picks({"kind": "endgame", "mode": "llc"}, inventory=[7])
        self.assertIsNone(pick)
        self.assertEqual((record["acted"], record["inventory_empty"], record["n_feasible"]), (False, False, 3))
        self.assertIsNone(self.picks({"kind": "endgame", "mode": "none"})[0])

    def test_bad_specs(self):
        for spec in ({"kind": "endgame", "mode": "random"}, {"kind": "endgame", "mode": "llc", "seed": 1},
                     {"kind": "endgame", "mode": "best"}, {"kind": "endgame", "mode": "llc", "extra": 1},
                     {"kind": "endgame", "mode": "random_safe"},
                     {"kind": "endgame", "mode": "low_top_safe", "seed": 2}):
            with self.subTest(spec), self.assertRaises(ValueError):
                endgame.full_spec(spec)


class ArenaReading(unittest.TestCase):
    def test_reading(self):
        def table(pct, better):
            return pd.DataFrame({"cascade_percentile": pct, "better_exists": better})

        self.assertEqual(endgame_arena.reading(table([0.95] * 7 + [0.2] * 2, [False] * 7 + [True] * 2)), "no room")
        self.assertEqual(endgame_arena.reading(table([0.95] * 7 + [0.2] * 2, [True] * 3 + [False] * 6)), "room")
        self.assertEqual(endgame_arena.reading(table([0.5] * 5 + [0.95] * 4, [False] * 9)), "no floor")

    def test_prefix_stops_at_seven_left(self):
        steps = pd.DataFrame({"wall": "a", "outcome": ["placed", "no_placement", "placed", "placed"],
                              "n_remaining": [9, 8, 8, 7], "stone": [1, 2, 2, 3], "rot": 0, "x": [0, 5, 5, 9], "y": 0})
        self.assertEqual(endgame_arena.prefix(steps, "a"), [(1, 0, 0, 0), (2, 0, 5, 0)])


class PlanLevel(unittest.TestCase):
    def test_levels(self):
        self.assertIsNone(batch_plan.filter_policy_problem("30214+random:7"))
        self.assertIsNotNone(batch_plan.filter_policy_problem("30214+random"))
        self.assertIsNotNone(batch_plan.filter_policy_problem("3021+llc"))
        for level in ("30214+low_top_safe", "30214+random_safe:3", "30214+pick_safe"):
            self.assertIsNone(batch_plan.filter_policy_problem(level), level)
        self.assertIsNotNone(batch_plan.filter_policy_problem("30214+random_safe"))
        self.assertEqual(batch_plan.endgame_spec("30214+pick_safe"),
                         {"kind": "pick", "llm": {"kind": "mock", "mode": "pick_safe"}, "lookahead": True})
        self.assertEqual(batch_plan.policy_args({"filter_policy": "30214+llc"}),
                         ["--policy", "paper", "--order", "3,0,2,1,4",
                          "--controller", '{"kind": "endgame", "mode": "llc"}'])


if __name__ == "__main__":
    unittest.main()
