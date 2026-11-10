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


def context(inventory):
    return {"step": 15, "phase": "single", "sample_trial": 0, "side": "left", "features_raw": RAW,
            "weights": np.ones(5), "inventory_ids": inventory, "y": Y, "height": HEIGHT}


class Picker(unittest.TestCase):
    def picks(self, spec, inventory=()):
        records = []
        policy = endgame.EndgamePicker(spec, records.append)({"policy": "paper", "order": (3, 0, 2, 1, 4), "tol": 1e-9},
                                                              context(list(inventory)))
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

    def test_no_pick_while_stones_are_drawn_or_in_mode_none(self):
        pick, record = self.picks({"kind": "endgame", "mode": "llc"}, inventory=[7])
        self.assertIsNone(pick)
        self.assertEqual((record["acted"], record["inventory_empty"], record["n_feasible"]), (False, False, 3))
        self.assertIsNone(self.picks({"kind": "endgame", "mode": "none"})[0])

    def test_bad_specs(self):
        for spec in ({"kind": "endgame", "mode": "random"}, {"kind": "endgame", "mode": "llc", "seed": 1},
                     {"kind": "endgame", "mode": "best"}, {"kind": "endgame", "mode": "llc", "extra": 1}):
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
        self.assertEqual(batch_plan.policy_args({"filter_policy": "30214+llc"}),
                         ["--policy", "paper", "--order", "3,0,2,1,4",
                          "--controller", '{"kind": "endgame", "mode": "llc"}'])


if __name__ == "__main__":
    unittest.main()
