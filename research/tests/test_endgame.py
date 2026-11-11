"""research/controller/endgame.py: when it acts, what each mode picks, and its batch_plan level."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from research.analysis import endgame_arena, whole_wall_rules
from research.controller import endgame
from research.llm_walls import whole_wall
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

    def test_cascade_safe_leaves_the_cascade_unless_its_row_is_unsafe(self):
        spec = {"kind": "endgame", "mode": "cascade_safe"}
        pick, record = self.picks(spec)
        self.assertEqual((record["cascade_index"], pick, record["n_safe"]), (1, 3, 2))  # row 1 unsafe: rows 0, 3 alone
        alone = endgame.select(RAW[[0, 3]], np.ones(5), "paper", (3, 0, 2, 1, 4), 1e-9).index
        self.assertEqual(pick, [0, 3][alone])
        safe = np.array([[0, 2], [1, -1], [-1, -1], [1, -1]])  # the cascade's row 1 is safe, row 0 is not
        self.assertEqual(self.picks(spec, after=safe)[0], 1)
        unsafe = np.array([[0, 2], [0, -1], [-1, -1], [1, 0]])
        self.assertEqual(self.picks(spec, after=unsafe)[0], 1)  # no row is safe: the cascade

    def test_whole_acts_with_stones_in_the_inventory_and_earlier_specs_keep(self):
        pick, record = self.picks({"kind": "endgame", "mode": "borda", "whole": True}, inventory=[7])
        self.assertEqual((pick, record["acted"], record["inventory_empty"]), (3, True, False))
        self.assertEqual(record["controller"], {"kind": "endgame", "mode": "borda", "seed": None, "whole": True})
        pick, record = self.picks({"kind": "endgame", "mode": "cascade_safe", "whole": True}, inventory=[7])
        self.assertEqual((pick, record["n_safe"]), (3, 2))
        self.assertEqual(endgame.EndgamePicker({"kind": "endgame", "mode": "cascade_safe", "whole": True}).lookahead,
                         "always")
        self.assertIs(endgame.EndgamePicker({"kind": "endgame", "mode": "cascade_safe"}).lookahead, True)
        self.assertFalse(endgame.EndgamePicker({"kind": "endgame", "mode": "borda", "whole": True}).lookahead)
        self.assertEqual(endgame.full_spec({"kind": "endgame", "mode": "llc"}),
                         {"kind": "endgame", "mode": "llc", "seed": None})

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
                     {"kind": "endgame", "mode": "low_top_safe", "seed": 2},
                     {"kind": "endgame", "mode": "borda", "whole": False}):
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


class WholeWallRules(unittest.TestCase):
    def test_reading(self):
        read = whole_wall_rules.reading
        self.assertEqual([read(k, 7) for k in (8, 7)], ["the rule reaches the LLM's count without an LLM"] * 2)
        self.assertEqual([read(k, 7) for k in (6, 5)], ["not separated"] * 2)
        self.assertEqual([read(k, 7) for k in (4, 0)], ["the LLM is 3 or more walls ahead of the rule"] * 2)

    def test_the_blocks_are_the_llms_and_each_has_its_batch(self):
        self.assertEqual(len(whole_wall_rules.BLOCKS), 10)
        for seed, scene in whole_wall_rules.BLOCKS:
            config = Path(batch_plan.__file__).parent / "endgame" / f"whole-rules-s{seed}-{scene}.yaml"
            text = config.read_text(encoding="utf-8")
            with self.subTest(seed=seed, scene=scene):
                self.assertTrue(all(f'"30214+{mode}"' in text for mode in whole_wall_rules.RULES.values()))
                self.assertIn(f"seed: [{seed}]", text)
                self.assertIn(f"wall_type: {{{scene}}}", text)

    def test_not_below_wang(self):
        t = pd.DataFrame({"complete": [True, True, True, False], "mean_z": [0.1, 0.0, 0.1, 0.1], "wang_mean_z": 0.0,
                          "n_placed": [18, 18, 17, 18], "wang_n_placed": 18})
        self.assertEqual(whole_wall_rules.not_below_wang(t).tolist(), [True, True, False, False])


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
        # docs/plans/whole-wall-rules.md: the three rules, pick_safe on the budget of the LLM's whole wall
        for level in ("30214+" + mode for mode in whole_wall_rules.RULES.values()):
            self.assertIsNone(batch_plan.filter_policy_problem(level), level)
        self.assertEqual(batch_plan.endgame_spec("30214+whole_cascade_safe"),
                         {"kind": "endgame", "mode": "cascade_safe", "whole": True})
        spec = batch_plan.endgame_spec("30214+whole_pick_safe")
        self.assertEqual(spec, {"kind": "pick", "llm": {"kind": "mock", "mode": "pick_safe"}, "lookahead": True,
                                "whole_wall": True, "max_tool_calls": 4, "max_decisions": 25})
        self.assertEqual([spec[k] for k in ("whole_wall", "max_tool_calls", "max_decisions")],
                         [whole_wall.SPEC[k] for k in ("whole_wall", "max_tool_calls", "max_decisions")])
        self.assertEqual(batch_plan.policy_args({"filter_policy": "30214+llc"}),
                         ["--policy", "paper", "--order", "3,0,2,1,4",
                          "--controller", '{"kind": "endgame", "mode": "llc"}'])


if __name__ == "__main__":
    unittest.main()
