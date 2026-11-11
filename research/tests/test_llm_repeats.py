"""The pick-4 repeats (docs/plans/llm-pick-repeats.md): the batch (research/llm_walls/repeat_wall.py) and the
reading (research/analysis/llm_repeats.py)."""
import math
import unittest
from unittest import mock

import pandas as pd

from research.analysis import llm_repeats
from research.llm_walls import repeat_wall, whole_confirm_wall, whole_wall
from research.runner import pick_wall as runner


class Batch(unittest.TestCase):
    def test_forty_walls_a_round_at_a_time_with_the_whole_wall_spec_unchanged(self):
        states, blocks = repeat_wall.STATES, [*whole_wall.STATES, *whole_confirm_wall.STATES]
        self.assertEqual(states, [(seed, (scene, r)) for r in (1, 2, 3, 4) for seed, scene in blocks])
        self.assertEqual((len(states), len({repeat_wall.name_of(*s) for s in states})), (40, 40))
        self.assertIs(repeat_wall.SPEC, whole_wall.SPEC)
        self.assertEqual((repeat_wall.spec_problems(), repeat_wall.WALL_LIMITS), ([], whole_wall.WALL_LIMITS))
        self.assertEqual((repeat_wall.name_of(313, ("TH_T80", 2)), repeat_wall.wall_args(("TH_T20", 3))),
                         ("repeat-s313-th-t80-r2", whole_wall.wall_args("TH_T20")))

    def test_the_runner_builds_them_with_repeats(self):
        with mock.patch.object(runner, "run", return_value=[]) as run:
            self.assertEqual(runner.main(["run", "--repeats", "--loopback", "--limit", "1"]), 0)
        self.assertEqual((run.call_args.kwargs["x"], run.call_args.kwargs["limit"]), (repeat_wall, 1))


class Reading(unittest.TestCase):
    def test_the_pooled_sd_weighs_each_block_by_its_degrees_of_freedom(self):
        t = pd.DataFrame({"seed": [1, 1, 2, 2, 2, 3], "scene": "TH_T80", "x": [0, 2, 1, 1, 4, 9]})
        s, df = llm_repeats.pooled(t, "x")  # variances 2 and 3 with 1 and 2 degrees; a block of one wall adds nothing
        self.assertEqual((round(s, 12), df), (round(math.sqrt(8 / 3), 12), 3))
        self.assertTrue(math.isnan(llm_repeats.pooled(t[t.seed == 3], "x")[0]))
        self.assertEqual([round(v, 3) for v in llm_repeats.interval(1.0, 30)], [0.799, 1.337])

    def test_two_walls_split_at_their_first_different_pick(self):
        for a, b, out in (((1, 2, 3), (1, 2, 3), None), ((1, 2, 3), (1, 5, 3), 2), ((1, 2), (1, 2, 3), 3)):
            with self.subTest(a=a, b=b):
                self.assertEqual(llm_repeats.first_split(a, b), out)

    def test_every_pair_of_a_block_is_judged_as_in_the_lessons_reading(self):
        t = pd.DataFrame({"seed": [1, 1, 1, 2], "scene": "TH_T80", "complete": True,
                          "llm_mean_z": [1.0, 0.0, 1.0, 5.0], "llm_n_placed": [18, 18, 17, 18]})
        self.assertEqual(llm_repeats.pairs(t), ["win", "win", "tie"])

    def test_the_blocks_needed(self):
        self.assertEqual([llm_repeats.blocks_mean(0.343, d, r) for d, r in ((0.5, 1), (0.25, 1), (0.25, 4))],
                         [6, 24, 6])
        # at even odds 5 wins of 5 reject at 0.05 and 3 of 3 at 0.20; 7 of 8 reject, and 0.9 a block wins them 81%
        self.assertEqual([llm_repeats.blocks_sign(p, a) for p, a in ((1.0, 0.05), (1.0, 0.20), (0.9, 0.05))],
                         [5, 3, 8])
        self.assertIsNone(llm_repeats.blocks_sign(0.5, 0.05, top=50))
        row = llm_repeats.sizes(0.343).iloc[1]  # a true 0.25 with 2 walls per arm
        self.assertEqual((row.delta, row.walls_per_arm, row.blocks_mean), (0.25, 2, 12))


if __name__ == "__main__":
    unittest.main()
