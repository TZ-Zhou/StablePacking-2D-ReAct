"""docs/plans/llm-safe-top-hint.md: the list and its paragraph (research/controller/pick.py's safe_top_llc), the batch
(research/llm_walls/safe_top_wall.py) and the reading (research/analysis/llm_safe_top.py)."""
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_pick  # noqa: E402

from research.analysis import llm_safe_top  # noqa: E402
from research.controller import order_controller as oc  # noqa: E402
from research.controller import pick  # noqa: E402
from research.llm_walls import repeat_wall, safe_top_wall, whole_wall  # noqa: E402
from research.paths import REPO  # noqa: E402
from research.runner import pick_wall as runner  # noqa: E402

PICK_4 = "c9b8128adf32529b074a7e22ae1013acb47759829bcb2c787527281f939c4242"  # the paid whole-wall prompt
PLAN = (REPO / "docs" / "plans" / "llm-safe-top-hint.md").read_text(encoding="utf-8")
WHOLE = {"lookahead": True, "whole_wall": True}


def listed(**patched):
    with mock.patch.multiple(test_pick, **{"RAW": test_pick.RAW, **patched}):
        return test_pick.run("pick_safe", inventory=[7], **WHOLE, safe_top_llc=True)[1]["observation"]


class Hint(unittest.TestCase):
    def test_the_list_is_the_safe_candidates_at_their_highest_llc(self):
        obs = listed()  # candidates 0, 1, 2 with LLC 1.0, 2.0, 1.5; candidate 1 leaves S1 at 0
        self.assertEqual((obs["observation_version"], obs["safe_top_llc"], len(obs["candidates"])), ("pick-5", [2], 3))
        raw = test_pick.RAW.copy()
        raw[0, 3] = 1.5
        self.assertEqual(listed(RAW=raw)["safe_top_llc"], [0, 2])  # a tie: both
        unsafe = np.where(test_pick.AFTER > 0, 0, test_pick.AFTER)
        self.assertEqual(listed(AFTER=unsafe)["safe_top_llc"], [])  # no safe candidate: empty
        plain = test_pick.run("pick_safe", inventory=[7], **WHOLE)[1]["observation"]
        self.assertEqual((plain["observation_version"], "safe_top_llc" in plain), ("pick-4", False))
        self.assertEqual({**obs, "observation_version": "pick-4"}, {**plain, "safe_top_llc": [2]})

    def test_the_prompt_is_pick_4_and_the_plans_paragraph_and_without_the_key_it_is_unchanged(self):
        quoted = [line[2:] for line in PLAN.splitlines() if line.startswith("> ")]
        self.assertEqual(quoted, [pick.SAFE_TOP.rstrip("\n")])
        self.assertEqual(pick.SYSTEM_SAFE.replace(pick.SAFE_TOP, ""), pick.SYSTEM_WHOLE)
        llm = oc.MockLLM("pick_safe")
        plain = pick.PickPolicy(llm, max_tool_calls=4, **WHOLE)
        told = pick.PickPolicy(llm, max_tool_calls=4, **WHOLE, safe_top_llc=True)
        self.assertEqual((plain.prompt_sha256, plain.system), (PICK_4, pick.SYSTEM_WHOLE))
        self.assertEqual((told.system, told.tools, told.prompt_sha256 != PICK_4), (pick.SYSTEM_SAFE, plain.tools, True))
        mocked = {"kind": "pick", "llm": {"kind": "mock", "mode": "pick_safe"}}
        oc.validate_spec({**mocked, **WHOLE, "safe_top_llc": True})
        for spec in ({"lookahead": True, "safe_top_llc": True}, {**WHOLE, "safe_top_llc": 1}):
            with self.subTest(spec), self.assertRaises(ValueError):
                oc.validate_spec({**mocked, **spec})


class Batch(unittest.TestCase):
    def test_the_states_of_the_repeats_with_the_key_added_to_the_whole_wall_spec(self):
        self.assertIs(safe_top_wall.STATES, repeat_wall.STATES)
        self.assertEqual(len({safe_top_wall.name_of(*s) for s in safe_top_wall.STATES}), 40)
        self.assertEqual(safe_top_wall.SPEC, {**whole_wall.SPEC, "safe_top_llc": True})
        self.assertEqual(safe_top_wall.spec_problems(), [])
        self.assertNotEqual(safe_top_wall.spec_problems(whole_wall.SPEC), [])
        self.assertEqual((safe_top_wall.WALL_LIMITS, safe_top_wall.EXPERIMENT_LIMITS),
                         (whole_wall.WALL_LIMITS, repeat_wall.EXPERIMENT_LIMITS))
        self.assertEqual((safe_top_wall.name_of(313, ("TH_T80", 2)), safe_top_wall.wall_args(("TH_T20", 3))),
                         ("safetop-s313-th-t80-r2", whole_wall.wall_args("TH_T20")))

    def test_the_runner_builds_them_with_safe_top(self):
        with mock.patch.object(runner, "run", return_value=[]) as run:
            self.assertEqual(runner.main(["run", "--safe-top", "--loopback", "--limit", "1"]), 0)
        self.assertEqual((run.call_args.kwargs["x"], run.call_args.kwargs["limit"]), (safe_top_wall, 1))


class Reading(unittest.TestCase):
    def test_a_block_difference_in_the_spread_of_both_batches(self):
        cols = dict.fromkeys((*llm_safe_top.llm_repeats.COLS, "llm_n_placed"), 0.0)
        new = pd.DataFrame({"seed": [1, 1, 2, 2], "scene": "TH_T80", **cols, "llm_mean_z": [1.0, 3.0, 2.0, 2.0]})
        old = new.assign(llm_mean_z=[0.0, 2.0, 1.0, 1.0])
        row = llm_safe_top.reading(new, old).set_index("quantity").loc["llm_mean_z"]
        # both blocks differ by 1; the variances 2 and 0 pool to 1 on each side; se = 1 * sqrt(2 * (1/2 + 1/2)) / 2
        self.assertEqual((row.mean_diff, row.sd_within, row.df, row.sd_blocks), (1.0, 1.0, 4, 0.0))
        self.assertAlmostEqual(row.se_within, math.sqrt(2) / 2)
        self.assertAlmostEqual(row.in_se_within, math.sqrt(2))

    def test_the_four_readings(self):
        for f_lr, mean_z, said in ((2.0, -1.9, "F_LR raised and the mean z kept"),
                                   (2.5, -2.0, "F_LR raised but the mean z lowered"),
                                   (1.9, 5.0, "no difference seen"), (-2.0, 0.0, "F_LR lowered"),
                                   (float("nan"), 0.0, "not read: no spread measured")):
            with self.subTest(f_lr=f_lr, mean_z=mean_z):
                self.assertEqual(llm_safe_top.verdict(f_lr, mean_z), said)


if __name__ == "__main__":
    unittest.main()
