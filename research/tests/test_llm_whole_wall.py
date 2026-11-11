"""docs/plans/llm-whole-wall-pick.md: its prompt, spec, batch and rule (research/controller/pick.py's whole_wall,
research/llm_walls/whole_wall.py, research/analysis/llm_whole_wall.py), and its confirmation on the held-out seeds
(docs/plans/llm-whole-wall-confirm.md, research/llm_walls/whole_confirm_wall.py)."""

import json
import os
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_gpt_smoke  # noqa: E402
import test_pick  # noqa: E402

from research.analysis import llm_whole_wall as ww  # noqa: E402
from research.controller import endgame, pick  # noqa: E402
from research.controller import order_controller as oc  # noqa: E402
from research.llm_walls import gpt_wall, lookahead_wall, whole_confirm_wall, whole_wall  # noqa: E402
from research.paths import DATASETS_DIR, REPO  # noqa: E402
from research.runner import pick_wall as pick_runner  # noqa: E402
from research.runner import run_wall  # noqa: E402

PLAN = (REPO / "docs" / "plans" / "llm-whole-wall-pick.md").read_text(encoding="utf-8")
ROWS = [[c.strip() for c in line.split("|")[1:-1]] for line in PLAN.splitlines()
        if line.startswith(("| 313 |", "| 484 |", "| 717 |"))]
CONFIRM = (REPO / "docs" / "plans" / "llm-whole-wall-confirm.md").read_text(encoding="utf-8")
CONFIRM_ROWS = [[c.strip() for c in line.split("|")[1:-1]] for line in CONFIRM.splitlines()
                if line.startswith(("| 720 |", "| 761 |"))]
MOCK = {"kind": "pick", "llm": {"kind": "mock", "mode": "pick_safe"}}


def limits(plan, x):
    """(the limits a plan states, those of its batch module x)."""
    stated = re.search(r"每面牆最多 (\d+) 個請求、(\d+)k tokens、US\$([\d.]+)；整批最多 (\d+) 面、(\d+) 個請求、"
                       r"([\d.]+)M tokens、\*\*US\$(\d+)\*\*", plan).groups()
    wall, batch = x.WALL_LIMITS, x.EXPERIMENT_LIMITS
    return [float(v) for v in stated], [wall["http_requests"], wall["tokens"] / 1e3, wall["cost_usd"], batch["walls"],
                                        batch["http_requests"], batch["tokens"] / 1e6, batch["cost_usd"]]


def table(llm, n=18, complete=True):
    return pd.DataFrame({"llm_mean_z": llm, "wang_mean_z": 0.0, "llm_n_placed": n, "wang_n_placed": 18,
                         "complete": complete})


class WholeWall(unittest.TestCase):
    def test_the_prompt_is_the_plans_and_the_earlier_prompts_keep(self):
        quoted = [line[2:] for line in PLAN.splitlines() if line.startswith("> ")]
        self.assertEqual(pick.SYSTEM_WHOLE, "\n".join(quoted))
        self.assertIn("All four metrics must be good: F_SF, F_AH, F_AV and F_LR. Do not sacrifice the others for one "
                      "of them.", pick.SYSTEM_WHOLE)
        self.assertNotIn(pick.LLC_NOTE, pick.SYSTEM_WHOLE)
        llm = oc.MockLLM("pick_safe")
        whole = pick.PickPolicy(llm, max_tool_calls=4, lookahead=True, whole_wall=True)
        self.assertEqual(whole.system, pick.SYSTEM_WHOLE)
        self.assertEqual([t["name"] for t in whole.tools], ["check_after", "commit_pick"])
        self.assertIn("each other stone on offer now", whole.tools[0]["description"])
        # pick-1 to pick-4 as the paid walls recorded them (step 3, lookahead, LLC note, whole wall)
        for case, spec in (("20261110-2141-TH_T80-pc1", {}), ("20261111-0106-TH_T80-pc1", {"lookahead": True}),
                           ("20261111-0528-TH_T80-pc1", {"lookahead": True, "llc_note": True}),
                           ("20261111-1101-TH_T80-pc1", {"lookahead": True, "whole_wall": True})):
            trace = next((REPO / "research" / "cases" / case).glob("result_*/trace/controller.jsonl"))
            record = next(r for r in map(json.loads, trace.read_text(encoding="utf-8").splitlines())
                          if r.get("prompt_sha256"))
            policy = pick.PickPolicy(llm, max_tool_calls=record["controller"]["max_tool_calls"], **spec)
            with self.subTest(case):
                self.assertEqual(policy.prompt_sha256, record["prompt_sha256"])

    def test_every_selection_with_two_candidates_is_a_decision(self):
        policy, record = test_pick.run("pick_safe", inventory=[7], lookahead=True, whole_wall=True)
        self.assertEqual((record["decision_point"], record["inventory_empty"], record["source"]), (True, False, "llm"))
        self.assertEqual((record["observation"]["observation_version"], record["observation"]["n_remaining"]),
                         ("pick-4", 3))
        self.assertEqual(policy["pick"], 3)  # as in the endgame: candidate 1 leaves S1 at 0
        self.assertFalse(test_pick.run("pick_safe", inventory=[7], lookahead=True)[1]["decision_point"])
        controller = oc.make_controller({**MOCK, "lookahead": True, "whole_wall": True}, lambda r: None,
                                        transport="inline", ledger=lambda e: None)
        self.assertEqual(controller.lookahead, "always")  # build_wall: the table at every selection
        for spec in ({"whole_wall": True}, {"lookahead": True, "whole_wall": 1},
                     {"lookahead": True, "llc_note": True, "whole_wall": True}):
            with self.subTest(spec), self.assertRaises(ValueError):
                oc.validate_spec({**MOCK, **spec})

    def test_a_table_with_stones_in_the_inventory_is_checked_on_the_stones_sampled_again(self):
        raw = np.vstack([test_pick.RAW[:4], [[0.0] * 5] * 4])  # 2 stones x 4 rotations
        told = np.full((8, 2), -1)
        told[1] = [3, -1]  # row 1 places stone 17 and tells 3 feasible poses of 41
        first = {"sampled_ids": np.array([41, 17]), "features_raw": raw, "selected": np.array(1), "after": told,
                 "inventory_ids": np.array([7])}
        events = [({"event": 0, "step": 3}, first), None]
        for nxt, problems in ((np.array([7]), 0),  # 41 not sampled again: nothing to compare
                              (np.array([41]), 0), (np.array([41, 7]), 1)):  # 41 with 3 feasible poses, then 2
            rows = test_pick.RAW[:4] if len(nxt) == 1 else np.vstack([test_pick.RAW[:4]] * 2)
            events[1] = ({"event": 1, "step": 4}, {"sampled_ids": nxt, "features_raw": rows, "selected": np.array(0)})
            with self.subTest(nxt=nxt):
                self.assertEqual(len(endgame.lookahead_problems(events)), problems)
        first["inventory_ids"] = np.array([])  # the endgame: every stone told must be there
        self.assertEqual(len(endgame.lookahead_problems(events)), 1)

    def test_the_batch_is_the_plans(self):
        self.assertEqual(whole_wall.spec_problems(), [])
        for bad in ({"whole_wall": False}, {"llc_note": True}, {"max_decisions": 26}, {"max_cost_usd": 0.6}):
            with self.subTest(bad):
                self.assertTrue(whole_wall.spec_problems({**whole_wall.SPEC, **bad}))
        self.assertEqual(whole_wall.SPEC, {**lookahead_wall.SPEC, "whole_wall": True, "max_decisions": 25,
                                           "max_total_tokens": 250_000, "max_cost_usd": 0.50})
        self.assertEqual(whole_wall.STATES, [(int(r[0]), r[1]) for r in ROWS])
        self.assertEqual(*limits(PLAN, whole_wall))
        names = [whole_wall.name_of(*s) for s in whole_wall.STATES]
        self.assertTrue(all(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,40}", n) for n in names), names)  # a run name
        with mock.patch.object(gpt_wall, "journals", return_value={names[0]: {"state": "finished", "started_at": "1"}}):
            self.assertEqual(pick_runner.next_state(False, whole_wall), (313, "TH_T20", names[1], names[0]))
        self.assertEqual(whole_wall.wall_args("TH_T20"), ["thangavelu-noise20"])

    def test_the_confirmation_is_the_plans(self):
        c = whole_confirm_wall
        self.assertEqual(c.spec_problems(), [])
        self.assertTrue(c.spec_problems({**c.SPEC, "llc_note": True}))
        self.assertEqual((c.SPEC, c.WALL_LIMITS, c.WALL_ARGS),  # the batch repeated unchanged
                         (whole_wall.SPEC, whole_wall.WALL_LIMITS, whole_wall.WALL_ARGS))
        self.assertEqual(c.STATES, [(int(r[0]), r[1]) for r in CONFIRM_ROWS])
        self.assertFalse(set(c.STATES) & set(whole_wall.STATES))
        self.assertFalse({seed for seed, _ in c.STATES} & {313, 484, 717})  # held out
        self.assertEqual(*limits(CONFIRM, c))
        names = [c.name_of(*s) for s in c.STATES]
        self.assertTrue(all(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,40}", n) for n in names), names)
        with mock.patch.object(gpt_wall, "journals", return_value={}):
            self.assertEqual(pick_runner.next_state(False, c), (720, "TH_T80", names[0], None))
        # the chances the plan states: a random other order not below Wang in at least 4, 3, 2 of the 4 walls
        chance = np.array([1.0])
        for row in CONFIRM_ROWS:
            chance = np.convolve(chance, [1 - int(row[5]) / 119, int(row[5]) / 119])
        stated = dict(re.findall(r"^\| (\d) \| ([\d.]+) \|", CONFIRM, re.M))
        self.assertEqual(sorted(stated), ["2", "3", "4"])
        for k, p in stated.items():
            with self.subTest(walls=k):
                self.assertAlmostEqual(chance[int(k):].sum(), float(p), delta=0.005)

    @unittest.skipUnless((DATASETS_DIR / "factorial-s313-TH_T20" / "walls.csv").exists(), "no factorial tables")
    def test_wangs_walls_are_the_plans(self):
        for seed, scene, wall, stones, z, others in ROWS + CONFIRM_ROWS:
            w, block = ww.wang(int(seed), scene)
            passing = ((block.mean_z >= w.mean_z) & (block.n_placed >= w.n_placed)).sum() - 1
            with self.subTest(seed=seed, scene=scene):
                self.assertEqual((w.wall, f"{w.n_placed} / {w.n_input}", f"{w.mean_z:.3f}".replace("-", "−"), passing),
                                 (wall, stones, z, int(others)))

    def test_rule(self):
        self.assertEqual(ww.NEEDED, 4)
        self.assertEqual(ww.reading(table([0.1] * 3 + [0.0] + [-0.1] * 2)), "pass: the whole wall is not below Wang")
        self.assertEqual(ww.reading(table([0.1] * 3 + [-0.1] * 3)), "catching up with Wang not shown")
        self.assertEqual(ww.reading(table([0.1] * 6, n=[17] * 3 + [18] * 3)), "catching up with Wang not shown")
        self.assertEqual(ww.reading(table([0.1] * 6, complete=[False] * 3 + [True] * 3)),
                         "catching up with Wang not shown")

    def test_the_confirmation_rule(self):
        self.assertEqual(ww.CONFIRM_NEEDED, 3)
        confirmed = "confirmed: the whole wall is not below Wang on the held-out seeds"
        self.assertEqual(ww.reading(table([0.1, 0.0, 0.1, -0.1]), confirm=True), confirmed)
        self.assertEqual(ww.reading(table([0.1, 0.1, -0.1, -0.1]), confirm=True),
                         "not confirmed on the held-out seeds")
        self.assertEqual(ww.reading(table([0.1] * 4, n=[17, 17, 18, 18]), confirm=True),
                         "not confirmed on the held-out seeds")
        diffs = pd.DataFrame({f"diff_{c}": [0.0, 0.5] for c in ww.step3.ZCOLS})
        diffs.iloc[1, 0] = -0.1  # one metric below Wang
        self.assertEqual(ww.all_four(diffs).tolist(), [True, False])


@unittest.skipUnless(test_pick.VENV and Path(sys.prefix).resolve() == run_wall.ENV_PREFIX,
                     "needs the baseline env and the venv")
class LoopbackWall(unittest.TestCase):
    def test_a_loopback_whole_wall_passes(self):
        """TH_T20 through the venv worker: decisions from the first stone, each check_after row checked against the
        next selection (controller verdict PASS)."""
        test_gpt_smoke.isolate(self)
        server = test_gpt_smoke.Server()
        self.addCleanup(server.shutdown)
        env = {**test_gpt_smoke.CLEAN_ENV, "OPENAI_API_KEY": test_gpt_smoke.DUMMY, "OPENAI_BASE_URL": server.url}
        spec = {**test_pick.LOOKAHEAD_SPEC, "whole_wall": True, "max_decisions": 25, "max_total_tokens": 250_000,
                "max_cost_usd": 0.50}
        with mock.patch.dict(os.environ, env, clear=True):
            receipt = run_wall.run(["thangavelu-noise20", "--seed", "313", "--policy", "paper", "--order", "3,0,2,1,4",
                                    "--worker-venv", "--controller", json.dumps(spec)])
        report = receipt["trace_check"]["controller"][0]
        self.assertEqual((receipt["status"], report["verdict"]), ("PROCESS_PASS", "PASS"), receipt["trace_check"])
        self.assertGreater(report["decision_points"], 10)  # the endgame alone has at most 3
        self.assertEqual(len(server.requests), 2 * report["decision_points"])  # check_after, then commit_pick
        self.assertTrue(all(r["body"]["instructions"] == pick.SYSTEM_WHOLE for r in server.requests))


if __name__ == "__main__":
    unittest.main()
