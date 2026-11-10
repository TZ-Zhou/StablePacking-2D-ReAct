"""research/controller/pick.py: when the LLM pick acts, what it sends, how replies are validated and its fallbacks."""

import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_gpt_smoke  # noqa: E402

from research.analysis import llm_endgame_pick  # noqa: E402
from research.controller import (
    llm_api,  # noqa: E402
    pick,  # noqa: E402
)
from research.controller import order_controller as oc  # noqa: E402
from research.controller import worker as controller_worker  # noqa: E402
from research.llm_walls import gpt_smoke, gpt_wall, pick_wall  # noqa: E402
from research.runner import pick_wall as pick_runner  # noqa: E402
from research.runner import run_wall  # noqa: E402

# 2 stones x 2 rotations; row 2 is infeasible (LLC 0). rows: NHR, GDS, IWR, LLC, LFR
RAW = np.array([[0.1, 0.5, 0.2, 1.0, 0.0],
                [0.3, 0.4, 0.1, 2.0, 0.1],
                [0.9, 0.9, 0.9, 0.0, 0.9],
                [0.2, 0.6, 0.3, 1.5, 0.2]] + [[0.0] * 5] * 4)
BASE = types.MappingProxyType({"policy": "paper", "order": (3, 0, 2, 1, 4), "tol": 1e-9})


# the check_after table (rows x stones 41, 17): row 1 (stone 17) leaves stone 41 at 0
AFTER = np.array([[-1, 2], [0, -1], [-1, -1], [-1, 1]] + [[-1, -1]] * 4)


def context(inventory=(), step=15):
    return {"step": step, "phase": "single", "sample_trial": 0, "side": "left", "features_raw": RAW,
            "weights": np.ones(5), "sampled_ids": [41, 17], "inventory_ids": list(inventory),
            "x": np.array([0, 5, 9, 2, 0, 0, 0, 0]), "y": np.array([10, 4, 0, 3, 0, 0, 0, 0]),
            "width": np.array([4] * 8), "height": np.array([5, 5, 5, 9, 1, 1, 1, 1]),
            "wall": np.zeros((20, 30)), "all_bricks": None, "after": AFTER}


def run(mode, inventory=(), **spec):
    records = []
    controller = oc.make_controller({"kind": "pick", "llm": {"kind": "mock", "mode": mode}, **spec},
                                    records.append, transport="inline", ledger=lambda e: None)
    return controller(BASE, context(inventory)), records[0]


class Lookahead(unittest.TestCase):
    def test_check_after_reads_the_table_and_pick_safe_keeps_to_the_safe_candidates(self):
        policy, record = run("pick_safe", lookahead=True)
        self.assertEqual(record["observation"]["observation_version"], "pick-2")
        self.assertEqual([s["tool"] for s in record["react"]], ["check_after", "commit_pick"])
        self.assertEqual(record["react"][0]["feedback"], {"after": {"0": {"S2": 2}, "1": {"S1": 0}, "2": {"S2": 1}}})
        self.assertEqual(policy["pick"], 3)  # candidate 1 (LLC 2.0) is unsafe; of 0 and 2 (LLC 1.0, 1.5): 2, row 3
        self.assertEqual((record["final"]["candidate"], record["final"]["switched"]), (2, True))

    def test_the_tool_and_the_prompt_only_with_lookahead(self):
        llm = oc.MockLLM("pick_llc")
        plain, ahead = pick.PickPolicy(llm), pick.PickPolicy(llm, max_tool_calls=4, lookahead=True)
        self.assertEqual([t["name"] for t in plain.tools], ["commit_pick"])
        self.assertEqual([t["name"] for t in ahead.tools], ["check_after", "commit_pick"])
        self.assertEqual(ahead.system.replace(pick.LOOKAHEAD, ""), plain.system)
        recorded = Path(__file__).resolve().parents[1] / "cases" / "20261110-2141-TH_T80-pc1"
        record = json.loads(next(recorded.glob("result_*/trace/controller.jsonl")).read_text(encoding="utf-8")
                            .splitlines()[11])
        self.assertEqual(plain.prompt_sha256, record["prompt_sha256"])  # step 3's prompt is unchanged
        self.assertEqual(run("pick_llc")[1]["observation"]["observation_version"], "pick-1")

    def test_check_after_ids_are_validated(self):
        policy = pick.PickPolicy(oc.MockLLM("pick_llc"), lookahead=True)
        request = {"observation": {"candidates": [{}] * 3}}
        for ids, ok in (("0,2", True), (" 1 ", True), ("0,0", False), ("3", False), ("a", False), ("", False)):
            reply = {"thought": "", "tool": "check_after", "arguments": {"candidates": ids}}
            with self.subTest(ids):
                self.assertEqual(policy.problem(reply, request) is None, ok)
        with self.assertRaises(ValueError):
            oc.validate_spec({"kind": "pick", "llm": {"kind": "mock", "mode": "pick_safe"}, "lookahead": 1})


class Pick(unittest.TestCase):
    def test_observation_lists_every_feasible_candidate_without_stone_ids(self):
        policy, record = run("pick_llc")
        obs = record["observation"]
        self.assertEqual([c["id"] for c in obs["candidates"]], [0, 1, 2])  # rows 0, 1, 3
        self.assertEqual([c["stone"] for c in obs["candidates"]], ["S1", "S2", "S2"])
        self.assertEqual(obs["candidates"][2]["rotation"], 90)
        self.assertNotIn("41", json.dumps(obs))
        self.assertEqual((policy["pick"], record["source"], record["final"]["switched"]), (1, "llm", False))
        policy, record = run("pick_first")
        self.assertEqual((policy["pick"], record["final"]["cascade_index"], record["final"]["switched"]), (0, 1, True))

    def test_low_top_mock_and_ties(self):
        self.assertEqual(run("pick_low_top")[0]["pick"], 1)  # tops 15, 9, 12

    def test_no_request_while_stones_are_drawn(self):
        policy, record = run("pick_llc", inventory=[7])
        self.assertNotIn("pick", policy)
        self.assertEqual((record["decision_point"], record["source"], record["observation"]),
                         (False, "not_a_decision_point", None))

    def test_a_rejected_id_is_retried_then_falls_back(self):
        policy, record = run("pick_retry")
        self.assertEqual((policy["pick"], [s["status"] for s in record["react"]]), (1, ["rejected", "ok"]))
        policy, record = run("pick_invalid", max_tool_calls=2)
        self.assertNotIn("pick", policy)
        self.assertEqual((record["source"], record["final"]["index"]), ("fallback:no_valid_commit", 1))

    def test_wall_budget(self):
        records = []
        controller = oc.make_controller({"kind": "pick", "llm": {"kind": "mock", "mode": "pick_first"},
                                         "max_decisions": 1}, records.append, transport="inline",
                                        ledger=lambda e: None)
        controller(BASE, context())
        self.assertNotIn("pick", controller(BASE, context(step=16)))
        self.assertEqual([r["source"] for r in records], ["llm", "skip:wall_budget"])

    def test_worker_replies_are_validated(self):
        for reply, ok in (({"pick": 2, "source": "llm"}, True), ({"pick": None, "source": "fallback:llm_error"}, True),
                          ({"pick": 3, "source": "llm"}, False), ({"pick": 1, "source": "fallback:llm_error"}, False),
                          ({"pick": True, "source": "llm"}, False), ({"order": "3-0-2-1-4", "source": "llm"}, False)):
            with self.subTest(reply):
                self.assertEqual(pick.reply_problem(reply, 3) is None, ok)

    def test_bad_specs(self):
        mock = {"kind": "mock", "mode": "pick_llc"}
        for spec in ({"kind": "pick"}, {"kind": "pick", "llm": mock, "tool_feedback": False}):
            with self.subTest(spec), self.assertRaises((ValueError, NotImplementedError, TypeError)):
                oc.validate_spec(spec)


class Harness(unittest.TestCase):
    def test_the_spec_fits_and_the_states_are_the_arena_states(self):
        self.assertEqual(pick_wall.spec_problems(), [])
        self.assertTrue(pick_wall.spec_problems({**pick_wall.SPEC, "max_cost_usd": 0.25}))
        arena = Path(__file__).resolve().parents[2] / "docs" / "plans" / "endgame-arena.md"
        rows = [line.split("|")[1:3] for line in arena.read_text(encoding="utf-8").splitlines()
                if line.startswith("| 3") or line.startswith("| 4") or line.startswith("| 7")]
        self.assertEqual(pick_wall.STATES, [(int(s), o.strip()) for s, o in rows])
        self.assertEqual(pick_wall.wall_args("30214"), ["--order", "3,0,2,1,4"])

    def test_the_lookahead_spec_fits_and_its_states_are_the_plans_new_states(self):
        from research.llm_walls import lookahead_wall
        self.assertEqual(lookahead_wall.spec_problems(), [])
        self.assertTrue(lookahead_wall.spec_problems({**lookahead_wall.SPEC, "lookahead": False}))
        self.assertTrue(lookahead_wall.spec_problems({**lookahead_wall.SPEC, "max_tool_calls": 5}))  # 50 requests
        plan = Path(__file__).resolve().parents[2] / "docs" / "plans" / "llm-endgame-lookahead.md"
        rows = [line.split("|")[1:3] for line in plan.read_text(encoding="utf-8").splitlines()
                if line.startswith(("| 313 |", "| 484 |", "| 717 |"))]
        self.assertEqual(lookahead_wall.STATES, [(int(s), o.strip()) for s, o in rows])
        self.assertFalse(set(lookahead_wall.STATES) & set(pick_wall.STATES))  # never step 3's states
        self.assertIs(oc.full_spec(lookahead_wall.SPEC)["lookahead"], True)
        names = [lookahead_wall.name_of(*s) for s in lookahead_wall.STATES]
        with mock.patch.object(gpt_wall, "journals", return_value={names[0]: {"state": "finished", "started_at": "1"}}):
            self.assertEqual(pick_runner.next_state(False, lookahead_wall), (313, "14302", names[1], names[0]))

    def test_next_state_resumes_after_the_finished_walls(self):
        names = [pick_wall.name_of(*s) for s in pick_wall.STATES]
        done = {names[0]: {"state": "finished", "started_at": "1"}, names[1]: {"state": "finished", "started_at": "2"}}
        with mock.patch.object(gpt_wall, "journals", return_value=done):
            self.assertEqual(pick_runner.next_state(False), (313, "34021", names[2], names[1]))
        every = {n: {"state": "finished", "started_at": n} for n in names}
        with mock.patch.object(gpt_wall, "journals", return_value=every):
            self.assertIsNone(pick_runner.next_state(False))


class Reading(unittest.TestCase):
    def table(self, llm, n=18, complete=True, switches=2):
        import pandas as pd
        return pd.DataFrame({"llm_mean_z": llm, "cascade_mean_z": 0.0, "low_top_mean_z": 0.2, "random_median": -0.3,
                             "llm_n_placed": n, "cascade_n_placed": 18, "complete": complete,
                             "switches": switches, "llm_decisions": 7})

    def test_rule(self):
        r = llm_endgame_pick.reading
        self.assertEqual(r(self.table([0.5] * 7 + [-0.1] * 2)), "go")
        self.assertEqual(r(self.table([0.1] * 9)), "no endgame contribution shown")  # (b): below low_top
        self.assertEqual(r(self.table([0.5] * 9, switches=1)), "no endgame contribution shown")  # (c): 9 of 63
        self.assertEqual(r(self.table([0.5] * 9, complete=[True] * 6 + [False] * 3)), "no endgame contribution shown")
        self.assertEqual(r(self.table([-0.5] * 5 + [0.5] * 4)), "stop direct pick")
        self.assertEqual(r(self.table([0.5] * 9, n=[17] * 3 + [18] * 6)), "stop direct pick")


VENV = controller_worker.VENV_MANIFEST.exists() and not controller_worker.venv_problems()
OPENAI = {"kind": "api", "provider": "openai", "model": gpt_smoke.MODEL, "max_tokens": 4096, "effort": "low",
          "request_timeout_s": 60.0, "max_retries": 0, "retry_backoff_s": 0.0, "price_per_mtok": gpt_smoke.PRICE}
LOOPBACK_SPEC = {"kind": "pick", "llm": OPENAI,
                 "llm_timeout": 90.0, "max_tool_calls": 3, "max_decisions": 10, "max_total_tokens": 100_000,
                 "max_cost_usd": 0.30, "max_llm_failures": 1}
LOOKAHEAD_SPEC = {**LOOPBACK_SPEC, "max_tool_calls": 4, "lookahead": True}


@unittest.skipUnless(VENV and Path(sys.prefix).resolve() == run_wall.ENV_PREFIX, "needs the baseline env and the venv")
class LoopbackWall(unittest.TestCase):
    """A whole TH_T80 wall: Wang's process -> venv worker -> real openai SDK -> a mock Responses API on 127.0.0.1."""

    def test_a_loopback_pick_wall_passes(self):
        test_gpt_smoke.isolate(self)
        server = test_gpt_smoke.Server()
        self.addCleanup(server.shutdown)
        env = {**test_gpt_smoke.CLEAN_ENV, "OPENAI_API_KEY": test_gpt_smoke.DUMMY, "OPENAI_BASE_URL": server.url}
        with mock.patch.dict(os.environ, env, clear=True):
            receipt = run_wall.run(["thangavelu-noise80", "--seed", "313", "--policy", "paper", "--order", "3,0,2,1,4",
                                    "--worker-venv", "--controller", json.dumps(LOOPBACK_SPEC)])
        report = receipt["trace_check"]["controller"][0]
        self.assertEqual((receipt["status"], report["verdict"]), ("PROCESS_PASS", "PASS"), receipt["trace_check"])
        self.assertEqual(len(server.requests), report["decision_points"])  # one request per decision
        self.assertGreater(report["decision_points"], 0)
        self.assertTrue(all(r["body"]["tools"][0]["name"] == "commit_pick" for r in server.requests))
        self.assertNotIn(llm_api.PAID_GATE, os.environ)

    def test_a_loopback_lookahead_wall_passes(self):
        """check_after then commit_pick at every decision (the mock server's script), checked against the wall."""
        test_gpt_smoke.isolate(self)
        server = test_gpt_smoke.Server()
        self.addCleanup(server.shutdown)
        env = {**test_gpt_smoke.CLEAN_ENV, "OPENAI_API_KEY": test_gpt_smoke.DUMMY, "OPENAI_BASE_URL": server.url}
        with mock.patch.dict(os.environ, env, clear=True):
            receipt = run_wall.run(["thangavelu-noise80", "--seed", "313", "--policy", "paper", "--order", "3,0,2,1,4",
                                    "--worker-venv", "--controller", json.dumps(LOOKAHEAD_SPEC)])
        report = receipt["trace_check"]["controller"][0]
        self.assertEqual((receipt["status"], report["verdict"]), ("PROCESS_PASS", "PASS"), receipt["trace_check"])
        self.assertEqual(len(server.requests), 2 * report["decision_points"])  # check_after, then commit_pick
        self.assertTrue(all([t["name"] for t in r["body"]["tools"]] == ["check_after", "commit_pick"]
                            for r in server.requests))


if __name__ == "__main__":
    unittest.main()
