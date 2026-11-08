"""Checks for research/perf/verify_speedup.py and research/runner/verify_speedup.py: run_wall.run and the checks are mocked, no wall is ever built.

Run from the repository root: python -B -m unittest research.tests.test_verify_speedup
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from research.paths import LEGACY_RUNS, REPO
from research.perf import profile_report, verify_speedup
from research.storage import case_id
from research.runner import environment as runner_environment
from research.walls import case_definition
from research.runner import run_wall
from research.runner import verify_speedup as speedup_runner
from research.walls import check_run

LABEL = "test-label"
ONE_CELL = ("--scenes", "fft-d2-0", "--policies", "paper", "--placements", "3")
FULL_CODE_ORIGINAL = ("--scenes", "bricks", "--policies", "code_original")
DIFFERENT = ["decisions.txt differs"]
SECOND_ROW = {
    "scene": "fft-d2-0",
    "policy": "paper",
    "mode": "persistent",
    "repeat": 1,
    "case_id": "case-02",
    "status": "PROCESS_PASS",
    "elapsed_seconds": 102.0,
    "placed": 7,
    "seconds_per_stone": 1.5,
    "trace_ok": True,
    "differences": DIFFERENT,
    "check_ok": True,
    "compared_to": "case-01",
    "error": None,
}


class ParserAccepted(Exception):
    """Raised where run_wall.run would start allocating a wall, right after its parser accepted the argv."""


def fake_receipt(argv, number, **changes):
    """What run_wall.run returns for `argv`: the pool follows the flag, and `changes` override any field."""
    return {
        "case_id": f"case-{number:02d}",
        "status": "PROCESS_PASS",
        "elapsed_seconds": 100.0 + number,
        "committed_placements": 7,
        "trace_check": {"ok": True},
        "worker_pool": "per_batch" if "--fresh-pool" in argv else "persistent",
        **changes,
    }


def summary_row(cell, seconds, **fields):
    """A row of the summary's input: `cell` is (scene, policy, mode); `fields` override the clean defaults."""
    scene, policy, mode = cell
    clean = {"differences": [], "archive_differences": None}
    return {"scene": scene, "policy": policy, "mode": mode, "seconds_per_stone": seconds, **clean, **fields}


class DriverTestCase(unittest.TestCase):
    """main() with a temporary output root; `outcomes` scripts run_wall.run by call number."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.folder = self.root / LABEL
        self.calls, self.outcomes, self.during_call = [], {}, lambda number: None
        self.prepare = self.patch(speedup_runner, "prepare_environment")
        self.seconds = self.patch(profile_report, "seconds_per_stone", return_value=1.5)
        self.compare = self.patch(check_run, "compare_runs", return_value=[])
        self.patch(verify_speedup, "stopped_differences", self.compare)  # a stopped wall has its own comparison
        self.check = self.patch(check_run, "check_run", return_value={"ok": True})
        self.patch(speedup_runner, "SPEEDUP_RUNS", self.root)
        self.patch(run_wall, "run", self.fake_run)

    def patch(self, target, name, *args, **kwargs):
        patcher = mock.patch.object(target, name, *args, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def fake_run(self, argv):
        self.calls.append(list(argv))
        number = len(self.calls)
        self.during_call(number)
        outcome = self.outcomes.get(number, {})
        if isinstance(outcome, BaseException):
            raise outcome
        return fake_receipt(argv, number, **outcome)

    def drive(self, *options):
        return speedup_runner.main([LABEL, *options])

    def report(self):
        return json.loads((self.folder / "report.json").read_text(encoding="utf-8"))

    def rows(self):
        return self.report()["rows"]

    def compared_pairs(self):
        return [call.args for call in self.compare.call_args_list]


class Arguments(unittest.TestCase):
    def test_defaults(self):
        options = speedup_runner.parse_args([LABEL])
        self.assertEqual(options.scenes, list(case_definition.CASES))
        self.assertEqual(options.policies, ["paper", "code_original"])
        self.assertEqual(options.modes, ["fresh", "persistent"])
        self.assertEqual((options.placements, options.repeat, options.workers, options.timeout), (0, 1, 10, 3600))

    def test_lists_are_split_on_commas(self):
        options = speedup_runner.parse_args([LABEL, "--scenes", "bricks,rubble", "--modes", "persistent,fresh"])
        self.assertEqual((options.scenes, options.modes), (["bricks", "rubble"], ["persistent", "fresh"]))

    def test_an_unknown_name_is_refused_before_any_wall(self):
        for flag, value in (("--scenes", "nowhere"), ("--policies", "bricks"), ("--modes", "lazy")):
            with self.subTest(flag), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                speedup_runner.parse_args([LABEL, flag, value])

    def test_argv_of_each_mode_and_policy(self):
        tail = ["--placements", "3", "--timeout", "60"]
        paper = ["--policy", "paper", "--order", "3,0,2,1,4"]
        code_original = ["--policy", "code_original"]
        cases = (
            ("paper", "fresh", [*paper, *tail, "--fresh-pool"]),
            ("paper", "persistent", [*paper, *tail]),
            ("code_original", "fresh", [*code_original, *tail, "--fresh-pool"]),
            ("code_original", "persistent", [*code_original, *tail]),
        )
        for policy, mode, expected in cases:
            with self.subTest(policy=policy, mode=mode):
                entry = {"scene": "bricks", "policy": policy, "mode": mode}
                self.assertEqual(verify_speedup.wall_argv(entry, 3, 60), ["bricks", *expected])

    def test_run_wall_parses_the_argv_of_each_mode_and_policy(self):
        policies, modes = list(verify_speedup.POLICIES), list(verify_speedup.MODE_FLAGS)
        environment = mock.patch.object(run_wall, "ENV_PREFIX", Path(sys.prefix).resolve())
        stop = mock.patch.object(run_wall, "prepare_output", side_effect=ParserAccepted)
        with environment, stop:
            for entry in verify_speedup.planned_walls(["bricks"], policies, modes, 1):
                with self.subTest(entry), self.assertRaises(ParserAccepted):
                    run_wall.run(verify_speedup.wall_argv(entry, 3, 60))


class Workers(unittest.TestCase):
    def test_the_call_is_replaced(self):
        staged = speedup_runner.with_workers(lambda case: f"{case}\nset_number_cpu(10)\n", 4)
        self.assertEqual(staged("bricks"), "bricks\nset_number_cpu(4)\n")

    def test_a_script_without_the_call_is_refused(self):
        with self.assertRaises(AssertionError):
            speedup_runner.with_workers(lambda case: "set_number_cpu(12)\n", 4)("bricks")

    def test_every_scene_holds_the_call(self):
        for case in case_definition.CASES:
            with self.subTest(case):
                self.assertIn("set_number_cpu(4)", speedup_runner.with_workers(case_definition.source_text, 4)(case))


class RunOrder(DriverTestCase):
    def test_two_repeats_alternate_the_modes(self):
        self.drive(*ONE_CELL, "--repeat", "2")
        self.assertEqual(["--fresh-pool" in argv for argv in self.calls], [True, False, True, False])
        self.assertEqual(
            [(r["mode"], r["repeat"]) for r in self.rows()],
            [("fresh", 1), ("persistent", 1), ("fresh", 2), ("persistent", 2)],
        )

    def test_the_given_mode_order_is_kept(self):
        self.drive(*ONE_CELL, "--modes", "persistent,fresh")
        self.assertEqual([r["mode"] for r in self.rows()], ["persistent", "fresh"])

    def test_every_repeat_of_a_scene_and_policy_comes_before_the_next_one(self):
        planned = verify_speedup.planned_walls(["a", "b"], ["p", "q"], ["fresh", "persistent"], 2)
        self.assertEqual(len(planned), 16)
        self.assertEqual(
            [tuple(e.values()) for e in planned[:5]],
            [
                ("a", "p", "fresh", 1),
                ("a", "p", "persistent", 1),
                ("a", "p", "fresh", 2),
                ("a", "p", "persistent", 2),
                ("a", "q", "fresh", 1),
            ],
        )
        self.assertEqual(tuple(planned[8].values()), ("b", "p", "fresh", 1))

    def test_walls_go_to_the_label_folder_with_the_requested_workers(self):
        seen = []
        self.patch(case_definition, "source_text", lambda case: "set_number_cpu(10)")
        self.during_call = lambda number: seen.append((case_id.CASES_DIR, case_definition.source_text("fft-d2-0")))
        self.drive(*ONE_CELL, "--modes", "persistent", "--workers", "4")
        self.assertEqual(seen, [(self.folder, "set_number_cpu(4)")])

    def test_the_environment_is_prepared_in_main_before_the_first_wall(self):
        prepared = []
        self.during_call = lambda number: prepared.append(self.prepare.called)
        self.drive(*ONE_CELL)
        self.assertEqual(prepared, [True, True])
        self.prepare.assert_called_once_with(REPO / ".env.local", self.folder / "tmp")
        self.assertTrue((self.folder / "tmp").is_dir())


class Rows(DriverTestCase):
    def test_a_row_holds_the_receipt_and_the_checks(self):
        self.compare.return_value = DIFFERENT
        self.drive(*ONE_CELL)
        first, second = self.rows()
        self.assertEqual(first["differences"], [])
        self.assertIsNone(first["compared_to"])
        self.assertEqual({key: second[key] for key in SECOND_ROW}, SECOND_ROW)

    def test_every_wall_is_compared_with_the_first_of_its_scene_and_policy(self):
        self.drive(*ONE_CELL, "--repeat", "2")
        first = self.folder / "case-01"
        self.assertEqual(self.compared_pairs(), [(self.folder / f"case-0{n}", first) for n in (2, 3, 4)])

    def test_each_scene_and_policy_has_its_own_first_wall(self):
        self.drive("--scenes", "fft-d2-0", "--policies", "paper,code_original", "--placements", "3")
        self.assertEqual(
            self.compared_pairs(),
            [(self.folder / "case-02", self.folder / "case-01"), (self.folder / "case-04", self.folder / "case-03")],
        )

    def test_check_run_and_seconds_read_the_case_folder(self):
        self.drive(*ONE_CELL, "--modes", "persistent")
        self.check.assert_called_once_with(self.folder / "case-01")
        self.seconds.assert_called_once_with(self.folder / "case-01")

    def test_a_stopped_wall_has_no_seconds_per_stone(self):
        self.seconds.return_value = None
        self.outcomes[1] = {"status": "STOPPED_AFTER_PLACEMENTS"}
        self.drive(*ONE_CELL, "--modes", "persistent")
        self.assertIsNone(self.rows()[0]["seconds_per_stone"])
        self.assertIsNone(self.rows()[0]["error"])


class Archive(DriverTestCase):
    def archived(self):
        return [pair for pair in self.compared_pairs() if pair[1].parent == LEGACY_RUNS]

    def test_a_full_persistent_code_original_wall_is_compared_with_the_archive(self):
        self.compare.side_effect = lambda new, old: DIFFERENT if old.parent == LEGACY_RUNS else []
        self.drive(*FULL_CODE_ORIGINAL)
        self.assertEqual(self.archived(), [(self.folder / "case-02", LEGACY_RUNS / "bricks-code-original-trace-02")])
        self.assertEqual(
            [(r["differences"], r["archive_differences"]) for r in self.rows()], [([], None), ([], DIFFERENT)]
        )

    def test_every_persistent_wall_of_a_repeat_is_compared(self):
        self.drive(*FULL_CODE_ORIGINAL, "--repeat", "2")
        self.assertEqual([pair[0].name for pair in self.archived()], ["case-02", "case-04"])

    def test_the_folder_of_each_scene(self):
        folders = {
            "thangavelu-noise80": "thangavelu-noise80-code-original-full-01",
            "bricks": "bricks-code-original-trace-02",
            "almeida-pr": "pr-code-original-full-01",
            "fft-d2-0": "fft-d2-0-code-original-notrace-01",
        }
        for scene, name in folders.items():
            with self.subTest(scene):
                row = {"scene": scene, "policy": "code_original", "mode": "persistent"}
                self.assertEqual(verify_speedup.archive_of(row, 0), LEGACY_RUNS / name)

    def test_no_archive_comparison_otherwise(self):
        row = {"scene": "bricks", "policy": "code_original", "mode": "persistent"}
        cases = (({"mode": "fresh"}, 0), ({"policy": "paper"}, 0), ({"scene": "rubble"}, 0), ({}, 3))
        for change, placements in cases:
            with self.subTest(change=change, placements=placements):
                self.assertIsNone(verify_speedup.archive_of({**row, **change}, placements))


class Failures(DriverTestCase):
    def test_a_wall_that_raises_does_not_stop_the_rest(self):
        self.outcomes[2] = RuntimeError("the case exploded")
        code = self.drive(*ONE_CELL, "--repeat", "2")
        rows = self.rows()
        self.assertEqual((len(self.calls), len(rows), code), (4, 4, 1))
        self.assertIn("RuntimeError: the case exploded", rows[1]["error"])
        self.assertEqual((rows[1]["status"], rows[1]["case_id"]), ("RAISED", None))
        self.assertEqual(rows[1]["differences"], [verify_speedup.NOT_COMPARED])
        self.assertIsNone(rows[2]["error"])
        self.assertTrue((self.folder / "done.txt").exists())

    def test_an_argv_refused_by_run_wall_does_not_stop_the_rest(self):
        self.outcomes[1] = SystemExit(2)
        self.drive(*ONE_CELL)
        self.assertEqual(len(self.calls), 2)
        self.assertIn("SystemExit", self.rows()[0]["error"])

    def test_keyboard_interrupt_stops_and_keeps_the_rows_so_far(self):
        self.outcomes[2] = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.drive(*ONE_CELL, "--repeat", "2")
        self.assertEqual((len(self.calls), len(self.rows())), (2, 1))
        self.assertFalse((self.folder / "done.txt").exists())

    def test_a_process_tree_not_stopped_is_recorded_from_its_receipt(self):
        receipt = fake_receipt(["bricks"], 1, status="TERMINATION_FAILED")
        self.outcomes[1] = run_wall.TerminationFailed(receipt)
        self.drive(*ONE_CELL)
        first = self.rows()[0]
        self.assertEqual((first["case_id"], first["status"], len(self.calls)), ("case-01", "TERMINATION_FAILED", 2))
        self.assertIn("TerminationFailed", first["error"])
        self.assertEqual(first["differences"], [verify_speedup.NOT_COMPARED])

    def test_a_failed_check_is_recorded_and_keeps_what_was_measured(self):
        self.check.side_effect = OSError("base.pkl is gone")
        self.drive(*ONE_CELL, "--modes", "persistent")
        row = self.rows()[0]
        self.assertIn("base.pkl is gone", row["error"])
        self.assertEqual((row["case_id"], row["seconds_per_stone"], row["differences"]), ("case-01", 1.5, []))

    def test_a_receipt_that_contradicts_the_mode_is_reported(self):
        self.outcomes[2] = {"worker_pool": "per_batch"}
        code = self.drive(*ONE_CELL)
        row = self.rows()[1]
        self.assertEqual(code, 1)
        self.assertIn("worker_pool 'per_batch' contradicts mode persistent", row["error"])
        self.assertEqual(row["differences"], [verify_speedup.NOT_COMPARED])
        self.assertIsNone(row["seconds_per_stone"])
        self.compare.assert_not_called()

    def test_a_receipt_without_the_field_is_reported(self):
        self.outcomes[1] = {"worker_pool": None}
        self.drive(*ONE_CELL, "--modes", "fresh")
        self.assertIn("contradicts mode fresh", self.rows()[0]["error"])

    def test_a_wall_that_did_not_succeed_is_not_compared_and_is_not_the_first(self):
        self.outcomes[1] = {"status": "TIMEOUT", "trace_check": {"ok": False}}
        self.drive(*ONE_CELL, "--repeat", "2")
        rows = self.rows()
        self.assertEqual((rows[0]["error"], rows[0]["trace_ok"]), ("status TIMEOUT", False))
        self.assertEqual(rows[0]["differences"], [verify_speedup.NOT_COMPARED])
        self.assertEqual((rows[1]["differences"], rows[1]["compared_to"]), ([], None))
        self.assertEqual(self.compared_pairs(), [(self.folder / f"case-0{n}", self.folder / "case-02") for n in (3, 4)])


class ExitCode(DriverTestCase):
    def test_zero_when_every_wall_is_identical_and_passes_its_check(self):
        self.assertEqual(self.drive(*ONE_CELL), 0)

    def test_one_with_a_difference(self):
        self.compare.return_value = DIFFERENT
        self.assertEqual(self.drive(*ONE_CELL), 1)

    def test_one_with_a_failed_check(self):
        self.check.return_value = {"ok": False}
        self.assertEqual(self.drive(*ONE_CELL), 1)

    def test_one_when_a_check_raised(self):
        self.check.side_effect = OSError("base.pkl is gone")
        self.assertEqual(self.drive(*ONE_CELL), 1)

    def test_one_with_a_difference_from_the_archive(self):
        self.compare.side_effect = lambda new, old: DIFFERENT if old.parent == LEGACY_RUNS else []
        self.assertEqual(self.drive(*FULL_CODE_ORIGINAL), 1)

    def test_one_when_no_wall_was_compared(self):
        self.assertEqual(self.drive(*ONE_CELL, "--modes", "persistent"), 1)

    def test_an_archive_without_a_trace_is_only_missing_its_commits(self):
        archive = LEGACY_RUNS / "fft-d2-0-code-original-notrace-01"
        self.compare.return_value = [verify_speedup.UNTRACED_ARCHIVE]
        self.assertEqual(verify_speedup.archive_differences("wall", archive), [])
        self.compare.return_value = [verify_speedup.UNTRACED_ARCHIVE, "decisions.txt differs"]
        self.assertEqual(verify_speedup.archive_differences("wall", archive), ["decisions.txt differs"])

    def test_a_traced_archive_must_have_the_commits(self):
        self.compare.return_value = [verify_speedup.UNTRACED_ARCHIVE]
        archive = LEGACY_RUNS / "bricks-code-original-trace-02"
        self.assertEqual(verify_speedup.archive_differences("wall", archive), [verify_speedup.UNTRACED_ARCHIVE])


class ReportFile(DriverTestCase):
    def test_the_report_is_written_after_every_wall(self):
        seen = []
        self.during_call = lambda number: seen.append(len(self.rows()) if number > 1 else None)
        self.drive(*ONE_CELL, "--repeat", "2")
        self.assertEqual(seen, [None, 1, 2, 3])
        self.assertEqual(len(self.rows()), 4)

    def test_a_partial_report_already_has_a_summary(self):
        seen = []
        self.during_call = lambda number: seen.append(self.report()["summary"]) if number == 2 else None
        self.drive(*ONE_CELL)
        self.assertEqual(seen[0]["fft-d2-0"]["paper"]["worst_seconds_per_stone"], {"fresh": 1.5})

    def test_the_summary_ends_the_report(self):
        self.drive(*ONE_CELL)
        self.assertEqual(list(self.report()), ["settings", "rows", "summary"])

    def test_done_is_written_last(self):
        seen = []
        self.during_call = lambda number: seen.append((self.folder / "done.txt").exists())
        self.drive(*ONE_CELL)
        self.assertEqual(seen, [False, False])
        self.assertTrue((self.folder / "done.txt").exists())

    def test_a_done_file_of_an_earlier_run_is_removed_first(self):
        self.folder.mkdir()
        (self.folder / "done.txt").write_text("done\n", encoding="utf-8")
        seen = []
        self.during_call = lambda number: seen.append((self.folder / "done.txt").exists())
        self.drive(*ONE_CELL, "--modes", "persistent")
        self.assertEqual(seen, [False])

    def test_a_file_held_open_does_not_end_the_run(self):
        with mock.patch.object(speedup_runner.os, "replace", side_effect=PermissionError("in use")):
            self.drive(*ONE_CELL)
        self.assertEqual(len(self.rows()), 2)


class Summary(unittest.TestCase):
    def test_worst_seconds_per_mode_and_counts(self):
        rows = [
            summary_row(("bricks", "paper", "fresh"), 3.0),
            summary_row(("bricks", "paper", "persistent"), 1.0),
            summary_row(("bricks", "paper", "fresh"), 4.0),
            summary_row(("bricks", "paper", "persistent"), None, differences=DIFFERENT),
            summary_row(("bricks", "code_original", "persistent"), 2.0, archive_differences=DIFFERENT),
            summary_row(("rubble", "paper", "fresh"), None),
        ]
        self.assertEqual(
            verify_speedup.summarize(rows),
            {
                "bricks": {
                    "paper": {
                        "worst_seconds_per_stone": {"fresh": 4.0, "persistent": 1.0},
                        "rows_with_differences": 1,
                        "rows_with_archive_differences": 0,
                    },
                    "code_original": {
                        "worst_seconds_per_stone": {"persistent": 2.0},
                        "rows_with_differences": 0,
                        "rows_with_archive_differences": 1,
                    },
                },
                "rubble": {
                    "paper": {
                        "worst_seconds_per_stone": {"fresh": None},
                        "rows_with_differences": 0,
                        "rows_with_archive_differences": 0,
                    }
                },
            },
        )

    def test_a_wall_that_was_not_compared_counts_as_different(self):
        row = verify_speedup.blank_row({"scene": "bricks", "policy": "paper", "mode": "fresh", "repeat": 1})
        self.assertEqual(verify_speedup.summarize([row])["bricks"]["paper"]["rows_with_differences"], 1)


class StoppedWalls(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(check_run, "load_result", return_value=({}, None))
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def wall(root, name, decisions, filters, commits=(1,)):
        result = Path(root) / name / "result_x"
        (result / "trace").mkdir(parents=True)
        for path, rows in (("decisions.txt", decisions), ("filter.txt", filters), ("trace/commits.jsonl", commits)):
            (result / path).write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")
        return result.parent

    def test_a_record_without_rows_in_one_wall_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = self.wall(tmp, "a", [1], []), self.wall(tmp, "b", [1], [1])
            self.assertEqual(verify_speedup.stopped_differences(a, b), ["filter.txt: no rows in one of the walls"])

    def test_a_missing_commits_file_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = self.wall(tmp, "a", [1], [1]), self.wall(tmp, "b", [1], [1])
            (b / "result_x" / "trace" / "commits.jsonl").unlink()
            self.assertEqual(
                verify_speedup.stopped_differences(a, b), ["trace/commits.jsonl: no rows in one of the walls"]
            )

    def test_a_wall_that_ran_a_little_longer_is_the_same_wall(self):
        with tempfile.TemporaryDirectory() as tmp:
            short, long = self.wall(tmp, "a", [1, 2, 3], [1, 2]), self.wall(tmp, "b", [1, 2, 3, 4], [1, 2, 3])
            self.assertEqual(verify_speedup.stopped_differences(short, long), [])
            self.assertEqual(verify_speedup.stopped_differences(long, short), [])

    def test_a_different_common_row_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = self.wall(tmp, "a", [1, 2, 3], [1]), self.wall(tmp, "b", [1, 9, 3, 4], [1])
            self.assertEqual(
                verify_speedup.stopped_differences(a, b), ["decisions.txt: the rows both walls have differ"]
            )

    def test_stopped_walls_use_the_common_rows_and_full_walls_every_byte(self):
        with mock.patch.object(verify_speedup, "stopped_differences", return_value=["s"]) as stopped, mock.patch.object(
            check_run, "compare_runs", return_value=["f"]
        ), mock.patch.object(verify_speedup, "limit_load_differences", return_value=["l"]):
            self.assertEqual(verify_speedup.wall_differences("a", "b", 3), ["s"])
            self.assertEqual(verify_speedup.wall_differences("a", "b", 0), ["f", "l"])
        stopped.assert_called_once_with("a", "b")


class LimitLoads(unittest.TestCase):
    def test_the_last_bits_of_the_solver_are_allowed(self):
        self.assertTrue(verify_speedup.same_limit_load(0.3203711097007615, 0.32037110970076155))

    def test_a_real_difference_is_not(self):
        self.assertFalse(verify_speedup.same_limit_load(0.32, 0.3200001))

    def test_a_side_without_a_value_only_equals_another_without_one(self):
        self.assertTrue(verify_speedup.same_limit_load(None, None))
        self.assertFalse(verify_speedup.same_limit_load(None, 0.3))


class Environment(unittest.TestCase):
    ENV_FILE = "\n".join(
        [
            "OPENAI_API_KEY=placeholder-one",
            "OTHER_SETTING=placeholder-two",
            'MOSEKLM_LICENSE_FILE="licenses/mosek.lic"',
            "PYTHONNOUSERSITE=1",
        ]
    )

    def prepared(self, text):
        """os.environ after prepare_environment, with every key it must drop present beforehand."""
        before = {key: "set" for key in runner_environment.DROPPED_KEYS}
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(os.environ, before):
            env_file, tmp = Path(folder) / ".env.local", Path(folder) / "tmp"
            env_file.write_text(text, encoding="utf-8")
            speedup_runner.prepare_environment(env_file, tmp)
            return dict(os.environ), tmp

    def test_the_paid_keys_are_removed_and_not_taken_back_from_the_file(self):
        environment, _ = self.prepared(self.ENV_FILE)
        self.assertFalse(set(runner_environment.DROPPED_KEYS) & set(environment))

    def test_only_two_values_are_taken_from_the_file(self):
        environment, _ = self.prepared(self.ENV_FILE)
        self.assertNotIn("OTHER_SETTING", environment)
        self.assertEqual(environment["MOSEKLM_LICENSE_FILE"], str(REPO / "licenses/mosek.lic"))
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")

    def test_an_absolute_license_path_is_kept(self):
        absolute = str(Path(tempfile.gettempdir()) / "mosek.lic")
        environment, _ = self.prepared(f"MOSEKLM_LICENSE_FILE={absolute}\n")
        self.assertEqual(environment["MOSEKLM_LICENSE_FILE"], absolute)

    def test_temp_files_go_to_the_given_folder(self):
        environment, tmp = self.prepared(self.ENV_FILE)
        self.assertEqual((environment["TEMP"], environment["TMP"]), (str(tmp), str(tmp)))


if __name__ == "__main__":
    unittest.main()
