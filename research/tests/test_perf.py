"""Checks for research/perf: staging, the report functions and the timer wrappers (no case is run).

Run from the repository root: python -B -m unittest research.tests.test_perf
"""

import ast
import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from research.paths import REPO, VALIDATION
from research.perf import profile_case, profile_report
from research.walls import run_wall

PERF_RUNTIME = REPO / "research" / "perf" / "perf_runtime.py"
FILTER = {"policy": "paper", "order": (3, 0, 2, 1, 4), "tol": 1e-9}


def load_perf_runtime():
    """perf_runtime as a staged case sees it: PERF_DIR set and the registered runtime importable by name."""
    os.environ.setdefault("PERF_DIR", tempfile.mkdtemp())
    sys.path.insert(0, str(VALIDATION))
    try:
        spec = importlib.util.spec_from_file_location("perf_runtime_under_test", PERF_RUNTIME)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(VALIDATION))
    return module


class Staging(unittest.TestCase):
    def test_only_the_runtime_import_changes(self):
        for case, (script, _) in run_wall.CASES.items():
            source = (REPO / script).read_text(encoding="utf-8")
            policy = {"filter": FILTER, "run_id": None, "controller": None, "worker_venv": False}
            registered = run_wall.stage(source, policy, seed=7)
            perf = profile_case.stage_perf(source, FILTER, 7)
            self.assertEqual(
                perf, registered.replace(profile_case.RUNTIME_IMPORT, "from perf_runtime import configure", 1), case
            )
            self.assertNotEqual(perf, registered, case)


class RunFolders(unittest.TestCase):
    def test_a_timing_run_gets_a_new_folder_of_its_own(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(profile_case, "PERF_RUNS", Path(tmp) / "runs"):
            output = profile_case.prepare_output("a")
            self.assertEqual(output, (Path(tmp) / "runs" / "a").resolve())
            self.assertTrue(output.is_dir())
            with self.assertRaises(FileExistsError):
                profile_case.prepare_output("a")

    def test_the_report_finds_a_path_a_timing_run_or_an_archived_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs, archive = Path(tmp) / "runs", Path(tmp) / "archive"
            (runs / "new").mkdir(parents=True)
            with mock.patch.object(profile_report, "PERF_RUNS", runs), mock.patch.object(
                profile_report, "LEGACY_RUNS", archive
            ):
                self.assertEqual(profile_report.resolve_run(tmp), Path(tmp))
                self.assertEqual(profile_report.resolve_run("new"), runs / "new")
                self.assertEqual(profile_report.resolve_run("old"), archive / "old")


class Report(unittest.TestCase):
    def test_phases_cover_launch_to_exit(self):
        events = {"build_enter": 20.0, "build_return": 120.0, "final_evaluation_complete": 125.0}
        rows = profile_report.phase_rows(events, {"configure_epoch": 1004.0}, 1000.0, 130.0)
        self.assertEqual([label for label, _ in rows][-1], "exit")
        self.assertAlmostEqual(sum(seconds for _, seconds in rows), 130.0)
        self.assertAlmostEqual(dict(rows)["build_wall_vendor"], 100.0)

    def test_phase_without_closing_event_is_left_out(self):
        rows = profile_report.phase_rows({"build_enter": 20.0}, {"configure_epoch": 1004.0}, 1000.0, 60.0)
        self.assertEqual(
            [label for label, _ in rows],
            ["interpreter and imports", "stone loading, shape factors, clustering", "exit"],
        )

    def test_main_rows_skip_the_task_counter(self):
        stats = {"main.pool.starmap": [5.0, 2], "main.pool.tasks": [8, 2], "worker.x.y": [9.0, 1], "main.a.b": [7.0, 3]}
        self.assertEqual(profile_report.main_rows(stats), [("main.a.b", 7.0, 3), ("main.pool.starmap", 5.0, 2)])

    def test_worker_sections_sum_over_calls(self):
        records = [
            {"total": 1.0, "sections": {"worker.k": [0.5, 1]}},
            {"total": 2.0, "sections": {"worker.k": [1.0, 2]}},
        ]
        self.assertEqual(profile_report.worker_sections(records), {"worker.k": [1.5, 3]})
        self.assertAlmostEqual(profile_report.average_busy_workers(records, 1.5), 2.0)


class TimerWrappers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = load_perf_runtime()

    def test_every_timer_target_exists(self):
        tree = ast.parse(PERF_RUNTIME.read_text(encoding="utf-8"))
        targets = [
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in ("SHARED_TARGETS", "MAIN_TARGETS")
        ]
        missing = []
        for module_name, owner_name, attributes in (t for group in targets for t in group):
            module = importlib.import_module(module_name)
            owner = getattr(module, owner_name) if owner_name else module
            missing += [f"{module_name}.{owner_name}.{a}" for a in attributes if not hasattr(owner, a)]
        self.assertEqual(missing, [])

    def test_wrapper_passes_result_arguments_and_errors_through(self):
        calls = []
        wrapped = self.runtime.timed("test.wrapper", lambda *a, **k: calls.append((a, k)) or "result")
        self.assertEqual(wrapped(1, 2, key=3), "result")
        self.assertEqual(calls, [((1, 2), {"key": 3})])

        def fail():
            raise KeyError("x")

        with self.assertRaises(KeyError):
            self.runtime.timed("test.fail", fail)()
        self.assertEqual(self.runtime.STATS["test.fail"][1], 1)

    def test_install_timers_wraps_the_attribute_in_place(self):
        module = types.ModuleType("perf_dummy")
        module.f = lambda x: x + 1
        sys.modules["perf_dummy"] = module
        try:
            self.runtime.install_timers("t.", (("perf_dummy", None, ("f",)),))
            self.assertEqual(module.f(1), 2)
            self.assertEqual(self.runtime.STATS["t.perf_dummy.f"][1], 1)
        finally:
            del sys.modules["perf_dummy"]

    def test_run_task_calls_with_the_given_arguments(self):
        constants = {k: v for k, v in self.runtime.settings().items() if k == "MU"}
        self.assertEqual(self.runtime.run_task(constants, lambda a, b: a * b, (3, 4)), 12)


if __name__ == "__main__":
    unittest.main()
