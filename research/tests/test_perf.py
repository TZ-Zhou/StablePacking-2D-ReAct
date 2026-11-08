"""Checks for research/perf: staging, the report functions and the timer wrappers (no case is run).

Run from the repository root: python -B -m unittest research.tests.test_perf
"""

import ast
import concurrent.futures
import importlib
import json
import os
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from research.paths import REPO, VALIDATION
from research.perf import profile_report
from research.runner import profile_case
from research.walls import case_definition

PERF_RUNTIME = REPO / "research" / "perf" / "perf_runtime.py"
FILTER = {"policy": "paper", "order": (3, 0, 2, 1, 4), "tol": 1e-9}
MB = 1024**2


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


def step_tasks(poses):
    """The tasks of one step as build_wall makes them: (pose, *arguments shared by the step)."""
    current_bricks, all_bricks, base = ["stone"], list(range(poses)), {"placed": [1, 2, 3]}
    return [(pose, current_bricks, all_bricks, base) for pose in range(poses)]


class FakeExecutor:
    """Takes the place of the ProcessPoolExecutor: nothing runs and every submission is counted."""

    def __init__(self):
        self.submitted = 0

    def submit(self, *call):
        self.submitted += 1
        future = concurrent.futures.Future()
        future.set_result("result")
        return future


def write_run(folder, events):
    """The files of a profile_case run that format_report reads."""
    perf = folder / "perf"
    perf.mkdir()
    record = {"case": "c", "wall_seconds": 30.0, "persistent_pool": True, "launch_epoch": 1000.0}
    (folder / "perf.json").write_text(json.dumps(record), encoding="utf-8")
    (folder / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    stats = {
        "main.pool.starmap": [1.0, 1],
        "main.pool.tasks": [4, 1],
        "main.payload.bytes": [8 * MB, 1],
        "main.payload.pickle": [0.25, 1],
    }
    stats_file = {"meta": {"configure_epoch": 1001.0}, "stats": stats}
    (perf / "main_stats.json").write_text(json.dumps(stats_file), encoding="utf-8")
    place = {"pid": 1, "total": 1.0, "sections": {"worker.k": [0.5, 1]}, "t_end": 1.0, "peak_wset": 50 * MB}
    (perf / "place_1.jsonl").write_text(json.dumps(place) + "\n", encoding="utf-8")


class Staging(unittest.TestCase):
    def test_only_the_runtime_import_changes(self):
        for case, (script, _) in case_definition.CASES.items():
            source = (REPO / script).read_text(encoding="utf-8")
            policy = {"filter": FILTER, "run_id": None, "controller": None, "worker_venv": False}
            registered = case_definition.stage(source, policy, seed=7)
            perf = profile_case.stage_perf(source, FILTER, 7)
            self.assertEqual(
                perf, registered.replace(profile_case.RUNTIME_IMPORT, "from perf_runtime import configure", 1), case
            )
            self.assertNotEqual(perf, registered, case)

    def test_dump_tasks_reaches_the_case_environment(self):
        args = profile_case.parse_args(["fft-d2-0", "run", "--dump-tasks", "3"])
        self.assertEqual(profile_case.case_environment(Path("output"), 1, args)["PERF_DUMP_TASKS"], "3")


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

    def test_main_rows_skip_the_payload_bytes_and_keep_the_pickling_seconds(self):
        stats = {"main.payload.bytes": [9e7, 3], "main.payload.pickle": [0.5, 3], "main.pool.starmap": [5.0, 2]}
        self.assertEqual(
            profile_report.main_rows(stats), [("main.pool.starmap", 5.0, 2), ("main.payload.pickle", 0.5, 3)]
        )

    def test_payload_line_gives_mb_per_task_total_gb_and_pickling_seconds(self):
        stats = {
            "main.payload.bytes": [2.5 * 1024**3, 3],
            "main.payload.pickle": [1.5, 3],
            "main.pool.tasks": [256, 9],
        }
        self.assertEqual(
            profile_report.payload_line(stats), "payload: 10.00 MB per task, 2.50 GB total, 1.50 s pickling"
        )

    def test_payload_line_of_a_run_without_the_counters(self):
        self.assertEqual(profile_report.payload_line({"main.pool.tasks": [8, 2]}), "payload: not available")

    def test_memory_line_gives_the_worst_worker_at_the_first_and_last_records(self):
        peaks = [(1, 100), (2, 120), (1, 300), (2, 150)]
        records = [{"pid": pid, "peak_wset": megabytes * MB} for pid, megabytes in peaks]
        self.assertEqual(
            profile_report.memory_line(records),
            "worker peak working set (worst worker): first record 120.0 MB, last record 300.0 MB",
        )

    def test_memory_line_of_records_without_the_peak(self):
        self.assertEqual(profile_report.memory_line([{"pid": 1}]), "worker peak working set: not available")

    def test_the_report_opens_with_seconds_per_stone_and_prints_the_new_lines(self):
        events = [
            {"event": "build_enter", "elapsed_seconds": 1.0},
            {"event": "build_return", "elapsed_seconds": 11.0, "placed_rocks": 5},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            write_run(Path(tmp), events)
            lines = profile_report.format_report(Path(tmp), 5).splitlines()
        self.assertEqual(lines[1], "seconds per stone: 2.00")
        self.assertIn("payload: 2.00 MB per task, 0.01 GB total, 0.25 s pickling", lines)
        self.assertIn("worker peak working set (worst worker): first record 50.0 MB, last record 50.0 MB", lines)

    def test_the_report_of_a_wall_that_did_not_return(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_run(Path(tmp), [{"event": "build_enter", "elapsed_seconds": 1.0}])
            lines = profile_report.format_report(Path(tmp), 5).splitlines()
        self.assertEqual(lines[1], "seconds per stone: not available")

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
        registered = self.runtime.registered
        constants = {k: v for k, v in registered._constants().items() if k == "MU"}
        self.assertEqual(registered._run_task(constants, lambda a, b: a * b, (3, 4)), 12)

    def test_each_place_record_has_the_peak_working_set(self):
        place_stone = types.SimpleNamespace(place_one_stone=lambda *args: "result")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(self.runtime, "PERF_DIR", Path(tmp)):
            with mock.patch.object(self.runtime.importlib, "import_module", return_value=place_stone):
                self.runtime.wrap_place_one_stone()
            placed = types.SimpleNamespace(placed_bricks=[])
            self.assertEqual(place_stone.place_one_stone(0, None, None, placed), "result")
            lines = [line for path in Path(tmp).glob("place_*.jsonl") for line in path.read_text("utf-8").splitlines()]
        self.assertEqual(len(lines), 1)
        self.assertGreater(json.loads(lines[0])["peak_wset"], 0)


class Payload(unittest.TestCase):
    """The real pool classes and their counting, with the worker processes replaced by stand-ins."""

    @classmethod
    def setUpClass(cls):
        cls.runtime = load_perf_runtime()
        cls.shared_pool = cls.runtime.registered.SharedPool

    def setUp(self):
        self.runtime.STATS.clear()

    def assert_payload(self, total_bytes, calls):
        self.assertEqual(self.runtime.STATS["main.payload.bytes"], [total_bytes, calls])
        self.assertEqual(self.runtime.STATS["main.payload.pickle"][1], calls)
        self.assertGreater(self.runtime.STATS["main.payload.pickle"][0], 0)

    def test_a_fresh_pool_counts_every_task_of_the_batch_before_the_clock_starts(self):
        tasks, counted_before_the_call = step_tasks(4), []

        def starmap(function, inputs):
            counted_before_the_call.append(self.runtime.STATS["main.payload.bytes"][1])
            return ["result"] * len(inputs)

        with mock.patch.object(self.runtime.multiprocessing, "Pool") as pool_class:
            pool_class.return_value.starmap = starmap
            with self.runtime.TimedPool(4) as pool:
                self.assertEqual(pool.starmap(len, tasks), ["result"] * 4)
        self.assertIs(pool_class.call_args.kwargs["initializer"], self.runtime.restore_worker)
        self.assert_payload(len(pickle.dumps(tasks[0], 5)) * 4, 1)
        self.assertEqual(counted_before_the_call, [1])
        self.assertEqual(self.runtime.STATS["main.pool.tasks"], [4, 1])

    def test_a_persistent_pool_counts_a_whole_step_at_its_first_batch_only(self):
        step, next_step, executor = step_tasks(5), step_tasks(3), FakeExecutor()
        step_bytes = len(pickle.dumps(step[0], 5)) * 5
        with mock.patch.multiple(self.shared_pool, executor=executor, shared=None, futures=None):
            pool = self.runtime.PersistentPool(2)
            self.assertEqual(pool.starmap(len, step[:2]), ["result"] * 2)
            self.assert_payload(step_bytes, 1)
            self.assertEqual(pool.starmap(len, step[2:4]), ["result"] * 2)
            self.assertEqual(pool.starmap(len, step[4:]), ["result"])
            self.assert_payload(step_bytes, 1)
            self.assertEqual(executor.submitted, 5)  # what the count above claims SharedPool sends
            pool.starmap(len, next_step)
        self.assert_payload(step_bytes + len(pickle.dumps(next_step[0], 5)) * 3, 2)
        self.assertEqual(executor.submitted, 8)
        self.assertEqual(self.runtime.STATS["main.pool.tasks"], [8, 4])

    def test_only_the_call_that_creates_the_executor_is_timed(self):
        registered = self.runtime.registered
        state = ({"NUMBER_CPU": 3}, None, None)
        with mock.patch.multiple(self.shared_pool, executor=None), mock.patch.object(
            registered, "ProcessPoolExecutor"
        ) as executor_class, mock.patch.object(registered, "_worker_state", return_value=state):
            self.runtime.PersistentPool(2)
            self.runtime.PersistentPool(2)
        executor_class.assert_called_once()
        self.assertEqual(executor_class.call_args.args, (self.runtime.POOL_SIZE or 3,))
        self.assertIs(executor_class.call_args.kwargs["initargs"][0], self.runtime.restore_worker)
        self.assertEqual(self.runtime.STATS["main.pool.create"][1], 1)


class TaskSamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = load_perf_runtime()

    def test_every_tenth_call_gives_its_middle_task_until_the_limit(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(self.runtime, "PERF_DIR", Path(tmp)):
            sampler = self.runtime.TaskSampler(2)
            for call in range(1, 41):
                sampler.take([(call, "first"), (call, "middle"), (call, "last")])
            saved = {path.name: pickle.loads(path.read_bytes()) for path in (Path(tmp) / "tasks").iterdir()}
        self.assertEqual(sorted(saved), ["task_0.pkl", "task_1.pkl"])
        self.assertEqual(saved["task_0.pkl"]["arguments"], (10, "middle"))
        self.assertEqual(saved["task_1.pkl"]["arguments"], (20, "middle"))
        self.assertEqual(saved["task_0.pkl"]["constants"].keys(), self.runtime.registered._constants().keys())

    def test_no_limit_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(self.runtime, "PERF_DIR", Path(tmp)):
            sampler = self.runtime.TaskSampler(0)
            for _ in range(10):
                sampler.take([("task",)])
            self.assertFalse((Path(tmp) / "tasks").exists())

    def test_a_task_that_cannot_be_pickled_raises_and_leaves_no_file(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(self.runtime, "PERF_DIR", Path(tmp)):
            sampler = self.runtime.TaskSampler(1)
            sampler.calls = 9  # the next call is a sampled one
            with self.assertRaises((pickle.PicklingError, AttributeError)):
                sampler.take([(lambda: None,)])
            self.assertEqual(list((Path(tmp) / "tasks").iterdir()), [])

    def test_every_starmap_call_is_offered_to_the_sampler(self):
        tasks = step_tasks(2)
        with mock.patch.object(self.runtime, "SAMPLER") as sampler:
            self.runtime.timed_starmap(lambda function, inputs: [], len, tasks, 2)
        sampler.take.assert_called_once_with(tasks)


if __name__ == "__main__":
    unittest.main()
