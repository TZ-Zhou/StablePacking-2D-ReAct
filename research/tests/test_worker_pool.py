"""The persistent workers of windows_case_runtime: reuse, settings, whole-step scheduling, failures (no wall is built).

Run from the repository root: python -B -m unittest research.tests.test_worker_pool
"""

import importlib
import os
import pickle
import subprocess
import sys
import time
import unittest
from concurrent.futures.process import BrokenProcessPool
from multiprocessing.shared_memory import SharedMemory
from unittest import mock

import numpy as np

from research.paths import REPO
from research.validation import windows_case_runtime as runtime


def process_exists(pid):
    """The registered environment has no psutil, and spawned workers import this module."""
    listed = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"], capture_output=True, text=True)
    return f'"{pid}"' in listed.stdout


SLOW_POSE_SECONDS = 2.0
EXIT_LIMIT_SECONDS = 30
LOST_PARENT_CASE = """
import os
from research.tests.test_worker_pool import one_step, worker_pid
from research.validation import windows_case_runtime as runtime

if __name__ == "__main__":
    runtime.SharedPool.size = 1
    with runtime.SharedPool(1) as pool:
        print(pool.starmap(worker_pid, one_step(1))[0], flush=True)
    os._exit(0)
"""


def one_step(poses):
    """The tasks of one step, as build_wall makes them: every task shares the same argument objects."""
    stones, all_poses = [], list(range(poses))
    return [(pose, stones, all_poses) for pose in range(poses)]


def step_with_arrays(poses):
    stones, all_poses = [np.arange(6.0).reshape(2, 3)], list(range(poses))
    return [(pose, stones, all_poses) for pose in range(poses)]


def arguments_seen(pose, stones, poses):
    return pose, stones, poses


def worker_pid(pose, stones, poses):
    return os.getpid()


def pose_index(pose, stones, poses):
    return pose


def number_cpu_seen(pose, stones, poses):
    return importlib.import_module("Stonepacker2D.utils.constant").get_number_cpu()


def start_and_end(pose, stones, poses):
    start = time.time()
    time.sleep(SLOW_POSE_SECONDS if pose == 0 else 0.05)
    return start, time.time()


def raise_error(pose, stones, poses):
    raise KeyError(pose)


def die(pose, stones, poses):
    os._exit(1)


class SharedPool(unittest.TestCase):
    def setUp(self):
        for name, value in (("executor", None), ("size", 2), ("shared", None), ("futures", None), ("block", None), ("waiting", None)):
            patch = mock.patch.object(runtime.SharedPool, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(self.close_workers)

    @staticmethod
    def close_workers():
        runtime.SharedPool.release()
        if runtime.SharedPool.executor is not None:
            runtime.SharedPool.executor.shutdown(wait=True, cancel_futures=True)

    @staticmethod
    def run_step(function, tasks):
        with runtime.SharedPool(len(tasks)) as pool:
            return pool.starmap(function, tasks)

    def test_two_pools_use_the_same_workers(self):
        first = set(self.run_step(worker_pid, one_step(4)))
        second = set(self.run_step(worker_pid, one_step(4)))
        self.assertLessEqual(len(first | second), 2)

    def test_a_setting_changed_after_the_workers_started_reaches_them(self):
        constants = importlib.import_module("Stonepacker2D.utils.constant")
        self.addCleanup(constants.set_number_cpu, constants.get_number_cpu())
        constants.set_number_cpu(3)
        self.assertEqual(set(self.run_step(number_cpu_seen, one_step(2))), {3})
        constants.set_number_cpu(5)
        self.assertEqual(set(self.run_step(number_cpu_seen, one_step(2))), {5})

    def test_results_follow_the_poses_over_the_batches_of_a_step(self):
        tasks = one_step(6)
        with runtime.SharedPool(2) as pool:
            batches = [pool.starmap(pose_index, tasks[first : first + 2]) for first in (0, 2, 4)]
        self.assertEqual(batches, [[0, 1], [2, 3], [4, 5]])

    def test_a_free_worker_does_not_wait_for_the_slowest_pose_of_its_batch(self):
        tasks = one_step(4)
        with runtime.SharedPool(2) as pool:
            first = pool.starmap(start_and_end, tasks[:2])
            second = pool.starmap(start_and_end, tasks[2:])
        self.assertLess(second[0][0], first[0][1], "the next batch started only after the slow pose ended")

    def test_a_task_of_another_step_is_refused(self):
        tasks = one_step(4)
        with runtime.SharedPool(2) as pool:
            pool.starmap(pose_index, tasks[:2])
            with self.assertRaises(AssertionError):
                pool.starmap(pose_index, one_step(4)[2:])

    def test_a_task_sees_what_direct_passing_would_give_it(self):
        tasks = step_with_arrays(3)
        results = self.run_step(arguments_seen, tasks)
        for task, (pose, stones, poses) in zip(tasks, results):
            self.assertEqual(pose, task[0])
            self.assertTrue(np.array_equal(stones[0], task[1][0]))
            self.assertEqual(poses, task[2])

    def test_two_tasks_of_one_step_get_different_objects(self):
        payload = pickle.dumps(step_with_arrays(2)[0][1:], 5)
        block = SharedMemory(create=True, size=len(payload))
        self.addCleanup(block.unlink)
        self.addCleanup(block.close)
        block.buf[: len(payload)] = payload
        first, second = (runtime._run_shared_task({}, arguments_seen, block.name, len(payload), j) for j in (0, 1))
        self.assertIsNot(first[1], second[1])
        self.assertIsNot(first[1][0], second[1][0])
        self.assertTrue(np.array_equal(first[1][0], second[1][0]))

    def test_the_shared_memory_is_released_after_a_step(self):
        tasks = one_step(4)
        with runtime.SharedPool(2) as pool:
            pool.starmap(pose_index, tasks[:2])
            name = runtime.SharedPool.block.name
            pool.starmap(pose_index, tasks[2:])
        self.assertIsNone(runtime.SharedPool.block)
        with self.assertRaises(FileNotFoundError):
            SharedMemory(name=name)

    def test_an_error_in_a_task_reaches_the_case(self):
        with self.assertRaises(KeyError):
            self.run_step(raise_error, one_step(2))

    def test_a_worker_that_dies_fails_the_case_instead_of_hanging(self):
        started = time.monotonic()
        with self.assertRaises(BrokenProcessPool):
            self.run_step(die, one_step(2))
        self.assertLess(time.monotonic() - started, EXIT_LIMIT_SECONDS)

    def test_a_worker_does_not_outlive_a_case_process_killed_alone(self):
        finished = subprocess.run(
            [sys.executable, "-B", "-c", LOST_PARENT_CASE], cwd=REPO, capture_output=True, text=True, timeout=120
        )
        self.assertEqual(finished.returncode, 0, finished.stderr)
        worker = int(finished.stdout.split()[-1])
        deadline = time.monotonic() + EXIT_LIMIT_SECONDS
        while process_exists(worker) and time.monotonic() < deadline:
            time.sleep(0.2)
        self.assertFalse(process_exists(worker))


if __name__ == "__main__":
    unittest.main()
