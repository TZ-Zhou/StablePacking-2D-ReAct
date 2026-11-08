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
from types import SimpleNamespace
from unittest import mock

import numpy as np

from research.paths import LEGACY_RUNS, REPO
from research.validation import windows_case_runtime as runtime

ARCHIVED_BASES = sorted((LEGACY_RUNS / "controllers" / "p1-r2" / "null-seed20").glob("result_*/base.pkl"))


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

    def test_a_worker_unpickles_the_arguments_of_a_step_once(self):
        block, sizes = runtime._share_step(step_with_arrays(2)[0][1:])
        self.addCleanup(block.unlink)
        self.addCleanup(block.close)
        first, second = (runtime._run_shared_task({}, arguments_seen, 7, block.name, sizes, j)[0] for j in (0, 1))
        next_step = runtime._run_shared_task({}, arguments_seen, 8, block.name, sizes, 0)[0]
        self.assertIs(first[1], second[1])
        self.assertIsNot(first[1], next_step[1])
        self.assertTrue(np.array_equal(first[1][0], next_step[1][0]))

    def test_the_shared_memory_is_released_after_a_step(self):
        tasks = one_step(4)
        with runtime.SharedPool(2) as pool:
            pool.starmap(pose_index, tasks[:2])
            pool.starmap(pose_index, tasks[2:])
        self.assertIsNone(runtime.SharedPool.block)

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


def children_of(value):
    """What a container or object holds; None for an array of numbers and a plain value."""
    if isinstance(value, np.ndarray):
        return value.ravel().tolist() if value.dtype == object else None
    if isinstance(value, (list, tuple)):
        return value
    if isinstance(value, dict):
        return list(value.items())
    return list(vars(value).items()) if hasattr(value, "__dict__") else None


def leaves(value, seen=None):
    """Every array and plain value reachable from value, in a fixed order; shared objects are visited once."""
    seen = set() if seen is None else seen
    children = children_of(value)
    if children is None:
        yield value
    elif id(value) not in seen:
        seen.add(id(value))
        for child in children:
            yield from leaves(child, seen)


def same_leaf(got, want):
    if isinstance(want, np.ndarray):
        return (got.shape, got.dtype, got.tobytes()) == (want.shape, want.dtype, want.tobytes())
    plain = want is None or isinstance(want, (bool, int, float, str, bytes, np.generic))
    return type(got) is type(want) and (not plain or repr(got) == repr(want))


def all_arguments(pose, *shared):
    return shared


def a_wall(value=1.0):
    return SimpleNamespace(
        matrix=np.full((6, 8), value),
        centers=[[1.0, 2.0]],
        rock_centers=[[1.0, 2.0]],
        rock_tops=[3],
        rock_left_tops=[[0, 3]],
        rock_right_tops=[[2, 3]],
        rock_right_bottoms=[[2, 0]],
    )


def a_pose(stone=7, rotation=0, height=4, fill=1.0):
    return SimpleNamespace(
        id=stone,
        rotate_from_ori=rotation,
        width=3,
        height=height,
        area=12,
        center=[1.5, 2.0],
        matrix=np.full((6, 8), fill),
    )


def batch_of(*heights):
    return [SimpleNamespace(height=height) for height in heights]


class PoseMemo(unittest.TestCase):
    result = (2, 3, 0.5, np.array([-0.25, 1.0, 2.0, 3.0, 4.0, 5.0]), 3.0, np.zeros(6))

    def setUp(self):
        self.memo = runtime.PoseMemo()
        self.results = self.memo.results_for(a_wall())

    def test_an_equal_pose_on_an_equal_wall_is_answered(self):
        runtime.PoseMemo.remember(self.results, a_pose(), self.result, None)
        results = self.memo.results_for(a_wall())
        self.assertIs(self.memo.known(results, a_pose(), batch_of(1, 2, 3)), self.result)

    def test_a_pose_not_evaluated_yet_is_not_answered(self):
        runtime.PoseMemo.remember(self.results, a_pose(), self.result, None)
        self.assertIsNone(self.memo.known(self.results, a_pose(rotation=-90), batch_of(1, 2, 3)))

    def test_a_pose_with_the_same_key_but_other_pixels_is_not_answered(self):
        runtime.PoseMemo.remember(self.results, a_pose(), self.result, None)
        self.assertIsNone(self.memo.known(self.results, a_pose(fill=2.0), batch_of(1, 2, 3)))

    def test_another_wall_starts_without_results(self):
        runtime.PoseMemo.remember(self.results, a_pose(), self.result, None)
        self.assertEqual(self.memo.results_for(a_wall(2.0)), {})

    def test_a_wall_and_its_mirror_image_are_both_kept_but_not_a_third(self):
        runtime.PoseMemo.remember(self.results, a_pose(), self.result, None)
        self.memo.results_for(a_wall(2.0))
        self.assertIs(self.memo.results_for(a_wall()), self.results)
        self.memo.results_for(a_wall(3.0))
        self.memo.results_for(a_wall(4.0))
        self.assertEqual(self.memo.results_for(a_wall()), {})

    def test_the_value_read_from_the_batch_is_computed_for_the_new_batch(self):
        runtime.PoseMemo.remember(self.results, a_pose(), self.result, 4)
        heights = batch_of(2, 3, 9, 10)
        got = self.memo.known(self.results, a_pose(), heights)
        expected = -abs(np.percentile([2.0, 3.0, 9.0, 10.0], 50) - 4) / 4
        self.assertEqual(got[3][0], expected)
        np.testing.assert_array_equal(got[3][1:], self.result[3][1:])
        self.assertEqual(self.result[3][0], -0.25)
        self.assertEqual(got[:3], self.result[:3])


def place_one_stone(pose, stones, poses, base):
    """Stands for the real function in SharedPool: a value that differs on every call shows whether a worker ran."""
    return pose, time.perf_counter_ns()


class PoseReuseInThePool(unittest.TestCase):
    def setUp(self):
        runtime.SharedPool.size = 1
        runtime.SharedPool.memo = runtime.PoseMemo()
        self.addCleanup(self.stop_workers)

    @staticmethod
    def stop_workers():
        runtime.SharedPool.release()
        if runtime.SharedPool.executor is not None:
            runtime.SharedPool.executor.shutdown(wait=True, cancel_futures=True)
            runtime.SharedPool.executor = None

    @staticmethod
    def attempt(poses, wall):
        shared = (batch_of(4, 4), poses, wall)
        return runtime.SharedPool(1).starmap(place_one_stone, [(index, *shared) for index in range(len(poses))])

    def test_only_the_poses_not_evaluated_on_this_wall_reach_a_worker(self):
        first = self.attempt([a_pose(stone=1), a_pose(stone=2)], a_wall())
        second = self.attempt([a_pose(stone=3), a_pose(stone=2)], a_wall())
        self.assertNotEqual(second[0], first[0])
        self.assertEqual(second[1], first[1])

    def test_a_new_wall_evaluates_the_same_pose_again(self):
        first = self.attempt([a_pose()], a_wall())
        self.assertNotEqual(self.attempt([a_pose()], a_wall(2.0)), first)


@unittest.skipUnless(ARCHIVED_BASES, "the archived wall is not present")
class RealStepPayload(unittest.TestCase):
    """A step's arguments rebuilt from an archived wall arrive through the shared block as through a per-task pickle."""

    def setUp(self):
        with ARCHIVED_BASES[0].open("rb") as stream:
            base = pickle.load(stream)
        rocks = list(base.placed_rocks)
        self.shared = (rocks, rocks, base, np.arange(12.0).reshape(6, 2), False, True, (0, 10, 0, 10))
        block, sizes = runtime._share_step(self.shared)
        self.addCleanup(block.unlink)
        self.addCleanup(block.close)
        self.received = list(leaves(runtime._run_shared_task({}, all_arguments, 9, block.name, sizes, 0)[0]))

    def test_it_equals_what_a_per_task_pickle_delivers(self):
        expected = list(leaves(pickle.loads(pickle.dumps(self.shared, 4))))
        self.assertEqual(len(self.received), len(expected))
        self.assertTrue(all(same_leaf(got, want) for got, want in zip(self.received, expected)))

    def test_no_array_arrives_read_only(self):
        arrays = [leaf for leaf in self.received if isinstance(leaf, np.ndarray)]
        self.assertTrue(arrays)
        self.assertTrue(all(array.flags.writeable for array in arrays))

    def test_writing_to_an_array_of_a_worker_leaves_the_block_as_it_was(self):
        block, sizes = runtime._share_step((np.zeros(4), 1))
        self.addCleanup(block.unlink)
        self.addCleanup(block.close)
        first = runtime._load_step(block.name, sizes)
        first[0][:] = 7.0
        self.assertEqual(runtime._load_step(block.name, sizes)[0].tolist(), [0.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
