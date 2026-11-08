"""Process-local Windows compatibility and observation for original test scripts."""
from concurrent.futures import Future, ProcessPoolExecutor
import importlib
import itertools
import json
import mmap
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import os
from pathlib import Path
import pickle
import random
import threading
import time

import cv2
import matplotlib.figure
import matplotlib.pyplot
import numpy as np


def _skip_figures():
    """A wall keeps data only: no figure is drawn or saved, in the case process or in a worker.

    Nothing reads a figure back and none of these calls draws a random number, so decisions do not change.
    """
    base = importlib.import_module("Stonepacker2D.entity.base").Base
    base.draw_assembly = base.draw_last_action = lambda self, *args, **kwargs: None
    matplotlib.figure.Figure.savefig = lambda self, *args, **kwargs: None
    matplotlib.pyplot.savefig = lambda *args, **kwargs: None
    cv2.imwrite = lambda *args, **kwargs: True


def _restore_worker(constants, numpy_state, random_state):
    _skip_figures()
    module = importlib.import_module("Stonepacker2D.utils.constant")
    module.__dict__.update(constants)
    np.random.set_state(numpy_state)
    random.setstate(random_state)


def _constants():
    """The uppercase settings of Stonepacker2D.utils.constant: what a worker has to see."""
    module = importlib.import_module("Stonepacker2D.utils.constant")
    return {key: value for key, value in vars(module).items() if key.isupper()}


def _worker_state():
    return _constants(), np.random.get_state(), random.getstate()


def _pool(processes):
    return multiprocessing.Pool(processes, initializer=_restore_worker, initargs=_worker_state())


def _run_task(constants, function, arguments):
    """A long-lived worker gets the settings before each task: a case changes them after the workers have started."""
    importlib.import_module("Stonepacker2D.utils.constant").__dict__.update(constants)
    return function(*arguments)


_step_arguments = (None, None)  # (step number, the step's shared arguments) this worker unpickled last


def _share_step(shared):
    """(block, sizes): the pickle of a step's arguments followed by their arrays, which pickling did not copy."""
    buffers = []
    payload = pickle.dumps(shared, 5, buffer_callback=buffers.append)
    pieces = [payload, *(buffer.raw() for buffer in buffers)]
    sizes = tuple(len(piece) for piece in pieces)
    block = SharedMemory(create=True, size=max(sum(sizes), 1))
    offset = 0
    for piece, size in zip(pieces, sizes):
        block.buf[offset : offset + size] = piece
        offset += size
    return block, sizes


def _load_step(name, sizes):
    """The step's arguments over a copy-on-write view of its block: an array that gets written becomes a private copy."""
    view = memoryview(mmap.mmap(-1, sum(sizes), tagname=name, access=mmap.ACCESS_COPY))
    ends = list(itertools.accumulate(sizes))
    return pickle.loads(view[: ends[0]], buffers=[view[start:end] for start, end in zip(ends, ends[1:])])


def _run_shared_task(constants, function, step, name, sizes, pose):
    """The arguments are unpickled once per step in each worker: place_one_stone only reads them."""
    global _step_arguments
    if _step_arguments[0] != step:
        _step_arguments = step, _load_step(name, sizes)
    result = _run_task(constants, function, (pose, *_step_arguments[1]))
    return result, _batch_median_height(function)


def _is_placement(function):
    return getattr(function, "__name__", "") == "place_one_stone"


def _batch_median_height(function):
    """The stone height the last call of `function` compared with the batch's median, None when it did not."""
    if not _is_placement(function):
        return None
    return importlib.import_module("Stonepacker2D.stack.place_stone").batch_median_height


def _wall_lists(base):
    return base.centers, base.rock_centers, base.rock_tops, base.rock_left_tops, base.rock_right_tops, base.rock_right_bottoms


class PoseMemo:
    """The results of the poses evaluated on the last two walls: a step evaluates a wall and its mirror image.

    A sampling attempt that failed or only chose the cluster is followed by one on the same wall, which often evaluates
    the same poses again. place_one_stone reads the pose, the wall and, when no stone is to the left of the pose, the
    median height of the batch; everything else it returns stands, and that one value is computed again for the new batch.
    A result is reused only after the wall and the pose are found equal to those it was computed for.
    """

    WALLS_KEPT = 2

    def __init__(self):
        self.walls = []  # [(wall image, lists of the wall, {(stone id, rotation): (pose, result, height)})], oldest first

    def _results_of(self, base):
        lists = _wall_lists(base)
        for image, kept, results in self.walls:
            if kept == lists and np.array_equal(image, base.matrix):
                return results
        return None

    def results_for(self, base):
        """The results of this wall; a wall not seen before replaces the oldest one."""
        results = self._results_of(base)
        if results is None:
            results = {}
            self.walls = [*self.walls[1 - self.WALLS_KEPT :], (base.matrix, _wall_lists(base), results)]
        return results

    @staticmethod
    def _same_pose(first, second):
        return (
            (first.width, first.height, first.area) == (second.width, second.height, second.area)
            and list(first.center) == list(second.center)
            and np.array_equal(first.matrix, second.matrix)
        )

    def known(self, results, pose, stones):
        """The result of `pose` for the batch `stones` when it was evaluated on this wall, else None."""
        entry = results.get((pose.id, pose.rotate_from_ori))
        if entry is None or not self._same_pose(entry[0], pose):
            return None
        _, result, height = entry
        if height is None:
            return result
        filters = result[3].copy()
        heights = np.array([stone.height for stone in stones], dtype=np.float64)
        filters[0] = -abs(np.percentile(heights, 50) - height) / height
        return (*result[:3], filters, *result[4:])

    @staticmethod
    def remember(results, pose, result, height):
        results[(pose.id, pose.rotate_from_ori)] = (pose, result, height)


def _exit_with_parent():
    """A worker blocked on its task queue would otherwise outlive a case process that was killed alone."""
    multiprocessing.parent_process().join()
    os._exit(1)


def _start_worker(initializer, *state):
    cv2.setNumThreads(1)  # the workers already use every core
    threading.Thread(target=_exit_with_parent, daemon=True).start()
    initializer(*state)


class SharedPool:
    """build_wall's `with Pool(n) as p: p.starmap(...)` over workers that live until the case process exits.

    A task is (pose index, *arguments shared by the step). The first batch of a step submits every pose of the step,
    so a free worker never waits for the slowest pose of a batch. place_one_stone reseeds both RNGs when it starts,
    so neither the order of the poses nor the worker that runs one can change a result. build_wall changes none of the
    shared arguments until it holds every result, so the background pickling sees what the step was given.
    """

    initializer = staticmethod(_restore_worker)
    size = 0  # 0: the case's NUMBER_CPU
    executor = None
    shared = None
    futures = None
    block = None
    waiting = None
    step = 0  # numbers the steps, so that a worker can tell its arguments are those of the task's step
    memo = PoseMemo()
    known = None  # the results of the wall of this step when the function is place_one_stone, else None
    reused = frozenset()  # the poses of this step answered from the memo

    def __init__(self, processes):
        if SharedPool.executor is None:
            state = _worker_state()
            SharedPool.executor = ProcessPoolExecutor(
                self.size or state[0]["NUMBER_CPU"], initializer=_start_worker, initargs=(self.initializer, *state)
            )

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    def starmap(self, function, inputs):
        if inputs[0][0] == 0:
            self._submit_step(function, inputs[0][1:])
        assert all(a is b for task in inputs for a, b in zip(task[1:], SharedPool.shared)), "not the poses of one step"
        try:
            outcomes = [SharedPool.futures[task[0]].result() for task in inputs]
        except BaseException:
            for future in SharedPool.futures:
                future.cancel()
            SharedPool.release()
            raise
        if SharedPool.known is not None:
            for task, (result, height) in zip(inputs, outcomes):
                if task[0] not in SharedPool.reused:
                    PoseMemo.remember(SharedPool.known, SharedPool.shared[1][task[0]], result, height)
        SharedPool.waiting -= {task[0] for task in inputs}
        if not SharedPool.waiting:
            SharedPool.release()
        return [result for result, _ in outcomes]

    @staticmethod
    def release():
        """Windows frees the block when its last handle closes; a worker holds a view until the next step's first task."""
        if SharedPool.block is not None:
            SharedPool.block.close()
            SharedPool.block.unlink()
            SharedPool.block = None

    def _submit_step(self, function, shared):
        SharedPool.release()
        constants = _constants()
        block, sizes = _share_step(shared)
        poses = range(len(shared[1]))
        SharedPool.block, SharedPool.shared, SharedPool.waiting = block, shared, set(poses)
        SharedPool.step += 1
        SharedPool.known = SharedPool.memo.results_for(shared[2]) if _is_placement(function) else None
        SharedPool.reused, SharedPool.futures = set(), []
        for pose in poses:
            result = None if SharedPool.known is None else SharedPool.memo.known(SharedPool.known, shared[1][pose], shared[0])
            if result is None:
                future = SharedPool.executor.submit(
                    _run_shared_task, constants, function, SharedPool.step, block.name, sizes, pose
                )
            else:
                SharedPool.reused.add(pose)
                future = Future()
                future.set_result((result, None))
            SharedPool.futures.append(future)


def configure(namespace):
    """Preserve original decisions; inherit parent settings in spawned workers."""
    _skip_figures()
    persistent = os.environ.get("BASELINE_PERSISTENT_POOL") == "1"
    importlib.import_module("Stonepacker2D.stack.build_wall").Pool = SharedPool if persistent else _pool
    events = Path(os.environ["BASELINE_CASE_EVENTS"])
    start = time.monotonic()

    def emit(kind, **values):
        record = {"event": kind, "elapsed_seconds": round(time.monotonic() - start, 3), **values}
        with events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print("BASELINE_EVENT " + json.dumps(record), flush=True)

    build_wall = importlib.import_module("Stonepacker2D.stack.build_wall")
    writer = None
    if multiprocessing.parent_process() is None and os.environ.get("BASELINE_SYNC_OUTPUT") != "1":
        import step_writer  # staged beside this file

        writer = step_writer.install(build_wall)
    original_commit = build_wall.trace_commit
    original_build = namespace["build_wall_vendor"]
    original_evaluate = namespace["evaluate_kine"]

    def commit(record):
        """The wall-clock time of a sampling attempt: an event, as commits.jsonl is compared byte for byte."""
        original_commit(record)
        emit("step_commit", step=record["step"], outcome=record["outcome"])

    def build(*args, **kwargs):
        constants = importlib.import_module("Stonepacker2D.utils.constant")
        constants.set_record_detail({name: False for name in constants.get_record_detail()})
        constants.set_record_time(False)  # after the script's own settings, before the pool copies them
        emit("build_enter", input_stones=args[1].get_number_stones(),
             candidates_per_iteration=args[2], build_options=kwargs,
             constants={key: value for key, value in vars(constants).items() if key.isupper()})
        result = original_build(*args, **kwargs)
        if writer is not None:
            writer.flush()
        emit("build_return", placed_rocks=len(result.placed_rocks), output_dir=constants.get_dir())
        return result

    def evaluate(*args, **kwargs):
        result = original_evaluate(*args, **kwargs)
        if kwargs.get("save_failure"):
            constants = importlib.import_module("Stonepacker2D.utils.constant")
            emit("final_evaluation_complete", placed_rocks=len(args[0].placed_rocks),
                 limit_force=float(result), output_dir=constants.get_dir())
            if os.environ["BASELINE_SELECT_FIRST_WALL"] == "1":
                # Select the first complete original replicate; do not change max_nb_runs.
                raise SystemExit(0)
        return result

    build_wall.trace_commit = commit
    namespace["build_wall_vendor"] = build
    namespace["evaluate_kine"] = evaluate
