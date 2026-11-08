"""Stage-level timers and an optional persistent worker pool for a staged case (copied next to the case script).

profile_case.py stages this file as perf_runtime.py beside the registered windows_case_runtime.py and points the
staged case at it. configure() first applies the registered runtime (events.jsonl, Windows-safe pool, RNG and
constant inheritance), then swaps the pool class used by build_wall and wraps functions with perf_counter timers.
The wrappers pass arguments and results through unchanged.

Pool modes (environment PERF_PERSISTENT):
- 0: TimedPool, a fresh pool per batch exactly as build_wall creates it, with create / starmap / terminate timed.
- 1: PersistentPool, worker processes created once and reused. Every batch re-sends the constants snapshot, so a worker
  sees what a fresh pool's initializer would give it; the RNG state is not re-sent because place_one_stone reseeds both
  global RNGs before it uses them.

Outputs under $PERF_DIR: main_stats.json (timers of the main process, run times) written at exit, and one
place_<pid>.jsonl line per place_one_stone call in a worker (plus cprof_<pid>.pstats with PERF_CPROFILE=1).

This file is imported by the staged case and by spawned workers, whose working directory is the case directory, so it
cannot import the research package (same constraint as windows_case_runtime).
"""

import atexit
import contextlib
import cProfile
import importlib
import json
import multiprocessing
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import windows_case_runtime as registered

PERF_DIR = Path(os.environ["PERF_DIR"])
USE_CPROFILE = os.environ.get("PERF_CPROFILE") == "1"
PERSISTENT = os.environ.get("PERF_PERSISTENT") == "1"
POOL_SIZE = int(os.environ.get("PERF_POOL_SIZE", "0"))  # 0: the case's set_number_cpu value

STATS = defaultdict(lambda: [0.0, 0])  # label -> [seconds, calls]
SHARED_TARGETS = (  # (module, owner class or None, attributes) timed in the main process and in every worker
    ("Stonepacker2D.entity.base", "Base", ("add_stone", "draw_assembly", "create_mirrored_base")),
    ("Stonepacker2D.entity.base", None, ("rotate_brick", "add_wedges")),
    (
        "Stonepacker2D.stack.place_stone",
        None,
        ("evaluate_convolve", "evaluate", "get_void_to_left", "deepcopy", "evaluate_kine"),
    ),
    (
        "Stonepacker2D.evaluate.tilting_angle",
        None,
        ("pixel_to_mesh", "base_to_elements", "clear_contps", "cal_Aglobal", "solve_force_rigid"),
    ),
    (
        "Stonepacker2D.entity.stone",
        "Stone",
        ("create_mirrored_stone", "rotated_by_90", "from_labeled_img", "rotate_min_shape_factor", "cal_shape_factor"),
    ),
)
MAIN_TARGETS = (
    ("Stonepacker2D.stack.build_wall", None, ("select", "controlled_policy", "deepcopy")),
    (
        "Stonepacker2D.stack.vendor",
        "Vendor",
        (
            "cluster_stones",
            "get_variant_stones",
            "get_same_stones",
            "get_random_stones",
            "return_stones",
        ),
    ),
)


@contextlib.contextmanager
def clock(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        STATS[label][0] += time.perf_counter() - start
        STATS[label][1] += 1


def timed(label, function):
    def wrapper(*args, **kwargs):
        with clock(label):
            return function(*args, **kwargs)

    wrapper.__wrapped__ = function
    return wrapper


def install_timers(tag, targets):
    """Replace each target attribute by a timed wrapper labelled '<tag><module or class>.<attribute>'."""
    for module_name, owner_name, attributes in targets:
        module = importlib.import_module(module_name)
        owner = getattr(module, owner_name) if owner_name else module
        scope = owner_name or module_name.rsplit(".", 1)[-1]
        for attribute in attributes:
            setattr(owner, attribute, timed(f"{tag}{scope}.{attribute}", getattr(owner, attribute)))


def settings():
    """Uppercase globals of Stonepacker2D.utils.constant: what a pool initializer hands to its workers."""
    module = importlib.import_module("Stonepacker2D.utils.constant")
    return {key: value for key, value in vars(module).items() if key.isupper()}


def snapshot_delta(before):
    return {
        label: [seconds - before.get(label, [0, 0])[0], calls - before.get(label, [0, 0])[1]]
        for label, (seconds, calls) in STATS.items()
        if calls - before.get(label, [0, 0])[1]
    }


def wrap_place_one_stone():
    """Worker side: time each place_one_stone call and append its record (a task unpickles the function by name)."""
    place_stone = importlib.import_module("Stonepacker2D.stack.place_stone")
    original, pid = place_stone.place_one_stone, os.getpid()
    profiler = cProfile.Profile() if USE_CPROFILE else None

    def place_one_stone(*args, **kwargs):
        before = {label: list(value) for label, value in STATS.items()}
        start = time.perf_counter()
        if profiler:
            profiler.enable()
        try:
            return original(*args, **kwargs)
        finally:
            if profiler:
                profiler.disable()
                profiler.dump_stats(str(PERF_DIR / f"cprof_{pid}.pstats"))
            record = {
                "pid": pid,
                "pose": args[0],
                "placed": len(args[3].placed_bricks),
                "total": time.perf_counter() - start,
                "sections": snapshot_delta(before),
                "t_end": time.time(),
            }
            with (PERF_DIR / f"place_{pid}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")

    place_stone.place_one_stone = place_one_stone


def restore_worker(constants, numpy_state, random_state):
    registered._restore_worker(constants, numpy_state, random_state)  # state first: the timers never touch the RNG
    install_timers("worker.", SHARED_TARGETS)
    wrap_place_one_stone()


def spawn_pool(processes):
    with clock("main.pool.create"):
        return multiprocessing.Pool(
            processes, initializer=restore_worker, initargs=(settings(), np.random.get_state(), random.getstate())
        )


def run_task(constants, function, arguments):
    importlib.import_module("Stonepacker2D.utils.constant").__dict__.update(constants)
    return function(*arguments)


class TimedPool:
    """build_wall's per-batch pool (context manager, starmap) with its three phases timed."""

    def __init__(self, processes):
        self.pool = spawn_pool(processes)

    def __enter__(self):
        return self

    def starmap(self, function, inputs):
        with clock("main.pool.starmap"):
            result = self.pool.starmap(function, inputs)
        STATS["main.pool.tasks"][0] += len(inputs)
        STATS["main.pool.tasks"][1] += 1
        return result

    def __exit__(self, *exc_info):
        with clock("main.pool.terminate"):
            self.pool.terminate()


class PersistentPool(TimedPool):
    """The same interface over one pool that lives until the process exits; leaving the context stops nothing."""

    shared = None

    def __init__(self, processes):
        if PersistentPool.shared is None:
            PersistentPool.shared = spawn_pool(POOL_SIZE or settings()["NUMBER_CPU"])
            atexit.register(PersistentPool.shared.terminate)
        self.pool = PersistentPool.shared

    def starmap(self, function, inputs):
        constants = settings()
        with clock("main.pool.starmap"):
            result = self.pool.starmap(run_task, [(constants, function, arguments) for arguments in inputs])
        STATS["main.pool.tasks"][0] += len(inputs)
        STATS["main.pool.tasks"][1] += 1
        return result

    def __exit__(self, *exc_info):
        return None


def write_stats(configured_at):
    meta = {"configure_epoch": configured_at, "exit_epoch": time.time(), "persistent_pool": PERSISTENT}
    (PERF_DIR / "main_stats.json").write_text(json.dumps({"meta": meta, "stats": STATS}), encoding="utf-8")


def configure(namespace):
    """Registered runtime first, then the pool swap and the timers."""
    configured_at = time.time()
    PERF_DIR.mkdir(parents=True, exist_ok=True)
    registered.configure(namespace)
    importlib.import_module("Stonepacker2D.stack.build_wall").Pool = PersistentPool if PERSISTENT else TimedPool
    install_timers("main.", SHARED_TARGETS + MAIN_TARGETS)
    atexit.register(write_stats, configured_at)
