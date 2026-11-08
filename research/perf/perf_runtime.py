"""Stage timers, payload and memory counters and task samples for a staged case (copied next to the case script).

profile_case.py stages this file as perf_runtime.py beside the registered windows_case_runtime.py and points the
staged case at it. configure() first applies the registered runtime (events.jsonl, Windows-safe pool, RNG and
constant inheritance), then swaps the pool class used by build_wall and wraps functions with perf_counter timers.
The wrappers pass arguments and results through unchanged, and nothing here draws a random number.

Pool modes (environment PERF_PERSISTENT):
- 0: TimedPool, a fresh pool per batch exactly as build_wall creates it, with create / starmap / terminate timed.
- 1: PersistentPool, the registered SharedPool with the same timers; PERF_POOL_SIZE overrides its size.

Outputs under $PERF_DIR: main_stats.json (timers and payload counters of the main process, run times) written at exit,
one place_<pid>.jsonl line per place_one_stone call in a worker (plus cprof_<pid>.pstats with PERF_CPROFILE=1), and with
PERF_DUMP_TASKS=N up to N files tasks/task_<i>.pkl, each one real task and the constants it ran under.

This file is imported by the staged case and by spawned workers, whose working directory is the case directory, so it
cannot import the research package (same constraint as windows_case_runtime).
"""

import atexit
import contextlib
import cProfile
import ctypes
import importlib
import json
import multiprocessing
import os
import pickle
import time
from collections import defaultdict
from ctypes import wintypes
from pathlib import Path

import windows_case_runtime as registered

PERF_DIR = Path(os.environ["PERF_DIR"])
USE_CPROFILE = os.environ.get("PERF_CPROFILE") == "1"


class MemoryCounters(ctypes.Structure):
    """PROCESS_MEMORY_COUNTERS of the Windows API."""

    _fields_ = [
        ("cb", wintypes.DWORD),
        ("page_fault_count", wintypes.DWORD),
        ("peak_working_set", ctypes.c_size_t),
        ("other_sizes", ctypes.c_size_t * 7),
    ]


def peak_working_set():
    """This process's peak working set in bytes; the registered environment has no psutil."""
    counters = MemoryCounters(cb=ctypes.sizeof(MemoryCounters))
    ctypes.windll.kernel32.K32GetProcessMemoryInfo(wintypes.HANDLE(-1), ctypes.byref(counters), counters.cb)
    return counters.peak_working_set


PERSISTENT = os.environ.get("PERF_PERSISTENT") == "1"
POOL_SIZE = int(os.environ.get("PERF_POOL_SIZE", "0"))  # 0: the case's set_number_cpu value
DUMP_TASKS = int(os.environ.get("PERF_DUMP_TASKS", "0"))
DUMP_EVERY = int(os.environ.get("PERF_DUMP_EVERY", "10"))  # starmap calls per task sample
PICKLE_PROTOCOL = 5

STATS = defaultdict(lambda: [0.0, 0])  # label -> [seconds, or the amount for a counter such as main.pool.tasks; calls]
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
        ("contps_of_image", "base_to_elements", "cal_Aglobal", "solve_force_rigid"),
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


def tally(label, amount):
    STATS[label][0] += amount
    STATS[label][1] += 1


@contextlib.contextmanager
def clock(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        tally(label, time.perf_counter() - start)


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
                "peak_wset": peak_working_set(),
            }
            with (PERF_DIR / f"place_{pid}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")

    place_stone.place_one_stone = place_one_stone


def restore_worker(constants, numpy_state, random_state):
    registered._restore_worker(constants, numpy_state, random_state)  # state first: the timers never touch the RNG
    install_timers("worker.", SHARED_TARGETS)
    wrap_place_one_stone()


class TaskSampler:
    """Saves real tasks for comparisons without a wall: from every DUMP_EVERY-th starmap call, up to `limit` samples."""

    def __init__(self, limit):
        self.limit = limit
        self.calls = 0
        self.saved = 0

    def take(self, inputs):
        self.calls += 1
        if self.saved >= self.limit or self.calls % DUMP_EVERY:
            return
        sample = {"constants": registered._constants(), "arguments": inputs[len(inputs) // 2]}
        folder = PERF_DIR / "tasks"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"task_{self.saved}.pkl").write_bytes(pickle.dumps(sample, PICKLE_PROTOCOL))
        self.saved += 1


SAMPLER = TaskSampler(DUMP_TASKS)


def record_payload(task, sent):
    """Count `sent` tasks of the size of `task`; one is pickled and scaled because a whole batch would cost a wall."""
    if not sent:
        return
    start = time.perf_counter()
    size = len(pickle.dumps(task, PICKLE_PROTOCOL))
    tally("main.payload.pickle", (time.perf_counter() - start) * sent)
    tally("main.payload.bytes", size * sent)


def timed_starmap(starmap, function, inputs, sent):
    """Run `starmap` under its clock; `sent` is how many tasks this call hands to the workers."""
    SAMPLER.take(inputs)
    record_payload(inputs[0], sent)
    with clock("main.pool.starmap"):
        result = starmap(function, inputs)
    tally("main.pool.tasks", len(inputs))
    return result


class TimedPool:
    """build_wall's per-batch pool (context manager, starmap) with its three phases timed."""

    def __init__(self, processes):
        with clock("main.pool.create"):
            self.pool = multiprocessing.Pool(processes, initializer=restore_worker, initargs=registered._worker_state())

    def __enter__(self):
        return self

    def starmap(self, function, inputs):
        return timed_starmap(self.pool.starmap, function, inputs, len(inputs))

    def __exit__(self, *exc_info):
        with clock("main.pool.terminate"):
            self.pool.terminate()


class PersistentPool(registered.SharedPool):
    """The registered SharedPool, whose workers run the timed restore_worker, with its main-process calls timed."""

    initializer = staticmethod(restore_worker)
    size = POOL_SIZE

    def __init__(self, processes):
        with clock("main.pool.create") if self.executor is None else contextlib.nullcontext():
            super().__init__(processes)

    def starmap(self, function, inputs):
        sent = len(inputs[0][2]) if inputs[0][0] == 0 else 0  # the first batch of a step submits the whole step
        return timed_starmap(super().starmap, function, inputs, sent)


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
