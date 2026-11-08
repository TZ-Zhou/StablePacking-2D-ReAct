"""Compare two versions of place_one_stone on saved real tasks, without building a wall.

Usage (from the repository root, in the baseline environment):
  python -B -m research.perf.task_results run <out.json> <folder with perf/tasks> [...] [--poses N]
  python -B -m research.perf.task_results compare <old.json> <new.json>

`run` calls place_one_stone in this process on up to N poses of every saved task (research/perf/perf_runtime.py saves
them with PERF_DUMP_TASKS) and writes a digest of each return value and the seconds it took. Set PYTHONPATH to the tree
to test: without it the environment imports the main checkout's Stonepacker2D. `compare` reports every pose whose digest
differs and the total time of both runs; its exit code is 0 when no pose differs.

The per-pose timing array that place_one_stone returns is left out of the digest.
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

import Stonepacker2D
from Stonepacker2D.stack.place_stone import place_one_stone
from Stonepacker2D.utils import constant as constants

TIMES_FIELD = -1  # the last item of place_one_stone's return value is its timing array


def pose_indexes(count, poses):
    """Up to `poses` indexes spread over `count` poses, always the same ones for the same arguments."""
    return list(range(0, count, max(1, count // poses)))[:poses]


def canonical(value):
    """A byte string with the exact bits and types of a return value, the same in every process."""
    if isinstance(value, np.ndarray):
        return b"a" + str(value.dtype).encode() + repr(value.shape).encode() + value.tobytes()
    if isinstance(value, np.generic):
        return b"s" + str(value.dtype).encode() + value.tobytes()
    if isinstance(value, float):
        return b"f" + np.float64(value).tobytes()
    if isinstance(value, (list, tuple)):
        return (b"l" if isinstance(value, list) else b"t") + b"".join(canonical(item) for item in value) + b"]"
    if isinstance(value, dict):
        return b"d" + b"".join(canonical(key) + canonical(value[key]) for key in sorted(value)) + b"]"
    return b"o" + repr(value).encode()


def digest(result):
    """The digest of a place_one_stone return value, its timing array left out."""
    kept = result[:TIMES_FIELD] if isinstance(result, tuple) else result
    return hashlib.sha256(canonical(kept)).hexdigest()


def task_files(folders):
    """The saved tasks under each folder, in a fixed order, as (name, path)."""
    found = []
    for folder in map(Path, folders):
        paths = sorted((folder / "perf" / "tasks").glob("task_*.pkl"), key=lambda path: int(path.stem.split("_")[1]))
        found += [(f"{folder.name}/{path.stem}", path) for path in paths]
    return found


def run_task(sample, indexes):
    """{pose index: (digest, seconds)} of one saved task."""
    vars(constants).update(sample["constants"])
    arguments = list(sample["arguments"])
    results = {}
    for index in indexes:
        arguments[0] = index
        start = time.perf_counter()
        result = place_one_stone(*arguments)
        results[index] = (digest(result), time.perf_counter() - start)
    return results


def run(out, folders, poses):
    """Write the digest and time of every pose of the saved tasks to `out`."""
    rows = {}
    for name, path in task_files(folders):
        sample = pickle.loads(path.read_bytes())
        for index, (value, seconds) in run_task(sample, pose_indexes(len(sample["arguments"][2]), poses)).items():
            rows[f"{name}/{index}"] = {"digest": value, "seconds": seconds}
    record = {"package": str(Path(Stonepacker2D.__file__).parent), "rows": rows}
    Path(out).write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"{len(rows)} poses from {record['package']}: {sum(r['seconds'] for r in rows.values()):.2f} s")


def difference_lines(old_rows, new_rows):
    """One line for every pose that is in one run only or has another digest."""
    lines = [f"only in one run: {key}" for key in sorted(old_rows.keys() ^ new_rows.keys())]
    shared = sorted(old_rows.keys() & new_rows.keys())
    return lines + [f"differs: {key}" for key in shared if old_rows[key]["digest"] != new_rows[key]["digest"]]


def compare(old_path, new_path):
    """Print the differences and the times of two `run` outputs; returns the exit code."""
    old, new = (json.loads(Path(path).read_text(encoding="utf-8")) for path in (old_path, new_path))
    lines = difference_lines(old["rows"], new["rows"])
    print("\n".join(lines[:50]))
    total_old, total_new = (sum(r["seconds"] for r in run_["rows"].values()) for run_ in (old, new))
    print(f"{len(old['rows'])} poses, {len(lines)} differences; {total_old:.2f} s -> {total_new:.2f} s")
    print(f"old {old['package']}\nnew {new['package']}")
    if os.path.normcase(old["package"]) == os.path.normcase(new["package"]):
        print("both runs used the same Stonepacker2D: this comparison proves nothing")
        return 1
    return 1 if lines else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("run")
    build.add_argument("out")
    build.add_argument("folders", nargs="+")
    build.add_argument("--poses", type=int, default=8)
    check = commands.add_parser("compare")
    check.add_argument("old")
    check.add_argument("new")
    options = parser.parse_args(argv)
    if options.command == "compare":
        return compare(options.old, options.new)
    run(options.out, options.folders, options.poses)
    return 0


if __name__ == "__main__":
    sys.exit(main())
