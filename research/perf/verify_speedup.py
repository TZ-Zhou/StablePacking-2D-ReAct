"""What the speedup acceptance builds and compares: the planned walls and whether two walls are the same wall.

research/runner/verify_speedup.py builds the walls (usage and output there). Every wall is compared with the first good
wall of its scene and policy, and every full persistent code_original wall also with its archived wall in research/runs,
which is only read.
"""

import json
import math
import statistics

import numpy as np

from research.paths import LEGACY_RUNS
from research.perf import profile_report
from research.walls import case_definition, check_run, wall_metrics

POLICIES = {
    "paper": ("--policy", "paper", "--order", "3,0,2,1,4"),
    "code_original": ("--policy", "code_original"),
}
MODE_FLAGS = {"fresh": ("--fresh-pool",), "persistent": ()}
WORKER_POOLS = {"fresh": "per_batch", "persistent": "persistent"}
ARCHIVED_WALLS = {
    "thangavelu-noise80": "thangavelu-noise80-code-original-full-01",
    "bricks": "bricks-code-original-trace-02",
    "almeida-pr": "pr-code-original-full-01",
    "fft-d2-0": "fft-d2-0-code-original-notrace-01",
}
NOT_COMPARED = "not compared: see error"
UNTRACED_ARCHIVE = "trace/commits.jsonl: present in only one run"
RAISED = "RAISED"
PREFIX_RECORDS = ("decisions.txt", "filter.txt", "trace/commits.jsonl")
LIMIT_LOAD_TOLERANCE = 1e-12  # MOSEK's default path varies in the last bits between two runs of the same wall


def planned_walls(scenes, policies, modes, repeat):
    """One entry per wall in build order: per scene and policy, every mode in the given order, `repeat` times over."""
    return [
        {"scene": scene, "policy": policy, "mode": mode, "repeat": number}
        for scene in scenes
        for policy in policies
        for number in range(1, repeat + 1)
        for mode in modes
    ]


def wall_argv(entry, placements, timeout):
    """The run_wall.run argv of one planned wall."""
    return [
        entry["scene"],
        *POLICIES[entry["policy"]],
        "--placements",
        str(placements),
        "--timeout",
        str(timeout),
        *MODE_FLAGS[entry["mode"]],
    ]


def blank_row(entry):
    """The row of a wall nothing is known about yet; `differences` stays non-empty until a comparison has run."""
    return {
        **entry,
        "case_id": None,
        "status": RAISED,
        "elapsed_seconds": None,
        "placed": None,
        "seconds_per_stone": None,
        "stones": None,
        "trace_ok": None,
        "differences": [NOT_COMPARED],
        "compared_to": None,
        "check_ok": None,
        "archive_differences": None,
        "error": None,
    }


def receipt_fields(receipt):
    """The row fields that come straight from a receipt."""
    return {
        "case_id": receipt["case_id"],
        "status": receipt["status"],
        "elapsed_seconds": receipt.get("elapsed_seconds"),
        "placed": receipt.get("committed_placements"),
        "trace_ok": (receipt.get("trace_check") or {}).get("ok"),
    }


def receipt_problem(receipt, mode):
    """Why a wall proves nothing about `mode`, or None: its receipt names another pool, or the wall did not succeed."""
    pool = receipt.get("worker_pool")
    if pool != WORKER_POOLS[mode]:
        return f"receipt worker_pool {pool!r} contradicts mode {mode}"
    if receipt["status"] not in case_definition.SUCCESS:
        return f"status {receipt['status']}"
    return None


def archive_of(row, placements):
    """The archived wall a row is compared with, or None: only full persistent code_original walls have one."""
    eligible = not placements and row["policy"] == "code_original" and row["mode"] == "persistent"
    return LEGACY_RUNS / ARCHIVED_WALLS[row["scene"]] if eligible and row["scene"] in ARCHIVED_WALLS else None


def record_rows(path):
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def record_difference(name, a, b):
    """What is wrong with the rows of one record of two stopped walls, or None: a wall without rows proves nothing."""
    if not a or not b:
        return f"{name}: no rows in one of the walls"
    return f"{name}: the rows both walls have differ" if a[: len(b)] != b[: len(a)] else None


def result_prefix_differences(x, y):
    """The records and trace events two result folders have in common must be equal."""
    records = (record_difference(name, record_rows(x / name), record_rows(y / name)) for name in PREFIX_RECORDS)
    found = [line for line in records if line]
    if (x / "trace" / "commits.jsonl").exists() and (y / "trace" / "commits.jsonl").exists():
        found += check_run.event_differences(check_run.load_result(x)[0], check_run.load_result(y)[0])
    return found


def stopped_differences(a, b):
    """Differences between two walls stopped early: the launcher polls, so each may overrun by a few stones."""
    ra, rb = check_run.result_dirs(a), check_run.result_dirs(b)
    if len(ra) != len(rb):
        return [f"{len(ra)} vs {len(rb)} result directories"]
    return [found for x, y in zip(ra, rb) for found in result_prefix_differences(x, y)]


def same_limit_load(a, b):
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=LIMIT_LOAD_TOLERANCE)


def limit_load_differences(a, b):
    """The limit loads of two full walls, compared within the tolerance of the solver."""
    found = []
    for x, y in zip(check_run.result_dirs(a), check_run.result_dirs(b)):
        la, lb = wall_metrics.mechanism_lambdas(x), wall_metrics.mechanism_lambdas(y)
        found += [
            f"limit load {side}: {la[side]} vs {lb[side]}" for side in la if not same_limit_load(la[side], lb[side])
        ]
    return found


def wall_differences(wall, first, placements):
    """What keeps `wall` from being the same wall as `first`; empty means identical."""
    if placements:
        return stopped_differences(wall, first)
    return check_run.compare_runs(wall, first) + limit_load_differences(wall, first)


def archive_differences(wall, archive):
    """Differences from an archived wall; a wall archived without tracing has no commits to compare with."""
    found = check_run.compare_runs(wall, archive)
    return [line for line in found if line != UNTRACED_ARCHIVE] if "notrace" in archive.name else found


def golden_walls(folder):
    """{(scene, policy): wall folder} of an earlier run of this tool, from the walls whose own checks passed."""
    rows = json.loads((folder / "report.json").read_text(encoding="utf-8"))["rows"]
    return {
        (row["scene"], row["policy"]): folder / row["case_id"]
        for row in rows
        if row["error"] is None and row["check_ok"] and row["status"] in case_definition.SUCCESS
    }


def stone_summary(wall):
    """Count, mean, median, p90, maximum and threshold counts of a wall's seconds per stone, without the first stone."""
    seconds = wall_metrics.stone_seconds(wall / "events.jsonl")
    if not seconds:
        return None
    return {
        "counted": len(seconds),
        "mean": round(statistics.mean(seconds), 3),
        "median": round(statistics.median(seconds), 3),
        "p90": round(float(np.percentile(seconds, 90)), 3),
        "max": max(seconds),
        "over_2s": sum(value > 2 for value in seconds),
        "over_half_s": sum(value > 0.5 for value in seconds),
    }


def fill_checks(row, wall, first, placements):
    """Measure a built wall into its row, field by field so a later check that raises keeps the earlier results.

    `first` is the folder of the wall this one must equal, None when this one is the first of its scene and policy.
    """
    row["seconds_per_stone"] = profile_report.seconds_per_stone(wall)
    row["stones"] = stone_summary(wall)
    row["differences"] = [] if first is None else wall_differences(wall, first, placements)
    row["compared_to"] = None if first is None else first.name
    row["check_ok"] = check_run.check_run(wall)["ok"]
    archive = archive_of(row, placements)
    if archive is not None:
        row["archive_differences"] = archive_differences(wall, archive)


def worst_seconds(rows, mode):
    """The largest seconds per stone of a mode, None when no wall of that mode was timed."""
    seconds = (row["seconds_per_stone"] for row in rows if row["mode"] == mode and row["seconds_per_stone"] is not None)
    return max(seconds, default=None)


def group_summary(rows):
    """The summary of the rows of one scene and policy."""
    return {
        "worst_seconds_per_stone": {mode: worst_seconds(rows, mode) for mode in dict.fromkeys(r["mode"] for r in rows)},
        "rows_with_differences": sum(bool(row["differences"]) for row in rows),
        "rows_with_archive_differences": sum(bool(row["archive_differences"]) for row in rows),
    }


def summarize(rows):
    """{scene: {policy: group summary}} in the order the walls were built."""
    groups = {}
    for row in rows:
        groups.setdefault(row["scene"], {}).setdefault(row["policy"], []).append(row)
    return {
        scene: {policy: group_summary(group) for policy, group in by_policy.items()}
        for scene, by_policy in groups.items()
    }


def report_of(options, rows):
    """The content of report.json; the summary comes last."""
    return {"settings": vars(options), "rows": rows, "summary": summarize(rows)}


def row_passed(row):
    """Nothing raised, the wall equals what it was compared with and check_run accepts it."""
    return row["error"] is None and not row["differences"] and row["check_ok"] and not row["archive_differences"]


def exit_code(rows):
    """0 when every wall passed and every scene and policy had a wall to compare; a group of one proves nothing."""
    groups = {(row["scene"], row["policy"]) for row in rows}
    compared = {(row["scene"], row["policy"]) for row in rows if row["compared_to"]}
    return 0 if compared == groups and all(map(row_passed, rows)) else 1
