"""Prove that walls built by the persistent worker pool equal walls built the old way, and time both.

Usage (from the repository root, in the baseline environment):
  python -B -m research.runner.verify_speedup <label> [--scenes a,b] [--policies paper,code_original] [--placements N]
                                            [--modes fresh,persistent] [--repeat R] [--workers W] [--timeout S]
                                            [--golden RUN]

The comparison of docs/plans/case-speedup-2s-per-stone.md. For each scene and policy the modes are built in the
given order, R times over (R=2: fresh, persistent, fresh, persistent), one wall at a time in this process. Mode fresh
is run_wall --fresh-pool, mode persistent is run_wall's default; --placements 0 builds the full wall.

Every wall is compared with the first good wall of its scene and policy, or with that scene and policy's wall of an
earlier run named by --golden, and every full persistent code_original wall also with its archived wall in
research/runs, which is only read. No paid API: the keys are removed from the
environment and .env.local gives only the MOSEK license and PYTHONNOUSERSITE.

Output in runs/speedup/<label>/: the case folders, tmp/ for temporary files, report.json (one row per wall and a
summary) replaced after every wall, and done.txt written last. The exit code is 0 when every scene and policy had a
wall to compare, and every wall raised nothing, showed no difference, also from its archive, and passed check_run.
"""

import argparse
import json
import os
import traceback
from unittest import mock

from research.paths import REPO, SPEEDUP_RUNS
from research.perf import verify_speedup
from research.runner import run_wall
from research.runner.environment import prepare_environment
from research.storage import case_id
from research.walls import case_definition

WORKERS_CALL = "set_number_cpu(10)"
TRACEBACK_TAIL = 900


def names_in(allowed):
    """An argparse type: a comma-separated list whose items are all in `allowed`."""

    def parse(text):
        names = text.split(",")
        unknown = [name for name in names if name not in allowed]
        if unknown:
            raise argparse.ArgumentTypeError(f"unknown {unknown}; choose from {list(allowed)}")
        return names

    return parse


def parse_args(argv):
    """The command line; an unknown scene, policy or mode is refused here, before any wall is built."""
    parser = argparse.ArgumentParser(description="Build walls in both worker-pool modes, compare and time them.")
    parser.add_argument("label", help="output folder under runs/speedup")
    parser.add_argument("--scenes", type=names_in(case_definition.CASES), default=list(case_definition.CASES))
    parser.add_argument("--policies", type=names_in(verify_speedup.POLICIES), default=list(verify_speedup.POLICIES))
    parser.add_argument("--placements", type=int, default=0, help="stop each wall after N placements (0 = full wall)")
    parser.add_argument(
        "--modes", type=names_in(verify_speedup.WORKER_POOLS), default=list(verify_speedup.WORKER_POOLS)
    )
    parser.add_argument("--repeat", type=int, default=1, help="how many times every mode is built")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=3600, help="seconds allowed per wall")
    parser.add_argument("--golden", help="run folder of walls every wall must equal: a label or an absolute path")
    return parser.parse_args(argv)


def with_workers(source_text, workers):
    """`source_text` staging `workers` workers; the size of the pool does not decide the wall."""

    def staged(case):
        text = source_text(case)
        assert WORKERS_CALL in text, case
        return text.replace(WORKERS_CALL, f"set_number_cpu({workers})")

    return staged


def build_wall(argv, root, workers):
    """One wall in a new case folder under `root`, with `workers` workers; returns its receipt."""
    staged = with_workers(case_definition.source_text, workers)
    with mock.patch.object(case_definition, "source_text", staged), mock.patch.object(case_id, "CASES_DIR", root):
        return run_wall.run(argv)


def described(error):
    """The type and message of the exception being handled, and the end of its traceback."""
    return f"{type(error).__name__}: {error}\n{traceback.format_exc()[-TRACEBACK_TAIL:]}"


def verified_row(entry, options, root, first):
    """Build and check one wall; its row records any failure, so that only KeyboardInterrupt gets through."""
    row = verify_speedup.blank_row(entry)
    try:
        receipt = build_wall(
            verify_speedup.wall_argv(entry, options.placements, options.timeout), root, options.workers
        )
        row.update(verify_speedup.receipt_fields(receipt))
        row["error"] = verify_speedup.receipt_problem(receipt, row["mode"])
        if row["error"] is None:
            verify_speedup.fill_checks(row, root / receipt["case_id"], first, options.placements)
    except run_wall.TerminationFailed as error:
        row.update(verify_speedup.receipt_fields(error.receipt), error=described(error))
    except (Exception, SystemExit) as error:  # run_wall's argparse refuses an argv with SystemExit
        row["error"] = described(error)
    return row


def save_report(folder, report):
    """Replace report.json in one step, so a run killed during the write keeps the previous version."""
    text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    target, partial = folder / "report.json", folder / "report.json.tmp"
    partial.write_text(text, encoding="utf-8")
    try:
        os.replace(partial, target)
    except PermissionError:  # on Windows a reader or virus scanner holding either file must not end a long run
        target.write_text(text, encoding="utf-8")


def run_matrix(options, folder):
    """Build every planned wall in order, replacing report.json after each one; returns the rows."""
    rows, first_walls = [], verify_speedup.golden_walls(SPEEDUP_RUNS / options.golden) if options.golden else {}
    for entry in verify_speedup.planned_walls(options.scenes, options.policies, options.modes, options.repeat):
        key = entry["scene"], entry["policy"]
        row = verified_row(entry, options, folder, first_walls.get(key))
        rows.append(row)
        if row["differences"] != [verify_speedup.NOT_COMPARED]:
            first_walls.setdefault(key, folder / row["case_id"])
        save_report(folder, verify_speedup.report_of(options, rows))
    return rows


def main(argv=None):
    """Run the matrix of the command line; returns the exit code described in the module docstring."""
    options = parse_args(argv)
    folder = SPEEDUP_RUNS / options.label
    tmp = folder / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    (folder / "done.txt").unlink(missing_ok=True)
    prepare_environment(REPO / ".env.local", tmp)
    rows = run_matrix(options, folder)
    (folder / "done.txt").write_text("done\n", encoding="utf-8")
    return verify_speedup.exit_code(rows)


if __name__ == "__main__":
    raise SystemExit(main())
