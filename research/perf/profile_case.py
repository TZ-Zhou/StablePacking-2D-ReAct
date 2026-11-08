"""Run an original test case once with stage timers (untraced, outside the registered run_wall receipt chain).

Usage: python -B -m research.perf.profile_case <case> <run name> [--policy code_original|paper] [--order 3,0,2,1,4]
                                              [--seed S] [--persistent-pool] [--pool-size N] [--cprofile]
Then:  python -B -m research.perf.profile_report show <run name>

Staging follows run_wall (same CASES, same stage(), copied inputs, shell cleanup refused); only the runtime that the
staged case imports changes: perf_runtime.py (timers, optional persistent pool) on top of the registered
windows_case_runtime.py. Output goes to a new research/runs/<name>: perf.json (launcher record), perf/ (timer output),
events.jsonl, execution.log and the case's own result directory. The case runs without trace and without a controller,
so nothing here is a registered run; to check that a pool mode leaves the result unchanged, run the same case, policy,
order and seed twice and compare: python -B -m research.walls.check_run <run a> --compare <run b>.
The case needs MOSEK: load .env.local first (. .\\research\\load_env.ps1); the API keys are dropped from the case
environment (run_wall.quiet_env).
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from research.paths import REPO, VALIDATION
from research.walls import run_wall

RUNTIME_IMPORT = "from windows_case_runtime import configure"
PERF_RUNTIME = Path(__file__).with_name("perf_runtime.py")
PYTHON_HASH_SEED = "20"  # run_wall's launcher value


def stage_perf(source, filter_policy, seed):
    """run_wall.stage with the runtime import pointed at perf_runtime (which applies the registered one first)."""
    staged = run_wall.stage(
        source, {"filter": filter_policy, "run_id": None, "controller": None, "worker_venv": False}, seed=seed
    )
    assert RUNTIME_IMPORT in staged, "staged case has no runtime import"
    return staged.replace(RUNTIME_IMPORT, "from perf_runtime import configure", 1)


def case_environment(output, replicates, args):
    perf = {
        "PERF_DIR": str(output / "perf"),
        "PERF_PERSISTENT": "1" if args.persistent_pool else "0",
        "PERF_CPROFILE": "1" if args.cprofile else "0",
        "PERF_POOL_SIZE": str(args.pool_size),
    }
    return {
        **run_wall.quiet_env(),
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(output / "matplotlib-config"),
        "PYTHONHASHSEED": PYTHON_HASH_SEED,
        "PYTHONDONTWRITEBYTECODE": "1",
        "BASELINE_CASE_EVENTS": str(output / "events.jsonl"),
        "BASELINE_SELECT_FIRST_WALL": "1" if replicates > 1 else "0",
        **perf,
    }


def stage_inputs(output, source_name, filter_policy, seed):
    original = REPO / source_name
    target = output / original.name
    target.write_text(stage_perf(original.read_text(encoding="utf-8"), filter_policy, seed), encoding="utf-8")
    shutil.copy2(VALIDATION / "windows_case_runtime.py", output / "windows_case_runtime.py")
    shutil.copy2(PERF_RUNTIME, output / "perf_runtime.py")
    shutil.copytree(original.parent / "data", output / "data")
    if (original.parent / "config.json").exists():
        shutil.copy2(original.parent / "config.json", output / "config.json")
    return target


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=run_wall.CASES)
    parser.add_argument("name")
    parser.add_argument("--policy", default="code_original", choices=["code_original", "paper"])
    parser.add_argument("--order", type=lambda s: tuple(int(c) for c in s.split(",")))
    parser.add_argument("--seed", type=int, help="seed_everything(seed) instead of the source default (20)")
    parser.add_argument("--persistent-pool", action="store_true", help="reuse one worker pool for the whole case")
    parser.add_argument("--pool-size", type=int, default=0, help="persistent pool size (0: the case set_number_cpu)")
    parser.add_argument(
        "--cprofile", action="store_true", help="cProfile every worker (slower; for function-level data)"
    )
    return parser.parse_args(argv)


def run_case(args, output, target, replicates):
    """Run the staged case in its directory; returns (launch epoch, wall seconds, return code)."""
    launched, start = time.time(), time.monotonic()
    with (output / "execution.log").open("w", encoding="utf-8") as log:
        returncode = subprocess.run(
            [sys.executable, "-B", "-u", str(target)],
            cwd=output,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=case_environment(output, replicates, args),
        ).returncode
    return launched, round(time.monotonic() - start, 3), returncode


def main(argv=None):
    args = parse_args(argv)
    assert Path(sys.prefix).resolve() == run_wall.ENV_PREFIX, sys.prefix
    output = run_wall.prepare_output(args.name)
    source_name, replicates = run_wall.CASES[args.case]
    filter_policy = {"policy": args.policy, "order": args.order, "tol": 1e-9}
    target = stage_inputs(output, source_name, filter_policy, args.seed)
    launched, wall_seconds, returncode = run_case(args, output, target, replicates)
    record = {
        "case": args.case,
        "run_id": args.name,
        "filter_policy": filter_policy,
        "seed": args.seed,
        "persistent_pool": args.persistent_pool,
        "pool_size": args.pool_size,
        "cprofile": args.cprofile,
        "launch_epoch": launched,
        "wall_seconds": wall_seconds,
        "returncode": returncode,
    }
    (output / "perf.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0 if returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
