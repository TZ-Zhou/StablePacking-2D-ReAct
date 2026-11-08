"""Build the walls of a batch: any part of its plan, one wall at a time, resumable.

  python -B -m research.runner.batch plan <config.yaml>
  python -B -m research.runner.batch show <name>
  python -B -m research.runner.batch run <name> [--cases 3-10,15] [--first N] [--last M] [--retry-failed] [--dry-run]

The config file and how cases are drawn: research/walls/batch_plan.py. run builds the cases chosen by --cases, --first
and --last (together: every case any of them names; none: all) and shows [current/total] over the chosen cases with a
progress bar of each wall's stones. status.jsonl gets one line per finished wall, so running the same command again
resumes: a finished wall is skipped, a wall cut off by Ctrl+C or a crash is built again from its first stone in a new
folder, and a failed wall stays failed unless --retry-failed. The exit code is 0 when every chosen case succeeded. No
paid API: the keys are removed from the environment.

Output in runs/batch/<name>/: config.yaml, plan.csv, status.jsonl, launcher.log, tmp/ and the walls in cases/.
"""

import argparse
import contextlib
import json
import traceback
from datetime import datetime

from research.paths import REPO
from research.runner import run_wall
from research.runner.environment import prepare_environment
from research.walls import batch_plan, case_definition

RAISED = "RAISED"
TRACEBACK_TAIL = 900
INTERRUPTED_EXIT = 130
STOPPED_EXIT = 2


def statuses(folder):
    """{case number: its latest record}; a line cut short by a crash names no finished wall."""
    path = folder / "status.jsonl"
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else ():
        with contextlib.suppress(json.JSONDecodeError):
            record = json.loads(line)
            records[record["case"]] = record
    return records


def save_record(folder, record):
    stamped = {**record, "finished": datetime.now().isoformat(timespec="seconds")}
    with (folder / "status.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(stamped) + "\n")


def without_decisions(receipt):
    """True for a controller wall that never had a choice (identical stones): run_wall calls its check incomplete,
    but the wall is as good as any other."""
    reports = (receipt.get("trace_check") or {}).get("controller") or []
    no_choice = all(report["verdict"] == "PARTIAL" and report["decision_points"] == 0 for report in reports)
    return receipt["status"] == "INCOMPLETE_TRACE" and bool(reports) and no_choice


def receipt_record(number, receipt):
    """The status.jsonl line of a built wall."""
    return {
        "case": number,
        "status": receipt["status"],
        "no_decision_points": without_decisions(receipt),
        "case_id": receipt["case_id"],
        "placed": receipt.get("committed_placements"),
        "elapsed_seconds": receipt.get("elapsed_seconds"),
    }


def build_case(batch, case, label, launch):
    """Build one wall and return its record. KeyboardInterrupt leaves no record, so the wall is built again on resume;
    TerminationFailed is recorded and raised, because nothing else may be launched then."""
    number, argv = case["case"], batch_plan.wall_argv(case, batch.config)
    try:
        with (batch.folder / "launcher.log").open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            return receipt_record(number, launch(argv, root=batch.folder / "cases", label=label))
    except run_wall.TerminationFailed as error:
        save_record(batch.folder, receipt_record(number, error.receipt))
        raise
    except (Exception, SystemExit) as error:  # run_wall's argparse refuses an argv with SystemExit
        return {"case": number, "status": RAISED, "error": f"{error!r}\n{traceback.format_exc()[-TRACEBACK_TAIL:]}"}


def succeeded(record):
    return record is not None and (
        record["status"] in case_definition.SUCCESS or record.get("no_decision_points", False)
    )


def is_settled(record, retry_failed):
    """A wall is not built again once it succeeded, nor once it failed unless the failures are to be retried."""
    return succeeded(record) or (record is not None and not retry_failed)


def case_title(case):
    return f"case {case['case']}  {case['wall_type']}  {case['filter_policy']}  seed={case['seed']}"


def outcome_text(record):
    if record["status"] == RAISED:
        return f"{RAISED}: {record['error'].splitlines()[0]}"
    status = f"{record['status']} (no decision points)" if record.get("no_decision_points") else record["status"]
    return f"{status}  {record['placed']} stones  {record['elapsed_seconds']:.0f} s  {record['case_id']}"


def run_cases(batch, cases, retry_failed=False, launch=run_wall.run):
    """Build the chosen cases not settled yet, recording each; returns True when every chosen case has succeeded."""
    records = statuses(batch.folder)
    todo = [case for case in cases if not is_settled(records.get(case["case"]), retry_failed)]
    settled = len(cases) - len(todo)
    print(f"{batch.config['name']}: {len(cases)} cases chosen, {settled} settled before this run", flush=True)
    for position, case in enumerate(todo, settled + 1):
        tag = f"[{position}/{len(cases)}]"
        print(f"{tag} {case_title(case)}", flush=True)
        records[case["case"]] = build_case(batch, case, f"{tag} stones", launch)
        save_record(batch.folder, records[case["case"]])
        print(f"{tag} {outcome_text(records[case['case']])}", flush=True)
    return all(succeeded(records.get(case["case"])) for case in cases)


def show_batch(batch):
    """Print one line per case and how far the batch is."""
    records = statuses(batch.folder)
    done = sum(succeeded(record) for record in records.values())
    print(f"{batch.config['name']}: {len(batch.plan)} cases, {done} succeeded, {len(records) - done} failed")
    for case in batch.plan:
        record = records.get(case["case"])
        print(f"{case_title(case)}  {outcome_text(record) if record else 'not run'}")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Plan and run batches of walls from a config file.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan").add_argument("config", help="YAML file, see the module docstring")
    commands.add_parser("show").add_argument("name")
    run = commands.add_parser("run")
    run.add_argument("name")
    run.add_argument("--cases", help="case numbers, e.g. 3-10,15")
    run.add_argument("--first", type=int, help="the first N cases of the plan")
    run.add_argument("--last", type=int, help="the last M cases of the plan")
    run.add_argument("--retry-failed", action="store_true", help="build the failed walls again")
    run.add_argument("--dry-run", action="store_true", help="print each chosen case's run_wall arguments and stop")
    return parser.parse_args(argv)


def run_command(args):
    """The `run` command; returns the exit code."""
    batch = batch_plan.load_batch(args.name)
    numbers = batch_plan.chosen_numbers(len(batch.plan), args.cases, args.first, args.last)
    cases = [batch.plan[number - 1] for number in numbers]
    if args.dry_run:
        for case in cases:
            print(f"{case['case']}: {' '.join(batch_plan.wall_argv(case, batch.config))}")
        return 0
    (batch.folder / "tmp").mkdir(exist_ok=True)
    prepare_environment(REPO / ".env.local", batch.folder / "tmp")
    try:
        return 0 if run_cases(batch, cases, args.retry_failed) else 1
    except KeyboardInterrupt:
        print(f"interrupted: `run {args.name}` with the same choice resumes", flush=True)
        return INTERRUPTED_EXIT
    except run_wall.TerminationFailed as error:
        print(f"stopped: {error}", flush=True)
        return STOPPED_EXIT


def main(argv=None):
    args = parse_args(argv)
    if args.command == "run":
        return run_command(args)
    if args.command == "plan":
        batch = batch_plan.create_batch(args.config)
        print(f"{len(batch.plan)} cases in {batch.folder}")
        return 0
    show_batch(batch_plan.load_batch(args.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
