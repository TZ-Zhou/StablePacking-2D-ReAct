"""Build and summarize the walls of a fixed filter-order campaign, one wall at a time.

  python -B -m research.runner.campaign init <id> --case thangavelu-noise80 --arms code_original \
      paper=3,0,2,1,4 alt=2,4,3,1,0 random --n-seeds 3 [--master-seed 20261106] [--timeout 1200]
      [--extends <parent id>] [--note TEXT]
  python -B -m research.runner.campaign run <id> [--dry-run] [--max-runs N]
  python -B -m research.runner.campaign summarize <id>

What a campaign is, its seeds, arms and tables: research/walls/campaign.py. launch_case and log_run are shared by every
runner that builds the walls of a study.
"""

import argparse
import json
from datetime import datetime

from research.runner import run_wall
from research.storage import studies
from research.walls import campaign as definition
from research.walls import case_definition


def launch_case(study_id, argv, keys, launch=run_wall.run):
    """Build one wall of a study with `launch` (run_wall.run: a new case folder) and list it there as
    {'case_id', **keys}; returns its receipt. TERMINATION_FAILED lists the wall, then raises and stops the
    whole batch."""
    try:
        receipt = launch(argv)
    except run_wall.TerminationFailed as error:
        studies.add_case(study_id, {"case_id": error.receipt["case_id"], **keys})
        raise
    studies.add_case(study_id, {"case_id": receipt["case_id"], **keys})
    return receipt


def log_run(study_id, record):
    """Append one record to runs.jsonl beside the study's study.json."""
    with (studies.path_of(study_id).parent / "runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run_campaign(campaign, dry_run=False, max_runs=None, launch=run_wall.run):
    definition.require_study(campaign["id"])
    if case_definition.code_fingerprint() != campaign["code_sha256"]:
        raise RuntimeError("executed code differs from the code registered in the study")
    started = []
    for seed, arm, wall in definition.planned_runs(campaign):
        status = definition.run_status(wall)
        if status is not None:
            print(f"skip {wall}: already {status}")
            continue
        if max_runs is not None and len(started) >= max_runs:
            break
        argv = [
            campaign["case"],
            "--policy",
            arm["policy"],
            "--seed",
            str(seed),
            "--timeout",
            str(campaign["timeout_seconds"]),
        ]
        if arm["order"]:
            argv += ["--order", ",".join(map(str, arm["order"]))]
        if dry_run:
            print("would run", " ".join(argv))
            started.append(f"{arm['name']} seed {seed}")
            continue
        receipt = launch_case(campaign["id"], argv, {"arm": arm["name"], "seed": seed}, launch)
        started.append(receipt["case_id"])
        log_run(
            campaign["id"],
            {
                "run": receipt["case_id"],
                "seed": seed,
                "arm": arm["name"],
                "status": receipt["status"],
                "finished": datetime.now().isoformat(timespec="seconds"),
            },
        )
        if receipt["status"] == "FAILED":
            print(f"{receipt['case_id']} FAILED: campaign stopped for inspection")
            break
    return started


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("id")
    init.add_argument("--case", required=True)
    init.add_argument("--arms", nargs="+", required=True)
    init.add_argument("--n-seeds", type=int, required=True)
    init.add_argument("--master-seed", type=int, default=20261106)
    init.add_argument("--timeout", type=int, default=1200)
    init.add_argument("--extends", help="reuse runs of this campaign for arms and seeds it shares")
    init.add_argument("--note", action="append", default=[], help="recorded in the design, e.g. why an arm was chosen")
    run = sub.add_parser("run")
    run.add_argument("id")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--max-runs", type=int)
    summ = sub.add_parser("summarize")
    summ.add_argument("id")
    args = parser.parse_args(argv)
    if args.command == "init":
        campaign = definition.make_campaign(
            args.id, args.case, args.arms, args.n_seeds, args.master_seed, args.timeout, args.extends, args.note
        )
        definition.register(campaign)
        print(json.dumps({k: campaign[k] for k in ("id", "case", "arms", "seeds")}, indent=2))
    elif args.command == "run":
        run_campaign(definition.load(args.id), args.dry_run, args.max_runs)
    else:
        for row in definition.summarize(definition.load(args.id)):
            print(
                {
                    k: row.get(k)
                    for k in (
                        "arm",
                        "seed",
                        "status",
                        "n_placed",
                        "stone_filling",
                        "lambda_min",
                        "first_divergence_step_vs_code_original",
                    )
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
