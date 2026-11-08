"""Build and check the development walls.

  python -B -m research.runner.dev_walls init <id>       # research/studies/<id>/study.json, once
  python -B -m research.runner.dev_walls run <id>        # builds the walls not built yet, then check
  python -B -m research.runner.dev_walls check [<id>]    # check every wall, Null == Paper, writes manifest.csv

What the walls are and how they are checked: research/llm_walls/dev_walls.py.
"""

import argparse
import json
import sys
from datetime import datetime

from research.llm_walls import dev_walls
from research.runner import run_wall
from research.runner.campaign import launch_case, log_run
from research.walls import campaign, case_definition


def run(study_id, launch=run_wall.run):
    campaign.require_study(study_id)
    c = campaign.load(study_id)
    if case_definition.code_fingerprint() != c["code_sha256"]:
        raise SystemExit("executed code differs from the code registered in the study")
    for seed, arm, wall in campaign.planned_runs(c):
        status = campaign.run_status(wall)
        if status is not None:
            print(f"skip {wall}: already {status}")
            continue
        receipt = launch_case(study_id, dev_walls.argv_of(c, seed, arm), {"arm": arm["name"], "seed": seed}, launch)
        log_run(
            study_id,
            {
                "run": receipt["case_id"],
                "seed": seed,
                "arm": arm["name"],
                "status": receipt["status"],
                "finished": datetime.now().isoformat(timespec="seconds"),
            },
        )
        if receipt["status"] != "PROCESS_PASS":
            raise SystemExit(f"{receipt['case_id']} {receipt['status']}: stopped for inspection")
    return dev_walls.check(study_id)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["init", "run", "check"])
    ap.add_argument(
        "id",
        nargs="?",
        default=dev_walls.ID,
        help=f"study of the dev walls (default {dev_walls.ID}: archived, check only)",
    )
    a = ap.parse_args(argv)
    if a.command == "init":
        print(json.dumps(dev_walls.init(a.id), indent=1))
        return 0
    return 1 if (run(a.id) if a.command == "run" else dev_walls.check(a.id)) else 0


if __name__ == "__main__":
    sys.exit(main())
