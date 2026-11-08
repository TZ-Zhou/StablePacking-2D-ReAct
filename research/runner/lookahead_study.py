"""Build the lookahead walls of the P1 study and compare them with the final walls.

  python -B -m research.runner.lookahead_study run [--max-runs N]   builds the lookahead walls
  python -B -m research.runner.lookahead_study evaluate --out DIR    compares them with the final walls

What the study is and how a lookahead wall is valued: research/analysis/lookahead_study.py.
"""

import argparse
import json
import sys

from research.analysis import lookahead_study
from research.runner import run_wall
from research.runner.campaign import launch_case
from research.storage import case_id as cid
from research.walls import case_definition


def run(max_runs=None, launch=run_wall.run):
    """Build the lookahead walls not built yet; False at the first one that fails or differs from its wall."""
    lookahead_study.open_study()
    started = 0
    for (parent, step), alts in lookahead_study.points().items():
        n = lookahead_study.placed_before(cid.wall_dir(parent), step) + 1 + lookahead_study.K + 1
        for alt in alts:
            if lookahead_study.lookahead_wall(parent, step, alt) is not None:
                continue
            if max_runs is not None and started >= max_runs:
                return True
            args = [
                lookahead_study.CASE,
                "--parent",
                parent,
                "--policy",
                "paper",
                "--order",
                lookahead_study.ORDER,
                "--seed",
                str(alt["seed"]),
                "--timeout",
                str(lookahead_study.TIMEOUT),
                "--placements",
                str(n),
            ]
            args += ["--switch", alt["switch"]] if alt["switch"] else []
            receipt = launch_case(
                lookahead_study.STUDY,
                args,
                {"point": lookahead_study.point_name(parent, step), "alt": alt["name"]},
                launch,
            )
            started += 1
            built = receipt["status"] in case_definition.SUCCESS
            bad = (
                lookahead_study.same_events(cid.wall_dir(receipt["case_id"]), cid.wall_dir(alt["wall"]))
                if built
                else []
            )
            print(json.dumps({"run": receipt["case_id"], "status": receipt["status"], "differs": bad[:3]}), flush=True)
            if not built or bad:
                print("stopped for inspection")
                return False
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run").add_argument("--max-runs", type=int)
    sub.add_parser("evaluate").add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return 0 if run(a.max_runs) else 1
    lookahead_study.evaluate(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
