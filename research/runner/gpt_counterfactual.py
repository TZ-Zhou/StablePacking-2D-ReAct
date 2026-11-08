"""Build, check and summarize the counterfactual branches of a live GPT wall.

  python -B -m research.runner.gpt_counterfactual init            # the study gpt-001-cf, once
  python -B -m research.runner.gpt_counterfactual run [--only V]  # build the walls not built yet, check each, stop at a failure
  python -B -m research.runner.gpt_counterfactual check           # check every built wall (and the reused P6)
  python -B -m research.runner.gpt_counterfactual summarize       # wang-quality-v1 of every wall, G - P per step

What the branches are and how they are checked: research/llm_walls/gpt_counterfactual.py.
"""

import argparse
import json
import os
from datetime import datetime

from research.llm_walls import gpt_counterfactual
from research.runner import run_wall
from research.runner.campaign import launch_case, log_run
from research.storage import studies
from research.walls import campaign, case_definition


def run_plan(plan, only=None, launch=run_wall.run):
    if case_definition.code_fingerprint() != plan["code_sha256"]:
        raise SystemExit("executed code differs from the code registered in the plan")
    if any(os.environ.get(k) for k in gpt_counterfactual.KEY_VARS):
        raise SystemExit(f"unset {', '.join(gpt_counterfactual.KEY_VARS)}: a branch needs no key and no gate")
    for name in plan["build_order"]:
        if only and name not in only:
            continue
        if name != "V":
            v = gpt_counterfactual.check_wall(plan, "V") if gpt_counterfactual.built(plan, "V") else ["V not built"]
            if v:
                raise SystemExit(f"V has not passed: {v[:5]}; no branch is built")
            if (
                "P6" in plan["walls"]
                and plan["walls"]["P6"]["run"] == gpt_counterfactual.PAPER_WALL
                and (p6 := gpt_counterfactual.check_wall(plan, "P6"))
            ):
                raise SystemExit(f"the reused P6 fails its check: {p6[:5]}")
        if gpt_counterfactual.built(plan, name):
            print(f"skip {name}: already built")
            continue
        campaign.require_study(plan["id"])
        started = datetime.now()
        launch_case(plan["id"], gpt_counterfactual.branch_argv(plan, plan["walls"][name]), {"branch": name}, launch)
        problems = gpt_counterfactual.check_wall(plan, name)
        log_run(
            plan["id"],
            {
                "wall": name,
                "case_id": gpt_counterfactual.wall_ref(plan, name),
                "started": started.isoformat(timespec="seconds"),
                "finished": datetime.now().isoformat(timespec="seconds"),
                "problems": problems,
            },
        )
        print(f"{name}:", "OK" if not problems else problems[:10])
        if problems:
            raise SystemExit(f"{name} failed its check: stopped, nothing else is built")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["init", "run", "check", "summarize"])
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    if a.command == "init":
        if gpt_counterfactual.LEGACY_PLAN.exists():
            raise SystemExit(f"{gpt_counterfactual.LEGACY_PLAN} exists: a plan is registered once")
        plan = gpt_counterfactual.make_plan()
        studies.create(gpt_counterfactual.PLAN_ID, "gpt-counterfactual", plan)
        print(json.dumps({k: plan[k] for k in ("recorded_orders", "walls", "build_order")}, indent=1))
        return 0
    plan = gpt_counterfactual.load_plan()
    if a.command == "run":
        run_plan(plan, a.only)
        return 0
    if a.command == "check":
        bad = {}
        for name in plan["walls"]:
            if gpt_counterfactual.built(plan, name):
                bad[name] = gpt_counterfactual.check_wall(plan, name)
                print(name, "OK" if not bad[name] else bad[name][:10])
        return 1 if any(bad.values()) else 0
    print(json.dumps(gpt_counterfactual.summarize(plan), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
