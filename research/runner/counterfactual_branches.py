"""Plan, build and summarize one-step filter-order branches of a recorded paper-order wall.

  python -B -m research.runner.counterfactual_branches options <run> <target stone> [...]
  python -B -m research.runner.counterfactual_branches init <plan id> --wall <run>:<seed>:<target> [...] [--timeout 2400]
  python -B -m research.runner.counterfactual_branches run <plan id> [--max-runs N]
  python -B -m research.runner.counterfactual_branches summarize <plan id>
(<run> is a recorded wall: a case id of research/cases or a path in the research/runs archive.)

What a branch is and how it is checked: research/analysis/counterfactual_branches.py.
"""

import argparse

import pandas as pd

from research.analysis import counterfactual_branches
from research.runner import run_wall
from research.runner.campaign import launch_case, log_run
from research.storage import case_id as cid
from research.walls import campaign, case_definition


def run_plan(plan, max_runs=None, launch=run_wall.run):
    """True when every branch built so far passed; stops at the first failure."""
    from research.walls import check_run

    campaign.require_study(plan["id"])
    if case_definition.code_fingerprint() != plan["code_sha256"]:
        raise RuntimeError("executed code differs from the code registered in the plan")
    started = 0
    for b in plan["branches"]:
        if campaign.run_status(counterfactual_branches.branch_wall(plan, b)) is not None:
            continue
        if max_runs is not None and started >= max_runs:
            break
        argv = [
            plan["case"],
            "--parent",
            b["run"],
            "--policy",
            "paper",
            "--order",
            b["recorded_order"].replace("-", ","),
            "--seed",
            str(b["seed"]),
            "--timeout",
            str(plan["timeout_seconds"]),
            "--switch",
            f"{b['step']}={b['order'].replace('-', ',')}",
        ]
        receipt = launch_case(plan["id"], argv, {"branch": b["name"]}, launch)
        started += 1
        record = {
            "branch": b["name"],
            "case_id": receipt["case_id"],
            "status": receipt["status"],
            "check_run_ok": False,
            "check_branch_ok": False,
        }
        if receipt["status"] == "PROCESS_PASS":
            original, branch = (check_run.result_dirs(cid.wall_dir(r))[0] for r in (b["run"], receipt["case_id"]))
            run_check = check_run.check_run(cid.wall_dir(receipt["case_id"]))
            bc = counterfactual_branches.check_branch(
                original, branch, {b["step"]: tuple(int(c) for c in b["order"].split("-"))}
            )
            chosen = bc["switched_choices"].get(b["step"])
            if chosen is None or list(chosen) != b["pose"]:
                bc["ok"] = False
                bc["problems"].append(f"step {b['step']}: chose {chosen}, planned {b['pose']}")
            record.update(
                check_run_ok=run_check["ok"],
                check_branch_ok=bc["ok"],
                problems=run_check["problems"][:5] + bc["problems"][:5],
                identical_prefix_attempts=bc["identical_prefix_attempts"],
            )
        log_run(plan["id"], record)
        if not (record["check_run_ok"] and record["check_branch_ok"]):
            print(f"{b['name']}: {record}; plan stopped for inspection")
            return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    opt = sub.add_parser("options")
    opt.add_argument("run")
    opt.add_argument("targets", nargs="+", type=int)
    init = sub.add_parser("init")
    init.add_argument("id")
    init.add_argument("--wall", action="append", required=True, metavar="RUN:SEED:TARGET")
    init.add_argument("--timeout", type=int, default=2400)
    run = sub.add_parser("run")
    run.add_argument("id")
    run.add_argument("--max-runs", type=int)
    summ = sub.add_parser("summarize")
    summ.add_argument("id")
    args = parser.parse_args(argv)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    if args.command == "options":
        result = next(cid.wall_dir(args.run).rglob("trace/commits.jsonl")).parent.parent
        print(
            counterfactual_branches.summarize(
                counterfactual_branches.step_options(result, set(args.targets))
            ).to_string(index=False)
        )
        return 0
    if args.command == "init":
        walls = [(run, int(seed), int(target)) for run, seed, target in (w.rsplit(":", 2) for w in args.wall)]
        plan = counterfactual_branches.make_plan(args.id, walls, args.timeout)
        counterfactual_branches.register(plan)
        print(pd.DataFrame(plan["branches"]).drop(columns=["name", "run"]).to_string(index=False))
        return 0
    plan = counterfactual_branches.load_plan(args.id)
    if args.command == "run":
        return 0 if run_plan(plan, args.max_runs) else 1
    print(counterfactual_branches.summarize_plan(plan).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
