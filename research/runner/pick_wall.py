"""Build the LLM endgame pick walls under the guarantees of research/llm_walls/pick_wall.py.

  python -B -m research.runner.pick_wall preflight [--loopback]                  # the next state's wall
  STONEPACKING_ALLOW_PAID_API=1 python -B -m research.runner.pick_wall run --confirm-paid
  python -B -m research.runner.pick_wall run --loopback                          # a local mock server
  python -B -m research.runner.pick_wall ledger [--loopback]                     # what the walls spent

run builds the states of pick_wall.STATES in order, one wall at a time, each named pick-s<seed>-<order> and naming the
one before it (--after); states with a finished wall are skipped, so a run that stopped resumes. It stops at the first
wall that is not finished (uncertain or aborted: inspect it before anything else) and exits 1. Run it in the
stablepacking-baseline environment.
"""

import argparse
import json

from research.llm_walls import gpt_smoke, gpt_wall, pick_wall
from research.runner import gpt_wall as runner


def next_state(loopback):
    """(seed, order, name, after) of the first state without a finished wall, None if all are; the latest wall of
    the study is `after` (journal_problems refuses anything else)."""
    journals = gpt_wall.journals(loopback, pick_wall)
    runs = {n: j for n, j in journals.items() if j}
    after = max(runs, key=lambda n: runs[n].get("started_at", ""), default=None)
    for seed, order in pick_wall.STATES:
        name = pick_wall.name_of(seed, order)
        if (journals.get(name) or {}).get("state") != "finished":
            return seed, order, name, after
    return None


def run(loopback, launch=runner.wall_runner.run):
    """Every remaining state's wall; the reports of the walls built, the last one first if it is not finished."""
    reports = []
    while (state := next_state(loopback)) is not None:
        seed, order, name, after = state
        report = runner.run_wall(name, seed, after, loopback, launch=launch, x=pick_wall,
                                 wall_args=pick_wall.wall_args(order))
        reports.append({"name": name, **{k: v for k, v in report.items() if k != "requests"}})
        if gpt_wall.journals(loopback, pick_wall)[name].get("state") != "finished":
            break
    return reports


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("preflight", "run", "ledger"):
        p = sub.add_parser(cmd)
        p.add_argument("--loopback", action="store_true")
        if cmd == "run":
            p.add_argument("--confirm-paid", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "ledger":
        shown = ("state", "reasons", "spend", "gpt_complete", "case_id", "seed", "wall_args")
        print(json.dumps({"spent": gpt_wall.spent(a.loopback, pick_wall), "limits": pick_wall.EXPERIMENT_LIMITS,
                          "lock": gpt_smoke.read_journal(gpt_wall.lock_of(a.loopback, pick_wall)),
                          "walls": {n: {k: (j or {"state": "no journal"}).get(k) for k in shown}
                                    for n, j in gpt_wall.journals(a.loopback, pick_wall).items()}}, indent=1))
        return 0
    state = next_state(a.loopback)
    if a.cmd == "preflight":
        if state is None:
            print(json.dumps({"ok": True, "problems": [], "next": None}))
            return 0
        seed, order, name, after = state
        problems = gpt_wall.preflight(name, seed, after, a.loopback, x=pick_wall)
        print(json.dumps({"ok": not problems, "problems": problems, "next": name, "after": after,
                          "spent": gpt_wall.spent(a.loopback, pick_wall), "limits": pick_wall.EXPERIMENT_LIMITS},
                         indent=1))
        return 0 if not problems else 1
    if not (a.loopback or a.confirm_paid):
        gpt_smoke.refuse(["run: a paid wall needs --confirm-paid (or --loopback for a local mock server)"])
    reports = run(a.loopback)
    print(json.dumps(reports, indent=1))
    done = gpt_wall.journals(a.loopback, pick_wall)
    return 0 if all((done.get(pick_wall.name_of(*s)) or {}).get("state") == "finished" for s in pick_wall.STATES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
