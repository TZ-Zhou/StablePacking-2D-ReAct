"""Build the LLM endgame pick walls under the guarantees of research/llm_walls/pick_wall.py.

  python -B -m research.runner.pick_wall preflight [--loopback]                  # the next state's wall
  STONEPACKING_ALLOW_PAID_API=1 python -B -m research.runner.pick_wall run --confirm-paid
  python -B -m research.runner.pick_wall run --loopback                          # a local mock server
  python -B -m research.runner.pick_wall ledger [--loopback]                     # what the walls spent

--lookahead: the walls of research/llm_walls/lookahead_wall.py (docs/plans/llm-endgame-lookahead.md, study
lookahead-walls) instead of pick_wall.py's (study pick-walls), the same commands. --llc control|note: the walls of
research/llm_walls/llc_control_wall.py or llc_note_wall.py (docs/plans/llm-endgame-llc-note.md); run --llc both builds
the two batches in turn, state by state, the control first, and stops at the first wall that is not finished.
--whole: the whole-wall picks of research/llm_walls/whole_wall.py (docs/plans/llm-whole-wall-pick.md, study
whole-walls), one per (seed, scene); --whole --confirm: the same on the held-out seeds, the walls of
research/llm_walls/whole_confirm_wall.py (docs/plans/llm-whole-wall-confirm.md, study whole-confirm-walls).

run builds the states of pick_wall.STATES in order, one wall at a time, each named pick-s<seed>-<order> and naming the
one before it (--after); states with a finished wall are skipped, so a run that stopped resumes. It stops at the first
wall that is not finished (uncertain or aborted: inspect it before anything else) and exits 1. Run it in the
stablepacking-baseline environment.
"""

import argparse
import json

from research.llm_walls import (
    gpt_smoke,
    gpt_wall,
    llc_control_wall,
    llc_note_wall,
    lookahead_wall,
    pick_wall,
    whole_confirm_wall,
    whole_wall,
)
from research.runner import gpt_wall as runner


def next_state(loopback, x=pick_wall):
    """(seed, order, name, after) of the first state without a finished wall, None if all are; the latest wall of
    the study is `after` (journal_problems refuses anything else)."""
    journals = gpt_wall.journals(loopback, x)
    runs = {n: j for n, j in journals.items() if j}
    after = max(runs, key=lambda n: runs[n].get("started_at", ""), default=None)
    for seed, order in x.STATES:
        name = x.name_of(seed, order)
        if (journals.get(name) or {}).get("state") != "finished":
            return seed, order, name, after
    return None


LLC = {"control": llc_control_wall, "note": llc_note_wall}


def run(loopback, launch=runner.wall_runner.run, x=pick_wall, limit=None):
    """Every remaining state's wall (at most limit); the reports of the walls built, the last one first if it is not
    finished."""
    reports = []
    while (limit is None or len(reports) < limit) and (state := next_state(loopback, x)) is not None:
        seed, order, name, after = state
        report = runner.run_wall(name, seed, after, loopback, launch=launch, x=x, wall_args=x.wall_args(order))
        reports.append({"name": name, **{k: v for k, v in report.items() if k != "requests"}})
        if gpt_wall.journals(loopback, x)[name].get("state") != "finished":
            break
    return reports


def run_pair(loopback, launch=runner.wall_runner.run, xs=(llc_control_wall, llc_note_wall)):
    """The batches of xs in turn, one wall each, until all are done or a wall is not finished."""
    reports = []
    while any(next_state(loopback, x) is not None for x in xs):
        for x in xs:
            if next_state(loopback, x) is None:
                continue
            reports += run(loopback, launch, x, limit=1)
            if gpt_wall.journals(loopback, x)[reports[-1]["name"]].get("state") != "finished":
                return reports
    return reports


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("preflight", "run", "ledger"):
        p = sub.add_parser(cmd)
        p.add_argument("--loopback", action="store_true")
        p.add_argument("--lookahead", action="store_true")
        p.add_argument("--whole", action="store_true")
        p.add_argument("--confirm", action="store_true")
        p.add_argument("--llc", choices=(*LLC, "both") if cmd == "run" else tuple(LLC))
        if cmd == "run":
            p.add_argument("--confirm-paid", action="store_true")
    a = ap.parse_args(argv)
    if a.confirm and not a.whole:
        ap.error("--confirm goes with --whole")
    x = LLC.get(a.llc) or (whole_confirm_wall if a.confirm else whole_wall if a.whole else
                           lookahead_wall if a.lookahead else pick_wall)
    if a.cmd == "ledger":
        shown = ("state", "reasons", "spend", "gpt_complete", "case_id", "seed", "wall_args")
        print(json.dumps({"spent": gpt_wall.spent(a.loopback, x), "limits": x.EXPERIMENT_LIMITS,
                          "lock": gpt_smoke.read_journal(gpt_wall.lock_of(a.loopback, x)),
                          "walls": {n: {k: (j or {"state": "no journal"}).get(k) for k in shown}
                                    for n, j in gpt_wall.journals(a.loopback, x).items()}}, indent=1))
        return 0
    state = next_state(a.loopback, x)
    if a.cmd == "preflight":
        if state is None:
            print(json.dumps({"ok": True, "problems": [], "next": None}))
            return 0
        seed, order, name, after = state
        problems = gpt_wall.preflight(name, seed, after, a.loopback, x=x)
        print(json.dumps({"ok": not problems, "problems": problems, "next": name, "after": after,
                          "spent": gpt_wall.spent(a.loopback, x), "limits": x.EXPERIMENT_LIMITS},
                         indent=1))
        return 0 if not problems else 1
    if not (a.loopback or a.confirm_paid):
        gpt_smoke.refuse(["run: a paid wall needs --confirm-paid (or --loopback for a local mock server)"])
    xs = tuple(LLC.values()) if a.llc == "both" else (x,)
    reports = run_pair(a.loopback, xs=xs) if a.llc == "both" else run(a.loopback, x=x)
    print(json.dumps(reports, indent=1))
    return 0 if all((gpt_wall.journals(a.loopback, x).get(x.name_of(*s)) or {}).get("state") == "finished"
                    for x in xs for s in x.STATES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
