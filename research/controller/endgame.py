"""A non-LLM controller that picks a candidate directly in the endgame, and its check.

Spec: {"kind": "endgame", "mode": M, "seed": S}. While stones are still drawn from the inventory it never acts: the
wall follows the base policy (the receipt's paper order). At a selection whose inventory is empty (every remaining
stone is on offer; TH_T80's last 7 placements) with two or more feasible rows (LLC > 0 and five finite scores,
selection.paper_feasible) it returns the base policy plus {'pick': row}, by mode:
- none: never acts (the controller is attached, the wall is the base policy's);
- random: a uniformly random feasible row, from numpy default_rng(seed), one draw per acting call;
- llc: the highest LLC;
- borda: the highest sum over the five scores of the feasible rows' ranks (ties share the mean rank);
- low_top: the lowest top y + height of the pose before stabilisation.
Ties go to the lowest row. Every call writes a record to trace/controller.jsonl (order_controller.write_record).

check() re-executes the receipt's spec on the traced inputs (order_controller.context_from_trace): the records must
equal the re-execution, each traced policy must be the base plus the record's pick, replaying select with it must
give the traced selection, and the trace chain and run manifest must be complete (the order controller's checks).
"""

import json

import numpy as np
from scipy.stats import rankdata

from Stonepacker2D.stack.selection import paper_feasible, select

MODES = ("none", "random", "llc", "borda", "low_top")


def full_spec(spec):
    """The spec with its defaults; unknown keys, modes or seeds raise."""
    if not isinstance(spec, dict) or spec.get("kind") != "endgame" or set(spec) - {"kind", "mode", "seed"}:
        raise ValueError(f'an endgame spec is {{"kind": "endgame", "mode": ..., "seed": ...}}, got {spec!r}')
    full = {"kind": "endgame", "mode": spec.get("mode"), "seed": spec.get("seed")}
    if full["mode"] not in MODES:
        raise ValueError(f'endgame mode is one of {MODES}, got {full["mode"]!r}')
    if (full["mode"] == "random") != (type(full["seed"]) is int):
        raise ValueError(f'the random mode needs an int seed and the others none, got {full["seed"]!r}')
    return full


def choose(mode, raw, feasible, y, height, rng):
    """Row picked among the feasible rows of raw (features_raw)."""
    rows = np.flatnonzero(feasible)
    if mode == "random":
        return int(rows[rng.integers(len(rows))])
    if mode == "llc":
        score = raw[rows, 3]
    elif mode == "borda":
        score = sum(rankdata(raw[rows, k]) for k in range(5))
    else:  # low_top
        score = -(np.asarray(y, dtype=float)[rows] + np.asarray(height, dtype=float)[rows])
    return int(rows[np.argmax(score)])  # argmax takes the first of equal maxima: the lowest row


class EndgamePicker:
    def __init__(self, spec, sink=None):
        from research.controller.order_controller import write_record
        self.spec_, self.sink, self.calls = full_spec(spec), sink or write_record, 0
        self.rng = np.random.default_rng(self.spec_["seed"]) if self.spec_["mode"] == "random" else None

    def spec(self):
        return self.spec_

    def __call__(self, policy, ctx):
        raw = np.asarray(ctx["features_raw"])
        feasible = paper_feasible(raw)
        endgame = len(ctx["inventory_ids"]) == 0
        acts = self.spec_["mode"] != "none" and endgame and int(feasible.sum()) >= 2
        pick = None
        if acts:
            height = ctx["height"] if ctx.get("height") is not None else [b.height for b in ctx["all_bricks"]]
            pick = choose(self.spec_["mode"], raw, feasible, ctx["y"], height, self.rng)
        cascade = select(raw, ctx["weights"], "paper", policy["order"], policy["tol"]).index
        self.sink({"event": self.calls, "step": int(ctx["step"]), "phase": ctx["phase"],
                   "sample_trial": int(ctx["sample_trial"]), "side": ctx["side"], "controller": self.spec_,
                   "inventory_empty": endgame, "n_feasible": int(feasible.sum()), "acted": acts, "pick": pick,
                   "cascade_index": None if cascade is None else int(cascade)})
        self.calls += 1
        return {**policy, "pick": pick} if acts else policy


def check(result, records, spec, stopped=False):
    """The order_controller.check report of an endgame wall (see the module docstring)."""
    from research.controller.order_controller import (
        base_policy,
        context_from_trace,
        manifest_problems,
        run_of,
        trace_events,
    )
    from research.walls.check_run import chain_problems
    run = run_of(result)
    receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
    base, events = base_policy(run), list(trace_events(result))
    expected = []
    picker = EndgamePicker(spec, expected.append)
    for meta, a in events:
        picker(dict(base), context_from_trace(meta, a))
    expected = json.loads(json.dumps(expected))
    problems = chain_problems(result, events, stopped)
    if len(records) != len(events):
        problems.append(f"{len(records)} records for {len(events)} selections")
    for r, x, (meta, a) in zip(records, expected, events):
        where = f"event {meta['event']} (step {meta['step']} {meta['phase']} trial {meta['sample_trial']})"
        if r != x:
            problems.append(f"{where}: record differs from the controller re-executed on the trace")
        traced = {k: v for k, v in meta["policy"].items() if k != "pick"}
        if traced != json.loads(json.dumps(base)) or meta["policy"].get("pick") != x["pick"]:
            problems.append(f"{where}: traced policy {meta['policy']}, base {base}, pick {x['pick']}")
        index = select(a["features_raw"], a["weights"], "paper", base["order"], base["tol"], x["pick"]).index
        if (-1 if index is None else int(index)) != int(a["selected"]):
            problems.append(f"{where}: replay selects {index}, traced {int(a['selected'])}")
    manifest = manifest_problems(result, run, receipt)
    problems += [f"manifest: {p}" for p in (manifest or [])[:5]]
    incomplete = ["no run manifest"] if manifest is None else []
    return {"ok": not problems, "verdict": "FAIL" if problems else "PARTIAL" if incomplete else "PASS",
            "problems": problems, "incomplete": incomplete,
            "records": len(records), "decision_points": sum(x["acted"] for x in expected),
            "switches": sum(x["acted"] and x["pick"] != x["cascade_index"] for x in expected)}
