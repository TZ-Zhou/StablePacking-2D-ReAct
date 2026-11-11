"""A non-LLM controller that picks a candidate directly in the endgame, and its check.

Spec: {"kind": "endgame", "mode": M, "seed": S}. While stones are still drawn from the inventory it never acts: the
wall follows the base policy (the receipt's paper order). At a selection whose inventory is empty (every remaining
stone is on offer; TH_T80's last 7 placements) with two or more feasible rows (LLC > 0 and five finite scores,
selection.paper_feasible) it returns the base policy plus {'pick': row}, by mode:
- none: never acts (the controller is attached, the wall is the base policy's);
- random: a uniformly random feasible row, from numpy default_rng(seed), one draw per acting call;
- llc: the highest LLC;
- borda: the highest sum over the five scores of the feasible rows' ranks (ties share the mean rank);
- low_top: the lowest top y + height of the pose before stabilisation;
- low_top_safe, random_safe: low_top, random among the safe rows, the feasible rows whose check_after row has no 0
  (every other remaining stone keeps a feasible pose at the next selection; docs/plans/llm-endgame-lookahead.md), or
  among every feasible row if none is safe. These two ask build_wall for the table (lookahead), traced as 'after';
  their records also give n_safe;
- cascade_safe: the base cascade's row if it is safe or no row is; else the cascade among the safe rows alone
  (lookahead and n_safe as above).
With "whole": true (docs/plans/whole-wall-rules.md) it acts at every such selection, the inventory empty or not, and
a safe mode's table is computed at each of them (lookahead 'always', as pick.py's whole_wall).
Ties go to the lowest row. Every call writes a record to trace/controller.jsonl (order_controller.write_record).

check() re-executes the receipt's spec on the traced inputs (order_controller.context_from_trace): the records must
equal the re-execution, each traced policy must be the base plus the record's pick, replaying select with it must
give the traced selection, every traced check_after row of a placed pose must equal the feasible poses at the
selections that follow the placement (lookahead_problems), and the trace chain and run manifest must be complete (the
order controller's checks).
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from Stonepacker2D.stack.selection import paper_feasible, select

MODES = ("none", "random", "llc", "borda", "low_top", "low_top_safe", "random_safe", "cascade_safe")
SAFE = ("low_top_safe", "random_safe", "cascade_safe")


def full_spec(spec):
    """The spec with its defaults (whole only if set, so earlier walls keep their spec); unknown keys, modes or seeds
    raise."""
    if (not isinstance(spec, dict) or spec.get("kind") != "endgame" or set(spec) - {"kind", "mode", "seed", "whole"}
            or spec.get("whole", True) is not True):
        raise ValueError(f'an endgame spec is {{"kind": "endgame", "mode": ..., "seed": ...[, "whole": true]}}, '
                         f'got {spec!r}')
    full = {"kind": "endgame", "mode": spec.get("mode"), "seed": spec.get("seed")}
    if full["mode"] not in MODES:
        raise ValueError(f'endgame mode is one of {MODES}, got {full["mode"]!r}')
    if (full["mode"] in ("random", "random_safe")) != (type(full["seed"]) is int):
        raise ValueError(f'the random mode needs an int seed and the others none, got {full["seed"]!r}')
    return {**full, **({"whole": True} if "whole" in spec else {})}


def safe_rows(feasible, after):
    """The feasible rows whose check_after row leaves no remaining stone without a feasible pose."""
    return feasible & ~(np.asarray(after) == 0).any(axis=1)


def choose(mode, raw, feasible, y, height, rng, after=None):
    """Row picked among the feasible rows of raw (features_raw); a safe mode among the safe rows if there are any."""
    if mode in SAFE:
        safe = safe_rows(feasible, after)
        feasible, mode = (safe if safe.any() else feasible), mode[:-len("_safe")]
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


def cascade_safe(raw, safe, weights, order, tol):
    """The cascade's row; if it is not safe and a safe row exists, the cascade among the safe rows alone."""
    pick = select(raw, weights, "paper", order, tol).index
    if safe.any() and not safe[pick]:
        masked = np.array(raw, dtype=float)
        masked[~safe, 3] = 0  # LLC 0: not feasible for the cascade
        pick = select(masked, weights, "paper", order, tol).index
    return int(pick)


class EndgamePicker:
    def __init__(self, spec, sink=None):
        from research.controller.order_controller import write_record
        self.spec_, self.sink, self.calls = full_spec(spec), sink or write_record, 0
        self.rng = np.random.default_rng(self.spec_["seed"]) if self.spec_["seed"] is not None else None
        self.whole = self.spec_.get("whole", False)
        # build_wall computes the check_after table for this controller: in the endgame, or 'always'
        self.lookahead = self.spec_["mode"] in SAFE and ("always" if self.whole else True)

    def spec(self):
        return self.spec_

    def __call__(self, policy, ctx):
        raw = np.asarray(ctx["features_raw"])
        feasible = paper_feasible(raw)
        endgame = len(ctx["inventory_ids"]) == 0
        acts = self.spec_["mode"] != "none" and (endgame or self.whole) and int(feasible.sum()) >= 2
        pick = None
        if acts and self.spec_["mode"] == "cascade_safe":
            pick = cascade_safe(raw, safe_rows(feasible, ctx["after"]), ctx["weights"], policy["order"], policy["tol"])
        elif acts:
            height = ctx["height"] if ctx.get("height") is not None else [b.height for b in ctx["all_bricks"]]
            pick = choose(self.spec_["mode"], raw, feasible, ctx["y"], height, self.rng, ctx.get("after"))
        cascade = select(raw, ctx["weights"], "paper", policy["order"], policy["tol"]).index
        self.sink({"event": self.calls, "step": int(ctx["step"]), "phase": ctx["phase"],
                   "sample_trial": int(ctx["sample_trial"]), "side": ctx["side"], "controller": self.spec_,
                   "inventory_empty": endgame, "n_feasible": int(feasible.sum()), "acted": acts, "pick": pick,
                   "cascade_index": None if cascade is None else int(cascade),
                   **({"n_safe": int(safe_rows(feasible, ctx["after"]).sum()) if acts else None}
                      if self.lookahead else {})})
        self.calls += 1
        return {**policy, "pick": pick} if acts else policy


def commits_of(result):
    """The attempts of a traced result directory (trace/commits.jsonl)."""
    lines = (Path(result) / "trace" / "commits.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def lookahead_problems(events, commits=None):
    """Where a traced check_after table is not what the wall did next: the row of the placed pose must give, for each
    remaining stone, its feasible poses at the next selection (and -1 for no stone left). A table traced with stones
    still in the inventory (pick.py's whole_wall) is compared on the stones the next selection samples again.
    commits (trace/commits.jsonl): the placed pose is the chosen side's selection of a 'placed' attempt, and 'next' is
    every selection on that side before the following placement, as they all see the wall it left (a two-sided,
    sample-twice wall; the table of a probe or of the side not taken is not checked: nothing shows its wall).
    Without commits every selection is taken as placed and followed by the next one (a wall built from one side)."""
    if commits is None:
        pairs = [(e, [n]) for e, n in zip(events, events[1:])] + [(events[-1], [])] if events else []
    else:
        by_id = {meta["event"]: (meta, a) for meta, a in events}
        placed = [c for c in commits if c["outcome"] == "placed"]
        last = [max(v["event"] for v in c["sides"].values() if v["event"] is not None) for c in placed]
        pairs = []
        for c, lo, hi in zip(placed, last, [*last[1:], float("inf")]):
            e = by_id[c["sides"][c["chosen_side"]]["event"]]
            pairs.append((e, [n for n in events if lo < n[0]["event"] <= hi and n[0]["side"] == e[0]["side"]]))
    problems = []
    for (meta, a), nxt in pairs:
        if "after" not in a or int(a["selected"]) < 0:
            continue
        ids, row = [int(s) for s in a["sampled_ids"]], a["after"][int(a["selected"])]
        told = {s: int(c) for s, c in zip(ids, row) if c >= 0}
        seen = {}
        for _, n in nxt:
            n_ids = [int(s) for s in n["sampled_ids"]]
            feasible = paper_feasible(np.asarray(n["features_raw"]))
            seen.update({s: int(feasible[k::len(n_ids)].sum()) for k, s in enumerate(n_ids)})
        if len(a["inventory_ids"]):
            told, seen = ({s: c for s, c in d.items() if s in told and s in seen} for d in (told, seen))
        if told != seen:
            problems.append(f"event {meta['event']} (step {meta['step']}): check_after {told}, next selection {seen}")
    return problems


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
    problems = chain_problems(result, events, stopped) + lookahead_problems(events, commits_of(result))
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
