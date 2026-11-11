"""An LLM that picks the endgame's candidate directly (controller kind 'pick').

Spec: {'kind': 'pick', 'llm': ..., 'max_tool_calls', 'max_decisions', 'llm_timeout', 'max_llm_failures',
'max_total_tokens', 'max_cost_usd'} (order_controller.SPEC_DEFAULTS; the budget keys as a react spec's). While stones
are still drawn from the inventory it never acts. A decision point is a selection whose inventory is empty (every
remaining stone is on offer; TH_T80's last 7 placements) with two or more feasible candidates (LLC > 0 and five finite
scores, selection.paper_feasible). There the LLM gets every feasible candidate in its first message (one request per
decision, no listing tool) and answers with commit_pick; the wall places that candidate (the base policy plus
{'pick': row}). A skip or a fallback keeps the cascade of the base order.

With "lookahead": true (docs/plans/llm-endgame-lookahead.md) the observation version is 'pick-2' (the same fields)
and the LLM also has check_after: for listed candidate ids, the feasible poses each other remaining stone has at the
next placement if that candidate is placed (build_wall's table, traced as 'after'; the request carries it per
candidate, so the tool only reads it). check() then also requires each placed candidate's row to equal the next
selection (endgame.lookahead_problems). With "llc_note": true as well (docs/plans/llm-endgame-llc-note.md) the
observation version is 'pick-3': the system prompt of 'pick-2' plus one paragraph saying LLC and F_LR are the same
kind of quantity (LLC_NOTE); nothing else changes.

With "whole_wall": true (with lookahead, without llc_note; docs/plans/llm-whole-wall-pick.md) the LLM picks from the
first stone: every selection with two or more feasible candidates is a decision point, the observation version is
'pick-4' (the same fields; SYSTEM_WHOLE says all four metrics must be good) and build_wall computes the check_after
table at every such selection (lookahead 'always'). While the inventory is not empty the next selection is a new
sample, so check() compares a placed candidate's row only on the stones sampled again.

It is the order controller with its own observation, request and reply (PickClient over ControllerClient: the same
worker, budget, ledger and LLM transcripts; PickPolicy over ReActPolicy: the same LLM call, retries on a rejected
reply and fallbacks). The observation version is 'pick-1', its own: no filter order, no is_paper or n_orders, no
geometry, no memory. check() re-executes the receipt's spec on the trace with the recorded LLM replies (ReplayLLM):
records, transcripts and ledger must equal the re-execution, the traced policy must be the base plus the record's
pick and replaying select must give the traced selection.
"""

import hashlib
import json
import re
import types

import numpy as np

from research.controller import order_controller as oc
from Stonepacker2D.stack.selection import paper_feasible, select

SYSTEM = (
    "You finish a dry-stone wall. The wall is built course by course with Wang's method; you act only at the end, "
    "when every remaining stone is on offer. At each placement you choose which candidate to place: a stone in one "
    "of four rotations, at the position Wang's image convolution found for it. Every listed candidate is feasible "
    "(LLC > 0). Your goal is the finished wall, judged on four metrics: F_SF, stone filling (the share of the wall "
    "area that is stone, higher is better); F_AH, the horizontal joint path index (lower is better: no long "
    "continuous horizontal joints); F_AV, the vertical joint path index (higher is better: vertical joints are "
    "staggered); F_LR, the smaller of the left and right kinematic limit loads of the wall divided by the friction "
    "coefficient (higher is better). A stone that cannot be placed is lost to the wall.\n"
    "Data. Stones are labelled S1, S2, ... in the order they were first sampled. Coordinates are pixels in the "
    "evaluation frame (mirrored when building from the right, so 'left' is the side the course is built from); rows "
    "count upward from the ground. Each candidate: id, stone, rotation, its pose x and y, the width and height of "
    "its bounding box, and Wang's five filter scores (higher is better for each): NHR course height (0 = level with "
    "its left neighbour), GDS global density, IWR interlocking (its vertical joint far from the joints below), LLC "
    "the kinematic limit load of the wall with the stone added and stabilized, divided by the friction coefficient, "
    "LFR local void (0 = flush on the left and on the bottom). wall summarizes the current wall; unplaceable lists "
    "stones on offer with no feasible candidate now. Nothing about the final wall is known.\n"
    "Compare the main candidates on all five scores and on what they leave for the remaining stones; in the "
    "rationale state each one's main advantages and costs. Finish with commit_pick.")
LOOKAHEAD = (
    "Before committing you may call check_after with one or more candidate ids (comma-separated). For each, it "
    "returns how many feasible poses (0 to 4, at most one per rotation) every other remaining stone would have at the "
    "very next placement if that candidate were placed now. 0 means that stone could not be placed next; the counts "
    "can change again after later placements. It gives no scores.\n")
SYSTEM_LOOKAHEAD = SYSTEM.replace("Compare the main candidates", LOOKAHEAD + "Compare the main candidates")
assert SYSTEM_LOOKAHEAD.count(LOOKAHEAD) == 1
LLC_NOTE = (
    "LLC and F_LR are the same kind of quantity, a kinematic limit load of the wall divided by the friction "
    "coefficient: LLC is that of the wall right after this placement, F_LR that of the finished wall. Giving up a "
    "higher LLC now can therefore leave a lower F_LR at the end. Weigh this together with everything else; it is "
    "not a rule.\n")
SYSTEM_NOTE = SYSTEM_LOOKAHEAD.replace("Compare the main candidates", LLC_NOTE + "Compare the main candidates")
assert SYSTEM_NOTE.count(LLC_NOTE) == 1
WHOLE = (  # (old, new) on SYSTEM_LOOKAHEAD
    ("You finish a dry-stone wall. The wall is built course by course with Wang's method; you act only at the end, "
     "when every remaining stone is on offer. At each placement you choose which candidate to place: a stone in one "
     "of four rotations",
     "You build a dry-stone wall from the first stone to the last. The wall is built course by course with Wang's "
     "method. At each placement Wang's sampling puts some of the remaining stones on offer (all of them once only a "
     "few are left), and you choose which candidate to place: a stone on offer in one of four rotations"),
    ("(higher is better). A stone",
     "(higher is better). All four metrics must be good: F_SF, F_AH, F_AV and F_LR. Do not sacrifice the others for "
     "one of them. A stone"),
    ("bottom). wall summarizes",
     "bottom). n_remaining counts the stones not yet placed, on offer or not; wall summarizes"),
    ("every other remaining stone would", "every other stone on offer now would"),
    ("after later placements. It gives",
     "after later placements. While more stones remain than are on offer, the next placement's offer is a new "
     "sample, so a stone on offer now may not be on offer next. It gives"))
SYSTEM_WHOLE = SYSTEM_LOOKAHEAD
for _old, _new in WHOLE:
    assert SYSTEM_WHOLE.count(_old) == 1
    SYSTEM_WHOLE = SYSTEM_WHOLE.replace(_old, _new)
CHECK = {"name": "check_after", "description": "For listed candidates: the feasible poses each other remaining stone "
         "would have at the next placement if that candidate were placed. No scores.",
         "parameters": {"type": "object", "additionalProperties": False, "required": ["candidates"],
                        "properties": {"candidates": {"type": "string", "description":
                                                      "Comma-separated ids of listed candidates, e.g. '0,3'."}}}}
CHECK_WHOLE = {**CHECK,
               "description": CHECK["description"].replace("other remaining stone", "other stone on offer now")}
assert CHECK_WHOLE != CHECK
COMMIT = {"name": "commit_pick", "description": "Final action: the candidate to place.",
          "parameters": {"type": "object", "additionalProperties": False, "required": ["candidate", "rationale"],
                         "properties": {"candidate": {"type": "string", "description": "The id of a listed candidate."},
                                        "rationale": {"type": "string", "description":
                                                      "The main advantages and costs of the compared candidates, "
                                                      "and why this one."}}}}
# before the lessons of a spec (docs/plans/llm-lessons.md); a spec without lessons leaves SYSTEM_WHOLE as it is
LESSONS = ("\nLessons. The advice below was written after studying finished walls, good and bad, built with this "
           "method from other sets of stones. It is advice, not rules: weigh it against what this placement shows.")
OBSERVATION_VERSION = "pick-1"
OBSERVATION_VERSION_LOOKAHEAD = "pick-2"
OBSERVATION_VERSION_NOTE = "pick-3"
OBSERVATION_VERSION_WHOLE = "pick-4"
REPLY_KEYS = {"pick", "source", "prompt_sha256", "react", "transcripts", "restart"}
ID_TEXT = re.compile(r"[0-9]{1,4}")


class Observer:
    """The pick observation of a selection: the wall and every feasible candidate, stones labelled in order of first
    sampling (so it is the same under any renaming of the stone ids); also the feasible rows, in candidate id order."""

    def __init__(self, version=OBSERVATION_VERSION):
        self.labels, self.geometry, self.version = {}, None, version  # geometry: order_controller.rerun's slot, unused

    label = oc.Observer.label

    def __call__(self, policy, ctx):
        raw = np.asarray(ctx["features_raw"])
        feasible, sampled = paper_feasible(raw), [int(s) for s in ctx["sampled_ids"]]
        n, rows = len(sampled), np.flatnonzero(paper_feasible(raw))
        for s in sampled:
            self.label(s)
        widths, heights = oc.candidate_sizes(ctx)
        wall = np.asarray(ctx["wall"])
        candidates = [{"id": k, "stone": self.label(sampled[i % n]), "rotation": int(i // n * 90),
                       "x": int(ctx["x"][i]), "y": int(ctx["y"][i]), "width": widths[i], "height": heights[i],
                       "scores": {oc.FILTERS[c]: round(float(raw[i, c]), 6) for c in range(5)}}
                      for k, i in enumerate(rows)]
        reachable = {sampled[i % n] for i in rows}
        obs = {"observation_version": self.version, "step": int(ctx["step"]), "phase": ctx["phase"],
               "sample_trial": int(ctx["sample_trial"]), "side": ctx["side"], "frame": "evaluation",
               "n_remaining": len({int(s) for s in ctx["inventory_ids"]} | set(sampled)),
               "wall_size": {"width": int(wall.shape[1]), "height": int(wall.shape[0])}, "wall": oc.wall_state(wall),
               "unplaceable": [self.label(s) for s in sampled if s not in reachable], "candidates": candidates}
        return oc.plain(obs), rows, feasible


class PickPolicy(oc.ReActPolicy):
    """ReActPolicy with commit_pick as its only tool (and check_after with lookahead) and the whole observation in
    the first message."""

    def __init__(self, llm, max_tool_calls=3, llm_timeout=60.0, lookahead=False, llc_note=False,
                 whole_wall=False, lessons=None):
        self.llm, self.max_tool_calls, self.llm_timeout = llm, max_tool_calls, llm_timeout
        self.system, self.tools = (SYSTEM_LOOKAHEAD, [CHECK, COMMIT]) if lookahead else (SYSTEM, [COMMIT])
        if llc_note:
            self.system = SYSTEM_NOTE
        if whole_wall:
            self.system, self.tools = SYSTEM_WHOLE, [CHECK_WHOLE, COMMIT]
        if lessons:
            self.system += LESSONS + "".join(f"\n{i}. {text}" for i, text in enumerate(lessons, 1))
        self.prompt_sha256 = hashlib.sha256(json.dumps([self.system, self.tools]).encode()).hexdigest()

    def view(self, obs):
        return obs

    def answer(self, request, choice):
        return {"pick": choice}

    def problem(self, reply, request):
        problem = oc.reply_problem(reply, self.tools)
        if problem:
            return problem
        n, args = len(request["observation"]["candidates"]), reply["arguments"]
        if reply["tool"] == "check_after":
            ids = [c.strip() for c in args["candidates"].split(",")]
            if not all(ID_TEXT.fullmatch(c) and int(c) < n for c in ids) or len(set(map(int, ids))) != len(ids):
                return f"candidates must be distinct ids of listed candidates, 0 to {n - 1}, got {args['candidates']!r}"
            return None
        if not (ID_TEXT.fullmatch(args["candidate"].strip()) and int(args["candidate"]) < n):
            return f"candidate must be the id of a listed candidate, 0 to {n - 1}, got {args['candidate']!r}"
        if not args["rationale"].strip():
            return "rationale must not be empty"
        return None

    def tool(self, name, args, request):
        if name == "check_after":
            ids = [str(int(c)) for c in args["candidates"].split(",")]
            return {"after": {c: request["after"][c] for c in ids}}, None
        return {"ok": True, "candidate": int(args["candidate"])}, int(args["candidate"])


def reply_problem(reply, n):
    """Why a worker's reply is not a pick policy's answer for n candidates, None if it is."""
    if not isinstance(reply, dict) or set(reply) - REPLY_KEYS or not {"pick", "source"} <= set(reply):
        return "the reply is not an object with a pick and a source"
    if not isinstance(reply["source"], str) or reply["source"] not in oc.SOURCES["pick"]:
        return "source is not one of the pick policy"
    if reply["source"] == "llm" and not (type(reply["pick"]) is int and 0 <= reply["pick"] < n):
        return f"the pick is not a candidate id 0 to {n - 1}"
    if reply["source"] != "llm" and reply["pick"] is not None:
        return "a fallback picks nothing"
    return None


class PickClient(oc.ControllerClient):
    """ControllerClient with the pick observation, request, reply and record (module docstring)."""

    def __init__(self, spec, transport, sink=None, ledger=None):
        super().__init__(spec, transport, sink, ledger)
        self.whole = self.spec().get("whole_wall", False)
        # build_wall computes the check_after table if true (in the endgame) or 'always' (at every selection)
        self.lookahead = "always" if self.whole else self.spec().get("lookahead", False)
        version = (OBSERVATION_VERSION_WHOLE if self.whole else OBSERVATION_VERSION_NOTE
                   if self.spec().get("llc_note") else OBSERVATION_VERSION_LOOKAHEAD
                   if self.lookahead else OBSERVATION_VERSION)
        self.observer, self.n = Observer(version), 0

    def request_of(self, obs, info):
        after = {"after": info["after"]} if self.lookahead else {}
        return {"observation": obs, "base_order": oc.key(info["base"]), **after}, {}

    def keep(self, info):
        return None

    def reply_problem(self, reply, base):
        return reply_problem(reply, self.n)

    def decision(self, reply):
        return reply["pick"], {k: v for k, v in reply.items() if k not in ("pick", "restart")}

    def __call__(self, policy, ctx):
        obs, rows, feasible = self.observer(policy, ctx)
        base, raw = tuple(policy["order"] or oc.PAPER_ORDER), np.asarray(ctx["features_raw"])
        endgame = len(ctx["inventory_ids"]) == 0
        decision_point, self.n = (endgame or self.whole) and len(rows) >= 2, len(rows)
        info = {"base": base}
        if decision_point and self.lookahead:  # per candidate id: {stone label: feasible poses at the next placement}
            sampled, after = [int(s) for s in ctx["sampled_ids"]], np.asarray(ctx["after"])
            info["after"] = {str(k): {self.observer.label(s): int(c) for s, c in zip(sampled, after[i]) if c >= 0}
                             for k, i in enumerate(rows)}
        choice, detail = self.ask(obs, info) if decision_point else (None, {"source": "not_a_decision_point"})
        pick = None if choice is None else int(rows[choice])
        cascade = select(raw, ctx["weights"], "paper", base, policy["tol"]).index
        cascade = None if cascade is None else int(cascade)
        index = cascade if pick is None else pick
        n = len(ctx["sampled_ids"])
        self.sink({"event": self.calls, "step": obs["step"], "phase": obs["phase"], "sample_trial": obs["sample_trial"],
                   "side": obs["side"], "controller": self.spec(), "decision_point": decision_point,
                   "inventory_empty": endgame, "n_candidates": len(rows),
                   "observation": obs if decision_point else None, **detail,
                   "final": {"candidate": choice, "pick": pick, "index": index, "cascade_index": cascade,
                             "switched": index != cascade,
                             "stone_id": None if index is None else int(ctx["sampled_ids"][index % n])}})
        self.calls += 1
        return policy if pick is None else {**policy, "pick": pick}


def mock_reply(mode, messages):
    """order_controller.MockLLM for a pick (also the 'fake' provider's): pick_llc commits the highest LLC, pick_low_top
    the lowest top (y + height), pick_first id 0; pick_retry first commits an id not listed, then as pick_llc;
    pick_invalid only ids not listed; pick_safe (lookahead) first calls check_after on every candidate, then commits
    the highest LLC among the safe ones (no remaining stone left at 0), among all if none is. Ties go to the lowest
    id, as in research/controller/endgame.py."""
    candidates = json.loads(messages[1]["content"])["candidates"]
    if mode == "pick_safe":
        told = json.loads(messages[-1]["content"]).get("after") if messages[-1]["role"] == "tool" else None
        if told is None:
            ids = ",".join(str(c["id"]) for c in candidates)
            return {"thought": "check every candidate", "tool": "check_after", "arguments": {"candidates": ids}}
        safe = [c for c in candidates if 0 not in told[str(c["id"])].values()] or candidates
        best = max(safe, key=lambda c: (c["scores"]["LLC"], -c["id"]))
        why = f"candidate {best['id']} (pick_safe, {len(safe)} safe)"
        return {"thought": why, "tool": "commit_pick", "arguments": {"candidate": str(best["id"]), "rationale": why}}

    def commit(c, why):
        return {"thought": why, "tool": "commit_pick", "arguments": {"candidate": str(c), "rationale": why}}
    if mode == "pick_invalid" or (mode == "pick_retry" and len(messages) == 2):
        return commit(len(candidates), "an id that is not listed")
    if mode in ("pick_llc", "pick_retry"):
        best = max(candidates, key=lambda c: (c["scores"]["LLC"], -c["id"]))
    elif mode == "pick_low_top":
        best = min(candidates, key=lambda c: (c["y"] + c["height"], c["id"]))
    elif mode == "pick_first":
        best = candidates[0]
    else:
        raise ValueError(f"unknown pick mock mode {mode!r}")
    return commit(best["id"], f"candidate {best['id']} ({mode})")


def check(result, records, spec, stopped=False):
    """The order_controller.check report of a pick wall (see the module docstring); verdict PASS needs the ledger and
    the run manifest."""
    from research.controller.endgame import commits_of, lookahead_problems
    from research.walls.check_run import chain_problems
    run = oc.run_of(result)
    receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
    base, full, events = oc.base_policy(run), oc.full_spec(spec), list(oc.trace_events(result))
    llm = oc.ReplayLLM(records, *oc.llm_identity(full["llm"]))
    worker = {**{k: full[k] for k in oc.WORKER_KEYS["pick"]}, "lookahead": full.get("lookahead", False),
              "llc_note": full.get("llc_note", False), "whole_wall": full.get("whole_wall", False),
              "lessons": full.get("lessons")}
    transport = oc.ReplayedFailures(lambda: PickPolicy(llm, **worker), records)
    expected, ledger = [], []
    client = PickClient(spec, transport, expected.append, ledger.append)
    try:
        for meta, a in events:
            client(types.MappingProxyType(dict(base)), oc.context_from_trace(meta, a))
    finally:
        transport.close()
    expected = json.loads(json.dumps(expected, default=lambda v: v.item()))
    problems = chain_problems(result, events, stopped) + lookahead_problems(events, commits_of(result))
    if not len(events) <= len(records) <= len(events) + int(stopped):  # ponytail: a trailing record is not validated
        problems.append(f"{len(records)} records for {len(events)} selections")
    plain_base = json.loads(json.dumps(base))
    for r, x, (meta, a) in zip(records, expected, events):
        where = f"event {meta['event']} (step {meta['step']} {meta['phase']} trial {meta['sample_trial']})"
        if oc.comparable(r) != oc.comparable(x):
            problems.append(f"{where}: record differs from the controller re-executed on the trace")
        traced = {k: v for k, v in meta["policy"].items() if k != "pick"}
        if traced != plain_base or meta["policy"].get("pick") != x["final"]["pick"]:
            problems.append(f"{where}: traced policy {meta['policy']}, base {base}, pick {x['final']['pick']}")
        index = select(a["features_raw"], a["weights"], "paper", base["order"], base["tol"], x["final"]["pick"]).index
        if (-1 if index is None else int(index)) != int(a["selected"]):
            problems.append(f"{where}: replay selects {index}, traced {int(a['selected'])}")
    problems += [f"replay diverged at {d}" for d in llm.diverged[:5]]
    found = oc.ledger_problems(result, ledger, events, stopped)
    problems += [f"ledger: {p}" for p in found or []]
    manifest = oc.manifest_problems(result, run, receipt)
    problems += [f"manifest: {p}" for p in (manifest or [])[:5]]
    missing = (("transport ledger", found is None), ("run manifest", manifest is None))
    incomplete = [f"no {name}" for name, absent in missing if absent]
    return {"ok": not problems, "verdict": "FAIL" if problems else "PARTIAL" if incomplete else "PASS",
            "problems": problems, "incomplete": incomplete, "records": len(records),
            "decision_points": sum(x["decision_point"] for x in expected),
            "switches": sum(x["final"]["switched"] for x in expected)}
