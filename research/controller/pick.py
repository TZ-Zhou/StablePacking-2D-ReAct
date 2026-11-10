"""An LLM that picks the endgame's candidate directly (controller kind 'pick').

Spec: {'kind': 'pick', 'llm': ..., 'max_tool_calls', 'max_decisions', 'llm_timeout', 'max_llm_failures',
'max_total_tokens', 'max_cost_usd'} (order_controller.SPEC_DEFAULTS; the budget keys as a react spec's). While stones
are still drawn from the inventory it never acts. A decision point is a selection whose inventory is empty (every
remaining stone is on offer; TH_T80's last 7 placements) with two or more feasible candidates (LLC > 0 and five finite
scores, selection.paper_feasible). There the LLM gets every feasible candidate in its first message (one request per
decision, no listing tool) and answers with commit_pick; the wall places that candidate (the base policy plus
{'pick': row}). A skip or a fallback keeps the cascade of the base order.

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
COMMIT = {"name": "commit_pick", "description": "Final action: the candidate to place.",
          "parameters": {"type": "object", "additionalProperties": False, "required": ["candidate", "rationale"],
                         "properties": {"candidate": {"type": "string", "description": "The id of a listed candidate."},
                                        "rationale": {"type": "string", "description":
                                                      "The main advantages and costs of the compared candidates, "
                                                      "and why this one."}}}}
OBSERVATION_VERSION = "pick-1"
REPLY_KEYS = {"pick", "source", "prompt_sha256", "react", "transcripts", "restart"}
ID_TEXT = re.compile(r"[0-9]{1,4}")


class Observer:
    """The pick observation of a selection: the wall and every feasible candidate, stones labelled in order of first
    sampling (so it is the same under any renaming of the stone ids); also the feasible rows, in candidate id order."""

    def __init__(self):
        self.labels, self.geometry = {}, None  # geometry: order_controller.rerun's slot, unused

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
        obs = {"observation_version": OBSERVATION_VERSION, "step": int(ctx["step"]), "phase": ctx["phase"],
               "sample_trial": int(ctx["sample_trial"]), "side": ctx["side"], "frame": "evaluation",
               "n_remaining": len({int(s) for s in ctx["inventory_ids"]} | set(sampled)),
               "wall_size": {"width": int(wall.shape[1]), "height": int(wall.shape[0])}, "wall": oc.wall_state(wall),
               "unplaceable": [self.label(s) for s in sampled if s not in reachable], "candidates": candidates}
        return oc.plain(obs), rows, feasible


class PickPolicy(oc.ReActPolicy):
    """ReActPolicy with commit_pick as its only tool and the whole observation in the first message."""

    def __init__(self, llm, max_tool_calls=3, llm_timeout=60.0):
        self.llm, self.max_tool_calls, self.llm_timeout = llm, max_tool_calls, llm_timeout
        self.system, self.tools = SYSTEM, [COMMIT]
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
        if not (ID_TEXT.fullmatch(args["candidate"].strip()) and int(args["candidate"]) < n):
            return f"candidate must be the id of a listed candidate, 0 to {n - 1}, got {args['candidate']!r}"
        if not args["rationale"].strip():
            return "rationale must not be empty"
        return None

    def tool(self, name, args, request):
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
        self.observer, self.n = Observer(), 0

    def request_of(self, obs, info):
        return {"observation": obs, "base_order": oc.key(info["base"])}, {}

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
        decision_point, self.n = endgame and len(rows) >= 2, len(rows)
        choice, detail = self.ask(obs, {"base": base}) if decision_point else (None, {"source": "not_a_decision_point"})
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
    pick_invalid only ids not listed. Ties go to the lowest id, as in research/controller/endgame.py."""
    candidates = json.loads(messages[1]["content"])["candidates"]

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
    from research.walls.check_run import chain_problems
    run = oc.run_of(result)
    receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
    base, full, events = oc.base_policy(run), oc.full_spec(spec), list(oc.trace_events(result))
    llm = oc.ReplayLLM(records, *oc.llm_identity(full["llm"]))
    worker = {k: full[k] for k in oc.WORKER_KEYS["pick"]}
    transport = oc.ReplayedFailures(lambda: PickPolicy(llm, **worker), records)
    expected, ledger = [], []
    client = PickClient(spec, transport, expected.append, ledger.append)
    try:
        for meta, a in events:
            client(types.MappingProxyType(dict(base)), oc.context_from_trace(meta, a))
    finally:
        transport.close()
    expected = json.loads(json.dumps(expected, default=lambda v: v.item()))
    problems = chain_problems(result, events, stopped)
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
