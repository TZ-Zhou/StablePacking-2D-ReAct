"""The reading of docs/plans/llm-endgame-pick.md: each state's LLM pick wall against the same state's arena endings.

Usage: python -B -m research.analysis.llm_endgame_pick [--loopback] [--out FILE]

The LLM walls are the pick-walls study's (pick-walls-loopback with --loopback: only to test this script), one per
state of research/llm_walls/pick_wall.STATES, each with a finished journal. Scores are as in endgame_arena: F_* on the
simulation grid after the last placement (the wall's derived/steps.csv), z in the seed's factorial TH_T80 block,
F_AH negated, the mean over the four metrics. The cascade, random and rule endings are the arena's walls
(research/datasets/endgame-arena-s<seed>-TH_T80, batch_tables). Rows per state, then the rule, read once: stop
first, then go ((a), (b), (c)), else 'no endgame contribution shown'.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.analysis import myopia_tables as mt
from research.analysis.endgame_arena import METRICS
from research.llm_walls import gpt_wall, pick_wall
from research.paths import DATASETS_DIR
from research.storage import case_id as cid

ZCOLS = [f"z_{m}" for m in METRICS]


def zscores(frame, block):
    z = pd.concat([(frame[m] - block[m].mean()) / block[m].std() * s for m, s in METRICS.items()], axis=1)
    return z.add_prefix("z_").assign(mean_z=lambda t: t[ZCOLS].mean(axis=1))


def llm_wall(case):
    """Grid F_*, stones placed and the decisions of one pick wall's case folder."""
    result = mt.traced_result(case)
    placed = pd.read_csv(case / "derived" / "steps.csv").dropna(subset=["F_LR"]).iloc[-1]
    lines = (result / "trace" / "controller.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    decisions = [r for r in records if r["decision_point"]]
    latency = [t["latency_s"] for r in decisions for t in r.get("transcripts") or []]
    return {**{m: placed[m] for m in METRICS}, "n_placed": mt.readable_metrics(case.name, case, result)["n_placed"],
            "decisions": len(decisions), "llm_decisions": sum(r["source"] == "llm" for r in decisions),
            "switches": sum(r["source"] == "llm" and r["final"]["switched"] for r in decisions),
            "complete": bool(decisions) and all(r["source"] == "llm" for r in decisions),
            "llm_seconds": round(sum(latency), 3)}


def state_row(seed, order, journal):
    arena = pd.read_csv(DATASETS_DIR / f"endgame-arena-s{seed}-TH_T80" / "walls.csv", dtype={"filter_policy": str})
    block = pd.read_csv(DATASETS_DIR / f"factorial-s{seed}-TH_T80" / "walls.csv", dtype={"filter_policy": str})
    arena = pd.concat([arena, zscores(arena, block)], axis=1)
    block = pd.concat([block, zscores(block, block)], axis=1)
    mine = arena[arena.filter_policy.str.partition("+")[0] == order]
    kind = mine.filter_policy.str.partition("+")[2].str.partition(":")[0]
    cascade = block[block.filter_policy == order].iloc[0]
    llm = llm_wall(cid.wall_dir(journal["case_id"]))
    llm = {**llm, **zscores(pd.DataFrame([llm]), block).iloc[0].to_dict()}
    random = mine[kind == "random"].mean_z.to_numpy()
    rules = {r: mine[kind == r].iloc[0] for r in ("llc", "borda", "low_top")}
    spend = journal.get("spend") or {}
    return {"seed": seed, "order": order, "case_id": journal["case_id"], "complete": llm["complete"],
            "llm_mean_z": llm["mean_z"], **{f"llm_{c}": llm[c] for c in ZCOLS}, "llm_n_placed": llm["n_placed"],
            "cascade_mean_z": cascade.mean_z, "cascade_n_placed": int(cascade.n_placed),
            "random_median": np.median(random),
            "llm_percentile": ((random < llm["mean_z"]).sum() + 0.5 * (random == llm["mean_z"]).sum()) / len(random),
            **{f"{r}_mean_z": w.mean_z for r, w in rules.items()},
            "decisions": llm["decisions"], "llm_decisions": llm["llm_decisions"], "switches": llm["switches"],
            "tokens": spend.get("tokens"), "cost_usd": spend.get("cost_usd"), "llm_seconds": llm["llm_seconds"]}


def reading(t):
    """The rule of docs/plans/llm-endgame-pick.md on the 9 state rows."""
    if (t.llm_mean_z < t.random_median).sum() >= 5 or (t.llm_n_placed < t.cascade_n_placed).sum() >= 3:
        return "stop direct pick"
    a = (t.complete & (t.llm_mean_z > t.cascade_mean_z) & (t.llm_n_placed >= t.cascade_n_placed)).sum() >= 7
    b = (t.complete & (t.llm_mean_z >= t.low_top_mean_z)).sum() >= 5
    c = t.switches.sum() >= 0.25 * t.llm_decisions.sum()
    return "go" if a and b and c else "no endgame contribution shown"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--loopback", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    journals = gpt_wall.journals(args.loopback, pick_wall)
    missing = [pick_wall.name_of(*s) for s in pick_wall.STATES
               if (journals.get(pick_wall.name_of(*s)) or {}).get("state") != "finished"]
    if missing:
        print(f"not read: no finished wall for {missing}")
        return 1
    table = pd.DataFrame([state_row(s, o, journals[pick_wall.name_of(s, o)]) for s, o in pick_wall.STATES])
    if args.out:
        table.round(6).to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    print(table.round(3).to_string(index=False))
    t = table
    print(f"(a) {(t.complete & (t.llm_mean_z > t.cascade_mean_z) & (t.llm_n_placed >= t.cascade_n_placed)).sum()}/9 "
          f"(b) {(t.complete & (t.llm_mean_z >= t.low_top_mean_z)).sum()}/9 "
          f"(c) {t.switches.sum()}/{t.llm_decisions.sum()} switched; below random median "
          f"{(t.llm_mean_z < t.random_median).sum()}/9, fewer stones {(t.llm_n_placed < t.cascade_n_placed).sum()}/9")
    print("reading:", reading(table))
    return 0


if __name__ == "__main__":
    sys.exit(main())
