"""The reading of docs/plans/whole-wall-rules.md: three rules without an LLM, each picking a whole wall, against Wang's
Paper wall and the whole-wall LLM pick on the same 10 TH blocks.

Usage: python -B -m research.analysis.whole_wall_rules [--out FILE]

A block's rule walls are the batch whole-rules-s<seed>-<scene> (research/walls/endgame, built by
research/runner/batch.py into runs/batch), one wall per rule of RULES. Wang's wall, the scores and 'not below Wang' are
llm_whole_wall's: F_* on the simulation grid after the last placement, z in the seed's factorial block, F_AH negated,
the mean over the four metrics; a wall counts if it is complete (PROCESS_PASS, trace and controller checks PASS, every
decision the rule's), its mean z is not below Wang's and it places no fewer stones. The LLM's walls are the paid ones
of the whole-walls and whole-confirm-walls studies, read as they are. The rule, read once when the 30 walls pass the
checks: each rule's count of the 10 against the LLM's (reading); the tool's credit is the wang_safe row's.
Reported with it: the walls no decision of which differs from the cascade (Wang's wall again), the LLM's wall against
the rule's on the same block, and where the check_after table could matter (mixed: decisions with both safe and
unsafe candidates).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.analysis import llm_endgame_lookahead as la
from research.analysis import llm_endgame_pick as step3
from research.analysis import llm_whole_wall as ww
from research.analysis import myopia_tables as mt
from research.analysis.endgame_arena import METRICS
from research.controller import order_controller as oc
from research.controller.endgame import safe_rows
from research.llm_walls import whole_confirm_wall, whole_wall
from research.paths import BATCH_RUNS
from Stonepacker2D.stack.selection import paper_feasible

RULES = {"wang_safe": "whole_cascade_safe", "pick_safe": "whole_pick_safe", "borda": "whole_borda"}  # plan: level
BLOCKS = [*whole_wall.STATES, *whole_confirm_wall.STATES]


def cases(seed, scene):
    """{rule: case folder} of a block's batch; a rule without a finished wall is left out."""
    folder = BATCH_RUNS / f"whole-rules-s{seed}-{scene}"
    if not (folder / "status.jsonl").exists():
        return {}
    plan = pd.read_csv(folder / "plan.csv", dtype={"filter_policy": str})
    status = pd.read_json(folder / "status.jsonl", lines=True).drop_duplicates("case", keep="last")
    level = plan.merge(status, on="case").set_index(plan.filter_policy.str.partition("+")[2].rename("level")).case_id
    return {rule: folder / "cases" / level[mode] for rule, mode in RULES.items() if mode in level}


def tool_use(result):
    """(decisions with both safe and unsafe candidates, unsafe candidates placed although a safe one existed) of a
    traced wall; (None, None) for a wall without check_after tables."""
    tables = [(paper_feasible(np.asarray(a["features_raw"])), a) for _, a in oc.trace_events(result) if "after" in a]
    safe = [(f, safe_rows(f, a["after"]), int(a["selected"])) for f, a in tables]
    if not safe:
        return None, None
    return sum(0 < s.sum() < f.sum() for f, s, _ in safe), sum(bool(s.any() and not s[k]) for _, s, k in safe)


def rule_row(seed, scene, rule, case, llm):
    w, block = ww.wang(seed, scene)
    receipt = json.loads((case / "receipt.json").read_text(encoding="utf-8"))
    report, result = receipt["trace_check"]["controller"][0], mt.traced_result(case)
    placed = pd.read_csv(case / "derived" / "steps.csv").dropna(subset=["F_LR"]).iloc[-1]
    wall = {**{m: placed[m] for m in METRICS}, "n_placed": mt.readable_metrics(case.name, case, result)["n_placed"]}
    wall = {**wall, **step3.zscores(pd.DataFrame([wall]), block).iloc[0].to_dict()}
    lines = (result / "trace" / "controller.jsonl").read_text(encoding="utf-8").splitlines()
    fallbacks = sum(r.get("decision_point", False) and r["source"] != "llm" for r in map(json.loads, lines))
    mixed, unsafe = tool_use(result)
    mine, his = ww.first_offer(case), ww.first_offer(BATCH_RUNS / f"factorial-s{seed}-{scene}" / "cases" / w.wall)
    return {"seed": seed, "scene": scene, "rule": rule, "case_id": case.name,
            "complete": bool(receipt["status"] == "PROCESS_PASS" and receipt["trace_check"]["ok"]
                             and report["verdict"] == "PASS" and not fallbacks),
            "mean_z": wall["mean_z"], "wang_mean_z": w.mean_z, "mean_z_diff": wall["mean_z"] - w.mean_z,
            **{f"diff_{c}": wall[c] - w[c] for c in step3.ZCOLS},
            "n_placed": wall["n_placed"], "wang_n_placed": int(w.n_placed), "n_input": int(w.n_input),
            "percentile": la.percentile(block.mean_z.to_numpy(), wall["mean_z"]),
            "decisions": report["decision_points"], "switches": report["switches"],
            # no decision differs from the cascade: Wang's wall again, so its scores must be Wang's
            "as_wang": report["switches"] == 0,
            "scores_as_wang": bool(wall["n_placed"] == w.n_placed and np.isclose(wall["mean_z"], w.mean_z)),
            "mixed": mixed, "unsafe_picks": unsafe,
            "same_first_offer": all(np.array_equal(x, y, equal_nan=True) for x, y in zip(mine, his)),
            "llm_mean_z": llm.llm_mean_z, "llm_n_placed": int(llm.llm_n_placed),
            "llm_not_below_wang": bool(ww.not_below_wang(llm.to_frame().T.infer_objects()).iloc[0])}


def not_below_wang(t):
    return t.complete & (t.mean_z >= t.wang_mean_z) & (t.n_placed >= t.wang_n_placed)


def llm_not_below(t):
    """The blocks where the LLM's wall is not below the rule's (the same standard, the rule in Wang's place)."""
    return (t.llm_mean_z >= t.mean_z) & (t.llm_n_placed >= t.n_placed)


def rule_not_below_llm(t):
    return t.complete & (t.mean_z >= t.llm_mean_z) & (t.n_placed >= t.llm_n_placed)


def reading(count, llm):
    """A rule's count of walls not below Wang against the LLM's (docs/plans/whole-wall-rules.md)."""
    return ("the rule reaches the LLM's count without an LLM" if count >= llm
            else "the LLM is 3 or more walls ahead of the rule" if count <= llm - 3 else "not separated")


def problems(t):
    """Why the rule is not read: a wall missing, failed or not on Wang's stones."""
    out = [f"{len(t)} of the {len(BLOCKS) * len(RULES)} walls are built"] if len(t) < len(BLOCKS) * len(RULES) else []
    out += [f"{r.case_id} ({r.seed} {r.scene} {r.rule}): not complete" for r in t[~t.complete].itertuples()]
    out += [f"{r.case_id}: the first offer is not Wang's" for r in t[~t.same_first_offer].itertuples()]
    out += [f"{r.case_id}: no decision differs from the cascade, yet the scores are not Wang's"
            for r in t[t.as_wang & ~t.scores_as_wang].itertuples()]
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="CSV of the per-wall rows")
    args = parser.parse_args(argv)
    llm = pd.concat([ww.table(False, whole_wall), ww.table(False, whole_confirm_wall)]).set_index(["seed", "scene"])
    t = pd.DataFrame([rule_row(seed, scene, rule, case, llm.loc[(seed, scene)])
                      for seed, scene in BLOCKS for rule, case in cases(seed, scene).items()])
    if args.out:
        t.round(6).to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(t.drop(columns=["case_id", "n_input", "scores_as_wang", "same_first_offer"]).round(3).to_string(index=False))
    n_llm = int(ww.not_below_wang(llm).sum())
    print(f"LLM: not below Wang {n_llm}/{len(llm)}")
    found = problems(t) if len(t) else ["no wall is built"]
    for rule, r in t.groupby("rule", sort=False) if len(t) else ():
        ok = not_below_wang(r)
        tool = "" if r.mixed.isna().all() else (f"; safe and unsafe candidates at {int(r.mixed.sum())} decisions, "
                                                 f"unsafe placed {int(r.unsafe_picks.sum())}")
        print(f"{rule}: not below Wang {ok.sum()}/{len(r)} ({(ok & ~r.as_wang).sum()} of them not Wang's wall again; "
              f"{r.as_wang.sum()} walls are Wang's again), mean z - Wang's {r.mean_z_diff.mean():.3f}, "
              + ", ".join(f"{c} {r[f'diff_{c}'].mean():.3f}" for c in step3.ZCOLS)
              + f"; {r.switches.sum()}/{r.decisions.sum()} decisions differ from the cascade{tool}; the LLM's wall "
              f"not below the rule's {llm_not_below(r).sum()}/{len(r)}, the rule's not below the LLM's "
              f"{rule_not_below_llm(r).sum()}/{len(r)}, mean z LLM - rule {(r.llm_mean_z - r.mean_z).mean():.3f}"
              + ("" if found else f"; reading: {reading(ok.sum(), n_llm)}"))
    print("\n".join(found) if found else "checks: no problem")
    tool = not_below_wang(t[t.rule == "wang_safe"]).sum() if len(t) else 0
    print("reading:", "not read" if found else f"wang_safe (Wang's choice with check_after): {reading(tool, n_llm)}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
