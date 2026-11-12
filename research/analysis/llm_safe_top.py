"""The reading of docs/plans/llm-safe-top-hint.md: the whole-wall LLM pick told which candidates are safe and at the
highest LLC ('pick-5', the walls of research/llm_walls/safe_top_wall.py) against the pick-4 repeats of the same
blocks (research/llm_walls/repeat_wall.py), 4 walls against 4 a block, and described against pick_safe's wall.

Usage: python -B -m research.analysis.llm_safe_top [--loopback]

Read (reading): per quantity the mean over the blocks of (the mean of the block's new walls minus that of its
pick-4 repeats), its standard error from the within-block sd pooled over both batches, and next to it the standard
error from the sd of the block differences; verdict is the plan's table on the F_LR z and the mean z. Described: the
new walls against Wang, wang_safe and pick_safe (llm_repeats_against); per decision whether the pick is in the list
(decisions; for the pick-4 repeats the list they would have been shown), and every pick outside it with its
rationale. Tables go to research/studies/safe-top-walls[-loopback]/.
"""

import argparse
import json
import math
import sys

import numpy as np
import pandas as pd

from research.analysis import llm_repeats
from research.analysis import llm_repeats_against as against
from research.controller import order_controller as oc
from research.llm_walls import gpt_wall, repeat_wall, safe_top_wall
from research.paths import STUDIES_DIR
from research.storage import case_id as cid
from Stonepacker2D.stack.selection import paper_feasible

KEY = llm_repeats.KEY
THRESHOLD = 2  # standard errors (the plan)
SPENT = ["http_requests", "tokens", "cost_usd", "ids_checked", "decisions"]


def decisions(x, loopback=False):
    """One row per LLM pick of the finished walls of x: the list (rebuilt from the trace; equal to the recorded one
    where there is one), the pick, whether it is listed, and the rationale."""
    journals, rows = gpt_wall.journals(loopback, x), []
    for seed, (scene, r) in x.STATES:
        j = journals.get(x.name_of(seed, (scene, r))) or {}
        if j.get("state") != "finished":
            continue
        result = next(cid.wall_dir(j["case_id"]).glob("result*"))
        records = (result / "trace" / "controller.jsonl").read_text(encoding="utf-8").splitlines()
        for rec, (_, a) in zip(map(json.loads, records), oc.trace_events(result)):
            k = rec["final"]["candidate"]
            if k is None:
                continue
            raw, after = np.asarray(a["features_raw"]), np.asarray(a["after"])
            feasible = np.flatnonzero(paper_feasible(raw))
            llc, safe = raw[feasible, 3].round(6), np.array([not (after[i] == 0).any() for i in feasible])
            listed = [int(i) for i in np.flatnonzero(safe & (llc == llc[safe].max()))] if safe.any() else []
            assert rec["observation"].get("safe_top_llc", listed) == listed, (j["case_id"], rec["step"])
            rationale = rec["react"][-1]["arguments"].get("rationale", "")
            rows.append({"seed": seed, "scene": scene, "round": r, "step": rec["step"], "candidates": len(feasible),
                         "safe": int(safe.sum()), "listed": " ".join(map(str, listed)), "pick": k,
                         "in_list": k in listed, "names_list": "safe_top_llc" in rationale, "rationale": rationale})
    return pd.DataFrame(rows)


def reading(new, old):
    """Per quantity: the mean block difference new minus old, in the standard errors of both spreads."""
    rows, n = [], (1 / new.groupby(KEY).size() + 1 / old.groupby(KEY).size()).dropna()
    for c in (*llm_repeats.COLS, "llm_n_placed"):
        d = (new.groupby(KEY)[c].mean() - old.groupby(KEY)[c].mean()).dropna()
        parts = [(s, df) for s, df in (llm_repeats.pooled(new, c), llm_repeats.pooled(old, c)) if df]
        df = sum(f for _, f in parts)
        sd = math.sqrt(sum(f * s * s for s, f in parts) / df) if df else float("nan")
        se, se_blocks = sd * math.sqrt(n.sum()) / len(d), d.std() / math.sqrt(len(d))
        rows.append({"quantity": c, "blocks": len(d), "mean_diff": d.mean(), "sd_within": sd, "df": df,
                     "se_within": se, "in_se_within": against.in_se(d.mean(), se)[0], "sd_blocks": d.std(),
                     "se_blocks": se_blocks, "in_se_blocks": against.in_se(d.mean(), se_blocks)[0]})
    return pd.DataFrame(rows)


def verdict(f_lr, mean_z):
    """The plan's reading of the two differences in standard errors."""
    if math.isnan(f_lr) or math.isnan(mean_z):
        return "not read: no spread measured"
    if f_lr >= THRESHOLD:
        return "F_LR raised and the mean z kept" if mean_z > -THRESHOLD else "F_LR raised but the mean z lowered"
    return "F_LR lowered" if f_lr <= -THRESHOLD else "no difference seen"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--loopback", action="store_true")
    args = parser.parse_args(argv)
    new, old = (llm_repeats.table(args.loopback, x) for x in (safe_top_wall, repeat_wall))
    if new is None or old is None:
        print("not read: no finished wall on one side")
        return 1
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    folder = STUDIES_DIR / safe_top_wall.STUDIES[args.loopback]
    new.assign(picks=new.picks.map(lambda p: " ".join(map(str, p)))).to_csv(
        folder / "llm-safe-top.csv", index=False, lineterminator="\n")
    print(f"{len(new)} of the {len(safe_top_wall.STATES)} walls are finished; pick-4 repeats: {len(old)}")
    parts = []
    for scope, a, b in (("all", new, old), *((s, new[new.scene == s], old[old.scene == s])
                                             for s in sorted(set(new.scene)))):
        parts.append(reading(a, b).assign(scope=scope))
    r = pd.concat(parts, ignore_index=True)
    r.to_csv(folder / "llm-safe-top-reading.csv", index=False, lineterminator="\n")
    print(r.round(3).to_string(index=False))
    m = r[r.scope == "all"].set_index("quantity").in_se_within
    print("\nreading:", verdict(m["diff_z_F_LR"], m["llm_mean_z"]),
          "" if len(new) == len(safe_top_wall.STATES) else "(NOT FINAL: not every wall is finished)")
    walls = new.rename(columns={"llm_mean_z": "mean_z", "llm_n_placed": "n_placed"})
    totals = pd.concat([against.summary(against.differences(walls, other)).assign(opponent=name)
                        for name, other in against.opponents(walls).items()], ignore_index=True)
    totals.to_csv(folder / "llm-safe-top-against.csv", index=False, lineterminator="\n")
    print("\nagainst the walls without spread:\n" + totals.round(3).to_string(index=False))
    print("\nper wall, new and pick-4:\n" + pd.DataFrame({"new": new[SPENT].mean(), "pick4": old[SPENT].mean()})
          .round(3).to_string())
    d, was = decisions(safe_top_wall, args.loopback), decisions(repeat_wall, args.loopback)
    out = d[~d.in_list]
    out.to_csv(folder / "llm-safe-top-unlisted.csv", index=False, lineterminator="\n")
    print(f"\npicks in the list: new {d.in_list.mean():.3f} of {len(d)}, pick-4 (the list it was not shown) "
          f"{was.in_list.mean():.3f} of {len(was)}; new picks outside the list {len(out)}, "
          f"of them {int(out.names_list.sum())} with safe_top_llc in the rationale")
    return 0


if __name__ == "__main__":
    sys.exit(main())
