"""The post-hoc description of docs/plans/llm-pick-repeats.md (its last section): the 40 pick-4 repeats against the
walls without run-to-run spread on the same 10 TH blocks, Wang's Paper wall and the rules wang_safe and pick_safe
(runs/batch/whole-wall-rules.csv). Made after the walls were read: a description, nothing passes or fails.

Usage: python -B -m research.analysis.llm_repeats_against

Per quantity (the mean z, the four metric z's, the stones) a difference is the LLM's wall minus the opponent's wall
of the block; the opponent's wall is one fixed wall, so the spread of a difference is the LLM's. Printed and written
to research/studies/pick-repeats: per block the mean of its 4 differences in standard errors of its scene's pooled
within-block sd (llm-repeats-against-blocks.csv); over all blocks and per scene the mean difference, its standard
error from the pooled within-block sd, and next to it the sd of the block means and the standard error that gives
(llm-repeats-against.csv); the outcome (llm_lessons.outcome) of each wall against the opponent's.
"""

import math
import sys

import pandas as pd

from research.analysis import llm_endgame_pick as step3
from research.analysis import llm_lessons, llm_repeats
from research.llm_walls import repeat_wall
from research.paths import BATCH_RUNS, STUDIES_DIR

KEY = llm_repeats.KEY
COLS = ["mean_z", *(f"diff_{c}" for c in step3.ZCOLS), "n_placed"]  # the metric z's as differences from Wang's wall
RULES = ("wang_safe", "pick_safe")
OUTSIDE = 2  # standard errors from which a difference is described as outside the spread
OUTCOMES = ("win", "tie", "loss")


def opponents(walls):
    """{name: one row per block with complete and COLS}: Wang's wall of the repeats' table and the rule walls."""
    wang = walls.drop_duplicates(KEY).assign(complete=True, mean_z=lambda d: d.wang_mean_z,
                                             n_placed=lambda d: d.wang_n_placed, **dict.fromkeys(COLS[1:-1], 0.0))
    rules = pd.read_csv(BATCH_RUNS / "whole-wall-rules.csv")
    return {"wang": wang, **{r: rules[rules.rule == r] for r in RULES}}


def differences(walls, other):
    """Per LLM wall its COLS minus the block's opponent wall, and its outcome against that wall."""
    m = walls.merge(other[[*KEY, "complete", *COLS]], on=KEY, suffixes=("", "_other"), validate="m:1")
    d = m[[*KEY, "round"]].copy()
    for c in COLS:
        d[c] = m[c] - m[f"{c}_other"]
    d["outcome"] = [llm_lessons.outcome(a, {k: a[f"{k}_other"] for k in ("complete", "mean_z", "n_placed")})
                    for a in m.to_dict("records")]
    return d


def in_se(diff, se):
    """A difference in standard errors and how it is described; without spread there is nothing to compare."""
    if not se > 0:
        return float("nan"), "no spread"
    return diff / se, "outside the spread" if abs(diff / se) >= OUTSIDE else "inside the spread"


def summary(d):
    """Per quantity the mean difference of the walls of d with its two standard errors: from the pooled
    within-block sd (these blocks), and from the sd of the block means (other blocks)."""
    rows = []
    for c in COLS:
        means = d.groupby(KEY)[c].mean()
        sd = llm_repeats.pooled(d, c)[0]
        ratio, said = in_se(means.mean(), sd / math.sqrt(len(d)))
        rows.append({"quantity": c, "blocks": len(means), "walls": len(d), "mean_diff": means.mean(),
                     "sd_within": sd, "se_within": sd / math.sqrt(len(d)), "in_se_within": ratio, "said": said,
                     "sd_blocks": means.std(), "se_blocks": means.std() / math.sqrt(len(means)),
                     "in_se_blocks": in_se(means.mean(), means.std() / math.sqrt(len(means)))[0]})
    return pd.DataFrame(rows)


def per_block(d):
    """Per block the mean difference of each quantity, in standard errors of its scene's pooled sd, and the
    outcomes of its walls."""
    rows = []
    for (seed, scene), g in d.groupby(KEY, sort=False):
        row = {"seed": seed, "scene": scene, **{o: int((g.outcome == o).sum()) for o in OUTCOMES}}
        for c in COLS:
            se = llm_repeats.pooled(d[d.scene == scene], c)[0] / math.sqrt(len(g))
            row[c], row[f"{c}_in_se"] = g[c].mean(), in_se(g[c].mean(), se)[0]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    t = llm_repeats.table(False)
    if t is None or len(t) < len(repeat_wall.STATES):
        print("not described: not every repeat is finished")
        return 1
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    walls = t.rename(columns={"llm_mean_z": "mean_z", "llm_n_placed": "n_placed"})
    totals, blocks = [], []
    for name, other in opponents(walls).items():
        d = differences(walls, other)
        blocks.append(per_block(d).assign(opponent=name))
        for scope, part in (("all", d), *((s, d[d.scene == s]) for s in sorted(set(d.scene)))):
            totals.append(summary(part).assign(opponent=name, scope=scope,
                                               **{o: int((part.outcome == o).sum()) for o in OUTCOMES}))
    totals, blocks = (pd.concat(x, ignore_index=True) for x in (totals, blocks))
    totals, blocks = (x[["opponent", *x.columns.drop("opponent")]] for x in (totals, blocks))
    folder = STUDIES_DIR / repeat_wall.STUDIES[False]
    totals.to_csv(folder / "llm-repeats-against.csv", index=False, lineterminator="\n")
    blocks.to_csv(folder / "llm-repeats-against-blocks.csv", index=False, lineterminator="\n")
    print(totals.round(3).to_string(index=False))
    print(blocks.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
