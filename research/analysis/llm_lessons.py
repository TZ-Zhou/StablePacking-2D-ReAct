"""The reading of docs/plans/llm-lessons.md, stage A: per TH block the lessons wall (research/llm_walls/
lessons_wall.py) against the whole-wall pick without lessons ('pick-4', the paid walls of whole_wall.py and
whole_confirm_wall.py) and against the free rules pick_safe and wang_safe (runs/batch/whole-wall-rules.csv).

Usage: python -B -m research.analysis.llm_lessons [--loopback]

A is not below B if A is complete, its mean z is not lower and it places no fewer stones (z in the block's 120
factorial walls). On a block A wins if it is not below B and B is not not below it, loses the other way round, and
ties otherwise; a lessons wall that is not complete loses. Wins and losses go to a one-sided sign test; against
pick-4 p <= SCREEN reads 'a sign of help, worth confirming', the same the other way round 'possibly harmful', else
'no difference seen'. Read once, when all 10 walls are finished; --loopback reads the rehearsal's walls (any number).
The table goes to research/studies/lessons-walls[-loopback]/llm-lessons.csv.
"""

import argparse
import math
import sys

import pandas as pd

from research.analysis import llm_whole_wall as ww
from research.llm_walls import lessons_wall, whole_confirm_wall, whole_wall
from research.paths import BATCH_RUNS, STUDIES_DIR

SCREEN = 0.20
OPPONENTS = ("pick4", "pick_safe", "wang_safe")


def outcome(a, b):
    """'win', 'loss' or 'tie' of wall a against wall b on one block; a wall is {complete, mean_z, n_placed}."""
    def not_below(p, q):
        return bool(p["complete"]) and p["mean_z"] >= q["mean_z"] and p["n_placed"] >= q["n_placed"]
    if not a["complete"]:
        return "loss"
    ab, ba = not_below(a, b), not_below(b, a)
    return "win" if ab and not ba else "loss" if ba and not ab else "tie"


def sign_p(wins, losses):
    """One-sided sign test: the chance of at least `wins` of wins + losses at even odds."""
    n = wins + losses
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def reading(outcomes):
    wins, losses = outcomes.count("win"), outcomes.count("loss")
    return ("a sign of help, worth confirming" if sign_p(wins, losses) <= SCREEN else
            "possibly harmful" if sign_p(losses, wins) <= SCREEN else "no difference seen")


def table(loopback=False):
    """One row per finished lessons wall with its three opponents and outcomes, None if there is none."""
    mine = ww.table(loopback, lessons_wall, partial=True)
    if mine is None:
        return None
    key = ["seed", "scene"]
    names = {"llm_mean_z": "mean_z", "llm_n_placed": "n_placed"}
    plain = pd.concat([ww.table(False, whole_wall), ww.table(False, whole_confirm_wall)]).rename(columns=names)
    rules = pd.read_csv(BATCH_RUNS / "whole-wall-rules.csv")
    others = {"pick4": plain, **{r: rules[rules.rule == r] for r in OPPONENTS[1:]}}
    out = mine.rename(columns=names)[[*key, "case_id", "complete", "mean_z", "n_placed", "n_input", "wang_mean_z",
                                      "wang_n_placed", "same_first_offer", "cost_usd"]].copy()
    out["not_below_wang"] = ww.not_below_wang(mine).to_numpy()
    for name, t in others.items():
        t = out[key].merge(t, on=key, how="left", validate="1:1")
        out[f"{name}_mean_z"], out[f"{name}_n_placed"] = t.mean_z.to_numpy(), t.n_placed.to_numpy()
        out[f"vs_{name}"] = [outcome(a, b) for a, b in zip(out.to_dict("records"), t.to_dict("records"))]
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--loopback", action="store_true")
    args = parser.parse_args(argv)
    t = table(args.loopback)
    if t is None or (len(t) < len(lessons_wall.STATES) and not args.loopback):
        print("not read: not every lessons wall is finished")
        return 1
    t.to_csv(STUDIES_DIR / lessons_wall.STUDIES[args.loopback] / "llm-lessons.csv", index=False, lineterminator="\n")
    print(t.round(3).to_string(index=False))
    print(f"lessons walls not below Wang: {int(t.not_below_wang.sum())} of {len(t)}")
    for name in OPPONENTS:
        o = list(t[f"vs_{name}"])
        wins, losses = o.count("win"), o.count("loss")
        print(f"against {name}: {wins} wins, {losses} losses, {o.count('tie')} ties, one-sided sign test p = "
              f"{sign_p(wins, losses):.3f}" + (f": {reading(o)}" if name == "pick4" else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
