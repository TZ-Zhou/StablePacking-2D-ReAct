"""The reading of docs/plans/llm-pick-repeats.md: how much the whole-wall LLM pick ('pick-4', the walls of
research/llm_walls/repeat_wall.py) differs from run to run on the same stones, and how many blocks a later
experiment needs to tell a difference from that spread. A measurement: nothing passes or fails.

Usage: python -B -m research.analysis.llm_repeats [--loopback]

Per wall the row of llm_whole_wall (z in the block's 120 factorial walls; the four metric z's as differences from
Wang's wall of the block, a constant per block). Printed: per block the mean, sd, min and max over its repeats, the
number of distinct walls and the first decision at which two repeats split; the pooled within-block sd with a
chi-square interval, over all blocks and per scene; the outcomes (llm_lessons.outcome) of every pair of repeats of a
block; the walls not below Wang per block; the mean per round; the old pick-4 wall and the lessons wall of each
block against the block's repeats (described only, no reading is reopened); the blocks a paired experiment needs
(sizes). The per-wall table goes to research/studies/pick-repeats[-loopback]/llm-repeats.csv.
"""

import argparse
import itertools
import json
import math
import sys

import pandas as pd
from scipy import stats

from research.analysis import llm_endgame_pick as step3
from research.analysis import llm_lessons
from research.analysis import llm_whole_wall as ww
from research.llm_walls import gpt_wall, lessons_wall, repeat_wall, whole_confirm_wall, whole_wall
from research.paths import STUDIES_DIR
from research.storage import case_id as cid

KEY = ["seed", "scene"]
COLS = ["llm_mean_z", *(f"diff_{c}" for c in step3.ZCOLS)]
DELTAS, REPEATS = (0.25, 0.5, 0.75, 1.0), (1, 2, 4)
ALPHA, POWER = 0.05, 0.80


def picks(case_id):
    """The selection of each decision of a wall, in order: equal for two walls of a block means the same wall."""
    trace = next(cid.wall_dir(case_id).glob("result*")) / "trace" / "controller.jsonl"
    records = (json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines())
    return tuple(r["final"]["index"] for r in records if r.get("decision_point"))


def table(loopback=False, x=repeat_wall):
    """One row per finished repeat: llm_whole_wall's row, its round and its picks; None if there is none. x: the
    batch (safe_top_wall has the same states)."""
    journals = gpt_wall.journals(loopback, x)
    rows = []
    for seed, (scene, r) in x.STATES:
        j = journals.get(x.name_of(seed, (scene, r))) or {}
        if j.get("state") == "finished":
            rows.append({**ww.state_row(seed, scene, j), "round": r, "picks": picks(j["case_id"])})
    return pd.DataFrame(rows) if rows else None


def first_split(a, b):
    """The first decision (from 1) at which two walls pick differently, None if they never do."""
    return next((i + 1 for i, (p, q) in enumerate(itertools.zip_longest(a, b)) if p != q), None)


def blocks(t):
    """Per block: its repeats, the distinct walls among them, the earliest split and the spread of each number."""
    rows = []
    for (seed, scene), g in t.groupby(KEY, sort=False):
        splits = [s for a, b in itertools.combinations(g.picks, 2) if (s := first_split(a, b))]
        row = {"seed": seed, "scene": scene, "walls": len(g), "distinct": g.picks.nunique(),
               "first_split": min(splits, default=None), "not_below_wang": int(ww.not_below_wang(g).sum())}
        for c in (*COLS, "llm_n_placed"):
            row.update({f"{c}_{k}": getattr(g[c], k)() for k in ("mean", "std", "min", "max")})
        rows.append(row)
    return pd.DataFrame(rows)


def pooled(t, col):
    """(the pooled within-block sd of a column, its degrees of freedom); nan without a block of two walls."""
    g = t.groupby(KEY)[col]
    df = int((g.count() - 1).sum())
    return (math.sqrt(((g.count() - 1) * g.var()).sum() / df) if df else float("nan")), df


def interval(s, df, level=0.95):
    """The chi-square interval of an sd with df degrees of freedom (normal walls assumed: a rough guide)."""
    lo, hi = stats.chi2.ppf([(1 + level) / 2, (1 - level) / 2], df)
    return s * math.sqrt(df / lo), s * math.sqrt(df / hi)


def pairs(t):
    """llm_lessons.outcome of every pair of repeats of a block: the same prompt against itself."""
    walls = t.rename(columns={"llm_mean_z": "mean_z", "llm_n_placed": "n_placed"})
    return [llm_lessons.outcome(a, b) for _, g in walls.groupby(KEY, sort=False)
            for a, b in itertools.combinations(g.to_dict("records"), 2)]


def reach(alpha=ALPHA, power=POWER):
    """The true difference a one-sided test at alpha finds with this power, in standard errors."""
    return stats.norm.isf(alpha) + stats.norm.ppf(power)


def blocks_mean(s, delta, r):
    """Blocks a paired comparison of means needs to find a true difference delta: r walls per arm and block, the
    within-block sd s in both arms, the same effect in every block (a lower bound otherwise)."""
    return math.ceil(reach() ** 2 * 2 * s * s / (r * delta * delta))


def blocks_sign(p, alpha, power=POWER, top=1000):
    """The fewest blocks at which a one-sided sign test at alpha has this power when a block is won with chance p
    (no ties); None above top."""
    for b in range(1, top + 1):
        k = stats.binom.isf(alpha, b, 0.5) + 1  # the fewest wins that reject
        if stats.binom.sf(k - 1, b, p) >= power:
            return b
    return None


def sizes(s):
    """Per true difference and walls per arm: the blocks needed by the means and by the sign test at 0.05 and 0.20."""
    rows = []
    for delta, r in itertools.product(DELTAS, REPEATS):
        p = float(stats.norm.cdf(delta / (s * math.sqrt(2 / r))))
        rows.append({"delta": delta, "walls_per_arm": r, "blocks_mean": blocks_mean(s, delta, r), "p_win": p,
                     "blocks_sign_05": blocks_sign(p, 0.05), "blocks_sign_20": blocks_sign(p, 0.20)})
    return pd.DataFrame(rows)


def old_walls(t):
    """{name: per block the old wall minus the mean of the block's repeats} for the pick-4 walls built once and the
    lessons walls; the repeats per block."""
    g = t.groupby(KEY)
    plain = pd.concat([ww.table(False, whole_wall), ww.table(False, whole_confirm_wall)])
    return {name: (other.set_index(KEY)[COLS] - g[COLS].mean()).dropna()
            for name, other in (("pick4", plain), ("lessons", ww.table(False, lessons_wall)))}, g.size()


def report(t):
    b = blocks(t)
    print(b.round(3).to_string(index=False))
    print("\npooled within-block sd (95% interval, degrees of freedom):")
    for scope, part in (("all", t), *((s, t[t.scene == s]) for s in sorted(set(t.scene)))):
        for c in (*COLS, "llm_n_placed"):
            s, df = pooled(part, c)
            lo, hi = interval(s, df) if df else (float("nan"),) * 2
            print(f"  {scope} {c}: {s:.3f} ({lo:.3f}-{hi:.3f}, {df})")
    o = pairs(t)
    if o:
        print(f"\npairs of repeats of a block: {len(o)}, decided {len(o) - o.count('tie')} "
              f"({1 - o.count('tie') / len(o):.0%}), tied {o.count('tie')}")
    q = b.not_below_wang / b.walls
    print(f"not below Wang {int(b.not_below_wang.sum())} of {int(b.walls.sum())} walls; one wall a block: expected "
          f"{q.sum():.2f} of {len(b)} blocks, sd {math.sqrt((q * (1 - q)).sum()):.2f}")
    print("\nper round:\n" + t.groupby("round").llm_mean_z.agg(["count", "mean"]).round(3).to_string())
    old, n = old_walls(t)
    for c in ("llm_mean_z", "diff_z_F_LR"):
        s, df = pooled(t, c)
        if not s > 0:
            print(f"\n{c}: no spread measured, nothing to compute")
            continue
        for name, d in old.items():
            se = s * math.sqrt((1 + 1 / n[d.index]).sum()) / len(d)
            print(f"\n{c}: old {name} wall minus the block's repeats, in sd: "
                  + ", ".join(f"{seed} {scene[3:]} {v:+.2f}" for (seed, scene), v in (d[c] / s).items())
                  + f"; mean {d[c].mean():+.3f}, se {se:.3f}")
        for label, sigma in (("the estimate", s), ("the interval's upper end", interval(s, df)[1])):
            print(f"\n{c}: blocks needed at sd {sigma:.3f} ({label}):")
            print(sizes(sigma).round(3).to_string(index=False))
            print("smallest difference found, means: " + ", ".join(
                f"{nb} blocks x {r}: {reach() * sigma * math.sqrt(2 / (r * nb)):.2f}"
                for nb in (10, 20) for r in REPEATS))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--loopback", action="store_true")
    args = parser.parse_args(argv)
    t = table(args.loopback)
    if t is None:
        print("not read: no repeat is finished")
        return 1
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    t.assign(picks=t.picks.map(lambda p: " ".join(map(str, p)))).to_csv(
        STUDIES_DIR / repeat_wall.STUDIES[args.loopback] / "llm-repeats.csv", index=False, lineterminator="\n")
    print(f"{len(t)} of the {len(repeat_wall.STATES)} repeats are finished")
    report(t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
