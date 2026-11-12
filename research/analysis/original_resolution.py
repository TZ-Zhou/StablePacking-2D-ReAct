"""The recheck of docs/plans/original-resolution-recheck.md: the finished walls of the 10 TH blocks with the four
metrics recomputed on the paper's image (02 at the original resolution, then 03) instead of the simulation grid, and
three conclusions read again by their 2 standard error rules. Free: nothing is built, no LLM is called.

Usage: python -B -m research.analysis.original_resolution walls   (once: the metrics of every wall, about 10 minutes)
       python -B -m research.analysis.original_resolution read    (the checks, then the tables)

Walls: per block its 120 factorial walls (the z scale; Wang's wall is the Paper order's), wang_safe and pick_safe
(runs/batch/whole-wall-rules.csv), the 4 pick-4 repeats and the 4 pick-5 walls. Renderings, each with its own z scale
and never mixed in a comparison: G the recorded grid values; A wall_metrics.wall_quality on the author canvas (02 as
it is); B on the corrected canvas (stones whole); C the corrected canvas with every stone drawn, computed where B
leaves a stone out and B's values elsewhere. The comparisons are llm_repeats_against's and llm_safe_top.reading's.

Checks, each stops the reading: (1, walls) the port equals the values Wang's own code gives for the 7 walls of
docs/reviews/wall-pictures/metrics-check.csv at CHECKED; (2) G through this path gives the published tables, the
rule walls' z rounded to 6 decimals as their source has them; (3)
F_LR is the recorded one in every rendering; (4) every wall is evaluated, no metric missing. A conclusion 'holds'
when A and B both meet its rule, 'does not hold' when neither does, else 'depends on the rendering'.
Tables go to research/studies/original-resolution-recheck/.
"""

import io
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from research.analysis import llm_endgame_pick as step3
from research.analysis import llm_repeats, llm_safe_top
from research.analysis import llm_repeats_against as against
from research.analysis import llm_whole_wall as ww
from research.analysis import myopia_tables as mt
from research.analysis.endgame_arena import METRICS
from research.llm_walls import gpt_wall, repeat_wall, safe_top_wall
from research.paths import BATCH_RUNS, DATASETS_DIR, REPO, STUDIES_DIR
from research.storage import case_id as cid
from research.walls import wall_metrics as wm

FOLDER = STUDIES_DIR / "original-resolution-recheck"
CHECKED = "e85b4ae"  # the commit of the independent check with Wang's own code
KEY = llm_repeats.KEY
SEEDS, SCENES = (313, 484, 717, 720, 761), ("TH_T80", "TH_T20")
RENDERINGS = {"A": {"canvas": "author"}, "B": {"canvas": "corrected"},
              "C": {"canvas": "corrected", "leave_out": False}}
LLM = {"pick4": repeat_wall, "pick5": safe_top_wall}
OPPONENTS = ("wang", *against.RULES)
WALLS = {"factorial": 1200, "wang_safe": 10, "pick_safe": 10, "pick4": 40, "pick5": 40}
TOLERANCE = 1e-9
COLS = [*against.COLS, *METRICS]  # the quantities of a comparison: the z's, the stones and the raw metrics
# (conclusion, comparison, quantity, the rule on the difference in standard errors)
CONCLUSIONS = (("LLM above Wang", "pick4-wang", "mean_z", lambda x: x >= against.OUTSIDE),
               ("LLM level with pick_safe", "pick4-pick_safe", "mean_z", lambda x: abs(x) < against.OUTSIDE),
               ("pick-5 lowers F_AV", "pick5-pick4", "diff_z_F_AV", lambda x: x <= -against.OUTSIDE),
               ("pick-5 raises F_LR", "pick5-pick4", "diff_z_F_LR", lambda x: x >= against.OUTSIDE))


def grid(case):
    """The recorded grid metrics of a traced wall: its last placement."""
    placed = pd.read_csv(case / "derived" / "steps.csv").dropna(subset=["F_LR"]).iloc[-1]
    return {f"{m}_G": placed[m] for m in METRICS}


def inventory():
    """One row per wall: its group, block, folder, stones and recorded grid metrics; (the table, the case folders)."""
    rows, cases = [], []
    rules = pd.read_csv(BATCH_RUNS / "whole-wall-rules.csv")
    for seed in SEEDS:
        for scene in SCENES:
            batch = BATCH_RUNS / f"factorial-s{seed}-{scene}"
            block = pd.read_csv(DATASETS_DIR / batch.name / "walls.csv", dtype={"filter_policy": str})
            assert (block.status == mt.FINISHED_STATUS).all(), batch.name
            for w in block.itertuples():
                rows.append({"group": "factorial", "seed": seed, "scene": scene, "filter_policy": w.filter_policy,
                             "wall": w.wall, "complete": True, "n_placed": w.n_placed,
                             **{f"{m}_G": getattr(w, m) for m in METRICS}})
                cases.append(batch / "cases" / w.wall)
            mine = rules[(rules.seed == seed) & (rules.scene == scene) & rules.rule.isin(against.RULES)]
            for w in mine.itertuples():
                cases.append(BATCH_RUNS / f"whole-rules-s{seed}-{scene}" / "cases" / w.case_id)
                rows.append({"group": w.rule, "seed": seed, "scene": scene, "wall": w.case_id,
                             "complete": w.complete, "n_placed": w.n_placed, **grid(cases[-1])})
    for group, x in LLM.items():
        journals = gpt_wall.journals(False, x)
        for seed, (scene, r) in x.STATES:
            cases.append(cid.wall_dir(journals[x.name_of(seed, (scene, r))]["case_id"]))
            wall = step3.llm_wall(cases[-1])
            rows.append({"group": group, "seed": seed, "scene": scene, "round": r, "wall": cases[-1].name,
                         "complete": wall["complete"], "n_placed": wall["n_placed"],
                         **{f"{m}_G": wall[m] for m in METRICS}})
    t = pd.DataFrame(rows)
    assert t.group.value_counts().to_dict() == WALLS, t.group.value_counts()
    return t, cases


def measure(case, r):
    """The four metrics of a wall in rendering r, the stones 02's overlap rule leaves out and the pixels cut."""
    result, = Path(case).glob("result*")
    q = wm.wall_quality(case, result, **RENDERINGS[r])
    stones = q["render"]["stones"]
    return {**{m: q[m] for m in METRICS}, "left_out": sum(bool(s["left_out"]) for s in stones),
            "cut_px": sum(s["lost_to_crop"] for s in stones)}


def render(case):
    """A wall in A, B and C (B's values unless B leaves a stone out); an error is kept, to stop the reading."""
    try:
        a, b = measure(case, "A"), measure(case, "B")
        c = measure(case, "C") if b["left_out"] else b
    except Exception as e:  # noqa: BLE001  whatever fails, the wall is listed and nothing is read
        return {"error": f"{type(e).__name__}: {e}"}
    return {**{f"{k}_{r}": v for r, q in zip("ABC", (a, b, c)) for k, v in q.items()}, "error": ""}


def show(name):
    out = subprocess.run(["git", "show", f"{CHECKED}:docs/reviews/wall-pictures/{name}"], cwd=REPO,
                         capture_output=True, encoding="utf-8", check=True).stdout
    return pd.read_csv(io.StringIO(out))


def against_wangs_code():
    """Check 1: the largest difference between this port and Wang's own 02 and 03 on the 7 walls of CHECKED."""
    t = show("walls.csv")[["key", "case"]].merge(show("metrics-check.csv"), on="key")
    return max(abs(q[m] - w[f"{m}_{canvas}"]) for w in t.to_dict("records")
               for r, canvas in (("A", "author"), ("B", "corrected"))
               for q in [measure(w["case"], r)] for m in ("F_SF", "F_AH", "F_AV"))


def walls():
    if (FOLDER / "walls.csv").exists():
        print("refused: walls.csv exists, the walls are measured once")
        return 1
    worst = against_wangs_code()
    print(f"check 1: largest difference from Wang's own code on the 7 walls of {CHECKED}: {worst:.3g}")
    if not worst < TOLERANCE:
        return 1
    t, cases = inventory()
    with Pool(8) as pool:
        t = pd.concat([t, pd.DataFrame(pool.map(render, cases, chunksize=10))], axis=1)
    whole = [i for i in t.index if t.error[i] == "" and t.left_out_B[i] == 0][:10]
    for i in whole:  # C is B where B draws every stone: the premise of copying B's values
        c = measure(cases[i], "C")
        assert all(c[k] == t.loc[i, f"{k}_B"] for k in c), cases[i]
    FOLDER.mkdir(exist_ok=True)
    t.to_csv(FOLDER / "walls.csv", index=False, lineterminator="\n")
    failed = t[t.error != ""]
    print(f"{len(t)} walls, {len(failed)} not evaluated; C equals B on {len(whole)} walls without a stone left out")
    print(failed[["group", "seed", "scene", "wall", "error"]].to_string(index=False) if len(failed) else "", end="")
    return 1 if len(failed) else 0


def scored(w, r):
    """The walls with the z scores of rendering r in their block's 120 walls, Wang's wall also as the group 'wang',
    and each z as a difference from Wang's wall (the columns of llm_whole_wall's rows)."""
    v, parts = w.rename(columns={f"{m}_{r}": m for m in METRICS}), []
    for _, g in v.groupby(KEY, sort=False):
        block = g[g.group == "factorial"]
        g = pd.concat([g, block[block.filter_policy == ww.PAPER].assign(group="wang")], ignore_index=True)
        g = pd.concat([g[[*KEY, "group", "round", "complete", "n_placed", *METRICS]], step3.zscores(g, block)], axis=1)
        wang = g[g.group == "wang"].iloc[0]
        parts.append(g.assign(wang_mean_z=wang.mean_z, wang_n_placed=wang.n_placed,
                              **{f"diff_{c}": g[c] - wang[c] for c in step3.ZCOLS}))
    return pd.concat(parts, ignore_index=True)


def comparisons(s):
    """(per comparison, scope and quantity the mean difference in standard errors, as llm_repeats_against.summary
    and llm_safe_top.reading give it; per comparison and block the mean difference of COLS)."""
    rows = []
    scopes = [("all", s), *((scene, s[s.scene == scene]) for scene in sorted(set(s.scene)))]
    for scope, part in scopes:
        for a in LLM:
            for b in OPPONENTS:
                d = against.differences(part[part.group == a], part[part.group == b])
                rows.append(against.summary(d).assign(comparison=f"{a}-{b}", scope=scope))
        new, old = (part[part.group == g].rename(columns={"mean_z": "llm_mean_z", "n_placed": "llm_n_placed"})
                    for g in ("pick5", "pick4"))
        rows.append(llm_safe_top.reading(new, old).assign(comparison="pick5-pick4", scope=scope))
    t = pd.concat(rows, ignore_index=True).drop(columns=["walls", "said", "df"])
    t["quantity"] = t.quantity.str.replace("llm_", "")
    m = s[s.group != "factorial"].groupby([*KEY, "group"])[COLS].mean()
    blocks = pd.concat([(m.xs(a, level="group") - m.xs(b, level="group")).reset_index().assign(comparison=f"{a}-{b}")
                        for a, b in [*((a, b) for a in LLM for b in OPPONENTS), ("pick5", "pick4")]])
    up = []
    for row in t.itertuples():  # the same mean by the other path, and the blocks on its side of zero
        d = blocks[(blocks.comparison == row.comparison) & ((row.scope == "all") | (blocks.scene == row.scope))]
        assert np.isclose(d[row.quantity].mean(), row.mean_diff, rtol=0, atol=TOLERANCE), row
        up.append(int((d[row.quantity] > 0).sum()))
    return t.assign(blocks_positive=up), blocks


def conclusions(reading):
    """Per conclusion and rendering the difference and whether it meets the rule; the verdict is A's and B's."""
    rows = []
    for name, comparison, quantity, rule in CONCLUSIONS:
        r = reading[(reading.comparison == comparison) & (reading.quantity == quantity) & (reading.scope == "all")]
        r = r.assign(meets=r.in_se_within.map(rule)).set_index("rendering")
        both = int(r.meets["A"]) + int(r.meets["B"])
        verdict = ("does not hold", "depends on the rendering", "holds")[both]
        rows += [{"conclusion": name, "rendering": k, "comparison": comparison, "quantity": quantity,
                  **x[["mean_diff", "se_within", "in_se_within", "se_blocks", "in_se_blocks", "blocks_positive",
                       "meets"]].to_dict(), "verdict": verdict} for k, x in r.iterrows()]
    return pd.DataFrame(rows)


def published():
    """The published grid tables in the form of comparisons(): comparison, scope, quantity and three numbers."""
    old = pd.read_csv(STUDIES_DIR / "pick-repeats" / "llm-repeats-against.csv")
    new = pd.read_csv(STUDIES_DIR / "safe-top-walls" / "llm-safe-top-against.csv").assign(scope="all")
    both = pd.read_csv(STUDIES_DIR / "safe-top-walls" / "llm-safe-top-reading.csv").assign(comparison="pick5-pick4")
    t = pd.concat([old.assign(comparison="pick4-" + old.opponent), new.assign(comparison="pick5-" + new.opponent),
                   both.assign(quantity=both.quantity.str.replace("llm_", ""))])
    return t[["comparison", "scope", "quantity", "mean_diff", "se_within", "se_blocks"]]


def off_published(s):
    """Check 2: the largest difference between the comparisons of the grid scores s and the published tables. The
    rule walls' z are rounded to 6 decimals as runs/batch/whole-wall-rules.csv, the published tables' source, has
    them; the reading itself is not rounded."""
    rule, z = s.group.isin(against.RULES), ["mean_z", *(c for c in against.COLS if c.startswith("diff_"))]
    s = s.copy()
    s.loc[rule, z] = s.loc[rule, z].round(6)
    m = published().merge(comparisons(s)[0], on=["comparison", "scope", "quantity"],
                          suffixes=("", "_new"), validate="1:1")
    return max(float((m[c] - m[f"{c}_new"]).abs().max()) for c in ("mean_diff", "se_within", "se_blocks"))


def groups(w, s, r):
    """Per group of walls in rendering r: the walls with a stone left out, the pixels cut and the mean scores."""
    left = (w.assign(left=w[f"left_out_{r}"], cut=w[f"cut_px_{r}"]) if r != "G" else w.assign(left=0, cut=0))
    left = left[left.group != "factorial"].groupby("group").agg(
        walls=("left", "size"), walls_left_out=("left", lambda x: int((x > 0).sum())),
        stones_left_out=("left", "sum"), cut_px_mean=("cut", "mean"))
    llm = s[s.group.isin(list(LLM))]
    below = (llm.complete & (llm.mean_z >= llm.wang_mean_z) & (llm.n_placed >= llm.wang_n_placed)).groupby(llm.group)
    return left.join(s[s.group != "factorial"].groupby("group")[["mean_z", *METRICS]].mean()).join(
        below.sum().rename("not_below_wang")).reset_index()


def measured():
    return pd.read_csv(FOLDER / "walls.csv", dtype={"filter_policy": str}, float_precision="round_trip")


def read():
    w = measured()
    failed = w[w.error.notna()]
    values = w[[f"{m}_{r}" for m in METRICS for r in "GABC"]]
    f_lr = max(float((w[f"F_LR_{r}"] - w.F_LR_G).abs().max()) for r in "ABC")
    print(f"check 4: {len(failed)} walls not evaluated, {int(values.isna().sum().sum())} metrics missing")
    print(f"check 3: largest difference of F_LR from the recorded one: {f_lr:.3g}")
    if len(failed) or values.isna().any().any() or not f_lr < TOLERANCE:
        return 1
    scores = {r: scored(w, r) for r in "GABC"}
    off = off_published(scores["G"])
    print(f"check 2: largest difference of the grid rows from the published tables: {off:.3g}")
    if not off < TOLERANCE:
        return 1
    compared = {r: comparisons(s) for r, s in scores.items()}
    reading = pd.concat([t.assign(rendering=r) for r, (t, _) in compared.items()], ignore_index=True)
    c = conclusions(reading)
    first = ["rendering", "comparison", "scope", "quantity"]
    reading[[*first, *reading.columns.drop(first)]].to_csv(FOLDER / "reading.csv", index=False, lineterminator="\n")
    pd.concat([b.assign(rendering=r) for r, (_, b) in compared.items()]).to_csv(
        FOLDER / "blocks.csv", index=False, lineterminator="\n")
    g = pd.concat([groups(w, s, r).assign(rendering=r) for r, s in scores.items()], ignore_index=True)
    g.to_csv(FOLDER / "groups.csv", index=False, lineterminator="\n")
    c.to_csv(FOLDER / "conclusions.csv", index=False, lineterminator="\n")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print(c.round(3).to_string(index=False))
    print(g.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit({"walls": walls, "read": read}[sys.argv[1]]())
