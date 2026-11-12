"""The reading of docs/plans/th80-pick-safe-gap.md: where the whole-wall LLM pick ('pick-4', the 40 repeats) chooses
differently from the rule pick_safe, what it wrote, and what F_LR does afterwards. Offline from the records: no wall
is built and no LLM is called. A description after the walls were read: nothing passes or fails.

Usage: python -B -m research.analysis.pick_safe_gap

At every decision of an LLM wall the rule's choice in that same state is recomputed from the traced scores and the
traced check_after table of every candidate (rule), and the decision gets a class (classify): same; unsafe (a safe
candidate existed, the LLM placed an unsafe one); lower_llc; same_llc (another of the candidates the rule ranks
first: it differs from the rule only in the rule's tie-break, the lowest id). The LLC of a candidate is the wall's
F_LR after placing it, rounded as LLC is, so llc_diff (the LLM's LLC minus the rule's) is what the step gives up at
once. The other four scores of both candidates are kept as the LLM was shown them. said() finds the sentence of the
rationale about the rule's candidate and tags it by keyword (TAGS): a rough aid, the rationale itself is in the
table.

Four checks, each fatal (nothing is written if one fails): the rule recomputed on the pick_safe walls is the pick
they placed; the records of the LLM walls agree with each other; after every placement F_LR rounded as LLC is the
placed candidate's LLC, and the last F_LR is the wall's recorded one; an LLM wall and its block's pick_safe wall get
the same offers up to their first different pick, which is the first decision not of class same.

Written to research/studies/th80-pick-safe-gap: decisions.csv (a row per decision of an LLM wall), walls.csv (a row
per LLM wall, against its block's pick_safe wall) and paths.csv (F_LR after every placement of the 50 walls).
The question is TH_T80's; TH_T20 is listed next to it as the contrast.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from research.analysis import llm_repeats
from research.analysis import whole_wall_rules as rules
from research.controller import order_controller as oc
from research.controller.endgame import safe_rows
from research.paths import BATCH_RUNS, STUDIES_DIR
from research.storage import case_id as cid
from research.walls.wall_metrics import MU
from Stonepacker2D.stack.selection import paper_feasible

FOLDER = STUDIES_DIR / "th80-pick-safe-gap"
RECORDED = STUDIES_DIR / "original-resolution-recheck" / "walls.csv"  # F_LR_G: every wall's recorded F_LR
TOLERANCE = 1e-9
LLC_DECIMALS = 1  # Stonepacker2D.utils.constant.KINEMATIC_RESULT_DECIMAL
CLASSES = ("same", "unsafe", "lower_llc", "same_llc")
TAGS = {"height": "NHR|height|level|course", "density": "GDS|densit", "interlock": "IWR|interlock|stagger",
        "void": "LFR|void|local fit|flush", "llc": "LLC|stabil|limit load",
        "next": "pose|next|feasible|flexib|leaves|retain|preserv"}
OTHERS = ("NHR", "GDS", "IWR", "LFR")  # the scores next to LLC, as the LLM was shown them
BUT = r"\b(?:but|though|although|however)\b"


def rule(raw, after):
    """(the rows pick_safe ranks first in this state, the safe rows): the highest LLC among the safe rows, among the
    feasible rows if none is safe. It places the first of them (pick.mock_reply's pick_safe)."""
    feasible = paper_feasible(raw)
    safe = safe_rows(feasible, after)
    rows = np.flatnonzero(safe if safe.any() else feasible)
    return rows[raw[rows, 3] == raw[rows, 3].max()], safe


def classify(raw, after, pick):
    """The class of placing row pick where the rule places its row, and what goes with it."""
    top, safe = rule(raw, after)
    row = int(top[0])
    diff = float(raw[pick, 3] - raw[row, 3])
    kind = ("same" if pick == row else "unsafe" if safe.any() and not safe[pick]
            else "lower_llc" if diff < -TOLERANCE else "same_llc")
    return {"top_rows": top, "kind": kind, "n_safe": int(safe.sum()), "n_top": len(top), "llm_safe": bool(safe[pick]),
            "llc_llm": float(raw[pick, 3]), "llc_rule": float(raw[row, 3]), "llc_diff": diff,
            "higher_safe": bool(safe.any() and raw[safe, 3].max() > raw[pick, 3] + TOLERANCE)}


def tags(text):
    return "+".join(k for k, p in TAGS.items() if re.search(p, text, re.I))


def said(rationale, c):
    """How a rationale names candidate c ('pose': stone and rotation, or its id; 'stone'; 'none') and the first
    sentence or clause that does."""
    pose = (rf"\b{c['stone']}\b[^.;]*?\b{c['rotation']}\s*°|\b{c['stone']}\b (?:at )?rotation {c['rotation']}\b"
            rf"|\({c['id']}\)|\bcandidate {c['id']}\b")
    parts = re.split(r"(?<=[.;])\s+|\n+", rationale)
    for level, pattern in (("pose", pose), ("stone", rf"\b{c['stone']}\b")):
        found = next((p for p in parts if re.search(pattern, p)), None)
        if found:
            return level, found
    return "none", ""


def events(case):
    """Per selection of a wall its controller record and its traced arrays, in order."""
    result = next(Path(case).glob("result*"))
    records = [json.loads(x) for x in (result / "trace" / "controller.jsonl").read_text(encoding="utf-8").splitlines()]
    traced = list(oc.trace_events(result))
    if [r["event"] for r in records] != [m["event"] for m, _ in traced]:
        raise ValueError(f"{case}: the controller records are not the traced events")
    return [(r, a) for r, (_, a) in zip(records, traced)]


def placed(case):
    """The placements of a wall by event: nth counts them (step also counts the selections that place nothing),
    F_LR_before is nan before the first."""
    s = pd.read_csv(Path(case) / "derived" / "steps.csv")
    s = s[s.outcome == "placed"].set_index("event")
    return s.assign(nth=range(1, len(s) + 1), F_LR_before=s.F_LR.shift())


def decisions(case):
    """(a row per decision of a pick wall, the problems of check 2)."""
    steps, rows, problems = placed(case), [], []
    for r, a in events(case):
        if not r["decision_point"]:
            continue
        raw, f, cands = np.asarray(a["features_raw"]), r["final"], r["observation"]["candidates"]
        ids = np.flatnonzero(paper_feasible(raw))  # candidate id k is the k-th feasible row
        where = f"{Path(case).name} event {r['event']}"
        if not (len(cands) == len(ids) == r["n_candidates"]
                and ids[f["candidate"]] == f["index"] == int(a["selected"])):
            problems.append(f"{where}: the candidates or the pick are not the traced ones")
            continue
        if r["event"] not in steps.index or steps.stone_id[r["event"]] != f["stone_id"] or abs(
                steps["01_LLC"][r["event"]] - raw[f["index"], 3]) > TOLERANCE:
            problems.append(f"{where}: the pick is not the placement of steps.csv")
            continue
        c, step = classify(raw, a["after"], f["index"]), steps.loc[r["event"]]
        top = [cands[k] for k in np.searchsorted(ids, c.pop("top_rows"))]  # the candidates the rule ranks first
        mine, his = cands[f["candidate"]], top[0]
        asked = {int(i) for k in r["react"] if k["tool"] == "check_after"
                 for i in re.findall(r"\d+", str(k["arguments"].get("candidates")))}
        rationale = next(k["arguments"].get("rationale", "") for k in r["react"][::-1] if k["tool"] == "commit_pick")
        level, sentence = said(rationale, his) if c["kind"] != "same" else ("", "")
        against = re.split(BUT, sentence, maxsplit=1, flags=re.I)
        rows.append({"event": r["event"], "nth": step.nth, "n_candidates": len(cands), **c,
                     "llm_id": mine["id"], "llm_stone": mine["stone"], "llm_rotation": mine["rotation"],
                     "rule_id": his["id"], "rule_stone": his["stone"], "rule_rotation": his["rotation"],
                     **{f"{who}_{k}": x["scores"][k] for who, x in (("llm", mine), ("rule", his)) for k in OTHERS},
                     "cascade_is_llm": f["cascade_index"] == f["index"],
                     "cascade_is_rule": cands[int(np.searchsorted(ids, f["cascade_index"]))]["id"] == his["id"],
                     "asked_llm": mine["id"] in asked, "asked_rule": his["id"] in asked, "n_asked": len(asked),
                     "asked_top": any(k["id"] in asked for k in top),
                     "named_top": any(said(rationale, k)[0] == "pose" for k in top),
                     "F_LR_before": step.F_LR_before, "F_LR_after": step.F_LR,
                     "drop": bool(step.F_LR < step.F_LR_before - TOLERANCE),
                     "mention": level, "tags": tags(sentence), "against_tags": tags(against[1]) if against[1:] else "",
                     "sentence": sentence, "rationale": rationale})
    return rows, problems


def rule_problems(case):
    """Check 1: the rule recomputed at every decision of a pick_safe wall is the row the wall placed."""
    return [f"{Path(case).name} event {r['event']}: the rule gives another row than the wall placed"
            for r, a in events(case)
            if r["decision_point"] and rule(np.asarray(a["features_raw"]), a["after"])[0][0] != r["final"]["index"]]


def lr_problems(case, recorded):
    """Check 3: F_LR after a placement, rounded as LLC is, is the placed candidate's LLC; the last F_LR is the
    wall's recorded one."""
    s, name = placed(case), Path(case).name
    off = (np.round(s.F_LR * MU, LLC_DECIMALS) / MU - s["01_LLC"]).abs()
    return ([f"{name} placement {n}: F_LR does not round to the placed candidate's LLC" for n in s.nth[off > TOLERANCE]]
            + ([] if abs(s.F_LR.iloc[-1] - recorded[name]) < TOLERANCE else [f"{name}: the last F_LR is not recorded"]))


def split_problems(case, other, rows):
    """Check 4: up to their first different pick an LLM wall and the pick_safe wall of its block get the same offer,
    and that pick is the first decision of rows not of class same."""
    first = next((r["event"] for r in rows if r["kind"] != "same"), None)
    for (r, a), (q, b) in zip(events(case), events(other)):
        if not all(np.array_equal(a[k], b[k], equal_nan=True) for k in ("features_raw", "sampled_ids", "after")):
            return [f"{Path(case).name} event {r['event']}: not the offer of {Path(other).name} before the walls split"]
        if r["final"]["index"] != q["final"]["index"]:
            return [] if r["event"] == first else [f"{Path(case).name}: the walls split at event {r['event']}, the "
                                                   f"first decision not of class same is event {first}"]
    return [] if first is None else [f"{Path(case).name}: never splits from {Path(other).name}, yet event {first}"]


def path(case):
    return placed(case).reset_index()[["nth", "stone_id", "F_LR"]]


def below_from(mine, his):
    """The placement from which the LLM wall's F_LR stays below the pick_safe wall's to the last common one; None if
    it does not end below."""
    m = mine.merge(his, on="nth", suffixes=("", "_other"))
    not_below = m.nth[m.F_LR >= m.F_LR_other - TOLERANCE]
    last = not_below.max() if len(not_below) else 0
    return None if last == m.nth.max() else int(last + 1)


def read():
    """(decisions, walls, paths, problems) of the 40 LLM walls against the 10 pick_safe walls."""
    t = llm_repeats.table(False)
    recorded = pd.read_csv(RECORDED).set_index("wall").F_LR_G
    scored = pd.read_csv(BATCH_RUNS / "whole-wall-rules.csv").query("rule == 'pick_safe'").set_index(["seed", "scene"])
    safe = {block: rules.cases(*block)["pick_safe"] for block in rules.BLOCKS}
    problems = [p for case in safe.values() for p in rule_problems(case) + lr_problems(case, recorded)]
    paths = [path(case).assign(seed=seed, scene=scene, wall="pick_safe") for (seed, scene), case in safe.items()]
    table, walls = [], []
    for w in t.itertuples():
        case, other = cid.wall_dir(w.case_id), safe[(w.seed, w.scene)]
        rows, found = decisions(case)
        problems += found + lr_problems(case, recorded) + (split_problems(case, other, rows) if not found else [])
        d = pd.DataFrame(rows).assign(seed=w.seed, scene=w.scene, round=w.round, case_id=w.case_id)
        mine, his = path(case), path(other)
        start = below_from(mine, his)
        walls.append({"seed": w.seed, "scene": w.scene, "round": w.round, "case_id": w.case_id, "decisions": len(d),
                      **{k: int((d.kind == k).sum()) for k in CLASSES}, "llc_diff_sum": d.llc_diff.sum(),
                      "first_diff": d.nth[d.kind != "same"].min(), "drops": int(d["drop"].sum()),
                      "drops_higher_safe": int((d["drop"] & d.higher_safe).sum()),
                      "F_LR": mine.F_LR.iloc[-1], "F_LR_pick_safe": his.F_LR.iloc[-1],
                      "F_LR_diff": mine.F_LR.iloc[-1] - his.F_LR.iloc[-1],
                      "mean_z_diff": w.llm_mean_z - scored.mean_z[(w.seed, w.scene)], "below_from": start,
                      "below_from_kind": None if start is None else "".join(d.kind[d.nth == start])})
        table.append(d)
        paths.append(mine.assign(seed=w.seed, scene=w.scene, wall=f"r{w.round}"))
    key = ["seed", "scene", "round", "case_id"]
    d = pd.concat(table, ignore_index=True)
    return d[[*key, *d.columns.drop(key)]], pd.DataFrame(walls), pd.concat(paths, ignore_index=True), problems


def main():
    d, walls, paths, problems = read()
    if problems:
        print("\n".join(problems[:40]), f"\n{len(problems)} problems: nothing is read or written")
        return 1
    print("checks: no problem")
    FOLDER.mkdir(exist_ok=True)
    paths = paths[["seed", "scene", "wall", *paths.columns[:3]]]
    for name, x in (("decisions", d), ("walls", walls), ("paths", paths)):
        x.to_csv(FOLDER / f"{name}.csv", index=False, lineterminator="\n")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print(pd.crosstab([d.scene, d.seed], d.kind, margins=True).reindex(columns=[*CLASSES, "All"]).to_string())
    d = d.assign(change=d.F_LR_after - d.F_LR_before, named=d.mention.isin(["pose", "stone"]),
                 third=pd.cut(d.nth, [0, 6, 12, 99], labels=["1-6", "7-12", "13+"]),
                 **{f"llm_higher_{k}": d[f"llm_{k}"] > d[f"rule_{k}"] + TOLERANCE for k in OTHERS})
    print("\nper class: the decisions, the summed llc_diff, the summed change of F_LR at them, and counts of them")
    print(d.groupby(["scene", "kind"]).agg(
        n=("kind", "size"), llc_diff=("llc_diff", "sum"), F_LR_change=("change", "sum"), falls=("drop", "sum"),
        asked_rule=("asked_rule", "sum"), named=("named", "sum"), asked_top=("asked_top", "sum"),
        named_top=("named_top", "sum"), cascade_is_llm=("cascade_is_llm", "sum"),
        cascade_is_rule=("cascade_is_rule", "sum"), **{f"llm_higher_{k}": (f"llm_higher_{k}", "sum") for k in OTHERS},
    ).round(2).to_string())
    low = d[d.kind == "lower_llc"]
    print("\nlower_llc: on how many of the four other scores the LLM's candidate is the higher one")
    print(pd.crosstab(low.scene, low[[f"llm_higher_{k}" for k in OTHERS]].sum(axis=1)).to_string())
    print("\nlower_llc: F_LR at the placement (as LLC, against F_LR before it) with the LLM's candidate and the rule's")
    before = np.round(low.F_LR_before * MU, LLC_DECIMALS) / MU  # nan at a wall's first placement: left out
    way = {-1: "falls", 0: "stays", 1: "rises"}
    print(pd.crosstab([low.scene, np.sign((low.llc_llm - before).round(6)).map(way).rename("llm")],
                      np.sign((low.llc_rule - before).round(6)).map(way).rename("rule"), margins=True).to_string())
    print("\nlower_llc: what the sentence about the rule's candidate holds against it (after but/though), by tag")
    low = low[low.named]
    print(pd.DataFrame({scene: {"sentences": len(g), **{k: g.against_tags.str.contains(k).sum() for k in TAGS}}
                        for scene, g in low.groupby("scene")}).T.to_string())
    print("\nthe classes by the third of the wall, and the summed llc_diff of lower_llc")
    given_up = d[d.kind == "lower_llc"].groupby(["scene", "third"], observed=True).llc_diff.sum()
    print(pd.crosstab([d.scene, d.third], d.kind).assign(llc_diff=given_up).round(2).to_string())
    print("\nthe last decision of each wall: the LLM's LLC, the rule's in that state, the two final F_LR")
    last = d.sort_values("nth").groupby("case_id").tail(1).merge(walls[["case_id", "F_LR", "F_LR_pick_safe"]])
    print(last.sort_values(["scene", "seed", "round"])[["scene", "seed", "round", "nth", "kind", "llc_llm", "llc_rule",
                                                        "F_LR", "F_LR_pick_safe"]].round(3).to_string(index=False))
    print()
    print(walls.drop(columns="case_id").round(3).to_string(index=False))
    for scene, g in walls.groupby("scene"):
        rho = stats.spearmanr(g.llc_diff_sum, g.F_LR_diff)
        print(f"{scene}: F_LR minus the pick_safe wall's {g.F_LR_diff.mean():.3f} (below in {(g.F_LR_diff < 0).sum()} "
              f"of {len(g)}); Spearman of the summed llc_diff and that difference {rho.statistic:.2f} "
              f"(p {rho.pvalue:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
