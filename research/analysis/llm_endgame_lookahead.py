"""The reading of docs/plans/llm-endgame-lookahead.md: the free opponents of the 9 new states, the ceiling check, and
each state's LLM wall with the check_after tool against them.

Usage: python -B -m research.analysis.llm_endgame_lookahead opponents [--out FILE]
       python -B -m research.analysis.llm_endgame_lookahead read [--loopback] [--out FILE]

opponents reads research/datasets/lookahead-arena-s<seed>-TH_T80 and factorial-s<seed>-TH_T80 (batch_tables.py,
--no-quality) and the walls in runs/batch, with endgame_arena's gates: every none wall equals its factorial wall
(G1); every wall PROCESS_PASS with its trace and controller checks PASS (the controller check includes each placed
check_after row against the next selection) and its placements before the endgame equal the factorial wall's (G2).
One row per new state (cascade, random and random_safe medians and stone losses, llc, borda, low_top, low_top_safe,
pick_safe) and the ceiling rows: low_top_safe on the 9 arena states, next to their cascade and step 3 LLM walls.

read adds the LLM walls of the lookahead-walls study (lookahead-walls-loopback with --loopback: only to test this
script), one per state of lookahead_wall.STATES with a finished journal, and the rule, read once: stop first (as in
step 3), then go ((a), (b), (c) of step 3 and (d), (e)), else 'no endgame contribution shown'. Scores as in
endgame_arena: z in the seed's factorial TH_T80 block, F_AH negated, the mean over the four grid metrics. A state
loses stones when fewer than its 18 stones are placed.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.analysis import endgame_arena as ea
from research.analysis import llm_endgame_pick as step3
from research.controller import order_controller as oc
from research.llm_walls import gpt_wall, lookahead_wall, pick_wall
from research.paths import BATCH_RUNS, DATASETS_DIR
from research.storage import case_id as cid
from Stonepacker2D.stack.selection import paper_feasible

STONES = 18  # TH_T80
RULES = ("llc", "borda", "low_top", "low_top_safe", "pick_safe")
NEW = {s: [o for t, o in lookahead_wall.STATES if t == s] for s in (313, 484, 717)}


def seed_walls(seed):
    """The lookahead arena walls of a seed with z and kind, the factorial block with z, and the gate problems."""
    name, factorial = f"lookahead-arena-s{seed}-TH_T80", f"factorial-s{seed}-TH_T80"
    walls = pd.read_csv(DATASETS_DIR / name / "walls.csv", dtype={"filter_policy": str})
    steps = pd.read_csv(DATASETS_DIR / name / "steps.csv")
    block = pd.read_csv(DATASETS_DIR / factorial / "walls.csv", dtype={"filter_policy": str})
    block_steps = pd.read_csv(DATASETS_DIR / factorial / "steps.csv")
    walls[["order", "mode"]] = walls.filter_policy.str.split("+", n=1, expand=True)
    walls["kind"] = walls["mode"].str.partition(":")[0]
    walls = pd.concat([walls, step3.zscores(walls, block)], axis=1)
    block = pd.concat([block, step3.zscores(block, block)], axis=1)
    problems = []
    for w in walls.itertuples():
        cascade = block[block.filter_policy == w.order].iloc[0]
        case = BATCH_RUNS / name / "cases" / w.wall
        ok, verdicts = ea.controller_verdicts(case)
        if w.status != "PROCESS_PASS" or not ok or verdicts != ["PASS"]:
            problems.append(f"G2 {w.wall} ({w.filter_policy}): {w.status}, trace ok {ok}, controller {verdicts}")
        elif ea.prefix(steps, w.wall) != ea.prefix(block_steps, cascade.wall):
            problems.append(f"G2 {w.wall} ({w.filter_policy}): placements before the endgame differ")
        if w.kind == "none":
            found = ea.identical(case, BATCH_RUNS / factorial / "cases" / cascade.wall)
            problems += [f"G1 {w.wall} ({w.filter_policy}): {d}" for d in found[:5]]
    return walls, block, problems


def percentile(sample, value):
    return ((sample < value).sum() + 0.5 * (sample == value).sum()) / len(sample)


def opponent_row(seed, order, walls, block):
    mine = walls[walls.order == order]
    cascade = block[block.filter_policy == order].iloc[0]
    row = {"seed": seed, "order": order, "cascade_wall": cascade.wall, "cascade_mean_z": cascade.mean_z,
           "cascade_n_placed": int(cascade.n_placed)}
    for kind in ("random", "random_safe"):
        k = mine[mine.kind == kind]
        row.update({f"{kind}_n": len(k), f"{kind}_median": k.mean_z.median(),
                    f"{kind}_losing": int((k.n_placed < STONES).sum()),
                    f"{kind}_lost": int((STONES - k.n_placed).sum())})
    for rule in RULES:
        w = mine[mine.kind == rule].iloc[0]
        row.update({f"{rule}_mean_z": w.mean_z, f"{rule}_n_placed": int(w.n_placed)})
    row["cascade_percentile_random_safe"] = percentile(mine[mine.kind == "random_safe"].mean_z.to_numpy(),
                                                       cascade.mean_z)
    return row


def ceiling_rows(seed, walls, block):
    """low_top_safe on the seed's arena states, with their cascade and step 3 LLM walls' stones."""
    journals = gpt_wall.journals(False, pick_wall)
    rows = []
    for order in [o for s, o in pick_wall.STATES if s == seed]:
        w = walls[(walls.order == order) & (walls.kind == "low_top_safe")].iloc[0]
        cascade = block[block.filter_policy == order].iloc[0]
        journal = journals.get(pick_wall.name_of(seed, order)) or {}
        llm = step3.llm_wall(cid.wall_dir(journal["case_id"]))["n_placed"] if journal.get("case_id") else None
        rows.append({"seed": seed, "order": order, "low_top_safe_n_placed": int(w.n_placed),
                     "low_top_safe_mean_z": w.mean_z, "cascade_n_placed": int(cascade.n_placed),
                     "cascade_mean_z": cascade.mean_z, "step3_llm_n_placed": llm})
    return rows


def opponents():
    rows, ceiling, problems, frames = [], [], [], {}
    for seed in NEW:
        walls, block, p = seed_walls(seed)
        problems += p
        frames[seed] = (walls, block)
        rows += [opponent_row(seed, o, walls, block) for o in NEW[seed]]
        ceiling += ceiling_rows(seed, walls, block)
    return pd.DataFrame(rows), pd.DataFrame(ceiling), problems, frames


def decisions(case):
    """Per LLM decision of a lookahead pick wall: tool used, ids checked, the pick unsafe, pick_safe's choice equal."""
    result = next(Path(case).glob("result_*"))
    records = {json.loads(line)["event"]: json.loads(line)
               for line in (result / "trace" / "controller.jsonl").read_text(encoding="utf-8").splitlines()}
    out = []
    for meta, a in oc.trace_events(result):
        r = records[meta["event"]]
        if not r["decision_point"]:
            continue
        checks = [s for s in r.get("react") or [] if s.get("tool") == "check_after" and s.get("status") == "ok"]
        raw, after, sel = np.asarray(a["features_raw"]), np.asarray(a["after"]), int(a["selected"])
        safe = paper_feasible(raw) & ~(after == 0).any(axis=1)
        pool = np.flatnonzero(safe) if safe.any() else np.flatnonzero(paper_feasible(raw))
        out.append({"source": r["source"], "checked": len(checks),
                    "ids_checked": sum(len(s["arguments"]["candidates"].split(",")) for s in checks),
                    "unsafe_pick": bool(safe.any() and not safe[sel]),
                    "as_pick_safe": sel == int(pool[np.argmax(raw[pool, 3])])})
    return out


def state_row(seed, order, journal, walls, block):
    row = opponent_row(seed, order, walls, block)
    case = cid.wall_dir(journal["case_id"])
    llm = step3.llm_wall(case)
    llm = {**llm, **step3.zscores(pd.DataFrame([llm]), block).iloc[0].to_dict()}
    d = pd.DataFrame(decisions(case))
    mine = walls[walls.order == order]
    spend = journal.get("spend") or {}
    return {**row, "case_id": journal["case_id"], "complete": llm["complete"], "llm_mean_z": llm["mean_z"],
            **{f"llm_{c}": llm[c] for c in step3.ZCOLS}, "llm_n_placed": llm["n_placed"],
            "llm_percentile": percentile(mine[mine.kind == "random"].mean_z.to_numpy(), llm["mean_z"]),
            "llm_percentile_random_safe": percentile(mine[mine.kind == "random_safe"].mean_z.to_numpy(), llm["mean_z"]),
            "decisions": llm["decisions"], "llm_decisions": llm["llm_decisions"], "switches": llm["switches"],
            "decisions_checked": int((d.checked > 0).sum()), "ids_checked": int(d.ids_checked.sum()),
            "unsafe_picks": int(d.unsafe_pick.sum()), "as_pick_safe": int(d.as_pick_safe.sum()),
            "tokens": spend.get("tokens"), "cost_usd": spend.get("cost_usd"), "llm_seconds": llm["llm_seconds"]}


def not_below_low_top_safe(t):
    return t.complete & (t.llm_mean_z >= t.low_top_safe_mean_z) & (t.llm_n_placed >= t.low_top_safe_n_placed)


def reading(t):
    """The rule of docs/plans/llm-endgame-lookahead.md on the 9 state rows: step 3's, then (d) and (e)."""
    first = step3.reading(t)
    if first == "stop direct pick":
        return first
    d = (t.llm_n_placed < STONES).sum() <= (t.cascade_n_placed < STONES).sum()
    e = not_below_low_top_safe(t).sum() >= 5
    return "go" if first == "go" and d and e else "no endgame contribution shown"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cmd", choices=("opponents", "read"))
    parser.add_argument("--loopback", action="store_true")
    parser.add_argument("--out", type=Path, help="CSV of the per-state rows")
    args = parser.parse_args(argv)
    table, ceiling, problems, frames = opponents()
    pd.set_option("display.width", 250)
    if args.cmd == "opponents":
        print(table.round(3).to_string(index=False))
        print("\nceiling (low_top_safe on the arena states):")
        print(ceiling.round(3).to_string(index=False))
        if args.out:
            table.round(6).to_csv(args.out, index=False)
            ceiling.round(6).to_csv(args.out.with_name(args.out.stem + "-ceiling.csv"), index=False)
        print("\n".join(problems) or "G1 and G2: no problem")
        return 1 if problems else 0
    if problems:
        print("\n".join(problems) + "\nnot read: G1 or G2 failed")
        return 1
    journals = gpt_wall.journals(args.loopback, lookahead_wall)
    missing = [lookahead_wall.name_of(*s) for s in lookahead_wall.STATES
               if (journals.get(lookahead_wall.name_of(*s)) or {}).get("state") != "finished"]
    if missing:
        print(f"not read: no finished wall for {missing}")
        return 1
    t = pd.DataFrame([state_row(s, o, journals[lookahead_wall.name_of(s, o)], *frames[s])
                      for s, o in lookahead_wall.STATES])
    if args.out:
        t.round(6).to_csv(args.out, index=False)
    print(t.round(3).to_string(index=False))
    print(f"(a) {(t.complete & (t.llm_mean_z > t.cascade_mean_z) & (t.llm_n_placed >= t.cascade_n_placed)).sum()}/9 "
          f"(b) {(t.complete & (t.llm_mean_z >= t.low_top_mean_z)).sum()}/9 "
          f"(c) {t.switches.sum()}/{t.llm_decisions.sum()} switched "
          f"(d) losing states LLM {(t.llm_n_placed < STONES).sum()}, cascade {(t.cascade_n_placed < STONES).sum()} "
          f"(e) {not_below_low_top_safe(t).sum()}/9; "
          f"below random median {(t.llm_mean_z < t.random_median).sum()}/9, "
          f"fewer stones than cascade {(t.llm_n_placed < t.cascade_n_placed).sum()}/9")
    print("reading:", reading(t))
    return 0


if __name__ == "__main__":
    sys.exit(main())
