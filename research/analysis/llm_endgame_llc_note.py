"""The reading of docs/plans/llm-endgame-llc-note.md: the free opponents of its 9 states, the control batch (pick-2) and
the note batch (pick-3) against them, and the note against the control.

Usage: python -B -m research.analysis.llm_endgame_llc_note opponents [--out FILE]
       python -B -m research.analysis.llm_endgame_llc_note read [--loopback] [--out FILE]

opponents reads research/datasets/llc-note-arena-s<seed>-TH_T80 (batch_tables.py, --no-quality) with the gates of
llm_endgame_lookahead (G1, G2). read adds the walls of the studies llc-control-walls and llc-note-walls
(-loopback with --loopback: only to test this script), each read once:
- the note batch with the rule of llm_endgame_lookahead ((a)-(e) after step 3's stop) and (f): not below pick_safe in
  mean z, with no fewer stones, in at least 5 states;
- the control batch with the same rule, reported only;
- the note's effect: (g) the note wall above the control wall in mean z (strictly), with no fewer stones, in at least
  7 states; (h) fewer decisions giving up the highest safe LLC in all. (g) and (h): 'the note helps'; (h) only: 'the
  LLM picks differently, the wall not shown better'; else 'no effect of the note shown'.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from research.analysis import llm_endgame_lookahead as la
from research.analysis import llm_endgame_pick as step3
from research.llm_walls import gpt_wall, llc_control_wall, llc_note_wall

ARENA = "llc-note-arena"
STATES = llc_control_wall.STATES
NEW = {s: [o for t, o in STATES if t == s] for s in (313, 484, 717)}
BATCHES = {"control": llc_control_wall, "note": llc_note_wall}


def opponents():
    rows, problems, frames = [], [], {}
    for seed in NEW:
        walls, block, p = la.seed_walls(seed, ARENA)
        problems += p
        frames[seed] = (walls, block)
        rows += [la.opponent_row(seed, o, walls, block) for o in NEW[seed]]
    return pd.DataFrame(rows), problems, frames


def not_below_pick_safe(t):
    return t.complete & (t.llm_mean_z >= t.pick_safe_mean_z) & (t.llm_n_placed >= t.pick_safe_n_placed)


def reading(t):
    """llm_endgame_lookahead's rule, then (f)."""
    first = la.reading(t)
    if first != "go":
        return first
    return "go" if not_below_pick_safe(t).sum() >= 5 else "no endgame contribution shown"


def above_control(note, control):
    n, c = note.reset_index(drop=True), control.reset_index(drop=True)
    return n.complete & (n.llm_mean_z > c.llm_mean_z) & (n.llm_n_placed >= c.llm_n_placed)


def note_effect(note, control):
    """(g) and (h) on the two batches' rows, state by state in the same order."""
    g = above_control(note, control).sum() >= 7
    h = note.gave_up_llc.sum() < control.gave_up_llc.sum()
    return ("the note helps" if g and h else "the LLM picks differently, the wall not shown better" if h
            else "no effect of the note shown")


def summary(t):
    return (f"(a) {(t.complete & (t.llm_mean_z > t.cascade_mean_z) & (t.llm_n_placed >= t.cascade_n_placed)).sum()}/9 "
            f"(b) {(t.complete & (t.llm_mean_z >= t.low_top_mean_z)).sum()}/9 "
            f"(c) {t.switches.sum()}/{t.llm_decisions.sum()} switched "
            f"(d) losing states LLM {(t.llm_n_placed < la.STONES).sum()}, "
            f"cascade {(t.cascade_n_placed < la.STONES).sum()} "
            f"(e) {la.not_below_low_top_safe(t).sum()}/9 (f) {not_below_pick_safe(t).sum()}/9; mean z LLM "
            f"{t.llm_mean_z.mean():.3f}, pick_safe {t.pick_safe_mean_z.mean():.3f}; highest safe LLC given up "
            f"{t.gave_up_llc.sum()}/{t.llm_decisions.sum()} ({t.gave_up_llc_for_nhr.sum()} for a higher NHR); "
            f"another top-LLC tie than pick_safe {t.other_of_tie.sum()}/{t.top_tie_decisions.sum()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cmd", choices=("opponents", "read"))
    parser.add_argument("--loopback", action="store_true")
    parser.add_argument("--out", type=Path, help="CSV of the per-state rows")
    args = parser.parse_args(argv)
    table, problems, frames = opponents()
    pd.set_option("display.width", 250)
    if args.cmd == "opponents":
        print(table.round(3).to_string(index=False))
        if args.out:
            table.round(6).to_csv(args.out, index=False)
        print("\n".join(problems) or "G1 and G2: no problem")
        return 1 if problems else 0
    if problems:
        print("\n".join(problems) + "\nnot read: G1 or G2 failed")
        return 1
    tables = {}
    for batch, x in BATCHES.items():
        journals = gpt_wall.journals(args.loopback, x)
        missing = [x.name_of(*s) for s in STATES if (journals.get(x.name_of(*s)) or {}).get("state") != "finished"]
        if missing:
            print(f"not read: no finished {batch} wall for {missing}")
            return 1
        tables[batch] = pd.DataFrame([la.state_row(s, o, journals[x.name_of(s, o)], *frames[s]) for s, o in STATES])
    note, control = tables["note"], tables["control"]
    if args.out:
        pd.concat([t.assign(batch=b) for b, t in tables.items()]).round(6).to_csv(args.out, index=False)
    for batch, t in tables.items():
        print(f"\n{batch}: {summary(t)}")
        print(f"{batch} reading{'' if batch == 'note' else ' (reported only)'}: {reading(t)}")
    diff = pd.DataFrame({"seed": note.seed, "order": note.order, "note_mean_z": note.llm_mean_z,
                         "control_mean_z": control.llm_mean_z, "mean_z_diff": note.llm_mean_z - control.llm_mean_z,
                         "F_LR_z_diff": note.llm_z_F_LR - control.llm_z_F_LR,
                         "note_n_placed": note.llm_n_placed, "control_n_placed": control.llm_n_placed,
                         "note_gave_up_llc": note.gave_up_llc, "control_gave_up_llc": control.gave_up_llc})
    print("\nnote against control:")
    print(diff.round(3).to_string(index=False))
    print(f"(g) {above_control(note, control).sum()}/9 (h) gave up the highest safe LLC: note "
          f"{note.gave_up_llc.sum()}, control {control.gave_up_llc.sum()}; mean of the differences: mean z "
          f"{diff.mean_z_diff.mean():.3f}, F_LR z {diff.F_LR_z_diff.mean():.3f}")
    print("note effect:", note_effect(note, control))
    print("\nexploratory, not in the rule: LLM - pick_safe, per metric z")
    for batch, t in tables.items():
        gap = pd.DataFrame({c: t[f"llm_{c}"] - t[f"pick_safe_{c}"] for c in step3.ZCOLS}).assign(seed=t.seed,
                                                                                                order=t.order)
        print(batch)
        print(gap.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
