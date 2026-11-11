"""The reading of docs/plans/llm-whole-wall-pick.md: each whole-wall LLM pick wall against Wang's Paper wall of the same
seed and scene.

Usage: python -B -m research.analysis.llm_whole_wall [--loopback] [--confirm] [--out FILE]

The LLM walls are the whole-walls study's (whole-walls-loopback with --loopback: only to test this script), one per
(seed, scene) of research/llm_walls/whole_wall.STATES, each with a finished journal. Wang's wall is the Paper order's
(30214) of the seed's factorial block (research/datasets/factorial-s<seed>-<scene>); scores as in llm_endgame_pick:
F_* on the simulation grid after the last placement, z in that block, F_AH negated, the mean over the four metrics.
The rule, read once: a wall counts if it is complete (every decision the LLM's), its mean z is not below Wang's and it
places no fewer stones; at least NEEDED of the 6 walls: 'pass: the whole wall is not below Wang', else 'catching up
with Wang not shown'. Reported with it: the per-metric z differences, the decisions that differ from Wang's cascade,
the use of check_after and whether the first selection offered the same candidates as Wang's wall (same_first_offer).

--confirm: the reading of docs/plans/llm-whole-wall-confirm.md, the 4 walls of whole_confirm_wall.STATES (study
whole-confirm-walls) on the held-out seeds: at least CONFIRM_NEEDED not below Wang: 'confirmed: the whole wall is not
below Wang on the held-out seeds', else 'not confirmed on the held-out seeds'; with it the count over both batches.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.analysis import llm_endgame_lookahead as la
from research.analysis import llm_endgame_pick as step3
from research.controller import order_controller as oc
from research.llm_walls import gpt_wall, whole_confirm_wall, whole_wall
from research.paths import BATCH_RUNS, DATASETS_DIR
from research.storage import case_id as cid

PAPER = "30214"
NEEDED = 4
CONFIRM_NEEDED = 3  # of the 4 walls of docs/plans/llm-whole-wall-confirm.md


def wang(seed, scene):
    """(the Paper wall's row, the block) of a seed's factorial block, with z scores."""
    block = pd.read_csv(DATASETS_DIR / f"factorial-s{seed}-{scene}" / "walls.csv", dtype={"filter_policy": str})
    block = pd.concat([block, step3.zscores(block, block)], axis=1)
    return block[block.filter_policy == PAPER].iloc[0], block


def first_offer(case):
    """The stones and the scores of a wall's first selection."""
    _, a = next(iter(oc.trace_events(next(Path(case).glob("result_*")))))
    return np.asarray(a["sampled_ids"]), np.asarray(a["features_raw"])


def state_row(seed, scene, journal):
    w, block = wang(seed, scene)
    case = cid.wall_dir(journal["case_id"])
    llm = step3.llm_wall(case)
    llm = {**llm, **step3.zscores(pd.DataFrame([llm]), block).iloc[0].to_dict()}
    d = pd.DataFrame(la.decisions(case))
    mine, his = first_offer(case), first_offer(BATCH_RUNS / f"factorial-s{seed}-{scene}" / "cases" / w.wall)
    spend = journal.get("spend") or {}
    return {"seed": seed, "scene": scene, "case_id": journal["case_id"], "complete": llm["complete"],
            "llm_mean_z": llm["mean_z"], "wang_mean_z": w.mean_z, "mean_z_diff": llm["mean_z"] - w.mean_z,
            **{f"diff_{c}": llm[c] - w[c] for c in step3.ZCOLS},
            "llm_n_placed": llm["n_placed"], "wang_n_placed": int(w.n_placed), "n_input": int(w.n_input),
            "llm_percentile": la.percentile(block.mean_z.to_numpy(), llm["mean_z"]),
            "decisions": llm["decisions"], "llm_decisions": llm["llm_decisions"], "switches": llm["switches"],
            "decisions_checked": int((d.checked > 0).sum()), "ids_checked": int(d.ids_checked.sum()),
            "left_at_zero": int(d.unsafe_pick.sum()),
            "same_first_offer": all(np.array_equal(x, y, equal_nan=True) for x, y in zip(mine, his)),
            "http_requests": spend.get("http_requests"), "tokens": spend.get("tokens"),
            "cost_usd": spend.get("cost_usd"), "llm_seconds": llm["llm_seconds"]}


def not_below_wang(t):
    return t.complete & (t.llm_mean_z >= t.wang_mean_z) & (t.llm_n_placed >= t.wang_n_placed)


def all_four(t):
    """The walls at or above Wang on each of the four metrics (reported, not in the rule)."""
    return (t[[f"diff_{c}" for c in step3.ZCOLS]] >= 0).all(axis=1)


def reading(t, confirm=False):
    """The rule of docs/plans/llm-whole-wall-pick.md on the 6 wall rows; confirm: that of llm-whole-wall-confirm.md
    on the 4."""
    if confirm:
        return ("confirmed: the whole wall is not below Wang on the held-out seeds"
                if not_below_wang(t).sum() >= CONFIRM_NEEDED else "not confirmed on the held-out seeds")
    return ("pass: the whole wall is not below Wang" if not_below_wang(t).sum() >= NEEDED
            else "catching up with Wang not shown")


def table(loopback, x):
    """The wall rows of an experiment (whole_wall or whole_confirm_wall), None unless every wall is finished."""
    journals = gpt_wall.journals(loopback, x)
    names = {s: x.name_of(*s) for s in x.STATES}
    if any((journals.get(n) or {}).get("state") != "finished" for n in names.values()):
        return None
    return pd.DataFrame([state_row(*s, journals[n]) for s, n in names.items()])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--loopback", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--out", type=Path, help="CSV of the per-wall rows")
    args = parser.parse_args(argv)
    x = whole_confirm_wall if args.confirm else whole_wall
    t = table(args.loopback, x)
    if t is None:
        print(f"not read: a wall of {x.STUDIES[args.loopback]} is not finished")
        return 1
    if args.out:
        t.round(6).to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(t.round(3).to_string(index=False))
    print(f"not below Wang {not_below_wang(t).sum()}/{len(t)} (needed {CONFIRM_NEEDED if args.confirm else NEEDED}), "
          f"on all four metrics {all_four(t).sum()}/{len(t)}; mean of the differences: mean z "
          f"{t.mean_z_diff.mean():.3f}, " + ", ".join(f"{c} {t[f'diff_{c}'].mean():.3f}" for c in step3.ZCOLS)
          + f"; {t.switches.sum()}/{t.llm_decisions.sum()} decisions differ from the cascade; check_after at "
          f"{t.decisions_checked.sum()}/{t.decisions.sum()}; first offer as Wang's {t.same_first_offer.sum()}/{len(t)}"
          f"; US${t.cost_usd.sum():.3f}, {t.tokens.sum()} tokens, {t.http_requests.sum()} requests")
    print("reading:", reading(t, args.confirm))
    first = table(args.loopback, whole_wall) if args.confirm else None
    if first is not None:  # described, not in the rule: the development walls were read before
        both = pd.concat([first, t])
        print(f"both batches: not below Wang {not_below_wang(both).sum()}/{len(both)}, on all four metrics "
              f"{all_four(both).sum()}/{len(both)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
