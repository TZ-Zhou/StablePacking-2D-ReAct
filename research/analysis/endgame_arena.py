"""The TH_T80 endgame arena of docs/plans/endgame-arena.md: gates G1 and G2, one row per state and the reading.

Usage: python -B -m research.analysis.endgame_arena [--seeds 313,484,717] [--out FILE]

Reads research/datasets/endgame-arena-s<seed>-TH_T80 and factorial-s<seed>-TH_T80 (batch_tables.py, --no-quality)
and the walls in runs/batch. An arena wall's filter_policy is "<order>+<mode>[:<seed>]"; its state is the factorial
wall of the same seed and order, whose own ending is the cascade's. G1: every none wall equals its factorial wall
(check_run.compare_runs: decisions, filter, transformation, commits, base.pkl matrices, trace events; and mechanism.csv
bytes). G2: every arena wall PROCESS_PASS with its trace and controller checks PASS, and its placements before the
first step with 7 or fewer stones left equal the factorial wall's. Scores are z in the factorial block (120 walls,
mean and SD with ddof 1), signed so that + is better, and their mean over the four grid metrics (those with a value).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.paths import BATCH_RUNS, DATASETS_DIR
from research.walls import check_run

METRICS = {"F_SF": 1, "F_AH": -1, "F_AV": 1, "F_LR": 1}
RULES = ("llc", "borda", "low_top")
TOLERANCE = 0.1  # SD: an admissible ending is no worse than the cascade's by more on any metric
BETTER = 0.1  # SD: and better on the mean z by at least this much


def prefix(steps, wall):
    """Placements (stone, rot, x, y) before the first step with 7 or fewer stones left."""
    rows = steps[(steps.wall == wall) & (steps.outcome == "placed")]
    first = rows.index[rows.n_remaining <= 7][0]
    rows = rows[rows.index < first]
    return list(zip(rows.stone, rows.rot, rows.x, rows.y))


def identical(arena_wall, factorial_wall):
    differences = check_run.compare_runs(arena_wall, factorial_wall)
    for a, b in zip(check_run.result_dirs(arena_wall), check_run.result_dirs(factorial_wall)):
        if (a / "mechanism.csv").read_bytes() != (b / "mechanism.csv").read_bytes():
            differences.append("mechanism.csv differs")
    return differences


def controller_verdicts(case):
    trace = json.loads((case / "receipt.json").read_text(encoding="utf-8"))["trace_check"]
    return trace["ok"], [c["verdict"] for c in trace.get("controller", [])]


def seed_rows(seed):
    name, factorial = f"endgame-arena-s{seed}-TH_T80", f"factorial-s{seed}-TH_T80"
    walls = pd.read_csv(DATASETS_DIR / name / "walls.csv", dtype={"filter_policy": str})
    steps = pd.read_csv(DATASETS_DIR / name / "steps.csv")
    block = pd.read_csv(DATASETS_DIR / factorial / "walls.csv", dtype={"filter_policy": str})
    block_steps = pd.read_csv(DATASETS_DIR / factorial / "steps.csv")

    def z(frame):
        return pd.concat([(frame[m] - block[m].mean()) / block[m].std() * s for m, s in METRICS.items()], axis=1)

    walls[["order", "mode"]] = walls.filter_policy.str.split("+", n=1, expand=True)
    walls["kind"] = walls["mode"].str.partition(":")[0]
    walls = pd.concat([walls, z(walls).add_prefix("z_")], axis=1)
    block = pd.concat([block, z(block).add_prefix("z_")], axis=1)
    zcols = [f"z_{m}" for m in METRICS]
    walls["mean_z"], block["mean_z"] = walls[zcols].mean(axis=1), block[zcols].mean(axis=1)
    rows, problems = [], []
    for order, arena in walls.groupby("order", sort=False):
        cascade = block[block.filter_policy == order].iloc[0]
        expected = prefix(block_steps, cascade.wall)
        for w in arena.itertuples():
            case = BATCH_RUNS / name / "cases" / w.wall
            ok, verdicts = controller_verdicts(case)
            if w.status != "PROCESS_PASS" or not ok or verdicts != ["PASS"]:
                problems.append(f"G2 {w.wall} ({w.filter_policy}): {w.status}, trace ok {ok}, controller {verdicts}")
            elif prefix(steps, w.wall) != expected:
                problems.append(f"G2 {w.wall} ({w.filter_policy}): placements before the endgame differ")
            if w.kind == "none":
                found = identical(case, BATCH_RUNS / factorial / "cases" / cascade.wall)
                problems += [f"G1 {w.wall} ({w.filter_policy}): {d}" for d in found[:5]]
        endings = arena[arena.kind != "none"]
        random = endings[endings.kind == "random"].mean_z.to_numpy()
        admissible = endings[(endings.n_placed >= cascade.n_placed)
                             & (endings[zcols].sub(cascade[zcols].astype(float)) >= -TOLERANCE).all(axis=1)]
        best = admissible.loc[admissible.mean_z.idxmax()] if len(admissible) else None
        percentile = ((random < cascade.mean_z).sum() + 0.5 * (random == cascade.mean_z).sum()) / len(random)
        rows.append({
            "seed": seed, "order": order, "cascade_wall": cascade.wall, "cascade_mean_z": cascade.mean_z,
            **{f"cascade_{c}": cascade[c] for c in zcols}, "cascade_n_placed": int(cascade.n_placed),
            "random_n": len(random), "cascade_percentile": percentile, "random_median": np.median(random),
            "random_p90": np.quantile(random, 0.9), "random_min": random.min(), "random_max": random.max(),
            "random_n_placed_min": int(endings[endings.kind == "random"].n_placed.min()),
            **{f"{r}_mean_z": endings[endings.kind == r].mean_z.iloc[0] for r in RULES},
            **{f"{r}_n_placed": int(endings[endings.kind == r].n_placed.iloc[0]) for r in RULES},
            "best_admissible": None if best is None else best["mode"],
            "best_admissible_gain": None if best is None else best.mean_z - cascade.mean_z,
            "better_exists": best is not None and best.mean_z - cascade.mean_z >= BETTER,
        })
    return rows, problems


def reading(table):
    no_room = (table.cascade_percentile >= 0.9).sum() >= 7 and (~table.better_exists).sum() >= 7
    no_floor = (table.cascade_percentile <= 0.5).sum() >= 5
    return "no room" if no_room else "no floor" if no_floor else "room"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", default="313,484,717")
    parser.add_argument("--out", type=Path, help="CSV of the per-state rows")
    args = parser.parse_args(argv)
    rows, problems = [], []
    for seed in map(int, args.seeds.split(",")):
        r, p = seed_rows(seed)
        rows += r
        problems += p
    table = pd.DataFrame(rows)
    if args.out:
        table.round(6).to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    print(table.round(3).to_string(index=False))
    print("\n".join(problems) or "G1 and G2: no problem")
    print("reading:", "not read (G1 or G2 failed)" if problems else reading(table))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
