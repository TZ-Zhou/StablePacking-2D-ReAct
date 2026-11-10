"""B3 rough estimate: Paper's filter order against the median and the post-hoc best of the 120 orders (descriptive).

Usage: python -B -m research.analysis.b3_ceiling <dataset folder>... [--out FILE]

Each <dataset folder> is a research/datasets/<name>/ written by batch_tables.py; its walls.csv rows are grouped
into blocks of seed x scene, and every block must hold the 120 filter orders once. Per block and metric (the grid
F_SF, F_AH, F_AV, F_LR and wall_height), in standard deviations of the block's walls, signed so that + is better:
paper_vs_median_sd, best_vs_paper_sd and best_vs_median_sd, plus paper_pct_beaten (share of orders strictly better
than Paper), the best order and how many orders tie it. Walls without a value are left out of that metric (n).
Prints the medians over blocks and the expected best of 120 independent standard normal draws, the size a
post-hoc best reaches by luck alone. No significance test (docs/plans/simulation-to-llm-agent.md B3).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from Stonepacker2D.stack.selection import PAPER_ORDER

BETTER = {"F_SF": 1, "F_AH": -1, "F_AV": 1, "F_LR": 1, "wall_height": -1}  # sign of the better direction
PAPER = "".join(map(str, PAPER_ORDER))
N_ORDERS = 120


def block_rows(walls):
    """One row per metric of one seed x scene block of 120 walls."""
    orders = walls.filter_policy
    if len(walls) != N_ORDERS or orders.nunique() != N_ORDERS or PAPER not in set(orders):
        raise ValueError(f"block needs the {N_ORDERS} orders once each, got {len(walls)} walls")
    paper = walls[orders == PAPER].iloc[0]
    rows = []
    for metric, sign in BETTER.items():
        values = walls[metric].dropna()
        good, sd = sign * values, values.std()
        best = walls.loc[good.idxmax()]
        rows.append(dict(
            scene=paper.wall_type, seed=int(paper.seed), metric=metric, n=len(values),
            distinct_walls=walls.unique_id.nunique(), paper=paper[metric], median=values.median(), best=best[metric],
            best_order=best.filter_policy, n_best=int((good == good.max()).sum()), sd=sd,
            paper_vs_median_sd=sign * (paper[metric] - values.median()) / sd,
            best_vs_paper_sd=sign * (best[metric] - paper[metric]) / sd,
            best_vs_median_sd=sign * (best[metric] - values.median()) / sd,
            paper_pct_beaten=float((good > sign * paper[metric]).mean())))
    return rows


def ceiling_table(folders):
    """block_rows of every seed x scene block in the walls.csv of the given dataset folders."""
    walls = pd.concat([pd.read_csv(Path(f) / "walls.csv", dtype={"filter_policy": str}) for f in folders])
    return pd.DataFrame([row for _, block in walls.groupby(["wall_type", "seed"]) for row in block_rows(block)])


def expected_best_of(n=N_ORDERS, draws=20000, seed=0):
    """Mean of the largest of n independent standard normal draws."""
    return float(np.random.default_rng(seed).standard_normal((draws, n)).max(axis=1).mean())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="+", type=Path, help="research/datasets/<name> folders")
    parser.add_argument("--out", type=Path, help="CSV of the per-block rows")
    args = parser.parse_args(argv)
    table = ceiling_table(args.folders)
    if args.out:
        table.round(4).to_csv(args.out, index=False)
    sd_columns = ["paper_vs_median_sd", "best_vs_paper_sd", "best_vs_median_sd", "paper_pct_beaten"]
    print(f"{table.groupby(['scene', 'seed']).ngroups} blocks; median over blocks:")
    print(table.groupby("metric")[sd_columns].median().round(2).to_string())
    quality = table[table.metric != "wall_height"]
    print(f"Paper beaten by more than half the orders: {(quality.paper_pct_beaten > 0.5).sum()} of {len(quality)}")
    print(f"most frequent best orders: {quality.best_order.value_counts().head(3).to_dict()}")
    print(f"expected best of {N_ORDERS} standard normal draws: {expected_best_of():.2f} SD")


if __name__ == "__main__":
    sys.exit(main())
