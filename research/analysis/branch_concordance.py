"""C2 and C4 of docs/plans/preregistration.md: do scores of the children at a branch point agree with the final walls?

Usage: python -B -m research.analysis.branch_concordance <dataset folder>... [--family H2b,H3b,H4a]
       [--baselines c_GDS,H0_unlevel] [--out FILE]

Each <dataset folder> is research/datasets/<name>/ of batch_tables.py (walls.csv, steps.csv) for one seed x scene
block; scores named H* are read from the h.csv of placement_h.py in <dataset folder>-h, the others (c_NHR, c_GDS,
c_IWR, c_LLC, c_LFR: the placed candidate's raw scores at the branch point) from steps.csv.

Per block (A3): the prefix tree of the PROCESS_PASS walls' placement sequences (stone, rot, x, y of each placed row
of steps.csv). A branch point is a node with two or more children; a child's score is the one of its placement (h
after it, joined on wall and trace event; c_* at the branch point), and its label is the mean z of the distinct
walls (unique_id) below it, z taken per scene over every wall of the given folders, signed so that + is better (wall
metrics F_SF, F_AH, F_AV, F_LR on the simulation grid; a wall without a value is left out of that metric). Each pair
of children whose scores differ and whose labels differ is comparable, and concordant when the higher score has the
better label. D_b = concordant - discordant of block b; the statistic is sum_b D_b, two-sided block sign-flip test
(100,000 flips, numpy default_rng(20261109), p = (1 + #|flipped| >= |observed|) / (1 + 100,000)). Holm runs over the
family's score x metric tests; baselines get the same columns but no p. passes (C4's rule): Holm p < 0.05, a
positive statistic and a rate above every baseline's rate on that metric. A child whose walls disagree on a score
keeps its first wall's value and is counted in n_score_mismatch.
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = {"F_SF": 1, "F_AH": -1, "F_AV": 1, "F_LR": 1}  # sign of the better direction
FLIPS = 100_000
FLIP_SEED = 20261109


def block_children(walls, steps, h, scores):
    """{node: {child: (score values, unique_ids below)}} of one block, and the number of score mismatches."""
    placed = steps[steps.outcome == "placed"]
    if h is not None:
        placed = placed.merge(h, on=["wall", "event"], how="left", validate="one_to_one")
    unique = walls.set_index("wall").unique_id
    tree, mismatch = {}, 0
    for wall, rows in placed.groupby("wall", sort=False):
        if wall not in unique.index:
            continue
        sequence = list(zip(rows.stone, rows.rot, rows.x, rows.y))
        values = rows[scores].to_numpy(dtype=float)
        for t, child in enumerate(sequence):
            node = tree.setdefault(tuple(sequence[:t]), {})
            if child in node:
                kept, leaves = node[child]
                mismatch += not np.array_equal(kept, values[t], equal_nan=True)
                leaves.add(unique[wall])
            else:
                node[child] = (values[t], {unique[wall]})
    return {node: kids for node, kids in tree.items() if len(kids) > 1}, mismatch


def block_counts(branches, z, scores):
    """Comparable and concordant pair counts per score and metric of one block."""
    counts = {(s, m): [0, 0] for s in scores for m in z.columns}
    for kids in branches.values():
        labels = {child: z.loc[sorted(leaves)].mean() for child, (_, leaves) in kids.items()}
        for a, b in combinations(kids, 2):
            for i, s in enumerate(scores):
                sign_score = np.sign(kids[a][0][i] - kids[b][0][i])
                if not abs(sign_score) > 0:  # equal or NaN
                    continue
                for m in z.columns:
                    sign_label = np.sign(labels[a][m] - labels[b][m])
                    if abs(sign_label) > 0:  # not equal and not NaN
                        counts[s, m][0] += 1
                        counts[s, m][1] += int(sign_score == sign_label)
    return counts


def sign_flip_p(d, flips=FLIPS, seed=FLIP_SEED):
    """Two-sided block sign-flip p of sum(d)."""
    signs = np.random.default_rng(seed).choice([-1, 1], size=(flips, len(d)))
    return (1 + int((np.abs(signs @ d) >= abs(d.sum())).sum())) / (1 + flips)


def holm(p):
    """Holm-adjusted p values, in the input order."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]).clip(max=1)
    out = np.empty_like(p)
    out[order] = adjusted
    return out


def concordance(folders, family, baselines):
    """One row per score and metric over the blocks in folders, with the per-block D_b."""
    scores = list(family) + list(baselines)
    h_scores = [s for s in scores if s.startswith("H")]
    blocks = []
    for folder in map(Path, folders):
        walls = pd.read_csv(folder / "walls.csv", dtype={"filter_policy": str})
        walls = walls[walls.status == "PROCESS_PASS"]
        steps = pd.read_csv(folder / "steps.csv", usecols=["wall", "event", "outcome", "stone", "rot", "x", "y"]
                            + [s for s in scores if s.startswith("c_")])
        h = pd.read_csv(Path(f"{folder}-h") / "h.csv", usecols=["wall", "event", *h_scores]) if h_scores else None
        blocks.append((walls, steps, h))
    every = pd.concat([w for w, _, _ in blocks])
    z = pd.concat([(every[m] - every.groupby("wall_type")[m].transform("mean"))
                   / every.groupby("wall_type")[m].transform("std") * sign for m, sign in METRICS.items()], axis=1)
    z.index = every.unique_id.to_numpy()
    z = z[~z.index.duplicated()]
    per_block, rows = [], []
    for walls, steps, h in blocks:
        branches, mismatch = block_children(walls, steps, h, scores)
        counts = block_counts(branches, z, scores)
        per_block.append(dict(scene=walls.wall_type.iloc[0], seed=int(walls.seed.iloc[0]), branch_points=len(branches),
                              n_score_mismatch=mismatch, counts=counts))
    for s in scores:
        for m in METRICS:
            comparable = np.array([b["counts"][s, m][0] for b in per_block])
            concordant = np.array([b["counts"][s, m][1] for b in per_block])
            d = 2 * concordant - comparable
            rows.append(dict(score=s, metric=m, role="family" if s in family else "baseline", blocks=len(per_block),
                             comparable=int(comparable.sum()), concordant=int(concordant.sum()),
                             rate=concordant.sum() / comparable.sum() if comparable.sum() else np.nan,
                             statistic=int(d.sum()), blocks_positive=int((d > 0).sum()),
                             p=sign_flip_p(d) if s in family else np.nan))
    table = pd.DataFrame(rows)
    in_family = table.role == "family"
    table.loc[in_family, "p_holm"] = holm(table.loc[in_family, "p"])
    best_baseline = table[~in_family].groupby("metric").rate.max()
    table.loc[in_family, "passes"] = ((table.p_holm < 0.05) & (table.statistic > 0)
                                      & (table.rate > table.metric.map(best_baseline).fillna(-np.inf)))[in_family]
    return table, pd.DataFrame([{k: v for k, v in b.items() if k != "counts"} for b in per_block])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="+", help="research/datasets/<name> folders, one per seed x scene block")
    parser.add_argument("--family", default="H2b,H3b,H4a", help="scores tested and Holm-corrected together")
    parser.add_argument("--baselines", default="c_GDS,H0_unlevel", help="scores reported without a test")
    parser.add_argument("--out", type=Path, help="CSV of the score x metric rows")
    args = parser.parse_args(argv)
    table, blocks = concordance(args.folders, args.family.split(","), [s for s in args.baselines.split(",") if s])
    if args.out:
        table.round(6).to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    print(blocks.to_string(index=False))
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
