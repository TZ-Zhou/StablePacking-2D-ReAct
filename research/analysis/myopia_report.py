"""Recompute the numbers of "Temporal myopia on TH_T80 (recorded walls)" in research/findings.md (offline).

Usage: python -B -m research.analysis.myopia_report <tables dir>

Input, in <tables dir>:
- steps.csv: one row per recorded step of every wall (scores, feasible counts, skyline state before the placement).
- walls.csv: one row per wall with status, result directory, final metrics and F_SF, F_AH, F_AV, F_LR. Its row order
  matters: the best-of-N bootstrap draws row positions.
- unique.csv: the deduplicated walls (columns wall, result), one per seed and placement sequence.
Also reads, without writing, the recorded traces, base.pkl and time.txt of those walls, the branch plans under
research/runs/branches, research/quality/wang-quality-v1/quality.csv and the stone label image of the scene.

Output: to stdout, one labelled block per subsection of the findings section, in the same order. The subsection
"Vertical-drop model of the skyline" is not covered here. Nothing is written to disk and no wall is built.
Everything except the causal branches is association between recorded walls.
"""

import argparse
import pickle
from itertools import combinations
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import pandas as pd
from scipy import stats
from skimage.measure import label, regionprops

from research.paths import LEGACY_RUNS, QUALITY_DIR, REPO
from Stonepacker2D.stack.selection import ALL_ORDERS, load_trace, select  # noqa: E402

STONES_IMAGE = REPO / "Tests" / "Thangavelu" / "data" / "stones_noise_80_labels.png"
QUALITY_CSV = QUALITY_DIR / "wang-quality-v1" / "quality.csv"
BRANCH_PLANS_DIR = LEGACY_RUNS / "branches"
BRANCH_PLANS = ("paper-hardstone-01", "smoke-001-cf")

FINISHED_STATUS = "PROCESS_PASS"
N_STONES = 18
MIN_STONE_AREA = 50  # labelled regions below this many pixels are noise, not stones
N_LARGEST = 5
HARD_STONE = 13  # the stone left unplaced most often

SCORES = ("LLC", "NHR", "IWR", "GDS", "LFR")
IMMEDIATE = SCORES + ("sum",)
FINALS = ("F_LR", "F_AV", "F_AH", "wall_height", "n_placed")
TIE_FINALS = FINALS[:4]  # n_placed is too coarse for the tied-LLC agreement
BETTER = {"F_LR": 1, "F_AV": 1, "F_AH": -1, "wall_height": -1, "n_placed": 1}  # sign of the better direction
POSE_KEYS = ["stone", "rot", "x", "y"]
TIE_TOL = 1e-9  # same tolerance as the paper-policy selection

PAPER_ARM = "paper_3-0-2-1-4"
CODE_ORIGINAL_ARM = "code_original"
VARIANCE_METRICS = ("F_SF", "F_AH", "F_AV", "F_LR", "wall_height", "n_placed")

LLC_COLUMN = 3
N_FILTERS = 5
SELECTION_POLICY = "paper"
GROUND_ROWS = 2  # rows 0-1 of a wall matrix are ground; the outer two columns are the frame
ROTATIONS = 4  # poses evaluated per stone

DIVERGENCE_EDGES = [0, 3, 6, 12, 18]
DIVERGENCE_LABELS = ["t 1-3", "t 4-6", "t 7-12", "t 13-18"]
REACH_EDGES = [0, 6, 12, 18]
REACH_LABELS = ["t 1-6", "t 7-12", "t 13-18"]
EARLY_POINT_LAST_STEP = 6
LATE_POINT_FIRST_STEP = 13

BUDGET_STEP = 13
ENDGAME_STEPS = range(13, 19)
MID_WALL_STEPS = range(3, 13)
LLC_REPORT_STEPS = (2, 17, 18)

SURFACE_STEPS = (3, 6, 9, 12, 15)
SURFACE_RELATIONS = (
    ("peak_minus_mean", "wall_height"),
    ("peak_minus_mean", "F_AV"),
    ("peak_minus_mean", "F_AH"),
    ("peak_minus_mean", "F_LR"),
    ("c_GDS", "wall_height"),
    ("c_GDS", "F_AV"),
    ("c_LLC", "F_LR"),
    ("c_IWR", "F_LR"),
)
SURFACE_SCORES = ("GDS", "LLC", "IWR")

ENUMERATION_TAILS = (3, 4, 5, 6)
TIME_COLUMNS = 6
TIME_KINEMATICS_COLUMN = 5
TIME_FIRST_POSE_COLUMN = 2  # placement, typology evaluation, stabilization, kinematics evaluation

BOOTSTRAP_SEED = 0
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SIZES = (1, 3, 5, 10, 20)  # all are drawn: the sizes share one random stream
BOOTSTRAP_REPORTED = (1, 5, 10)


class Study(NamedTuple):
    finished: pd.DataFrame  # finished walls with final metrics, in walls.csv order
    unique: pd.DataFrame  # the deduplicated subset, in walls.csv order
    placed: pd.DataFrame  # placements of the finished walls with step t (1-based) and stone area
    placed_unique: pd.DataFrame  # placements of the deduplicated walls
    areas: pd.Series  # stone area in pixels by stone id


# ----- loading ------------------------------------------------------------------------------------------------------


def stone_areas(image_path):
    """Pixel area of every stone of the label image, by stone id (regionprops label)."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    regions = regionprops(label(image.astype(np.uint8)))
    return pd.Series({r.label: r.area for r in regions if r.area >= MIN_STONE_AREA}, name="area").rename_axis("stone")


def placements(steps, areas):
    """Placed steps with their 1-based placement index t per wall and the stone area."""
    placed = steps[steps.outcome == "placed"].copy()
    placed["t"] = placed.groupby("wall").cumcount() + 1
    placed["area"] = placed.stone.map(areas)
    return placed


def load_study(tables_dir):
    """Read the three tables of `tables_dir` into the frames the statistics use."""
    tables_dir = Path(tables_dir)
    walls = pd.read_csv(tables_dir / "walls.csv")
    unique_ids = pd.read_csv(tables_dir / "unique.csv").wall
    areas = stone_areas(STONES_IMAGE)
    placed = placements(pd.read_csv(tables_dir / "steps.csv"), areas)
    finished = walls[(walls.status == FINISHED_STATUS) & walls.F_LR.notna()]
    unique = finished[finished.wall.isin(unique_ids)]
    return Study(
        finished, unique, placed[placed.wall.isin(finished.wall)], placed[placed.wall.isin(unique.wall)], areas
    )


def result_dir(result):
    """A wall's result directory; `result` may be absolute or relative to the repository."""
    return Path(REPO, result)


def trace_path(result, event):
    """The trace file of one recorded event of a wall."""
    return result_dir(result) / "trace" / f"e{int(event):05d}.npz"


# ----- shared statistics -------------------------------------------------------------------------------------------


def spearman(table, x, y):
    """(rho, p) of two columns of `table`, rows with a missing value dropped."""
    rows = table[[x, y]].dropna()
    rho, p = stats.spearmanr(rows[x], rows[y])
    return float(rho), float(p)


def demean_within_seed(table, columns):
    """`table` with each of `columns` replaced by its difference from the mean of that row's seed."""
    return table.assign(**{c: table[c] - table.groupby("seed")[c].transform("mean") for c in columns})


def concordance(points, score, final):
    """(concordant, comparable) over `points`: the higher score ends better on `final`, in its better direction.

    Points tied on the score or on the final value are not comparable.
    """
    d_score, d_final = points[f"d_{score}"], points[f"f_{final}"]
    comparable = (d_score.abs() > TIE_TOL) & (d_final.abs() > TIE_TOL)
    agree = np.sign(d_score[comparable]) * BETTER[final] * np.sign(d_final[comparable]) > 0
    return int(agree.sum()), int(comparable.sum())


# ----- filter order ------------------------------------------------------------------------------------------------


def variance_shares(table, metric):
    """(arm, seed, residual) share of the total sum of squares of `metric` (balanced two-way, no interaction)."""
    data = table[["arm", "seed", metric]].dropna()
    grand = data[metric].mean()
    total = ((data[metric] - grand) ** 2).sum()
    arm = ((data.groupby("arm")[metric].transform("mean") - grand) ** 2).sum() / total
    seed = ((data.groupby("seed")[metric].transform("mean") - grand) ** 2).sum() / total
    return float(arm), float(seed), float(1 - arm - seed)


def paired_comparison(difference, sign):
    """Wins, non-zero seeds, mean difference and Wilcoxon p of one arm's per-seed differences from Paper."""
    better = difference * sign
    changed = better[better != 0]
    p = float(stats.wilcoxon(changed).pvalue) if len(changed) else np.nan
    return {"wins": int((better > 0).sum()), "n": len(changed), "mean": float(difference.mean()), "p": p}


def arm_vs_paper(quality, metric):
    """One row per non-Paper arm: per-seed `metric` of the arm minus Paper (wins count the better direction)."""
    table = quality.pivot(index="seed", columns="arm", values=metric)
    differences = table.drop(columns=PAPER_ARM).sub(table[PAPER_ARM], axis=0)
    rows = {arm: paired_comparison(differences[arm], BETTER[metric]) for arm in differences}
    return pd.DataFrame.from_dict(rows, orient="index")


def pareto_mask(quality):
    """True for walls no other wall dominates: maximise F_SF, F_AV, F_LR and minimise F_AH."""
    objectives = quality[["F_SF", "F_AV", "F_LR"]].assign(neg_AH=-quality.F_AH).to_numpy()
    no_worse = (objectives[None] >= objectives[:, None]).all(axis=2)
    strictly_better = (objectives[None] > objectives[:, None]).any(axis=2)
    return ~(no_worse & strictly_better).any(axis=1)


# ----- immediate scores and divergence -------------------------------------------------------------------------------


def wall_sequences(placed):
    """Placements of every wall in placement order, by wall id."""
    return {wall: group.sort_values("t") for wall, group in placed.groupby("wall")}


def first_divergence(poses_a, poses_b):
    """Index of the first differing pose over the common length of two pose arrays, or None if they never differ."""
    length = min(len(poses_a), len(poses_b))
    differs = (poses_a[:length] != poses_b[:length]).any(axis=1)
    return int(np.argmax(differs)) if differs.any() else None


def same_candidates(result_a, event_a, result_b, event_b):
    """True when two recorded events offered the same candidates with the same scores (the same state, seen twice)."""
    _, trace_a = load_trace(trace_path(result_a, event_a))
    _, trace_b = load_trace(trace_path(result_b, event_b))
    raw_a, raw_b = trace_a["features_raw"], trace_b["features_raw"]
    return raw_a.shape == raw_b.shape and np.allclose(raw_a, raw_b)


def oriented_pair(seed, t, step_a, step_b, final_a, final_b):
    """One divergence as a row, differences a minus b, oriented by the sorted picks so a pair and its mirror agree."""
    pick_a, pick_b = (tuple(int(step[key]) for key in POSE_KEYS) for step in (step_a, step_b))
    sign = -1 if pick_a > pick_b else 1
    row = {"seed": seed, "t": t, "pick_a": min(pick_a, pick_b), "pick_b": max(pick_a, pick_b)}
    row.update({f"d_{s}": sign * (step_a[f"c_{s}"] - step_b[f"c_{s}"]) for s in SCORES})
    row["d_sum"] = sum(row[f"d_{s}"] for s in SCORES)
    row.update({f"f_{m}": sign * (final_a[m] - final_b[m]) for m in FINALS})
    return row


def divergence_row(seed, a, b, sequences, finals):
    """The row of walls a and b at their first differing placement, or None if they never differ or differ in state."""
    if a not in sequences or b not in sequences:
        return None
    path_a, path_b = sequences[a], sequences[b]
    index = first_divergence(path_a[POSE_KEYS].to_numpy(), path_b[POSE_KEYS].to_numpy())
    if index is None:
        return None
    step_a, step_b = path_a.iloc[index], path_b.iloc[index]
    if not same_candidates(finals.loc[a, "result"], step_a.event, finals.loc[b, "result"], step_b.event):
        return None
    return oriented_pair(seed, index + 1, step_a, step_b, finals.loc[a], finals.loc[b])


def divergence_pairs(walls, placed):
    """Same-seed wall pairs that first differ at one placement from an identical candidate set."""
    sequences, finals = wall_sequences(placed), walls.set_index("wall")
    ordered = walls.sort_values("wall")
    pairs = [(seed, a, b) for seed, group in ordered.groupby("seed") for a, b in combinations(group.wall, 2)]
    rows = (divergence_row(seed, a, b, sequences, finals) for seed, a, b in pairs)
    return pd.DataFrame([row for row in rows if row is not None])


def branch_points(pairs):
    """Merge pairs that share a seed, step and unordered pair of picks: scores are the same, finals are averaged."""
    aggregation = {**{f"d_{s}": "first" for s in IMMEDIATE}, **{f"f_{m}": "mean" for m in FINALS}, "t": "size"}
    grouped = pairs.groupby(["seed", "t", "pick_a", "pick_b"]).agg(aggregation)
    return grouped.rename(columns={"t": "n_pairs"}).reset_index()


def tied_agreement_range(points):
    """Lowest and highest agreement rate of the non-LLC scores with the finals over the LLC-tied points."""
    tied = points[points.d_LLC.abs() < TIE_TOL]
    counts = (concordance(tied, score, final) for score in SCORES[1:] for final in TIE_FINALS)
    rates = [c / n for c, n in counts if n]
    return len(tied), min(rates), max(rates)


def divergence_effect_sizes(points):
    """Median absolute final difference of F_LR and height per divergence-step bucket."""
    buckets = pd.cut(points.t, DIVERGENCE_EDGES, labels=DIVERGENCE_LABELS)
    columns = ["f_F_LR", "f_wall_height"]
    return points[columns].abs().groupby(buckets, observed=True).median()


def branch_effect(row, plan, sequences, finals):
    """Final metrics of one recorded branch minus its base wall, or a row noting why there is no difference."""
    branch, base = f"branches/{plan}/{row.name}", row.run
    if branch not in finals.index or base not in finals.index:
        return {"note": "missing"}
    index = first_divergence(sequences[branch][POSE_KEYS].to_numpy(), sequences[base][POSE_KEYS].to_numpy())
    if index is None:
        return {"note": "neutral"}
    return {
        "note": "branch",
        "seed": row.seed,
        "step": index + 1,
        **{m: finals.loc[branch, m] - finals.loc[base, m] for m in FINALS},
    }


def causal_branches(finished, placed):
    """Effect of every branch of the hand-made plans: the branch wall minus the wall it was forked from."""
    sequences, finals = wall_sequences(placed), finished.set_index("wall")
    tables = ((plan, pd.read_csv(BRANCH_PLANS_DIR / plan / "branches.csv")) for plan in BRANCH_PLANS)
    rows = [branch_effect(row, plan, sequences, finals) for plan, table in tables for row in table.itertuples()]
    return pd.DataFrame(rows)


# ----- action reach and stone consumption ----------------------------------------------------------------------------


def action_reach(placed):
    """Per-step feasible candidates, outcomes the 120 orders reach and how often the chosen LLC is the highest."""
    return {
        "feasible_median": placed.n_feasible.median(),
        "feasible_mean": placed.n_feasible.mean(),
        "reach_median": placed.n_reach.median(),
        "reach_mean": placed.n_reach.mean(),
        "one_outcome": (placed.n_reach == 1).mean(),
        "chosen_top_llc": ((placed.c_LLC - placed.max_LLC).abs() < TIE_TOL).mean(),
    }


def step_feasible_stones(raw, weights, stone_ids):
    """(feasible stones, stones picked by at least one of the 120 filter orders) of one trace step."""
    feasible = (raw[:, LLC_COLUMN] > 0) & np.isfinite(raw[:, :N_FILTERS]).all(axis=1)
    picks = (select(raw, weights, SELECTION_POLICY, order, TIE_TOL).index for order in ALL_ORDERS)
    return set(stone_ids[feasible].tolist()), {int(stone_ids[pick]) for pick in picks if pick is not None}


def step_reach_rows(step, result):
    """One row per feasible stone of a recorded step: whether some filter order picks it."""
    _, trace = load_trace(trace_path(result, step.event))
    feasible, reachable = step_feasible_stones(trace["features_raw"], trace["weights"], trace["stone_id"])
    return [{"t": step.t, "stone": stone, "reachable": stone in reachable} for stone in feasible]


def feasible_stone_reach(placed, results):
    """Feasible-stone table of every placement, with the placement phase. Re-runs the selection from the traces."""
    rows = [row for step in placed.itertuples() for row in step_reach_rows(step, results[step.wall])]
    table = pd.DataFrame(rows)
    table["phase"] = pd.cut(table.t, REACH_EDGES, labels=REACH_LABELS)
    return table


def reach_by_phase(reach, stones):
    """Share of feasible rows of `stones` that some order picks, per phase."""
    return reach[reach.stone.isin(stones)].groupby("phase", observed=True).reachable.mean()


def stone_mean_steps(placed, areas):
    """Mean placement step and area of every stone."""
    return placed.groupby("stone").t.mean().rename("mean_t").to_frame().join(areas)


def largest_are_last(table):
    """True when the five largest stones are the five placed last on average."""
    return set(table.area.nlargest(N_LARGEST).index) == set(table.mean_t.nlargest(N_LARGEST).index)


def large_last_tendency(placed):
    """Spearman of placement step and stone area within each wall (positive: large stones go later)."""
    return placed.groupby("wall")[["t", "area"]].apply(lambda g: stats.spearmanr(g.t, g.area)[0]).rename("large_last")


def largest_stones_step(unique, placed, areas):
    """Finals of each wall with the mean step of its five largest stones, demeaned within seed."""
    largest = areas.nlargest(N_LARGEST).index
    mean_step = placed[placed.stone.isin(largest)].groupby("wall").t.mean().rename("large_step")
    table = unique.set_index("wall")[["seed", *FINALS]].join(mean_step)
    return demean_within_seed(table, ["large_step", *FINALS])


# ----- endgame -------------------------------------------------------------------------------------------------------


def final_skyline(shape, rock_masks):
    """Height decomposition of a finished wall: rock masks are boolean matrices over the wall of `shape`."""
    rows, columns = shape
    inner = np.logical_or.reduce(rock_masks)[:, 1:-1]
    column_top = np.where(inner.any(axis=0), rows - np.argmax(inner[::-1], axis=0), 0)
    heights = np.maximum(column_top - GROUND_ROWS, 0)
    width = columns - GROUND_ROWS
    rock_tops = [np.nonzero(mask)[0].max() for mask in rock_masks]
    return {
        "peak": heights.max(),
        "ideal": int(inner.sum()) / width,
        "trap": (heights - inner.sum(axis=0)).clip(0).sum() / width,
        "unlevel": heights.max() - heights.mean(),
        "top_from_end": len(rock_masks) - (int(np.argmax(rock_tops)) + 1),
    }


def wall_skyline(result):
    """`final_skyline` of the base.pkl of one result directory."""
    with open(result_dir(result) / "base.pkl", "rb") as file:
        base = pickle.load(file)
    return final_skyline(base.matrix.shape, [rock.matrix != 0 for rock in base.placed_rocks])


def skyline_decomposition(results):
    """Final skyline table of the given result directories, with the height above the ideal."""
    table = pd.DataFrame([wall_skyline(result) for result in results])
    return table.assign(excess=table.peak - table.ideal)


def height_budget(unique, placed):
    """Complete walls with their skyline before placement BUDGET_STEP and the height gained after it."""
    complete = unique[unique.n_placed == N_STONES].set_index("wall")
    before = placed[placed.t == BUDGET_STEP].set_index("wall")[["h_max", "h_mean"]]
    table = complete.join(before)
    return table.assign(gain=table.wall_height - table.h_max)


def unplaced_counts(unique, placed, stones):
    """How many incomplete walls leave each stone out (only stones left out at least once)."""
    incomplete = unique[unique.n_placed < N_STONES].wall
    times_placed = placed[placed.wall.isin(incomplete)].groupby("stone").size().reindex(stones, fill_value=0)
    left_out = len(incomplete) - times_placed
    return left_out[left_out > 0]


# ----- surface features ----------------------------------------------------------------------------------------------


def surface_table(unique, placed, step):
    """Features after `step` placements (cumulative mean scores, peak minus mean before the next one) with finals."""
    columns = [f"c_{s}" for s in SURFACE_SCORES]
    scores = placed[placed.t <= step].groupby("wall")[columns].mean()
    following = placed[placed.t == step + 1].set_index("wall")
    peak = (following.h_max - following.h_mean).rename("peak_minus_mean")
    table = unique.set_index("wall")[["seed", *FINALS]].join(scores, how="inner").join(peak, how="inner")
    return demean_within_seed(table, ["peak_minus_mean", *columns, *FINALS])


def surface_correlations(unique, placed):
    """Within-seed Spearman of each registered feature-final pair at each step, and n per step."""
    tables = {step: surface_table(unique, placed, step) for step in SURFACE_STEPS}
    rhos = {pair: [spearman(tables[step], *pair)[0] for step in SURFACE_STEPS] for pair in SURFACE_RELATIONS}
    return pd.DataFrame(rhos, index=SURFACE_STEPS).T, {step: len(table) for step, table in tables.items()}


# ----- search cost and restart baseline ------------------------------------------------------------------------------


def enumeration_poses(n_feasible, n_remaining):
    """Pose evaluations to enumerate a tail of a recorded path: every feasible pose opens a node of the next step."""
    nodes = np.r_[1.0, np.cumprod(np.maximum(n_feasible, 1.0))[:-1]]
    return float((nodes * n_remaining * ROTATIONS).sum())


def enumeration_cost(unique, placed):
    """Median and p90 of `enumeration_poses` per tail length over the complete walls."""
    complete = placed[placed.wall.isin(unique[unique.n_placed == N_STONES].wall)]
    records = [
        {
            "k": k,
            "evals": enumeration_poses(
                g[g.t > N_STONES - k].n_feasible.to_numpy(float), g[g.t > N_STONES - k].n_remaining.to_numpy(float)
            ),
        }
        for _, g in complete.sort_values("t").groupby("wall")
        for k in ENUMERATION_TAILS
    ]
    table = pd.DataFrame(records).groupby("k").evals
    return pd.DataFrame({"median": table.quantile(0.5), "p90": table.quantile(0.9)})


def read_time_table(path):
    """The numeric rows of one time.txt, or None when it is absent, unreadable or of an unexpected shape."""
    if not path.exists():
        return None
    try:
        table = np.loadtxt(path, delimiter=";", skiprows=1, ndmin=2)
    except ValueError:
        return None
    return table if len(table) and table.shape[1] == TIME_COLUMNS else None


def timed_pose_seconds(table):
    """Mean recorded seconds of one pose (rows with a kinematics evaluation); the convolution is not timed there."""
    evaluated = table[table[:, TIME_KINEMATICS_COLUMN] > 0]
    return float(evaluated[:, TIME_FIRST_POSE_COLUMN:].sum(axis=1).mean())


def pose_seconds(results):
    """`timed_pose_seconds` over the time.txt of every given result directory."""
    tables = [read_time_table(result_dir(result) / "time.txt") for result in results]
    return timed_pose_seconds(np.vstack([table for table in tables if table is not None]))


def best_of_n_means(complete, n, rng):
    """Mean F_LR and height of the z-sum pick (F_LR up, height down) of n walls drawn with replacement."""
    f_lr, height = complete.F_LR.to_numpy(), complete.wall_height.to_numpy()
    z = (f_lr - f_lr.mean()) / f_lr.std(ddof=1) + (height.mean() - height) / height.std(ddof=1)
    draws = rng.integers(0, len(complete), (BOOTSTRAP_RESAMPLES, n))
    best = draws[np.arange(len(draws)), np.argmax(z[draws], axis=1)]
    return float(f_lr[best].mean()), float(height[best].mean())


def best_of_n_bootstrap(complete):
    """`best_of_n_means` for every size in turn from one random stream; depends on the row order of `complete`."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return {n: best_of_n_means(complete, n, rng) for n in BOOTSTRAP_SIZES}


# ----- report blocks: each returns the lines of one findings subsection -----------------------------------------------


def spread(series):
    """Range, mean and SD of a final metric as text."""
    return f"{series.min():.3f}-{series.max():.3f} (mean {series.mean():.3f}, SD {series.std():.3f})"


def concordance_text(points, score, final):
    """'c/n p=...' of `concordance`, with the two-sided binomial p against 1/2."""
    agree, n = concordance(points, score, final)
    p = stats.binomtest(agree, n, 0.5).pvalue if n else np.nan
    return f"{agree}/{n} p={p:.3f}"


def block_sample(study, points, pooled_pairs):
    """Lines of the Sample subsection."""
    finished, unique = study.finished, study.unique
    unique_llc = concordance(points, "LLC", "F_LR")
    pooled_llc = concordance(pooled_pairs, "LLC", "F_LR")
    return [
        f"finished walls {len(finished)}; unique {len(unique)}; seeds {unique.seed.nunique()};"
        f" complete {int((unique.n_placed == N_STONES).sum())} of {len(unique)}",
        f"LLC -> F_LR concordance: unique branch points {unique_llc[0]}/{unique_llc[1]};"
        f" pooled over all pairs {pooled_llc[0]}/{pooled_llc[1]} = {pooled_llc[0] / pooled_llc[1]:.1%}",
        f"F_LR {spread(unique.F_LR)}",
        f"height px {spread(unique.wall_height)}",
        f"Spearman F_LR-height {spearman(unique, 'F_LR', 'wall_height')[0]:.2f},"
        f" F_AV-height {spearman(unique, 'F_AV', 'wall_height')[0]:.2f}",
    ]


def block_filter_order(quality):
    """Lines of the Filter order subsection."""
    shares = (f"{m} {'/'.join(f'{s:.2f}' for s in variance_shares(quality, m))}" for m in VARIANCE_METRICS)
    f_lr = arm_vs_paper(quality, "F_LR")
    arms = (f"{arm} {r.wins}/{r.n} seeds better, mean {r['mean']:+.3f}, p={r.p:.2f}" for arm, r in f_lr.iterrows())
    code = {m: arm_vs_paper(quality, m).loc[CODE_ORIGINAL_ARM] for m in ("F_AV", "wall_height")}
    return [
        "share of variance, arm/seed/residual: " + ", ".join(shares),
        "F_LR vs Paper: " + "; ".join(arms),
        f"{CODE_ORIGINAL_ARM} vs Paper: F_AV {code['F_AV']['mean']:+.2f} ({code['F_AV'].wins}/{code['F_AV'].n} seeds,"
        f" p={code['F_AV'].p:.2f}), height {code['wall_height']['mean']:+.1f} px"
        f" ({code['wall_height'].wins}/{code['wall_height'].n}, p={code['wall_height'].p:.2f})",
        f"Pareto-optimal walls: {int(pareto_mask(quality).sum())} of {len(quality)}",
    ]


def block_immediate(points, pairs):
    """Lines of the Immediate scores subsection: all score-final concordances and the point counts."""
    rows = [f"{s}: " + "; ".join(f"{m} {concordance_text(points, s, m)}" for m in FINALS) for s in IMMEDIATE]
    tied, low, high = tied_agreement_range(points)
    return [
        f"{len(points)} unique branch points, {len(pairs)} wall pairs, {points.seed.nunique()} seeds",
        *rows,
        f"tests in the table: {len(IMMEDIATE) * len(FINALS)}",
        f"at the {tied} LLC-tied points the other scores agree {low:.1%}-{high:.1%}",
        f"points at t <= {EARLY_POINT_LAST_STEP}: {int((points.t <= EARLY_POINT_LAST_STEP).sum())};"
        f" at t >= {LATE_POINT_FIRST_STEP}: {int((points.t >= LATE_POINT_FIRST_STEP).sum())}",
    ]


def block_action_reach(placed, reach, areas):
    """Lines of the Action reach subsection."""
    r = action_reach(placed)
    largest = set(areas.nlargest(N_LARGEST).index)
    groups = {"five largest": largest, "other stones": set(areas.index) - largest, f"stone {HARD_STONE}": {HARD_STONE}}
    rates = (
        f"{name} " + " / ".join(f"{v:.1%}" for v in reach_by_phase(reach, stones)) for name, stones in groups.items()
    )
    return [
        f"feasible candidates per step: median {r['feasible_median']:.0f} (mean {r['feasible_mean']:.1f})",
        f"outcomes reachable by the 120 orders: median {r['reach_median']:.0f} (mean {r['reach_mean']:.2f});"
        f" one outcome only on {r['one_outcome']:.1%}; chosen has the highest LLC on {r['chosen_top_llc']:.1%}",
        "feasible stone picked by some order (" + " / ".join(REACH_LABELS) + "): " + "; ".join(rates),
        f"overall {reach.reachable.mean():.1%}",
    ]


def block_stone_consumption(unique, placed, areas):
    """Lines of the Stone consumption subsection."""
    mean_steps = stone_mean_steps(placed, areas)
    rho, p = spearman(mean_steps, "area", "mean_t")
    late = large_last_tendency(placed)
    within = largest_stones_step(unique, placed, areas)
    correlation = (
        f"{m} {spearman(within, 'large_step', m)[0]:+.2f} (p={spearman(within, 'large_step', m)[1]:.3f})"
        for m in ("wall_height", "F_AH", "n_placed")
    )
    return [
        f"area vs mean placement step: Spearman {rho:.2f} (p={p:.3f})",
        f"five largest stones are the last five on average: {largest_are_last(mean_steps)}",
        f"walls placing large stones later: {(late > 0).mean():.1%}",
        "within seed, mean step of the five largest vs: " + ", ".join(correlation),
    ]


def block_endgame(unique, placed, areas):
    """Lines of the Endgame subsection."""
    skyline = skyline_decomposition(unique.result)
    budget = height_budget(unique, placed)
    by_step = placed.groupby("t")
    feasible = by_step.n_feasible.mean()
    llc = by_step.c_LLC.mean()
    left_out = unplaced_counts(unique, placed, areas.index)
    return [
        f"excess over ideal {skyline.excess.mean():.1f} px: uneven top {skyline.unlevel.mean():.1f},"
        f" void under the skyline {skyline.trap.mean():.1f}",
        f"peak set by the last placement on {int((skyline.top_from_end == 0).sum())} of {len(skyline)},"
        f" by the second-last on {int((skyline.top_from_end == 1).sum())}",
        f"t = {BUDGET_STEP} ({len(budget)} complete walls): peak {budget.h_max.mean():.1f},"
        f" mean height {budget.h_mean.mean():.1f}; final {budget.wall_height.mean():.1f},"
        f" last six stones add {budget.gain.mean():.1f} px (SD {budget.gain.std():.1f})",
        f"Spearman with final height: peak {spearman(budget, 'h_max', 'wall_height')[0]:.2f},"
        f" later gain {spearman(budget, 'gain', 'wall_height')[0]:.2f}",
        "feasible poses t = 13-18: "
        + ", ".join(f"{feasible[t]:.1f}" for t in ENDGAME_STEPS)
        + f" (t = 3-12: {feasible[list(MID_WALL_STEPS)].min():.1f}-{feasible[list(MID_WALL_STEPS)].max():.1f})",
        "chosen LLC " + ", ".join(f"t = {t}: {llc[t]:.2f}" for t in LLC_REPORT_STEPS),
        f"{int((unique.n_placed < N_STONES).sum())} incomplete walls leave {int(left_out.sum())} stones unplaced: "
        + ", ".join(
            f"stone {int(stone)} x{int(count)}" for stone, count in left_out.sort_values(ascending=False).items()
        ),
    ]


def block_divergence(points, branches):
    """Lines of the Size of one divergence subsection."""
    effects = divergence_effect_sizes(points)
    real = branches[branches.note == "branch"]
    bucket_rows = [
        f"{name}: F_LR {row.f_F_LR:.2f}, height {row.f_wall_height:.1f} px (n={int((points.t.between(lo, hi)).sum())})"
        for (name, row), lo, hi in zip(effects.iterrows(), DIVERGENCE_EDGES[:-1], DIVERGENCE_EDGES[1:])
    ]
    points_seen = real[["seed", "step"]].drop_duplicates()
    return [
        "median |final difference| by divergence step:",
        *bucket_rows,
        f"{len(real)} non-neutral causal branches at {len(points_seen)} points"
        f" ({int((branches.note == 'neutral').sum())} neutral, {int((branches.note == 'missing').sum())} missing):"
        f" F_LR better {int((real.F_LR > 0).sum())}, n_placed +1 in {int((real.n_placed > 0).sum())},"
        f" height worse {int((real.wall_height > 0).sum())} ({int((real.wall_height == 0).sum())} tie),"
        f" F_AV worse {int((real.F_AV < 0).sum())}, F_AH worse {int((real.F_AH > 0).sum())}",
    ]


def block_surface(unique, placed):
    """Lines of the Surface features subsection."""
    rhos, n = surface_correlations(unique, placed)
    rows = [
        f"{feature} vs {final}: " + " / ".join(f"{v:+.2f}" for v in row) for (feature, final), row in rhos.iterrows()
    ]
    return ["T = " + " / ".join(map(str, SURFACE_STEPS)) + f"; n per T {sorted(set(n.values()))}", *rows]


def block_search_cost(unique, placed):
    """Lines of the Search cost and restart baseline subsection."""
    cost = enumeration_cost(unique, placed)
    best = best_of_n_bootstrap(unique[unique.n_placed == N_STONES])
    n_complete = int((unique.n_placed == N_STONES).sum())
    return [
        "pose evaluations, median (p90): "
        + ", ".join(f"k = {k} {r['median']:,.0f} ({r['p90']:,.0f})" for k, r in cost.iterrows()),
        f"timed part of one pose: {pose_seconds(unique.result):.3f} s",
        f"best of N, bootstrap over {n_complete} complete walls, F_LR / height: "
        + ", ".join(f"N = {n} {best[n][0]:.3f} / {best[n][1]:.1f}" for n in BOOTSTRAP_REPORTED),
    ]


def print_block(title, lines):
    """Print one labelled block."""
    print(f"### {title}")
    print("\n".join(lines))
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("tables", type=Path, help="directory with steps.csv, walls.csv and unique.csv")
    args = parser.parse_args(argv)
    study = load_study(args.tables)
    quality = pd.read_csv(QUALITY_CSV)
    pairs = divergence_pairs(study.unique, study.placed_unique)
    points = branch_points(pairs)
    pooled_pairs = divergence_pairs(study.finished, study.placed)
    reach = feasible_stone_reach(study.placed_unique, study.unique.set_index("wall").result)
    print_block("Sample", block_sample(study, points, pooled_pairs))
    print_block("Filter order", block_filter_order(quality))
    print_block("Immediate scores vs final outcome", block_immediate(points, pairs))
    print_block("Action reach", block_action_reach(study.placed_unique, reach, study.areas))
    print_block("Stone consumption", block_stone_consumption(study.unique, study.placed_unique, study.areas))
    print_block("Endgame", block_endgame(study.unique, study.placed_unique, study.areas))
    print_block("Size of one divergence", block_divergence(points, causal_branches(study.finished, study.placed)))
    print_block("Surface features at step T vs final result", block_surface(study.unique, study.placed_unique))
    print_block("Search cost and restart baseline", block_search_cost(study.unique, study.placed_unique))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
