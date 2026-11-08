"""Per-step and per-wall tables of every traced TH_T80 wall in the legacy archive (offline).

Usage: python -B -m research.analysis.myopia_tables <output dir>

Reads research/runs/legacy-index.csv and, for each TH_T80 entry that is not a copy, the last result directory that
holds a trace; research/runs is read only and no wall is built. Writes three CSV files to <output dir>:
- steps.csv: one row per sampling attempt with a trace event: candidate and feasible counts, what the 120 filter
  orders could have selected, the wall state before the decision and, when a candidate was selected, its pose and
  its score against the feasible set (columns c_/mean_/max_/pct_ per filter).
- walls.csv: one row per wall: index fields, wall_metrics values and the formal quality metrics of
  wall_quality(..., 'corrected') for PROCESS_PASS walls (F_SF, F_AH, F_AV, F_LR, n_left_out; err if one failed).
- unique.csv: columns wall,result: the PROCESS_PASS walls with a quality value, one per (seed, placement sequence);
  the archive holds the same wall several times under different names, so statistics must use this list.
Heights are in simulation-grid pixels above the ground. A wall that cannot be read is reported on stderr and skipped.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from research.analysis.surface_fit import GROUND_ROWS, IGNORE_VALUE, skyline
from research.paths import LEGACY_RUNS
from research.walls import wall_metrics as wm
from Stonepacker2D.stack.selection import ALL_ORDERS, PAPER_ORDER, load_trace, select

SCENE = "TH_T80"
FINISHED_STATUS = "PROCESS_PASS"
PLACED_OUTCOME = "placed"
QUALITY_CANVAS = "corrected"
SELECTION_POLICY = "paper"
DEFAULT_TOLERANCE = 1e-9

LLC_COLUMN = 3  # column of features_raw
FILTER_COLUMNS = list(PAPER_ORDER)
JUMP_PX = 8  # a height step between neighbouring columns larger than this counts as a jump
LLC_LEVEL_DECIMALS = 6
PERCENTILE_MARGIN = 1e-12
LLC_TIE_MARGIN = 1e-9
ERROR_CHARS = 200

INDEX_COLUMNS = ("seed", "policy", "order", "switch", "controller", "llm", "status", "placements_limit")
METRIC_COLUMNS = (
    "n_input",
    "n_placed",
    "placed_ratio",
    "stone_filling",
    "wall_density",
    "wall_height",
    "height_ratio",
    "lambda_left",
    "lambda_right",
    "lambda_min",
    "attempts",
    "n_no_placement",
    "n_stabilized",
    "termination",
    "n_wedges",
)
QUALITY_COLUMNS = ("F_SF", "F_AH", "F_AV", "F_LR")
POSE_COLUMNS = ["stone", "rot", "x", "y"]


def step_state(wall):
    """Surface statistics of one wall matrix (row 0 at the bottom; rows 0-1 ground, frame IGNORE_VALUE)."""
    heights = skyline(wall)
    inner = wall[:, 1:-1]
    filled = ((inner != 0) & (inner != IGNORE_VALUE))[GROUND_ROWS:].sum(axis=0)
    void = int((heights - filled).clip(0).sum())
    under = int(heights.sum())
    peak = int(heights.max())
    steps = np.abs(np.diff(heights))
    return dict(
        h_max=peak,
        h_min=int(heights.min()),
        h_mean=float(heights.mean()),
        h_std=float(heights.std()),
        h_range=peak - int(heights.min()),
        roughness=float(steps.mean()),
        void_under=void,
        trapped_ratio=void / under if under else 0.0,
        unlevel=float((peak - heights).sum() / (len(heights) * max(peak, 1))),
        n_jumps=int((steps > JUMP_PX).sum()),
    )


def chosen_event(commit):
    """Trace event number of the chosen side of a commit record, or None if the attempt has no event."""
    side = commit["sides"].get(commit.get("chosen_side")) or next(iter(commit["sides"].values()))
    return side.get("event")


def selected_row(arrays):
    """Candidate row the recorded selection took, or None."""
    selected = arrays["selected"]
    return int(selected) if selected.size and int(selected) >= 0 else None


def selection_tolerance(meta):
    """Tolerance the recorded selection used for its mean comparisons."""
    policy = meta["policy"]
    return policy.get("tol", DEFAULT_TOLERANCE) if isinstance(policy, dict) else DEFAULT_TOLERANCE


def order_picks(raw, weights, tolerance):
    """(selections per candidate row over all filter orders, row picked by the paper order or None)."""
    picks, paper_pick = Counter(), None
    for order in ALL_ORDERS:
        index = select(raw, weights, SELECTION_POLICY, order, tolerance).index
        if index is not None:
            picks[int(index)] += 1
        if order == PAPER_ORDER and index is not None:
            paper_pick = int(index)
    return picks, paper_pick


def chosen_columns(raw, arrays, selected, feasible):
    """Pose of the selected candidate and where each of its filter scores sits within the feasible set."""
    row = dict(
        stone=int(arrays["stone_id"][selected]),
        rot=int(arrays["rotation_world"][selected]),
        x=int(arrays["x_world"][selected]),
        y=int(arrays["y"][selected]),
        width=int(arrays["width"][selected]),
        height_px=int(arrays["height"][selected]),
    )
    for column, name in sorted(wm.FILTER_NAMES.items()):
        value, scores = raw[selected, column], raw[feasible, column]
        row[f"c_{name}"] = float(value)
        row[f"mean_{name}"] = float(scores.mean())
        row[f"max_{name}"] = float(scores.max())
        row[f"pct_{name}"] = float((scores <= value + PERCENTILE_MARGIN).mean())
    llc_gap = np.abs(raw[feasible, LLC_COLUMN] - raw[selected, LLC_COLUMN])
    row["llc_tie_share"] = float((llc_gap < LLC_TIE_MARGIN).mean())
    return row


def commit_columns(wall_id, commit):
    """The steps.csv columns taken from a commit record."""
    return dict(
        wall=wall_id,
        event=chosen_event(commit),
        step=commit["step"],
        outcome=commit["outcome"],
        sample_trial=commit.get("sample_trial"),
        termination=commit.get("termination"),
    )


def candidate_counts(arrays, feasible):
    """Sizes of the candidate set, the feasible subset and the stones still to place."""
    return dict(
        n_candidates=int(arrays["features_raw"].shape[0]),
        n_feasible=int(feasible.sum()),
        n_feasible_stones=len(set(arrays["stone_id"][feasible].tolist())),
        n_sampled=len(arrays["sampled_ids"]),
        n_remaining=len(arrays["sampled_ids"]) + len(arrays["inventory_ids"]),
    )


def step_row(wall_id, commit, meta, arrays):
    """One steps.csv row from a commit record and its loaded trace event."""
    raw, stones = arrays["features_raw"], arrays["stone_id"]
    feasible = (raw[:, LLC_COLUMN] > 0) & np.isfinite(raw[:, FILTER_COLUMNS]).all(axis=1)
    selected = selected_row(arrays)
    picks, paper_pick = order_picks(raw, arrays["weights"], selection_tolerance(meta))
    row = dict(
        **commit_columns(wall_id, commit),
        **candidate_counts(arrays, feasible),
        n_reach=len(picks),
        n_orders_diff_from_paper=sum(count for pick, count in picks.items() if pick != paper_pick),
        branch=meta.get("branch"),
        chosen_is_paper_pick=selected == paper_pick,
        **step_state(arrays["wall"]),
        n_reach_stones=len({int(stones[pick]) for pick in picks}),
        n_feasible_llc_levels=len(set(np.round(raw[feasible, LLC_COLUMN], LLC_LEVEL_DECIMALS).tolist())),
    )
    if selected is not None and feasible.any():
        row.update(chosen_columns(raw, arrays, selected, feasible))
    return row


def wall_row(entry, result, n_steps, metrics):
    """One walls.csv row before the quality columns; metrics holds the METRIC_COLUMNS values, or nothing."""
    path = entry["path"]
    return {
        "wall": path,
        "group": path.split("/")[0],
        **{key: entry[key] for key in INDEX_COLUMNS},
        "result": str(result),
        "n_steps": n_steps,
        **metrics,
    }


def placement_signatures(steps):
    """Per wall, the (stone, rot, x, y) tuples of its placements in the order they were made."""
    placed = steps[steps["outcome"] == PLACED_OUTCOME]
    ordered = placed.assign(t=placed.groupby("wall").cumcount() + 1).sort_values("t")
    return {
        wall: tuple(map(tuple, group[POSE_COLUMNS].astype(int).to_numpy().tolist()))
        for wall, group in ordered.groupby("wall")
    }


def unique_walls(walls, steps):
    """Columns wall,result of the finished walls with a quality value, one per (seed, placement sequence).

    Of walls that are duplicates the first by name is kept.
    """
    finished = walls[(walls["status"] == FINISHED_STATUS) & walls["F_LR"].notna()]
    signatures = placement_signatures(steps[steps["wall"].isin(finished["wall"])])
    keys = [(seed, signatures.get(wall, ())) for seed, wall in zip(finished["seed"], finished["wall"])]
    return (
        finished.assign(seed_signature=keys).sort_values("wall").drop_duplicates("seed_signature")[["wall", "result"]]
    )


def traced_result(run):
    """The last result directory of a run that holds a trace, or None."""
    results = [path for path in sorted(run.glob("result*"), key=str) if (path / "trace" / "commits.jsonl").exists()]
    return results[-1] if results else None


def wall_steps(wall_id, result):
    """All step rows of one result directory; only the traces of attempts with an event are read."""
    trace = result / "trace"
    commits = [commit for commit in wm.read_jsonl(trace / "commits.jsonl") if chosen_event(commit) is not None]
    return [step_row(wall_id, commit, *load_trace(trace / f"e{chosen_event(commit):05d}.npz")) for commit in commits]


def warn(wall_id, what, error):
    """Report a wall that could not be read in full."""
    print(f"{wall_id}: {what}: {error}", file=sys.stderr)


def readable_steps(wall_id, result):
    """Step rows of a wall, or None if its trace cannot be read; one broken wall must not stop the table."""
    try:
        return wall_steps(wall_id, result)
    except Exception as error:
        warn(wall_id, "steps", error)
        return None


def readable_metrics(wall_id, run, result):
    """The METRIC_COLUMNS of a wall; empty if its records cannot be read, the wall stays in the table."""
    try:
        metrics = wm.wall_metrics(run, result)
        return {key: metrics[key] for key in METRIC_COLUMNS}
    except Exception as error:
        warn(wall_id, "metrics", error)
        return {}


def extract_wall(entry):
    """(step rows, wall row) of one index entry, or None if it has no readable trace."""
    wall_id = entry["path"]
    run = LEGACY_RUNS / wall_id
    result = traced_result(run)
    if result is None:
        warn(wall_id, "steps", "no trace")
        return None
    steps = readable_steps(wall_id, result)
    if steps is None:
        return None
    return steps, wall_row(entry, result, len(steps), readable_metrics(wall_id, run, result))


def extract_tables():
    """Steps and walls DataFrames of every TH_T80 entry of the archive index that is not a copy."""
    index = pd.read_csv(LEGACY_RUNS / "legacy-index.csv", keep_default_na=False)
    entries = index[(index["scene"] == SCENE) & (index["copy_of"] == "")].to_dict("records")
    extracted = [found for found in map(extract_wall, entries) if found is not None]
    steps = [row for rows, _ in extracted for row in rows]
    return pd.DataFrame(steps), pd.DataFrame([wall for _, wall in extracted])


def quality_row(wall_id, result):
    """The quality columns of one wall, or its error message; rendering can fail on a single wall."""
    try:
        quality = wm.wall_quality(LEGACY_RUNS / wall_id, Path(result), QUALITY_CANVAS)
        left_out = sum(1 for stone in quality["render"]["stones"] if stone.get("left_out"))
        return {"wall": wall_id, **{key: quality[key] for key in QUALITY_COLUMNS}, "n_left_out": left_out}
    except Exception as error:
        return {"wall": wall_id, "err": str(error)[:ERROR_CHARS]}


def quality_table(walls):
    """Quality columns of the finished walls; this is the slow step."""
    finished = walls[walls["status"] == FINISHED_STATUS]
    return pd.DataFrame([quality_row(wall, result) for wall, result in zip(finished["wall"], finished["result"])])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Per-step and per-wall tables of the archived TH_T80 walls.")
    parser.add_argument("out", type=Path, help="output directory")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    steps, walls = extract_tables()
    walls = walls.merge(quality_table(walls), on="wall", how="left")
    unique = unique_walls(walls, steps)
    steps.to_csv(args.out / "steps.csv", index=False)
    walls.to_csv(args.out / "walls.csv", index=False)
    unique.to_csv(args.out / "unique.csv", index=False)
    print(f"{len(walls)} walls, {len(steps)} steps, {len(unique)} unique walls written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
