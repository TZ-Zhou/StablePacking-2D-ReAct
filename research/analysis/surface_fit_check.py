"""Check the surface-fit heuristics of research/analysis/surface_fit.py on recorded walls (offline, read-only).

Usage: python -B -m research.analysis.surface_fit_check <unique.csv>

<unique.csv> has columns wall,result; result is the absolute path of a result directory holding trace/e*.npz. No wall
is built and no file is written. Printed to stdout:
- fidelity, over every recorded placement that added exactly one brick: how often the vertical-drop rest height equals
  the height where the stone really landed, how far below it the others landed, the share of real placements with
  their centre of mass between the outermost contact columns, and the void under a placed stone over its area;
- timing, in ms per candidate at three stages of one wall: each heuristic against a 2-D convolution of all poses.
These are the quantities reported in research/findings.md under "Vertical-drop model of the skyline".
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

from research.analysis.surface_fit import (
    CONTACT_TOLERANCE,
    GROUND_ROWS,
    envelope_h,
    fit_scores,
    fluid_level,
    make_pose,
    rollout,
    rollout_best_fit,
    skyline,
    stone_poses,
)
from Stonepacker2D.stack.selection import load_trace

TIMING_WALL_PREFIX = "campaigns"  # with the seed below, picks the wall the timing runs on
TIMING_WALL_SEED = "seed762733368"
TIMING_STEPS = (1, 7, 13)  # 1-based placements of that wall: 17, 11 and 5 stones left
ENVELOPE_WIDTH = 40  # px; pockets narrower than this count as dead in the timed envelope
QUARTER_TURNS = 4
LOW_PERCENTILE = 0.05
HIGH_PERCENTILE = 0.9
REPEAT_CHEAP = 30  # calls averaged per timing: more for the fast heuristics, fewer for the slow ones
REPEAT_FIT = 10
REPEAT_BEST_FIT = 3
REPEAT_CONVOLVE = 2


def load_events(result):
    """The trace arrays of every event of one result directory, in event order."""
    return [load_trace(path)[1] for path in sorted(Path(result, "trace").glob("e*.npz"))]


def selected_row(event):
    """Candidate row the recorded selection chose in a trace event, or -1 when it chose none."""
    return int(event["selected"]) if event["selected"].size else -1


def crop(mask):
    """A boolean mask cropped to its bounding box, plus the box's lowest row and leftmost column."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1], int(rows[0]), int(cols[0])


def new_label(wall_before, wall_after):
    """The brick label that appears only in the later wall, or None unless exactly one appeared."""
    new = np.setdiff1d(wall_after, wall_before)
    return new[0] if len(new) == 1 else None


def placement_records(events):
    """One (stone id, cropped mask, lowest row, leftmost column, wall before) per event placing exactly one brick."""
    records = []
    for before, after in zip(events, events[1:]):
        chosen = selected_row(before)
        label = new_label(before["wall"], after["wall"])
        if chosen < 0 or label is None:
            continue
        mask, row, col = crop(after["wall"] == label)
        records.append((int(before["stone_id"][chosen]), mask, row, col, before["wall"]))
    return records


def compare_landing(s, pose, x, y):
    """Compare a real landing at inner column x, height y with the vertical drop onto skyline s.

    diff is the real height minus the drop's rest height (px, negative under an overhang), sealed_ratio the void under
    the stone over its area, supported whether the centre of mass lies between the outermost contact columns.
    """
    window = s[x : x + len(pose.bottom)]
    gap = y + pose.bottom - window
    contact = np.flatnonzero(gap <= CONTACT_TOLERANCE)
    return {
        "diff": y - int((window - pose.bottom).max()),
        "sealed_ratio": float(gap.clip(0).sum() / pose.area),
        "supported": bool(contact.size and contact[0] <= pose.center <= contact[-1]),
    }


def fidelity_table(wall_events):
    """(one row per placement, first mask seen of each stone) over an iterable of (wall name, trace events)."""
    rows, masks = [], {}
    for wall, events in wall_events:
        for stone, mask, row, col, before in placement_records(events):
            masks.setdefault(stone, mask)
            rows.append({"wall": wall, **compare_landing(skyline(before), make_pose(mask), col - 1, row - GROUND_ROWS)})
    return pd.DataFrame(rows), masks


def fidelity_summary(table):
    """The fidelity numbers of a fidelity_table: shares in 0-1, differences in px."""
    return {
        "placements": len(table),
        "equal": float((table["diff"] == 0).mean()),
        "lower": float((table["diff"] < 0).mean()),
        "diff_p5": float(table["diff"].quantile(LOW_PERCENTILE)),
        "diff_min": int(table["diff"].min()),
        "supported": float(table["supported"].mean()),
        "void_median": float(table["sealed_ratio"].median()),
        "void_p90": float(table["sealed_ratio"].quantile(HIGH_PERCENTILE)),
    }


def timed(fn, repeat):
    """Mean milliseconds of one call of fn."""
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - start) / repeat * 1000


def remaining_stones(event):
    """Sorted ids of the stones still to place after the placement chosen in a trace event."""
    chosen = int(event["stone_id"][selected_row(event)])
    return sorted((set(event["sampled_ids"].tolist()) | set(event["inventory_ids"].tolist())) - {chosen})


def stage_costs(event, masks, poses_of):
    """Milliseconds per candidate of each heuristic on the wall state of one trace event."""
    stones = remaining_stones(event)
    remaining = [poses_of[stone] for stone in stones]
    s = skyline(event["wall"])
    occupied = (event["wall"] != 0).astype(float)
    # Reversed on both axes so that fftconvolve gives the stone's overlap with the wall at every offset.
    shapes = [np.rot90(masks[stone], k).astype(float)[::-1, ::-1] for stone in stones for k in range(QUARTER_TURNS)]
    area = sum(poses[0].area for poses in remaining)
    return {
        "stones_left": len(remaining),
        "fluid": timed(lambda: fluid_level(s, area), REPEAT_CHEAP),
        "envelope": timed(lambda: envelope_h(s, 0, ENVELOPE_WIDTH), REPEAT_CHEAP),
        "fit": timed(lambda: fit_scores(s, remaining), REPEAT_FIT),
        "rollout": timed(lambda: rollout(s, remaining), REPEAT_FIT),
        "best_fit_rollout": timed(lambda: rollout_best_fit(s, remaining), REPEAT_BEST_FIT),
        "convolve_all_poses": timed(lambda: [fftconvolve(occupied, m, mode="same") for m in shapes], REPEAT_CONVOLVE),
    }


def timing_table(events, masks, steps=TIMING_STEPS):
    """Per-candidate cost of each heuristic at the given 1-based placements of one wall's trace events."""
    placed = [event for event in events if selected_row(event) >= 0]
    poses_of = {stone: stone_poses(mask) for stone, mask in masks.items()}
    return pd.DataFrame([stage_costs(placed[step - 1], masks, poses_of) for step in steps]).round(3)


def timing_wall(unique):
    """The row of unique.csv whose wall the timing runs on."""
    chosen = unique.wall.str.startswith(TIMING_WALL_PREFIX) & unique.wall.str.contains(TIMING_WALL_SEED, regex=False)
    return unique[chosen].iloc[0]


def print_fidelity(summary):
    """Print the numbers of fidelity_summary."""
    print(f"placements compared: {summary['placements']:,}")
    print(f"real height equals vertical-drop rest: {summary['equal']:.1%}")
    print(f"real height below vertical-drop rest: {summary['lower']:.1%}")
    print(f"real minus rest, 5th percentile: {summary['diff_p5']:.1f} px, minimum: {summary['diff_min']} px")
    print(f"centre of mass between outermost contacts: {summary['supported']:.1%}")
    print(f"void under a placed stone / its area: median {summary['void_median']:.3f}, p90 {summary['void_p90']:.3f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check the surface-fit heuristics on recorded walls.")
    parser.add_argument("unique", type=Path, help="csv with columns wall,result")
    args = parser.parse_args(argv)
    unique = pd.read_csv(args.unique)
    print(f"walls: {len(unique)}")
    table, masks = fidelity_table((row.wall, load_events(row.result)) for row in unique.itertuples())
    print_fidelity(fidelity_summary(table))
    wall = timing_wall(unique)
    print(f"timing wall: {wall.wall} (ms per candidate)")
    print(timing_table(load_events(wall.result), masks).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
