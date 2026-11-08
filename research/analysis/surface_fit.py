"""Candidate heuristics for "do the remaining stones still fit the new top surface?" (pure functions, not wired in).

Design and variants: docs/plans/surface-fit-heuristic.md. Everything works on a skyline: one height per inner column
of the wall, in simulation-grid pixels above the ground, assuming a stone drops vertically. No physics, no RNG, no
file access; every score is comparable only between candidates of the same wall state, and larger is better.
"""

from typing import NamedTuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import grey_closing

GROUND_ROWS = 2  # wall rows 0-1 are ground
IGNORE_VALUE = 255  # the frame: top row and both side columns
CONTACT_TOLERANCE = 2  # px between a stone's bottom and the skyline that still counts as contact
GOOD_VOID_RATIO = 0.25  # sealed void / stone area of a seat worth counting; recorded placements: median 0.17
VOID_WEIGHT = 1.0  # px of height one px of mean sealed void costs when ranking seats


class Pose(NamedTuple):
    """Column profiles of one stone pose, rows counted upward from the box bottom."""

    bottom: np.ndarray
    top: np.ndarray
    center: float
    area: int


def skyline(wall):
    """Height above the ground of the top occupied pixel in each inner column of a wall matrix (row 0 at the bottom)."""
    inner = (wall[:, 1:-1] != 0) & (wall[:, 1:-1] != IGNORE_VALUE)
    return inner.shape[0] - np.argmax(inner[::-1], axis=0) - GROUND_ROWS


def make_pose(mask):
    """Column profiles of a boolean stone mask cropped to its bounding box."""
    columns = np.arange(mask.shape[1])
    center = float((mask.sum(axis=0) * columns).sum() / mask.sum())
    return Pose(mask.argmax(axis=0), mask.shape[0] - mask[::-1].argmax(axis=0), center, int(mask.sum()))


def stone_poses(mask):
    """The four right-angle poses of one stone."""
    return [make_pose(np.rot90(mask, k)) for k in range(4)]


def place(s, x, y, pose):
    """Skyline after a stone box lands at column x, height y; also the void it seals."""
    end = x + len(pose.bottom)
    sealed = int((y + pose.bottom - s[x:end]).clip(0).sum())
    new = s.copy()
    new[x:end] = np.maximum(new[x:end], y + pose.top)
    return new, sealed


def seats(s, pose):
    """All vertical-drop seats of one pose: rest height, sealed void, top, supported flag (one entry per column)."""
    window = sliding_window_view(s, len(pose.bottom))
    rest = (window - pose.bottom).max(axis=1)
    gap = rest[:, None] + pose.bottom - window
    contact = gap <= CONTACT_TOLERANCE
    left = contact.argmax(axis=1)
    right = contact.shape[1] - 1 - contact[:, ::-1].argmax(axis=1)
    return rest, gap.sum(axis=1), rest + pose.top.max(), (left <= pose.center) & (pose.center <= right)


def fluid_level(s, area):
    """Water level reached by pouring area onto skyline s."""
    rising = np.sort(s)
    level = (area + rising.cumsum()) / np.arange(1, len(s) + 1)
    return float(level[np.argmax(level <= np.append(rising[1:], np.inf))])


def surface_terms(s, width):
    """Open area below the current peak, and the part of it in pockets narrower than width."""
    peak = int(s.max())
    padded = np.pad(s, width, constant_values=peak)
    closed = grey_closing(padded, size=width)[width:-width]
    return int((peak - s).sum()), int((closed - s).clip(0).sum())


def envelope_h(s_new, sealed, width, dead_weight=1.0):
    """Larger is better: little open area, few dead pockets, little sealed void."""
    open_area, dead = surface_terms(s_new, width)
    return -(open_area + dead_weight * (dead + sealed))


def stone_fit(s, poses):
    """Good-seat count, least sealed void and lowest top over the supported seats of one stone."""
    count, void, low_top = 0, np.inf, np.inf
    for pose in poses:
        _, sealed, top, supported = seats(s, pose)
        if supported.any():
            count += int((sealed[supported] <= GOOD_VOID_RATIO * pose.area).sum())
            void = min(void, sealed[supported].min())
            low_top = min(low_top, top[supported].min())
    return count, void, low_top


def fit_scores(s, remaining):
    """Three aggregations of stone_fit over the remaining stones (each a list of poses); larger is better."""
    if not remaining:
        return dict(worst_top=-float(s.max()), scarcity=0.0, void=0.0)
    count, void, low_top = np.array([stone_fit(s, poses) for poses in remaining], dtype=float).T
    areas = np.array([poses[0].area for poses in remaining], dtype=float)
    tallest = np.array([max(pose.top.max() for pose in poses) for poses in remaining], dtype=float)
    low_top = np.where(np.isfinite(low_top), low_top, s.max() + tallest)
    void = np.where(np.isfinite(void), void, areas)
    return dict(
        worst_top=-float(low_top.max()),
        scarcity=-float((areas / (1 + count)).sum() / areas.sum()),
        void=-float(void.sum()),
    )


def best_seat(s, poses):
    """Cheapest supported drop of one stone: (cost, column, height, pose), or None."""
    best = None
    for pose in poses:
        rest, sealed, top, supported = seats(s, pose)
        if not supported.any():
            continue
        cost = np.where(supported, top + VOID_WEIGHT * sealed / len(pose.bottom), np.inf)
        k = int(cost.argmin())
        if best is None or cost[k] < best[0]:
            best = (float(cost[k]), k, int(rest[k]), pose)
    return best


def rollout(s, remaining):
    """Drop stones largest first without mechanics: (final height, unplaced stones, sealed void)."""
    unplaced, sealed_total = 0, 0
    for poses in sorted(remaining, key=lambda p: -p[0].area):
        seat = best_seat(s, poses)
        if seat is None:
            unplaced += 1
            continue
        s, sealed = place(s, seat[1], seat[2], seat[3])
        sealed_total += sealed
    return int(s.max()), unplaced, sealed_total


def rollout_best_fit(s, remaining):
    """Like rollout, but each drop takes the stone with the cheapest seat."""
    pool = list(remaining)
    sealed_total = 0
    while pool:
        found = [(seat, i) for i, seat in enumerate(best_seat(s, poses) for poses in pool) if seat is not None]
        if not found:
            break
        seat, i = min(found, key=lambda item: item[0][0])
        s, sealed = place(s, seat[1], seat[2], seat[3])
        sealed_total += sealed
        pool.pop(i)
    return int(s.max()), len(pool), sealed_total
