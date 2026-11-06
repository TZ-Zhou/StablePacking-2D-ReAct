"""Hierarchical filtering of candidate poses (pure numpy, no I/O except write_trace).

Feature columns produced by place_one_stone:
0 course_height (NHR), 1 global_density (GDS), 2 interlocking (IWR), 3 kinematic (LLC),
4 local_void (LFR), 5 local_void_center (unused by selection).

policy="code_original" reproduces the author's cascade in build_wall.py (tag
baseline-author-original): filter order[0] is applied twice, order[4] is never read and the
fallback/final sums use fixed columns. policy="paper" follows Algorithm 2 of Wang et al.
(EJOR 2024): fixed LLC>0 gate, then each of the five filters once in the given order.
"""
import hashlib
import itertools
import json
import os
from collections import namedtuple

import numpy as np

FEATURE_NAMES = ('course_height', 'global_density', 'interlocking', 'kinematic',
                 'local_void', 'local_void_center')
PAPER_ORDER = (3, 0, 2, 1, 4)          # LLC, NHR, IWR/IS, GDS, LFR
CODE_ORIGINAL_ORDER = (1, 2, 3, 4, 0)  # active filter_order in the author's build_wall.py
ALL_ORDERS = tuple(itertools.permutations(range(5)))

# index: selected row or None; stages: survivor index arrays in the order they are written to
# filter.txt (stage 0 = LLC>0 gate); branch: which exit was taken; weighted: raw scaled by weights
Selection = namedtuple('Selection', 'index stages branch weighted')


def select(raw, weights, policy='code_original', order=None, tol=1e-9):
    if order is not None and sorted(order) != [0, 1, 2, 3, 4]:
        raise ValueError(f'order must be a permutation of 0..4, got {order}')
    weights = np.asarray(weights, dtype=float)
    if policy == 'code_original':
        # The author's matmul: a non-finite value in any column spreads to the whole row (0*NaN).
        weighted = np.matmul(raw, np.diag(weights))
        index, stages, branch = _code_original(raw, weighted, order or CODE_ORIGINAL_ORDER)
    elif policy == 'paper':
        # Element-wise, so the unused sixth column cannot contaminate the five filter scores.
        weighted = raw * weights
        index, stages, branch = _paper(raw, weighted, order or PAPER_ORDER, tol)
    else:
        raise ValueError(f'unknown policy {policy}')
    return Selection(index, stages, branch, weighted)


def _code_original(filters_matrix_ori, filters_matrix, filter_order):
    # Statement-for-statement copy of build_wall.py L383-454 at baseline-author-original,
    # with the filter.txt writes moved to the caller. Keep expressions identical (float order).
    stages = []
    result_0 = np.argwhere(filters_matrix_ori[:, 3] > 0).reshape((1, -1))
    if not result_0[0].shape[0] > 0:
        return None, stages, 'no_feasible'
    stages.append(result_0[0])
    filter3 = np.argwhere(filters_matrix[:, filter_order[0]] >= np.average(
        filters_matrix[result_0, filter_order[0]])).reshape((1, -1))
    result_0_1 = np.intersect1d(result_0, filter3)
    filter0 = np.argwhere(filters_matrix[:, filter_order[0]] >= np.average(
        filters_matrix[result_0_1, filter_order[0]])).reshape((1, -1))
    result_1 = np.intersect1d(filter0, result_0_1)
    if not result_1.shape[0] > 0:
        return np.argmax(filters_matrix[:, 3]), stages, 'fallback_1'
    stages.append(result_1)
    filter2 = np.argwhere(filters_matrix[:, filter_order[1]] >= np.average(
        filters_matrix[result_1, filter_order[1]])).reshape((1, -1))
    result_2 = np.intersect1d(result_1, filter2)
    if not result_2.shape[0] > 0:
        return result_1[np.argmax(filters_matrix[result_1, 3]+filters_matrix[result_1, 0])], stages, 'fallback_2'
    stages.append(result_2)
    filter1 = np.argwhere(filters_matrix[:, filter_order[2]] >= np.average(
        filters_matrix[result_2, filter_order[2]])).reshape((1, -1))
    result_3 = np.intersect1d(result_2, filter1)
    if not result_3.shape[0] > 0:
        return result_2[np.argmax(
            filters_matrix[result_2, 3]+filters_matrix[result_2, 0]+filters_matrix[result_2, 2])], stages, 'fallback_3'
    stages.append(result_3)
    filter4 = np.argwhere(filters_matrix[:, filter_order[3]] >= np.average(
        filters_matrix[result_3, filter_order[3]])).reshape((1, -1))
    result_4 = np.intersect1d(result_3, filter4)
    if not result_4.shape[0] > 0:
        return result_3[np.argmax(
            filters_matrix[result_3, 3]+filters_matrix[result_3, 0]+filters_matrix[result_3, 2]+filters_matrix[result_3, 1])], stages, 'fallback_4'
    stages.append(result_4)
    return result_4[np.argmax(
        filters_matrix[result_4, 3]+filters_matrix[result_4, 0]+filters_matrix[result_4, 2]+filters_matrix[result_4, 1]+filters_matrix[result_4, 4])], stages, 'full'


def _paper(raw, weighted, order, tol):
    # Algorithm 2: Pi_1 = {LLC > 0}; Pi_{k+1} = {x in Pi_k : f_k(x) >= mean f_k(Pi_k)}.
    # Rows with a non-finite score in any of the five filters are infeasible.
    # tol keeps values within float error of the mean (ties of rounded LLC); tol=0 is strict.
    # If a stage empties, pick argmax of the weighted sum of the filters applied so far
    # (including the failing one) over the previous set (Eqs. 15-21); else Eq. 22.
    cols = list(order)
    feasible = (raw[:, 3] > 0) & np.isfinite(raw[:, cols]).all(axis=1)
    survivors = np.flatnonzero(feasible)
    if survivors.size == 0:
        return None, [], 'no_feasible'
    stages = [survivors]
    for k, col in enumerate(cols):
        values = weighted[survivors, col]
        mean = values.mean()
        kept = survivors[values >= mean - tol * max(1.0, abs(mean))]
        if kept.size == 0:
            score = weighted[np.ix_(survivors, cols[:k + 1])].sum(axis=1)
            return survivors[np.argmax(score)], stages, f'fallback_{k + 1}'
        survivors = kept
        stages.append(survivors)
    score = weighted[np.ix_(survivors, cols)].sum(axis=1)
    return survivors[np.argmax(score)], stages, 'full'


def to_world(side, x, rotation, width, world_width):
    """Candidate poses from the evaluation frame to the wall (world) frame.

    World frame: columns of base.matrix from the left, rows from the ground; (x, y) is the
    Stone.transformed() translation of the pose's bounding box; rotation r means r/90
    applications of Stone.rotated_by_90 to the vendor stone. Right-side candidates are
    evaluated on np.fliplr(wall) with mirrored stones; build_wall maps them back with
    x_world = W - x - width (as the author's code does) and the mirror reverses the turn
    direction, so rotation_world = (360 - r) % 360 (90 <-> 270). y is unchanged. Infeasible
    candidates keep x = -1. decisions.txt writes x_world but the evaluation-frame label r.
    """
    x, rotation = np.asarray(x), np.asarray(rotation)
    if side == 'left':
        return x, rotation
    return np.where(x < 0, x, world_width - x - np.asarray(width)), (360 - rotation) % 360


def footprint(matrix, side):
    """Shape fingerprint of a candidate's pixels in the world frame (position excluded).

    Equal fingerprints at equal (x_world, y) occupy exactly the same pixels, e.g. a
    rectangle at rotation 0 and 180, or two identical bricks with different ids.
    """
    pixels = np.asarray(matrix) != 0
    if not pixels.any():
        return ''
    rows, cols = np.nonzero(pixels)
    crop = pixels[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    if side == 'right':
        crop = np.fliplr(crop)
    return hashlib.sha1(np.packbits(crop).tobytes() + str(crop.shape).encode()).hexdigest()[:16]


def tie_count(raw, weights, selection, policy='code_original', order=None, tol=1e-9):
    """How many candidates share the maximum of the score the final argmax was taken over.

    >1 means the pick was decided by candidate index order (np.argmax takes the first).
    Rebuilds the argmax's candidate set and score from selection.branch; 0 if none.
    """
    return len(tied_candidates(raw, weights, selection, policy, order, tol))


def tied_candidates(raw, weights, selection, policy='code_original', order=None, tol=1e-9):
    """Row indices sharing the final argmax maximum (see tie_count)."""
    w, branch = selection.weighted, selection.branch
    if branch == 'no_feasible':
        return np.array([], dtype=int)
    if policy == 'code_original':
        cols = [3, 0, 2, 1, 4]
        if branch == 'fallback_1':
            pool, used = np.arange(len(w)), cols[:1]
        else:
            k = 5 if branch == 'full' else int(branch[-1])
            pool, used = selection.stages[k - 1 if branch != 'full' else 4], cols[:k]
    else:
        cols = list(order or PAPER_ORDER)
        k = 5 if branch == 'full' else int(branch[-1])
        pool, used = selection.stages[-1], cols[:k]
    score = w[np.ix_(pool, used)].sum(axis=1)
    best = score.max()
    return np.asarray(pool)[score >= best - tol * max(1.0, abs(best))]


def filter_log(step, ids, selection):
    """filter.txt lines exactly as the author's build_wall.py writes them."""
    lines = [f"{step};result{k};" + "".join(f"{ids[i]};" for i in survivors) + "\n"
             for k, survivors in enumerate(selection.stages)]
    if selection.index is not None:
        lines.append(f"{step};select;{ids[selection.index]}\n")
    return "".join(lines)


_events = itertools.count()


def next_event():
    return next(_events)


def write_trace(directory, meta, **arrays):
    """One decision event per .npz; written via rename so a killed run leaves no partial file."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"e{meta['event']:05d}.npz")
    tmp = path[:-4] + '.tmp.npz'
    np.savez_compressed(tmp, meta=json.dumps(meta), **arrays)
    os.replace(tmp, path)


def load_trace(path):
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files if key != 'meta'}
        return json.loads(str(data['meta'])), arrays
