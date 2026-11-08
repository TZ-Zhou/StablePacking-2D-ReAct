"""Observation version 3, layer A (P0 draft, not wired into order_controller.Observer): a local quality
preview of each candidate placement, and the Greedy-Preview rule that uses it.

Inputs are only what the controller has at the decision: the wall matrix (world frame, row 0 at
the ground; 1 = ground, 255 = frame, k >= 2 = the k-th placed brick), the values of the placed
wedges, and the candidate's own pixels at its pose before stabilization. No later step, no final
metric, no physics.

quality(wall, wedges) of a partial wall, on the simulation grid:
  density   rock px / ((W - 2) * built height): wall_metrics' wall_density of the partial wall
            (the local counterpart of F_SF; F_SF itself divides by the whole wall area);
  ah, av    03's F_AH / F_AV (wall_metrics.joint_image, joint_paths, path_metric) of its rock label
            image as it is; above the skyline these paths run through the air; None while the
            image is too small for 03;
  ah_band, av_band  the same on the rows below the lowest skyline column, the band every column
            has built up to; None until that band is tall enough;
  height    built height in px (rows above the 2 ground rows up to the top rock pixel).
preview(wall, pixels, wedges) of one candidate: {'after': quality with its pixels pasted,
  'n_supports': distinct bricks (the ground counts as one) right under its bottom pixels; 2 or
  more means it spans the joint between them}. The change is after - quality(wall, wedges).
Rocks only: ground, frame and wedges are left out of every label image, as in the author's.
"""
import numpy as np

from research.walls import wall_metrics

GROUND_TOP = 2  # rows 0-1 are ground (unplaced_stones.wall_state)
KEYS = ('density', 'ah', 'av', 'ah_band', 'av_band', 'height')
# Greedy-Preview objectives, fixed before any validation: (field, +1 higher is better / -1 lower);
# ah, av: 03 on the partial wall as it is (band variants: the secondary rule); llc: the
# candidate's LLC score, the local counterpart of F_LR
OBJECTIVES = {'full': (('density', 1), ('ah', -1), ('av', 1), ('llc', 1)),
              'band': (('density', 1), ('ah_band', -1), ('av_band', 1), ('llc', 1))}


def rock_labels(wall, wedges=()):
    wall = np.asarray(wall)
    rock = (wall >= 2) & (wall != 255) & ~np.isin(wall, list(wedges))
    return np.where(rock, wall - 1, 0).astype(int)


def paths_metric(labels, vertical):
    if not labels.any():
        return None
    try:
        img = wall_metrics.joint_image(labels)
    except ValueError:  # nothing left after the erosion
        return None
    return wall_metrics.path_metric(wall_metrics.joint_paths(img, vertical))


def quality(wall, wedges=()):
    wall = np.asarray(wall)
    labels = rock_labels(wall, wedges)
    rows = np.flatnonzero(labels.any(axis=1))
    if not rows.size:
        return dict.fromkeys(KEYS)
    height = int(rows.max()) - GROUND_TOP + 1
    occupied = (wall != 0) & (wall != 255)
    inner = occupied[:, 1:-1]
    sky = np.where(inner.any(axis=0), inner.shape[0] - np.argmax(inner[::-1], axis=0), 0)  # rows up to the top pixel
    band = labels[:int(sky.min())]
    return {'density': float(np.count_nonzero(labels) / ((wall.shape[1] - 2) * height)),
            'ah': paths_metric(labels, False), 'av': paths_metric(labels, True),
            'ah_band': paths_metric(band, False), 'av_band': paths_metric(band, True), 'height': height}


def paste(wall, pixels):
    """The wall with the candidate's pixels as the next brick (on free pixels only)."""
    wall = np.asarray(wall)
    after = wall.copy()
    after[np.asarray(pixels, bool) & (wall == 0)] = wall[(wall != 255)].max() + 1
    return after


def supports(wall, pixels):
    pixels = np.asarray(pixels, bool)
    bottom = pixels & ~np.vstack([np.zeros((1, pixels.shape[1]), bool), pixels[:-1]])  # nothing of it right below
    rows, cols = np.nonzero(bottom)
    under = np.asarray(wall)[np.maximum(rows - 1, 0), cols]
    return len({int(v) for v in under if v not in (0, 255)})


def preview(wall, pixels, wedges=()):
    return {'after': quality(paste(wall, pixels), wedges), 'n_supports': supports(wall, pixels)}


def candidate_pixels(crop, x_world, y, shape):
    """A candidate's pixels in the world frame: its footprint crop with the bottom-left corner at
    (x_world, y), as Wang's trace records the pose; None if it does not fit."""
    h, w = crop.shape
    if x_world < 0 or y < 0 or y + h > shape[0] or x_world + w > shape[1]:
        return None
    out = np.zeros(shape, bool)
    out[y:y + h, x_world:x_world + w] = crop
    return out


def greedy_preview(outcomes, paper_id, rule='full', tol=1e-9):
    """Greedy-Preview, no weights: the outcome that Pareto-dominates Paper's on every objective of
    OBJECTIVES[rule] (each field of outcome['preview_change'] or 'llc'; an objective one of the two
    lacks counts only if both lack it) and is strictly better on at least one; among several,
    the one strictly better on most objectives, then the lowest id. Paper's outcome otherwise."""
    value = lambda o, f: o['llc'] if f == 'llc' else o['preview_change'].get(f)  # noqa: E731
    paper = next(o for o in outcomes if o['id'] == paper_id)
    best, best_n = paper, 0
    for o in sorted(outcomes, key=lambda o: o['id']):
        if o['id'] == paper_id:
            continue
        n_better, ok = 0, True
        for f, sense in OBJECTIVES[rule]:
            a, b = value(o, f), value(paper, f)
            if a is None or b is None:
                ok = a is None and b is None
            elif sense * (a - b) > tol:
                n_better += 1
            elif sense * (a - b) < -tol:
                ok = False
            if not ok:
                break
        if ok and n_better > best_n:
            best, best_n = o, n_better
    return best['id']
