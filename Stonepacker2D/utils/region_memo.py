"""Results about a placed stone or a wall, kept on the object: every pose of a step asks about the same ones."""

import numpy as np
from skimage.measure import regionprops

from .constant import get_ignore_pixel_value


def stone_region(stone):
    """(centroid, area) of the first region of `stone.matrix`, as regionprops gives them; None without a region."""
    if stone.__dict__.get("_region_of") is not stone.matrix:
        properties = regionprops(stone.matrix.astype(np.uint8))
        stone._region = (properties[0].centroid, properties[0].area) if properties else None
        stone._region_of = stone.matrix
    return stone._region


def occupied_bounds(base):
    """(any coordinate of a placed pixel is non-zero, first row, last row, first column) of the placed pixels."""
    if base.__dict__.get("_bounds_of") is not base.matrix:
        rows, cols = np.nonzero(np.where(base.matrix != get_ignore_pixel_value(), base.matrix, 0))
        found = bool(rows.any() or cols.any())
        base._bounds = (found, rows.min(), rows.max(), cols.min()) if found else (False, 0, 0, 0)
        base._bounds_of = base.matrix
    return base._bounds
