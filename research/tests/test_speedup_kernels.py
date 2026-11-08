"""The fast replacements of the wall-building hot spots give the same bits as the slow original.

Run from the repository root: python -B -m unittest research.tests.test_speedup_kernels
"""

import math
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from scipy import ndimage
from skimage.measure import regionprops

from Stonepacker2D.evaluate import local_void, tilting_angle
from Stonepacker2D.entity import transform
from Stonepacker2D.entity.base import Base
from Stonepacker2D.utils import region_memo
from Stonepacker2D.utils.kine_2d import Element, cal_Aglobal
from Stonepacker2D.utils.region_memo import stone_region

RNG = np.random.default_rng(20)


def stone_kernels():
    """Mirrored binary stone images: odd and even in both dimensions, with holes in the bounding box."""
    return [RNG.random((rows, cols)) < 0.7 for rows, cols in ((5, 7), (6, 7), (5, 8), (6, 8))]


class CandidateMetric(unittest.TestCase):
    def test_equals_the_full_convolution_bit_for_bit(self):
        matrix = RNG.random((60, 80)) * 1e4
        for kernel in stone_kernels():
            full = local_void.get_convolution_metric(matrix, kernel)
            rows, cols = np.meshgrid(np.arange(40), np.arange(60), indexing="ij")
            rows, cols = rows.ravel(), cols.ravel()
            pixel_rows, pixel_cols = local_void.stone_pixels(kernel)
            sparse = local_void.candidate_metric(matrix, pixel_rows, pixel_cols, rows, cols)
            self.assertEqual(sparse.tobytes(), full[rows, cols].tobytes())


class ColumnWindowMin(unittest.TestCase):
    def footprint_min(self, matrix, footprint, cval):
        origin = [-((footprint.shape[0] - 1) // 2), -((footprint.shape[1] - 1) // 2)]
        return ndimage.minimum_filter(matrix, footprint=footprint, mode="constant", cval=cval, origin=origin)

    def test_equals_the_filter_with_a_one_column_footprint(self):
        matrix = RNG.random((40, 50)) * 30
        for shape in ((5, 7), (6, 7), (5, 8), (6, 8)):
            even_rows, even_cols = 1 - shape[0] % 2, 1 - shape[1] % 2
            start = math.ceil(shape[0] / 2)
            cases = (
                (0, -even_rows, 1, 0, -even_cols, np.inf),
                (0, -even_rows, 1, -1, shape[1] - 1 - even_cols, np.inf),
                (slice(start, None), start - even_rows, shape[0] - start, 0, -even_cols, 0.0),
                (slice(start, None), start - even_rows, shape[0] - start, -1, shape[1] - 1 - even_cols, 0.0),
            )
            for rows, first_row, length, column, shift, cval in cases:
                footprint = np.zeros(shape, dtype=bool)
                footprint[rows, column] = True
                expected = self.footprint_min(matrix, footprint, cval)
                got = local_void.column_window_min(matrix, first_row, length, shift, cval, (0, 40, 0, 50))
                self.assertTrue(np.array_equal(got, expected))
                box = local_void.column_window_min(matrix, first_row, length, shift, cval, (3, 38, 11, 49))
                self.assertTrue(np.array_equal(box, expected[3:38, 11:49]))


class StoneRegion(unittest.TestCase):
    def test_equals_regionprops_and_is_none_without_a_region(self):
        matrix = np.zeros((20, 30), dtype=np.float64)
        matrix[3:9, 5:12] = 1
        properties = regionprops(matrix.astype(np.uint8))[0]
        self.assertEqual(stone_region(SimpleNamespace(matrix=matrix)), (properties.centroid, properties.area))
        self.assertIsNone(stone_region(SimpleNamespace(matrix=np.zeros((20, 30), dtype=np.uint8))))

    def test_a_new_image_of_the_stone_is_not_answered_from_the_old_one(self):
        first, second = np.zeros((20, 30)), np.zeros((20, 30))
        first[3:9, 5:12], second[10:12, 1:4] = 1, 1
        stone = SimpleNamespace(matrix=first)
        stone_region(stone)
        stone.matrix = second
        self.assertEqual(stone_region(stone)[1], 6)


class PrefillRegions(unittest.TestCase):
    def wall_of(self, images):
        wall = sum((label + 1) * image for label, image in enumerate(images))
        return SimpleNamespace(matrix=wall, placed_bricks=[SimpleNamespace(matrix=image) for image in images])

    def test_equals_regionprops_of_each_stone(self):
        first, second, empty = np.zeros((20, 30)), np.zeros((20, 30)), np.zeros((20, 30))
        first[3:9, 5:12] = 1
        second[11:15, 1:3] = 1
        second[12, 3] = 1
        base = self.wall_of([first, second, empty])
        region_memo.prefill_regions(base)
        for stone in base.placed_bricks[:2]:
            properties = regionprops(stone.matrix.astype(np.uint8))[0]
            self.assertEqual(stone._region, (properties.centroid, properties.area))
            self.assertIs(stone._region_of, stone.matrix)
        self.assertIsNone(base.placed_bricks[2]._region)

    def test_the_second_call_for_the_same_wall_does_nothing(self):
        image = np.zeros((20, 30))
        image[3:9, 5:12] = 1
        base = self.wall_of([image])
        region_memo.prefill_regions(base)
        base.placed_bricks[0]._region = "kept"
        region_memo.prefill_regions(base)
        self.assertEqual(base.placed_bricks[0]._region, "kept")


class OccupiedBounds(unittest.TestCase):
    def bounds_of(self, matrix):
        return region_memo.occupied_bounds(SimpleNamespace(matrix=matrix))

    def test_equals_the_coordinates_of_the_placed_pixels(self):
        matrix = np.zeros((20, 30))
        matrix[3:9, 5:12] = 2
        occupied = np.argwhere(matrix)
        self.assertEqual(
            self.bounds_of(matrix), (True, occupied[:, 0].min(), occupied[:, 0].max(), occupied[:, 1].min())
        )

    def test_a_wall_without_placed_pixels_is_not_found(self):
        self.assertFalse(self.bounds_of(np.zeros((20, 30)))[0])

    def test_a_single_pixel_at_the_origin_is_not_found_as_any_of_its_coordinates_says(self):
        matrix = np.zeros((20, 30))
        matrix[0, 0] = 1
        self.assertFalse(np.argwhere(matrix).any())
        self.assertFalse(self.bounds_of(matrix)[0])


def notched_stone(wall_shape, row, col):
    """A 7 x 9 stone with a missing corner and a hole in one row, at (row, col) of an image of wall_shape."""
    matrix = np.zeros(wall_shape)
    matrix[row : row + 7, col : col + 9] = 1
    matrix[row, col] = 0
    matrix[row + 3, col : col + 9] = 0
    return SimpleNamespace(matrix=matrix, center=[col + 4.5, row + 3.5], height=7, width=9, id=1, rotate_from_ori=0)


class VoidOfAPlacedStone(unittest.TestCase):
    def setUp(self):
        self.wall = np.zeros((40, 60))
        self.wall[0:10, :] = 1
        self.wall[10:16, 3:14] = 2
        self.wall[10:13, 30:44] = 3
        self.wall[:, [0, -1]] = 255
        modes = (
            mock.patch.object(local_void, "get_placement_optimization", return_value="image_convolve"),
            mock.patch.object(local_void, "get_record_detail", return_value={"RECORD_PLACEMENT_VALUE": False}),
        )
        for mode in modes:
            mode.start()
            self.addCleanup(mode.stop)

    def test_the_void_metrics_equal_the_ones_over_the_whole_image(self):
        base = SimpleNamespace(matrix=self.wall)
        checked = 0
        for row in range(10, 30, 3):
            for col in range(15, 48, 4):
                stone = notched_stone(self.wall.shape, row, col)
                if np.any(self.wall * stone.matrix):
                    continue
                _, void, void_center = local_void.evaluate(base, stone)
                self.assertEqual(
                    np.float64(local_void.get_void_to_left(base, stone)).tobytes(), np.float64(-void).tobytes()
                )
                self.assertEqual(
                    np.float64(local_void.center_void(base, stone)).tobytes(), np.float64(void_center).tobytes()
                )
                checked += 1
        self.assertGreater(checked, 20)

    def test_a_stone_overlapping_the_wall_or_without_pixels_has_the_value_of_evaluate(self):
        base = SimpleNamespace(matrix=self.wall)
        overlapping = notched_stone(self.wall.shape, 5, 20)
        empty = SimpleNamespace(**{**vars(overlapping), "matrix": np.zeros(self.wall.shape)})
        for stone in (overlapping, empty):
            self.assertEqual(local_void.center_void(base, stone), local_void.evaluate(base, stone)[2])


class NeedsNoRotation(unittest.TestCase):
    def built(self, *rectangles):
        """A wall of rectangles (first row, end row, first col, end col), the last one being the stone asked about."""
        stones = []
        wall = np.zeros((40, 60))
        for label, (top, bottom, left, right) in enumerate(rectangles, start=1):
            matrix = np.zeros(wall.shape)
            matrix[top:bottom, left:right] = 1
            wall += label * matrix
            stones.append(SimpleNamespace(matrix=matrix))
        return SimpleNamespace(matrix=wall, placed_bricks=stones)

    def test_a_stone_centred_on_the_stone_below_is_left_as_it_is(self):
        wall = self.built((0, 10, 5, 45), (10, 18, 10, 30))
        self.assertTrue(transform._needs_no_rotation(1, wall))
        self.assertTrue(transform.rotate_brick(1, wall))

    def test_a_stone_hanging_over_the_edge_of_the_stone_below_goes_through_the_rotation(self):
        self.assertFalse(transform._needs_no_rotation(1, self.built((0, 10, 5, 20), (10, 18, 15, 45))))

    def test_a_stone_on_the_floor_goes_through_the_rotation(self):
        self.assertFalse(transform._needs_no_rotation(0, self.built((0, 10, 5, 20))))


class ForPlacement(unittest.TestCase):
    def test_adding_to_the_copy_leaves_the_lists_of_the_wall_as_they_were(self):
        base = Base.__new__(Base)
        base.matrix = np.zeros((5, 5))
        names = (
            "centers",
            "placed_bricks",
            "rock_centers",
            "placed_rocks",
            "rock_tops",
            "rock_right_bottoms",
            "rock_left_tops",
            "rock_right_tops",
        )
        for name in names:
            setattr(base, name, ["old"])
        future = base.for_placement()
        for name in names:
            getattr(future, name).append("new")
            self.assertEqual(getattr(base, name), ["old"])


def blocks_image():
    """Three stones on a ground, some in contact on their sides and with a gap between others."""
    image = np.zeros((40, 60), dtype=np.uint8)
    image[0:6, :] = 1
    image[6:20, 4:20] = 2
    image[6:18, 20:34] = 3
    image[6:14, 40:50] = 4
    image[18:30, 21:33] = 5
    return image


def contact_summary(contps):
    return [(key, c.coor, c.cand, c.anta, c.orientation, c.cont_type.name, c.cont_type.mu) for key, c in contps.items()]


class ContactsOfImage(unittest.TestCase):
    def test_equals_the_objects_built_for_every_pixel(self):
        image = blocks_image()
        elems = {
            key: Element(key, [0, 0], 1, [], type="ground" if key == 1 else "stone")
            for key in np.unique(image)
            if key != 0
        }
        elems = {int(key): element for key, element in elems.items()}
        slow = tilting_angle.clear_contps(elems, tilting_angle.pixel_to_mesh(image))
        fast = tilting_angle.contps_of_image(elems, image)
        self.assertEqual(contact_summary(fast), contact_summary(slow))
        self.assertGreater(len(fast), 0)

    def assert_same_with_and_without_the_wall(self, wall, image):
        elems = {
            int(key): Element(int(key), [0, 0], 1, [], type="ground" if key == 1 else "stone")
            for key in np.unique(image)
            if key != 0
        }
        slow = tilting_angle.clear_contps(elems, tilting_angle.pixel_to_mesh(image))
        self.assertEqual(contact_summary(tilting_angle.contps_of_image(elems, image, wall)), contact_summary(slow))
        for key, element in elems.items():
            element.center = [key * 1.5, key * 0.25]
        contacts, matrix = tilting_angle.contact_matrix(elems, image, wall)
        self.assertEqual(matrix.tobytes(), cal_Aglobal(elems, slow).tobytes())
        self.assertEqual(matrix.shape, cal_Aglobal(elems, slow).shape)
        self.assertEqual(
            [contact.cont_type.mu for contact in contacts.values()], [c.cont_type.mu for c in slow.values()]
        )

    def test_a_stone_added_to_a_wall_equals_the_whole_image(self):
        image = blocks_image()
        wall = image.copy()
        wall[wall == 5] = 0
        self.assert_same_with_and_without_the_wall(wall, image)

    def test_a_stone_touching_the_edge_of_the_image_equals_the_whole_image(self):
        image = blocks_image()
        image[20:40, 50:60] = 6
        wall = image.copy()
        wall[wall == 6] = 0
        self.assert_same_with_and_without_the_wall(wall, image)

    def test_an_unchanged_wall_equals_the_whole_image(self):
        self.assert_same_with_and_without_the_wall(blocks_image(), blocks_image())

    def test_a_changed_placed_stone_falls_back_to_the_whole_image(self):
        image = blocks_image()
        wall = image.copy()
        wall[6:10, 4:10] = 3
        self.assert_same_with_and_without_the_wall(wall, image)

    def test_a_new_value_already_in_the_wall_falls_back_to_the_whole_image(self):
        image = blocks_image()
        image[30:34, 21:33] = 4
        wall = image.copy()
        wall[30:34, 21:33] = 0
        self.assert_same_with_and_without_the_wall(wall, image)

    def test_a_pixel_of_an_unknown_element_is_refused(self):
        elems = {1: Element(1, [0, 0], 1, [], type="ground")}
        with self.assertRaises(KeyError):
            tilting_angle.contps_of_image(elems, blocks_image())


if __name__ == "__main__":
    unittest.main()
