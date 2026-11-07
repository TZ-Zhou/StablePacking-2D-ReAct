"""research/preview.py (Observation V3 layer A, P0 draft) on small walls with known answers, the
candidate pixels rebuilt from footprints against a recorded wall, the Greedy-Preview rule, and
that the preview reads nothing but the wall and the candidate's pixels."""
import inspect
import json
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'research'), str(REPO / 'research' / 'tests')]
import preview  # noqa: E402
from Stonepacker2D.stack.selection import footprint  # noqa: E402
from test_controllers import WALL  # noqa: E402


def wall_with(blocks, shape=(40, 60)):
    """A wall matrix: frame columns and top row 255, ground rows 0-1 = 1, then each (r0, r1, c0, c1)
    block as brick 2, 3, ..."""
    w = np.zeros(shape)
    w[:2] = 1
    w[:, 0] = w[:, -1] = w[-1] = 255
    for k, (r0, r1, c0, c1) in enumerate(blocks, 2):
        w[r0:r1, c0:c1] = k
    return w


class Quality(unittest.TestCase):
    def test_density_and_height(self):
        w = wall_with([(2, 12, 1, 30), (2, 7, 30, 59)])
        q = preview.quality(w)
        self.assertEqual(q['height'], 10)
        self.assertAlmostEqual(q['density'], (10 * 29 + 5 * 29) / (58 * 10))
        self.assertEqual(preview.quality(wall_with([]))['density'], None)

    def test_wedges_are_not_rocks(self):
        w = wall_with([(2, 12, 1, 30), (2, 7, 30, 59)])
        self.assertAlmostEqual(preview.quality(w, wedges={3})['density'], 10 * 29 / (58 * 10))
        self.assertEqual(set(np.unique(preview.rock_labels(w, {3}))), {0, 1})

    def test_band_is_below_the_lowest_column(self):
        """Two courses of 4 bricks then one brick on the left: the band stops at the second course."""
        blocks = [(2 + 12 * r, 14 + 12 * r, 1 + 14 * c, 15 + 14 * c) for r in range(2) for c in range(4)] + [(26, 38, 1, 15)]
        w = wall_with(blocks, shape=(60, 60))
        w[:, 57:59] = 1  # (the last columns: ground, so every column has a top)
        labels = preview.rock_labels(w)
        band = labels[:26]
        self.assertEqual(preview.quality(w)['ah_band'], preview.paths_metric(band, False))
        self.assertEqual(preview.quality(w)['ah'], preview.paths_metric(labels, False))

    def test_paste_and_supports(self):
        w = wall_with([(2, 12, 1, 30), (2, 12, 30, 59)])
        px = np.zeros(w.shape, bool)
        px[12:20, 20:40] = True  # across the joint of bricks 2 and 3
        self.assertEqual(preview.supports(w, px), 2)
        self.assertEqual(int(preview.paste(w, px)[15, 25]), 4)
        one = np.zeros(w.shape, bool)
        one[12:20, 2:10] = True
        self.assertEqual(preview.supports(w, one), 1)
        ground = np.zeros(w.shape, bool)
        ground[2:5, 2:4] = True
        self.assertEqual(preview.supports(wall_with([]), ground), 1)  # the ground is one support

    def test_candidate_pixels(self):
        crop = np.array([[1, 0], [1, 1]], bool)
        px = preview.candidate_pixels(crop, 3, 4, (10, 10))
        self.assertEqual(np.argwhere(px).tolist(), [[4, 3], [5, 3], [5, 4]])
        self.assertIsNone(preview.candidate_pixels(crop, 9, 4, (10, 10)))
        self.assertIsNone(preview.candidate_pixels(crop, -1, 4, (10, 10)))

    def test_reads_only_its_arguments(self):
        """No future step, final value or hidden record can reach the preview: its functions take
        the wall, the wedges and the candidate's pixels, and open no file."""
        for f in (preview.quality, preview.preview, preview.paste, preview.supports, preview.candidate_pixels):
            src = inspect.getsource(f)
            self.assertNotIn('open(', src)
            self.assertNotIn('pickle', src)
        self.assertEqual(list(inspect.signature(preview.preview).parameters), ['wall', 'pixels', 'wedges'])


class Greedy(unittest.TestCase):
    def outcome(self, i, llc, **change):
        return {'id': i, 'llc': llc, 'preview_change': {'density': 0.0, 'ah': 0.0, 'av': 0.0, 'ah_band': None, 'av_band': None, **change}}

    def test_dominance_only(self):
        paper = self.outcome(0, 1.0)
        self.assertEqual(preview.greedy_preview([paper, self.outcome(1, 1.0, av=1.0)], 0), 1)
        self.assertEqual(preview.greedy_preview([paper, self.outcome(1, 2.0, av=1.0, ah=0.5)], 0), 0)  # worse F_AH
        self.assertEqual(preview.greedy_preview([paper, self.outcome(1, 1.0)], 0), 0)  # only equal
        best = preview.greedy_preview([paper, self.outcome(1, 1.0, av=1.0), self.outcome(2, 2.0, av=1.0, ah=-1.0)], 0)
        self.assertEqual(best, 2)  # strictly better on more objectives

    def test_undefined_objectives(self):
        paper = self.outcome(0, 1.0, av=None)
        self.assertEqual(preview.greedy_preview([paper, self.outcome(1, 2.0, av=None)], 0), 1)  # both lack av
        self.assertEqual(preview.greedy_preview([paper, self.outcome(1, 2.0, av=3.0)], 0), 0)  # only one has it
        self.assertEqual(preview.greedy_preview([self.outcome(0, 1.0), self.outcome(1, 2.0, ah_band=-1.0, av_band=1.0)], 0, 'band'), 0)


@unittest.skipUnless(WALL.exists(), 'recorded wall absent')
class RecordedWall(unittest.TestCase):
    def test_footprints_rebuild_the_placed_rocks(self):
        """Every rock placed without stabilization is its candidate's pixels at its pose: the
        footprint crop of the trace, at (x_world, y)."""
        import preview_study
        result = next(WALL.glob('result_*'))
        base = pickle.load(open(result / 'base.pkl', 'rb'))
        lib = preview_study.shape_library([WALL])
        n = 0
        for c in map(json.loads, open(result / 'trace' / 'commits.jsonl', encoding='utf-8')):
            if c['outcome'] != 'placed' or c.get('placed_rot_angle'):
                continue
            a = dict(np.load(result / 'trace' / f"e{c['sides'][c['chosen_side']]['event']:05d}.npz"))
            i = int(a['selected'])
            px = preview.candidate_pixels(lib[str(a['footprint'][i])], int(a['x_world'][i]), int(a['y'][i]), a['wall'].shape)
            rock = next(r for r in base.placed_rocks if r.id == c['stone_id'])
            self.assertTrue(np.array_equal(px, rock.matrix != 0), c['step'])
            self.assertEqual(footprint(px, 'left'), str(a['footprint'][i]))
            n += 1
        self.assertGreater(n, 5)


if __name__ == '__main__':
    unittest.main()
