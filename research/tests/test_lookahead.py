"""research/analysis/lookahead_study.py: the branch points it studies, Wang's tilting analysis of a wall
matrix alone against the recorded lambda, and the lookahead states read from a recorded trace."""
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'research' / 'tests')]
from research.analysis import lookahead_study as L  # noqa: E402
from research.walls import wall_metrics  # noqa: E402
from test_order_controller import WALL  # noqa: E402


@unittest.skipUnless(WALL.exists() and (L.RUNS / 'branches' / L.PLANS[0]).exists(), 'records absent')
class Lookahead(unittest.TestCase):
    def test_points(self):
        pts = L.points()
        self.assertEqual(len(pts), 6)
        self.assertEqual(sum(map(len, pts.values())), 17)
        for alts in pts.values():
            self.assertEqual((alts[0]['name'], alts[0]['switch']), ('recorded', None))
            self.assertTrue(all(a['switch'] and 'neutral' not in a['name'] for a in alts[1:]))

    def test_tilting_of_the_matrix_alone(self):
        result = next(WALL.glob('result_*'))
        lam = wall_metrics.mechanism_lambdas(result)
        lr = L.kine_lr(np.asarray(pickle.load(open(result / 'base.pkl', 'rb')).matrix))
        self.assertAlmostEqual(lr, min(lam.values()) / 0.58, places=12)

    def test_states_of_a_recorded_wall(self):
        """The recorded wall as its own lookahead: k counts placements, the last one ends at the
        finished wall, nothing is lost before the end."""
        n = sum(c['outcome'] == 'placed' for c in L.commits(WALL))
        s = L.states(WALL, n - 3)
        self.assertEqual([s[k]['end'] for k in L.DEPTHS], [False, False, True])
        self.assertEqual(s[0]['lost'], 0)
        self.assertLess(s[0]['sf'], s[2]['sf'])
        self.assertLessEqual(s[0]['seconds_from_decision'], s[2]['seconds_from_decision'])


if __name__ == '__main__':
    unittest.main()
