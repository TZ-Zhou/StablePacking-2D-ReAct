"""research/decision_signals.py: the Pareto front on a known matrix, every paper-cascade pick on
the front of a recorded wall, and the hardness order."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'research'), str(REPO / 'research' / 'tests')]
import decision_signals as d  # noqa: E402
from test_controllers import WALL  # noqa: E402


class Signals(unittest.TestCase):
    def test_front(self):
        raw = np.array([[1, 1, 1, 1, 1, 0], [2, 1, 1, 1, 1, 0], [0, 3, 0, 1, 0, 0],
                        [5, 5, 5, 0, 5, 0], [2, 1, 1, 1, 1, 0]], float)  # row 3: LLC 0, infeasible
        self.assertEqual(d.front(raw, np.ones(6)).tolist(), [1, 2, 4])  # 0 dominated; equal rows both kept

    def test_hardness_order(self):
        h = d.Hardness.__new__(d.Hardness)
        h.area = {1: 10, 2: 50, 3: 10, 4: 99}
        h.stat = {1: {'n_sampled': 1, 'n_infeasible': 1, 'poses': 9}, 2: {'n_sampled': 1, 'n_infeasible': 0, 'poses': 2},
                  3: {'n_sampled': 1, 'n_infeasible': 0, 'poses': 2}}
        self.assertEqual(h.rank([4, 3, 2, 1]), [1, 2, 3, 4])  # infeasible first, then poses, area; never sampled last
        h.update({'features_raw': np.array([[0, 0, 0, 1, 0, 0]], float), 'stone_id': np.array([4]), 'sampled_ids': np.array([4, 3])})
        self.assertEqual((h.stat[4]['poses'], h.stat[3]['n_infeasible']), (1, 1))

    @unittest.skipUnless(WALL.exists(), 'recorded wall absent')
    def test_every_order_picks_on_the_front(self):
        n = 0
        for _, a in d.events(WALL):
            if a['features_raw'].size:
                self.assertLessEqual(set(d.classes(a['features_raw'], a['weights'])), set(d.front(a['features_raw'], a['weights']).tolist()))
                n += 1
        self.assertGreater(n, 10)


if __name__ == '__main__':
    unittest.main()
