"""research/analysis/lookahead_study.py: the branch points it studies, Wang's tilting analysis of a wall
matrix alone against the recorded lambda, the lookahead states read from a recorded trace, and building
the lookahead walls as the cases of a study."""
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'research' / 'tests')]
from research.analysis import lookahead_study as L  # noqa: E402
from research.paths import LEGACY_RUNS  # noqa: E402
from research.storage import case_id  # noqa: E402
from research.storage import studies  # noqa: E402
from research.walls import wall_metrics  # noqa: E402
from test_order_controller import WALL  # noqa: E402


class LookaheadRun(unittest.TestCase):
    """run without records: a made-up recorded wall, and a launcher standing in for run_wall.run."""
    SWITCH = '3=0,1,2,3,4'
    ALTS = [{'name': 'recorded', 'wall': 'par', 'switch': None, 'seed': 7},
            {'name': 'b1', 'wall': 'par', 'switch': SWITCH, 'seed': 7}]

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for module, name, path in ((case_id, 'CASES_DIR', self.root / 'cases'),
                                   (case_id, 'LEGACY_RUNS', self.root / 'runs'),
                                   (studies, 'STUDIES_DIR', self.root / 'studies'),
                                   (L, 'points', lambda: {('par', 3): self.ALTS}), (L, 'same_events', lambda *a: [])):
            patcher = mock.patch.object(module, name, path)
            patcher.start()
            self.addCleanup(patcher.stop)
        trace = self.root / 'runs' / 'par' / 'result_x' / 'trace'
        trace.mkdir(parents=True)
        commits = [json.dumps({'step': step, 'outcome': 'placed'}) for step in (1, 2, 3)]
        (trace / 'commits.jsonl').write_text('\n'.join(commits) + '\n', encoding='utf-8')
        self.calls = []

    def launcher(self, status='STOPPED_AFTER_PLACEMENTS'):
        def launch(argv):
            self.calls.append(argv)
            wall, out = case_id.allocate(argv[0], hostname='GIAAIL-STRIX')
            receipt = {'case_id': wall, 'status': status}
            (out / 'receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
            return receipt
        return launch

    def test_builds_each_alternative_once_with_its_parent(self):
        self.assertTrue(L.run(launch=self.launcher()))
        self.assertEqual(len(self.calls), 2)
        for argv in self.calls:
            self.assertEqual(argv[argv.index('--parent') + 1], 'par')
            self.assertEqual(argv[argv.index('--placements') + 1], str(2 + 1 + L.K + 1))
        self.assertNotIn('--switch', self.calls[0])
        self.assertEqual(self.calls[1][self.calls[1].index('--switch') + 1], self.SWITCH)
        entries = studies.load(L.STUDY)['cases']
        self.assertEqual([(e['point'], e['alt']) for e in entries], [('par@3', 'recorded'), ('par@3', 'b1')])
        self.assertEqual(studies.load(L.STUDY)['kind'], 'lookahead')
        self.calls.clear()
        self.assertTrue(L.run(launch=self.launcher()))                 # resume: nothing left to build
        self.assertEqual(self.calls, [])

    def test_a_wall_of_the_archive_counts_as_built(self):
        (self.root / 'runs' / L.STUDY / 'par-step3-recorded').mkdir(parents=True)
        self.assertTrue(L.run(launch=self.launcher()))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(L.lookahead_wall('par', 3, self.ALTS[0]), f'{L.STUDY}/par-step3-recorded')
        self.assertTrue(case_id.is_case_id(L.lookahead_wall('par', 3, self.ALTS[1])))

    def test_stops_at_a_failed_wall_and_refuses_another_design(self):
        self.assertFalse(L.run(launch=self.launcher('FAILED')))
        self.assertEqual(len(self.calls), 1)
        with mock.patch.object(L, 'DESIGN', {**L.DESIGN, 'k': 9}), self.assertRaises(SystemExit):
            L.run(launch=self.launcher())


@unittest.skipUnless(WALL.exists() and (LEGACY_RUNS / 'branches' / L.PLANS[0]).exists(), 'records absent')
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
