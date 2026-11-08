"""research/walls/campaign_quality.py on a small campaign of copied walls: the output format, FAIL rows for
broken runs and failed evaluations, identical bytes on a rerun, the refusals, and that the run
records are only read."""
import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'research' / 'tests'))
from research.walls import campaign_quality as q  # noqa: E402
from test_order_controller import WALL  # noqa: E402
from test_wall_quality import FIVE, digest  # noqa: E402

ARMS = ['a', 'b', 'broken', 'absent']


@unittest.skipUnless(WALL.exists(), 'recorded wall absent')
class Batch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.runs = cls.root / 'runs'
        camp = cls.runs / 'campaigns' / 't'
        for arm in ARMS[:3]:
            shutil.copytree(WALL, camp / 'case' / arm / 'seed1')
        (next((camp / 'case' / 'broken' / 'seed1').glob('result_*')) / 'transformation.txt').unlink()
        (camp / 'campaign.json').write_text(json.dumps({'arms': [{'name': a} for a in ARMS], 'seeds': [1]}))
        with open(camp / 'manifest.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['arm', 'policy', 'order', 'seed', 'run'])
            for a in ARMS:
                w.writerow([a, 'paper', '3-0-2-1-4', 1, f'campaigns/t/case/{a}/seed1'])
        cls.before = digest(cls.runs)
        cls.failed = q.batch('t', cls.root / 'out', runs=cls.runs, check_sources=False)
        cls.rows = list(csv.DictReader(open(cls.root / 'out' / 'quality.csv', encoding='utf-8')))
        cls.info = json.loads((cls.root / 'out' / 'batch.json').read_text(encoding='utf-8'))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_rows(self):
        sf, ah, av, lam = FIVE[WALL][:4]
        self.assertEqual(list(self.rows[0]), q.COLUMNS)
        self.assertEqual([r['status'] for r in self.rows], ['OK', 'OK', 'FAIL', 'FAIL'])
        for r in self.rows[:2]:  # a wall 02's canvas does not cut: both canvases agree
            self.assertEqual([float(r[k]) for k in ('F_SF', 'F_AH', 'F_AV', 'lambda_min')], [sf, ah, av, lam])
            self.assertEqual([float(r[k + '_author']) for k in ('F_SF', 'F_AH', 'F_AV')], [sf, ah, av])
            self.assertEqual(float(r['F_LR']), lam / 0.58)
            self.assertEqual((r['placed_ratio'], r['n_placed'], r['n_input'], r['restored_px'], r['method']),
                             (repr(17 / 18), '17', '18', '0', 'wang-quality-v1'))
            self.assertNotEqual(r['wall_density'], '')
        self.assertIn('missing transformation.txt', self.rows[2]['problems'])
        self.assertEqual(self.rows[3]['problems'], 'no receipt.json')
        self.assertEqual([r['F_SF'] for r in self.rows[2:]], ['', ''])
        self.assertEqual(self.failed, 2)

    def test_batch_json_and_jsonl(self):
        self.assertEqual((self.info['method'], self.info['n_runs'], self.info['n_ok']), ('wang-quality-v1', 4, 2))
        self.assertEqual([f['arm'] for f in self.info['failed']], ['broken', 'absent'])
        self.assertEqual(self.info['resolution']['rescale'], 0.5)
        self.assertIn('research/walls/wall_metrics.py', self.info['sources_git_blob'])
        self.assertTrue(any(k.startswith('Stonepacker2D/') for k in self.info['sources_git_blob']))
        lines = [json.loads(line) for line in (self.root / 'out' / 'quality.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual([d['status'] for d in lines], ['OK', 'OK', 'FAIL', 'FAIL'])
        self.assertEqual(lines[0]['corrected']['render']['mode'], 'corrected')
        self.assertEqual(lines[0]['author']['render']['mode'], 'author')
        self.assertNotIn(str(self.root), (self.root / 'out' / 'batch.json').read_text(encoding='utf-8'))

    def test_rerun_writes_the_same_bytes_and_records_stay(self):
        q.batch('t', self.root / 'again', runs=self.runs, check_sources=False)
        for name in ('batch.json', 'quality.csv', 'quality.jsonl'):
            self.assertEqual((self.root / 'again' / name).read_bytes(), (self.root / 'out' / name).read_bytes(), name)
        self.assertEqual(digest(self.runs), self.before)

    def test_refusals(self):
        with self.assertRaises(SystemExit):  # written once
            q.batch('t', self.root / 'out', runs=self.runs, check_sources=False)
        with self.assertRaises(SystemExit):  # never inside the records
            q.batch('t', self.runs / 'campaigns' / 't' / 'quality', runs=self.runs, check_sources=False)
        hashes = q.sources()[0]
        with mock.patch.object(q, 'sources', return_value=(hashes, ['research/walls/wall_metrics.py'])), self.assertRaises(SystemExit):
            q.batch('t', self.root / 'dirty', runs=self.runs)
        changed = [(hashes, []), ({**hashes, 'research/walls/wall_metrics.py': '0'}, [])]
        with mock.patch.object(q, 'sources', side_effect=changed), self.assertRaises(SystemExit):
            q.batch('t', self.root / 'changed', runs=self.runs)
        self.assertFalse((self.root / 'dirty').exists() or (self.root / 'changed').exists())
        manifest = self.runs / 'campaigns' / 't' / 'manifest.csv'
        text = manifest.read_text(encoding='utf-8')
        try:
            manifest.write_text('\n'.join(text.splitlines()[:-1]) + '\n', encoding='utf-8')  # an arm x seed missing
            with self.assertRaises(SystemExit):
                q.batch('t', self.root / 'short', runs=self.runs, check_sources=False)
        finally:
            manifest.write_text(text, encoding='utf-8')

    def test_sensitivity_method(self):
        """s1 on a wall 02 leaves nothing out: the same values, its own method name and options."""
        q.batch('t', self.root / 's1', runs=self.runs, check_sources=False, method='wang-quality-s1-keep-all')
        rows = list(csv.DictReader(open(self.root / 's1' / 'quality.csv', encoding='utf-8')))
        info = json.loads((self.root / 's1' / 'batch.json').read_text(encoding='utf-8'))
        self.assertEqual((info['method'], info['options']), ('wang-quality-s1-keep-all', {'leave_out': False}))
        self.assertEqual({r['method'] for r in rows}, {'wang-quality-s1-keep-all'})
        for r, v1 in zip(rows, self.rows):
            self.assertEqual({k: r[k] for k in q.COLUMNS if k != 'method'}, {k: v1[k] for k in q.COLUMNS if k != 'method'})
        self.assertEqual(self.info['options'], {})
        with self.assertRaises(ValueError):
            q.evaluate(self.runs / 'campaigns' / 't' / 'case' / 'a' / 'seed1', 'wang-quality-v2')

    def test_a_failed_evaluation_is_a_fail_row(self):
        run = self.runs / 'campaigns' / 't' / 'case' / 'a' / 'seed1'
        with mock.patch.object(q.wall_metrics, 'wall_quality', side_effect=RuntimeError('boom')):
            row, detail = q.evaluate(run)
        self.assertEqual((row['status'], row['problems'], detail), ('FAIL', ['evaluation raised RuntimeError: boom'], {}))
        real = q.wall_metrics.wall_quality
        with mock.patch.object(q.wall_metrics, 'wall_quality', side_effect=lambda *a: {**real(*a), 'F_LR': None}):
            row, _ = q.evaluate(run)
        self.assertEqual((row['status'], row['problems']), ('FAIL', ['F_LR undefined', 'F_LR_author undefined']))


if __name__ == '__main__':
    unittest.main()
