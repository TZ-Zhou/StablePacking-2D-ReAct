"""research/storage/legacy_runs.py: the archive's index names copies, and a frozen archive has no writable file."""
import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'research' / 'tests')]
from research.storage import legacy_runs as lr  # noqa: E402
from test_case_id import RECEIPT  # noqa: E402
from test_case_index import read_csv  # noqa: E402


class Archive(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for path in ('pilot', 'api-smoke/copy'):  # the smoke test copies the pilot wall
            (self.root / path).mkdir(parents=True)
            (self.root / path / 'receipt.json').write_text(json.dumps({**RECEIPT, 'run_id': 'pilot'}), encoding='utf-8')

    def tearDown(self):
        for p in self.root.rglob('*'):  # Windows will not delete read-only files
            if p.is_file():
                p.chmod(stat.S_IREAD | stat.S_IWRITE)
        self.tmp.cleanup()

    def test_index_names_copies(self):
        out, n = lr.write_index(self.root)
        rows = read_csv(out)
        self.assertEqual([(r['path'], r['copy_of']) for r in rows], [('api-smoke/copy', 'pilot'), ('pilot', '')])
        self.assertEqual(tuple(rows[0]), lr.COLUMNS)
        self.assertNotIn('case_id', rows[0])

    def test_freeze(self):
        lr.write_index(self.root)
        self.assertEqual(lr.freeze(self.root), 3)
        self.assertEqual(lr.writable(self.root), [])
        self.assertEqual(lr.write_index(self.root)[1], 2)  # the index can still be rewritten


if __name__ == '__main__':
    unittest.main()
