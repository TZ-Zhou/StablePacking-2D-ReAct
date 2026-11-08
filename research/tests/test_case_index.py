"""research/storage/case_index.py: one receipt to one index row, and the index rebuilt from the case folders."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'research' / 'tests')]
from research.storage import case_id as cid  # noqa: E402
from research.storage import case_index as ci  # noqa: E402
from test_case_id import RECEIPT  # noqa: E402

CASE = {**RECEIPT, 'case_id': '20261108-1254-TH_T80-pc1', 'started': '2026-11-08T12:54:31+08:00', 'parent': None,
        'status': 'PROCESS_PASS'}


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


class Row(unittest.TestCase):
    def test_columns(self):
        row = ci.row(CASE)
        self.assertEqual(tuple(row), ci.COLUMNS)
        self.assertEqual((row['scene'], row['pc'], row['order'], row['switch'], row['llm'], row['parent']),
                         ('TH_T80', 'pc1', '3-0-2-1-4', '10:0-3-1-4-2', 'mock:max_llc', ''))
        self.assertEqual(row['spec_sha256'], cid.spec_sha256(CASE))

    def test_switch_text_sorts_steps_as_numbers(self):
        self.assertEqual(ci.switch_text({'12': [1, 0], '9': [2]}), '9:2 12:1-0')
        self.assertEqual(ci.switch_text(None), '')

    def test_llm_text(self):
        self.assertEqual(ci.llm_text({'kind': 'react', 'llm': {'kind': 'api', 'provider': 'openai', 'model': 'm-1'}}),
                         'openai:m-1')
        self.assertEqual(ci.llm_text({'kind': 'rule'}), '')


class Rebuild(unittest.TestCase):
    def test_rows_in_case_id_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            for case_id, seed in (('20261108-1255-TH_T80-pc1', 2), ('20261108-1254-TH_T80-pc1', 1)):
                (Path(tmp) / case_id).mkdir()
                receipt = {**CASE, 'case_id': case_id, 'seed': seed}
                (Path(tmp) / case_id / 'receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
            self.assertEqual(ci.rebuild(tmp), 2)
            self.assertEqual([r['seed'] for r in read_csv(Path(tmp) / ci.INDEX)], ['1', '2'])


if __name__ == '__main__':
    unittest.main()
