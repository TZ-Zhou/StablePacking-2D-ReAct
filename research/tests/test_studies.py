"""research/storage/studies.py: create a study once, add each case once, find cases by the study's keys."""
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO)]
from research.storage import studies as st  # noqa: E402


class Studies(unittest.TestCase):
    def test_create_add_find(self):
        with tempfile.TemporaryDirectory() as tmp:
            st.create('campaign-003', 'campaign', {'arms': ['paper', 'code_original']}, tmp)
            with self.assertRaises(FileExistsError):
                st.create('campaign-003', 'campaign', {}, tmp)
            for minute, arm in (('1254', 'paper'), ('1255', 'code_original'), ('1256', 'paper')):
                st.add_case('campaign-003', {'case_id': f'20261108-{minute}-TH_T80-pc1', 'arm': arm}, tmp)
            with self.assertRaises(ValueError):
                st.add_case('campaign-003', {'case_id': '20261108-1254-TH_T80-pc1', 'arm': 'paper'}, tmp)
            with self.assertRaises(ValueError):
                st.add_case('campaign-003', {'case_id': 'almeida-ir'}, tmp)
            found = st.find('campaign-003', tmp, arm='paper')
            self.assertEqual([e['case_id'][9:13] for e in found], ['1254', '1256'])
            self.assertEqual(st.load('campaign-003', tmp)['design'], {'arms': ['paper', 'code_original']})
            self.assertEqual(list(Path(tmp).rglob('*.tmp')), [])

    def test_study_ids(self):
        with self.assertRaises(ValueError):
            st.path_of('Campaign 3')


if __name__ == '__main__':
    unittest.main()
