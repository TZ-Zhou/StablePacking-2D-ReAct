"""research/llm_walls/gpt_wall_report.py: the committed gpt-001 report is what its records give (decisions,
switches, trajectory), and its information check finds no final-wall term in any request."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO)]
from research.controller import llm_api as llm_adapter  # noqa: E402
from research.llm_walls import gpt_wall_report as rep  # noqa: E402

RUN = rep.quality_batch.RUNS / 'gpt-walls' / 'gpt-001'
REPORT = REPO / 'research' / 'reports' / 'gpt-001'
PAPER = rep.quality_batch.RUNS / rep.DEV.format(arm='paper', seed=66794912)


class Names(unittest.TestCase):
    def test_filter_names_follow_paper_order(self):
        self.assertEqual(rep.names('3-0-2-1-4'), 'LLC>NHR>IWR>GDS>LFR')
        self.assertTrue(rep.QUALITY_TERMS.search('the final F_LR'))
        self.assertIsNone(rep.QUALITY_TERMS.search('LLC: the kinematic limit load (tilting table)'))


@unittest.skipUnless(RUN.exists() and PAPER.exists() and REPORT.exists(), 'gpt-001 records or report not present')
class Gpt001(unittest.TestCase):
    def setUp(self):
        self.result = next(RUN.glob('result_*'))
        self.records = [json.loads(l) for l in (self.result / 'trace' / 'controller.jsonl').read_text(encoding='utf-8').splitlines()]
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_the_committed_tables_are_the_records(self):
        rows, switches = rep.decision_rows(self.result, self.records, llm_adapter.price(rep.json.loads(
            (RUN / 'receipt.json').read_text(encoding='utf-8'))['controller']['llm']))
        traj = rep.trajectory(self.result, next(PAPER.glob('result_*')))
        rep.write_csv(self.tmp / 'decisions.csv', rows)
        rep.write_csv(self.tmp / 'trajectory.csv', traj)
        for name in ('decisions.csv', 'trajectory.csv'):
            self.assertEqual((self.tmp / name).read_bytes(), (REPORT / name).read_bytes(), name)
        self.assertEqual(switches, json.loads((REPORT / 'switches.json').read_text(encoding='utf-8')))
        self.assertEqual([r['category'] for r in rows].count('candidate changed'), 9)
        self.assertEqual([r['category'] for r in rows].count('order differs, same candidate'), 0)
        self.assertEqual(next(t['step'] for t in traj if not t['same_placement']), 6)

    def test_no_request_names_a_final_wall_metric(self):
        sent = [json.dumps(t['request']) for r in self.records for t in r.get('transcripts') or []]
        self.assertEqual(len(sent), 37)
        self.assertFalse([s for s in sent if rep.QUALITY_TERMS.search(s)])

    def test_a_report_is_written_once(self):
        with self.assertRaises(SystemExit):
            rep.main('gpt-001')


if __name__ == '__main__':
    unittest.main()
