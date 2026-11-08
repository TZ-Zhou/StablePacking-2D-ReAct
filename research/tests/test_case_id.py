"""research/storage/case_id.py: make, parse and check case ids, hash a wall's definition, reserve its folder."""
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO)]
from research.storage import case_id as cid  # noqa: E402

START = datetime(2026, 11, 8, 12, 54, 31)
RECEIPT = {'case': 'thangavelu-noise80', 'seed': 7, 'placements_limit': 40, 'timeout_seconds': 600, 'trace': True,
           'filter_policy': {'policy': 'paper', 'order': [3, 0, 2, 1, 4], 'tol': 1e-09,
                             'switch': {10: [0, 3, 1, 4, 2]}},
           'controller': {'kind': 'react', 'llm': {'kind': 'mock', 'mode': 'max_llc'}}}


class Ids(unittest.TestCase):
    def test_make_adds_a_suffix_on_collision(self):
        first = cid.make(START, 'TH_T80', 'pc1')
        self.assertEqual(first, '20261108-1254-TH_T80-pc1')
        self.assertEqual(cid.make(START, 'TH_T80', 'pc1', {first}), '20261108-1254b-TH_T80-pc1')

    def test_parse_round_trip(self):
        parsed = cid.parse('20261108-1254b-AL_IR-pc2')
        self.assertEqual(parsed, cid.CaseId('20261108-1254', 'b', 'AL_IR', 'pc2'))
        self.assertEqual(f'{parsed.minute}{parsed.suffix}-{parsed.scene}-{parsed.pc}', '20261108-1254b-AL_IR-pc2')

    def test_archive_paths_are_not_case_ids(self):
        for path in ('almeida-ir', 'gpt-walls/gpt-001', '20261108-1254-TH_T80'):
            self.assertFalse(cid.is_case_id(path), path)
        with self.assertRaises(ValueError):
            cid.parse('almeida-ir')

    def test_problems(self):
        self.assertEqual(cid.problems('20261108-1254-TH_T80-pc1'), [])
        self.assertEqual(len(cid.problems('20261399-1254-TH_T80-pc1')), 1)  # month 13
        self.assertEqual(len(cid.problems('20261108-1254-XX-pc1')), 1)  # no such scene
        self.assertEqual(len(cid.problems('gpt-walls/gpt-001')), 1)

    def test_scene_and_pc_codes(self):
        self.assertEqual(cid.scene_code('thangavelu-noise20'), 'TH_T20')
        self.assertEqual(cid.pc_code('giaail-strix'), 'pc2')
        with self.assertRaises(ValueError):
            cid.pc_code('unknown-host')
        with self.assertRaises(ValueError):
            cid.scene_code('no-such-case')


class Spec(unittest.TestCase):
    def test_hash_survives_a_json_round_trip(self):
        back = json.loads(json.dumps(RECEIPT))  # switch steps come back as str keys
        self.assertEqual(cid.spec_sha256(back), cid.spec_sha256(RECEIPT))

    def test_hash_follows_the_definition_only(self):
        base = cid.spec_sha256(RECEIPT)
        self.assertNotEqual(cid.spec_sha256({**RECEIPT, 'seed': 8}), base)
        self.assertEqual(cid.spec_sha256({**RECEIPT, 'timeout_seconds': 1, 'trace': False, 'worker_venv': {}}), base)
        self.assertEqual(cid.spec_sha256({**RECEIPT, 'scene': 'TH_T80'}), base)


class Allocate(unittest.TestCase):
    def test_two_walls_in_one_minute(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = cid.allocate('bricks', START, 'GIAAIL-STRIX', tmp)
            second = cid.allocate('bricks', START, 'GIAAIL-STRIX', tmp)
            self.assertEqual((first[0], second[0]), ('20261108-1254-BR-pc2', '20261108-1254b-BR-pc2'))
            self.assertTrue(second[1].is_dir())

    def test_refuses_the_archive(self):
        with self.assertRaises(ValueError):
            cid.allocate('bricks', START, 'GIAAIL-STRIX', cid.LEGACY_RUNS / 'new')

    def test_wall_dir(self):
        self.assertEqual(cid.wall_dir('20261108-1254-BR-pc1'), cid.CASES_DIR / '20261108-1254-BR-pc1')
        self.assertEqual(cid.wall_dir('almeida-ir'), cid.LEGACY_RUNS / 'almeida-ir')


if __name__ == '__main__':
    unittest.main()
