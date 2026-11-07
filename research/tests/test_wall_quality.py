"""research/metrics.wall_quality, the four formal wall metrics by the author's method: the
original-resolution rendering of Tests/Thangavelu/02_view_transformation_ori_reso.py
(render_original) then 03 (wang_metrics). Checked against a live run of the author's 02 -> 03, the
values of five recorded walls, the refusals on inconsistent data, and that no recorded run,
controller prompt or smoke record changed."""
import hashlib
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'research'))
sys.path.insert(0, str(REPO / 'research' / 'tests'))
import metrics  # noqa: E402
from test_controllers import WALL  # noqa: E402

AUTHOR = REPO / 'Tests' / 'Thangavelu'
RUNS = REPO / 'research' / 'runs'
BRANCHES = RUNS / 'branches' / 'paper-hardstone-01'

# wall: F_SF, F_AH, F_AV, lambda_min of the author's 02 -> 03 run on it (p22), the sha256 of its
# rendered label image, the stones 02's canvas crop cut (id: pixels lost)
FIVE = {
    WALL: (0.36848513718663883, 3.7344681717058616, 22.966398000168624, 0.06279094228486494,
           '781bf7d638828825c9ae9ffb61db8fff1e19e2817e0f9d991304fc59d919223d', {}),
    BRANCHES / 'seed184947496-step10-control-stone19-0-3-1-4-2': (
        0.3531103503234337, 5.680757235345211, 14.85373861871345, 0.13080192748860558,
        '59f7a7c9c4d0f64132a8ee0a31c09b1cb3ee7dc790d9b0a1a1b3f168dfe13f28', {18: 272, 1: 6590}),
    BRANCHES / 'seed184947496-step10-control-stone19-3-2-0-4-1': (
        0.40043450968520305, 8.01635990854515, 21.756938764772844, 0.12071498527306786,
        'f5bf148f72824579f5e51cc1c184e28d72945214f63acb9dd8a606418b96396c', {4: 1818}),
    BRANCHES / 'seed184947496-step10-control-stone19-3-4-0-1-2': (
        0.3976965644352042, 6.291020666296871, 21.089977019035672, 0.15720222966413225,
        'd8dc1f7abca5736ed0c257120ecfdcaf7c4bf407fabd653ad862c1047592d4fd', {4: 3156}),
    BRANCHES / 'seed184947496-step10-target-stone13-3-2-0-1-4': (
        0.40374007773093346, 7.012608463249623, 20.663834581265874, 0.2049128200086546,
        '77c3db374e275856b3172184d8b87e9f2959f4a76d145e0be9e38acbebbdf07d', {4: 362}),
}

# (files, digest) of record trees, taken after the five walls were checked unchanged by hash
RECORDS = {
    'campaigns': (27364, '284ac916db5fbeb45b8a49f4edca5943271c7199d9c56e361aee11f279ce9b23'),
    'api-smoke/smoke-001': (31, '3cba4d067fbfa145f6a801c30892419efbecb8b37f472242ce17db760905da88'),
    'api-smoke/smoke-002': (31, 'e525f6719788ac28eda42a205ae08a77c66ae26693f6771497980e1f654370d5'),
    'branches/paper-hardstone-01/seed184947496-step10-control-stone19-0-3-1-4-2':
        (511, '7e9d7f5f4e67aaf4b4f600eb94d432a2d8cad94ceb231a61e1bdfaa132b7f017'),
    'branches/paper-hardstone-01/seed184947496-step10-control-stone19-3-2-0-4-1':
        (508, '8287f35e9bf2c20e4a30d7471774807c3dd3637825efa9fef0db190711709253'),
    'branches/paper-hardstone-01/seed184947496-step10-control-stone19-3-4-0-1-2':
        (508, '5a963278676fe74ba6a8b53d6e2834e3be9e47ae25cbf76caac11406edefea23'),
    'branches/paper-hardstone-01/seed184947496-step10-target-stone13-3-2-0-1-4':
        (508, '74a0e400f59d5be745fde23422ed9efe025516b2be9dd41986d7bbd5e45df244'),
}
V2_PROMPT_SHA256 = '91a0f6f7e8efca4846080897556d2d7176b182f5315097768d4dda3101fbbbae'  # smoke-002's


def digest(d):
    h, n = hashlib.sha256(), 0
    for p in sorted(Path(d).rglob('*')):
        if p.is_file():
            h.update(p.relative_to(d).as_posix().encode() + b'\0' + hashlib.sha256(p.read_bytes()).digest())
            n += 1
    return n, h.hexdigest()


def author_02_03(wall, work):
    """The author's 02 then 03 on one wall in work, the scripts patched only in their path, noise
    level and loop lines; returns 02's label image and 03's F_SF, F_AV, F_AH, lambda_min."""
    result = next(wall.glob('result_*'))
    (work / 'data').mkdir()
    shutil.copy(metrics.stone_source(wall), work / 'data')
    shutil.copy(AUTHOR / 'graph.py', work)
    config = 'data_dir_prefix = "./result_x_wall"\nnoise_level = 80\n'
    for name, subs in (('02_view_transformation_ori_reso.py', [
            ('_rescale = 0.5\n', config + "wall_image = f'./data/stones_noise_{noise_level}_labels.png'\n_rescale = 0.5\n"),
            ('for i in range(5):', 'for i in range(1):')]),
            ('03_evaluate_wall_metrics.py', [
                ('#write metrics\nwith open', config + '#write metrics\nwith open'),
                ('for i in range(20):', 'for i in range(1):')])):
        s = (AUTHOR / name).read_text(encoding='utf-8')
        for a, b in subs:
            assert s.count(a) == 1, (name, a)
            s = s.replace(a, b)
        (work / name).write_text(s, encoding='utf-8')
    d = work / 'result_x_wall0'
    (d / 'img').mkdir(parents=True)
    shutil.copy(result / 'transformation.txt', d)
    for p in (result / 'img').glob('*_mechanism.png'):
        (d / 'img' / p.name).write_bytes(b'')  # 03 reads only the names
    env = {**os.environ, 'PYTHONPATH': str(REPO), 'MPLBACKEND': 'Agg', 'PYTHONDONTWRITEBYTECODE': '1'}
    env.pop('OPENAI_API_KEY', None)
    for name in ('02_view_transformation_ori_reso.py', '03_evaluate_wall_metrics.py'):
        subprocess.run([sys.executable, '-W', 'ignore', name], cwd=work, env=env, check=True, capture_output=True)
    row = (work / 'result_all' / 'metrics_80.txt').read_text().splitlines()[-1]
    return cv2.imread(str(d / 'stacked_wall_label.png'), cv2.IMREAD_GRAYSCALE), [float(v) for v in row.split(';') if v][1:]


@unittest.skipUnless(WALL.exists(), 'recorded wall absent')
class AgainstTheAuthor(unittest.TestCase):
    def test_02_and_03_on_the_paper_wall(self):
        result = next(WALL.glob('result_*'))
        before = digest(WALL)
        with tempfile.TemporaryDirectory() as d:
            image, (sf, av, ah, lam) = author_02_03(WALL, Path(d))
        labels, render = metrics.render_original(WALL, result)
        q = metrics.wall_quality(WALL, result)
        self.assertTrue(np.array_equal(image, labels))
        self.assertEqual((q['F_SF'], q['F_AV'], q['F_AH'], q['lambda_min']), (sf, av, ah, lam))
        self.assertEqual(q['F_LR'], lam / 0.58)
        self.assertEqual(digest(WALL), before)  # wall_quality only reads


class FiveWalls(unittest.TestCase):
    def test_values(self):
        for wall, (sf, ah, av, lam, sha, cut) in FIVE.items():
            if not wall.exists():
                continue
            with self.subTest(wall.name):
                result = next(wall.glob('result_*'))
                labels, render = metrics.render_original(wall, result)
                q = metrics.wall_quality(wall, result)
                self.assertEqual(hashlib.sha256(labels.tobytes()).hexdigest(), sha)
                self.assertEqual((q['F_SF'], q['F_AH'], q['F_AV'], q['lambda_min']), (sf, ah, av, lam))
                self.assertEqual(render['size'], [559, 810])
                self.assertEqual({s['id']: s['lost_to_crop'] for s in render['stones'] if s['lost_to_crop'] > 20}, cut)
                self.assertFalse(any(s['left_out'] for s in render['stones']))
                self.assertEqual(len(q['F_AH_factors']), 10)
                self.assertNotIn('wall_density', q)  # the formal metrics stay apart from wall_metrics


@unittest.skipUnless(WALL.exists(), 'recorded wall absent')
class Refusals(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name)
        shutil.copy(WALL / '00_run.py', self.run_dir)
        shutil.copytree(WALL / 'data', self.run_dir / 'data')
        source = next(WALL.glob('result_*'))
        self.result = self.run_dir / source.name
        self.result.mkdir()
        for name in ('base.pkl', 'transformation.txt'):
            shutil.copy(source / name, self.result)
        self.lines = (self.result / 'transformation.txt').read_text().splitlines()

    def tearDown(self):
        self.tmp.cleanup()

    def test_another_rescale(self):
        p = self.run_dir / '00_run.py'
        p.write_text(p.read_text(encoding='utf-8').replace('_rescale = 0.5', '_rescale = 0.25'), encoding='utf-8')
        with self.assertRaises(ValueError):
            metrics.render_original(self.run_dir, self.result)

    def test_a_repeated_or_unknown_stone(self):
        for extra in (self.lines[1], '1' + self.lines[1], '999;1;1;0'):
            (self.result / 'transformation.txt').write_text('\n'.join(self.lines + [extra]))
            with self.subTest(extra), self.assertRaises(ValueError):
                metrics.render_original(self.run_dir, self.result)

    def test_transformation_and_base_disagree(self):
        (self.result / 'transformation.txt').write_text('\n'.join(self.lines[:-1]))
        metrics.render_original(self.run_dir, self.result)  # renders, but is not this wall
        with self.assertRaises(ValueError):
            metrics.wall_quality(self.run_dir, self.result)

    def test_missing_transformation(self):
        (self.result / 'transformation.txt').unlink()
        with self.assertRaises(FileNotFoundError):
            metrics.wall_quality(self.run_dir, self.result)

    def test_an_overlapping_stone_is_left_out(self):
        """The last stone drawn next to the first, shifted right by dx: 02 leaves it out from an
        overlap of 200 px on (343 px at dx 168), draws it below (148 px at dx 176)."""
        last = pickle.load(open(self.result / 'base.pkl', 'rb')).placed_rocks[-1].id
        first = self.lines[1].split(';')
        for dx, overlap, out in ((168, 343, True), (176, 148, False)):
            moved = ';'.join([str(last), str(float(first[1]) + dx)] + first[2:])
            (self.result / 'transformation.txt').write_text('\n'.join(self.lines[:2] + [moved]))
            labels, render = metrics.render_original(self.run_dir, self.result)
            with self.subTest(dx):
                self.assertEqual((render['stones'][1]['overlap'], render['stones'][1]['left_out']), (overlap, out))
                self.assertEqual(2 in labels, not out)


class RecordsUnchanged(unittest.TestCase):
    def test_record_trees(self):
        for rel, pinned in RECORDS.items():
            if (RUNS / rel).exists():
                with self.subTest(rel):
                    self.assertEqual(digest(RUNS / rel), pinned)
        self.assertTrue((RUNS / 'never-created').exists())

    def test_v2_prompt_is_smoke_002s(self):
        import controllers
        self.assertEqual(controllers.ReActPolicy(None, observation_version=2).prompt_sha256, V2_PROMPT_SHA256)


if __name__ == '__main__':
    unittest.main()
