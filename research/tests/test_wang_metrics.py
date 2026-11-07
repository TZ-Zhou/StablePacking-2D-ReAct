"""The four wall metrics of Wang et al. Section 5.2 in research/metrics.py against the author's own
code (Tests/Thangavelu/graph.py and the compute_FAV / compute_FAH functions of
03_evaluate_wall_metrics.py, taken from its source without running the script body), on small
walls with known answers, their edge cases, and one recorded wall."""
import ast
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'research'))
sys.path.insert(0, str(REPO / 'research' / 'tests'))
import metrics  # noqa: E402
from test_controllers import WALL  # noqa: E402

AUTHOR = REPO / 'Tests' / 'Thangavelu'


def author_functions():
    spec = importlib.util.spec_from_file_location('author_graph', AUTHOR / 'graph.py')
    graph = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(graph)
    tree = ast.parse((AUTHOR / '03_evaluate_wall_metrics.py').read_text(encoding='utf-8'))
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('compute_FAV', 'compute_FAH')]
    namespace = {'np': np, 'nx': nx}
    exec(compile(ast.Module(defs, []), '03_evaluate_wall_metrics.py', 'exec'), namespace)
    return graph.build_graph, namespace['compute_FAV'], namespace['compute_FAH']


def author_joint_image(wall_image):
    """03's lines from 'remove all zeros columns on the right' to the padding, verbatim but for
    the image file and the debug imwrite."""
    import cv2
    from skimage.measure import regionprops
    wall_image = wall_image[:, :np.max(np.argwhere(np.sum(wall_image, axis=0) != 0)) + 1]
    wall_eroded_stones = np.zeros_like(wall_image)
    for region in regionprops(wall_image):
        stones_matrix = np.zeros_like(wall_image)
        stones_matrix[region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]] = region.image
        stones_matrix = stones_matrix.astype(np.uint8)
        stones_matrix = cv2.erode(stones_matrix, np.ones((3, 3), np.uint8), iterations=1)
        wall_eroded_stones += stones_matrix
    wall_eroded_stones = np.where(wall_eroded_stones != 0, 255, 0)
    wall_eroded_stones = wall_eroded_stones[:np.max(np.argwhere(np.sum(wall_eroded_stones, axis=1) != 0)) + 1, :]
    wall_eroded_stones = np.concatenate((np.zeros((1, wall_eroded_stones.shape[1])), wall_eroded_stones,
                                         np.zeros((2, wall_eroded_stones.shape[1]))), axis=0)
    return np.concatenate((np.zeros((wall_eroded_stones.shape[0], 1)), wall_eroded_stones,
                           np.zeros((wall_eroded_stones.shape[0], 2))), axis=1)


def author_metrics(labels):
    build_graph, compute_FAV, compute_FAH = author_functions()
    img = author_joint_image(labels.astype(np.uint8))
    G = build_graph(img, erosion=False)
    for u, v, e in G.edges(data=True):  # 03's main body
        e['weight'] = np.linalg.norm(np.asarray(u) - np.asarray(v))
    return compute_FAV(G, img), compute_FAH(G, img)


def bond(courses, width, height, offset):
    """Rectangular stones in courses; every other course shifted by offset px (0 = stack bond)."""
    labels, k = np.zeros((courses * height + 4, 120), int), 0
    for c in range(courses):
        x = -(offset if c % 2 else 0)
        while x < 120:
            k += 1
            labels[2 + c * height:2 + (c + 1) * height, max(x, 0):min(x + width, 120)] = k
            x += width
    return labels


class AgainstTheAuthor(unittest.TestCase):
    def test_joint_image_and_paths_equal_03(self):
        rng = np.random.default_rng(1)
        walls = [bond(5, 30, 12, 0), bond(5, 30, 12, 15), bond(6, 24, 10, 7)]
        for _ in range(3):  # irregular walls: random blobs in rows
            labels, k = np.zeros((70, 90), int), 0
            for r in range(2, 64, 12):
                for c in range(0, 84, int(rng.integers(12, 20))):
                    k += 1
                    h, w = int(rng.integers(6, 11)), int(rng.integers(8, 14))
                    labels[r:r + h, c:c + w][labels[r:r + h, c:c + w] == 0] = k
            walls.append(labels)
        truncated = 0
        for labels in walls:
            img = metrics.joint_image(labels)
            np.testing.assert_array_equal(img, author_joint_image(labels.astype(np.uint8)))
            free = img == 0
            truncated += (free[:, 1:] & free[:, :-1]).sum() > (free[1:, :] & free[:-1, :]).sum()
            fav, fah = author_metrics(labels)
            self.assertAlmostEqual(metrics.path_metric(metrics.joint_paths(img, True)), fav, places=9)
            self.assertAlmostEqual(metrics.path_metric(metrics.joint_paths(img, False)), fah, places=9)
        self.assertGreater(truncated, 0)  # build_graph's edge truncation is exercised

    def test_filling_equals_03(self):
        labels = bond(4, 30, 12, 15)
        labels[:, 100:] = 0  # empty columns on the right are cropped first
        img = labels[:, :100]
        H, W = img.shape
        self.assertEqual(metrics.filling_author(labels), np.count_nonzero(img) / (H * W - 2 * H - 3 * W))


class KnownAnswers(unittest.TestCase):
    def test_a_path_off_the_joint_climbs_to_it_and_back(self):
        """A joint image (0 = joint) with one horizontal joint at row 10 and the two side columns
        free: the path of row 10 is straight; a path starting at row r > 10 must climb the side
        column to row 11, cut the corner diagonally, cross, and come back down, so
        F_AH factor = (2 (r - 11) + 2 sqrt 2 - 2) / (W - 1). A straight joint counts only on the
        sampled rows themselves."""
        H, W = 40, 60
        img = np.full((H, W), 255)
        img[10, :], img[:, 0], img[:, -1] = 0, 0, 0
        paths = {p['start'][0]: p['factor'] for p in metrics.joint_paths(img, False)}
        self.assertEqual(sorted(paths), [10, 14, 18, 22, 26])
        self.assertEqual(paths[10], 0.0)
        for r in (14, 18, 22, 26):
            self.assertAlmostEqual(paths[r], (2 * (r - 11) + 2 * np.sqrt(2) - 2) / (W - 1), places=12)
        build_graph, _, compute_FAH = author_functions()
        G = build_graph(img, erosion=False)
        for u, v, e in G.edges(data=True):
            e['weight'] = np.linalg.norm(np.asarray(u) - np.asarray(v))
        self.assertAlmostEqual(metrics.path_metric(metrics.joint_paths(img, False)), compute_FAH(G, img), places=12)

    def test_running_bond_interlocks_more(self):
        """Staggered vertical joints lengthen the vertical paths: higher F_AV than stack bond."""
        stack, running = (metrics.joint_image(bond(5, 30, 12, o)) for o in (0, 15))
        self.assertGreater(metrics.path_metric(metrics.joint_paths(running, True)),
                           metrics.path_metric(metrics.joint_paths(stack, True)))

    def test_the_padding_is_a_joint(self):
        """Two stones over the whole height parted by one vertical joint (columns 28-32 of the
        padded joint image): the path starting in it is straight; every other path detours,
        either to that joint or round the wall along the zero padding of its sides."""
        labels = np.zeros((40, 60), int)
        labels[2:38, :28], labels[2:38, 31:] = 1, 2
        img = metrics.joint_image(labels)
        self.assertEqual(np.flatnonzero(img[5] == 0).tolist(), [0, 28, 29, 30, 31, 32, 61, 62])
        paths = {p['start'][1]: p['factor'] for p in metrics.joint_paths(img, True)}
        self.assertEqual(sorted(paths), [10, 16, 22, 28, 34, 40, 46, 52])
        self.assertEqual(paths[28], 0.0)
        self.assertTrue(all(f > 0 for c, f in paths.items() if c != 28))
        self.assertLess(paths[10], paths[16])  # column 10 is nearer the left padding than the joint


class EdgeCases(unittest.TestCase):
    def test_too_small_and_no_paths(self):
        tiny = np.zeros((5, 8), int)
        self.assertIsNone(metrics.joint_paths(tiny, True))  # W // 10 == 0: 03 raises here
        self.assertEqual(metrics.joint_paths(np.zeros((15, 40), int), False), [])  # range(10, H - 10) empty
        self.assertIsNone(metrics.path_metric([]))  # 03 returns nan
        self.assertIsNone(metrics.path_metric([{'factor': np.inf}]))  # unreachable end: 03 raises

    def test_empty_wall_and_lambda(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'img').mkdir()
            for name in ('left0.25_mechanism.png', 'right0.116_mechanism.png'):
                (Path(d) / 'img' / name).write_bytes(b'')
            m = metrics.wang_metrics(d, labels=np.zeros((30, 40), int))
            self.assertEqual((m['F_SF_author'], m['F_AH'], m['F_AV']), (None, None, None))  # 03 raises on an empty wall
            self.assertEqual(m['lambda_min'], 0.116)
            self.assertAlmostEqual(m['F_LR'], 0.116 / 0.58)
            (Path(d) / 'img' / 'right0.116_mechanism.png').unlink()
            self.assertEqual((metrics.wang_metrics(d, labels=bond(4, 30, 12, 0))['F_LR']), None)


@unittest.skipUnless(WALL.exists(), 'recorded wall absent')
class RecordedWall(unittest.TestCase):
    def test_paper_wall_on_the_simulation_grid(self):
        """Values of the author's 03 run on this wall's simulation-grid label image (rock_labels
        written as stacked_wall_label.png), reproduced exactly."""
        m = metrics.wang_metrics(next(WALL.glob('result_*')))
        self.assertEqual(m['F_SF_author'], 0.36963928397598833)
        self.assertEqual(m['F_AV'], 22.54835745077637)
        self.assertEqual(m['F_AH'], 5.034386775071311)
        self.assertEqual(m['lambda_min'], 0.06279094228486494)
        self.assertEqual(m['F_LR'], 0.06279094228486494 / 0.58)
        self.assertEqual((len(m['F_AH_paths']), len(m['F_AV_paths'])), (9, 10))


if __name__ == '__main__':
    unittest.main()
