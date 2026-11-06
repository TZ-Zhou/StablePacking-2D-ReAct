"""Checks for Stonepacker2D/stack/selection.py and the offline sensitivity replay.

Run from the repository root: python -B -m unittest discover -s research/tests
"""
import csv
import io
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

from Stonepacker2D.stack.selection import (ALL_ORDERS, PAPER_ORDER, filter_log, load_trace,
                                           select, write_trace)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'research'))
import run_case  # noqa: E402
import sensitivity  # noqa: E402


def original_cascade():
    """The author's cascade, executed verbatim from the baseline-author-original tag."""
    source = subprocess.run(['git', 'show', 'baseline-author-original:Stonepacker2D/stack/build_wall.py'],
                            cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()
    start = next(i for i, l in enumerate(source) if l.strip() == 'filter_order = [1,2,3,4,0]#brick wall width 24*12')
    end = next(i for i in range(start, len(source)) if 'return False,None,None,None,None,-np.inf' in source[i])
    block = textwrap.indent(textwrap.dedent('\n'.join(source[start:end + 1])), '    ')
    namespace = {'np': np}
    exec('def cascade(filters_matrix, all_bricks, construction_step_i, get_dir, open,'
         ' get_selection_weights, get_main_logger, nb_stones):\n' + block + '\n    return min_index\n', namespace)
    return namespace['cascade']


class Brick:
    def __init__(self, id):
        self.id = id


def run_original(cascade, raw, weights, step=7):
    """Returns (selected index or None, text written to filter.txt)."""
    buffer = io.StringIO()
    names = ['course_height', 'global_density', 'interlocking', 'kinematic', 'local_void', 'local_void_center']

    class Handle:
        def __enter__(self):
            return buffer

        def __exit__(self, *exc):
            return False

    result = cascade(raw.copy(), [Brick(100 + i) for i in range(len(raw))], step, lambda: '',
                     lambda *a: Handle(), lambda: dict(zip(names, weights)), lambda: mock.Mock(), len(raw))
    index = None if isinstance(result, tuple) else result
    return index, buffer.getvalue()


def random_matrix(rng):
    """Feature matrices shaped like place_one_stone output, rich in ties and edge cases."""
    stones = int(rng.integers(1, 13))
    n = stones * 4
    raw = np.empty((n, 6))
    raw[:, 0] = -np.round(rng.random(n), int(rng.integers(1, 4)))               # NHR <= 0
    gds = rng.random(n) * 1.2
    raw[:, 1] = np.where(gds > 0.7, gds, 0.0)                                   # H(GDS, 0.7)
    raw[:, 2] = rng.choice([0.5, 0.0, 1.0, 0.25], n) if rng.random() < 0.5 else rng.random(n)
    raw[:, 3] = np.round(rng.uniform(-0.3, 1.2, n), 1) / 0.58                   # rounded LLC / mu
    raw[:, 4] = -np.round(rng.random(n) * 0.3, 2)
    raw[:, 5] = rng.random(n)
    if rng.random() < 0.3:                                                      # failed placements
        raw[rng.random(n) < 0.3] = -1
    if rng.random() < 0.3:                                                      # symmetric stones
        raw[n // 2:] = raw[:n - n // 2]
    if rng.random() < 0.4:  # repeated value whose float mean exceeds it (e.g. 3 x 0.1): empty stage
        raw[raw[:, 3] > 0, int(rng.integers(5))] = rng.choice([0.1, 0.7, 0.1 / 0.58])
    if rng.random() < 0.05:  # includes the unused sixth column (Codex F3)
        raw[rng.integers(n), rng.integers(6)] = rng.choice([np.nan, np.inf])
    weights = np.ones(6) if rng.random() < 0.7 else rng.choice([0.0, 0.5, 1.0, 2.0], 6)
    return raw, weights


def reference_paper(raw, weights, order, tol):
    """Algorithm 2 written independently as plain set filtering."""
    weighted = raw * weights
    pi = [i for i in range(len(raw)) if raw[i, 3] > 0 and np.isfinite(raw[i, list(order)]).all()]
    if not pi:
        return None, []
    stages = [pi]
    for k, col in enumerate(order):
        mean = np.mean([weighted[i, col] for i in pi])
        kept = [i for i in pi if weighted[i, col] >= mean - tol * max(1.0, abs(mean))]
        if not kept:
            return max(pi, key=lambda i: (sum(weighted[i, c] for c in order[:k + 1]), -i)), stages
        pi = kept
        stages.append(pi)
    return max(pi, key=lambda i: (sum(weighted[i, c] for c in order), -i)), stages


warnings.simplefilter('ignore', RuntimeWarning)  # empty-slice means are part of the original behaviour


class CodeOriginalEquivalence(unittest.TestCase):
    def test_matches_author_code_on_random_matrices(self):
        cascade = original_cascade()
        rng = np.random.default_rng(1307)
        branches = set()
        for trial in range(5000):
            raw, weights = random_matrix(rng)
            expected_index, expected_log = run_original(cascade, raw, weights)
            sel = select(raw.copy(), weights)
            self.assertEqual(expected_index, sel.index, f'trial {trial}')
            self.assertEqual(expected_log, filter_log(7, [100 + i for i in range(len(raw))], sel), f'trial {trial}')
            branches.add(sel.branch)
        self.assertEqual(branches, {'full', 'no_feasible', 'fallback_1', 'fallback_2', 'fallback_3', 'fallback_4'})


class PaperSemantics(unittest.TestCase):
    #               NHR   GDS   IWR   LLC   LFR   (unused)
    RAW = np.array([[-0.1, 0.9, 0.9, 0.0, -0.1, 0],      # 0 LLC = 0: infeasible
                    [-0.1, 0.8, 0.6, 0.9, -0.2, 0],      # 1
                    [-0.2, 0.9, 0.8, 0.8, -0.1, 0],      # 2
                    [-0.6, 0.0, 0.9, 0.9, -0.1, 0],      # 3
                    [-0.1, 0.0, 0.2, 0.2, -0.3, 0],      # 4
                    [np.nan, 0.9, 0.9, 0.9, -0.1, 0]])   # 5 NaN: infeasible

    def test_hand_computed_algorithm_2(self):
        sel = select(self.RAW, np.ones(6), 'paper', PAPER_ORDER)
        # Pi1 {1,2,3,4}; LLC mean .7 -> {1,2,3}; NHR mean -.3 -> {1,2}; IWR mean .7 -> {2}; GDS, LFR keep {2}
        self.assertEqual([s.tolist() for s in sel.stages], [[1, 2, 3, 4], [1, 2, 3], [1, 2], [2], [2], [2]])
        self.assertEqual((sel.index, sel.branch), (2, 'full'))

    def test_each_filter_applied_once_in_order(self):
        rng = np.random.default_rng(20)
        for trial in range(300):
            raw, weights = random_matrix(rng)
            weights = np.where(weights == 0, 1.0, weights)  # Algorithm 2 assumes positive weights
            for order in ALL_ORDERS[::7]:
                for tol in (1e-9, 0.0):
                    sel = select(raw, weights, 'paper', order, tol)
                    index, stages = reference_paper(raw, weights, order, tol)
                    self.assertEqual(index, sel.index, (trial, order, tol))
                    self.assertEqual(stages, [s.tolist() for s in sel.stages], (trial, order, tol))
                    if sel.branch == 'full':
                        self.assertEqual(len(sel.stages), 6)  # gate + exactly five filters

    def test_no_feasible_candidate(self):
        raw = self.RAW.copy()
        raw[:, 3] = [0, -1, 0, -0.5, 0, 0.3]  # only row 5, which is NaN
        self.assertEqual(select(raw, np.ones(6), 'paper').branch, 'no_feasible')
        self.assertIsNone(select(raw, np.ones(6), 'paper').index)

    def test_rounded_llc_tie(self):
        # mean(0.1, 0.1, 0.1) == 0.10000000000000002 > 0.1: strict comparison empties the stage
        raw = np.array([[-0.3, 0, 0.5, 0.1, -0.1, 0], [-0.1, 0, 0.5, 0.1, -0.1, 0], [-0.2, 0, 0.5, 0.1, -0.1, 0]])
        tolerant = select(raw, np.ones(6), 'paper', PAPER_ORDER, 1e-9)
        strict = select(raw, np.ones(6), 'paper', PAPER_ORDER, 0.0)
        self.assertEqual(tolerant.stages[1].tolist(), [0, 1, 2])
        self.assertEqual(tolerant.index, 1)                                   # NHR then decides
        self.assertEqual(strict.branch, 'fallback_1')                         # Eq. 15: argmax LLC
        self.assertEqual(strict.index, 0)                                     # tie -> first row

    def test_unused_sixth_column_cannot_change_paper_selection(self):
        # Codex F3: matmul spread a NaN in local_void_center into all five weighted scores.
        raw = np.array([[-0.5, 0, 0.5, 0.5, -0.1, 0.0],    # weaker
                        [-0.1, 0, 0.5, 0.9, -0.1, 0.0]])   # stronger
        expected = select(raw, np.ones(6), 'paper')
        self.assertEqual((expected.index, expected.branch), (1, 'full'))
        for value in (np.nan, np.inf, -np.inf):
            contaminated = raw.copy()
            contaminated[0, 5] = value
            for order in ALL_ORDERS:
                self.assertEqual(select(contaminated, np.ones(6), 'paper', order).index,
                                 select(raw, np.ones(6), 'paper', order).index, (value, order))

    def test_invalid_order_rejected(self):
        for order in [(0, 1, 2, 3), (0, 0, 1, 2, 3), (1, 2, 3, 4, 5)]:
            with self.assertRaises(ValueError):
                select(self.RAW, np.ones(6), 'paper', order)


class AllOrders(unittest.TestCase):
    def test_every_order_returns_a_feasible_candidate(self):
        rng = np.random.default_rng(7)
        self.assertEqual(len(set(ALL_ORDERS)), 120)
        for _ in range(200):
            raw, weights = random_matrix(rng)
            for order in ALL_ORDERS:
                for tol in (1e-9, 0.0):
                    sel = select(raw, weights, 'paper', order, tol)
                    if sel.index is None:
                        self.assertEqual(sel.branch, 'no_feasible')
                    else:
                        self.assertIn(sel.index, sel.stages[0])

    def test_input_matrix_is_never_modified(self):
        raw, weights = random_matrix(np.random.default_rng(3))
        before = raw.copy()
        raw.setflags(write=False)  # any in-place write would raise
        for order in ALL_ORDERS:
            select(raw, weights, 'paper', order)
        select(raw, weights, 'code_original')
        np.testing.assert_array_equal(before, raw)


def synthetic_trace(directory, event, rng):
    raw, weights = random_matrix(rng)
    n = len(raw)
    sel = select(raw, weights)
    meta = {'run_id': 'unit', 'event': event, 'step': event // 2, 'side': 'left', 'phase': 'single',
            'policy': {'policy': 'code_original', 'order': None, 'tol': 1e-9}, 'branch': sel.branch,
            'stages': [s.tolist() for s in sel.stages]}
    write_trace(directory, meta, features_raw=raw, weights=weights, stone_id=np.arange(n) % (n // 4),
                rotation=np.arange(n) // (n // 4) * 90, x=np.arange(n), y=np.zeros(n, int),
                distance=np.zeros(n), wall=np.zeros((4, 4)),
                selected=np.array(-1 if sel.index is None else sel.index))


def forbidden(*args, **kwargs):
    raise AssertionError('offline replay must not run placement, convolution or kinematics')


class OfflineReplay(unittest.TestCase):
    def test_replay_uses_only_traced_matrices(self):
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as tmp:
            for event in range(6):
                synthetic_trace(Path(tmp) / 'trace', event, rng)
            meta, arrays = load_trace(Path(tmp) / 'trace' / 'e00000.npz')
            self.assertEqual(meta['event'], 0)
            # sys.modules lookup: the package re-exports a function named `evaluate` that
            # shadows the Stonepacker2D.evaluate subpackage attribute.
            patches = [mock.patch.object(sys.modules[module], name, forbidden) for module, name in (
                ('scipy.ndimage', 'convolve'),
                ('Stonepacker2D.stack.place_stone', 'place_one_stone'),
                ('Stonepacker2D.stack.build_wall', 'place_one_stone'),
                ('Stonepacker2D.stack.place_stone', 'evaluate_kine'),
                ('Stonepacker2D.stack.place_stone', 'evaluate_convolve'),
                ('Stonepacker2D.evaluate.tilting_angle', 'evaluate_kine'))]
            for p in patches:
                p.start()
            try:
                summary = sensitivity.main([str(Path(tmp) / 'trace'), str(Path(tmp) / 'out')])
            finally:
                for p in patches:
                    p.stop()
            self.assertEqual(summary['events'], 6)
            self.assertTrue(summary['live_replay_consistent'])
            orders = (Path(tmp) / 'out' / 'orders.csv').read_text(encoding='utf-8').splitlines()
            self.assertEqual(len(orders), 1 + 6 * 120)
            with self.assertRaises(FileExistsError):  # an existing analysis is never overwritten
                sensitivity.main([str(Path(tmp) / 'trace'), str(Path(tmp) / 'out')])


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def infeasible_trace(directory, event):
    raw = np.array([[-0.1, 0, 0.5, -0.2, -0.1, 0], [-0.2, 0, 0.5, 0.0, -0.1, 0]])  # LLC <= 0
    meta = {'run_id': 'unit', 'event': event, 'step': event, 'side': 'left', 'phase': 'single',
            'policy': {'policy': 'code_original', 'order': None, 'tol': 1e-9}}
    write_trace(directory, meta, features_raw=raw, weights=np.ones(6), stone_id=np.array([1, 2]),
                rotation=np.zeros(2, int), x=np.arange(2), y=np.zeros(2, int), selected=np.array(-1))


class InfeasibleEvents(unittest.TestCase):
    # Codex F2: None vs '' sentinel mismatch and IndexError on an empty candidates table.
    def test_all_infeasible_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            for event in range(2):
                infeasible_trace(Path(tmp) / 'trace', event)
            summary = sensitivity.main([str(Path(tmp) / 'trace'), str(Path(tmp) / 'out')])
            events = list(read_csv(Path(tmp) / 'out' / 'events.csv'))
            for e in events:
                self.assertEqual((e['n_feasible'], e['n_distinct_pose'], e['n_distinct_stone']), ('0', '0', '0'))
                self.assertEqual((e['orders_same_as_paper'], e['orders_without_selection']), ('120', '120'))
            self.assertEqual((Path(tmp) / 'out' / 'candidates.csv').read_text(encoding='utf-8').strip(),
                             ','.join(sensitivity.CANDIDATE_FIELDS))
            self.assertTrue(summary['live_replay_consistent'])

    def test_mixed_trace_counts_match_order_rows(self):
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as tmp:
            infeasible_trace(Path(tmp) / 'trace', 0)
            for event in range(1, 5):
                synthetic_trace(Path(tmp) / 'trace', event, rng)
            sensitivity.main([str(Path(tmp) / 'trace'), str(Path(tmp) / 'out')])
            orders = list(read_csv(Path(tmp) / 'out' / 'orders.csv'))
            for e in read_csv(Path(tmp) / 'out' / 'events.csv'):
                mine = [o for o in orders if o['event'] == e['event']]
                self.assertEqual(int(e['orders_same_as_paper']), sum(o['same_as_paper'] == 'True' for o in mine))
                self.assertEqual(int(e['n_distinct_pose']), len({o['pose_id'] for o in mine} - {''}))


class LauncherSafety(unittest.TestCase):
    def test_real_process_tree_is_verified_stopped(self):
        # Codex F1: a parent with a grandchild, killed and verified by PID + creation time.
        code = ('import subprocess, sys, time; '
                'subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"]); time.sleep(120)')
        parent = subprocess.Popen([sys.executable, '-c', code])
        try:
            deadline = time.monotonic() + 30
            while len(run_case.process_tree(run_case.process_table(), parent.pid)) < 2:
                self.assertLess(time.monotonic(), deadline, 'grandchild did not start')
                time.sleep(0.5)
            record = run_case.terminate(parent.pid)
            self.assertTrue(record['verified'], record)
            self.assertEqual(len(record['tree']), 2)
            self.assertEqual(record['alive_after'], [])
        finally:
            parent.kill()

    def test_unverified_termination_is_a_failure(self):
        # taskkill "succeeds" but a member stays alive: never reported as a successful stop.
        table = {100: (1, 'a'), 101: (100, 'b')}
        with mock.patch.object(run_case, 'process_table', return_value=table), \
                mock.patch.object(run_case.subprocess, 'run', return_value=mock.Mock(returncode=0, stdout='', stderr='')):
            record = run_case.terminate(100, settle_seconds=0)
        self.assertFalse(record['verified'])
        self.assertEqual(record['alive_after'], [100, 101])
        ok_trace = {'ok': True}
        self.assertEqual(run_case.final_status('STOPPED_AFTER_PLACEMENTS', ok_trace, record), 'TERMINATION_FAILED')
        self.assertEqual(run_case.final_status('TIMEOUT', ok_trace, record), 'TERMINATION_FAILED')

    def test_incomplete_trace_is_never_success(self):
        rng = np.random.default_rng(9)
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / 'result' / 'trace'
            self.assertFalse(run_case.check_trace(Path(tmp))['ok'])            # no trace at all
            for event in (0, 1, 3):                                            # gap at 2
                synthetic_trace(trace, event, rng)
            self.assertIn('not contiguous', run_case.check_trace(Path(tmp))['problems'][0])
            synthetic_trace(trace, 2, rng)
            self.assertTrue(run_case.check_trace(Path(tmp))['ok'])
            (trace / 'e00004.npz').write_bytes(b'truncated')                   # unreadable file
            check = run_case.check_trace(Path(tmp))
            self.assertFalse(check['ok'])
            for status in ('PROCESS_PASS', 'STOPPED_AFTER_PLACEMENTS'):
                self.assertEqual(run_case.final_status(status, check, None), 'INCOMPLETE_TRACE')
            self.assertEqual(run_case.final_status('FAILED', check, None), 'FAILED')

    def test_trace_write_failure_raises(self):
        # Construction must stop rather than continue with a missing event.
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / 'trace'
            blocker.write_text('a file where the trace directory should be')
            with self.assertRaises(OSError):
                write_trace(str(blocker), {'event': 0}, features_raw=np.zeros((1, 6)))

    def test_fingerprint_covers_untracked_selector(self):
        # Codex F4: changing the selector source must change the recorded provenance.
        with tempfile.TemporaryDirectory() as tmp:
            selector = Path(tmp) / 'Stonepacker2D' / 'stack' / 'selection.py'
            selector.parent.mkdir(parents=True)
            selector.write_text('A = 1\n')
            before = run_case.code_fingerprint(tmp)
            selector.write_text('A = 2\n')
            after = run_case.code_fingerprint(tmp)
            self.assertIn('Stonepacker2D/stack/selection.py', before)
            self.assertNotEqual(before, after)
        real = run_case.code_fingerprint()
        self.assertIn('Stonepacker2D/stack/selection.py', real)
        self.assertIn('research/run_case.py', real)


class OutputProtection(unittest.TestCase):
    def test_launcher_refuses_baseline_and_existing_directories(self):
        for name in ['../../baseline/cross-case/runs/almeida-pr-20261106-01', '..', '../../baseline']:
            with self.assertRaises((ValueError, FileExistsError)):
                run_case.prepare_output(name)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(run_case, 'RUNS', Path(tmp)):
            run_case.prepare_output('x')
            with self.assertRaises(FileExistsError):
                run_case.prepare_output('x')


if __name__ == '__main__':
    unittest.main()
