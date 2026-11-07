"""Checks for the filter-order controller hook (selection.controlled_policy) and research/controllers.py.

Run from the repository root: python -B -m unittest discover -s research/tests
Tests on a recorded wall replay its trace offline and are skipped when the run is absent.
Regression tests for the P0/P1 review findings are marked R1..R8, the second review S1..S6, the third T1..T9;
GeometryReplay G1..G5 covers the live outcome geometry of Mock wall 4.
"""
import copy
import json
import multiprocessing
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
import types
from types import SimpleNamespace
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'research'))
import check_case  # noqa: E402
import controller_worker  # noqa: E402
import controllers  # noqa: E402
import run_case  # noqa: E402

from Stonepacker2D.stack import selection  # noqa: E402
from Stonepacker2D.stack.selection import controlled_policy  # noqa: E402
from Stonepacker2D.utils import constant  # noqa: E402

PAPER = {'policy': 'paper', 'order': (3, 0, 2, 1, 4), 'tol': 1e-9}
WALL = REPO / 'research' / 'runs' / 'campaigns' / 'pilot-thangavelu-4arms' / 'thangavelu-noise80' / 'paper_3-0-2-1-4' / 'seed184947496'
RANDOM_WALL = REPO / 'research' / 'runs' / 'controllers' / 'p1' / 'random1-seed20-a'
NULL_WALL = REPO / 'research' / 'runs' / 'controllers' / 'p1-r2' / 'null-seed20'
REACT_WALL = REPO / 'research' / 'runs' / 'controllers' / 'p2-mock' / 'react-maxllc-seed20'  # Mock wall 4, live geometry


class Brick:
    """Picklable stand-in for a Stone: a wall-sized matrix, metadata and transformed()."""

    def __init__(self, sid=5, width=2):
        self.id, self.width, self.matrix = sid, width, np.zeros((10, 8))

    def transformed(self, xy):
        moved = Brick(self.id, self.width)
        moved.matrix[xy[1]:xy[1] + 2, xy[0]:xy[0] + 2] = 1
        return moved


def wall():
    w = np.zeros((10, 8))
    w[:2] = 1
    w[-1], w[:, 0], w[:, -1] = 255, 255, 255
    return w


def two_outcomes(phase='single'):
    """Selection inputs where Paper (LLC, NHR first) picks candidate 0 and GDS-first orders pick 1."""
    raw = np.array([[1, 0, 0, 1, 0, 0], [0, 1, 0, 1, 0, 0]], float)
    return {'step': 1, 'side': 'left', 'phase': phase, 'sample_trial': 0, 'wall': wall(),
            'all_bricks': [Brick(5), Brick(6)], 'sampled_ids': [5, 6], 'inventory_ids': [7],
            'x': [2, 4], 'y': [2, 2], 'distance': [0.0, 0.0], 'features_raw': raw, 'weights': [1] * 6}


def react(llm, **params):
    """A ReAct client around a test LLM callable, its policy in this process."""
    worker = {k: v for k, v in params.items() if k in controllers.WORKER_KEYS['react']}
    return controllers.ControllerClient({'kind': 'react', 'llm': {'kind': 'test'}, **params},
                                        controllers.InlineTransport(lambda: controllers.ReActPolicy(llm, **worker)))


def misbehaving_worker(conn, spec):
    """Worker main for ProcessTransport tests (spawned, so it lives at module level): spec['bad']
    says how it fails; spec['fail_at'] = n serves the real policy and fails only at request n."""
    spec = json.loads(spec)
    bad, fail_at = spec.pop('bad'), spec.pop('fail_at', 1)
    if bad == 'not_ready':
        conn.send_bytes(b'{"ready": false}')
        return
    policy = controllers.make_policy(spec)
    conn.send_bytes(b'{"ready": true}')
    for n in range(1, 1000):
        request = json.loads(conn.recv_bytes())
        if n < fail_at:
            conn.send_bytes(json.dumps({'ok': policy.decide(request)}).encode())
            continue
        if bad == 'hang':
            threading.Event().wait()
        if bad == 'exit':
            os._exit(3)
        conn.send_bytes({'garbage': b'\xff\xfe{not json', 'deep': b'[' * 200000 + b']' * 200000,
                         'huge': b'"' + b'x' * (controller_worker.MAX_BYTES + 10) + b'"',
                         'shape': b'{"ok": {}, "extra": 1}', 'error': b'{"error": "ValueError: boom"}',
                         'list_order': json.dumps({'ok': {'order': [3, 0, 2, 1, 4], 'source': 'null'}}).encode(),
                         'foreign_source': json.dumps({'ok': {'order': '0-1-2-3-4', 'source': 'llm'}}).encode()}[bad])


def process_client(spec, timeout=None, **bad):
    """A client whose ProcessTransport runs misbehaving_worker (bad given) or the real worker."""
    full = controllers.full_spec(spec)
    transport = controller_worker.ProcessTransport({**full, **bad}, timeout=timeout, **(
        {'target': misbehaving_worker} if bad else {}))
    return controllers.ControllerClient(spec, transport)


def reaped(client):
    """Kill the client's worker (if any); True if no worker process of this test is left."""
    client.transport.close()
    return client.transport.process is None and not multiprocessing.active_children()


def context():
    return {'step': 3, 'wall': np.zeros((5, 5)), 'features_raw': np.ones((2, 6)), 'all_bricks': [Brick()]}


class ControlledPolicy(unittest.TestCase):
    def tearDown(self):
        constant.set_filter_policy()

    def test_no_controller_or_unchanged_policy_is_the_same_object(self):
        self.assertIs(controlled_policy(PAPER, None, context()), PAPER)
        self.assertIs(controlled_policy(PAPER, lambda p, c: p, context()), PAPER)
        self.assertIs(controlled_policy(PAPER, lambda p, c: dict(p), context()), PAPER)

    def test_only_an_int_permutation_of_the_order_may_change(self):
        for order in ([0, 1, 2, 3, 4], (0, 1, 2, 3, 4)):
            self.assertEqual(controlled_policy(PAPER, lambda p, c: {**p, 'order': order}, context())['order'], (0, 1, 2, 3, 4))
        for bad in ({'order': (0, 0, 1, 2, 3)}, {'order': (0.0, 1.0, 2.0, 3.0, 4.0)}, {'order': '0-1-2-3-4'},
                    {'order': tuple(np.arange(5))}, {'tol': 0.0}, {'policy': 'code_original'}, {'extra': 1}):
            with self.assertRaises(ValueError, msg=bad):
                controlled_policy(PAPER, lambda p, c: {**p, **bad}, context())
        with self.assertRaises(ValueError):
            controlled_policy(PAPER, lambda p, c: [3, 0, 2, 1, 4], context())
        with self.assertRaises(ValueError):
            controlled_policy({**PAPER, 'policy': 'code_original'}, lambda p, c: p, context())

    def test_r1_in_place_policy_edits_are_impossible(self):
        constant.set_filter_policy('paper', (3, 0, 2, 1, 4))
        before = dict(constant.get_filter_policy())

        def mutate(p, c):
            p['policy'], p['tol'] = 'code_original', 1.0
            return p
        with self.assertRaises(TypeError):
            controlled_policy(constant.get_filter_policy(), mutate, context())

        def mutate_global(p, c):
            constant.get_filter_policy()['tol'] = 1.0
            return p
        with self.assertRaisesRegex(RuntimeError, 'global settings'):
            controlled_policy(constant.get_filter_policy(), mutate_global, context())
        constant.get_filter_policy()['tol'] = before['tol']
        self.assertEqual(constant.get_filter_policy(), before)

    def test_s1_global_setters_and_rebinding_are_caught(self):
        constant.set_filter_policy('paper', (3, 0, 2, 1, 4))
        saved = {k: copy.deepcopy(v) if k.isupper() else v for k, v in vars(constant).items() if not k.startswith('__')}
        attacks = {  # Codex's trigger first
            'filter policy setter': lambda: constant.set_filter_policy('paper', (3, 0, 2, 1, 4), tol=123),
            'weights setter': lambda: constant.set_selection_weights({'kinematic': -123}),
            'weights in place': lambda: constant.get_selection_weights().update(kinematic=-123),
            'switch': lambda: constant.set_filter_policy('paper', (3, 0, 2, 1, 4), switch={5: (0, 1, 2, 3, 4)}),
            'friction': lambda: constant.set_mu(0.1),
            'controller hook': lambda: constant.set_filter_controller(lambda p, c: p),
            'getter rebound': lambda: setattr(constant, 'get_filter_policy', lambda step=None: PAPER),
            'new global': lambda: setattr(constant, 'EXTRA_SETTING', 1)}
        for name, attack in attacks.items():
            def controller(p, c):
                attack()
                return p
            with self.assertRaisesRegex(RuntimeError, 'global settings', msg=name):
                controlled_policy(constant.get_filter_policy(), controller, context())
            for k in set(vars(constant)) - set(saved) - {k for k in vars(constant) if k.startswith('__')}:
                delattr(constant, k)
            for k, v in saved.items():
                setattr(constant, k, copy.deepcopy(v) if k.isupper() else v)
        self.assertIs(controlled_policy(constant.get_filter_policy(), lambda p, c: p, context()), constant.get_filter_policy())

    def test_s6_orders_equal_to_the_base_are_type_checked_too(self):
        for bad in ({'order': (3.0, 0.0, 2.0, 1.0, 4.0)}, {'order': [3.0, 0.0, 2.0, 1.0, 4.0]},
                    {'order': tuple(np.array([3, 0, 2, 1, 4]))}, {'tol': np.float64(1e-9)}, {'policy': np.str_('paper')}):
            with self.assertRaises(ValueError, msg=bad):
                controlled_policy(PAPER, lambda p, c: {**p, **bad}, context())
        self.assertIs(controlled_policy(PAPER, lambda p, c: {**p, 'order': [3, 0, 2, 1, 4]}, context()), PAPER)

    def test_r2_controller_gets_detached_read_only_inputs(self):
        real = two_outcomes()

        def tamper(p, c):
            c['all_bricks'][0].id, c['all_bricks'][0].width = 99, 7
            c['all_bricks'][1].matrix[0, 0] = 3
            c['features_raw'] = np.zeros((3, 4))
            return p
        controlled_policy(PAPER, tamper, real)
        self.assertEqual((real['all_bricks'][0].id, real['all_bricks'][0].width, real['all_bricks'][1].matrix[0, 0]), (5, 2, 0))
        self.assertEqual(real['features_raw'].shape, (2, 6))

        def write(p, c):
            c['features_raw'][0, 0] = 9
            return p
        with self.assertRaises(ValueError):
            controlled_policy(PAPER, write, real)
        self.assertEqual(real['features_raw'][0, 0], 1)

    def test_r2_digest_sees_shape_dtype_and_brick_metadata(self):
        a = two_outcomes()
        digest = selection._digest(a)
        for change in (lambda c: c.update(features_raw=c['features_raw'].reshape(3, 4)),
                       lambda c: c.update(features_raw=c['features_raw'].astype(np.float32)),
                       lambda c: setattr(c['all_bricks'][0], 'id', 99),
                       lambda c: setattr(c['all_bricks'][0], 'width', 7)):
            b = two_outcomes()
            change(b)
            self.assertNotEqual(selection._digest(b), digest)

    def test_global_rng_must_stay_untouched(self):
        def touch_np(p, c):
            np.random.rand()
            return p

        def touch_py(p, c):
            random.random()
            return p
        for bad in (touch_np, touch_py):
            with self.assertRaisesRegex(RuntimeError, 'RNG'):
                controlled_policy(PAPER, bad, context())


class Staging(unittest.TestCase):
    def test_controller_lines_only_when_given(self):
        for case, (script, _) in run_case.CASES.items():
            source = (REPO / script).read_text(encoding='utf-8')
            policy = {'filter': PAPER, 'run_id': 'x'}
            self.assertEqual(run_case.stage(source, policy), run_case.stage(source, {**policy, 'controller': None}))
            staged = run_case.stage(source, {**policy, 'controller': {'kind': 'null'}})
            added = set(staged.splitlines()) - set(run_case.stage(source, policy).splitlines())
            self.assertEqual(len(added), 4, case)
            self.assertIn("    set_filter_controller(make_controller({'kind': 'null'}))", added)

    def test_r8_launcher_rejects_bad_controller_specs_before_running(self):
        for spec in ('{"kind": "random", "seed": null}', '{"kind": "random"}', '{"kind": "null", "x": 1}',
                     '{"kind": "react", "llm": {"kind": "api", "model": "m"}}'):
            with self.assertRaises(SystemExit, msg=spec):
                run_case.main(['thangavelu-noise80', 'never-created', '--policy', 'paper', '--controller', spec])
        self.assertFalse((run_case.RUNS / 'never-created').exists())


class SpecsAndSchema(unittest.TestCase):
    def test_r8_random_needs_a_non_negative_int_seed(self):
        for seed in (None, 1.5, True, '1', -1):
            with self.assertRaises(ValueError, msg=seed):
                controllers.make_controller({'kind': 'random', 'seed': seed})
        with self.assertRaises(ValueError):
            controllers.make_controller({'kind': 'random', 'seed': 1, 'sed': 2})
        self.assertEqual(controllers.make_controller({'kind': 'random', 'seed': 0}).spec(), {'kind': 'random', 'seed': 0})

    def test_r7_orders_are_never_coerced(self):
        self.assertEqual(controllers.parse_order('3-0-2-1-4'), (3, 0, 2, 1, 4))
        for bad in ([0.9, 1.9, 2.9, 3.9, 4.9], ['0', '1', '2', '3', '4'], '0.9-1.9-2.9-3.9-4.9', '3,0,2,1,4',
                    ' 3-0-2-1-4', '3-0-2-1-4-', '0-0-1-2-3', (True, 0, 2, 3, 4), None, 30214):
            self.assertIsNone(controllers.parse_order(bad), bad)

    def test_r7_reply_schema_rejects_malformed_actions(self):
        tools = controllers.TOOLS
        ok = {'thought': 't', 'tool': 'commit_order', 'arguments': {'order': '3-0-2-1-4', 'rationale': 'r'}}
        self.assertIsNone(controllers.reply_problem(ok, tools))
        self.assertIsNone(controllers.reply_problem({'tool': 'list_order_outcomes'}, tools))
        bad = {'not an object': None,
               'arguments must be strings': {**ok, 'arguments': {'order': [0.9, 1.9, 2.9, 3.9, 4.9], 'rationale': 'r'}},
               "order must be 'a-b-c-d-e'": {**ok, 'arguments': {'order': '0.9-1.9-2.9-3.9-4.9', 'rationale': 'r'}},
               'missing arguments': {**ok, 'arguments': {'order': '3-0-2-1-4'}},
               'arguments must be an object': {**ok, 'arguments': ['3-0-2-1-4', 'r']},
               'rationale must not be empty': {**ok, 'arguments': {'order': '3-0-2-1-4', 'rationale': ' '}},
               'unexpected arguments': {**ok, 'arguments': {'order': '3-0-2-1-4', 'rationale': 'r', 'stone': 'S1'}},
               "got 'keep_paper'": {'tool': 'explain_order', 'arguments': {'order': 'keep_paper'}},
               'unknown tool': {'tool': 'place_stone', 'arguments': {}},
               'unexpected reply fields': {**ok, 'stone': 'S1'},
               'thought must be a string': {**ok, 'thought': 3}}
        for reason, reply in bad.items():
            problem = controllers.reply_problem(None if reason == 'not an object' else reply, tools)
            self.assertIn('reply must be an object' if reason == 'not an object' else reason, problem or '', reason)


class PlacementGeometry(unittest.TestCase):
    def test_height_and_trapped_void(self):
        on_ground, floating = np.zeros((10, 8), bool), np.zeros((10, 8), bool)
        on_ground[2:4, 2:4] = True
        floating[4:6, 2:4] = True
        g = controllers.placement_geometry(wall(), on_ground)
        self.assertEqual((g['d_h_max'], g['d_void_under_skyline_px'], g['stone_px']), (2, 0, 4))
        f = controllers.placement_geometry(wall(), floating)
        self.assertEqual((f['d_h_max'], f['d_void_under_skyline_px']), (4, 4))

    def test_live_observation_pastes_the_candidate(self):
        obs, _ = controllers.Observer()(PAPER, two_outcomes())
        self.assertEqual([(o['stone'], o['geometry']['d_h_max']) for o in obs['outcomes']], [('S1', 2), ('S2', 2)])
        self.assertEqual(obs['n_remaining'], 3)


class LiveDecisions(unittest.TestCase):
    """Controllers called the way build_wall calls them, on synthetic inputs."""

    def call(self, controller, phase='single'):
        records = []
        controller.sink = records.append
        policy = controlled_policy(PAPER, controller, two_outcomes(phase))
        return policy, records[-1]

    def test_r5_matched_random_keys_include_the_phase(self):
        controller = controllers.make_controller({'kind': 'random', 'seed': 3, 'steps': [[1, 'place', 0, 'left']]})
        _, probe = self.call(controller, 'probe')
        policy, place = self.call(controller, 'place')
        self.assertEqual((probe['source'], probe['final']['switched']), ('matched_random:not_matched', False))
        self.assertEqual((place['source'], place['final']['switched'], place['phase']), ('matched_random', True, 'place'))
        self.assertEqual(policy['order'], (1, 0, 2, 3, 4))  # the smallest order with GDS before NHR
        for steps in ([[1, 0, 'left']], [[1, 'place', 0, 'left'], [1, 'place', 0, 'left']]):
            with self.assertRaises(ValueError):
                controllers.make_controller({'kind': 'random', 'seed': 3, 'steps': steps})

    def test_s2_malformed_replies_of_any_json_type_fall_back_with_a_record(self):
        commit = {'order': '0-1-2-3-4', 'rationale': 'r'}
        replies = [{'tool': [], 'arguments': {}}, {'tool': {}, 'arguments': {}}, {'tool': None}, {'tool': 1}, {'tool': 1.5},
                   {'tool': True}, {'tool': ['commit_order'], 'arguments': commit}, {'tool': {'commit_order': 1}},
                   {'tool': 'commit_order', 'arguments': {1: 'x', 'order': '0-1-2-3-4', 'rationale': 'r'}},
                   {'tool': 'commit_order', 'arguments': {(1, 2): 'x'}},
                   {'tool': 'commit_order', 'arguments': {'order': None, 'rationale': 'r'}},
                   {'tool': 'commit_order', 'arguments': {'order': {'a': 1}, 'rationale': ['r']}},
                   {1: 'x', 'tool': 'commit_order', 'arguments': commit}, {(1, 2): 'x'},
                   {'tool': 'commit_order', 'arguments': commit, 'status': 'ok'},
                   {'thought': ['t'], 'tool': 'commit_order', 'arguments': commit}, [], 'commit_order', 3.5, None]
        for reply in replies:
            controller = react(lambda m, t, r=reply: copy.deepcopy(r))
            policy, record = self.call(controller)
            self.assertIs(policy, PAPER, reply)
            self.assertEqual(record['source'], 'fallback:no_valid_commit', reply)
            self.assertEqual([s['status'] for s in record['react']], ['rejected'] * 4, reply)
            self.assertTrue(all(s['feedback']['error'] for s in record['react']), reply)
            json.dumps(record)
        with mock.patch.object(controllers, 'reply_problem', side_effect=KeyError('validator bug')):
            policy, record = self.call(react(lambda m, t: {'tool': 'commit_order'}))
        self.assertEqual((record['source'], record['react'][0]['status']), ('fallback:no_valid_commit', 'rejected'))
        self.assertIn('unreadable reply', record['react'][0]['feedback']['error'])

    def test_s5_a_timed_out_call_is_abandoned_with_its_policy_and_does_not_hold_up_exit(self):
        # Inline: the call cannot be stopped, so the policy that made it is dropped (a worker would be killed)
        gate, calls, made = threading.Event(), [], []

        def llm(messages, tools):
            calls.append(len(calls))
            if len(calls) == 1:
                gate.wait(10)
            return {'tool': 'commit_order', 'arguments': {'order': '1-0-2-3-4', 'rationale': 'r'}}
        client = react(llm, llm_timeout=0.1, max_llm_failures=5)
        make = client.transport.make
        client.transport.make = lambda: made.append(1) or make()
        sources = [self.call(client)[1] for _ in range(2)]
        gate.set()
        self.assertEqual([r['source'] for r in sources], ['fallback:llm_timeout', 'llm'])
        self.assertEqual((sources[0]['react'][0]['status'], len(made), client.request_failures), ('timeout', 2, 1))
        self.assertNotIn('restart', sources[0])

        script = ("import sys, time; sys.path[:0] = [r'%s', r'%s']\n"
                  "import test_controllers as t, controllers\n"
                  "from Stonepacker2D.stack.selection import controlled_policy\n"
                  "c = t.react(lambda m, tools: time.sleep(30), llm_timeout=0.2)\n"
                  "c.sink = lambda r: print(r['source'])\n"
                  "controlled_policy(t.PAPER, c, t.two_outcomes())\n") % (REPO, REPO / 'research' / 'tests')
        start = time.monotonic()
        done = subprocess.run([sys.executable, '-B', '-c', script], cwd=REPO, capture_output=True, text=True, timeout=25)
        self.assertEqual(done.stdout.split(), ['fallback:llm_timeout'], done.stderr[-500:])
        self.assertLess(time.monotonic() - start, 15)  # the 30 s call does not keep the process alive

    def test_r6_llm_timeout_error_and_bad_replies_fall_back_and_are_logged(self):
        calls = []

        def slow(messages, tools):
            calls.append('slow')
            time.sleep(0.5)
            return {'tool': 'commit_order', 'arguments': {'order': '0-1-2-3-4', 'rationale': 'late'}}

        def broken(messages, tools):
            calls.append('broken')
            raise ConnectionError('api down')

        def silent(messages, tools):
            calls.append('silent')
            return None
        for llm, source, status in ((slow, 'fallback:llm_timeout', 'timeout'), (broken, 'fallback:llm_error', 'error'),
                                    (silent, 'fallback:no_valid_commit', 'rejected')):
            controller = react(llm, llm_timeout=0.1)
            policy, record = self.call(controller)
            self.assertIs(policy, PAPER)
            self.assertEqual((record['source'], record['final']['switched']), (source, False))
            self.assertEqual(record['react'][0]['status'], status)
            self.assertTrue(record['react'][0].get('error') or record['react'][0]['feedback']['error'])
        self.assertEqual(calls.count('silent'), 4)

        controller = react(broken, max_llm_failures=2)
        calls.clear()
        records = [self.call(controller)[1] for _ in range(3)]
        self.assertEqual([r['source'] for r in records], ['fallback:llm_error', 'fallback:llm_error', 'skip:llm_unavailable'])
        self.assertEqual((len(calls), controller.request_failures), (2, 2))
        self.assertNotIn('tool_table_sha256', records[2])  # a skipped decision sends no request

    def test_r7_rejected_replies_are_logged_with_their_reason(self):
        replies = iter([{'tool': 'commit_order', 'arguments': {'order': [0.9, 1.9, 2.9, 3.9, 4.9], 'rationale': 'r'}},
                        {'tool': 'commit_order', 'arguments': {'order': '0-1-2-3-4'}},
                        {'tool': 'commit_order', 'arguments': {'order': '0-1-2-3-4', 'rationale': 'ok'}}])
        controller = react(lambda m, t: next(replies))
        policy, record = self.call(controller)
        self.assertEqual([s['status'] for s in record['react']], ['rejected', 'rejected', 'ok'])
        self.assertIn('strings', record['react'][0]['feedback']['error'])
        self.assertIn('missing', record['react'][1]['feedback']['error'])
        self.assertEqual((record['source'], policy['order']), ('llm', (0, 1, 2, 3, 4)))


@unittest.skipUnless(WALL.exists(), 'existing run directories not present')
class OfflineControllers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = check_case.result_dirs(WALL)[0]
        cls.null = controllers.replay(cls.result, {'kind': 'null'})

    def test_null_keeps_paper_and_passes_the_trace_check(self):
        self.assertFalse(any(r['final']['switched'] for r in self.null))
        self.assertEqual({r['final']['order'] for r in self.null}, {'3-0-2-1-4'})
        report = controllers.check(self.result, self.null, {'kind': 'null'})
        self.assertTrue(report['ok'], report['problems'][:3])
        self.assertEqual((report['records'], report['decision_points']), (23, 14))

    def test_r4_check_recomputes_instead_of_trusting_records(self):
        def tampered(edit):
            records = json.loads(json.dumps(self.null))
            edit(records)
            return controllers.check(self.result, records, {'kind': 'null'})
        edits = [lambda rs: rs[5]['final'].update(order='0-1-2-3-4'),
                 lambda rs: rs[5]['final'].update(switched=True),
                 lambda rs: rs[5]['final'].update(stone_id=99, outcome=1),
                 lambda rs: rs[5].update(decision_point=not rs[5]['decision_point'], n_outcomes=9),
                 lambda rs: rs[5].update(controller={'kind': 'rule'}),
                 lambda rs: rs[5]['observation'].update(base_order='0-1-2-3-4'),
                 lambda rs: rs.pop()]
        for edit in edits:
            self.assertFalse(tampered(edit)['ok'])
        self.assertFalse(controllers.check(self.result, self.null, {'kind': 'random', 'seed': 1})['ok'])
        self.assertFalse(controllers.check(self.result, self.null)['ok'])  # this wall's receipt names no controller

    def test_observation_holds_no_ids_rng_or_outcomes_of_the_wall(self):
        for r in self.null:
            obs = r['observation']
            text = json.dumps(obs)
            for word in ('rng', 'selected', 'stone_id', 'inventory_ids', 'sampled_ids', 'lambda', 'placed'):
                self.assertNotIn(word, text)
            labels = {o['stone'] for o in obs['outcomes']} | {s['stone'] for s in obs['sampled']} | set(obs['memory']['stones'])
            self.assertTrue(all(re.fullmatch(r'S\d+', l) for l in labels))
            self.assertTrue(all(m['last_step'] < obs['step'] or (m['last_step'] == obs['step'] and obs['sample_trial'] > 0)
                                for m in obs['memory']['stones'].values()))
        self.assertEqual(self.null[0]['observation']['memory'], {'stones': {}, 'recent': []})

    def test_outcome_classes_are_the_distinct_picks_of_the_120_orders(self):
        import counterfactual
        poses = counterfactual.step_options(self.result, set()).groupby('step').apply(
            lambda g: len(g.groupby(['stone', 'rotation', 'x', 'y'])))
        n = {r['step']: r['n_outcomes'] for r in self.null}
        self.assertEqual({s: n[s] for s in poses.index}, poses.to_dict())

    def test_random_is_reproducible_and_seeded(self):
        a, b, c = (controllers.replay(self.result, {'kind': 'random', 'seed': s}) for s in (1, 1, 2))
        self.assertEqual(a, b)
        self.assertNotEqual([r['final']['order'] for r in a], [r['final']['order'] for r in c])

    def test_matched_random_switches_only_at_the_given_keys_and_never_to_paper(self):
        keys = [list(controllers.decision_key(r)) for r in self.null if r['decision_point']][::3]
        records = controllers.replay(self.result, {'kind': 'random', 'seed': 3, 'steps': keys})
        for r in records:
            matched = list(controllers.decision_key(r)) in keys
            self.assertEqual(r['final']['switched'], matched)
            if matched:
                self.assertFalse(r['observation']['outcomes'][r['final']['outcome']]['is_paper'])

    def test_rule_switches_follow_its_conditions(self):
        records = controllers.replay(self.result, {'kind': 'rule'})
        for r in (r for r in records if r['final']['switched']):
            obs, pick = r['observation'], r['observation']['outcomes'][r['final']['outcome']]
            self.assertIn(r['source'], ('rule:scarcity', 'rule:llc_drop'))
            if r['source'] == 'rule:scarcity':
                self.assertLessEqual(obs['n_remaining'], 6)
                self.assertLessEqual(next(s for s in obs['sampled'] if s['stone'] == pick['stone'])['feasible_rotations'], 1)

    def test_react_tool_flow_is_logged_and_commits_a_legal_order(self):
        records = controllers.replay(self.result, {'kind': 'react', 'llm': {'kind': 'mock', 'mode': 'max_llc'}})
        for r in (r for r in records if r['decision_point']):
            self.assertEqual([s['tool'] for s in r['react']], ['list_order_outcomes', 'explain_order', 'commit_order'])
            self.assertTrue(all(s['status'] == 'ok' and s['thought'] and 'feedback' in s for s in r['react']))
            self.assertEqual(r['react'][-1]['arguments']['order'], r['final']['order'])
            best = max(r['observation']['outcomes'], key=lambda o: (o['scores']['LLC'], -o['id']))
            self.assertEqual(r['final']['outcome'], best['id'])

    def test_invalid_actions_fall_back_to_paper(self):
        for mode, error in (('invalid', 'permutation'), ('stone', 'unexpected arguments')):
            records = controllers.replay(self.result, {'kind': 'react', 'llm': {'kind': 'mock', 'mode': mode}})
            for r in (r for r in records if r['decision_point']):
                self.assertEqual((r['source'], r['final']['switched'], len(r['react'])), ('fallback:no_valid_commit', False, 4))
                self.assertEqual(r['react'][0]['status'], 'rejected')
                self.assertIn(error, r['react'][0]['feedback']['error'])

    def test_wall_budget_falls_back_to_paper(self):
        records = controllers.replay(self.result, {'kind': 'react', 'llm': {'kind': 'mock', 'mode': 'max_llc'}, 'max_decisions': 2})
        sources = [r['source'] for r in records if r['decision_point']]
        self.assertEqual(sources[:2], ['llm', 'llm'])
        self.assertEqual(set(sources[2:]), {'skip:wall_budget'})

    def test_without_tool_feedback_only_commit_is_offered(self):
        seen = []

        def llm(messages, tools):
            seen.append(([t['name'] for t in tools], json.loads(messages[1]['content'])))
            return {'thought': 't', 'tool': 'commit_order', 'arguments': {'order': 'keep_paper', 'rationale': 't'}}
        controller = react(llm, tool_feedback=False)
        controller.sink = lambda r: None
        for meta, a in controllers.trace_events(self.result):
            if not seen:
                controller(PAPER, controllers.context_from_trace(meta, a))
        self.assertEqual(seen[0][0], ['commit_order'])
        self.assertNotIn('outcomes', seen[0][1])


@unittest.skipUnless(RANDOM_WALL.exists(), 'P1 random run not present')
class InterventionTrace(unittest.TestCase):
    """A wall built with the Random controller: its traced orders are applied orders, not the base."""

    @classmethod
    def setUpClass(cls):
        cls.result = check_case.result_dirs(RANDOM_WALL)[0]
        cls.base = controllers.base_policy(RANDOM_WALL)

    def test_r3_replay_refuses_and_rerun_starts_from_the_receipt_base(self):
        with self.assertRaisesRegex(ValueError, 'controller'):
            controllers.replay(self.result, {'kind': 'null'})
        null = controllers.rerun(self.result, {'kind': 'null'}, self.base)
        self.assertEqual({r['final']['order'] for r in null}, {'3-0-2-1-4'})
        self.assertEqual({r['observation']['base_order'] for r in null}, {'3-0-2-1-4'})
        traced = {tuple(m['policy']['order']) for m, _ in controllers.trace_events(self.result)}
        self.assertGreater(len(traced), 1)

    def test_r4_rerun_records_pass_and_tampered_ones_fail(self):
        records = controllers.rerun(self.result, {'kind': 'random', 'seed': 1}, self.base)
        report = controllers.check(self.result, records)
        self.assertTrue(report['ok'], report['problems'][:3])
        self.assertGreater(report['switches'], 0)
        for edit in (lambda r: r['final'].update(switched=False), lambda r: r['final'].update(stone_id=99),
                     lambda r: r.update(controller={'kind': 'null'})):
            bad = json.loads(json.dumps(records))
            for r in bad:
                edit(r)
            self.assertFalse(controllers.check(self.result, bad)['ok'])
        self.assertFalse(controllers.check(self.result, controllers.rerun(self.result, {'kind': 'null'}, self.base))['ok'])

    def test_s3_missing_or_inconsistent_trace_fails_the_check(self):
        spec = {'kind': 'random', 'seed': 1}
        events = list(controllers.trace_events(self.result))

        def report(evs, stopped=False):
            with mock.patch.object(controllers, 'trace_events', lambda result: iter(copy.deepcopy(evs))):
                return controllers.check(self.result, controllers.rerun(self.result, spec, self.base)[:len(evs)],
                                         stopped=stopped)

        def edit(k, **arrays):
            return [(m, {**a, **arrays}) if m['event'] == k else (m, a) for m, a in events]
        self.assertTrue(report(events)['ok'])
        first = events[0][1]
        other = int(first['inventory_ids'][0])
        cases = {  # Codex's two triggers first
            'last record and event removed': events[:-1],
            'stone ids of event 0 rewritten': edit(0, stone_id=np.full_like(first['stone_id'], -987654)),
            'sampled stone swapped consistently': edit(0, stone_id=np.full_like(first['stone_id'], other),
                                                       sampled_ids=np.array([other])),
            'middle event removed': events[:5] + events[6:],
            'rotation labels changed': edit(0, rotation=first['rotation'][::-1].copy()),
            'world pose changed': edit(0, x_world=first['x_world'] + 1)}
        for name, evs in cases.items():
            self.assertFalse(report(evs)['ok'], name)
        self.assertFalse(report(events[:-1], stopped=True)['ok'])  # its commit still points at the removed event


def null_records():
    """NULL_WALL's controller records as the current code writes them (the wall predates
    tool_table_sha256), with the live outcome geometry of its own records."""
    src = check_case.result_dirs(NULL_WALL)[0]
    geometry = controllers.LiveGeometry(jsonl(src / 'trace' / 'controller.jsonl'))
    return [json.dumps(r) for r in controllers.rerun(src, {'kind': 'null'}, controllers.base_policy(NULL_WALL), geometry=geometry)]


@unittest.skipUnless(NULL_WALL.exists(), 'P1-r2 Null run not present')
class StoppedRuns(unittest.TestCase):
    """run_case.verified_trace on copies of a Null wall, cut the way --placements N leaves a run."""

    def copy_run(self, dst, events=None, records=None, committed=None):
        """dst: receipt, decisions, commits and the first `events` trace events of NULL_WALL; records:
        how many controller records to keep (default one per kept event), or a list of them;
        committed: only attempts whose events all come before it have their commit."""
        src = check_case.result_dirs(NULL_WALL)[0]
        out = dst / src.relative_to(NULL_WALL)
        (out / 'trace').mkdir(parents=True)
        shutil.copy2(NULL_WALL / 'receipt.json', dst / 'receipt.json')
        n = len(list((src / 'trace').glob('e*.npz'))) if events is None else events
        for k in range(n):
            shutil.copy2(src / 'trace' / f'e{k:05d}.npz', out / 'trace' / f'e{k:05d}.npz')
        commits = [l for l in (src / 'trace' / 'commits.jsonl').read_text(encoding='utf-8').splitlines()
                   if all(s['event'] < (n if committed is None else committed) for s in json.loads(l)['sides'].values())]
        placed = sum(json.loads(l)['outcome'] == 'placed' for l in commits)
        (out / 'trace' / 'commits.jsonl').write_text(''.join(l + '\n' for l in commits), encoding='utf-8')
        rows = (src / 'decisions.txt').read_text(encoding='utf-8').splitlines()[:placed]
        (out / 'decisions.txt').write_text(''.join(l + '\n' for l in rows), encoding='utf-8')
        lines = null_records()
        keep = lines[:n] if records is None else lines[:records] if isinstance(records, int) else records
        if keep is not False:
            (out / 'trace' / 'controller.jsonl').write_text(''.join(l + '\n' for l in keep), encoding='utf-8')
        return dst

    def test_s4_stopped_controller_runs_are_checked_too(self):
        args, stop = SimpleNamespace(no_trace=False, controller={'kind': 'null'}), {'verified': True}
        lines = null_records()
        wrong = json.loads(lines[3])
        wrong['final']['switched'] = True
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            full = run_case.verified_trace(self.copy_run(tmp / 'full'), args, 'PROCESS_PASS', None)
            self.assertTrue(full['ok'], full['problems'])
            stopped = run_case.verified_trace(self.copy_run(tmp / 'stopped', 10, 11), args, 'STOPPED_AFTER_PLACEMENTS', stop)
            self.assertTrue(stopped['ok'], stopped['problems'])
            self.assertEqual(stopped['controller'][0]['unverified_trailing_records'], 1)
            for name, records in (('a record missing', lines[:3] + lines[4:10]), ('a wrong record', lines[:3] + [json.dumps(wrong)] + lines[4:10]),
                                  ('no records', False), ('two extra records', 12)):
                trace = run_case.verified_trace(self.copy_run(tmp / name, 10, records), args, 'STOPPED_AFTER_PLACEMENTS', stop)
                self.assertFalse(trace['ok'], name)
                self.assertEqual(run_case.final_status('STOPPED_AFTER_PLACEMENTS', trace, stop), 'INCOMPLETE_TRACE', name)
            unfinished = self.copy_run(tmp / 'unfinished', 10, committed=9)  # event 9 has no commit yet
            self.assertTrue(run_case.verified_trace(unfinished, args, 'STOPPED_AFTER_PLACEMENTS', stop)['ok'])
            self.assertFalse(controllers.check(check_case.result_dirs(unfinished)[0], stopped=False)['ok'])

    def test_t9_a_trailing_record_must_be_the_next_call_of_the_receipt_controller(self):
        args, stop = SimpleNamespace(no_trace=False, controller={'kind': 'null'}), {'verified': True}
        lines = null_records()
        nxt = json.loads(lines[10])
        bad = {'wrong event number': {**nxt, 'event': 99}, 'another controller': {**nxt, 'controller': {'kind': 'random', 'seed': 1}},
               'observation of another selection': {**nxt, 'step': nxt['step'] + 5},
               'no final order': {k: v for k, v in nxt.items() if k != 'final'}, 'not a record': [1, 2]}
        with tempfile.TemporaryDirectory() as tmp:
            for name, record in bad.items():
                run = self.copy_run(Path(tmp) / name, 10, lines[:10] + [json.dumps(record)])
                trace = run_case.verified_trace(run, args, 'STOPPED_AFTER_PLACEMENTS', stop)
                self.assertFalse(trace['ok'], name)
                self.assertTrue(any('trailing record' in p for p in trace['problems']), name)


TWO_SIDED_WALL = REPO / 'research' / 'runs' / 'validation-pr-paper-right-01'


def copy_wall(run, dst):
    """receipt, decisions.txt and trace of a run's (first) result directory; returns the copy's result directory."""
    src = check_case.result_dirs(run)[0]
    out = dst / src.relative_to(run)
    shutil.copytree(src / 'trace', out / 'trace')
    shutil.copy2(src / 'decisions.txt', out / 'decisions.txt')
    shutil.copy2(run / 'receipt.json', dst / 'receipt.json')
    return out


def jsonl(path, items=None):
    if items is None:
        return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()]
    path.write_text(''.join(json.dumps(i) + '\n' for i in items), encoding='utf-8')


def cut(result, events, commits, rows):
    """Keep the first `events` trace events, `commits` commits and `rows` decisions.txt lines."""
    for p in selection.published_events(result / 'trace'):
        if int(Path(p).stem[1:]) >= events:
            Path(p).unlink()
    jsonl(result / 'trace' / 'commits.jsonl', jsonl(result / 'trace' / 'commits.jsonl')[:commits])
    lines = (result / 'decisions.txt').read_text(encoding='utf-8').splitlines()[:rows]
    (result / 'decisions.txt').write_text(''.join(l + '\n' for l in lines), encoding='utf-8')


def edit_event(result, k, **arrays):
    meta, a = selection.load_trace(result / 'trace' / f'e{k:05d}.npz')
    selection.write_trace(str(result / 'trace'), meta, **{**a, **arrays})


@unittest.skipUnless(NULL_WALL.exists() and TWO_SIDED_WALL.exists(), 'P1-r2 Null run or two-sided validation run not present')
class TraceChain(unittest.TestCase):
    """check_case.chain_problems (run_case.check_commits, controllers.check): the third review's
    counterexamples (T1..T8; T9 is in StoppedRuns) on copies of real walls, each consistent but in one link."""

    def problems(self, run, stopped=False):
        return run_case.check_commits(run, stopped)

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def wall(self, name, run=NULL_WALL):
        return copy_wall(run, self.tmp / name)

    def test_unchanged_walls_pass(self):
        self.wall('null')
        self.assertEqual(self.problems(self.tmp / 'null'), [])
        self.wall('two', TWO_SIDED_WALL)
        self.assertEqual(self.problems(self.tmp / 'two', stopped=True), [])

    def test_t1_a_commit_must_hold_what_its_event_selects(self):
        # Codex: commit index 6 -> 7 with stone, pose and decisions.txt all synced to candidate 7
        result = self.wall('index')
        commits, rows = jsonl(result / 'trace' / 'commits.jsonl'), (result / 'decisions.txt').read_text(encoding='utf-8').splitlines()
        j = next(i for i, c in enumerate(commits) if c['outcome'] == 'placed')
        c, side = commits[j], commits[j]['sides'][commits[j]['chosen_side']]
        a = selection.load_trace(result / 'trace' / f"e{side['event']:05d}.npz")[1]
        n = len(a['sampled_ids'])  # the same stone at another feasible rotation: inventory stays consistent
        other = next(k for k in range(side['index'] % n, len(a['x']), n) if k != side['index'] and a['x'][k] >= 0)
        pose = check_case.candidate_pose(a, other)
        side['index'] = other
        c.update(stone_id=pose[0], x_world=pose[1], y=pose[2], rotation_world=pose[3], rotation_label=pose[4])
        rows[0] = ';'.join(map(str, (c['step'], pose[0], pose[1], pose[2], pose[4])))
        jsonl(result / 'trace' / 'commits.jsonl', commits)
        (result / 'decisions.txt').write_text(''.join(l + '\n' for l in rows), encoding='utf-8')
        self.assertTrue(any('commit holds' in p for p in self.problems(self.tmp / 'index')))

    def test_t2_a_stopped_run_ends_only_in_a_prefix_of_the_write_order(self):
        # one event per attempt in the Null wall; attempt j is commit j and event j
        commits = jsonl(check_case.result_dirs(NULL_WALL)[0] / 'trace' / 'commits.jsonl')
        j = [i for i, c in enumerate(commits) if c['outcome'] == 'placed'][4]
        before = sum(c['outcome'] == 'placed' for c in commits[:j])
        cases = {  # (events, commits, decisions rows): ok when stopped
            'event without its commit': ((j + 1, j, before), True),
            'event and decision without the commit': ((j + 1, j, before + 1), True),
            'decision whose event is hidden (Codex)': ((j, j, before + 1), False),
            'two attempts without commits': ((j + 1, j - 1, before), False),
            'decision of an earlier attempt again': ((j + 1, j, before + 1), 'wrong row')}
        for name, ((events, kept, rows), ok) in cases.items():
            result = self.wall(name)
            cut(result, events, kept, rows)
            if ok == 'wrong row':
                lines = (result / 'decisions.txt').read_text(encoding='utf-8').splitlines()
                lines[-1] = lines[-2]
                (result / 'decisions.txt').write_text(''.join(l + '\n' for l in lines), encoding='utf-8')
            self.assertEqual(self.problems(self.tmp / name, stopped=True) == [], ok is True, name)
            self.assertNotEqual(self.problems(self.tmp / name), [], name)  # never for a finished run

    def test_t3_stone_ids_must_be_distinct_ints_and_nb_stones(self):
        result = self.wall('ids')
        a = selection.load_trace(result / 'trace' / 'e00003.npz')[1]
        inv, sampled = a['inventory_ids'], a['sampled_ids']
        cases = {'duplicate remaining id (Codex)': {'inventory_ids': np.append(inv, inv[0])},
                 'remaining id also sampled': {'inventory_ids': np.append(inv, sampled[0])},
                 'float ids': {'inventory_ids': inv.astype(float)},
                 'fewer sampled stones than nb_stones': {'sampled_ids': sampled[:-1],
                                                         'stone_id': a['stone_id'][:len(a['stone_id']) // len(sampled) * (len(sampled) - 1)]}}
        for name, arrays in cases.items():
            shutil.copy2(check_case.result_dirs(NULL_WALL)[0] / 'trace' / 'e00003.npz', result / 'trace' / 'e00003.npz')
            edit_event(result, 3, **arrays)
            self.assertNotEqual(self.problems(self.tmp / 'ids'), [], name)

    def test_t4_unpublished_events_are_read_by_no_one_and_tolerated_only_at_a_stop(self):
        result = self.wall('tmp')
        n = len(selection.published_events(result / 'trace'))
        shutil.copy2(result / 'trace' / 'e00000.npz', result / 'trace' / f'e{n:05d}.tmp.npz')
        self.assertEqual(len(list(controllers.trace_events(result))), n)
        self.assertEqual(len(check_case.load_result(result)[0]), n)
        self.assertEqual(run_case.check_trace(self.tmp / 'tmp')['files'], n)
        self.assertTrue(any('unpublished' in p for p in self.problems(self.tmp / 'tmp')))
        self.assertEqual(self.problems(self.tmp / 'tmp', stopped=True), [])
        shutil.move(result / 'trace' / f'e{n:05d}.tmp.npz', result / 'trace' / 'e00001.tmp.npz')
        self.assertNotEqual(self.problems(self.tmp / 'tmp', stopped=True), [])  # not the next event

    def test_t5_the_chosen_side_follows_build_walls_rule(self):
        result = self.wall('side', TWO_SIDED_WALL)
        commits = jsonl(result / 'trace' / 'commits.jsonl')
        j, c = next((i, c) for i, c in enumerate(commits) if not c['side_tie'] and all(s['ok'] for s in c['sides'].values()))
        flipped = {'left': 'right', 'right': 'left'}[c['chosen_side']]
        a = selection.load_trace(result / 'trace' / f"e{c['sides'][flipped]['event']:05d}.npz")[1]
        pose = check_case.candidate_pose(a, c['sides'][flipped]['index'])
        c.update(chosen_side=flipped, stone_id=pose[0], x_world=pose[1], y=pose[2], rotation_world=pose[3], rotation_label=pose[4])
        placed = sum(x['outcome'] == 'placed' for x in commits[:j])
        lines = (result / 'decisions.txt').read_text(encoding='utf-8').splitlines()
        if c['outcome'] == 'placed':
            lines[placed] = ';'.join(map(str, (c['step'], pose[0], pose[1], pose[2], pose[4])))
        (result / 'decisions.txt').write_text(''.join(l + '\n' for l in lines), encoding='utf-8')
        jsonl(result / 'trace' / 'commits.jsonl', commits)
        self.assertTrue(any('chosen side' in p for p in self.problems(self.tmp / 'side', stopped=True)))
        c.update(chosen_side={'left': 'right', 'right': 'left'}[flipped], side_tie=True)
        jsonl(result / 'trace' / 'commits.jsonl', commits)
        self.assertTrue(any('chosen side' in p for p in self.problems(self.tmp / 'side', stopped=True)))

    def test_t6_a_commit_side_is_its_own_selection(self):
        cases = {'sampled stones of another attempt': lambda c, cs: c.update(sampled_ids=cs[0]['sampled_ids'][::-1]),
                 'event of another commit': lambda c, cs: c['sides']['left'].update(event=cs[0]['sides']['left']['event']),
                 'side missing': lambda c, cs: c.update(sides={}),
                 'wrong weighted GDS': lambda c, cs: c['sides']['left'].update(gds_weighted=c['sides']['left']['gds_weighted'] + 1)}
        for name, change in cases.items():
            result = self.wall(name)
            commits = jsonl(result / 'trace' / 'commits.jsonl')
            change(commits[3], commits)
            jsonl(result / 'trace' / 'commits.jsonl', commits)
            self.assertNotEqual(self.problems(self.tmp / name), [], name)

    def test_t7_decisions_txt_lists_exactly_the_placed_commits(self):
        cases = {'wrong x': lambda rows: rows.__setitem__(1, ';'.join(rows[1].split(';')[:2] + ['999'] + rows[1].split(';')[3:])),
                 'line missing': lambda rows: rows.pop(), 'extra line': lambda rows: rows.append(rows[-1])}
        for name, change in cases.items():
            result = self.wall(name)
            rows = (result / 'decisions.txt').read_text(encoding='utf-8').splitlines()
            change(rows)
            (result / 'decisions.txt').write_text(''.join(l + '\n' for l in rows), encoding='utf-8')
            self.assertTrue(any('decisions.txt' in p for p in self.problems(self.tmp / name)), name)

    def test_t8_the_base_policy_itself_is_validated(self):
        ok = lambda p, c: p
        for base in ({**PAPER, 'order': [3, 0, 2, 1, 4]}, {**PAPER, 'order': (3, 0, 2, 1, 4.0)}, {**PAPER, 'order': (3, 0, 2, 1, 1)},
                     {**PAPER, 'tol': 0}, {**PAPER, 'switch': {}}, {'policy': 'paper', 'order': None}):
            with self.assertRaises(ValueError, msg=base):
                controlled_policy(base, ok, context())
        self.assertIs(controlled_policy(PAPER, ok, context()), PAPER)


@unittest.skipUnless(WALL.exists(), 'existing run directories not present')
class ProcessIsolation(unittest.TestCase):
    """controller_worker.ProcessTransport: the policy in a spawned worker, JSON bytes only (step 3)."""

    def call(self, client):
        records = []
        client.sink = records.append
        np_state, py_state = np.random.get_state(), random.getstate()
        policy = controlled_policy(PAPER, client, two_outcomes())  # raises if RNG, settings or inputs changed
        self.assertTrue(np.array_equal(np_state[1], np.random.get_state()[1]) and py_state == random.getstate())
        json.dumps(records[-1])
        return policy, records[-1]

    def test_inline_and_process_give_the_same_records(self):
        result = check_case.result_dirs(WALL)[0]
        base = controllers.base_policy(WALL)
        llm = controllers.rerun(result, {'kind': 'react', 'llm': {'kind': 'mock', 'mode': 'max_llc'}}, base)
        steps = [list(controllers.decision_key(r)) for r in llm if r['final']['switched']]
        for spec in controllers.OFFLINE[:4] + [{'kind': 'random', 'seed': 1, 'steps': steps}]:
            inline, process = (controllers.rerun(result, spec, base, t) for t in ('inline', 'process'))
            self.assertEqual(controllers.comparable(inline), controllers.comparable(process), spec)
            self.assertTrue(sum(r['final']['switched'] for r in process) or spec['kind'] == 'null', spec)
        self.assertFalse(multiprocessing.active_children())

    def test_a_timed_out_worker_is_killed_and_reaped_and_restarted_lazily(self):
        client, seen = process_client({'kind': 'null'}, timeout=1.0, bad='hang'), []
        close = client.transport.close
        client.transport.close = lambda: (seen.append(client.transport.process), close())
        for k in range(2):
            start = time.monotonic()
            policy, record = self.call(client)
            self.assertIs(policy, PAPER)
            self.assertEqual((record['source'], record['final']['order']), ('fallback:worker_timeout', '3-0-2-1-4'))
            self.assertLess(time.monotonic() - start, 30)
            self.assertTrue(seen[-1] is not None and not seen[-1].is_alive() and seen[-1].exitcode is not None)
            self.assertEqual((client.transport.process, client.transport.started, client.request_failures), (None, k + 1, k + 1))
        self.assertTrue(reaped(client))

    def test_bad_messages_are_recorded_safely_and_fall_back(self):
        cases = {'exit': ('worker_error', 'code 3'), 'not_ready': ('worker_error', 'did not start'),
                 'error': ('worker_error', 'boom'), 'garbage': ('worker_protocol_error', 'not JSON'),
                 'deep': ('worker_protocol_error', 'not JSON'), 'huge': ('worker_protocol_error', 'over'),
                 'shape': ('worker_protocol_error', 'neither'), 'list_order': ('worker_protocol_error', 'order'),
                 'foreign_source': ('worker_protocol_error', 'source')}
        for bad, (status, text) in cases.items():
            client = process_client({'kind': 'null'}, bad=bad)
            policy, record = self.call(client)
            self.assertIs(policy, PAPER, bad)
            self.assertEqual(record['source'], f'fallback:{status}', bad)
            self.assertIn(text, record['error'], bad)
            self.assertLess(len(record['error']), 400, bad)
            self.assertTrue(reaped(client), bad)

    def test_llm_hangs_and_crashes_in_the_worker_fall_back_and_count_as_failures(self):
        spec = {'kind': 'react', 'llm': {'kind': 'mock', 'mode': 'hang@1'}, 'llm_timeout': 1.0, 'max_llm_failures': 2}
        client = controllers.make_controller(spec)
        _, timed_out = self.call(client)
        self.assertEqual((timed_out['source'], timed_out['react'][0]['status']), ('fallback:llm_timeout', 'timeout'))
        self.assertIsNone(client.transport.process)  # killed with the abandoned call in it
        client.spec_['llm'] = {'kind': 'mock', 'mode': 'crash@1'}  # the record names the spec; the worker is new
        client.transport.spec = json.dumps(client.spec_)
        _, crashed = self.call(client)
        self.assertEqual(crashed['source'], 'fallback:worker_error')
        self.assertIn('code 3', crashed['error'])
        _, skipped = self.call(client)
        self.assertEqual((skipped['source'], client.transport.started, client.request_failures), ('skip:llm_unavailable', 2, 2))
        self.assertTrue(reaped(client))

    def test_a_worker_failure_of_a_deterministic_controller_fails_acceptance(self):
        result = check_case.result_dirs(WALL)[0]
        base, records = controllers.base_policy(WALL), []
        client = process_client({'kind': 'random', 'seed': 1}, bad='garbage', fail_at=4)
        client.sink = records.append
        for meta, a in controllers.trace_events(result):
            client(types.MappingProxyType(dict(base)), controllers.context_from_trace(meta, a))
        self.assertTrue(reaped(client))
        self.assertIn('fallback:worker_protocol_error', [r['source'] for r in records])
        report = controllers.check(result, records, {'kind': 'random', 'seed': 1})
        self.assertFalse(report['ok'])
        self.assertTrue(any('deterministic controller failed' in p for p in report['problems']))

    def test_the_worker_does_not_outlive_its_main_process(self):
        script = ("import sys; sys.path[:0] = [r'%s', r'%s']\n"
                  "import test_controllers as t\n"
                  "from Stonepacker2D.stack.selection import controlled_policy\n"
                  "if __name__ == '__main__':\n"
                  "    idle = t.process_client({'kind': 'null'})\n"
                  "    busy = t.process_client({'kind': 'null'}, timeout=600, bad='hang', fail_at=2)\n"
                  "    for c in (idle, busy):\n"
                  "        c.sink = lambda r: None\n"
                  "        controlled_policy(t.PAPER, c, t.two_outcomes())\n"
                  "    busy.transport.conn.send_bytes(b'{}')  # the worker now hangs in its second request\n"
                  "    print(idle.transport.process.pid, busy.transport.process.pid, flush=True)\n"
                  ) % (REPO, REPO / 'research' / 'tests')
        done = subprocess.run([sys.executable, '-B', '-c', script], cwd=REPO, capture_output=True, text=True, timeout=120)
        pids = done.stdout.split()
        self.assertEqual(len(pids), 2, done.stderr[-800:])
        time.sleep(1)
        listed = subprocess.run(['tasklist', '/FO', 'CSV', '/NH'], capture_output=True, text=True).stdout
        for pid in pids:
            self.assertNotIn(f'"{pid}"', listed)


class Evil:
    """A reply object whose every dunder raises: it must never be called."""
    def __repr__(self):
        raise RuntimeError('repr called')
    __str__ = __eq__ = __hash__ = __len__ = __iter__ = __repr__


@unittest.skipUnless(WALL.exists(), 'existing run directories not present')
class RecordAndReplay(unittest.TestCase):
    """Transcripts of every LLM call and their replay in check(); no LLM or API is called (step 4)."""

    KEEP = {'kind': 'react', 'llm': {'kind': 'mock', 'mode': 'keep'}}
    MAX_LLC = {'kind': 'react', 'llm': {'kind': 'mock', 'mode': 'max_llc'}}

    @classmethod
    def setUpClass(cls):
        cls.result, cls.base = check_case.result_dirs(WALL)[0], controllers.base_policy(WALL)
        cls.keep = controllers.rerun(cls.result, cls.KEEP, cls.base)  # keep_paper: consistent with this base wall
        cls.max_llc = controllers.rerun(cls.result, cls.MAX_LLC, cls.base)

    def replayed(self, records, spec):
        llm = controllers.ReplayLLM(records)
        worker = {k: controllers.full_spec(spec)[k] for k in controllers.WORKER_KEYS['react']}
        transport = controllers.ReplayedFailures(lambda: controllers.ReActPolicy(llm, **worker), records)
        with mock.patch.object(controllers.MockLLM, '__call__', side_effect=AssertionError('an LLM was called')):
            return controllers.rerun(self.result, spec, self.base, transport), llm.diverged

    def test_every_call_has_a_transcript_whose_hash_is_its_request(self):
        for r in (r for r in self.max_llc if r['decision_point']):
            self.assertEqual(len(r['transcripts']), len(r['react']))
            for k, t in enumerate(r['transcripts']):
                self.assertEqual((t['call'], t['status'], t['request']['model']), (k, 'ok', 'mock:max_llc'))
                self.assertEqual(t['request_sha256'], controllers.request_sha256(t['request']))
                self.assertEqual(len(t['request']['messages']), 2 + 2 * k)
                self.assertEqual({**r['react'][k]}, {**t['response']['parsed'], 'status': 'ok', 'feedback': r['react'][k]['feedback']})

    def test_replay_reproduces_the_records_without_any_llm(self):
        for records, spec in ((self.max_llc, self.MAX_LLC), (self.keep, self.KEEP)):
            replayed, diverged = self.replayed(records, spec)
            self.assertEqual(diverged, [])
            self.assertEqual(controllers.comparable(replayed), controllers.comparable(records))
        with mock.patch.object(controllers.MockLLM, '__call__', side_effect=AssertionError('an LLM was called')):
            report = controllers.check(self.result, self.keep, self.KEEP)
        self.assertTrue(report['ok'], report['problems'][:3])

    def test_tampered_transcripts_or_requests_fail_the_check(self):
        k = next(i for i, r in enumerate(self.keep) if r['decision_point'])

        def tampered(edit):
            records = copy.deepcopy(self.keep)
            edit(records[k])
            with mock.patch.object(controllers.MockLLM, '__call__', side_effect=AssertionError('an LLM was called')):
                return controllers.check(self.result, records, self.KEEP)
        edits = {'request message': lambda r: r['transcripts'][0]['request']['messages'][1].update(content='{}'),
                 'request hash': lambda r: r['transcripts'][0].update(request_sha256='0' * 64),
                 'recorded reply': lambda r: r['transcripts'][0]['response']['parsed']['arguments'].update(rationale='other'),
                 'recorded failure': lambda r: r['transcripts'][0].update(status='timeout'),
                 'transcript removed': lambda r: r['transcripts'].pop(),
                 'no transcripts': lambda r: r.pop('transcripts')}
        for name, edit in edits.items():
            report = tampered(edit)
            self.assertFalse(report['ok'], name)
        self.assertTrue(any('replay diverged' in p for p in tampered(edits['request hash'])['problems']))

    def test_recorded_timeouts_errors_and_worker_failures_recur_without_waiting(self):
        def slow(messages, tools):
            time.sleep(5)

        def broken(messages, tools):
            raise ConnectionError('api down')
        for llm, source in ((slow, 'fallback:llm_timeout'), (broken, 'fallback:llm_error')):
            live = []
            client = react(llm, llm_timeout=0.3)
            client.sink = live.append
            controlled_policy(PAPER, client, two_outcomes())
            replay, again = controllers.ReplayLLM(live), []
            client = controllers.ControllerClient(client.spec_, controllers.ReplayedFailures(
                lambda: controllers.ReActPolicy(replay, llm_timeout=0.3), live), again.append)
            start = time.monotonic()
            controlled_policy(PAPER, client, two_outcomes())
            self.assertLess(time.monotonic() - start, 0.25)
            self.assertEqual((live[0]['source'], replay.diverged), (source, []))
            self.assertEqual(controllers.comparable(again), controllers.comparable(live))
        live = [{**self.max_llc[0], 'source': 'fallback:worker_error', 'error': 'the worker exited with code 3'}]
        transport = controllers.ReplayedFailures(lambda: self.fail('the policy ran'), live)
        request = {'observation': live[0]['observation'], 'base_order': '3-0-2-1-4', 'orders': {}}
        self.assertEqual(transport.decide(request), ('worker_error', 'the worker exited with code 3'))

    def test_replies_are_parsed_into_bounded_plain_json(self):
        deep = []
        for _ in range(100):
            deep = [deep]
        cases = {'text': ('{"tool": "list_order_outcomes"}', {'tool': 'list_order_outcomes'}),
                 'not json text': ('<<commit 3-0-2-1-4>>', {'__type__': 'unparseable text'}),
                 'oversized': ({'thought': 'x' * 200000}, {'__type__': 'oversized reply'}),
                 'evil object': ({'tool': Evil()}, {'tool': {'__type__': 'Evil'}}),
                 'nan': ({'thought': float('nan')}, {'thought': {'__type__': 'float'}}),
                 'non-str keys': ({1: 'x'}, {'__type__': 'dict'})}
        for name, (reply, parsed) in cases.items():
            self.assertEqual(controllers.parse_reply(reply), parsed, name)
        flat = json.dumps(controllers.parse_reply(deep))
        self.assertIn('__type__', flat)
        self.assertLessEqual(flat.count('['), 33)
        policy, record = LiveDecisions.call(None, react(lambda m, t: Evil()))
        self.assertEqual((record['source'], record['react'][0]['status']), ('fallback:no_valid_commit', 'rejected'))
        json.dumps(record)


@unittest.skipUnless(REACT_WALL.exists(), 'P2 Mock wall 4 (react max_llc) not present')
class GeometryReplay(unittest.TestCase):
    """A live ReAct wall whose LLM saw outcome geometry (list_order_outcomes): the trace keeps no
    stone pixels, so Transcript Replay takes the geometry from the live records by decision key and
    candidate pose, and Geometry Validation recomputes it from base.pkl where it can."""

    @classmethod
    def setUpClass(cls):
        cls.result = check_case.result_dirs(REACT_WALL)[0]
        cls.records = jsonl(cls.result / 'trace' / 'controller.jsonl')
        cls.points = [i for i, r in enumerate(cls.records) if r['decision_point'] and len(r['observation']['outcomes']) > 1]

    def check(self, records):
        with mock.patch.object(controllers.MockLLM, '__call__', side_effect=AssertionError('an LLM was called')):
            return controllers.check(self.result, records=records)

    def test_g1_the_live_wall_replays_without_an_llm_and_its_geometry_is_recomputed(self):
        report = self.check(copy.deepcopy(self.records))
        self.assertTrue(report['ok'], report['problems'])
        self.assertEqual((report['switches'], report['geometry']['replay_problems']), (6, 0))
        v = report['geometry']['validation']
        self.assertGreater(v['verified'], 0)
        self.assertEqual((v['mismatched'], v['outcomes']), (0, sum(len(r['observation']['outcomes']) for r in self.records)))

    def test_g2_tampered_missing_or_mispaired_geometry_fails(self):
        a, b = self.points[0], self.points[1]

        def outcome(rs, i=a, k=0):
            return rs[i]['observation']['outcomes'][k]

        def swap(rs):
            o = rs[a]['observation']['outcomes']
            o[0]['geometry'], o[1]['geometry'] = o[1]['geometry'], o[0]['geometry']

        def drop(rs):
            del rs[a]['observation']['outcomes'][-1]
            rs[a]['n_outcomes'] -= 1

        def strip(rs):
            for r in rs:
                for o in r['observation']['outcomes']:
                    o['geometry'] = None

        cases = {'one geometry missing': lambda rs: outcome(rs).update(geometry=None),
                 'all geometry missing': strip,
                 'geometry swapped between two outcomes': swap,
                 'geometry of another decision': lambda rs: outcome(rs).update(geometry=outcome(rs, b)['geometry']),
                 'a value changed': lambda rs: outcome(rs)['geometry'].update(d_h_max=outcome(rs)['geometry']['d_h_max'] + 1),
                 'malformed geometry': lambda rs: outcome(rs).update(geometry={'d_h_max': 'x'}),
                 'an outcome dropped': drop,
                 'a candidate moved': lambda rs: outcome(rs).update(x=outcome(rs)['x'] + 1)}
        for name, change in cases.items():
            records = copy.deepcopy(self.records)
            change(records)
            self.assertFalse(self.check(records)['ok'], name)

    def test_g3_geometry_validation_catches_a_changed_value_on_its_own(self):
        events = list(controllers.trace_events(self.result))
        self.assertEqual(controllers.validate_geometry(self.result, self.records, events)['mismatched'], 0)
        i = next(i for i in self.points if controllers.validate_geometry(self.result, [self.records[i]], events)['verified'])
        records = copy.deepcopy(self.records)
        for o in records[i]['observation']['outcomes']:
            o['geometry']['stone_px'] += 1
        v = controllers.validate_geometry(self.result, records, events)
        self.assertEqual(v['mismatched'], controllers.validate_geometry(self.result, [self.records[i]], events)['verified'])
        with tempfile.TemporaryDirectory() as tmp:  # no base.pkl: nothing verified and nothing passed
            v = controllers.validate_geometry(copy_wall(REACT_WALL, Path(tmp)), self.records, events)
        self.assertEqual((v['verified'], v['mismatched'], v['unverifiable']), (0, 0, v['outcomes']))

    def test_g4_live_geometry_is_found_by_pose_not_by_position(self):
        live_record = copy.deepcopy(self.records[self.points[0]])
        obs = copy.deepcopy(live_record['observation'])
        for o in obs['outcomes']:
            o['geometry'] = None
        live_record['observation']['outcomes'].reverse()
        live = controllers.LiveGeometry([live_record])
        live.fill(obs)
        self.assertEqual((obs, live.problems), (self.records[self.points[0]]['observation'], []))
        for name, change in (('another decision key', lambda o: o.update(step=o['step'] + 1000)),
                             ('a moved candidate', lambda o: o['outcomes'][0].update(x=o['outcomes'][0]['x'] + 1))):
            other = copy.deepcopy(obs)
            change(other)
            live.problems.clear()
            live.fill(other)
            self.assertTrue(live.problems, name)

    def test_g5_acceptance_of_a_built_wall_requires_live_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = copy_wall(NULL_WALL, Path(tmp))
            path = out / 'trace' / 'controller.jsonl'
            records = jsonl(path)
            self.assertNotIn('controller records without live outcome geometry',
                             run_case.check_controller(Path(tmp), {'ok': True, 'problems': []})['problems'])
            for r in records:
                for o in r['observation']['outcomes']:
                    o['geometry'] = None
            jsonl(path, records)
            stripped = run_case.check_controller(Path(tmp), {'ok': True, 'problems': []})
        self.assertFalse(stripped['ok'])
        self.assertIn('controller records without live outcome geometry', stripped['problems'])


if __name__ == '__main__':
    unittest.main()
