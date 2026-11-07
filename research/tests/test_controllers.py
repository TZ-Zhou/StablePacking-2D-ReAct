"""Checks for the filter-order controller hook (selection.controlled_policy) and research/controllers.py.

Run from the repository root: python -B -m unittest discover -s research/tests
Tests on a recorded wall replay its trace offline and are skipped when the run is absent.
Regression tests for the P0/P1 review findings are marked R1..R8.
"""
import json
import random
import re
import sys
import time
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'research'))
import check_case  # noqa: E402
import controllers  # noqa: E402
import run_case  # noqa: E402

from Stonepacker2D.stack import selection  # noqa: E402
from Stonepacker2D.stack.selection import controlled_policy  # noqa: E402
from Stonepacker2D.utils import constant  # noqa: E402

PAPER = {'policy': 'paper', 'order': (3, 0, 2, 1, 4), 'tol': 1e-9}
WALL = REPO / 'research' / 'runs' / 'campaigns' / 'pilot-thangavelu-4arms' / 'thangavelu-noise80' / 'paper_3-0-2-1-4' / 'seed184947496'
RANDOM_WALL = REPO / 'research' / 'runs' / 'controllers' / 'p1' / 'random1-seed20-a'


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
        with self.assertRaisesRegex(RuntimeError, 'base policy'):
            controlled_policy(constant.get_filter_policy(), mutate_global, context())
        constant.get_filter_policy()['tol'] = before['tol']
        self.assertEqual(constant.get_filter_policy(), before)

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
            controller = controllers.ReActController(llm, {'kind': 'test'}, llm_timeout=0.1)
            policy, record = self.call(controller)
            self.assertIs(policy, PAPER)
            self.assertEqual((record['source'], record['final']['switched']), (source, False))
            self.assertEqual(record['react'][0]['status'], status)
            self.assertTrue(record['react'][0].get('error') or record['react'][0]['feedback']['error'])
        self.assertEqual(calls.count('silent'), 4)

        controller = controllers.ReActController(broken, {'kind': 'test'}, max_llm_failures=2)
        calls.clear()
        sources = [self.call(controller)[1]['source'] for _ in range(3)]
        self.assertEqual(sources, ['fallback:llm_error', 'fallback:llm_error', 'fallback:llm_unavailable'])
        self.assertEqual(len(calls), 2)

    def test_r7_rejected_replies_are_logged_with_their_reason(self):
        replies = iter([{'tool': 'commit_order', 'arguments': {'order': [0.9, 1.9, 2.9, 3.9, 4.9], 'rationale': 'r'}},
                        {'tool': 'commit_order', 'arguments': {'order': '0-1-2-3-4'}},
                        {'tool': 'commit_order', 'arguments': {'order': '0-1-2-3-4', 'rationale': 'ok'}}])
        controller = controllers.ReActController(lambda m, t: next(replies), {'kind': 'test'})
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
        self.assertEqual(set(sources[2:]), {'fallback:wall_budget'})

    def test_without_tool_feedback_only_commit_is_offered(self):
        seen = []

        def llm(messages, tools):
            seen.append(([t['name'] for t in tools], json.loads(messages[1]['content'])))
            return {'thought': 't', 'tool': 'commit_order', 'arguments': {'order': 'keep_paper', 'rationale': 't'}}
        controller = controllers.ReActController(llm, tool_feedback=False, sink=lambda r: None)
        meta, a = next(controllers.trace_events(self.result))
        controller.decide(*controller.observer(PAPER, controllers.context_from_trace(meta, a)), None)
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


if __name__ == '__main__':
    unittest.main()
