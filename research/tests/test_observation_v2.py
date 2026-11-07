"""Observation version 2 (controllers.Observer.version_2, SYSTEM_V2, TOOLS_V2) and the
compatibility of version 1, the default. No API, no wall is built; tests on recorded walls are
skipped when the run is absent.

V1Compatibility  version 1 is unchanged: the observation stream and prompt hashes of the code before
                 version 2 existed, the recorded prompt hashes, smoke-001's offline check.
Version2         its content: filter definitions from the code, geometry availability (pixels,
                 summary, none) without invented values, only current and earlier information,
                 the same observation for the same state, Wang's selection untouched.
"""
import copy
import hashlib
import json
import sys
import types
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'research'))
sys.path.insert(0, str(REPO / 'research' / 'tests'))
import controllers  # noqa: E402
from test_controllers import PAPER, REACT_WALL, WALL, two_outcomes  # noqa: E402

from Stonepacker2D.stack import selection  # noqa: E402
from Stonepacker2D.stack.selection import FEATURE_NAMES, controlled_policy, select  # noqa: E402

SMOKE = REPO / 'research' / 'runs' / 'api-smoke' / 'smoke-001'
# Computed with the controllers.py of 3805b86 (before version 2) on WALL's 23 selections and for ReActPolicy.
V1_STREAM_SHA256 = '5d360d88d76a1adc67ada0f89a211e4de0c8151fe8c13e3cf7b2a5add57621f5'
V1_PROMPT_SHA256 = {True: '911f511f494e1d3a3be13b34ee73997d233d5d255bae2c7ed688764032378a39',
                    False: 'd4e6c0665ce7e1531cfb2b610431060abeac74923bdf2f55f0c253cff78451d9'}
V1_KEYS = {'step', 'phase', 'sample_trial', 'side', 'n_remaining', 'base_order', 'frame', 'wall', 'now', 'sampled',
           'memory', 'outcomes'}
V2_KEYS = V1_KEYS | {'observation_version', 'filter_weights', 'wall_size', 'geometry_available', 'geometry_note',
                     'remaining', 'trend'}
V2_SPEC = {'kind': 'react', 'llm': {'kind': 'mock', 'mode': 'keep'}, 'observation_version': 2}


def result_of(run):
    return next(Path(run).rglob('trace/commits.jsonl')).parent.parent


def stream(run, observer):
    base = controllers.base_policy(run)
    return [observer(types.MappingProxyType(dict(base)), controllers.context_from_trace(m, a))
            for m, a in controllers.trace_events(result_of(run))]


def live(ctx):
    """two_outcomes' bricks with a height (Wang's bricks have width and height)."""
    for b in ctx['all_bricks']:
        b.height = 2
    return ctx


def offline(ctx):
    """The same selection as a trace re-execution sees it: no bricks, their sizes as arrays."""
    return {**ctx, 'all_bricks': None, 'width': np.array([b.width for b in ctx['all_bricks']]),
            'height': np.array([b.height for b in ctx['all_bricks']])}


class V1Compatibility(unittest.TestCase):

    @unittest.skipUnless(WALL.exists(), 'recorded wall absent')
    def test_v1_stream_equals_the_code_before_version_2(self):
        for observer in (controllers.Observer(), controllers.Observer(version=1)):
            obs = [o for o, _ in stream(WALL, observer)]
            self.assertEqual(len(obs), 23)
            self.assertEqual(hashlib.sha256(json.dumps(obs, sort_keys=True).encode()).hexdigest(), V1_STREAM_SHA256)
            self.assertTrue(all(set(o) == V1_KEYS for o in obs))

    def test_v1_prompt_and_spec_unchanged(self):
        for feedback, digest in V1_PROMPT_SHA256.items():
            policy = controllers.ReActPolicy(None, tool_feedback=feedback)
            self.assertEqual((policy.prompt_sha256, policy.system), (digest, controllers.SYSTEM))
        spec = controllers.full_spec({'kind': 'react', 'llm': {'kind': 'mock', 'mode': 'keep'}})
        self.assertNotIn('observation_version', spec)  # earlier walls keep their recorded spec
        self.assertEqual(controllers.observation_version(spec), 1)

    @unittest.skipUnless(REACT_WALL.exists() and SMOKE.exists(), 'recorded walls absent')
    def test_recorded_prompt_hashes_are_version_1(self):
        for run in (REACT_WALL, SMOKE):
            path = result_of(run) / 'trace' / 'controller.jsonl'
            hashes = {r['prompt_sha256'] for r in map(json.loads, path.read_text(encoding='utf-8').splitlines())
                      if 'prompt_sha256' in r}
            self.assertEqual(hashes, {V1_PROMPT_SHA256[True]}, run)

    @unittest.skipUnless(SMOKE.exists(), 'smoke-001 absent')
    def test_smoke_001_still_replays(self):
        import api_smoke
        files = {p: p.read_bytes() for p in SMOKE.rglob('*') if p.is_file()}
        spec = json.loads((SMOKE / 'journal.json').read_text(encoding='utf-8'))['spec']
        report = api_smoke.offline_check(SMOKE, controllers.full_spec(spec))
        self.assertEqual((report['ok'], report['verdict'], report['records']), (True, 'PARTIAL', 23))
        self.assertEqual(len(report['expected_on_base_trajectory']), 2)
        self.assertEqual({p: p.read_bytes() for p in SMOKE.rglob('*') if p.is_file()}, files)  # nothing written


class Version2(unittest.TestCase):

    def test_spec(self):
        spec = controllers.validate_spec(V2_SPEC)
        self.assertEqual(controllers.observation_version(spec), 2)
        for bad in (3, 0, '2', True, 2.0):
            with self.assertRaises(ValueError):
                controllers.validate_spec({**V2_SPEC, 'observation_version': bad})
        with self.assertRaises(ValueError):  # react only
            controllers.validate_spec({'kind': 'null', 'observation_version': 2})
        client = controllers.make_controller(V2_SPEC, transport='inline')
        self.assertEqual(client.observer.version, 2)

    def test_prompt_defines_filters_from_the_code(self):
        policy = controllers.ReActPolicy(None, observation_version=2)
        self.assertEqual((policy.system, policy.tools), (controllers.SYSTEM_V2, controllers.TOOLS_V2))
        self.assertNotEqual(policy.prompt_sha256, V1_PROMPT_SHA256[True])
        text = controllers.SYSTEM_V2
        for c in range(5):  # number, abbreviation and the feature name of selection.FEATURE_NAMES
            self.assertIn(f'- {c} {controllers.FILTERS[c]} {FEATURE_NAMES[c]}:', text)
        for needle in ('higher score is better', 'LLC > 0', '>= the mean over the candidates still kept',
                       'largest sum of all five', '3-0-2-1-4', 'advantages and costs', 'Never cite a geometry value that is null',
                       'median height of the sampled stones', '0.7 or less', 'friction coefficient'):
            self.assertIn(needle, text)
        commit = {t['name']: t for t in controllers.TOOLS_V2}['commit_order']
        self.assertIn('advantages and costs', commit['parameters']['properties']['rationale']['description'])
        self.assertEqual(controllers.reply_problem(  # the schema still accepts the same commits
            {'tool': 'commit_order', 'arguments': {'order': 'keep_paper', 'rationale': 'x'}}, controllers.TOOLS_V2), None)

    def test_geometry_availability(self):
        obs, _ = controllers.Observer(version=2)(PAPER, live(two_outcomes()))
        self.assertEqual(obs['geometry_available'], 'pixels')
        self.assertTrue(all(o['geometry'] is not None and o['shape'] == {'width': 2, 'height': 2} for o in obs['outcomes']))
        obs, _ = controllers.Observer(version=2)(PAPER, offline(live(two_outcomes())))
        self.assertEqual((obs['geometry_available'], obs['geometry_note']), ('summary', controllers.GEOMETRY_NOTES['summary']))
        self.assertTrue(all(o['geometry'] is None and o['shape'] == {'width': 2, 'height': 2} for o in obs['outcomes']))
        self.assertIn('not the wall after placement', obs['geometry_note'])
        bare = {**two_outcomes(), 'all_bricks': None}
        obs, _ = controllers.Observer(version=2)(PAPER, bare)
        self.assertEqual(obs['geometry_available'], 'none')
        self.assertTrue(all(o['geometry'] is None and o['shape'] is None for o in obs['outcomes']))
        self.assertEqual(obs['remaining']['sampled_sizes'], [])

    def test_live_and_replay_observations_agree(self):
        """A live wall's check re-executes it from the trace with LiveGeometry: same observation."""
        live_obs, _ = controllers.Observer(version=2)(PAPER, live(two_outcomes()))

        class Fill:  # stands in for LiveGeometry: the recorded geometry by pose
            def fill(self, obs, classes):
                for o, g in zip(obs['outcomes'], live_obs['outcomes']):
                    o['geometry'] = g['geometry']
        replayed, _ = controllers.Observer(geometry=Fill(), version=2)(PAPER, offline(live(two_outcomes())))
        self.assertEqual(replayed, live_obs)

    @unittest.skipUnless(WALL.exists(), 'recorded wall absent')
    def test_only_current_and_earlier_information(self):
        obs = [o for o, _ in stream(WALL, controllers.Observer(version=2))]
        _, arrays = controllers.load_trace(sorted((result_of(WALL) / 'trace').glob('e*.npz'))[0])
        seen = set()
        for k, o in enumerate(obs):
            self.assertEqual(set(o), V2_KEYS)
            self.assertNotIn('rng', json.dumps(o))
            self.assertEqual(o['geometry_available'], 'summary')
            self.assertTrue(all(x['geometry'] is None for x in o['outcomes']))
            seen |= {s['stone'] for s in o['sampled']}
            order = [int(s['stone'][1:]) for s in o['remaining']['sampled_sizes']]
            self.assertEqual(order, sorted(order))  # label order: stone ids stay hidden
            sizes = {s['stone'] for s in o['remaining']['sampled_sizes']}
            self.assertLessEqual(sizes, seen)  # only stones sampled now or before
            self.assertEqual(o['remaining']['n'], o['n_remaining'])
            self.assertEqual(o['remaining']['n_never_sampled'], o['n_remaining'] - len(sizes))
            self.assertEqual([r['step'] for r in o['memory']['recent']], [x['step'] for x in obs[max(0, k - 10):k]])
            self.assertEqual(o['trend']['since_step'], obs[k - 1]['step'] if k else None)
        self.assertEqual(obs[0]['trend'], {'since_step': None, 'feasible_share': None, 'mean_feasible_llc': None})
        self.assertEqual(obs[0]['wall_size'], {'width': arrays['wall'].shape[1], 'height': arrays['wall'].shape[0]})
        # the observation at a selection never depends on later ones: a fresh observer fed the prefix agrees
        for k in (0, 5, 12, len(obs) - 1):
            observer, base = controllers.Observer(version=2), controllers.base_policy(WALL)
            for j, (m, a) in enumerate(controllers.trace_events(result_of(WALL))):
                o, _ = observer(types.MappingProxyType(dict(base)), controllers.context_from_trace(m, a))
                if j == k:
                    self.assertEqual(o, obs[k])
                    break

    @unittest.skipUnless(WALL.exists(), 'recorded wall absent')
    def test_same_state_same_observation(self):
        first = [o for o, _ in stream(WALL, controllers.Observer(version=2))]
        again = [o for o, _ in stream(WALL, controllers.Observer(version=2))]
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(again, sort_keys=True))
        # the unsampled inventory's order and the traced RNG state carry no information into it
        base, observer = controllers.base_policy(WALL), controllers.Observer(version=2)
        for (m, a), expected in zip(controllers.trace_events(result_of(WALL)), first):
            a = dict(a)
            a['inventory_ids'] = a['inventory_ids'][::-1]
            a['np_rng_keys'], a['py_rng_state'] = np.zeros_like(a['np_rng_keys']), np.zeros_like(a['py_rng_state'])
            m = {**m, 'np_rng': None, 'py_rng': None}
            o, _ = observer(types.MappingProxyType(dict(base)), controllers.context_from_trace(m, a))
            self.assertEqual(o, expected)

    @unittest.skipUnless(WALL.exists(), 'recorded wall absent')
    def test_wang_selection_untouched(self):
        """A version 2 client keeping Paper picks what Wang traced at every selection, and the
        version 2 observer leaves its inputs as they were."""
        records = controllers.rerun(result_of(WALL), V2_SPEC, controllers.base_policy(WALL))
        events = list(controllers.trace_events(result_of(WALL)))
        self.assertEqual(len(records), len(events))
        for r, (m, a) in zip(records, events):
            traced = int(a['selected'])  # -1: no feasible candidate (record: None)
            self.assertEqual(r['final']['index'], None if traced == -1 else traced)
            self.assertEqual(r['final']['index'], select(a['features_raw'], a['weights'], 'paper', (3, 0, 2, 1, 4), 1e-9).index)
            self.assertEqual(r['observation']['observation_version'], 2)
            self.assertIn(r['source'], ('llm', 'not_a_decision_point'))
            for t in r.get('transcripts', []):
                self.assertEqual(t['request']['messages'][0]['content'], controllers.SYSTEM_V2)
        m, a = events[0]
        ctx = controllers.context_from_trace(m, a)
        before = copy.deepcopy(dict(ctx))
        controllers.Observer(version=2)(PAPER, ctx)
        self.assertEqual(set(ctx), set(before))
        self.assertTrue(all(np.array_equal(ctx[k], before[k]) if isinstance(before[k], np.ndarray) else ctx[k] == before[k]
                            for k in before))

    def test_through_the_build_wall_hook(self):
        """controlled_policy (what build_wall calls) accepts a version 2 controller: global RNGs,
        settings and the selection inputs unchanged, and 'keep' returns the base policy itself."""
        client = controllers.make_controller(V2_SPEC, sink=lambda r: None, transport='inline')
        ctx = live(two_outcomes())
        self.assertIs(controlled_policy(PAPER, client, ctx), PAPER)
        client = controllers.make_controller({**V2_SPEC, 'llm': {'kind': 'mock', 'mode': 'max_llc'}},
                                             sink=lambda r: None, transport='inline')
        policy = controlled_policy(PAPER, client, live(two_outcomes()))
        self.assertEqual(set(policy), {'policy', 'order', 'tol'})


if __name__ == '__main__':
    unittest.main()
