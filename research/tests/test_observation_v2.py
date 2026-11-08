"""Observation version 2 (order_controller.Observer.version_2, SYSTEM_V2, TOOLS_V2) and the
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
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'research' / 'tests'))
from research.controller import order_controller  # noqa: E402
from test_order_controller import PAPER, REACT_WALL, WALL, two_outcomes  # noqa: E402

from Stonepacker2D.stack import selection  # noqa: E402
from Stonepacker2D.stack.selection import FEATURE_NAMES, controlled_policy, select  # noqa: E402

SMOKE = REPO / 'research' / 'runs' / 'api-smoke' / 'smoke-001'
# Computed with the order_controller.py of 3805b86 (before version 2) on WALL's 23 selections and for ReActPolicy.
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
    base = order_controller.base_policy(run)
    return [observer(types.MappingProxyType(dict(base)), order_controller.context_from_trace(m, a))
            for m, a in order_controller.trace_events(result_of(run))]


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
        for observer in (order_controller.Observer(), order_controller.Observer(version=1)):
            obs = [o for o, _ in stream(WALL, observer)]
            self.assertEqual(len(obs), 23)
            self.assertEqual(hashlib.sha256(json.dumps(obs, sort_keys=True).encode()).hexdigest(), V1_STREAM_SHA256)
            self.assertTrue(all(set(o) == V1_KEYS for o in obs))

    def test_v1_prompt_and_spec_unchanged(self):
        for feedback, digest in V1_PROMPT_SHA256.items():
            policy = order_controller.ReActPolicy(None, tool_feedback=feedback)
            self.assertEqual((policy.prompt_sha256, policy.system), (digest, order_controller.SYSTEM))
        spec = order_controller.full_spec({'kind': 'react', 'llm': {'kind': 'mock', 'mode': 'keep'}})
        self.assertNotIn('observation_version', spec)  # earlier walls keep their recorded spec
        self.assertEqual(order_controller.observation_version(spec), 1)

    @unittest.skipUnless(REACT_WALL.exists() and SMOKE.exists(), 'recorded walls absent')
    def test_recorded_prompt_hashes_are_version_1(self):
        for run in (REACT_WALL, SMOKE):
            path = result_of(run) / 'trace' / 'controller.jsonl'
            hashes = {r['prompt_sha256'] for r in map(json.loads, path.read_text(encoding='utf-8').splitlines())
                      if 'prompt_sha256' in r}
            self.assertEqual(hashes, {V1_PROMPT_SHA256[True]}, run)

    @unittest.skipUnless(SMOKE.exists(), 'smoke-001 absent')
    def test_smoke_001_still_replays(self):
        from research.llm_walls import gpt_smoke
        files = {p: p.read_bytes() for p in SMOKE.rglob('*') if p.is_file()}
        spec = json.loads((SMOKE / 'journal.json').read_text(encoding='utf-8'))['spec']
        report = gpt_smoke.offline_check(SMOKE, order_controller.full_spec(spec))
        self.assertEqual((report['ok'], report['verdict'], report['records']), (True, 'PARTIAL', 23))
        self.assertEqual(len(report['expected_on_base_trajectory']), 2)
        self.assertEqual({p: p.read_bytes() for p in SMOKE.rglob('*') if p.is_file()}, files)  # nothing written


class Version2(unittest.TestCase):

    def test_spec(self):
        spec = order_controller.validate_spec(V2_SPEC)
        self.assertEqual(order_controller.observation_version(spec), 2)
        for bad in (3, 0, '2', True, 2.0):
            with self.assertRaises(ValueError):
                order_controller.validate_spec({**V2_SPEC, 'observation_version': bad})
        with self.assertRaises(ValueError):  # react only
            order_controller.validate_spec({'kind': 'null', 'observation_version': 2})
        client = order_controller.make_controller(V2_SPEC, transport='inline')
        self.assertEqual(client.observer.version, 2)

    def test_prompt_defines_filters_from_the_code(self):
        policy = order_controller.ReActPolicy(None, observation_version=2)
        self.assertEqual((policy.system, policy.tools), (order_controller.SYSTEM_V2, order_controller.TOOLS_V2))
        self.assertNotEqual(policy.prompt_sha256, V1_PROMPT_SHA256[True])
        text = order_controller.SYSTEM_V2
        for c in range(5):  # number, abbreviation and the feature name of selection.FEATURE_NAMES
            self.assertIn(f'- {c} {order_controller.FILTERS[c]} {FEATURE_NAMES[c]}:', text)
        for needle in ('higher score is better', 'LLC > 0', 'w >= m - tol x max(1, |m|)', "tol being the base policy's tolerance",
                       'largest sum of all five', '3-0-2-1-4', 'advantages and costs', 'Never cite a geometry value that is null',
                       'median stored height of the sampled stones', '0.7 or less', 'friction coefficient',
                       # the two states, what turning updates and what it leaves (test_stabilization_turns_only_the_pixels)
                       'LFR: the candidate at the pose convolution found', 'NHR, GDS, IWR and LLC: a copy of the current wall',
                       "Turning replaces only the stone's pixels", 'stay those of the unturned candidate',
                       'before its own stabilization', "candidate's own unstabilized pixels", 'not a state any filter uses',
                       # the frame (test_lfr_as_the_prompt_defines_it_is_wangs, test_filter_rules_on_the_recorded_trace,
                       # test_iwr_touch_as_the_prompt_defines_it)
                       'never a wall pixel for LFR, GDS or LLC', "its sides are edges for IWR's touch test",
                       'frame pixels count as empty', 'ground included, frame excluded',
                       '-(L + D) / (matrix width x stone height + matrix height x stone width)',
                       'flush on the left and on the bottom', "that stone's stored span", 'over at least 3 rows',
                       'upper half of the side',
                       # the cascade's fallback (test_cascade_fallback_as_the_prompt_defines_it)
                       'With finite w and m the best candidate always passes', 'If a stage keeps none',
                       'this one included, and the later filters are not used'):
            self.assertIn(needle, text)
        commit = {t['name']: t for t in order_controller.TOOLS_V2}['commit_order']
        self.assertIn('advantages and costs', commit['parameters']['properties']['rationale']['description'])
        self.assertEqual(order_controller.reply_problem(  # the schema still accepts the same commits
            {'tool': 'commit_order', 'arguments': {'order': 'keep_paper', 'rationale': 'x'}}, order_controller.TOOLS_V2), None)

    def test_lfr_as_the_prompt_defines_it_is_wangs(self):
        """SYSTEM_V2's LFR, implemented from its text alone, against local_void.get_void_to_left
        (place_stone's filters[4] = -get_void_to_left) on small walls; row 0 is the bottom."""
        from Stonepacker2D.evaluate.local_void import get_void_to_left

        def as_prompt(wall, stone, center):
            material = (wall != 0) & (wall != 255)  # wall pixels: the ground counts, the 255 boundary does not
            rows, cols = np.nonzero(stone)
            h, w = rows.max() - rows.min() + 1, cols.max() - cols.min() + 1
            left = 0
            for r in set(rows):  # each row: empty pixels between the leftmost stone pixel and the wall left of the center column
                c0 = cols[rows == r].min()
                walls = np.flatnonzero(material[r, :int(center[0])])
                left += max(0, c0 - (walls.max() if walls.size else -1) - 1)
            down = 0
            for c in set(cols):  # each column: empty pixels between the lowest stone pixel and the wall below the center row
                r0 = rows[cols == c].min()
                walls = np.flatnonzero(material[:int(center[1]), c])
                down += max(0, r0 - (walls.max() if walls.size else -1) - 1)
            return -(left + down) / (wall.shape[1] * h + wall.shape[0] * w)

        def wall_():
            w = np.zeros((12, 14))
            w[:2] = 1  # ground
            w[-1], w[:, 0], w[:, -1] = 255, 255, 255
            return w
        cases = []
        w = wall_()
        w[2:6, 1:4] = 2  # a placed stone on the left
        s = np.zeros_like(w)
        s[4:8, 6:10] = 1  # gap of 2 to the left, 2 above the ground
        cases.append((w, s))
        s = np.zeros_like(w)
        s[2:6, 4:8] = 1  # flush on the left and on the ground
        cases.append((w, s))
        w2 = wall_()
        s = np.zeros_like(w2)
        s[3:6, 5:9] = 1  # no wall to the left: the gap runs to the matrix edge
        s[6, 6:8] = 1  # an irregular top row
        cases.append((w2, s))
        w3 = wall_()
        w3[:, 1:5] = 255  # a thick frame directly left of the stone: frame pixels count as empty
        cases.append((w3, s))
        values = []
        for wall, stone in cases:
            rows, cols = np.nonzero(stone)
            center = ((cols.min() + cols.max()) / 2, (rows.min() + rows.max()) / 2)
            h, wd = rows.max() - rows.min() + 1, cols.max() - cols.min() + 1
            wang = -get_void_to_left(types.SimpleNamespace(matrix=wall),
                                     types.SimpleNamespace(matrix=stone, center=center, height=h, width=wd))
            self.assertAlmostEqual(as_prompt(wall, stone, center), wang, places=12)
            values.append(wang)
        self.assertLess(values[0], 0)
        self.assertEqual(values[1], 0)  # flush: 0
        self.assertLess(values[2], 0)
        self.assertEqual(values[3], values[2])  # the frame changes nothing

    def test_iwr_touch_as_the_prompt_defines_it(self):
        """IWR's touch test, by local_void.evaluate_convolve at one pinned pose: a vertical edge of the
        wall matrix over at least 3 rows within 1 px, along the upper half of the side (one row lower
        for an even height); the frame's side is an edge."""
        from skimage.measure import regionprops
        from Stonepacker2D.evaluate.local_void import evaluate_convolve

        def left_touch(h, rows, frame=False):
            wall = np.zeros((30, 30))
            wall[:2], wall[-1], wall[:, 0], wall[:, -1] = 1, 255, 255, 255
            if frame:
                wall[:, :10] = 255
            for r in rows:
                wall[2 + r, 9] = 2  # wall pixels directly left of the stone's row r
            stone = np.zeros((30, 30))
            stone[:h, :6] = 1
            box = regionprops(stone.astype(np.uint8))[0].image
            pinned = np.zeros((30, 30))
            pinned[2, 10] = 1  # the only allowed pose: bottom-left corner at row 2, column 10
            r = evaluate_convolve(types.SimpleNamespace(matrix=wall, rock_left_tops=[], rock_right_tops=[]),
                                  types.SimpleNamespace(matrix=stone, width=6, height=h), np.flip(box), pinned)
            self.assertEqual(tuple(r['best_loc']), (2, 10))
            return r['best_left_touch'] <= 1
        self.assertTrue(left_touch(9, range(4, 9)))  # upper half (rows 4-8 of 9)
        self.assertFalse(left_touch(9, range(0, 4)))  # lower half only
        self.assertTrue(left_touch(8, range(0, 4)))  # even height: one row lower
        self.assertTrue(left_touch(9, range(6, 9)))  # 3 rows make an edge
        self.assertFalse(left_touch(9, range(6, 8)))  # 2 rows do not
        self.assertFalse(left_touch(9, []))  # no wall beside it
        self.assertTrue(left_touch(9, [], frame=True))  # the frame's side

    def test_stabilization_turns_only_the_pixels(self):
        """Wang's Base.add_stone with the 'rotation' method on a stone overhanging a lower stone:
        rotate_brick turns its pixels, but its stored center, width, height and the corner points and
        centers the wall stored for it stay those of the unturned stone (what NHR and IWR read)."""
        from Stonepacker2D.entity.base import Base
        from Stonepacker2D.entity.stone import Stone
        from Stonepacker2D.utils import constant
        saved = (constant.get_world_size(), constant.get_rotate_state(), constant.get_stabalize_method())
        self.addCleanup(lambda: (constant.set_world_size(*saved[0]), constant.set_rotate_state(saved[1]),
                                 constant.set_stabalize_method(saved[2])))
        constant.set_world_size(40, 40)
        constant.set_rotate_state(True)
        constant.set_stabalize_method('rotation')

        def stone(rows, cols, id_):
            s = Stone()
            s.matrix = np.zeros((40, 40))
            s.matrix[:rows, :cols] = 1
            s.center, s.height, s.width, s.id, s.cluster, s.area = [(cols - 1) / 2, (rows - 1) / 2], rows, cols, id_, 0, rows * cols
            return s
        base = Base()
        ground = stone(2, 40, constant.get_base_id())
        base.add_stone(ground, [0, 0])
        base.add_stone(stone(8, 10, 1), [2, 2])
        top = stone(6, 12, 2)
        unturned = top.transformed([11, 10])
        base.add_stone(top, [11, 10])  # center past the lower stone's right side: turned
        rock = base.placed_rocks[-1]
        pixels, before = np.argwhere(rock.matrix), np.argwhere(unturned.matrix)
        self.assertNotEqual(rock.rot_angle, 0)
        self.assertFalse(np.array_equal(rock.matrix != 0, unturned.matrix != 0))  # the pixels turned
        self.assertEqual(np.ptp(pixels[:, 0]) + 1, 7)  # the turned stone is 7 rows high
        self.assertEqual((rock.height, rock.width), (6, 12))  # stored: unturned
        self.assertEqual(list(rock.center), list(unturned.center))
        self.assertEqual(list(base.rock_centers[-1]), list(unturned.center))
        self.assertEqual(base.rock_left_tops[-1], [before[:, 1].min(), before[:, 0].max()])
        self.assertEqual(base.rock_right_tops[-1], [before[:, 1].max(), before[:, 0].max()])

    def test_cascade_fallback_as_the_prompt_defines_it(self):
        """selection._paper: with finite scores the best candidate always passes; a stage keeps none
        only when the arithmetic is not finite, and then the pick is the largest sum of the filters
        applied so far, this one included, over that stage's input."""
        rng = np.random.default_rng(0)
        for _ in range(300):
            raw = rng.normal(size=(int(rng.integers(1, 12)), 6)) * 10.0 ** rng.integers(-6, 7)
            raw[:, 3] = np.abs(raw[:, 3]) + 1e-12
            for order in ((3, 0, 2, 1, 4), (4, 1, 2, 0, 3)):
                self.assertEqual(select(raw, [1.0] * 6, 'paper', order, 1e-9).branch, 'full')
        raw = np.array([[1e308, 0, 0, 1, 0, 0], [1.7e308, 1, 0, 1, 0, 0], [1e308, 5, 0, 1, 0, 0]], dtype=float)
        s = select(raw, [1.0] * 6, 'paper', (3, 0, 2, 1, 4), 1e-9)  # the NHR mean overflows at stage 2
        self.assertEqual(s.branch, 'fallback_2')
        self.assertEqual(s.index, 1)  # largest LLC + NHR among the stage's input, not the largest GDS

    @unittest.skipUnless(WALL.exists(), 'recorded wall absent')
    def test_filter_rules_on_the_recorded_trace(self):
        """What the trace can confirm without stone pixels, at the first selection (nothing placed yet):
        NHR is the no-left-neighbour rule with the sampled stones' median height, IWR 0.5 (no stone
        below), GDS 0: B is the ground band. Were the frame (255, all around the wall) a wall pixel, B
        would be the whole matrix and every GDS 1."""
        m, a = next(iter(order_controller.trace_events(result_of(WALL))))
        raw, n = a['features_raw'], len(a['sampled_ids'])
        median = np.percentile(a['height'][:n], 50)  # rotation-0 rows: the sampled stones themselves
        np.testing.assert_allclose(raw[:, 0], -np.abs(median - a['height']) / a['height'])
        np.testing.assert_allclose(raw[:, 2], 0.5)
        self.assertTrue((a['wall'][-1] == 255).all() and (a['wall'][:, 0] == 255).all())
        np.testing.assert_allclose(raw[:, 1], 0.0)

    @unittest.skipUnless(WALL.exists(), 'recorded wall absent')
    def test_stone_ids_stay_anonymous(self):
        """Renaming every stone id (one bijection for the whole wall; sampling order, candidates and
        scores unchanged) gives byte-identical serialized observations and order tables in version 2."""
        events = list(order_controller.trace_events(result_of(WALL)))
        ids = sorted({int(s) for _, a in events for k in ('sampled_ids', 'inventory_ids') for s in a[k]})
        rename = dict(zip(ids, reversed([1000 + i for i in range(len(ids))])))  # reverses the id order
        base = order_controller.base_policy(WALL)

        def run(mapping):
            observer, out = order_controller.Observer(version=2), []
            for meta, a in events:
                a = {**a, **{k: np.array([mapping.get(int(s), int(s)) for s in a[k]]) for k in ('sampled_ids', 'inventory_ids')}}
                obs, info = observer(types.MappingProxyType(dict(base)), order_controller.context_from_trace(meta, a))
                out.append(json.dumps(obs) + json.dumps(order_controller.tool_table(info, observer.label)))
            return out
        original, renamed = run({}), run(rename)
        self.assertEqual(len(original), len(events))
        for k, (x, y) in enumerate(zip(original, renamed)):
            self.assertEqual(x, y, f'event {k}')
        # version 1 keeps its recorded behaviour (stone-id order in memory.stones), unchanged here
        v1 = order_controller.Observer()
        for meta, a in events[:10]:
            obs, _ = v1(types.MappingProxyType(dict(base)), order_controller.context_from_trace(meta, a))
        labels = {int(s): v1.label(s) for s in ids if int(s) in v1.labels}
        self.assertEqual(list(obs['memory']['stones']), [labels[s] for s in sorted(labels) if labels[s] in obs['memory']['stones']])

    def test_geometry_availability(self):
        obs, _ = order_controller.Observer(version=2)(PAPER, live(two_outcomes()))
        self.assertEqual(obs['geometry_available'], 'pixels')
        self.assertTrue(all(o['geometry'] is not None and o['shape'] == {'width': 2, 'height': 2} for o in obs['outcomes']))
        obs, _ = order_controller.Observer(version=2)(PAPER, offline(live(two_outcomes())))
        self.assertEqual((obs['geometry_available'], obs['geometry_note']), ('summary', order_controller.GEOMETRY_NOTES['summary']))
        self.assertTrue(all(o['geometry'] is None and o['shape'] == {'width': 2, 'height': 2} for o in obs['outcomes']))
        self.assertIn('not the wall after placement', obs['geometry_note'])
        bare = {**two_outcomes(), 'all_bricks': None}
        obs, _ = order_controller.Observer(version=2)(PAPER, bare)
        self.assertEqual(obs['geometry_available'], 'none')
        self.assertTrue(all(o['geometry'] is None and o['shape'] is None for o in obs['outcomes']))
        self.assertEqual(obs['remaining']['sampled_sizes'], [])

    def test_live_and_replay_observations_agree(self):
        """A live wall's check re-executes it from the trace with LiveGeometry: same observation."""
        live_obs, _ = order_controller.Observer(version=2)(PAPER, live(two_outcomes()))

        class Fill:  # stands in for LiveGeometry: the recorded geometry by pose
            def fill(self, obs, classes):
                for o, g in zip(obs['outcomes'], live_obs['outcomes']):
                    o['geometry'] = g['geometry']
        replayed, _ = order_controller.Observer(geometry=Fill(), version=2)(PAPER, offline(live(two_outcomes())))
        self.assertEqual(replayed, live_obs)

    @unittest.skipUnless(WALL.exists(), 'recorded wall absent')
    def test_only_current_and_earlier_information(self):
        obs = [o for o, _ in stream(WALL, order_controller.Observer(version=2))]
        _, arrays = order_controller.load_trace(sorted((result_of(WALL) / 'trace').glob('e*.npz'))[0])
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
            observer, base = order_controller.Observer(version=2), order_controller.base_policy(WALL)
            for j, (m, a) in enumerate(order_controller.trace_events(result_of(WALL))):
                o, _ = observer(types.MappingProxyType(dict(base)), order_controller.context_from_trace(m, a))
                if j == k:
                    self.assertEqual(o, obs[k])
                    break

    @unittest.skipUnless(WALL.exists(), 'recorded wall absent')
    def test_same_state_same_observation(self):
        first = [o for o, _ in stream(WALL, order_controller.Observer(version=2))]
        again = [o for o, _ in stream(WALL, order_controller.Observer(version=2))]
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(again, sort_keys=True))
        # the unsampled inventory's order and the traced RNG state carry no information into it
        base, observer = order_controller.base_policy(WALL), order_controller.Observer(version=2)
        for (m, a), expected in zip(order_controller.trace_events(result_of(WALL)), first):
            a = dict(a)
            a['inventory_ids'] = a['inventory_ids'][::-1]
            a['np_rng_keys'], a['py_rng_state'] = np.zeros_like(a['np_rng_keys']), np.zeros_like(a['py_rng_state'])
            m = {**m, 'np_rng': None, 'py_rng': None}
            o, _ = observer(types.MappingProxyType(dict(base)), order_controller.context_from_trace(m, a))
            self.assertEqual(o, expected)

    @unittest.skipUnless(WALL.exists(), 'recorded wall absent')
    def test_wang_selection_untouched(self):
        """A version 2 client keeping Paper picks what Wang traced at every selection, and the
        version 2 observer leaves its inputs as they were."""
        records = order_controller.rerun(result_of(WALL), V2_SPEC, order_controller.base_policy(WALL))
        events = list(order_controller.trace_events(result_of(WALL)))
        self.assertEqual(len(records), len(events))
        for r, (m, a) in zip(records, events):
            traced = int(a['selected'])  # -1: no feasible candidate (record: None)
            self.assertEqual(r['final']['index'], None if traced == -1 else traced)
            self.assertEqual(r['final']['index'], select(a['features_raw'], a['weights'], 'paper', (3, 0, 2, 1, 4), 1e-9).index)
            self.assertEqual(r['observation']['observation_version'], 2)
            self.assertIn(r['source'], ('llm', 'not_a_decision_point'))
            for t in r.get('transcripts', []):
                self.assertEqual(t['request']['messages'][0]['content'], order_controller.SYSTEM_V2)
        m, a = events[0]
        ctx = order_controller.context_from_trace(m, a)
        before = copy.deepcopy(dict(ctx))
        order_controller.Observer(version=2)(PAPER, ctx)
        self.assertEqual(set(ctx), set(before))
        self.assertTrue(all(np.array_equal(ctx[k], before[k]) if isinstance(before[k], np.ndarray) else ctx[k] == before[k]
                            for k in before))

    @unittest.skipUnless(WALL.exists(), 'recorded wall absent')
    def test_first_step(self):
        """first_step: earlier decision points are skipped without a request or a decision of the
        budget; without it (every earlier spec) the first decision point is answered as before."""
        for bad in (0, -1, '10', True, 2.5):
            with self.assertRaises(ValueError):
                order_controller.validate_spec({**V2_SPEC, 'first_step': bad})
        with self.assertRaises(ValueError):  # react only: the other controllers are unchanged
            order_controller.validate_spec({'kind': 'random', 'seed': 1, 'first_step': 10})
        mock_ = {'kind': 'mock', 'mode': 'max_llc'}
        for first, step in ((None, 1), (10, 10)):
            spec = {**V2_SPEC, 'llm': mock_, 'max_decisions': 1, **({} if first is None else {'first_step': first})}
            records = order_controller.rerun(result_of(WALL), spec, order_controller.base_policy(WALL))
            asked = [r for r in records if r['source'] == 'llm']
            self.assertEqual([(r['event'], r['step']) for r in asked], [(step - 1, step)])
            before = {r['source'] for r in records if r['decision_point'] and r['step'] < step}
            self.assertEqual(before, set() if first is None else {'skip:before_first_step'})
            self.assertEqual({r['source'] for r in records if r['decision_point'] and r['step'] > step}, {'skip:wall_budget'})
            self.assertTrue(all('transcripts' not in r for r in records if r not in asked))

    def test_through_the_build_wall_hook(self):
        """controlled_policy (what build_wall calls) accepts a version 2 controller: global RNGs,
        settings and the selection inputs unchanged, and 'keep' returns the base policy itself."""
        client = order_controller.make_controller(V2_SPEC, sink=lambda r: None, transport='inline')
        ctx = live(two_outcomes())
        self.assertIs(controlled_policy(PAPER, client, ctx), PAPER)
        client = order_controller.make_controller({**V2_SPEC, 'llm': {'kind': 'mock', 'mode': 'max_llc'}},
                                             sink=lambda r: None, transport='inline')
        policy = controlled_policy(PAPER, client, live(two_outcomes()))
        self.assertEqual(set(policy), {'policy', 'order', 'tol'})


if __name__ == '__main__':
    unittest.main()
