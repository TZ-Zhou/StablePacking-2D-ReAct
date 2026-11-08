"""Tests of the API smoke entry (research/llm_walls/gpt_smoke.py); no real API anywhere.

SmokeRules     the preflight: spec limits, environment, lock and journal (any environment).
SenderClient   the real openai SDK client the worker builds: no redirect, nothing from the
               environment, refused while a connection setting is present (the isolated venv).
LoopbackRuns   whole runs through the spawned worker and the real openai SDK against a mock HTTP
               server on 127.0.0.1 with a dummy key (only where the openai SDK is installed: the
               isolated venv). Every case counts the HTTP requests the server received.
"""
import http.server
import importlib.util
import json
import multiprocessing
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from research.llm_walls import gpt_smoke  # noqa: E402
from research.controller import order_controller  # noqa: E402
from research.controller import llm_api  # noqa: E402

DUMMY = 'sk-smoke-dummy-' + 'k' * 40
SDK = importlib.util.find_spec('openai') is not None
DROP = {'OPENAI_API_KEY', llm_api.PAID_GATE, *llm_api.CONNECTION_ENV}
CLEAN_ENV = {k: v for k, v in os.environ.items() if k.upper() not in DROP}


def spec(**over):
    s = gpt_smoke.spec_of(1)
    llm = {**s['llm'], **over.pop('llm', {})}
    return {**s, 'llm': llm, **over}


class SmokeRules(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.out)

    def test_the_spec_is_checked_against_the_limits(self):
        self.assertEqual(gpt_smoke.spec_problems(gpt_smoke.spec_of(1)), [])
        self.assertEqual(gpt_smoke.max_requests(gpt_smoke.spec_of(1)), 3)  # 1 decision x 3 calls x 1 attempt
        self.assertEqual(gpt_smoke.max_requests(gpt_smoke.spec_of(3)), 9)
        bad = {'max_retries 0': spec(llm={'max_retries': 1}, llm_timeout=120.0), 'max_cost_usd <=': spec(max_cost_usd=0.6),
               'max_cost_usd': spec(max_cost_usd=None), 'max_total_tokens': {k: v for k, v in spec().items() if k != 'max_total_tokens'},
               '0 < max_total_tokens <=': spec(max_total_tokens=200_000), 'max_output_tokens': spec(llm={'max_tokens': 8192}),
               'request_timeout_s <=': spec(llm={'request_timeout_s': 90.0}, llm_timeout=100.0),
               'call_bound <= llm_timeout': spec(llm_timeout=30.0), 'llm_timeout <= 120': spec(llm_timeout=200.0),
               'max_tool_calls <=': spec(max_tool_calls=4), 'max_decisions <=': spec(max_decisions=4),
               '1 <= max_decisions': spec(max_decisions=0), 'max_llm_failures 1': spec(max_llm_failures=3),
               'the price table': spec(llm={'price_per_mtok': None}), 'the price table PRICE': spec(llm={'price_per_mtok': [0, 0, 0, 0]}),
               'model gpt-6.1-terra': spec(llm={'model': 'gpt-other'}), 'not a valid spec': spec(llm={'api_key_env': 'X'})}
        for needle, s in bad.items():
            problems = gpt_smoke.spec_problems(s)
            self.assertTrue(any(needle in p for p in problems), (needle, problems))

    def env(self, env, paid=False, loopback=False):
        with mock.patch.dict(os.environ, {**CLEAN_ENV, 'OPENAI_API_KEY': DUMMY, **env}, clear=True), \
                mock.patch.object(importlib.util, 'find_spec', return_value=object()):
            return gpt_smoke.env_problems(loopback, paid)

    def test_the_environment_rules(self):
        with mock.patch.dict(os.environ, CLEAN_ENV, clear=True), mock.patch.object(importlib.util, 'find_spec', return_value=object()):
            problems = gpt_smoke.env_problems(False, True)
        self.assertEqual(len(problems), 2, problems)  # no gate, no key
        self.assertEqual(self.env({llm_api.PAID_GATE: '1'}, paid=True), [])
        for name in llm_api.CONNECTION_ENV:  # every connection setting, the official endpoint too, in any case
            for spelled in (name, name.lower()):
                value = 'https://api.openai.com/v1' if name == 'OPENAI_BASE_URL' else 'secret-value-x'
                problems = self.env({llm_api.PAID_GATE: '1', spelled: value}, paid=True)
                self.assertEqual(len(problems), 1, (spelled, problems))
                self.assertIn(f'{name} must be unset'.upper(), problems[0].upper())  # Windows names are upper case
                self.assertNotIn(value, problems[0])  # names, never values
        self.assertEqual(len(self.env({'OPENAI_BASE_URL': 'http://127.0.0.1:9/v1'})), 1)  # a paid run refuses even a loopback endpoint
        self.assertEqual(self.env({'OPENAI_BASE_URL': 'http://127.0.0.1:9/v1'}, loopback=True), [])
        self.assertTrue(any('--loopback needs' in p for p in self.env({}, loopback=True)))
        self.assertTrue(self.env({'OPENAI_BASE_URL': 'https://api.openai.com/v1'}, loopback=True))
        self.assertTrue(self.env({'OPENAI_BASE_URL': 'http://127.0.0.1:9/v1', 'HTTPS_PROXY': 'http://p:1'}, loopback=True))
        with mock.patch.dict(os.environ, {**CLEAN_ENV, 'OPENAI_API_KEY': DUMMY}, clear=True), mock.patch.object(importlib.util, 'find_spec', return_value=None):
            self.assertIn('openai SDK is not importable', gpt_smoke.env_problems(False, False)[0])

    def journal(self, name, state, started='2026-11-07T10:00:00', **more):
        (self.out / name).mkdir(exist_ok=True)
        (self.out / name / 'journal.json').write_text(json.dumps({'state': state, 'started_at': started, **more}), encoding='utf-8')

    def test_a_run_is_never_resent_unknowingly(self):
        problems = lambda name, after=None, decisions=1: gpt_smoke.journal_problems(self.out, name, after, decisions)
        self.assertEqual(problems('smoke-001'), [])
        self.assertTrue(problems('Smoke 1'))
        self.assertTrue(problems('smoke-002', decisions=2))  # more than one decision only after an approved run
        self.journal('smoke-001', 'started')  # a run that never ended (killed): its billing is unknown
        self.assertTrue(any('smoke-001 is started' in p for p in problems('smoke-002', after='smoke-001')))
        self.assertTrue(problems('smoke-001'))  # a name is used once
        for state in ('aborted', 'uncertain'):  # an uncertain run blocks like an unfinished one, with its reasons
            self.journal('smoke-001', state, reasons=['unknown usage'])
            self.assertTrue(any(f'smoke-001 is {state} (unknown usage)' in p for p in problems('smoke-002', after='smoke-001')))
        self.journal('smoke-001', 'finished')
        self.assertTrue(any('--after smoke-001' in p for p in problems('smoke-002')))
        self.assertEqual(problems('smoke-002', after='smoke-001'), [])
        self.assertEqual(problems('smoke-002', after='smoke-001', decisions=3), [])
        self.assertTrue(problems('smoke-002', after='smoke-000'))
        (self.out / gpt_smoke.LOCK).write_text('{}', encoding='utf-8')  # another run holds the lock, or a killed one left it
        self.assertTrue(any('.lock exists' in p for p in problems('smoke-002', after='smoke-001')))
        self.assertEqual(gpt_smoke.journal_problems(self.out, 'smoke-002', 'smoke-001', 1, holding=True), [])
        (self.out / gpt_smoke.LOCK).unlink()
        (self.out / 'stray').mkdir()  # a directory without a journal blocks too
        self.assertTrue(any('stray has no journal' in p for p in problems('smoke-002', after='smoke-001')))

    def test_a_refused_run_creates_nothing_and_sends_nothing(self):
        with mock.patch.dict(os.environ, {**CLEAN_ENV, 'OPENAI_API_KEY': DUMMY}, clear=True):  # no gate
            with self.assertRaises(SystemExit) as refused:
                gpt_smoke.run_smoke('smoke-001', gpt_smoke.spec_of(1), out=self.out)
        self.assertIn(llm_api.PAID_GATE, str(refused.exception))
        self.assertEqual(list(self.out.iterdir()), [])
        with mock.patch.dict(os.environ, {**CLEAN_ENV, 'OPENAI_API_KEY': DUMMY}, clear=True):
            with self.assertRaises(SystemExit):
                gpt_smoke.main(['run', '--name', 'smoke-001'])  # neither --confirm-paid nor --loopback


TARGET = '184947496:10:9:single:0:left'  # smoke-002: the Paper wall's step 10, event 9
SMOKE_001 = REPO / 'research' / 'runs' / 'api-smoke' / 'smoke-001'


@unittest.skipUnless(gpt_smoke.WALL.exists(), 'pilot wall not present')
class TargetRules(unittest.TestCase):
    """--observation-version and --target: version 1 specs unchanged, the target pinned exactly."""

    def test_version_1_spec_unchanged(self):
        self.assertNotIn('observation_version', gpt_smoke.spec_of(1))
        self.assertNotIn('first_step', gpt_smoke.spec_of(1))
        self.assertEqual(gpt_smoke.spec_of(1), gpt_smoke.spec_of(1, 1, None))
        if SMOKE_001.exists():  # the spec smoke-001 ran with is still what the default builds
            journal = json.loads((SMOKE_001 / 'journal.json').read_text(encoding='utf-8'))
            self.assertEqual(order_controller.full_spec(gpt_smoke.spec_of(1)), journal['spec'])

    def test_the_target_is_pinned_exactly(self):
        v2 = lambda step=10, decisions=1: gpt_smoke.spec_of(decisions, 2, step)
        target = gpt_smoke.parse_target(TARGET)
        self.assertEqual(target, {'seed': 184947496, 'step': 10, 'event': 9, 'phase': 'single', 'sample_trial': 0, 'side': 'left'})
        self.assertEqual(gpt_smoke.spec_problems(v2()), [])
        self.assertEqual(gpt_smoke.target_problems(v2(), target), [])
        plan = gpt_smoke.plan_of(v2())
        self.assertEqual(plan['llm_decisions'], [{'event': 9, 'step': 10, 'phase': 'single', 'sample_trial': 0, 'side': 'left', 'n_outcomes': 5}])
        self.assertEqual((plan['skipped'], plan['observation_version'], len(plan['requests_of_the_mock_flow'])),
                         ({'skip:before_first_step': 7, 'skip:wall_budget': 6}, 2, 3))
        for wrong in ('184947496:10:8:single:0:left', '184947496:10:9:single:0:right', '184947496:10:9:single:1:left',
                      '184947496:10:9:sample_twice:0:left', '1:10:9:single:0:left',
                      '184947496:2:1:single:0:left'):  # step 2 is no decision point: never moved on to step 3
            t = gpt_smoke.parse_target(wrong)
            problems = gpt_smoke.target_problems(v2(t['step']), t)
            self.assertTrue(problems and all(p.startswith('target:') for p in problems), (wrong, problems))
        self.assertTrue(gpt_smoke.target_problems(v2(9), target))  # first_step and target disagree
        self.assertTrue(gpt_smoke.target_problems(v2(), None))  # first_step needs a target
        self.assertTrue(gpt_smoke.target_problems(v2(10, 2), target))
        self.assertTrue(any('first_step only with max_decisions 1' in p for p in gpt_smoke.spec_problems(v2(10, 2))))
        for bad in ('184947496:10:9:single:0', '184947496:10:9:single:0:up', 'a:10:9:single:0:left', None):
            with self.assertRaises(ValueError):
                gpt_smoke.parse_target(bad)

    def test_a_refused_target_creates_nothing_and_sends_nothing(self):
        out = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, out)
        wrong = gpt_smoke.parse_target('184947496:10:8:single:0:left')
        with mock.patch.dict(os.environ, {**CLEAN_ENV, 'OPENAI_API_KEY': DUMMY, llm_api.PAID_GATE: '1'}, clear=True), \
                mock.patch.object(importlib.util, 'find_spec', return_value=object()):
            with self.assertRaises(SystemExit) as refused:
                gpt_smoke.run_smoke('smoke-001', gpt_smoke.spec_of(1, 2, 10), out=out, target=wrong)
        self.assertIn('target: trace event 8', str(refused.exception))
        self.assertEqual(list(out.iterdir()), [])


@unittest.skipUnless(SDK, 'the openai SDK is only in the isolated venv')
class SenderClient(unittest.TestCase):
    def test_the_real_client_follows_no_redirect_and_reads_nothing_from_the_environment(self):
        with mock.patch.dict(os.environ, {**CLEAN_ENV, 'OPENAI_API_KEY': DUMMY}, clear=True):
            sender = llm_api.OpenAISender(llm_api.full(gpt_smoke.spec_of(1)['llm']))
            http = sender.client._client
            self.assertEqual((http.follow_redirects, http.trust_env, sender.client.max_retries), (False, False, 0))
            self.assertEqual(str(sender.client.base_url), 'https://api.openai.com/v1/')
            for name in llm_api.CONNECTION_ENV:
                value = 'https://example.com/v1' if name == 'OPENAI_BASE_URL' else 'secret-value-x'
                with mock.patch.dict(os.environ, {name: value}):
                    with self.assertRaises(ValueError) as refused:
                        llm_api.OpenAISender(llm_api.full(gpt_smoke.spec_of(1)['llm']))
                self.assertIn(name, str(refused.exception))
                self.assertNotIn(value, str(refused.exception))


class Server(http.server.ThreadingHTTPServer):
    """A mock Responses API on 127.0.0.1: script holds the behaviour of each next request."""

    def __init__(self):
        super().__init__(('127.0.0.1', 0), Handler)
        self.requests, self.script, self.lock = [], [], threading.Lock()
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f'http://127.0.0.1:{self.server_address[1]}/v1'


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        with self.server.lock:
            self.server.requests.append({'path': self.path, 'auth': self.headers.get('Authorization'), 'body': body})
            n, kind = len(self.server.requests), self.server.script.pop(0) if self.server.script else 'ok'
        if kind == 'sleep':
            time.sleep(4)
        if kind == 'kill':  # the worker dies with the request sent and no reply
            for child in multiprocessing.active_children():
                child.kill()
            return
        if kind in ('307', '308'):  # a redirect to this same server: following it would be a second POST
            return self.reply(int(kind), {}, n, location=f'{self.server.url}/elsewhere')
        if kind == '500':
            return self.reply(500, {'error': {'message': 'internal', 'type': 'server_error', 'code': None}}, n)
        listed = body['input'][-1].get('type') == 'function_call_output'
        order = '0-1-2-3-4' if kind == 'switch' else kind[len('order:'):] if kind.startswith('order:') else 'keep_paper'
        name, args = ('commit_order', {'order': order, 'rationale': 'smoke'}) if listed else ('list_order_outcomes', {})
        if kind == 'explain' and listed:  # explain_order after the listing; the next request commits
            name, args = 'explain_order', {'order': '3-2-0-1-4'}
        data = {'id': f'resp_{n}', 'object': 'response', 'created_at': 1, 'model': gpt_smoke.MODEL, 'status': 'completed', 'error': None,
                'incomplete_details': None, 'output': [
                    {'type': 'reasoning', 'id': f'rs_{n}', 'summary': [], 'encrypted_content': f'gAAAA-{n}'},
                    {'type': 'function_call', 'id': f'fc_{n}', 'call_id': f'call_{n}', 'name': name, 'arguments': json.dumps(args), 'status': 'completed'}],
                'usage': None if kind == 'no_usage' else {'input_tokens': 1200, 'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0},
                                                           'output_tokens': 90, 'output_tokens_details': {'reasoning_tokens': 60}, 'total_tokens': 1290}}
        self.reply(200, data, n)

    def reply(self, status, data, n, location=None):
        raw = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('x-request-id', f'req_{n}')
        if location:
            self.send_header('Location', location)
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        try:
            self.wfile.write(raw)
        except ConnectionError:  # the client gave up (the timeout case)
            pass


@unittest.skipUnless(SDK, 'the openai SDK is only in the isolated venv')
@unittest.skipUnless(gpt_smoke.WALL.exists(), 'pilot wall not present')
class LoopbackRuns(unittest.TestCase):
    """Whole smoke runs: main process -> ProcessTransport worker -> real SDK -> 127.0.0.1."""

    def setUp(self):
        self.out, self.server = self.fresh(), Server()
        self.addCleanup(self.server.shutdown)

    def fresh(self):
        out = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, out)
        return out

    def env(self, url=None):
        return mock.patch.dict(os.environ, {**CLEAN_ENV, 'OPENAI_API_KEY': DUMMY, 'OPENAI_BASE_URL': url or self.server.url,
                                            llm_api.PAID_GATE: '1'}, clear=True)

    def run_smoke(self, name, s=None, url=None, after=None, out=None, target=None):
        with self.env(url):
            return gpt_smoke.run_smoke(name, s or spec(), out=out or self.out, after=after, loopback=True, target=target)

    def journal(self, name, out=None):
        return json.loads(((out or self.out) / name / 'journal.json').read_text(encoding='utf-8'))

    def assert_ended(self, name, report, state, clean, api_success=False, out=None):
        out = out or self.out
        journal = self.journal(name, out)
        self.assertEqual((journal['state'], journal['clean'], journal['api_success'], journal['max_http_requests'], journal['mode']),
                         (state, clean, api_success, 3, 'loopback mock'), journal)
        self.assertEqual((report['clean'], report['api_success']), (clean, api_success))
        self.assertEqual((report['check']['ok'], report['key_scan'], report['workers']['reaped']), (True, [], True), report['check'])
        self.assertTrue(report['geometry_validation'].startswith('PARTIAL'))
        self.assertEqual(report['check']['verdict'], 'PARTIAL')  # no live outcome geometry: never PASS here
        self.assertFalse((out / gpt_smoke.LOCK).exists())  # released after the final journal state
        self.assertFalse(multiprocessing.active_children())
        for path in (out / name).rglob('*'):
            if path.is_file():
                self.assertNotIn(DUMMY[:12].encode(), path.read_bytes(), path)

    def test_a_clean_run_is_recorded_checked_and_never_resent(self):
        report = self.run_smoke('smoke-001')
        self.assertEqual(len(self.server.requests), 2)  # list_order_outcomes, then commit_order
        self.assertEqual(self.server.requests[0]['auth'], f'Bearer {DUMMY}')  # the key of OPENAI_API_KEY, sent only to the server
        self.assertEqual((report['http_requests_recorded'], report['decisions'][0]['source'], report['unknown_usage']), (2, 'llm', 0))
        self.assertEqual(report['client']['unknown_usage'], 0)
        row = report['requests'][0]
        self.assertEqual((row['outcome'], row['billed'], row['request_id'], row['served_model'], row['usage']['input_tokens']),
                         ('reply', 'reported', 'req_1', gpt_smoke.MODEL, 1200))
        self.assertAlmostEqual(row['cost_usd_estimate'], (1200 * 1.00 + 90 * 5.00) / 1e6)
        self.assertAlmostEqual(report['cost_usd_estimate'], 2 * (1200 * 1.00 + 90 * 5.00) / 1e6)
        self.assertEqual(report['skips'], ['not_a_decision_point', 'skip:wall_budget'])  # the rest of the trajectory sent nothing
        self.assert_ended('smoke-001', report, 'finished', True, api_success=True)
        for name, after in (('smoke-001', None), ('smoke-002', None)):  # the same name, or a new one without --after
            with self.assertRaises(SystemExit):
                self.run_smoke(name, after=after)
        self.assertEqual(len(self.server.requests), 2)  # nothing resent
        with mock.patch.object(gpt_smoke, 'OUT', self.out):
            self.assertEqual(gpt_smoke.main(['check', '--name', 'smoke-001']), 0)  # offline again: Transcript Replay, key scan
            with mock.patch.dict(os.environ, {**CLEAN_ENV, 'OPENAI_API_KEY': DUMMY}, clear=True):
                self.assertEqual(gpt_smoke.main(['preflight', '--name', 'smoke-002']), 1)  # --after smoke-001 missing
                self.assertEqual(gpt_smoke.main(['preflight', '--name', 'smoke-002', '--after', 'smoke-001']), 0)
        self.assertEqual(len(self.server.requests), 2)

    def records(self, name):
        result = gpt_smoke.check_run.result_dirs(self.out / name)[0]
        return [json.loads(l) for l in (result / 'trace' / 'controller.jsonl').read_text(encoding='utf-8').splitlines()]

    def test_a_v2_run_calls_only_the_target_decision_point(self):
        self.server.script = ['ok', 'explain', 'order:3-2-0-1-4']  # list, explain, commit S11 (a switch at step 10)
        target, s = gpt_smoke.parse_target(TARGET), gpt_smoke.spec_of(1, 2, 10)
        report = self.run_smoke('smoke-001', s, target=target)
        self.assertEqual(len(self.server.requests), 3)
        policy = order_controller.ReActPolicy(None, observation_version=2)
        # what GPT gets is built from events 0..9 alone (the observer fed the prefix of the trace)
        observer, base = order_controller.Observer(version=2), order_controller.base_policy(gpt_smoke.WALL)
        for meta, a in list(order_controller.trace_events(gpt_smoke.check_run.result_dirs(gpt_smoke.WALL)[0]))[:10]:
            obs, _ = observer(base, order_controller.context_from_trace(meta, a))
        view = json.dumps({k: v for k, v in obs.items() if k != 'outcomes'})
        for r in self.server.requests:
            self.assertEqual((r['body']['instructions'], r['body']['input'][0]['content']), (order_controller.SYSTEM_V2, view))
            self.assertEqual([t['name'] for t in r['body']['tools']], ['list_order_outcomes', 'explain_order', 'commit_order'])
        listing = json.loads(self.server.requests[1]['body']['input'][-1]['output'])  # list_order_outcomes' feedback
        self.assertEqual(listing['outcomes'], obs['outcomes'])
        self.assertTrue(all(o['geometry'] is None and o['shape'] for o in listing['outcomes']))
        explained = json.loads(self.server.requests[2]['body']['input'][-1]['output'])
        self.assertEqual(explained['branch'], 'full')
        self.assertEqual(report['decisions'], [{'event': 9, 'source': 'llm', 'order': '3-2-0-1-4', 'switched': True, 'calls': 3}])
        self.assertEqual(report['target'], {**target, 'requests_elsewhere': []})
        self.assertEqual(report['skips'], ['not_a_decision_point', 'skip:before_first_step', 'skip:wall_budget'])
        self.assertEqual(report['client']['llm_decisions'], 1)
        records = self.records('smoke-001')
        self.assertTrue(all(r['observation']['observation_version'] == 2 and r['controller'] == order_controller.full_spec(s) for r in records))
        early = [r for r in records if r['step'] < 10 and r['decision_point']]
        self.assertEqual({r['source'] for r in early}, {'skip:before_first_step'})
        self.assertTrue(all('transcripts' not in r for r in records if r['event'] != 9))
        at = next(r for r in records if r['event'] == 9)
        self.assertEqual((at['prompt_sha256'], at['final']['stone_id']), (policy.prompt_sha256, 13))
        self.assertTrue(all(t['request']['messages'][0]['content'] == order_controller.SYSTEM_V2 for t in at['transcripts']))
        journal = self.journal('smoke-001')
        self.assertEqual((journal['spec'], journal['target']), (order_controller.full_spec(s), target))
        receipt = json.loads((self.out / 'smoke-001' / 'receipt.json').read_text(encoding='utf-8'))
        self.assertEqual((receipt['controller']['observation_version'], receipt['controller']['first_step']), (2, 10))
        self.assertEqual(len(report['check']['expected_on_base_trajectory']), 2)  # the switch at event 9, listed apart
        self.assertTrue(all(p.startswith('event 9 ') for p in report['check']['expected_on_base_trajectory']))
        self.assert_ended('smoke-001', report, 'finished', True, api_success=True)
        with mock.patch.object(gpt_smoke, 'OUT', self.out):  # offline again: Transcript Replay with version 2
            self.assertEqual(gpt_smoke.main(['check', '--name', 'smoke-001']), 0)
        self.assertEqual(len(self.server.requests), 3)

    def test_keeping_paper_at_the_target_is_a_legal_decision(self):
        report = self.run_smoke('smoke-001', gpt_smoke.spec_of(1, 2, 10), target=gpt_smoke.parse_target(TARGET))
        self.assertEqual(len(self.server.requests), 2)
        self.assertEqual(report['decisions'], [{'event': 9, 'source': 'llm', 'order': '3-0-2-1-4', 'switched': False, 'calls': 2}])
        self.assertEqual(report['check']['expected_on_base_trajectory'], [])
        self.assert_ended('smoke-001', report, 'finished', True, api_success=True)

    def test_a_request_for_another_decision_point_is_refused_before_sending(self):
        """The guard right before the worker, should the preflight's pin ever be bypassed: another
        event, decision key or seed than the target's sends nothing and ends the run 'aborted'."""
        target = gpt_smoke.parse_target(TARGET)
        for k, wrong in enumerate(({**target, 'event': 8, 'step': 9}, {**target, 'seed': 1}, {**target, 'event': 10})):
            name = f'smoke-00{k + 1}'
            with mock.patch.object(gpt_smoke, 'target_problems', return_value=[]), \
                    mock.patch.object(gpt_smoke, 'journal_problems', return_value=[]):  # each run aborts and would block the next
                with self.assertRaises(RuntimeError) as refused:
                    self.run_smoke(name, gpt_smoke.spec_of(1, 2, 10), target=wrong)  # the step-10 spec, a wrong target
            self.assertIn('refused before sending', str(refused.exception))
            self.assertEqual(len(self.server.requests), 0, wrong)
            journal = self.journal(name)
            self.assertEqual(journal['state'], 'aborted')
            self.assertTrue(any('refused before sending' in r for r in journal['reasons']), journal['reasons'])
            self.assertFalse((self.out / gpt_smoke.LOCK).exists())
        self.assertFalse(multiprocessing.active_children())

    def test_a_request_outside_the_target_ends_the_run_uncertain(self):
        """The ledger check after the run, a second net behind the guard (disabled here)."""
        wrong = {**gpt_smoke.parse_target(TARGET), 'event': 8, 'step': 9}
        with mock.patch.object(gpt_smoke, 'target_problems', return_value=[]), mock.patch.object(gpt_smoke, 'guard'):
            report = self.run_smoke('smoke-001', gpt_smoke.spec_of(1, 2, 10), target=wrong)
        self.assertEqual(report['target']['requests_elsewhere'], [9])  # one ledger line per decision request
        journal = self.journal('smoke-001')
        self.assertEqual((journal['state'], journal['clean'], journal['api_success']), ('uncertain', False, False))
        self.assertIn('a request outside the target decision point', journal['reasons'])

    def test_a_switched_order_is_checked_on_the_base_trajectory(self):
        self.server.script = ['ok', 'switch']
        report = self.run_smoke('smoke-001')
        self.assertEqual((report['decisions'][0]['order'], report['check']['ok']), ('0-1-2-3-4', True), report['check'])
        self.assertTrue(report['check']['expected_on_base_trajectory'])  # traced Paper, controller 0-1-2-3-4: listed apart
        self.assert_ended('smoke-001', report, 'finished', True, api_success=True)
        run, spec_ = self.out / 'smoke-001', order_controller.full_spec(self.journal('smoke-001')['spec'])
        result = gpt_smoke.check_run.result_dirs(run)[0]
        path = result / 'trace' / 'controller.jsonl'
        records = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()]
        switched = next(r for r in records if r['final']['order'] == '0-1-2-3-4')
        # a controller selection error at the switched event: neither its record nor its pick is excused
        bad = [{**r, 'final': {**r['final'], 'index': r['final']['index'] + 1}} if r is switched else r for r in records]
        path.write_text(''.join(json.dumps(r) + '\n' for r in bad), encoding='utf-8')
        report = gpt_smoke.offline_check(run, spec_)
        self.assertFalse(report['ok'])
        at = [p for p in report['problems'] if p.startswith(f"event {switched['event']} ")]
        self.assertTrue(any('record differs' in p for p in at) and any('traced policy' in p for p in at), at)  # the order difference no longer excused
        self.assertEqual(report['expected_on_base_trajectory'], [])
        path.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
        # a traced policy that is not the base one (another tol) at the switched event is no expected difference
        real = order_controller.trace_events

        def other_tol(result_):
            for meta, a in real(result_):
                yield ({**meta, 'policy': {**meta['policy'], 'tol': meta['policy']['tol'] + 0.5}} if meta['event'] == switched['event'] else meta), a
        with mock.patch.object(order_controller, 'trace_events', other_tol):
            report = gpt_smoke.offline_check(run, spec_)
        self.assertFalse(report['ok'])
        self.assertTrue(any('traced policy' in p and f"event {switched['event']} " in p for p in report['problems']), report['problems'])
        self.assertTrue(gpt_smoke.offline_check(run, spec_)['ok'])  # the records restored: the check passes again

    def stopped(self, name, report, requests, outcome, out=None):
        self.assertEqual(len(self.server.requests), requests)
        decision = report['decisions'][0]
        self.assertEqual((decision['source'], decision['calls']), ('fallback:llm_error', 1))
        self.assertGreaterEqual(report['unknown_usage'], 1)
        self.assertEqual([(r['outcome'], r['billed']) for r in report['requests']], [(outcome, 'unknown')])
        self.assert_ended(name, report, 'uncertain', False, out=out)
        self.assertIn('unknown usage', self.journal(name, out)['reasons'])

    def test_a_timeout_stops_after_one_request(self):
        self.server.script = ['sleep']
        report = self.run_smoke('smoke-001', spec(llm={'request_timeout_s': 1.0}, llm_timeout=10.0))
        self.stopped('smoke-001', report, 1, 'failed')
        self.assertIn('APITimeoutError', report['requests'][0]['error'])

    def test_a_connection_error_stops_after_one_attempt(self):
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            closed = f'http://127.0.0.1:{s.getsockname()[1]}/v1'
        report = self.run_smoke('smoke-001', url=closed)
        self.stopped('smoke-001', report, 0, 'failed')  # nothing reached any server
        self.assertIn('APIConnectionError', report['requests'][0]['error'])

    def test_a_5xx_a_redirect_or_a_reply_without_usage_stops_after_one_request(self):
        for kind, outcome in (('500', 'failed'), ('307', 'failed'), ('308', 'failed'), ('no_usage', 'reply')):
            self.server.requests.clear()
            self.server.script, out = [kind], self.fresh()
            report = self.run_smoke('smoke-001', out=out)
            self.stopped('smoke-001', report, 1, outcome, out=out)  # a redirect is never followed: one POST
            if kind in ('307', '308'):
                self.assertEqual((report['requests'][0]['status_code'], self.server.requests[0]['path']), (int(kind), '/v1/responses'))

    def test_a_worker_killed_after_sending_is_unknown_usage(self):
        self.server.script = ['kill']
        report = self.run_smoke('smoke-001')
        self.assertEqual(len(self.server.requests), 1)
        self.assertTrue(report['decisions'][0]['source'].startswith('fallback:worker'), report['decisions'])
        self.assertEqual(report['decisions'][0]['calls'], 0)  # no transcript came back
        self.assertEqual((report['unknown_usage'], report['client']['unknown_usage']), (1, 1))
        self.assertEqual([(r['outcome'], r['billed']) for r in report['requests']], [('worker_failure', 'unknown')])
        self.assertEqual(len(report['worker_failures']), 1)
        self.assert_ended('smoke-001', report, 'uncertain', False)
        self.assertEqual(self.journal('smoke-001')['reasons'], ['unknown usage', 'a worker failure'])
        with self.assertRaises(SystemExit) as refused:  # uncertain: no later run, --after or not
            self.run_smoke('smoke-002', after='smoke-001')
        self.assertIn('smoke-001 is uncertain', str(refused.exception))
        self.assertEqual(len(self.server.requests), 1)

    def test_an_insufficient_budget_sends_nothing(self):
        report = self.run_smoke('smoke-001', spec(max_cost_usd=0.0001))
        self.assertEqual((len(self.server.requests), report['http_requests_recorded']), (0, 0))
        self.assertEqual(report['decisions'][0]['source'], 'fallback:llm_error')
        self.assert_ended('smoke-001', report, 'finished', True, api_success=False)  # safe and certain, but no API success

    def test_two_runs_of_different_names_never_run_at_once(self):
        self.server.script = ['sleep']  # the first run's request takes 4 s
        errors = []

        def first():
            try:
                gpt_smoke.run_smoke('smoke-001', spec(), out=self.out, loopback=True)
            except BaseException as error:
                errors.append(error)
        with self.env():
            thread = threading.Thread(target=first)
            thread.start()
            while not self.server.requests and thread.is_alive():
                time.sleep(0.05)
            self.assertTrue((self.out / gpt_smoke.LOCK).exists())
            with self.assertRaises(SystemExit) as refused:
                gpt_smoke.run_smoke('smoke-002', spec(), out=self.out, loopback=True)
            thread.join()
        self.assertIn('.lock exists', str(refused.exception))
        self.assertEqual((errors, sorted(p.name for p in self.out.iterdir())), ([], ['smoke-001']))
        self.assertEqual(len(self.server.requests), 2)  # the first run's two requests, none of the second
        self.assertEqual(self.journal('smoke-001')['state'], 'finished')

    def test_a_lock_left_behind_or_an_unfinished_run_blocks_every_new_one(self):
        (self.out / gpt_smoke.LOCK).write_text('{"name": "smoke-000"}', encoding='utf-8')  # a run killed while it held the lock
        with self.assertRaises(SystemExit) as refused:
            self.run_smoke('smoke-001')
        self.assertIn('.lock exists', str(refused.exception))
        self.assertEqual((self.out / gpt_smoke.LOCK).read_text(encoding='utf-8'), '{"name": "smoke-000"}')  # never removed automatically
        (self.out / gpt_smoke.LOCK).unlink()
        (self.out / 'smoke-001').mkdir()
        (self.out / 'smoke-001' / 'journal.json').write_text(json.dumps({'state': 'started', 'started_at': 'x'}), encoding='utf-8')
        with self.assertRaises(SystemExit) as refused:
            self.run_smoke('smoke-002', after='smoke-001')
        self.assertIn('smoke-001 is started', str(refused.exception))
        self.assertEqual((len(self.server.requests), sorted(p.name for p in self.out.iterdir())), (0, ['smoke-001']))


if __name__ == '__main__':
    unittest.main()
