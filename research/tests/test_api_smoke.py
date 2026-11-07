"""Tests of the API smoke entry (research/api_smoke.py); no real API anywhere.

SmokeRules     the preflight: spec limits, environment, journal (any environment).
LoopbackRuns   whole runs through the spawned worker and the real openai SDK against a mock HTTP
               server on 127.0.0.1 with a dummy key (only where the openai SDK is installed: the
               isolated venv). Every case counts the HTTP requests the server received.
"""
import http.server
import importlib.util
import json
import multiprocessing
import os
import socket
import sys
import tempfile
import shutil
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'research'))
import api_smoke  # noqa: E402
import llm_adapter  # noqa: E402

DUMMY = 'sk-smoke-dummy-' + 'k' * 40
SDK = importlib.util.find_spec('openai') is not None
CLEAN_ENV = {k: v for k, v in os.environ.items()
             if k not in ('OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_CUSTOM_HEADERS', 'OPENAI_LOG', llm_adapter.PAID_GATE)}


def spec(**over):
    s = api_smoke.spec_of(1)
    llm = {**s['llm'], **over.pop('llm', {})}
    return {**s, 'llm': llm, **over}


class SmokeRules(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.out)

    def test_the_spec_is_checked_against_the_limits(self):
        self.assertEqual(api_smoke.spec_problems(api_smoke.spec_of(1)), [])
        self.assertEqual(api_smoke.max_requests(api_smoke.spec_of(1)), 3)  # 1 decision x 3 calls x 1 attempt
        self.assertEqual(api_smoke.max_requests(api_smoke.spec_of(3)), 9)
        bad = {'max_retries 0': spec(llm={'max_retries': 1}, llm_timeout=120.0), 'max_cost_usd <=': spec(max_cost_usd=0.6),
               'max_cost_usd': spec(max_cost_usd=None), 'max_total_tokens': {k: v for k, v in spec().items() if k != 'max_total_tokens'},
               '0 < max_total_tokens <=': spec(max_total_tokens=200_000), 'max_output_tokens': spec(llm={'max_tokens': 8192}),
               'request_timeout_s <=': spec(llm={'request_timeout_s': 90.0}, llm_timeout=100.0),
               'call_bound <= llm_timeout': spec(llm_timeout=30.0), 'llm_timeout <= 120': spec(llm_timeout=200.0),
               'max_tool_calls <=': spec(max_tool_calls=4), 'max_decisions <=': spec(max_decisions=4),
               '1 <= max_decisions': spec(max_decisions=0), 'max_llm_failures 1': spec(max_llm_failures=3),
               'a price per million': spec(llm={'price_per_mtok': None}), 'model gpt-6.1-terra': spec(llm={'model': 'gpt-other'}),
               'not a valid spec': spec(llm={'api_key_env': 'X'})}
        for needle, s in bad.items():
            problems = api_smoke.spec_problems(s)
            self.assertTrue(any(needle in p for p in problems), (needle, problems))

    def test_the_environment_rules(self):
        cases = [({}, True, False, ['STONEPACKING_ALLOW_PAID_API', 'OPENAI_API_KEY']),
                 ({'OPENAI_API_KEY': DUMMY, llm_adapter.PAID_GATE: '1', 'OPENAI_BASE_URL': 'https://example.com/v1'}, True, False, ['OPENAI_BASE_URL must be unset']),
                 ({'OPENAI_API_KEY': DUMMY, 'OPENAI_CUSTOM_HEADERS': 'x'}, False, False, ['OPENAI_CUSTOM_HEADERS must be unset']),
                 ({'OPENAI_API_KEY': DUMMY, 'OPENAI_LOG': 'debug'}, False, False, ['OPENAI_LOG must be unset']),
                 ({'OPENAI_API_KEY': DUMMY, 'OPENAI_BASE_URL': 'https://api.openai.com/v1'}, False, True, ['--loopback needs']),
                 ({'OPENAI_API_KEY': DUMMY, 'OPENAI_BASE_URL': 'http://127.0.0.1:9/v1'}, False, True, [])]
        for env, paid, loopback, expected in cases:
            with mock.patch.dict(os.environ, {**CLEAN_ENV, **env}, clear=True), mock.patch.object(importlib.util, 'find_spec', return_value=object()):
                problems = api_smoke.env_problems(loopback, paid)
            self.assertEqual(len(problems), len(expected), (env, problems))
            for text in expected:
                self.assertTrue(any(text in p for p in problems), (text, problems))
        with mock.patch.dict(os.environ, {**CLEAN_ENV, 'OPENAI_API_KEY': DUMMY}, clear=True), mock.patch.object(importlib.util, 'find_spec', return_value=None):
            self.assertIn('openai SDK is not importable', api_smoke.env_problems(False, False)[0])

    def journal(self, name, state, started='2026-11-07T10:00:00'):
        (self.out / name).mkdir()
        (self.out / name / 'journal.json').write_text(json.dumps({'state': state, 'started_at': started}), encoding='utf-8')

    def test_a_run_is_never_resent_unknowingly(self):
        problems = lambda name, after=None, decisions=1: api_smoke.journal_problems(self.out, name, after, decisions)
        self.assertEqual(problems('smoke-001'), [])
        self.assertTrue(problems('Smoke 1'))
        self.assertTrue(problems('smoke-002', decisions=2))  # more than one decision only after an approved run
        self.journal('smoke-001', 'started')  # a run that never ended (killed): its billing is unknown
        self.assertTrue(any('smoke-001 is started' in p for p in problems('smoke-002', after='smoke-001')))
        self.assertTrue(problems('smoke-001'))  # a name is used once
        (self.out / 'smoke-001' / 'journal.json').write_text(json.dumps({'state': 'finished', 'started_at': '2026-11-07T10:00:00'}), encoding='utf-8')
        self.assertTrue(any('--after smoke-001' in p for p in problems('smoke-002')))
        self.assertEqual(problems('smoke-002', after='smoke-001'), [])
        self.assertEqual(problems('smoke-002', after='smoke-001', decisions=3), [])
        self.assertTrue(problems('smoke-002', after='smoke-000'))
        (self.out / 'stray').mkdir()  # a directory without a journal blocks too
        self.assertTrue(any('stray has no journal' in p for p in problems('smoke-002', after='smoke-001')))

    def test_a_refused_run_creates_nothing_and_sends_nothing(self):
        with mock.patch.dict(os.environ, {**CLEAN_ENV, 'OPENAI_API_KEY': DUMMY}, clear=True):  # no gate
            with self.assertRaises(SystemExit) as refused:
                api_smoke.run_smoke('smoke-001', api_smoke.spec_of(1), out=self.out)
        self.assertIn(llm_adapter.PAID_GATE, str(refused.exception))
        self.assertEqual(list(self.out.iterdir()), [])
        with mock.patch.dict(os.environ, {**CLEAN_ENV, 'OPENAI_API_KEY': DUMMY}, clear=True):
            with self.assertRaises(SystemExit):
                api_smoke.main(['run', '--name', 'smoke-001'])  # neither --confirm-paid nor --loopback


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
        if kind == '500':
            return self.reply(500, {'error': {'message': 'internal', 'type': 'server_error', 'code': None}}, n)
        listed = body['input'][-1].get('type') == 'function_call_output'
        order = '0-1-2-3-4' if kind == 'switch' else 'keep_paper'
        name, args = ('commit_order', {'order': order, 'rationale': 'smoke'}) if listed else ('list_order_outcomes', {})
        data = {'id': f'resp_{n}', 'object': 'response', 'created_at': 1, 'model': api_smoke.MODEL, 'status': 'completed', 'error': None,
                'incomplete_details': None, 'output': [
                    {'type': 'reasoning', 'id': f'rs_{n}', 'summary': [], 'encrypted_content': f'gAAAA-{n}'},
                    {'type': 'function_call', 'id': f'fc_{n}', 'call_id': f'call_{n}', 'name': name, 'arguments': json.dumps(args), 'status': 'completed'}],
                'usage': None if kind == 'no_usage' else {'input_tokens': 1200, 'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0},
                                                           'output_tokens': 90, 'output_tokens_details': {'reasoning_tokens': 60}, 'total_tokens': 1290}}
        self.reply(200, data, n)

    def reply(self, status, data, n):
        raw = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('x-request-id', f'req_{n}')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        try:
            self.wfile.write(raw)
        except ConnectionError:  # the client gave up (the timeout case)
            pass


@unittest.skipUnless(SDK, 'the openai SDK is only in the isolated venv')
@unittest.skipUnless(api_smoke.WALL.exists(), 'pilot wall not present')
class LoopbackRuns(unittest.TestCase):
    """Whole smoke runs: main process -> ProcessTransport worker -> real SDK -> 127.0.0.1."""

    def setUp(self):
        self.out, self.server = Path(tempfile.mkdtemp()), Server()
        self.addCleanup(shutil.rmtree, self.out)
        self.addCleanup(self.server.shutdown)

    def run_smoke(self, name, s=None, url=None, after=None):
        env = {**CLEAN_ENV, 'OPENAI_API_KEY': DUMMY, 'OPENAI_BASE_URL': url or self.server.url, llm_adapter.PAID_GATE: '1'}
        with mock.patch.dict(os.environ, env, clear=True):
            return api_smoke.run_smoke(name, s or spec(), out=self.out, after=after, loopback=True)

    def assert_finished(self, name, report, clean):
        journal = json.loads((self.out / name / 'journal.json').read_text(encoding='utf-8'))
        self.assertEqual((journal['state'], journal['clean'], journal['max_http_requests'], journal['mode']), ('finished', clean, 3, 'loopback mock'))
        self.assertEqual((report['check']['ok'], report['key_scan'], report['workers']), (True, [], {'reaped': True, 'ledger_problems': []}), report['check'])
        self.assertFalse(multiprocessing.active_children())
        for path in (self.out / name).rglob('*'):
            if path.is_file():
                self.assertNotIn(DUMMY[:12].encode(), path.read_bytes(), path)

    def test_a_clean_run_is_recorded_checked_and_never_resent(self):
        report = self.run_smoke('smoke-001')
        self.assertEqual(len(self.server.requests), 2)  # list_order_outcomes, then commit_order
        self.assertEqual(self.server.requests[0]['auth'], f'Bearer {DUMMY}')  # the key of OPENAI_API_KEY, sent only to the server
        self.assertEqual((report['http_requests_recorded'], report['decisions'][0]['source'], report['clean']), (2, 'llm', True))
        row = report['requests'][0]
        self.assertEqual((row['outcome'], row['billed'], row['request_id'], row['served_model'], row['usage']['input_tokens']),
                         ('reply', 'reported', 'req_1', api_smoke.MODEL, 1200))
        self.assertAlmostEqual(row['cost_usd_estimate'], (1200 * 1.00 + 90 * 5.00) / 1e6)
        self.assertAlmostEqual(report['cost_usd_estimate'], 2 * (1200 * 1.00 + 90 * 5.00) / 1e6)
        self.assertEqual(report['skips'], ['not_a_decision_point', 'skip:wall_budget'])  # the rest of the trajectory sent nothing
        self.assert_finished('smoke-001', report, True)
        for name, after in (('smoke-001', None), ('smoke-002', None)):  # the same name, or a new one without --after
            with self.assertRaises(SystemExit):
                self.run_smoke(name, after=after)
        self.assertEqual(len(self.server.requests), 2)  # nothing resent
        with mock.patch.object(api_smoke, 'OUT', self.out):
            self.assertEqual(api_smoke.main(['check', '--name', 'smoke-001']), 0)  # offline again: Transcript Replay, key scan
            with mock.patch.dict(os.environ, {**CLEAN_ENV, 'OPENAI_API_KEY': DUMMY}, clear=True):
                self.assertEqual(api_smoke.main(['preflight', '--name', 'smoke-002']), 1)  # --after smoke-001 missing
                self.assertEqual(api_smoke.main(['preflight', '--name', 'smoke-002', '--after', 'smoke-001']), 0)
        self.assertEqual(len(self.server.requests), 2)

    def test_a_switched_order_is_checked_on_the_base_trajectory(self):
        self.server.script = ['ok', 'switch']
        report = self.run_smoke('smoke-001')
        self.assertEqual((report['decisions'][0]['order'], report['check']['ok']), ('0-1-2-3-4', True), report['check'])
        self.assertTrue(report['check']['expected_on_base_trajectory'])  # traced Paper, controller 0-1-2-3-4: listed apart
        self.assert_finished('smoke-001', report, True)

    def stopped(self, name, report, requests, needle):
        self.assertEqual(len(self.server.requests), requests)
        decision = report['decisions'][0]
        self.assertEqual((decision['source'], decision['calls'], report['clean']), ('fallback:llm_error', 1, False))
        self.assertGreaterEqual(report['unknown_usage'], 1)
        self.assertEqual([(r['outcome'], r['billed']) for r in report['requests']], [(r, 'unknown') for r in (needle,)])
        self.assert_finished(name, report, False)

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

    def test_a_5xx_or_a_reply_without_usage_stops_after_one_request(self):
        for name, kind, outcome in (('smoke-001', '500', 'failed'), ('smoke-002', 'no_usage', 'reply')):
            self.server.requests.clear()
            self.server.script = [kind]
            report = self.run_smoke(name, after='smoke-001' if name == 'smoke-002' else None)
            self.stopped(name, report, 1, outcome)

    def test_an_insufficient_budget_sends_nothing(self):
        report = self.run_smoke('smoke-001', spec(max_cost_usd=0.0001))
        self.assertEqual((len(self.server.requests), report['http_requests_recorded']), (0, 0))
        self.assertEqual(report['decisions'][0]['source'], 'fallback:llm_error')
        self.assert_finished('smoke-001', report, True)  # nothing unknown: nothing was sent

    def test_an_unfinished_run_blocks_every_new_one(self):
        (self.out / 'smoke-001').mkdir()
        (self.out / 'smoke-001' / 'journal.json').write_text(json.dumps({'state': 'started', 'started_at': 'x'}), encoding='utf-8')
        with self.assertRaises(SystemExit) as refused:
            self.run_smoke('smoke-002', after='smoke-001')
        self.assertIn('smoke-001 is started', str(refused.exception))
        self.assertEqual((len(self.server.requests), sorted(p.name for p in self.out.iterdir())), (0, ['smoke-001']))


if __name__ == '__main__':
    unittest.main()
