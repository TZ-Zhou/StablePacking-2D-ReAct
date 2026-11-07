"""Tests of the API smoke entry (research/api_smoke.py); no real API anywhere.

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
sys.path.insert(0, str(REPO / 'research'))
import api_smoke  # noqa: E402
import controllers  # noqa: E402
import llm_adapter  # noqa: E402

DUMMY = 'sk-smoke-dummy-' + 'k' * 40
SDK = importlib.util.find_spec('openai') is not None
DROP = {'OPENAI_API_KEY', llm_adapter.PAID_GATE, *llm_adapter.CONNECTION_ENV}
CLEAN_ENV = {k: v for k, v in os.environ.items() if k.upper() not in DROP}


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
               'the price table': spec(llm={'price_per_mtok': None}), 'the price table PRICE': spec(llm={'price_per_mtok': [0, 0, 0, 0]}),
               'model gpt-6.1-terra': spec(llm={'model': 'gpt-other'}), 'not a valid spec': spec(llm={'api_key_env': 'X'})}
        for needle, s in bad.items():
            problems = api_smoke.spec_problems(s)
            self.assertTrue(any(needle in p for p in problems), (needle, problems))

    def env(self, env, paid=False, loopback=False):
        with mock.patch.dict(os.environ, {**CLEAN_ENV, 'OPENAI_API_KEY': DUMMY, **env}, clear=True), \
                mock.patch.object(importlib.util, 'find_spec', return_value=object()):
            return api_smoke.env_problems(loopback, paid)

    def test_the_environment_rules(self):
        with mock.patch.dict(os.environ, CLEAN_ENV, clear=True), mock.patch.object(importlib.util, 'find_spec', return_value=object()):
            problems = api_smoke.env_problems(False, True)
        self.assertEqual(len(problems), 2, problems)  # no gate, no key
        self.assertEqual(self.env({llm_adapter.PAID_GATE: '1'}, paid=True), [])
        for name in llm_adapter.CONNECTION_ENV:  # every connection setting, the official endpoint too, in any case
            for spelled in (name, name.lower()):
                value = 'https://api.openai.com/v1' if name == 'OPENAI_BASE_URL' else 'secret-value-x'
                problems = self.env({llm_adapter.PAID_GATE: '1', spelled: value}, paid=True)
                self.assertEqual(len(problems), 1, (spelled, problems))
                self.assertIn(f'{name} must be unset'.upper(), problems[0].upper())  # Windows names are upper case
                self.assertNotIn(value, problems[0])  # names, never values
        self.assertEqual(len(self.env({'OPENAI_BASE_URL': 'http://127.0.0.1:9/v1'})), 1)  # a paid run refuses even a loopback endpoint
        self.assertEqual(self.env({'OPENAI_BASE_URL': 'http://127.0.0.1:9/v1'}, loopback=True), [])
        self.assertTrue(any('--loopback needs' in p for p in self.env({}, loopback=True)))
        self.assertTrue(self.env({'OPENAI_BASE_URL': 'https://api.openai.com/v1'}, loopback=True))
        self.assertTrue(self.env({'OPENAI_BASE_URL': 'http://127.0.0.1:9/v1', 'HTTPS_PROXY': 'http://p:1'}, loopback=True))
        with mock.patch.dict(os.environ, {**CLEAN_ENV, 'OPENAI_API_KEY': DUMMY}, clear=True), mock.patch.object(importlib.util, 'find_spec', return_value=None):
            self.assertIn('openai SDK is not importable', api_smoke.env_problems(False, False)[0])

    def journal(self, name, state, started='2026-11-07T10:00:00', **more):
        (self.out / name).mkdir(exist_ok=True)
        (self.out / name / 'journal.json').write_text(json.dumps({'state': state, 'started_at': started, **more}), encoding='utf-8')

    def test_a_run_is_never_resent_unknowingly(self):
        problems = lambda name, after=None, decisions=1: api_smoke.journal_problems(self.out, name, after, decisions)
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
        (self.out / api_smoke.LOCK).write_text('{}', encoding='utf-8')  # another run holds the lock, or a killed one left it
        self.assertTrue(any('.lock exists' in p for p in problems('smoke-002', after='smoke-001')))
        self.assertEqual(api_smoke.journal_problems(self.out, 'smoke-002', 'smoke-001', 1, holding=True), [])
        (self.out / api_smoke.LOCK).unlink()
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


@unittest.skipUnless(SDK, 'the openai SDK is only in the isolated venv')
class SenderClient(unittest.TestCase):
    def test_the_real_client_follows_no_redirect_and_reads_nothing_from_the_environment(self):
        with mock.patch.dict(os.environ, {**CLEAN_ENV, 'OPENAI_API_KEY': DUMMY}, clear=True):
            sender = llm_adapter.OpenAISender(llm_adapter.full(api_smoke.spec_of(1)['llm']))
            http = sender.client._client
            self.assertEqual((http.follow_redirects, http.trust_env, sender.client.max_retries), (False, False, 0))
            self.assertEqual(str(sender.client.base_url), 'https://api.openai.com/v1/')
            for name in llm_adapter.CONNECTION_ENV:
                value = 'https://example.com/v1' if name == 'OPENAI_BASE_URL' else 'secret-value-x'
                with mock.patch.dict(os.environ, {name: value}):
                    with self.assertRaises(ValueError) as refused:
                        llm_adapter.OpenAISender(llm_adapter.full(api_smoke.spec_of(1)['llm']))
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
        order = '0-1-2-3-4' if kind == 'switch' else 'keep_paper'
        name, args = ('commit_order', {'order': order, 'rationale': 'smoke'}) if listed else ('list_order_outcomes', {})
        data = {'id': f'resp_{n}', 'object': 'response', 'created_at': 1, 'model': api_smoke.MODEL, 'status': 'completed', 'error': None,
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
@unittest.skipUnless(api_smoke.WALL.exists(), 'pilot wall not present')
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
                                            llm_adapter.PAID_GATE: '1'}, clear=True)

    def run_smoke(self, name, s=None, url=None, after=None, out=None):
        with self.env(url):
            return api_smoke.run_smoke(name, s or spec(), out=out or self.out, after=after, loopback=True)

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
        self.assertFalse((out / api_smoke.LOCK).exists())  # released after the final journal state
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
                         ('reply', 'reported', 'req_1', api_smoke.MODEL, 1200))
        self.assertAlmostEqual(row['cost_usd_estimate'], (1200 * 1.00 + 90 * 5.00) / 1e6)
        self.assertAlmostEqual(report['cost_usd_estimate'], 2 * (1200 * 1.00 + 90 * 5.00) / 1e6)
        self.assertEqual(report['skips'], ['not_a_decision_point', 'skip:wall_budget'])  # the rest of the trajectory sent nothing
        self.assert_ended('smoke-001', report, 'finished', True, api_success=True)
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
        self.assert_ended('smoke-001', report, 'finished', True, api_success=True)
        run, spec_ = self.out / 'smoke-001', controllers.full_spec(self.journal('smoke-001')['spec'])
        result = api_smoke.check_case.result_dirs(run)[0]
        path = result / 'trace' / 'controller.jsonl'
        records = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()]
        switched = next(r for r in records if r['final']['order'] == '0-1-2-3-4')
        # a controller selection error at the switched event: neither its record nor its pick is excused
        bad = [{**r, 'final': {**r['final'], 'index': r['final']['index'] + 1}} if r is switched else r for r in records]
        path.write_text(''.join(json.dumps(r) + '\n' for r in bad), encoding='utf-8')
        report = api_smoke.offline_check(run, spec_)
        self.assertFalse(report['ok'])
        at = [p for p in report['problems'] if p.startswith(f"event {switched['event']} ")]
        self.assertTrue(any('record differs' in p for p in at) and any('traced policy' in p for p in at), at)  # the order difference no longer excused
        self.assertEqual(report['expected_on_base_trajectory'], [])
        path.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
        # a traced policy that is not the base one (another tol) at the switched event is no expected difference
        real = controllers.trace_events

        def other_tol(result_):
            for meta, a in real(result_):
                yield ({**meta, 'policy': {**meta['policy'], 'tol': meta['policy']['tol'] + 0.5}} if meta['event'] == switched['event'] else meta), a
        with mock.patch.object(controllers, 'trace_events', other_tol):
            report = api_smoke.offline_check(run, spec_)
        self.assertFalse(report['ok'])
        self.assertTrue(any('traced policy' in p and f"event {switched['event']} " in p for p in report['problems']), report['problems'])
        self.assertTrue(api_smoke.offline_check(run, spec_)['ok'])  # the records restored: the check passes again

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
                api_smoke.run_smoke('smoke-001', spec(), out=self.out, loopback=True)
            except BaseException as error:
                errors.append(error)
        with self.env():
            thread = threading.Thread(target=first)
            thread.start()
            while not self.server.requests and thread.is_alive():
                time.sleep(0.05)
            self.assertTrue((self.out / api_smoke.LOCK).exists())
            with self.assertRaises(SystemExit) as refused:
                api_smoke.run_smoke('smoke-002', spec(), out=self.out, loopback=True)
            thread.join()
        self.assertIn('.lock exists', str(refused.exception))
        self.assertEqual((errors, sorted(p.name for p in self.out.iterdir())), ([], ['smoke-001']))
        self.assertEqual(len(self.server.requests), 2)  # the first run's two requests, none of the second
        self.assertEqual(self.journal('smoke-001')['state'], 'finished')

    def test_a_lock_left_behind_or_an_unfinished_run_blocks_every_new_one(self):
        (self.out / api_smoke.LOCK).write_text('{"name": "smoke-000"}', encoding='utf-8')  # a run killed while it held the lock
        with self.assertRaises(SystemExit) as refused:
            self.run_smoke('smoke-001')
        self.assertIn('.lock exists', str(refused.exception))
        self.assertEqual((self.out / api_smoke.LOCK).read_text(encoding='utf-8'), '{"name": "smoke-000"}')  # never removed automatically
        (self.out / api_smoke.LOCK).unlink()
        (self.out / 'smoke-001').mkdir()
        (self.out / 'smoke-001' / 'journal.json').write_text(json.dumps({'state': 'started', 'started_at': 'x'}), encoding='utf-8')
        with self.assertRaises(SystemExit) as refused:
            self.run_smoke('smoke-002', after='smoke-001')
        self.assertIn('smoke-001 is started', str(refused.exception))
        self.assertEqual((len(self.server.requests), sorted(p.name for p in self.out.iterdir())), (0, ['smoke-001']))


if __name__ == '__main__':
    unittest.main()
