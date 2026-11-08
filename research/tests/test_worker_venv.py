"""The controller worker in the registered venv (controller_worker python=...): the manifest check,
the environment the worker gets, its failures (the same as a multiprocessing worker's), that
Wang's spawn executable is never touched, that the key leaves Wang's process, and the paid gate's
loopback exemption."""
import importlib.util
import json
import multiprocessing
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'research' / 'tests')]
from research.controller import worker as w  # noqa: E402
from research.controller import order_controller  # noqa: E402
from research.controller import llm_api  # noqa: E402
from test_order_controller import misbehaving_worker  # noqa: E402

VENV = w.VENV_MANIFEST.exists() and not w.venv_problems()
REQUEST = {'observation': {'step': 1, 'phase': 'single', 'sample_trial': 0, 'side': 'left', 'n_remaining': 5},
           'base_order': '3-0-2-1-4', 'orders': []}
FAKE_KEY = 'sk-loopback-' + 'x' * 30
API = {'kind': 'react', 'llm': {'kind': 'api', 'provider': 'openai', 'model': 'gpt-6.1-terra', 'price_per_mtok': [1, 5, 2.5, 0.05]},
       'max_total_tokens': 1000, 'max_cost_usd': 0.01}


def transport(spec, **kw):
    t = w.ProcessTransport(order_controller.full_spec(spec) | kw.pop('extra', {}), python=w.worker_python(), **kw)
    t.log = (log := []).append
    return t, log


class Environment(unittest.TestCase):
    def test_only_the_listed_variables_reach_a_worker(self):
        env = {'SYSTEMROOT': 'C:\\W', 'PATH': 'p', 'HTTPS_PROXY': 'http://p:1', 'OPENAI_API_KEY': FAKE_KEY, 'OTHER': '1',
               llm_api.PAID_GATE: '1', 'OPENAI_BASE_URL': 'https://example.com/v1', 'ANTHROPIC_API_KEY': 'a'}
        self.assertEqual(w.worker_env(order_controller.full_spec({'kind': 'null'}), env), {'SYSTEMROOT': 'C:\\W', 'PATH': 'p'})
        self.assertEqual(w.worker_env(order_controller.full_spec(API), env),
                         {'SYSTEMROOT': 'C:\\W', 'PATH': 'p', 'OPENAI_API_KEY': FAKE_KEY, llm_api.PAID_GATE: '1'})
        loop = w.worker_env(order_controller.full_spec(API), {**env, 'OPENAI_BASE_URL': 'http://127.0.0.1:9/v1'})
        self.assertEqual(loop['OPENAI_BASE_URL'], 'http://127.0.0.1:9/v1')  # a mock server only

    def test_the_gate_is_waived_only_for_a_loopback_endpoint(self):
        llm = API['llm']
        with mock.patch.dict(os.environ, {'OPENAI_API_KEY': FAKE_KEY}, clear=True):
            with self.assertRaises(NotImplementedError):
                llm_api.enabled(llm)
            for url in ('https://api.openai.com/v1', 'http://127.0.0.2.example.com/v1', 'http://10.0.0.1/v1'):
                with mock.patch.dict(os.environ, {'OPENAI_BASE_URL': url}), self.assertRaises(NotImplementedError):
                    llm_api.enabled(llm)
            with mock.patch.dict(os.environ, {'OPENAI_BASE_URL': 'http://127.0.0.1:9/v1'}):
                llm_api.enabled(llm)
                with self.assertRaises(NotImplementedError):  # anthropic has no loopback mode
                    llm_api.enabled({**llm, 'provider': 'anthropic', 'model': 'claude-opus-5-5'})

    @unittest.skipUnless(importlib.util.find_spec('openai'), 'the openai SDK is only in the venv')
    def test_without_the_gate_a_real_client_sends_only_to_loopback(self):
        with mock.patch.dict(os.environ, {'OPENAI_API_KEY': FAKE_KEY}, clear=True):
            sender = llm_api.OpenAISender(llm_api.full(API['llm']))
            with mock.patch.object(sender.client.responses, 'create', side_effect=AssertionError('sent')):
                with self.assertRaises(llm_api.ProviderError) as refused:
                    sender({'model': 'gpt-6.1-terra'}, 1)
            self.assertEqual(refused.exception.billed, 'none')


@unittest.skipUnless(VENV, 'the registered worker venv is absent or changed')
class VenvWorker(unittest.TestCase):
    def test_manifest_changes_are_refused(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        m = json.loads(w.VENV_MANIFEST.read_text(encoding='utf-8'))
        for change in ({'files': {**m['files'], 'pyvenv.cfg': '0' * 64}}, {'files': {**m['files'], 'extra.py': '0'}},
                       {'base_prefix': 'C:\\elsewhere'}, {'sdks': {'openai': '0.0.1'}}):
            (tmp / 'm.json').write_text(json.dumps({**m, **change}), encoding='utf-8')
            self.assertTrue(w.venv_problems(tmp / 'm.json'), change.keys())
            with self.assertRaises(RuntimeError):
                w.worker_python(tmp / 'm.json')

    def test_a_round_trip_leaves_spawn_alone_and_is_reaped(self):
        before = multiprocessing.spawn.get_executable()
        t, log = transport({'kind': 'null'})
        self.assertEqual(t.decide(REQUEST), ('ok', {'order': '3-0-2-1-4', 'source': 'null'}))
        self.assertEqual(t.decide(REQUEST)[0], 'ok')
        self.assertEqual(multiprocessing.spawn.get_executable(), before)  # Wang's Pool keeps this interpreter
        t.close()
        self.assertEqual([e['action'] for e in log], ['start', 'close'])
        self.assertIs(log[1]['alive'], False)
        self.assertEqual(order_controller.worker_problems(log, False)[0], [])

    def test_failures_are_those_of_a_multiprocessing_worker(self):
        for bad, status in (('exit', 'worker_error'), ('garbage', 'worker_protocol_error'), ('huge', 'worker_protocol_error'),
                            ('error', 'worker_error'), ('not_ready', 'worker_error')):
            t, log = transport({'kind': 'null'}, target=misbehaving_worker, extra={'bad': bad})
            self.assertEqual(t.decide(REQUEST)[0], status, bad)
            self.assertIsNone(t.process)  # killed and reaped at once
            self.assertEqual([e['action'] for e in log], ['start', 'close'], bad)
            self.assertIs(log[-1]['alive'], False, bad)
        t, log = transport({'kind': 'null'}, target=misbehaving_worker, extra={'bad': 'hang'}, timeout=3)
        self.assertEqual(t.decide(REQUEST)[0], 'worker_timeout')
        self.assertIs(log[-1]['alive'], False)

    def test_the_key_leaves_wangs_process(self):
        with mock.patch.dict(os.environ, {'OPENAI_API_KEY': FAKE_KEY, 'OPENAI_BASE_URL': 'http://127.0.0.1:9/v1'}):
            from research.llm_walls import gpt_wall
            client = order_controller.make_controller(gpt_wall.SPEC, sink=lambda r: None, worker_venv=True)
            self.assertNotIn('OPENAI_API_KEY', os.environ)
            self.assertEqual(client.transport.env['OPENAI_API_KEY'], FAKE_KEY)
            self.assertEqual(client.transport.python, w.worker_python())
        client.transport.close()


if __name__ == '__main__':
    unittest.main()
