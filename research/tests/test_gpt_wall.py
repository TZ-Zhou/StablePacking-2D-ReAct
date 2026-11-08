"""research/gpt_wall.py: its refusals (spec limits, environment, journal chain, experiment budget,
formal seeds, paid-only rules) and short live walls against a local mock Responses API with a
fake key: a clean one, a redirect and a reply without usage (both uncertain, blocking the next)."""
import datetime
import json
import os
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'research'), str(REPO / 'research' / 'tests')]
import controller_worker  # noqa: E402
import gpt_wall as g  # noqa: E402
import llm_adapter  # noqa: E402
from test_api_smoke import Server  # noqa: E402

KEY = g.LOOPBACK_KEY + 'k' * 40
SEED = 66794912  # development seed A
CLEAN = {k: v for k, v in os.environ.items() if k.upper() not in {'OPENAI_API_KEY', llm_adapter.PAID_GATE, *llm_adapter.CONNECTION_ENV}}
VENV = controller_worker.VENV_MANIFEST.exists() and not controller_worker.venv_problems()


def scratch(test):
    out = g.run_case.RUNS / 'tmp-tests' / uuid.uuid4().hex[:12]
    test.addCleanup(shutil.rmtree, out, True)
    return out


def journal(out, name, state='finished', started='2026-11-08T10:00:00', spend=None):
    (out / 'journals').mkdir(parents=True, exist_ok=True)
    (out / 'journals' / f'{name}.json').write_text(json.dumps({'state': state, 'started_at': started, 'spend': spend or {}}), encoding='utf-8')


class Rules(unittest.TestCase):
    def test_the_spec_fits_the_wall_limits(self):
        self.assertEqual(g.spec_problems(), [])
        self.assertEqual(g.max_requests(g.controllers.full_spec(g.SPEC)), 60)
        self.assertTrue(g.spec_problems({**g.SPEC, 'llm': {**g.SPEC['llm'], 'max_retries': 1}, 'llm_timeout': 200.0}))
        self.assertTrue(g.spec_problems({**g.SPEC, 'max_decisions': 21}))
        self.assertTrue(g.spec_problems({**g.SPEC, 'max_cost_usd': 0.61}))
        self.assertLessEqual(2 * g.WALL_LIMITS['cost_usd'], g.EXPERIMENT_LIMITS['cost_usd'])

    def test_environment(self):
        loop, real = {'OPENAI_API_KEY': KEY, 'OPENAI_BASE_URL': 'http://127.0.0.1:9/v1'}, {'OPENAI_API_KEY': 'sk-real' + 'r' * 40}
        self.assertEqual(g.env_problems(True, loop), [])
        self.assertTrue(g.env_problems(True, {**loop, llm_adapter.PAID_GATE: '1'}))  # loopback never with the gate
        self.assertTrue(g.env_problems(True, {**loop, 'OPENAI_API_KEY': 'sk-real' + 'r' * 40}))  # never a real key to a mock
        self.assertTrue(g.env_problems(True, {'OPENAI_API_KEY': KEY}))
        self.assertTrue(g.env_problems(True, {**loop, 'OPENAI_BASE_URL': 'https://api.openai.com/v1'}))
        self.assertTrue(g.env_problems(True, {**loop, 'HTTPS_PROXY': 'http://p:1'}))
        self.assertEqual(g.env_problems(False, {**real, llm_adapter.PAID_GATE: '1'}), [])
        self.assertTrue(g.env_problems(False, real))  # no gate
        self.assertTrue(g.env_problems(False, {**real, llm_adapter.PAID_GATE: '1', 'OPENAI_BASE_URL': 'http://127.0.0.1:9/v1'}))
        self.assertTrue(g.env_problems(False, {'OPENAI_API_KEY': KEY, llm_adapter.PAID_GATE: '1'}))
        self.assertTrue(g.env_problems(False, {llm_adapter.PAID_GATE: '1'}))

    def test_journal_chain_and_experiment_budget(self):
        out = scratch(self)
        self.assertEqual(g.journal_problems(out, 'w-1', None), [])
        self.assertTrue(g.journal_problems(out, 'W 1', None))
        journal(out, 'w-1', spend={'http_requests': 30, 'tokens': 100_000, 'cost_usd': 0.2})
        self.assertTrue(any('--after w-1' in p for p in g.journal_problems(out, 'w-2', None)))
        self.assertEqual(g.journal_problems(out, 'w-2', 'w-1'), [])
        self.assertTrue(any('used once' in p for p in g.journal_problems(out, 'w-1', 'w-1')))
        (out / g.LOCK).write_text('{}')
        self.assertTrue(any('.lock' in p for p in g.journal_problems(out, 'w-2', 'w-1')))
        (out / g.LOCK).unlink()
        journal(out, 'w-2', spend={'http_requests': 30, 'tokens': 100_000, 'cost_usd': 0.2}, started='2026-11-08T11:00:00')
        self.assertTrue(any('experiment: walls' in p for p in g.journal_problems(out, 'w-3', 'w-2')))  # 2 walls at most
        out2 = scratch(self)
        journal(out2, 'w-1', spend={'http_requests': 10, 'tokens': 10, 'cost_usd': 0.7})
        self.assertTrue(any('experiment: cost_usd' in p for p in g.journal_problems(out2, 'w-2', 'w-1')))
        for state in ('started', 'uncertain', 'aborted', 'unreadable'):
            out3 = scratch(self)
            journal(out3, 'w-1', state=state)
            self.assertTrue(any(f'w-1 is {state}' in p for p in g.journal_problems(out3, 'w-2', 'w-1')), state)
        out4 = scratch(self)
        (out4 / 'w-9').mkdir(parents=True)
        self.assertTrue(any('no journal' in p for p in g.journal_problems(out4, 'w-1', None)))

    def test_preflight_rules(self):
        out = scratch(self)
        with mock.patch.dict(os.environ, {**CLEAN, 'OPENAI_API_KEY': KEY, 'OPENAI_BASE_URL': 'http://127.0.0.1:9/v1'}, clear=True), \
                mock.patch.object(controller_worker, 'venv_problems', return_value=[]):
            self.assertEqual(g.preflight('w-1', SEED, out, loopback=True), [])
            self.assertTrue(any('formal campaign' in p for p in g.preflight('w-1', 20, out, loopback=True)))
        with mock.patch.dict(os.environ, {**CLEAN, 'OPENAI_API_KEY': 'sk-real' + 'r' * 40, llm_adapter.PAID_GATE: '1'}, clear=True), \
                mock.patch.object(controller_worker, 'venv_problems', return_value=[]):
            with mock.patch.object(g, 'code_dirty', return_value=' M research/gpt_wall.py'):
                self.assertTrue(any('commit them' in p for p in g.preflight('w-1', SEED, out)))
            with mock.patch.object(g, 'code_dirty', return_value=''):
                self.assertEqual(g.preflight('w-1', SEED, out, today=g.PRICE_CHECKED), [])
                late = g.PRICE_CHECKED + datetime.timedelta(days=g.PRICE_MAX_AGE + 1)
                self.assertTrue(any('price' in p for p in g.preflight('w-1', SEED, out, today=late)))

    def test_paid_needs_confirmation_and_placements_need_loopback(self):
        with self.assertRaises(SystemExit) as refused:
            g.main(['run', '--name', 'w-1', '--seed', str(SEED)])
        self.assertIn('--confirm-paid', str(refused.exception))
        with self.assertRaises(SystemExit):
            g.run_wall('w-1', SEED, scratch(self), placements=3)

    def test_admission_is_checked_again_under_the_lock(self):
        out = scratch(self)
        launch = mock.Mock()

        def admitted_then_doubtful(*a, **k):  # another run ends uncertain between preflight and the lock
            journal(out, 'w-0', state='uncertain')
            return []
        with mock.patch.object(g, 'preflight', side_effect=admitted_then_doubtful), self.assertRaises(SystemExit) as refused:
            g.run_wall('w-1', SEED, out, loopback=True, launch=launch)
        self.assertIn('w-0 is uncertain', str(refused.exception))
        launch.assert_not_called()
        self.assertFalse((out / g.LOCK).exists())
        self.assertFalse((out / 'journals' / 'w-1.json').exists())

    def test_a_failing_git_is_never_a_clean_tree(self):
        out = Path(tempfile.mkdtemp())  # outside the repository: not a git work tree
        self.addCleanup(shutil.rmtree, out, True)
        with mock.patch.object(g.run_case, 'REPO', out):
            for check in (g.code_dirty, g.head):
                with self.assertRaises(g.subprocess.CalledProcessError):
                    check()

    def test_an_exception_leaves_the_wall_aborted_and_blocking(self):
        out = scratch(self)
        boom = mock.Mock(side_effect=RuntimeError('case process tree not verified stopped'))
        with mock.patch.dict(os.environ, {**CLEAN, 'OPENAI_API_KEY': KEY, 'OPENAI_BASE_URL': 'http://127.0.0.1:9/v1'}, clear=True), \
                mock.patch.object(controller_worker, 'venv_problems', return_value=[]), \
                mock.patch.object(g.run_case, 'worker_venv_identity', return_value={}), self.assertRaises(RuntimeError):
            g.run_wall('w-1', SEED, out, loopback=True, launch=boom)
        j = json.loads((out / 'journals' / 'w-1.json').read_text(encoding='utf-8'))
        self.assertEqual(j['state'], 'aborted')
        self.assertFalse((out / g.LOCK).exists())
        self.assertTrue(any('w-1 is aborted' in p for p in g.journal_problems(out, 'w-2', 'w-1')))


class Launcher(unittest.TestCase):
    """run_case: children other than the case never get the key; an exception never leaves a case running."""
    BASELINE = Path(sys.prefix).resolve() == g.run_case.ENV_PREFIX

    def test_other_children_get_no_key(self):
        with mock.patch.dict(os.environ, {'OPENAI_API_KEY': KEY, 'ANTHROPIC_API_KEY': 'a', 'PATH': 'p'}):
            env = g.run_case.quiet_env()
        self.assertFalse({'OPENAI_API_KEY', 'ANTHROPIC_API_KEY'} & set(env))
        self.assertEqual(env['PATH'], 'p')

    def test_a_failing_process_listing_still_kills_and_is_unverified(self):
        child = g.subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
        self.addCleanup(child.kill)
        listing = g.subprocess.CalledProcessError(1, 'powershell')
        for failing in ([listing], [{}, listing]):  # before the kill, or while verifying it
            with mock.patch.object(g.run_case, 'process_table', side_effect=failing):
                if failing[0] == {}:
                    child = g.subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
                    self.addCleanup(child.kill)
                record = g.run_case.terminate(child.pid, settle_seconds=1)
            self.assertIs(record['verified'], False)
            self.assertIn('CalledProcessError', record['listing_error'])
            self.assertIsNotNone(child.wait(timeout=30))  # killed all the same

    @unittest.skipUnless(BASELINE, 'run_case builds walls only in the baseline environment')
    def test_an_exception_after_the_case_started_stops_it(self):
        name = f'tmp-tests/{uuid.uuid4().hex[:12]}'
        self.addCleanup(shutil.rmtree, g.run_case.RUNS / name, True)
        real, calls = g.run_case.committed_placements, []

        def interrupted(output):  # the launcher's loop fails with the case running
            calls.append(output)
            if len(calls) == 1:
                raise RuntimeError('launcher interrupted')
            return real(output)
        with mock.patch.object(g.run_case, 'committed_placements', side_effect=interrupted), self.assertRaises(RuntimeError):
            g.run_case.main(['thangavelu-noise80', name, '--policy', 'paper', '--seed', str(SEED), '--placements', '99'])
        receipt = json.loads((g.run_case.RUNS / name / 'receipt.json').read_text(encoding='utf-8'))
        self.assertEqual(receipt['status'], 'LAUNCHER_ERROR')
        self.assertIs(receipt['termination']['verified'], True)
        self.assertIsNotNone(receipt['returncode'])  # the case is gone


@unittest.skipUnless(VENV, 'the registered worker venv is absent or changed')
class LoopbackWalls(unittest.TestCase):
    """Live walls of 4 placements: Wang's process -> venv worker -> real SDK -> a mock on 127.0.0.1."""

    def wall(self, script=()):
        server, out = Server(), scratch(self)
        self.addCleanup(server.shutdown)
        server.script = list(script)
        env = {**CLEAN, 'OPENAI_API_KEY': KEY, 'OPENAI_BASE_URL': server.url}
        with mock.patch.dict(os.environ, env, clear=True):
            report = g.run_wall('lb-1', SEED, out, loopback=True, placements=4)
            blocked = g.journal_problems(out, 'lb-2', 'lb-1')
        j = json.loads((out / 'journals' / 'lb-1.json').read_text(encoding='utf-8'))
        self.assertFalse((out / g.LOCK).exists())
        for path in (out / 'lb-1').rglob('*'):
            if path.is_file():
                self.assertNotIn(KEY[:16].encode(), path.read_bytes(), path)
        self.assertEqual(report['key_scan'], [])
        return server, report, j, blocked

    def test_a_clean_wall_is_finished(self):
        server, report, j, blocked = self.wall()
        self.assertEqual((j['state'], j['reasons']), ('finished', []), report.get('check'))
        self.assertEqual(report['check']['verdict'], 'PASS')
        self.assertGreater(report['llm_decisions'], 0)
        self.assertEqual(len(server.requests), report['http_requests'])
        self.assertEqual(report['http_requests'], 2 * report['llm_decisions'])  # list, then commit
        self.assertTrue(all(r['auth'] == f'Bearer {KEY}' for r in server.requests))
        self.assertEqual((report['unknown_usage'], report['worker_failures'], report['workers_reaped']), (0, [], True))
        self.assertEqual(blocked, [])

    def test_a_redirect_is_one_post_and_uncertain(self):
        server, report, j, blocked = self.wall(['307'])
        self.assertEqual(j['state'], 'uncertain')
        self.assertIn('unknown usage', j['reasons'])
        self.assertEqual([r['path'] for r in server.requests], ['/v1/responses'])  # not followed, nothing after it
        self.assertTrue(any('lb-1 is uncertain' in p for p in blocked))

    def test_a_reply_without_usage_is_uncertain(self):
        server, report, j, blocked = self.wall(['no_usage'])
        self.assertEqual(j['state'], 'uncertain')
        self.assertEqual(len(server.requests), 1)  # the first unknown usage stops every later request
        self.assertTrue(blocked)


if __name__ == '__main__':
    unittest.main()
