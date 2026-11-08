"""research/llm_walls/gpt_wall.py: its refusals (spec limits, environment, journal chain, experiment budget,
formal seeds, paid-only rules), where a wall's records land (its case, the study) with the launch mocked,
and short live walls against a local mock Responses API with a fake key: a clean one, a redirect and a
reply without usage (both uncertain, blocking the next). Every test uses temporary cases, studies and archives."""
import contextlib
import datetime
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'research' / 'tests')]
from research.controller import worker as controller_worker  # noqa: E402
from research.llm_walls import gpt_wall as g  # noqa: E402
from research.runner import gpt_wall as runner  # noqa: E402
from research.walls import case_definition  # noqa: E402
from research.controller import llm_api  # noqa: E402
from research.storage import case_id, studies  # noqa: E402
import test_gpt_smoke  # noqa: E402
from test_gpt_smoke import Server  # noqa: E402

KEY = g.LOOPBACK_KEY + 'k' * 40
SEED = 66794912  # development seed A
CLEAN = {k: v for k, v in os.environ.items() if k.upper() not in {'OPENAI_API_KEY', llm_api.PAID_GATE, *llm_api.CONNECTION_ENV}}
VENV = controller_worker.VENV_MANIFEST.exists() and not controller_worker.venv_problems()
BASELINE = Path(sys.prefix).resolve() == runner.wall_runner.ENV_PREFIX  # run_wall builds walls only there


def isolate(test):
    """Temporary research/cases, research/studies and gpt-walls archives for one test; returns the root."""
    root = test_gpt_smoke.isolate(test)
    archive = root / 'archive'
    patcher = mock.patch.object(g, 'LEGACY', {False: archive / 'gpt-walls', True: archive / 'gpt-walls-loopback'})
    patcher.start()
    test.addCleanup(patcher.stop)
    return root


def journal(name, state='finished', started='2026-11-08T10:00:00', spend=None, loopback=True):
    """A wall of the (temporary) archive: LEGACY/journals/<name>.json."""
    journals = g.LEGACY[loopback] / 'journals'
    journals.mkdir(parents=True, exist_ok=True)
    text = json.dumps({'state': state, 'started_at': started, 'spend': spend or {}})
    (journals / f'{name}.json').write_text(text, encoding='utf-8')


def lock_file(loopback=True):
    lock = g.lock_of(loopback)
    lock.parent.mkdir(parents=True, exist_ok=True)
    return lock


def launcher():
    """A launch in place of run_wall.run, which builds nothing: a new case folder with a receipt and an
    empty result directory. Returns (launch, the argv of each call)."""
    calls = []

    def launch(argv):
        calls.append(argv)
        new_id, folder = case_id.allocate('thangavelu-noise80', hostname='GIAAIL-STRIX')
        (folder / 'result_20261108-000000_wall0').mkdir()
        receipt = {'schema_version': 2, 'case_id': new_id, 'scene': 'TH_T80', 'parent': None, 'run_id': new_id,
                   'status': 'PROCESS_PASS', 'process_status': 'PROCESS_PASS', 'elapsed_seconds': 1.0}
        (folder / 'receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
        return receipt
    return launch, calls


ACCOUNTED = {'http_requests': 2, 'http_requests_exact': True, 'tokens': 100, 'cost_usd': 0.001, 'unknown_usage': 0,
             'worker_failures': [], 'workers_reaped': True, 'decision_points': 1, 'llm_decisions': 1, 'switches': 0,
             'skips_and_fallbacks': [], 'gpt_complete': True, 'requests': []}


class Rules(unittest.TestCase):
    def test_the_spec_fits_the_wall_limits(self):
        self.assertEqual(g.spec_problems(), [])
        self.assertEqual(g.max_requests(g.order_controller.full_spec(g.SPEC)), 60)
        self.assertTrue(g.spec_problems({**g.SPEC, 'llm': {**g.SPEC['llm'], 'max_retries': 1}, 'llm_timeout': 200.0}))
        self.assertTrue(g.spec_problems({**g.SPEC, 'max_decisions': 21}))
        self.assertTrue(g.spec_problems({**g.SPEC, 'max_cost_usd': 0.61}))
        self.assertLessEqual(2 * g.WALL_LIMITS['cost_usd'], g.EXPERIMENT_LIMITS['cost_usd'])

    def test_environment(self):
        loop, real = {'OPENAI_API_KEY': KEY, 'OPENAI_BASE_URL': 'http://127.0.0.1:9/v1'}, {'OPENAI_API_KEY': 'sk-real' + 'r' * 40}
        self.assertEqual(g.env_problems(True, loop), [])
        self.assertTrue(g.env_problems(True, {**loop, llm_api.PAID_GATE: '1'}))  # loopback never with the gate
        self.assertTrue(g.env_problems(True, {**loop, 'OPENAI_API_KEY': 'sk-real' + 'r' * 40}))  # never a real key to a mock
        self.assertTrue(g.env_problems(True, {'OPENAI_API_KEY': KEY}))
        self.assertTrue(g.env_problems(True, {**loop, 'OPENAI_BASE_URL': 'https://api.openai.com/v1'}))
        self.assertTrue(g.env_problems(True, {**loop, 'HTTPS_PROXY': 'http://p:1'}))
        self.assertEqual(g.env_problems(False, {**real, llm_api.PAID_GATE: '1'}), [])
        self.assertTrue(g.env_problems(False, real))  # no gate
        self.assertTrue(g.env_problems(False, {**real, llm_api.PAID_GATE: '1', 'OPENAI_BASE_URL': 'http://127.0.0.1:9/v1'}))
        self.assertTrue(g.env_problems(False, {'OPENAI_API_KEY': KEY, llm_api.PAID_GATE: '1'}))
        self.assertTrue(g.env_problems(False, {llm_api.PAID_GATE: '1'}))

    def test_journal_chain_and_experiment_budget(self):
        isolate(self)
        problems = lambda name, after: g.journal_problems(name, after, True)
        self.assertEqual(problems('w-1', None), [])
        self.assertTrue(problems('W 1', None))
        journal('w-1', spend={'http_requests': 30, 'tokens': 100_000, 'cost_usd': 0.2})
        self.assertTrue(any('--after w-1' in p for p in problems('w-2', None)))
        self.assertEqual(problems('w-2', 'w-1'), [])
        self.assertTrue(any('used once' in p for p in problems('w-1', 'w-1')))
        lock_file().write_text('{}')
        self.assertTrue(any('.lock' in p for p in problems('w-2', 'w-1')))
        lock_file().unlink()
        journal('w-2', spend={'http_requests': 30, 'tokens': 100_000, 'cost_usd': 0.2}, started='2026-11-08T11:00:00')
        self.assertTrue(any('experiment: walls' in p for p in problems('w-3', 'w-2')))  # 2 walls at most
        isolate(self)
        journal('w-1', spend={'http_requests': 10, 'tokens': 10, 'cost_usd': 0.7})
        self.assertTrue(any('experiment: cost_usd' in p for p in problems('w-2', 'w-1')))
        for state in ('started', 'uncertain', 'aborted', 'unreadable'):
            isolate(self)
            journal('w-1', state=state)
            self.assertTrue(any(f'w-1 is {state}' in p for p in problems('w-2', 'w-1')), state)
        isolate(self)
        (g.LEGACY[True] / 'w-9').mkdir(parents=True)
        self.assertTrue(any('no journal' in p for p in problems('w-1', None)))

    def test_the_paid_and_the_loopback_walls_are_apart(self):
        isolate(self)
        journal('w-1', spend={'cost_usd': 0.2}, loopback=False)
        self.assertEqual(list(g.walls(False)), ['w-1'])
        self.assertEqual((g.walls(True), g.spent(True)['walls'], g.spent(False)['walls']), ({}, 0, 1))
        self.assertEqual(g.journal_problems('w-1', None, True), [])
        self.assertEqual((g.STUDIES[False], g.STUDIES[True]), ('gpt-walls', 'gpt-walls-loopback'))

    def test_the_archive_counts_with_the_study(self):
        """A legacy wall (gpt-001) keeps its name, its place in the --after chain and its spend."""
        isolate(self)
        journal('gpt-001', spend={'http_requests': 37, 'tokens': 145_293, 'cost_usd': 0.14}, loopback=False)
        wall = g.walls(False)['gpt-001']
        self.assertEqual(wall.folder, g.LEGACY[False] / 'gpt-001')
        self.assertEqual(wall.report, g.LEGACY[False] / 'gpt-001' / 'report.json')
        self.assertEqual(wall.journal, g.LEGACY[False] / 'journals' / 'gpt-001.json')
        self.assertTrue(any('used once' in p for p in g.journal_problems('gpt-001', 'gpt-001', False)))
        self.assertTrue(any('--after gpt-001' in p for p in g.journal_problems('gpt-002', None, False)))
        self.assertEqual(g.journal_problems('gpt-002', 'gpt-001', False), [])
        self.assertEqual(g.spent(False), {'walls': 1, 'http_requests': 37, 'tokens': 145_293, 'cost_usd': 0.14})

    def test_preflight_rules(self):
        isolate(self)
        with mock.patch.dict(os.environ, {**CLEAN, 'OPENAI_API_KEY': KEY, 'OPENAI_BASE_URL': 'http://127.0.0.1:9/v1'}, clear=True), \
                mock.patch.object(controller_worker, 'venv_problems', return_value=[]):
            self.assertEqual(g.preflight('w-1', SEED, loopback=True), [])
            self.assertTrue(any('formal campaign' in p for p in g.preflight('w-1', 20, loopback=True)))
        with mock.patch.dict(os.environ, {**CLEAN, 'OPENAI_API_KEY': 'sk-real' + 'r' * 40, llm_api.PAID_GATE: '1'}, clear=True), \
                mock.patch.object(controller_worker, 'venv_problems', return_value=[]):
            with mock.patch.object(g, 'code_dirty', return_value=' M research/llm_walls/gpt_wall.py'):
                self.assertTrue(any('commit them' in p for p in g.preflight('w-1', SEED)))
            with mock.patch.object(g, 'code_dirty', return_value=''):
                self.assertEqual(g.preflight('w-1', SEED, today=g.PRICE_CHECKED), [])
                late = g.PRICE_CHECKED + datetime.timedelta(days=g.PRICE_MAX_AGE + 1)
                self.assertTrue(any('price' in p for p in g.preflight('w-1', SEED, today=late)))

    def test_paid_needs_confirmation_and_placements_need_loopback(self):
        with self.assertRaises(SystemExit) as refused:
            runner.main(['run', '--name', 'w-1', '--seed', str(SEED)])
        self.assertIn('--confirm-paid', str(refused.exception))
        root = isolate(self)
        with self.assertRaises(SystemExit):
            runner.run_wall('w-1', SEED, placements=3)
        self.assertEqual([p.name for p in root.iterdir()], ['archive'])

    def test_admission_is_checked_again_under_the_lock(self):
        root = isolate(self)
        launch = mock.Mock()

        def admitted_then_doubtful(*a, **k):  # another run ends uncertain between preflight and the lock
            journal('w-0', state='uncertain')
            return []
        with mock.patch.object(g, 'preflight', side_effect=admitted_then_doubtful), self.assertRaises(SystemExit) as refused:
            runner.run_wall('w-1', SEED, loopback=True, launch=launch)
        self.assertIn('w-0 is uncertain', str(refused.exception))
        launch.assert_not_called()
        self.assertFalse(g.lock_of(True).exists())
        self.assertNotIn('w-1', g.walls(True))
        self.assertFalse((root / 'cases').exists())

    def test_a_failing_git_is_never_a_clean_tree(self):
        out = Path(tempfile.mkdtemp())  # outside the repository: not a git work tree
        self.addCleanup(shutil.rmtree, out, True)
        with mock.patch.object(g, 'REPO', out):
            for check in (g.code_dirty, g.head):
                with self.assertRaises(g.subprocess.CalledProcessError):
                    check()

    def live(self):
        """The environment a loopback wall is admitted in, without the worker venv."""
        loop = {**CLEAN, 'OPENAI_API_KEY': KEY, 'OPENAI_BASE_URL': 'http://127.0.0.1:9/v1'}
        env = mock.patch.dict(os.environ, loop, clear=True)
        venv = mock.patch.object(controller_worker, 'venv_problems', return_value=[])
        identity = mock.patch.object(case_definition, 'worker_venv_identity', return_value={})
        for patcher in (env, venv, identity):
            patcher.start()
            self.addCleanup(patcher.stop)

    @contextlib.contextmanager
    def checked(self):
        """A wall whose accounting and controller check pass, without any trace to read."""
        check = {'ok': True, 'verdict': 'PASS', 'problems': []}
        with mock.patch.object(g, 'accounting', return_value=dict(ACCOUNTED)), \
                mock.patch.object(g.order_controller, 'check', return_value=check):
            yield

    def test_a_launch_that_raises_leaves_the_lock_holding_the_journal(self):
        """The launch raised: its case is unknown, so the lock stays, with the final state, and blocks."""
        root = isolate(self)
        self.live()
        boom = mock.Mock(side_effect=RuntimeError('case process tree not verified stopped'))
        with self.assertRaises(RuntimeError):
            runner.run_wall('w-1', SEED, loopback=True, launch=boom)
        kept = json.loads(g.lock_of(True).read_text(encoding='utf-8'))
        self.assertEqual((kept['run'], kept['journal']['state']), ('w-1', 'aborted'))
        self.assertTrue(any('not verified stopped' in r for r in kept['journal']['reasons']))
        self.assertTrue(any('.lock' in p for p in g.journal_problems('w-2', None, True)))
        self.assertFalse((root / 'cases').exists())

    def test_an_exception_after_the_launch_leaves_the_wall_aborted_and_blocking(self):
        isolate(self)
        self.live()
        launch, _ = launcher()
        failing = mock.patch.object(g, 'accounting', side_effect=RuntimeError('records unreadable'))
        with failing, self.assertRaises(RuntimeError):
            runner.run_wall('w-1', SEED, loopback=True, launch=launch)
        wall = g.walls(True)['w-1']
        j = json.loads(wall.journal.read_text(encoding='utf-8'))
        self.assertEqual(j['state'], 'aborted')
        self.assertTrue(any('records unreadable' in r for r in j['reasons']))
        self.assertFalse(g.lock_of(True).exists())
        self.assertTrue(any('w-1 is aborted' in p for p in g.journal_problems('w-2', 'w-1', True)))

    def test_a_failed_listing_never_releases_the_lock(self):
        """A case the study does not list could not block: without its entry the lock stays."""
        isolate(self)
        self.live()
        launch, _ = launcher()
        failing = mock.patch.object(g.gpt_smoke.studies, 'add_case', side_effect=RuntimeError('study unwritable'))
        with failing, self.assertRaises(RuntimeError):
            runner.run_wall('w-1', SEED, loopback=True, launch=launch)
        self.assertEqual(json.loads(g.lock_of(True).read_text(encoding='utf-8'))['journal']['state'], 'aborted')

    def test_a_wall_lands_in_its_case_and_the_study(self):
        root = isolate(self)
        self.live()
        launch, calls = launcher()
        with self.checked():
            report = runner.run_wall('w-1', SEED, loopback=True, launch=launch)
        self.assertEqual(calls[0][0], 'thangavelu-noise80')  # no run name: run_wall allocates the case
        self.assertEqual(calls[0][1], '--policy')
        self.assertIn('--worker-venv', calls[0])
        self.assertEqual(json.loads(calls[0][calls[0].index('--controller') + 1]), g.SPEC)
        folder = next(p for p in (root / 'cases').iterdir())
        self.assertEqual(studies.load('gpt-walls-loopback')['kind'], 'gpt-walls')
        self.assertEqual(studies.find('gpt-walls-loopback', name='w-1'), [{'case_id': folder.name, 'name': 'w-1'}])
        self.assertEqual(g.walls(True)['w-1'], g.gpt_smoke.case_records(folder))
        j = json.loads((folder / 'llm' / 'journal.json').read_text(encoding='utf-8'))
        self.assertEqual((j['state'], j['reasons'], j['case_id']), ('finished', [], folder.name))
        self.assertEqual((j['study'], j['seed']), ('gpt-walls-loopback', SEED))
        self.assertEqual(j['spend']['http_requests'], 2)
        self.assertEqual(json.loads((folder / 'llm' / 'report.json').read_text(encoding='utf-8')), report)
        self.assertFalse(g.lock_of(True).exists())
        self.assertFalse(studies.path_of('gpt-walls').exists())  # a loopback wall never enters the paid study
        self.assertEqual(g.spent(True)['walls'], 1)
        self.assertTrue(any('--after w-1' in p for p in g.journal_problems('w-2', None, True)))
        self.assertTrue(any('used once' in p for p in g.journal_problems('w-1', 'w-1', True)))
        self.assertEqual(g.journal_problems('w-2', 'w-1', True), [])

    def test_a_key_or_a_failed_status_leaves_the_wall_uncertain(self):
        isolate(self)
        self.live()
        launch, _ = launcher()
        with self.checked(), mock.patch.object(g.gpt_smoke, 'key_scan', return_value=['receipt.json: the key']):
            runner.run_wall('w-1', SEED, loopback=True, launch=launch)
        j = json.loads(g.walls(True)['w-1'].journal.read_text(encoding='utf-8'))
        self.assertEqual((j['state'], j['reasons']), ('uncertain', ['the key in an output file']))
        self.assertTrue(any('w-1 is uncertain' in p for p in g.journal_problems('w-2', 'w-1', True)))


class Launcher(unittest.TestCase):
    """run_wall: children other than the case never get the key; an exception never leaves a case running."""

    def test_other_children_get_no_key(self):
        with mock.patch.dict(os.environ, {'OPENAI_API_KEY': KEY, 'ANTHROPIC_API_KEY': 'a', 'PATH': 'p'}):
            env = case_definition.quiet_env()
        self.assertFalse({'OPENAI_API_KEY', 'ANTHROPIC_API_KEY'} & set(env))
        self.assertEqual(env['PATH'], 'p')

    def test_a_failing_process_listing_still_kills_and_is_unverified(self):
        child = g.subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
        self.addCleanup(child.kill)
        listing = g.subprocess.CalledProcessError(1, 'powershell')
        for failing in ([listing], [{}, listing]):  # before the kill, or while verifying it
            with mock.patch.object(runner.wall_runner, 'process_table', side_effect=failing):
                if failing[0] == {}:
                    child = g.subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
                    self.addCleanup(child.kill)
                record = runner.wall_runner.terminate(child.pid, settle_seconds=1)
            self.assertIs(record['verified'], False)
            self.assertIn('CalledProcessError', record['listing_error'])
            self.assertIsNotNone(child.wait(timeout=30))  # killed all the same

    @unittest.skipUnless(BASELINE, 'run_wall builds walls only in the baseline environment')
    def test_an_exception_after_the_case_started_stops_it(self):
        root = isolate(self)
        real, calls = runner.wall_runner.committed_placements, []

        def interrupted(output):  # the launcher's loop fails with the case running
            calls.append(output)
            if len(calls) == 1:
                raise RuntimeError('launcher interrupted')
            return real(output)
        with mock.patch.object(runner.wall_runner, 'committed_placements', side_effect=interrupted), self.assertRaises(RuntimeError):
            runner.wall_runner.run(['thangavelu-noise80', '--policy', 'paper', '--seed', str(SEED), '--placements', '99'])
        receipt = json.loads((next((root / 'cases').iterdir()) / 'receipt.json').read_text(encoding='utf-8'))
        self.assertEqual(receipt['status'], 'LAUNCHER_ERROR')
        self.assertIs(receipt['termination']['verified'], True)
        self.assertIsNotNone(receipt['returncode'])  # the case is gone


@unittest.skipUnless(VENV and BASELINE, 'needs the registered worker venv, and the baseline environment to build walls')
class LoopbackWalls(unittest.TestCase):
    """Live walls of 4 placements: Wang's process -> venv worker -> real SDK -> a mock on 127.0.0.1."""

    def wall(self, script=()):
        server = Server()
        isolate(self)
        self.addCleanup(server.shutdown)
        server.script = list(script)
        env = {**CLEAN, 'OPENAI_API_KEY': KEY, 'OPENAI_BASE_URL': server.url}
        with mock.patch.dict(os.environ, env, clear=True):
            report = runner.run_wall('lb-1', SEED, loopback=True, placements=4)
            blocked = g.journal_problems('lb-2', 'lb-1', True)
        wall = g.walls(True)['lb-1']
        j = json.loads(wall.journal.read_text(encoding='utf-8'))
        self.assertFalse(g.lock_of(True).exists())
        for path in wall.folder.rglob('*'):
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
