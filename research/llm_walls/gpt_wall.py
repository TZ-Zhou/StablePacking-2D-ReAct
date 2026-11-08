"""Safe runner of a live GPT wall: one complete Thangavelu noise80 wall whose filter order is chosen
by the ReAct controller with an OpenAI model (Observation V2), every other part of the wall Wang's.

  python -B -m research.llm_walls.gpt_wall preflight --name gpt-002 --seed S [--after NAME] [--loopback]
  STONEPACKING_ALLOW_PAID_API=1 python -B -m research.llm_walls.gpt_wall run --name gpt-002 --seed S --after gpt-001 \
      --confirm-paid
  python -B -m research.llm_walls.gpt_wall run --name lb-001 --seed S --loopback [--placements N]   # a local mock server
  python -B -m research.llm_walls.gpt_wall ledger [--loopback]                                     # what the walls spent

Run it in the stablepacking-baseline environment (run_wall.py builds the wall there); the
controller worker runs in the registered venv (wall_runner --worker-venv), the only place the SDK and
the key are. It carries over gpt_smoke's guarantees, for a live wall instead of a replay:
- one spec, SPEC (no retries: max_retries 0, the SDK client follows no redirect and reads no proxy
  or CA from the environment, so one attempt is one POST), checked against WALL_LIMITS: requests
  (max_decisions x max_tool_calls), tokens, cost and seconds per wall; EXPERIMENT_LIMITS: walls and
  the sum of every earlier wall's reported spend (a wall not finished blocks everything instead);
- the key only from OPENAI_API_KEY (the worker reads it; this process only scans the outputs for
  it; Wang's process drops it once the controller is built); every llm_api.CONNECTION_ENV
  setting unset; paid: the gate, --confirm-paid, no OPENAI_BASE_URL, a clean committed tree and
  a price checked within PRICE_MAX_AGE days; --loopback: no gate, OPENAI_BASE_URL on a loopback
  host and a key starting LOOPBACK_KEY (never a real one), its walls in their own study;
- one run at a time: the study's lock (research/studies/<study>/.lock) created exclusively first. It
  holds the wall's journal until the wall's case exists, then <case>/llm/journal.json is created
  exclusively. The lock is removed only once the final journal state is in that case; if the launch
  raised, the case is unknown: the lock keeps the final state and stays until a person has looked.
  A name is used once; any wall not 'finished' (started, uncertain, aborted) or without a journal
  blocks every later run until a person has looked; a later wall names the latest one (--after);
- a wall is 'finished' only if its billing is certain: the case passed (PROCESS_PASS, or stopped
  by --placements in loopback mode), order_controller.check() PASS (transcript replay, geometry,
  ledger, manifest), no unknown usage (an attempt without usage, an abandoned call, a worker
  failure of a decision, a skip:usage_unknown), every worker reaped, requests, tokens, cost and
  seconds within WALL_LIMITS, and no key in any output file; else 'uncertain' ('aborted' on an
  exception). A timeout of the case kills it, so it is never finished.
gpt_complete (report) is apart from billing: every decision point decided by the LLM (no skip, no
fallback); a wall that is not is kept but left out of any quality comparison.

Each wall is a case built by run_wall.run, research/cases/<case id>/, with its LLM records in llm/:
journal.json and report.json (the request ledger stays in the result's trace/transport.jsonl, where
the checks read it). The study research/studies/gpt-walls (loopback: gpt-walls-loopback) names every
wall ({'case_id', 'name'}). The walls before research/cases (gpt-001; loopback lb-001) stay in
research/runs/gpt-walls[-loopback], read only (journals/<name>.json): their names, journals and spend
count like the study's for the name rule, --after and the experiment budget.
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys

from research.llm_walls import gpt_smoke
from research.controller import worker as controller_worker
from research.controller import order_controller
from research.controller import llm_api
from research.paths import LEGACY_RUNS
from research.storage import case_id as cid
from research.walls import run_wall as wall_runner

STUDIES = {False: 'gpt-walls', True: 'gpt-walls-loopback'}  # loopback? -> the study of the mode's walls
# the walls before research/cases; read only
LEGACY = {False: LEGACY_RUNS / 'gpt-walls', True: LEGACY_RUNS / 'gpt-walls-loopback'}
# run_wall's arguments of every GPT wall
WALL_ARGS = ['thangavelu-noise80', '--policy', 'paper', '--order', '3,0,2,1,4', '--worker-venv']
MODEL = gpt_smoke.MODEL
PRICE = gpt_smoke.PRICE
PRICE_CHECKED = datetime.date(2026, 11, 7)  # gpt_smoke.PRICE, developers.openai.com; check again before a paid wall
PRICE_MAX_AGE = 7
LOOPBACK_KEY = 'sk-loopback-'
SPEC = {'kind': 'react',
        'llm': {'kind': 'api', 'provider': 'openai', 'model': MODEL, 'max_tokens': 4096, 'effort': 'low',
                'request_timeout_s': 60.0, 'max_retries': 0, 'retry_backoff_s': 0.0, 'price_per_mtok': PRICE},
        'llm_timeout': 90.0, 'max_tool_calls': 3, 'max_decisions': 20, 'max_total_tokens': 300_000,
        'max_cost_usd': 0.60, 'max_llm_failures': 1, 'observation_version': 2}
WALL_LIMITS = {'http_requests': 60, 'tokens': 300_000, 'cost_usd': 0.60, 'seconds': 2400}
EXPERIMENT_LIMITS = {'walls': 2, 'http_requests': 120, 'tokens': 600_000, 'cost_usd': 1.20}
FORMAL = LEGACY_RUNS / 'campaigns' / 'thangavelu-5arms-10seeds' / 'campaign.json'
EXECUTED = ('research/llm_walls/gpt_wall.py', 'research/walls/run_wall.py', 'research/controller/order_controller.py',
            'research/controller/worker.py', 'research/controller/llm_api.py', 'research/llm_walls/gpt_smoke.py',
            'research/validation/worker-venv.json', 'research/storage/case_id.py', 'research/storage/case_index.py',
            'research/storage/studies.py')


def max_requests(spec):
    return spec['max_decisions'] * spec['max_tool_calls'] * (spec['llm']['max_retries'] + 1)


def spec_problems(spec=SPEC):
    """Why SPEC does not fit WALL_LIMITS (the structural bounds a wall cannot exceed)."""
    full, llm = order_controller.full_spec(spec), llm_api.full(spec['llm'])
    rules = {'max_retries 0': llm['max_retries'] == 0,
             f"max_decisions x max_tool_calls <= {WALL_LIMITS['http_requests']}": max_requests(full) <= WALL_LIMITS['http_requests'],
             f"max_total_tokens <= {WALL_LIMITS['tokens']}": full['max_total_tokens'] <= WALL_LIMITS['tokens'],
             f"max_cost_usd <= {WALL_LIMITS['cost_usd']}": full['max_cost_usd'] <= WALL_LIMITS['cost_usd'],
             'call_bound <= llm_timeout': llm_api.call_bound(llm) <= full['llm_timeout'],
             'max_llm_failures 1': full['max_llm_failures'] == 1, 'observation_version 2': full.get('observation_version') == 2,
             f'price {PRICE}': list(llm['price_per_mtok'] or []) == PRICE}
    return [f'spec: needs {rule}' for rule, ok in rules.items() if not ok]


def lock_of(loopback):
    """The lock of the mode's study: one wall at a time; it holds a wall's journal until the wall's case exists."""
    return gpt_smoke.lock_of(STUDIES[loopback])


def walls(loopback):
    """{name: gpt_smoke.Records} of every wall of the mode: the archive's (LEGACY/<name>, its journal
    LEGACY/journals/<name>.json; a folder there without a journal included), then the study's cases."""
    legacy = LEGACY[loopback]
    names = {p.stem for p in (legacy / 'journals').glob('*.json')} | \
        {d.name for d in (legacy.iterdir() if legacy.is_dir() else []) if d.is_dir() and d.name != 'journals'}
    old = {n: gpt_smoke.Records(legacy / n, legacy / 'journals' / f'{n}.json', legacy / n / 'report.json')
           for n in sorted(names)}
    return {**old, **gpt_smoke.study_runs(STUDIES[loopback])}


def journals(loopback):
    """{name: journal} of every wall of the mode; None for a wall without a journal."""
    return {n: gpt_smoke.read_journal(w.journal) for n, w in walls(loopback).items()}


def spent(loopback):
    """The reported spend of the mode's finished walls (a wall not finished blocks instead)."""
    done = [j.get('spend') or {} for j in journals(loopback).values() if j and j.get('state') == 'finished']
    return {'walls': len(done), **{k: sum(s.get(k) or 0 for s in done) for k in ('http_requests', 'tokens', 'cost_usd')}}


def journal_problems(name, after, loopback, holding=False):
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,40}', name):
        return [f'run: the name {name!r} is not [a-z0-9-]']
    lock = lock_of(loopback)
    if not holding and lock.exists():
        return [f'run: {lock} exists: another run is starting or running, or one ended without releasing it '
                '(a wall whose case is unknown leaves its journal there); inspect it']
    found = journals(loopback)
    runs = {n: j for n, j in found.items() if j is not None}
    problems = [f'run: {n} is {j.get("state")}: inspect it (its billing may be unknown) before any new run'
                for n, j in runs.items() if j.get('state') != 'finished']
    if name in found:
        problems.append(f'run: {name} was used: a run name is used once')
    problems += [f'run: {n} has no journal: inspect it before any new run' for n, j in found.items() if j is None]
    latest = max(runs, key=lambda n: runs[n].get('started_at', ''), default=None)
    if latest is not None and after != latest:
        problems.append(f'run: the latest wall is {latest}: name it with --after {latest}')
    if after is not None and after not in runs:
        problems.append(f'run: --after {after} names no run')
    s = spent(loopback)
    room = {'walls': s['walls'] + 1 <= EXPERIMENT_LIMITS['walls'],
            **{k: s[k] + WALL_LIMITS[k] <= EXPERIMENT_LIMITS[k] for k in ('http_requests', 'tokens', 'cost_usd')}}
    problems += [f'experiment: {k} spent {s[k]} + a wall {WALL_LIMITS.get(k, 1)} exceeds {EXPERIMENT_LIMITS[k]}' for k, ok in room.items() if not ok]
    return problems


def env_problems(loopback, environ=None):
    environ = os.environ if environ is None else environ
    problems, gate, key = [], environ.get(llm_api.PAID_GATE), environ.get('OPENAI_API_KEY')
    if key is None:
        problems.append('env: OPENAI_API_KEY is not set')
    if loopback:
        if gate is not None:
            problems.append(f'env: {llm_api.PAID_GATE} must be unset for --loopback')
        if not llm_api.loopback(SPEC['llm'], environ):
            problems.append('env: --loopback needs OPENAI_BASE_URL on 127.0.0.1 or localhost')
        if key is not None and not key.startswith(LOOPBACK_KEY):
            problems.append(f'env: --loopback takes only a fake key starting {LOOPBACK_KEY!r}')
    else:
        if gate != '1':
            problems.append(f'env: {llm_api.PAID_GATE}=1 is not set for this command')
        if any(k.upper() == 'OPENAI_BASE_URL' for k in environ):
            problems.append('env: OPENAI_BASE_URL must be unset for a paid wall')
        if key is not None and key.startswith(LOOPBACK_KEY):
            problems.append('env: a loopback key cannot make a paid wall')
    problems += [f'env: {v} must be unset (it changes where a request goes, its TLS, headers or logging)'
                 for v in llm_api.connection_env(environ)]
    return problems


def git(*args):
    """git's output; raises if git fails, so a failed check never reads as a clean tree."""
    return subprocess.run(['git', *args], capture_output=True, text=True, cwd=wall_runner.REPO, check=True, env=wall_runner.quiet_env()).stdout.strip()


def code_dirty():
    return git('status', '--porcelain', '--', *EXECUTED)


def head():
    sha = git('rev-parse', '--verify', 'HEAD')
    if not re.fullmatch(r'[0-9a-f]{40}', sha):
        raise RuntimeError(f'git HEAD {sha!r} is not a commit')
    return sha


def preflight(name, seed, after=None, loopback=False, holding=False, today=None):
    problems = (spec_problems() + journal_problems(name, after, loopback, holding) + env_problems(loopback)
                + controller_worker.venv_problems())
    if seed in json.loads(FORMAL.read_text(encoding='utf-8'))['seeds']:
        problems.append(f'seed: {seed} is a seed of the formal campaign; a development wall uses another')
    if not loopback:
        if code_dirty():
            problems.append('code: the executed files differ from the commit: commit them before a paid wall')
        if ((today or datetime.date.today()) - PRICE_CHECKED).days > PRICE_MAX_AGE:
            problems.append(f'price: PRICE was checked {PRICE_CHECKED}; check it again and update PRICE_CHECKED')
    return problems


def write_json(path, data, exclusive=False):
    gpt_smoke.write_json(path, data, exclusive)


def accounting(run, spec, stopped=False):
    """What the wall's records and ledger say it sent and spent; unknown counts every doubt."""
    result = next(run.glob('result_*'))
    records = [json.loads(l) for l in (result / 'trace' / 'controller.jsonl').read_text(encoding='utf-8').splitlines()]
    ledger = [json.loads(l) for l in (result / 'trace' / 'transport.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = gpt_smoke.requests_of(records, llm_api.price(spec['llm']))
    use = llm_api.spend(spec['llm'], [t for r in records for t in r.get('transcripts') or []])
    failures = [x for x in ledger if x.get('type') == 'request' and x.get('status') != 'ok']
    skips = sorted({r['source'] for r in records if r.get('decision_point') and r.get('source', '').startswith(('skip', 'fallback'))})
    decisions = [r for r in records if r.get('decision_point')]
    unseen = [x for x in rows if x.get('outcome') in ('worker_failure', 'abandoned')]  # requests that may have gone out unrecorded
    return {'http_requests': sum(1 for x in rows if x.get('outcome') in ('reply', 'retried', 'failed')),
            'http_requests_exact': not unseen and not failures,
            'tokens': use['tokens'] + use['estimated_tokens'], 'cost_usd': use['cost_usd'],
            'unknown_usage': use['unknown'] + len(failures) + ('skip:usage_unknown' in skips),
            'worker_failures': [{k: x.get(k) for k in ('event', 'status', 'error')} for x in failures],
            'workers_reaped': not order_controller.worker_problems(ledger, stopped)[0],
            'decision_points': len(decisions), 'llm_decisions': sum(r['source'] == 'llm' for r in decisions),
            'switches': sum(r['final']['switched'] for r in decisions), 'skips_and_fallbacks': skips,
            'gpt_complete': bool(decisions) and all(r['source'] == 'llm' for r in decisions), 'requests': rows}


def study_design(loopback):
    """The design a study of GPT walls is created with, by its first wall."""
    return {'runner': 'research/llm_walls/gpt_wall.py', 'mode': 'loopback mock' if loopback else 'paid',
            'run_wall': WALL_ARGS, 'spec': order_controller.full_spec(SPEC), 'wall_limits': WALL_LIMITS,
            'experiment_limits': EXPERIMENT_LIMITS}


def run_wall(name, seed, after=None, loopback=False, placements=0, launch=wall_runner.run):
    """One GPT wall (see the module docstring) under the study's lock; returns its report (<case>/llm/report.json)."""
    if placements and not loopback:
        raise SystemExit('refused: --placements only for a loopback wall')
    problems = preflight(name, seed, after, loopback)
    if problems:
        gpt_smoke.refuse(problems)
    study, lock = STUDIES[loopback], lock_of(loopback)
    # built before the lock: a git failure here leaves nothing behind
    journal = {'state': 'started', 'after': after, 'seed': seed, 'study': study,
               'mode': 'loopback mock' if loopback else 'paid', 'spec': order_controller.full_spec(SPEC), 'limits': WALL_LIMITS,
               'experiment_limits': EXPERIMENT_LIMITS, 'max_http_requests': max_requests(order_controller.full_spec(SPEC)),
               'placements': placements, 'code_dirty': bool(code_dirty()),
               'git_head': head(), 'worker_venv': wall_runner.worker_venv_identity()}
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_json(lock, {'run': name, 'pid': os.getpid()}, exclusive=True)
    except FileExistsError:
        gpt_smoke.refuse([f'run: {lock} appeared: another run is starting'])
    problems = journal_problems(name, after, loopback, holding=True)  # again, under the lock
    if problems:
        lock.unlink()
        gpt_smoke.refuse(problems)
    journal['started_at'] = datetime.datetime.now().isoformat(timespec='seconds')
    report, reasons, state, written = {}, [], 'aborted', False
    try:
        # the journal before the wall starts: in the lock, as run_wall only then allocates the case
        write_json(lock, {'run': name, 'pid': os.getpid(), 'journal': journal})
        returned = launch([*WALL_ARGS, '--seed', str(seed), '--timeout', str(WALL_LIMITS['seconds']),
                           '--controller', json.dumps(SPEC)] + (['--placements', str(placements)] if placements else []))
        journal['case_id'] = returned['case_id']
        run = cid.wall_dir(returned['case_id'])
        files = gpt_smoke.case_records(run)
        # the study lists the case first: from here on a case without a journal blocks too
        gpt_smoke.join_study(study, 'gpt-walls', study_design(loopback), {'case_id': returned['case_id'], 'name': name})
        files.journal.parent.mkdir()
        write_json(files.journal, journal, exclusive=True)
        written = True
        receipt = json.loads((run / 'receipt.json').read_text(encoding='utf-8'))
        ok_status = ('PROCESS_PASS', 'STOPPED_AFTER_PLACEMENTS') if placements else ('PROCESS_PASS',)
        report = accounting(run, SPEC, receipt.get('process_status') == 'STOPPED_AFTER_PLACEMENTS')
        report['check'] = {k: v for k, v in order_controller.check(next(run.glob('result_*'))).items()
                           if k in ('ok', 'verdict', 'problems', 'incomplete', 'records', 'decision_points', 'switches', 'geometry', 'provenance')}
        report.update(status=receipt['status'], elapsed_seconds=receipt.get('elapsed_seconds'))
        write_json(files.report, report)
        report['key_scan'] = gpt_smoke.key_scan(run)  # llm/report.json included
        over = [k for k in ('http_requests', 'tokens', 'cost_usd') if report[k] > WALL_LIMITS[k]]
        reasons = [why for why, bad in (
            (f"case status {receipt['status']}", receipt['status'] not in ok_status),
            (f"check verdict {report['check'].get('verdict')}", report['check'].get('verdict') != 'PASS'),
            ('unknown usage', report['unknown_usage']), ('a worker failure', report['worker_failures']),
            ('requests that may have gone out unrecorded', not report['http_requests_exact']),
            ('a worker not reaped', not report['workers_reaped']), (f'over the wall limits: {over}', over),
            ('over the time limit', (receipt.get('elapsed_seconds') or 1e9) > WALL_LIMITS['seconds']),
            ('the key in an output file', report['key_scan'])) if bad]
        write_json(files.report, report)
        state = 'uncertain' if reasons else 'finished'
    except BaseException as error:  # the journal keeps the experiment blocked for inspection
        reasons = ['an exception: ' + order_controller.error_text(error)]
        raise
    finally:
        final = {**journal, 'state': state, 'reasons': reasons,
                 'ended_at': datetime.datetime.now().isoformat(timespec='seconds'),
                 'spend': {k: report.get(k) for k in ('http_requests', 'tokens', 'cost_usd', 'unknown_usage')},
                 'gpt_complete': report.get('gpt_complete')}
        if written:
            write_json(files.journal, final)
            lock.unlink()
        else:  # no journal in a case of the study: the lock keeps the final state and blocks every later wall
            write_json(lock, {'run': name, 'pid': os.getpid(), 'journal': final})
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for cmd in ('preflight', 'run'):
        p = sub.add_parser(cmd)
        p.add_argument('--name', required=True)
        p.add_argument('--seed', type=int, required=True)
        p.add_argument('--after')
        p.add_argument('--loopback', action='store_true')
        if cmd == 'run':
            p.add_argument('--confirm-paid', action='store_true')
            p.add_argument('--placements', type=int, default=0)
    sub.add_parser('ledger').add_argument('--loopback', action='store_true')
    a = ap.parse_args(argv)
    if a.cmd == 'ledger':
        shown = ('state', 'reasons', 'spend', 'gpt_complete', 'case_id')
        listed = {n: {k: (j or {'state': 'no journal'}).get(k) for k in shown} for n, j in journals(a.loopback).items()}
        print(json.dumps({'spent': spent(a.loopback), 'limits': EXPERIMENT_LIMITS,
                          'lock': gpt_smoke.read_journal(lock_of(a.loopback)), 'walls': listed}, indent=1))
        return 0
    if a.cmd == 'preflight':
        problems = preflight(a.name, a.seed, a.after, a.loopback)
        print(json.dumps({'ok': not problems, 'problems': problems, 'max_http_requests': max_requests(order_controller.full_spec(SPEC))}, indent=1))
        return 0 if not problems else 1
    if not (a.loopback or a.confirm_paid):
        gpt_smoke.refuse(['run: a paid wall needs --confirm-paid (or --loopback for a local mock server)'])
    report = run_wall(a.name, a.seed, a.after, a.loopback, a.placements)
    print(json.dumps({k: v for k, v in report.items() if k != 'requests'}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
