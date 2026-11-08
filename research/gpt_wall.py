"""Safe runner of a live GPT wall: one complete Thangavelu noise80 wall whose filter order is chosen
by the ReAct controller with an OpenAI model (Observation V2), every other part of the wall Wang's.

  python -B research/gpt_wall.py preflight --name gpt-001 --seed S [--after NAME] [--loopback]
  STONEPACKING_ALLOW_PAID_API=1 python -B research/gpt_wall.py run --name gpt-001 --seed S --confirm-paid
  python -B research/gpt_wall.py run --name lb-001 --seed S --loopback [--placements N]   # a local mock server
  python -B research/gpt_wall.py ledger [--loopback]                                     # what the walls spent

Run it in the stablepacking-baseline environment (run_case.py builds the wall there); the
controller worker runs in the registered venv (run_case --worker-venv), the only place the SDK and
the key are. It carries over api_smoke's guarantees, for a live wall instead of a replay:
- one spec, SPEC (no retries: max_retries 0, the SDK client follows no redirect and reads no proxy
  or CA from the environment, so one attempt is one POST), checked against WALL_LIMITS: requests
  (max_decisions x max_tool_calls), tokens, cost and seconds per wall; EXPERIMENT_LIMITS: walls and
  the sum of every earlier wall's reported spend (a wall not finished blocks everything instead);
- the key only from OPENAI_API_KEY (the worker reads it; this process only scans the outputs for
  it; Wang's process drops it once the controller is built); every llm_adapter.CONNECTION_ENV
  setting unset; paid: the gate, --confirm-paid, no OPENAI_BASE_URL, a clean committed tree and
  a price checked within PRICE_MAX_AGE days; --loopback: no gate, OPENAI_BASE_URL on a loopback
  host and a key starting LOOPBACK_KEY (never a real one), its walls in their own directory;
- one run at a time: OUT/.lock created exclusively first, removed only after the final journal
  state; OUT/journals/<name>.json created exclusively before the wall starts; a name is used
  once; any wall not 'finished' (started, uncertain, aborted) blocks every later run until a
  person has looked; a later wall names the latest one (--after);
- a wall is 'finished' only if its billing is certain: the case passed (PROCESS_PASS, or stopped
  by --placements in loopback mode), controllers.check() PASS (transcript replay, geometry,
  ledger, manifest), no unknown usage (an attempt without usage, an abandoned call, a worker
  failure of a decision, a skip:usage_unknown), every worker reaped, requests, tokens, cost and
  seconds within WALL_LIMITS, and no key in any output file; else 'uncertain' ('aborted' on an
  exception). A timeout of the case kills it, so it is never finished.
gpt_complete (report) is apart from billing: every decision point decided by the LLM (no skip, no
fallback); a wall that is not is kept but left out of any quality comparison.
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import api_smoke  # noqa: E402
import controller_worker  # noqa: E402
import controllers  # noqa: E402
import llm_adapter  # noqa: E402
import run_case  # noqa: E402

OUT = run_case.RUNS / 'gpt-walls'
OUT_LOOPBACK = run_case.RUNS / 'gpt-walls-loopback'
LOCK = '.lock'
MODEL = api_smoke.MODEL
PRICE = api_smoke.PRICE
PRICE_CHECKED = datetime.date(2026, 11, 7)  # api_smoke.PRICE, developers.openai.com; check again before a paid wall
PRICE_MAX_AGE = 7
LOOPBACK_KEY = 'sk-loopback-'
SPEC = {'kind': 'react',
        'llm': {'kind': 'api', 'provider': 'openai', 'model': MODEL, 'max_tokens': 4096, 'effort': 'low',
                'request_timeout_s': 60.0, 'max_retries': 0, 'retry_backoff_s': 0.0, 'price_per_mtok': PRICE},
        'llm_timeout': 90.0, 'max_tool_calls': 3, 'max_decisions': 20, 'max_total_tokens': 300_000,
        'max_cost_usd': 0.60, 'max_llm_failures': 1, 'observation_version': 2}
WALL_LIMITS = {'http_requests': 60, 'tokens': 300_000, 'cost_usd': 0.60, 'seconds': 2400}
EXPERIMENT_LIMITS = {'walls': 2, 'http_requests': 120, 'tokens': 600_000, 'cost_usd': 1.20}
FORMAL = run_case.RUNS / 'campaigns' / 'thangavelu-5arms-10seeds' / 'campaign.json'
EXECUTED = ('research/gpt_wall.py', 'research/run_case.py', 'research/controllers.py', 'research/controller_worker.py',
            'research/llm_adapter.py', 'research/api_smoke.py', 'research/validation/worker-venv.json')


def max_requests(spec):
    return spec['max_decisions'] * spec['max_tool_calls'] * (spec['llm']['max_retries'] + 1)


def spec_problems(spec=SPEC):
    """Why SPEC does not fit WALL_LIMITS (the structural bounds a wall cannot exceed)."""
    full, llm = controllers.full_spec(spec), llm_adapter.full(spec['llm'])
    rules = {'max_retries 0': llm['max_retries'] == 0,
             f"max_decisions x max_tool_calls <= {WALL_LIMITS['http_requests']}": max_requests(full) <= WALL_LIMITS['http_requests'],
             f"max_total_tokens <= {WALL_LIMITS['tokens']}": full['max_total_tokens'] <= WALL_LIMITS['tokens'],
             f"max_cost_usd <= {WALL_LIMITS['cost_usd']}": full['max_cost_usd'] <= WALL_LIMITS['cost_usd'],
             'call_bound <= llm_timeout': llm_adapter.call_bound(llm) <= full['llm_timeout'],
             'max_llm_failures 1': full['max_llm_failures'] == 1, 'observation_version 2': full.get('observation_version') == 2,
             f'price {PRICE}': list(llm['price_per_mtok'] or []) == PRICE}
    return [f'spec: needs {rule}' for rule, ok in rules.items() if not ok]


def journals(out):
    found = {}
    for path in sorted((out / 'journals').glob('*.json')) if (out / 'journals').is_dir() else []:
        try:
            found[path.stem] = json.loads(path.read_text(encoding='utf-8'))
        except ValueError:
            found[path.stem] = {'state': 'unreadable'}
    return found


def spent(out):
    """The reported spend of the finished walls of out (a wall not finished blocks instead)."""
    done = [j.get('spend') or {} for j in journals(out).values() if j.get('state') == 'finished']
    return {'walls': len(done), **{k: sum(s.get(k) or 0 for s in done) for k in ('http_requests', 'tokens', 'cost_usd')}}


def journal_problems(out, name, after, holding=False):
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,40}', name):
        return [f'run: the name {name!r} is not [a-z0-9-]']
    if not holding and (out / LOCK).exists():
        return [f'run: {out / LOCK} exists: another run is starting or running, or one ended without releasing it; inspect it']
    runs = journals(out)
    problems = [f'run: {n} is {j.get("state")}: inspect it (its billing may be unknown) before any new run'
                for n, j in runs.items() if j.get('state') != 'finished']
    if name in runs or (out / name).exists():
        problems.append(f'run: {name} was used: a run name is used once')
    problems += [f'run: {d.name} has no journal: inspect it before any new run' for d in (out.iterdir() if out.is_dir() else [])
                 if d.is_dir() and d.name != 'journals' and d.name not in runs]
    latest = max(runs, key=lambda n: runs[n].get('started_at', ''), default=None)
    if latest is not None and after != latest:
        problems.append(f'run: the latest wall is {latest}: name it with --after {latest}')
    if after is not None and after not in runs:
        problems.append(f'run: --after {after} names no run')
    s = spent(out)
    room = {'walls': s['walls'] + 1 <= EXPERIMENT_LIMITS['walls'],
            **{k: s[k] + WALL_LIMITS[k] <= EXPERIMENT_LIMITS[k] for k in ('http_requests', 'tokens', 'cost_usd')}}
    problems += [f'experiment: {k} spent {s[k]} + a wall {WALL_LIMITS.get(k, 1)} exceeds {EXPERIMENT_LIMITS[k]}' for k, ok in room.items() if not ok]
    return problems


def env_problems(loopback, environ=None):
    environ = os.environ if environ is None else environ
    problems, gate, key = [], environ.get(llm_adapter.PAID_GATE), environ.get('OPENAI_API_KEY')
    if key is None:
        problems.append('env: OPENAI_API_KEY is not set')
    if loopback:
        if gate is not None:
            problems.append(f'env: {llm_adapter.PAID_GATE} must be unset for --loopback')
        if not llm_adapter.loopback(SPEC['llm'], environ):
            problems.append('env: --loopback needs OPENAI_BASE_URL on 127.0.0.1 or localhost')
        if key is not None and not key.startswith(LOOPBACK_KEY):
            problems.append(f'env: --loopback takes only a fake key starting {LOOPBACK_KEY!r}')
    else:
        if gate != '1':
            problems.append(f'env: {llm_adapter.PAID_GATE}=1 is not set for this command')
        if any(k.upper() == 'OPENAI_BASE_URL' for k in environ):
            problems.append('env: OPENAI_BASE_URL must be unset for a paid wall')
        if key is not None and key.startswith(LOOPBACK_KEY):
            problems.append('env: a loopback key cannot make a paid wall')
    problems += [f'env: {v} must be unset (it changes where a request goes, its TLS, headers or logging)'
                 for v in llm_adapter.connection_env(environ)]
    return problems


def git(*args):
    """git's output; raises if git fails, so a failed check never reads as a clean tree."""
    return subprocess.run(['git', *args], capture_output=True, text=True, cwd=run_case.REPO, check=True, env=run_case.quiet_env()).stdout.strip()


def code_dirty():
    return git('status', '--porcelain', '--', *EXECUTED)


def head():
    sha = git('rev-parse', '--verify', 'HEAD')
    if not re.fullmatch(r'[0-9a-f]{40}', sha):
        raise RuntimeError(f'git HEAD {sha!r} is not a commit')
    return sha


def preflight(name, seed, out, after=None, loopback=False, holding=False, today=None):
    problems = spec_problems() + journal_problems(out, name, after, holding) + env_problems(loopback) + controller_worker.venv_problems()
    if seed in json.loads(FORMAL.read_text(encoding='utf-8'))['seeds']:
        problems.append(f'seed: {seed} is a seed of the formal campaign; a development wall uses another')
    if not loopback:
        if code_dirty():
            problems.append('code: the executed files differ from the commit: commit them before a paid wall')
        if ((today or datetime.date.today()) - PRICE_CHECKED).days > PRICE_MAX_AGE:
            problems.append(f'price: PRICE was checked {PRICE_CHECKED}; check it again and update PRICE_CHECKED')
    return problems


def write_json(path, data, exclusive=False):
    api_smoke.write_json(path, data, exclusive)


def accounting(run, spec, stopped=False):
    """What the wall's records and ledger say it sent and spent; unknown counts every doubt."""
    result = next(run.glob('result_*'))
    records = [json.loads(l) for l in (result / 'trace' / 'controller.jsonl').read_text(encoding='utf-8').splitlines()]
    ledger = [json.loads(l) for l in (result / 'trace' / 'transport.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = api_smoke.requests_of(records, llm_adapter.price(spec['llm']))
    use = llm_adapter.spend(spec['llm'], [t for r in records for t in r.get('transcripts') or []])
    failures = [x for x in ledger if x.get('type') == 'request' and x.get('status') != 'ok']
    skips = sorted({r['source'] for r in records if r.get('decision_point') and r.get('source', '').startswith(('skip', 'fallback'))})
    decisions = [r for r in records if r.get('decision_point')]
    unseen = [x for x in rows if x.get('outcome') in ('worker_failure', 'abandoned')]  # requests that may have gone out unrecorded
    return {'http_requests': sum(1 for x in rows if x.get('outcome') in ('reply', 'retried', 'failed')),
            'http_requests_exact': not unseen and not failures,
            'tokens': use['tokens'] + use['estimated_tokens'], 'cost_usd': use['cost_usd'],
            'unknown_usage': use['unknown'] + len(failures) + ('skip:usage_unknown' in skips),
            'worker_failures': [{k: x.get(k) for k in ('event', 'status', 'error')} for x in failures],
            'workers_reaped': not controllers.worker_problems(ledger, stopped)[0],
            'decision_points': len(decisions), 'llm_decisions': sum(r['source'] == 'llm' for r in decisions),
            'switches': sum(r['final']['switched'] for r in decisions), 'skips_and_fallbacks': skips,
            'gpt_complete': bool(decisions) and all(r['source'] == 'llm' for r in decisions), 'requests': rows}


def run_wall(name, seed, out, after=None, loopback=False, placements=0, launch=run_case.main):
    if placements and not loopback:
        raise SystemExit('refused: --placements only for a loopback wall')
    problems = preflight(name, seed, out, after, loopback)
    if problems:
        api_smoke.refuse(problems)
    run = out / name
    journal_path = out / 'journals' / f'{name}.json'
    journal = {'state': 'started', 'after': after, 'seed': seed,  # built before the lock: a git failure here leaves nothing behind
               'mode': 'loopback mock' if loopback else 'paid', 'spec': controllers.full_spec(SPEC), 'limits': WALL_LIMITS,
               'experiment_limits': EXPERIMENT_LIMITS, 'max_http_requests': max_requests(controllers.full_spec(SPEC)),
               'placements': placements, 'run': run.relative_to(run_case.REPO).as_posix(), 'code_dirty': bool(code_dirty()),
               'git_head': head(), 'worker_venv': run_case.worker_venv_identity()}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'journals').mkdir(exist_ok=True)
    try:
        write_json(out / LOCK, {'run': name, 'pid': os.getpid()}, exclusive=True)
    except FileExistsError:
        api_smoke.refuse([f'run: {out / LOCK} appeared: another run is starting'])
    problems = journal_problems(out, name, after, holding=True)  # again, under the lock: a run may have ended meanwhile
    if problems:
        (out / LOCK).unlink()
        api_smoke.refuse(problems)
    journal['started_at'] = datetime.datetime.now().isoformat(timespec='seconds')
    report, reasons, state, written = {}, [], 'aborted', False
    try:
        write_json(journal_path, journal, exclusive=True)  # before the wall starts
        written = True
        launch(['thangavelu-noise80', run.relative_to(run_case.RUNS).as_posix(), '--policy', 'paper', '--order', '3,0,2,1,4',
                '--seed', str(seed), '--timeout', str(WALL_LIMITS['seconds']), '--controller', json.dumps(SPEC), '--worker-venv']
               + (['--placements', str(placements)] if placements else []))
        receipt = json.loads((run / 'receipt.json').read_text(encoding='utf-8'))
        ok_status = ('PROCESS_PASS', 'STOPPED_AFTER_PLACEMENTS') if placements else ('PROCESS_PASS',)
        report = accounting(run, SPEC, receipt.get('process_status') == 'STOPPED_AFTER_PLACEMENTS')
        report['check'] = {k: v for k, v in controllers.check(next(run.glob('result_*'))).items()
                           if k in ('ok', 'verdict', 'problems', 'incomplete', 'records', 'decision_points', 'switches', 'geometry', 'provenance')}
        report.update(status=receipt['status'], elapsed_seconds=receipt.get('elapsed_seconds'))
        write_json(run / 'report.json', report)
        report['key_scan'] = api_smoke.key_scan(run)  # report.json included
        over = [k for k in ('http_requests', 'tokens', 'cost_usd') if report[k] > WALL_LIMITS[k]]
        reasons = [why for why, bad in (
            (f"case status {receipt['status']}", receipt['status'] not in ok_status),
            (f"check verdict {report['check'].get('verdict')}", report['check'].get('verdict') != 'PASS'),
            ('unknown usage', report['unknown_usage']), ('a worker failure', report['worker_failures']),
            ('requests that may have gone out unrecorded', not report['http_requests_exact']),
            ('a worker not reaped', not report['workers_reaped']), (f'over the wall limits: {over}', over),
            ('over the time limit', (receipt.get('elapsed_seconds') or 1e9) > WALL_LIMITS['seconds']),
            ('the key in an output file', report['key_scan'])) if bad]
        write_json(run / 'report.json', report)
        state = 'uncertain' if reasons else 'finished'
    except BaseException as error:  # the journal keeps the experiment blocked for inspection
        reasons = ['an exception: ' + controllers.error_text(error)]
        raise
    finally:
        if written:  # never over another run's journal
            write_json(journal_path, {**journal, 'state': state, 'reasons': reasons, 'ended_at': datetime.datetime.now().isoformat(timespec='seconds'),
                                      'spend': {k: report.get(k) for k in ('http_requests', 'tokens', 'cost_usd', 'unknown_usage')},
                                      'gpt_complete': report.get('gpt_complete')})
        (out / LOCK).unlink()
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
    out = OUT_LOOPBACK if a.loopback else OUT
    if a.cmd == 'ledger':
        print(json.dumps({'spent': spent(out), 'limits': EXPERIMENT_LIMITS,
                          'walls': {n: {k: j.get(k) for k in ('state', 'reasons', 'spend', 'gpt_complete')} for n, j in journals(out).items()}}, indent=1))
        return 0
    if a.cmd == 'preflight':
        problems = preflight(a.name, a.seed, out, a.after, a.loopback)
        print(json.dumps({'ok': not problems, 'problems': problems, 'max_http_requests': max_requests(controllers.full_spec(SPEC))}, indent=1))
        return 0 if not problems else 1
    if not (a.loopback or a.confirm_paid):
        api_smoke.refuse(['run: a paid wall needs --confirm-paid (or --loopback for a local mock server)'])
    report = run_wall(a.name, a.seed, out, a.after, a.loopback, a.placements)
    print(json.dumps({k: v for k, v in report.items() if k != 'requests'}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
