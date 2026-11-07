"""API smoke test of the ReAct controller (P2): a few real OpenAI decisions on the decision
trajectory of an existing base-policy wall (the Thangavelu noise80 Paper pilot wall), never a new
wall and never a change to Wang's algorithm: the wall's trace is replayed offline (rerun) and
only the controller's first `decisions` decision points call the API.

  python research/api_smoke.py preflight --name smoke-001            # checks only, sends nothing
  STONEPACKING_ALLOW_PAID_API=1 python research/api_smoke.py run --name smoke-001 --confirm-paid
  python research/api_smoke.py check --name smoke-001                # offline: never the API

Run it with the isolated venv that has the openai SDK (the stablepacking-baseline environment
stays untouched). What a run guarantees:
- the spec is SPEC (gpt-6.1-terra, max_retries 0, budgets) with `decisions` (1 by default, at
  most 3) and is checked against LIMITS before anything is sent; the worker is always a
  ProcessTransport (no inline paid calls); at most decisions * max_tool_calls HTTP requests;
- the key comes only from OPENAI_API_KEY (read by the worker; this process reads it only to scan
  the outputs for it); OPENAI_BASE_URL, OPENAI_CUSTOM_HEADERS and OPENAI_LOG must be unset for a
  paid run (the key goes only to the official endpoint, the SDK logs nothing);
- journal.json is created (exclusively) before the first request: a run name is used once, and
  while any earlier run is not finished (started, aborted) no run starts at all; after a finished
  run the next one must name it (--after), so a request is never resent unknowingly;
- the first attempt of unknown usage stops every later request (APILLM, ControllerClient);
- afterwards: the controller check with Transcript Replay (ReplayLLM: the API sender cannot even
  be built), ledger and manifest, a key scan of every output file, and the worker cleanup check.
--loopback is for tests only: it requires OPENAI_BASE_URL on 127.0.0.1/localhost (a mock server).
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from unittest import mock
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_case  # noqa: E402
import controller_worker  # noqa: E402
import controllers  # noqa: E402
import llm_adapter  # noqa: E402
import run_case  # noqa: E402

WALL = run_case.RUNS / 'campaigns' / 'pilot-thangavelu-4arms' / 'thangavelu-noise80' / 'paper_3-0-2-1-4' / 'seed184947496'
OUT = run_case.RUNS / 'api-smoke'
MODEL = 'gpt-6.1-terra'
# USD per million tokens: input, output, cache write (the 12 h rate, the dearer), cached input:
# developers.openai.com pricing, checked 2026-11-07; check again before a paid run
PRICE = [1.00, 5.00, 2.50, 0.05]
LIMITS = {'decisions': 3, 'max_cost_usd': 0.50, 'max_total_tokens': 100_000, 'max_tokens': 4096,
          'request_timeout_s': 60.0, 'llm_timeout': 120.0, 'max_tool_calls': 3}
REPO_FILES = ('research/api_smoke.py', 'research/controllers.py', 'research/controller_worker.py', 'research/llm_adapter.py')


def spec_of(decisions=1):
    return {'kind': 'react',
            'llm': {'kind': 'api', 'provider': 'openai', 'model': MODEL, 'max_tokens': 4096, 'effort': 'low',
                    'request_timeout_s': 60.0, 'max_retries': 0, 'retry_backoff_s': 0.0, 'price_per_mtok': PRICE},
            'llm_timeout': 90.0, 'max_tool_calls': 3, 'max_decisions': decisions, 'max_total_tokens': 100_000,
            'max_cost_usd': 0.50, 'max_llm_failures': 1}


def max_requests(spec):
    return spec['max_decisions'] * spec['max_tool_calls'] * (spec['llm']['max_retries'] + 1)


def spec_problems(spec):
    """Why spec is not a smoke spec (LIMITS, no retry, a price, one LLM failure ends it)."""
    try:
        full, llm = controllers.full_spec(spec), llm_adapter.full(spec['llm'])
    except (ValueError, KeyError, TypeError) as error:
        return [f'not a valid spec: {error}']
    within = lambda v, lo, hi: type(v) in (int, float) and lo <= v <= hi
    rules = {'provider openai, model ' + MODEL: llm['provider'] == 'openai' and llm['model'] == MODEL,
             'max_retries 0 (no automatic resend)': llm['max_retries'] == 0,
             f"1 <= max_decisions <= {LIMITS['decisions']}": type(full['max_decisions']) is int and 1 <= full['max_decisions'] <= LIMITS['decisions'],
             f"0 < max_cost_usd <= {LIMITS['max_cost_usd']}": within(full.get('max_cost_usd'), 1e-9, LIMITS['max_cost_usd']),
             f"0 < max_total_tokens <= {LIMITS['max_total_tokens']}": type(full.get('max_total_tokens')) is int and 0 < full['max_total_tokens'] <= LIMITS['max_total_tokens'],
             f"max_tokens (max_output_tokens) <= {LIMITS['max_tokens']}": llm['max_tokens'] <= LIMITS['max_tokens'],
             f"request_timeout_s <= {LIMITS['request_timeout_s']}": within(llm['request_timeout_s'], 1, LIMITS['request_timeout_s']),
             f"call_bound <= llm_timeout <= {LIMITS['llm_timeout']}": within(full['llm_timeout'], llm_adapter.call_bound(llm), LIMITS['llm_timeout']),
             f"1 <= max_tool_calls <= {LIMITS['max_tool_calls']}": 1 <= full['max_tool_calls'] <= LIMITS['max_tool_calls'],
             'max_llm_failures 1': full['max_llm_failures'] == 1,
             'a price per million tokens (input, output, cache write, cached input)': llm['price_per_mtok'] is not None}
    return [f'spec: needs {rule}' for rule, ok in rules.items() if not ok]


def env_problems(loopback, paid):
    """Why this process may not send: the gate, the key variable, the endpoint, SDK logging, the SDK."""
    problems, env = [], os.environ
    if paid and env.get(llm_adapter.PAID_GATE) != '1':
        problems.append(f'env: {llm_adapter.PAID_GATE}=1 is not set for this command')
    if 'OPENAI_API_KEY' not in env:
        problems.append('env: OPENAI_API_KEY is not set')
    base = env.get('OPENAI_BASE_URL')
    if loopback:
        if base is None or urlparse(base).hostname not in ('127.0.0.1', 'localhost', '::1'):
            problems.append('env: --loopback needs OPENAI_BASE_URL on 127.0.0.1 or localhost')
    else:
        problems += [f'env: {v} must be unset (the key goes to the official endpoint only)' for v in ('OPENAI_BASE_URL', 'OPENAI_CUSTOM_HEADERS') if v in env]
    if 'OPENAI_LOG' in env:
        problems.append('env: OPENAI_LOG must be unset (no SDK logging)')
    if importlib.util.find_spec('openai') is None:
        problems.append(f'env: the openai SDK is not importable by {sys.executable} (use the isolated venv)')
    return problems


def journal_problems(out, name, after, decisions):
    """Why run `name` may not start: a used name, an unfinished earlier run, a missing --after."""
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,40}', name):
        return [f'run: the name {name!r} is not [a-z0-9-]']
    if (out / name).exists():
        return [f'run: {out / name} exists: a run name is used once, its requests are never resent']
    runs = {}
    for path in sorted(out.glob('*/journal.json')) if out.is_dir() else []:
        try:
            runs[path.parent.name] = json.loads(path.read_text(encoding='utf-8'))
        except ValueError:
            runs[path.parent.name] = {'state': 'unreadable'}
    problems = [f'run: {n} is {j.get("state")}: inspect it (its billing may be unknown) before any new run'
                for n, j in runs.items() if j.get('state') != 'finished']
    problems += [f'run: {d.name} has no journal: inspect it before any new run' for d in (out.iterdir() if out.is_dir() else [])
                 if d.is_dir() and d.name not in runs]
    latest = max(runs, key=lambda n: runs[n].get('started_at', ''), default=None)
    if latest is not None and after != latest:
        problems.append(f'run: an earlier run ({latest}) sent requests on this trajectory: name it with --after {latest}')
    if after is not None and after not in runs:
        problems.append(f'run: --after {after} names no run')
    if decisions > 1 and after is None:
        problems.append('run: more than one decision point only after an approved 1-decision run (--after)')
    return problems


def wall_problems():
    receipt = json.loads((WALL / 'receipt.json').read_text(encoding='utf-8')) if (WALL / 'receipt.json').is_file() else {}
    problems = [] if receipt.get('status') == 'PROCESS_PASS' else [f'wall: {WALL} is not a PROCESS_PASS run']
    if receipt.get('controller') is not None:
        problems.append('wall: built with a controller; the smoke needs the base-policy (Paper) trajectory')
    return problems


def preflight(name, spec, out=OUT, after=None, loopback=False, paid=False):
    """Every problem that forbids the run; empty: it may send at most max_requests(spec) requests."""
    return spec_problems(spec) + env_problems(loopback, paid) + wall_problems() + \
        journal_problems(out, name, after, spec.get('max_decisions', 0) if isinstance(spec, dict) else 0)


def write_json(path, data, exclusive=False):
    text = json.dumps(data, indent=1, sort_keys=True)
    if exclusive:  # fails if it exists: two starts of one run cannot both pass
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL), 'w', encoding='utf-8') as f:
            f.write(text)
    else:
        path.write_text(text, encoding='utf-8')


def write_jsonl(path, items):
    path.write_text(''.join(json.dumps(i) + '\n' for i in items), encoding='utf-8')


def provenance():
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=run_case.REPO).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', *REPO_FILES], capture_output=True, text=True, cwd=run_case.REPO).stdout.strip()
    spec_ = importlib.util.find_spec('openai')
    version = None
    if spec_ is not None:
        import openai
        version = openai.__version__
    return {'git_head': head, 'code_dirty': bool(dirty), 'python': sys.executable, 'openai_sdk': version,
            'code_sha256': {f: hashlib.sha256((run_case.REPO / f).read_bytes()).hexdigest() for f in REPO_FILES}}


def copy_wall(dst):
    """The wall's receipt and first result directory (trace, decisions.txt, base.pkl) under dst."""
    src = check_case.result_dirs(WALL)[0]
    result = dst / src.relative_to(WALL)
    shutil.copytree(src / 'trace', result / 'trace')
    for f in ('decisions.txt', 'base.pkl'):
        shutil.copy2(src / f, result / f)
    shutil.copy2(WALL / 'receipt.json', dst / 'receipt.json')
    return result


def requests_of(records, prices):
    """Every API request the records hold: decision, call, attempt, outcome, billing, HTTP status,
    request id, latency, error, and per call the usage and its estimated cost."""
    rows = []
    for r in records:
        for t in r.get('transcripts') or []:
            response = t.get('response') or {}
            usage = response.get('usage')
            call_cost = llm_adapter.cost(prices, usage) if llm_adapter.usage_ok(usage) else None
            for a in response.get('attempts') or []:
                rows.append({'event': r['event'], 'decision': list(controllers.decision_key(r)), 'call': t['call'],
                             **{k: a.get(k) for k in ('attempt', 'outcome', 'billed', 'status_code', 'request_id', 'latency_s', 'error')},
                             'usage': usage if a.get('outcome') == 'reply' else None,
                             'cost_usd_estimate': call_cost if a.get('outcome') == 'reply' else None,
                             'served_model': ((response.get('exchange') or {}).get('response') or {}).get('model') if a.get('outcome') == 'reply' else None})
            if t.get('status') == 'timeout':  # abandoned in flight: counted as unknown
                rows.append({'event': r['event'], 'decision': list(controllers.decision_key(r)), 'call': t['call'],
                             'outcome': 'abandoned', 'billed': 'unknown', 'error': response.get('error')})
    return rows


def offline_check(run, spec):
    """The controller check with Transcript Replay; on a base-policy trajectory a decision whose
    order differs from the traced Paper order is reported by check() as 'traced policy' and, if its
    pick differs, 'replay selects': those two, at exactly those events, are expected (the wall
    really was built with Paper) and listed apart. The API sender cannot be built meanwhile."""
    result = check_case.result_dirs(run)[0]
    records = [json.loads(l) for l in (result / 'trace' / 'controller.jsonl').read_text(encoding='utf-8').splitlines()]
    paper = controllers.key(controllers.PAPER_ORDER)
    moved = {r['event'] for r in records if r.get('final', {}).get('order') != paper}
    env = {k: v for k, v in os.environ.items() if k != llm_adapter.PAID_GATE}
    with mock.patch.dict(os.environ, env, clear=True), \
            mock.patch.object(llm_adapter.OpenAISender, '__init__', side_effect=AssertionError('the API sender was built in a check')):
        report = controllers.check(result, None, spec)
    expected = [p for p in report['problems'] if (m := re.match(r'event (\d+) \(.*?\): (traced policy|replay selects) ', p)) and int(m[1]) in moved]
    other = [p for p in report['problems'] if p not in expected]
    return {'ok': not other, 'problems': other, 'expected_on_base_trajectory': expected, 'verdict_of_check': report['verdict'],
            'incomplete': report.get('incomplete'), 'geometry': report.get('geometry'), 'provenance': report.get('provenance'), 'records': report.get('records'),
            'decision_points': report.get('decision_points'), 'switches': report.get('switches')}


def key_scan(run):
    """Files under run holding the key (or its first 12 characters) or an unmasked sk- key shape."""
    key, hits = os.environ.get('OPENAI_API_KEY', ''), []
    for path in sorted(p for p in run.rglob('*') if p.is_file()):
        data = path.read_bytes()
        if len(key) >= 12 and (key.encode() in data or key[:12].encode() in data):
            hits.append(f'{path.relative_to(run).as_posix()}: the key')
        elif path.suffix in ('.json', '.jsonl', '.txt', '.log') and \
                [m for m in controllers.KEY_PATTERN.findall(data.decode('utf-8', 'replace')) if m != 'sk-***']:
            hits.append(f'{path.relative_to(run).as_posix()}: an sk- key shape')
    return hits


def run_smoke(name, spec, out=OUT, after=None, loopback=False):
    """One smoke run (see the module docstring); returns its report, also written to report.json."""
    problems = preflight(name, spec, out, after, loopback, paid=True)
    if problems:
        raise SystemExit('refused, nothing sent:\n  ' + '\n  '.join(problems))
    run = out / name
    out.mkdir(parents=True, exist_ok=True)
    run.mkdir()  # fails if it exists: the name check above raced with another start
    full = controllers.full_spec(spec)
    journal = {'state': 'started', 'started_at': datetime.datetime.now().isoformat(timespec='seconds'), 'after': after,
               'wall': WALL.relative_to(run_case.REPO).as_posix(), 'spec': full, 'max_http_requests': max_requests(full),
               'mode': 'loopback mock' if loopback else 'paid', **provenance()}
    write_json(run / 'journal.json', journal, exclusive=True)  # before the first request
    report, transport = {}, None
    try:
        result = copy_wall(run)
        base, ledger = controllers.base_policy(run), []
        transport = controller_worker.ProcessTransport(full)
        records = controllers.rerun(result, full, base, transport=transport, ledger=ledger)
        write_jsonl(result / 'trace' / 'controller.jsonl', records)
        write_jsonl(result / 'trace' / 'transport.jsonl', ledger)
        receipt = json.loads((run / 'receipt.json').read_text(encoding='utf-8'))
        write_json(run / 'receipt.json', {**receipt, 'controller': full, 'smoke_of': receipt.get('status'), 'manifest': run_case.manifest(run)})
        reaped = transport.process is None and transport.unreaped is None and not multiprocessing.active_children()
        worker = controllers.worker_problems(ledger, False)[0]
        decisions = [r for r in records if r.get('transcripts')]
        spent = llm_adapter.spend(full['llm'], [t for r in records for t in r.get('transcripts') or []])
        rows = requests_of(records, llm_adapter.price(full['llm']))
        report = {'http_requests_recorded': sum(1 for x in rows if x.get('outcome') != 'abandoned'), 'max_http_requests': max_requests(full),
                  'decisions': [{'event': r['event'], 'source': r['source'], 'order': r['final']['order'], 'switched': r['final']['switched'],
                                 'calls': len(r['transcripts'])} for r in decisions],
                  'skips': sorted({r['source'] for r in records if not r.get('transcripts')}),
                  'requests': rows, 'tokens_reported': spent['tokens'], 'cost_usd_estimate': spent['cost_usd'], 'unknown_usage': spent['unknown'],
                  'workers': {'reaped': reaped, 'ledger_problems': worker}}
        report['check'] = offline_check(run, full)
        report['key_scan'] = key_scan(run)
        report['clean'] = bool(report['check']['ok'] and not report['key_scan'] and reaped and not worker and spent['unknown'] == 0)
        write_json(run / 'report.json', report)
        state = 'finished'
    except BaseException as error:  # the journal keeps the run blocked for inspection
        state, report = 'aborted', {'error': controllers.error_text(error)}
        raise
    finally:
        if transport is not None:
            transport.close('smoke_end')
        write_json(run / 'journal.json', {**journal, 'state': state, 'ended_at': datetime.datetime.now().isoformat(timespec='seconds'),
                                          'clean': report.get('clean'), 'error': report.get('error')})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('action', choices=['preflight', 'run', 'check'])
    parser.add_argument('--name', required=True)
    parser.add_argument('--decisions', type=int, default=1)
    parser.add_argument('--after', help='the earlier run this one follows (required once any run exists)')
    parser.add_argument('--confirm-paid', action='store_true', help='run: send real, paid requests')
    parser.add_argument('--loopback', action='store_true', help='tests only: OPENAI_BASE_URL is a local mock server')
    args = parser.parse_args(argv)
    spec = spec_of(args.decisions)
    if args.action == 'check':
        report = offline_check(OUT / args.name, controllers.full_spec(json.loads((OUT / args.name / 'journal.json').read_text(encoding='utf-8'))['spec']))
        report['key_scan'] = key_scan(OUT / args.name)
        print(json.dumps(report, indent=1))
        return 0 if report['ok'] and not report['key_scan'] else 1
    if args.action == 'preflight':
        problems = preflight(args.name, spec, OUT, after=args.after, loopback=args.loopback, paid=False)
        print(json.dumps({'spec': controllers.full_spec(spec), 'max_http_requests': max_requests(spec),
                          'problems': problems, 'ready_for_paid_run': not problems and not env_problems(args.loopback, True)}, indent=1))
        return 1 if problems else 0
    if not (args.confirm_paid or args.loopback):
        raise SystemExit('run needs --confirm-paid (real, paid requests)')
    report = run_smoke(args.name, spec, OUT, after=args.after, loopback=args.loopback)
    print(json.dumps({k: v for k, v in report.items() if k != 'requests'}, indent=1))
    return 0 if report.get('clean') else 1


if __name__ == '__main__':
    sys.exit(main())
