"""API smoke test of the ReAct controller (P2): a few real OpenAI decisions on the decision
trajectory of an existing base-policy wall (the Thangavelu noise80 Paper pilot wall), never a new
wall and never a change to Wang's algorithm: the wall's trace is replayed offline (rerun) and
only the controller's first `decisions` decision points call the API.

  python -m research.llm_walls.gpt_smoke preflight --name smoke-001            # checks only, sends nothing
  STONEPACKING_ALLOW_PAID_API=1 python -m research.llm_walls.gpt_smoke run --name smoke-001 --confirm-paid
  python -m research.llm_walls.gpt_smoke check --name smoke-001                # offline: never the API
  python -m research.llm_walls.gpt_smoke preflight --name smoke-002 --after smoke-001 \
      --observation-version 2 --target 184947496:10:9:single:0:left   # one decision point, pinned

Run it with the isolated venv that has the openai SDK (the stablepacking-baseline environment
stays untouched). What a run guarantees:
- the spec is SPEC (gpt-6.1-terra, max_retries 0, budgets) with `decisions` (1 by default, at
  most 3) and is checked against LIMITS before anything is sent; the worker is always a
  ProcessTransport (no inline paid calls); at most decisions * max_tool_calls HTTP requests;
- the key comes only from OPENAI_API_KEY (read by the worker; this process reads it only to scan
  the outputs for it); every setting of llm_api.CONNECTION_ENV (endpoint, headers, SDK log,
  organization/project, proxies, CA files, TLS key log) must be unset; the SDK client follows no
  redirect and reads no proxy or CA from the environment, so one attempt is one POST;
- one run at a time: the study's lock (research/studies/api-smoke/.lock) is created exclusively
  before anything else and removed only after the run's final journal state is written (a killed
  run leaves it: no run starts until a person has looked); llm/journal.json is created exclusively
  before the first request; a run name is used once (the archive's runs included); a run ends
  'finished' only if its billing is certain (no unknown usage, no worker failure, every worker reaped) and no key was found, otherwise 'uncertain' (or 'aborted' on an exception),
  and any run not finished blocks every later run until a person has looked; after a finished
  run the next one must name it (--after), so a request is never resent unknowingly;
- the first attempt of unknown usage stops every later request (APILLM, ControllerClient), and
  the run keeps the client's own counters (a worker that died after sending is unknown usage);
- afterwards: the controller check with Transcript Replay (ReplayLLM: the API sender cannot even
  be built), ledger and manifest, a key scan of every output file, and the worker cleanup check.
  The report keeps three results apart: api_success (a reply with known usage, a legal LLM
  decision, the replay passed, billing certain, workers reaped), clean (the run ended safely and
  checked, whether or not the API answered) and check.verdict (check()'s, once the expected
  differences of switched decisions are set aside; the raw one is verdict_of_check), PARTIAL here:
  the copied wall has no live outcome geometry, so the smoke validates the API flow, never
  geometry or wall quality.
--target SEED:STEP:EVENT:PHASE:TRIAL:SIDE pins the one decision point that may call the API: the
spec gets first_step STEP (every decision point of an earlier step is skip:before_first_step: its
observation is built, so the history is the trajectory's, but nothing is sent and no decision of
the budget is used) and needs max_decisions 1 (every later one is skip:wall_budget). The
preflight replays the trajectory offline with a scripted mock LLM (plan_of) and refuses unless
that one decision point is exactly the target (the wall's seed, event, step, phase, trial, side);
it never moves to another one. Right before each request reaches the worker the run checks the
copied wall's seed, the event and the request's decision key again (guard: else nothing is sent
and the run ends 'aborted'); after the run, a ledger request anywhere else ends it 'uncertain'.
--observation-version 2 gives the controller Observation and prompt version 2 (default 1).
--loopback is for tests only: it requires OPENAI_BASE_URL on 127.0.0.1/localhost (a mock server).

Each run is a new case, research/cases/<case id>/ (case_id.allocate with the copied wall's case):
receipt.json (the copied wall's, with schema_version 2, case_id, scene, started, parent: the copied
wall, the smoke's controller and spec_sha256; the run manifest once the replay has ended), the copied
result directory with the replay's controller.jsonl and transport.jsonl, and the LLM records
llm/journal.json and llm/report.json. The study research/studies/api-smoke names every run
({'case_id', 'name'}). The runs before research/cases (smoke-001, smoke-002) stay in
research/runs/api-smoke, read only: they are earlier runs of the same chain (names, --after, states).
"""
import argparse
import collections
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
import types
from unittest import mock

from research.walls import check_run
from research.controller import worker as controller_worker
from research.controller import order_controller
from research.controller import llm_api
from research.paths import LEGACY_RUNS, REPO
from research.storage import case_id as cid
from research.storage import studies
from research.walls import run_wall

WALL = LEGACY_RUNS / 'campaigns' / 'pilot-thangavelu-4arms' / 'thangavelu-noise80' / 'paper_3-0-2-1-4' / 'seed184947496'
STUDY = 'api-smoke'  # research/studies/api-smoke: one entry {'case_id', 'name'} per run
LEGACY = LEGACY_RUNS / 'api-smoke'  # the runs before research/cases, one folder per name; read only
LLM = 'llm'  # the folder of a case's LLM records: journal.json, report.json
MODEL = 'gpt-6.1-terra'
# USD per million tokens: input, output, cache write (the 12 h rate, the dearer), cached input:
# developers.openai.com pricing, checked 2026-11-07; check again before a paid run
PRICE = [1.00, 5.00, 2.50, 0.05]
LIMITS = {'decisions': 3, 'max_cost_usd': 0.50, 'max_total_tokens': 100_000, 'max_tokens': 4096,
          'request_timeout_s': 60.0, 'llm_timeout': 120.0, 'max_tool_calls': 3}
LOCK = '.lock'
GEOMETRY = 'PARTIAL: an offline smoke on the copied Paper trajectory: no live outcome geometry, no new wall; ' \
           'it validates the API flow, not the geometry, stability or quality of a wall built with the chosen orders'
REPO_FILES = ('research/llm_walls/gpt_smoke.py', 'research/controller/order_controller.py',
              'research/controller/worker.py', 'research/controller/llm_api.py')
Records = collections.namedtuple('Records', 'folder journal report')  # a run's folder, journal file and report file


def spec_of(decisions=1, observation_version=1, first_step=None):
    """The smoke spec; observation_version and first_step are in it only when they are not the default."""
    return {'kind': 'react',
            'llm': {'kind': 'api', 'provider': 'openai', 'model': MODEL, 'max_tokens': 4096, 'effort': 'low',
                    'request_timeout_s': 60.0, 'max_retries': 0, 'retry_backoff_s': 0.0, 'price_per_mtok': PRICE},
            'llm_timeout': 90.0, 'max_tool_calls': 3, 'max_decisions': decisions, 'max_total_tokens': 100_000,
            'max_cost_usd': 0.50, 'max_llm_failures': 1,
            **({'observation_version': observation_version} if observation_version != 1 else {}),
            **({'first_step': first_step} if first_step is not None else {})}


TARGET_KEYS = ('seed', 'step', 'event', 'phase', 'sample_trial', 'side')
DECISION_KEYS = ('step', 'phase', 'sample_trial', 'side')


def parse_target(text):
    """'SEED:STEP:EVENT:PHASE:TRIAL:SIDE' -> a dict of TARGET_KEYS; raises on anything else."""
    parts = text.split(':') if isinstance(text, str) else []
    if len(parts) != 6 or not all(p.isdigit() for p in parts[:3] + parts[4:5]) or not parts[3] or parts[5] not in ('left', 'right'):
        raise ValueError(f'a target is SEED:STEP:EVENT:PHASE:TRIAL:SIDE, got {text!r}')
    return dict(zip(TARGET_KEYS, [int(parts[0]), int(parts[1]), int(parts[2]), parts[3], int(parts[4]), parts[5]]))


def plan_of(spec):
    """Where the API would be called, offline: the wall's trajectory replayed inline with a scripted
    mock LLM ('max_llc': list_order_outcomes, explain_order, commit_order) in place of the API.
    Gives the decision points that would call the LLM ({event, decision key, n_outcomes}), the
    skipped ones by source and, for the first, each request the mock flow sends: bytes of the
    Responses API body and the adapter's reservation (what one attempt may cost at most). The
    real model's replies add their own output items (reasoning) to calls 2 and 3."""
    full = order_controller.full_spec(spec)
    mock_spec = {**full, 'llm': {'kind': 'mock', 'mode': 'max_llc'}}
    records = order_controller.rerun(check_run.result_dirs(WALL)[0], mock_spec, order_controller.base_policy(WALL))
    sizer = llm_api.APILLM(full['llm'], sender=object())  # reserve() only: no client, no key
    calls, estimate = [r for r in records if r.get('source') == 'llm'], []
    for t in calls[0]['transcripts'] if calls else []:
        payload = {'model': sizer.model, **sizer.params, 'messages': t['request']['messages'], 'tools': t['request']['tools']}
        body = llm_api.OpenAISender.request(types.SimpleNamespace(turns=[]), payload)
        tokens, usd = sizer.reserve(body)
        estimate.append({'call': t['call'], 'tool_called': (t['response']['parsed'] or {}).get('tool'),
                         'body_bytes': len(json.dumps(body, ensure_ascii=False).encode()),
                         'reserved_tokens': tokens, 'reserved_usd': round(usd, 6)})
    skipped = collections.Counter(r['source'] for r in records if r['decision_point'] and r['source'] != 'llm')
    return {'llm_decisions': [{'event': r['event'], **dict(zip(DECISION_KEYS, order_controller.decision_key(r))), 'n_outcomes': r['n_outcomes']}
                              for r in calls],
            'decision_points': sum(r['decision_point'] for r in records), 'skipped': dict(sorted(skipped.items())),
            'observation_version': order_controller.observation_version(full), 'requests_of_the_mock_flow': estimate}


def target_problems(spec, target):
    """Why the decision point the spec's controller would call the API at is not exactly the target
    (target None: then the spec may not set first_step)."""
    full = order_controller.full_spec(spec)
    if target is None:
        return ['target: first_step needs --target'] if 'first_step' in full else []
    problems = [] if full.get('first_step') == target['step'] else [f"target: spec first_step {full.get('first_step')}, target step {target['step']}"]
    if full['max_decisions'] != 1:
        problems.append('target: a pinned decision point allows max_decisions 1 only')
    seed = json.loads((WALL / 'receipt.json').read_text(encoding='utf-8')).get('seed')
    if seed != target['seed']:
        problems.append(f"target: the wall's seed is {seed}, not {target['seed']}")
    if problems:
        return problems
    want = {k: target[k] for k in TARGET_KEYS[1:]}
    traced = {meta['event']: {k: meta[k] for k in DECISION_KEYS} for meta, _ in order_controller.trace_events(check_run.result_dirs(WALL)[0])}
    if traced.get(target['event']) != {k: want[k] for k in DECISION_KEYS}:
        problems.append(f"target: trace event {target['event']} is {traced.get(target['event'])}, not {want}")
    calls = plan_of(spec)['llm_decisions']
    if [{k: c[k] for k in want} for c in calls] != [want]:
        problems.append(f"target: the controller would call the LLM at {calls}, not only at {want}")
    return problems


def max_requests(spec):
    return spec['max_decisions'] * spec['max_tool_calls'] * (spec['llm']['max_retries'] + 1)


def spec_problems(spec):
    """Why spec is not a smoke spec (LIMITS, no retry, a price, one LLM failure ends it)."""
    try:
        full, llm = order_controller.full_spec(spec), llm_api.full(spec['llm'])
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
             f"call_bound <= llm_timeout <= {LIMITS['llm_timeout']}": within(full['llm_timeout'], llm_api.call_bound(llm), LIMITS['llm_timeout']),
             f"1 <= max_tool_calls <= {LIMITS['max_tool_calls']}": 1 <= full['max_tool_calls'] <= LIMITS['max_tool_calls'],
             'max_llm_failures 1': full['max_llm_failures'] == 1,
             'first_step only with max_decisions 1': 'first_step' not in full or full['max_decisions'] == 1,
             f'the price table PRICE {PRICE}': list(llm['price_per_mtok'] or []) == PRICE}
    return [f'spec: needs {rule}' for rule, ok in rules.items() if not ok]


def env_problems(loopback, paid):
    """Why this process may not send: the gate, the key variable, any connection setting of
    llm_api.CONNECTION_ENV (names only), the SDK. --loopback needs OPENAI_BASE_URL on a
    loopback host (a mock server); otherwise it must be unset too."""
    problems, env = [], os.environ
    if paid and env.get(llm_api.PAID_GATE) != '1':
        problems.append(f'env: {llm_api.PAID_GATE}=1 is not set for this command')
    if 'OPENAI_API_KEY' not in env:
        problems.append('env: OPENAI_API_KEY is not set')
    refused = llm_api.connection_env() + ([] if loopback else [k for k in env if k.upper() == 'OPENAI_BASE_URL'])
    problems += [f'env: {v} must be unset (it changes where a request goes, its TLS, headers or logging)' for v in sorted(set(refused))]
    if loopback and not any(k.upper() == 'OPENAI_BASE_URL' for k in env if k not in refused):
        problems.append('env: --loopback needs OPENAI_BASE_URL on 127.0.0.1 or localhost')
    if importlib.util.find_spec('openai') is None:
        problems.append(f'env: the openai SDK is not importable by {sys.executable} (use the isolated venv)')
    return problems


def case_records(folder):
    """The records of a run that is a case: its LLM records are in <case>/llm/."""
    return Records(folder, folder / LLM / 'journal.json', folder / LLM / 'report.json')


def lock_of(study):
    """The lock of a study's runs (one run at a time), in the study's folder."""
    return studies.path_of(study).parent / LOCK


def study_runs(study):
    """{name: Records} of a study's runs in the order they were added; {} before its first run."""
    if not studies.path_of(study).exists():
        return {}
    return {e['name']: case_records(cid.wall_dir(e['case_id'])) for e in studies.load(study)['cases']}


def join_study(study, kind, design, entry):
    """Add a run's entry to its study, which the first run creates with `design`; only under the study's lock."""
    if not studies.path_of(study).exists():
        studies.create(study, kind, design)
    studies.add_case(study, entry)


def read_journal(path):
    """The JSON file at path; None when there is none, {'state': 'unreadable'} when it does not parse."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except ValueError:
        return {'state': 'unreadable'}


def all_runs():
    """{name: Records} of every smoke run: the archive's (LEGACY/<name>/journal.json), then the study's cases."""
    old = {d.name: Records(d, d / 'journal.json', d / 'report.json') for d in sorted(LEGACY.iterdir()) if d.is_dir()} \
        if LEGACY.is_dir() else {}
    return {**old, **study_runs(STUDY)}


def journal_problems(name, after, decisions, holding=False):
    """Why run `name` may not start: another run's lock, a used name, an earlier run not finished
    (started, aborted, uncertain) or without a journal, a missing --after. holding: this process holds the lock."""
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,40}', name):
        return [f'run: the name {name!r} is not [a-z0-9-]']
    lock = lock_of(STUDY)
    if not holding and lock.exists():
        return [f'run: {lock} exists: another run is starting or running, or one ended without releasing it; '
                'inspect it (it is never removed automatically)']
    found = all_runs()
    if name in found:
        return [f'run: {name} was used ({found[name].folder}): a run name is used once, its requests are never resent']
    journals = {n: read_journal(r.journal) for n, r in found.items()}
    runs = {n: j for n, j in journals.items() if j is not None}
    problems = [f'run: {n} is {j.get("state")}{" (" + "; ".join(j["reasons"]) + ")" if j.get("reasons") else ""}: '
                'inspect it (its billing may be unknown) before any new run' for n, j in runs.items() if j.get('state') != 'finished']
    problems += [f'run: {n} has no journal: inspect it before any new run' for n, j in journals.items() if j is None]
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


def preflight(name, spec, after=None, loopback=False, paid=False, holding=False, target=None):
    """Every problem that forbids the run; empty: it may send at most max_requests(spec) requests,
    all at the target decision point if one is given."""
    problems = spec_problems(spec) + env_problems(loopback, paid) + wall_problems() + \
        journal_problems(name, after, spec.get('max_decisions', 0) if isinstance(spec, dict) else 0, holding)
    return problems + (target_problems(spec, target) if not spec_problems(spec) and not wall_problems() else [])


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
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=REPO).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', *REPO_FILES], capture_output=True, text=True, cwd=REPO).stdout.strip()
    spec_ = importlib.util.find_spec('openai')
    version = None
    if spec_ is not None:
        import openai
        version = openai.__version__
    return {'git_head': head, 'code_dirty': bool(dirty), 'python': sys.executable, 'openai_sdk': version,
            'code_sha256': {f: hashlib.sha256((REPO / f).read_bytes()).hexdigest() for f in REPO_FILES}}


def copy_wall(dst):
    """The wall's first result directory (trace, decisions.txt, base.pkl) under dst."""
    src = check_run.result_dirs(WALL)[0]
    result = dst / src.relative_to(WALL)
    shutil.copytree(src / 'trace', result / 'trace')
    for f in ('decisions.txt', 'base.pkl'):
        shutil.copy2(src / f, result / f)
    return result


def case_receipt(source, case_id, started, spec):
    """The receipt of a smoke case: the copied wall's (source), with the case's own fields, the copied
    wall as its parent and the smoke's controller; the run manifest is added once the replay has ended."""
    receipt = {**source, 'schema_version': 2, 'case_id': case_id, 'scene': cid.scene_code(source['case']),
               'parent': WALL.relative_to(LEGACY_RUNS).as_posix(), 'started': started.isoformat(timespec='seconds'),
               'controller': spec, 'smoke_of': source.get('status')}
    return {**receipt, 'spec_sha256': cid.spec_sha256(receipt)}


def requests_of(records, prices):
    """Every API request the records hold: decision, call, attempt, outcome, billing, HTTP status,
    request id, latency, error, and per call the usage and its estimated cost; a decision whose
    worker failed (no transcript came back, an API request may have been sent) is one row of
    unknown billing."""
    rows = []
    for r in records:
        if r.get('source', '').startswith('fallback:worker'):
            rows.append({'event': r['event'], 'decision': list(order_controller.decision_key(r)), 'outcome': 'worker_failure',
                         'status': r['source'], 'billed': 'unknown', 'error': r.get('error')})
        for t in r.get('transcripts') or []:
            response = t.get('response') or {}
            usage = response.get('usage')
            call_cost = llm_api.cost(prices, usage) if llm_api.usage_ok(usage) else None
            for a in response.get('attempts') or []:
                rows.append({'event': r['event'], 'decision': list(order_controller.decision_key(r)), 'call': t['call'],
                             **{k: a.get(k) for k in ('attempt', 'outcome', 'billed', 'status_code', 'request_id', 'latency_s', 'error')},
                             'usage': usage if a.get('outcome') == 'reply' else None,
                             'cost_usd_estimate': call_cost if a.get('outcome') == 'reply' else None,
                             'served_model': ((response.get('exchange') or {}).get('response') or {}).get('model') if a.get('outcome') == 'reply' else None})
            if t.get('status') == 'timeout':  # abandoned in flight: counted as unknown
                rows.append({'event': r['event'], 'decision': list(order_controller.decision_key(r)), 'call': t['call'],
                             'outcome': 'abandoned', 'billed': 'unknown', 'error': response.get('error')})
    return rows


def expected_problems(result, records, base):
    """The exact check() problems an LLM decision with an order other than Paper must produce on a
    trajectory built with Paper ('traced policy': traced order Paper, controller order not;
    'replay selects': the controller's pick differs from the traced one), for the decisions whose
    structured fields show that nothing else differs: the source is 'llm', the traced policy is
    the base policy in full (tol and every other key) with the Paper order, the traced pick is
    select()'s with Paper, and the record's index is select()'s with its committed order. Each is
    the full message check() writes for that event (format: order_controller.check); anything else,
    or the same message at any other event, stays a problem."""
    paper, allowed = list(order_controller.PAPER_ORDER), set()
    rest = lambda p: {k: v for k, v in p.items() if k != 'order'}
    plain = json.loads(json.dumps(base))
    by_event = {r['event']: r for r in records if isinstance(r, dict) and isinstance(r.get('final'), dict)}
    for meta, a in order_controller.trace_events(result):
        r = by_event.get(meta['event'])
        if r is None or r.get('source') != 'llm' or r['final']['order'] == order_controller.key(paper):
            continue
        if rest(meta['policy']) != rest(plain) or list(meta['policy']['order'] or paper) != paper or list(base['order'] or paper) != paper:
            continue
        pick = lambda order: order_controller.select(a['features_raw'], a['weights'], 'paper', order, base['tol']).index
        traced, by_paper, index = int(a['selected']), pick(paper), pick(order_controller.parse_order(r['final']['order']))
        if (-1 if by_paper is None else int(by_paper)) != traced or r['final']['index'] != (None if index is None else int(index)):
            continue
        where = f"event {meta['event']} (step {meta['step']} {meta['phase']} trial {meta['sample_trial']} {meta['side']})"
        allowed.add(f"{where}: traced policy {meta['policy']}, base {base}, controller order {r['final']['order']}")
        allowed.add(f"{where}: replay selects {index}, traced {traced}, controller {r['final']['index']}")
    return allowed


def offline_check(run, spec):
    """The controller check with Transcript Replay; the problems an LLM decision with another order
    than the traced Paper one must produce (expected_problems, checked on structured fields) are
    listed apart, every other problem fails the check. The API sender cannot be built meanwhile."""
    result = check_run.result_dirs(run)[0]
    records = [json.loads(l) for l in (result / 'trace' / 'controller.jsonl').read_text(encoding='utf-8').splitlines()]
    env = {k: v for k, v in os.environ.items() if k != llm_api.PAID_GATE}
    with mock.patch.dict(os.environ, env, clear=True), \
            mock.patch.object(llm_api.OpenAISender, '__init__', side_effect=AssertionError('the API sender was built in a check')):
        report = order_controller.check(result, None, spec)
    allowed = expected_problems(result, records, order_controller.base_policy(run))
    expected = [p for p in report['problems'] if p in allowed]
    other = [p for p in report['problems'] if p not in allowed]
    verdict = 'FAIL' if other or not report['records'] else 'PARTIAL' if report.get('incomplete') else 'PASS'
    return {'ok': not other and bool(report['records']), 'problems': other, 'expected_on_base_trajectory': expected,
            'verdict': verdict, 'verdict_of_check': report['verdict'],
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
                [m for m in order_controller.KEY_PATTERN.findall(data.decode('utf-8', 'replace')) if m != 'sk-***']:
            hits.append(f'{path.relative_to(run).as_posix()}: an sk- key shape')
    return hits


def refuse(problems):
    raise SystemExit('refused, nothing sent:\n  ' + '\n  '.join(problems))


def run_smoke(name, spec, after=None, loopback=False, target=None):
    """One smoke run (see the module docstring) under the study's lock; returns its report (llm/report.json)."""
    problems = preflight(name, spec, after, loopback, paid=True, target=target)
    if problems:
        refuse(problems)
    lock = lock_of(STUDY)
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_json(lock, {'name': name, 'pid': os.getpid(), 'at': datetime.datetime.now().isoformat(timespec='seconds')}, exclusive=True)
    except FileExistsError:
        refuse(journal_problems(name, after, 0))
    try:
        problems = preflight(name, spec, after, loopback, paid=True, holding=True, target=target)  # again, under the lock
        if problems:
            refuse(problems)
        return locked_run(name, spec, after, loopback, target)
    finally:  # only after the run's final journal state is written (a killed run leaves the lock)
        lock.unlink()


def guard(transport, client, target, run):
    """Checks the target once more right before each request reaches the worker: the copied wall's
    seed, the event number (the client's call count) and the decision key of the observation the
    request carries must be the target's; otherwise it raises, nothing is sent and the run ends
    'aborted' (pinned by the preflight, this only fails if the trajectory changed in between)."""
    seed, send = json.loads((run / 'receipt.json').read_text(encoding='utf-8')).get('seed'), transport.decide
    at = (target['seed'], target['event'], *(target[k] for k in DECISION_KEYS))

    def decide(request, settle=None):
        obs = request['observation']
        here = (seed, client.calls, *(obs.get(k) for k in DECISION_KEYS))
        if here != at:
            raise RuntimeError(f'refused before sending: the request is for {here}, the target is {at}')
        return send(request, settle)
    transport.decide = decide


def locked_run(name, spec, after, loopback, target=None):
    """The run in a new case: allocated and named in the study first, its journal before the first request."""
    source = json.loads((WALL / 'receipt.json').read_text(encoding='utf-8'))
    started = datetime.datetime.now().astimezone()
    case_id, run = cid.allocate(source['case'], started)
    design = {'runner': 'research/llm_walls/gpt_smoke.py', 'wall': WALL.relative_to(LEGACY_RUNS).as_posix(), 'limits': LIMITS}
    join_study(STUDY, STUDY, design, {'case_id': case_id, 'name': name})
    files = case_records(run)
    files.journal.parent.mkdir()
    full = order_controller.full_spec(spec)
    journal = {'state': 'started', 'started_at': datetime.datetime.now().isoformat(timespec='seconds'), 'after': after,
               'case_id': case_id, 'wall': WALL.relative_to(REPO).as_posix(), 'spec': full,
               'max_http_requests': max_requests(full),
               'mode': 'loopback mock' if loopback else 'paid', **({'target': target} if target else {}), **provenance()}
    write_json(files.journal, journal, exclusive=True)  # before the first request
    report, transport, reasons = {}, None, []
    try:
        result = copy_wall(run)
        write_json(run / 'receipt.json', case_receipt(source, case_id, started, full))
        base, ledger, records = order_controller.base_policy(run), [], []
        transport = controller_worker.ProcessTransport(full)
        # order_controller.rerun's loop, with the client kept: its own unknown_usage counts a worker
        # failure after which no transcript came back (a request may have been sent)
        client = order_controller.ControllerClient(full, transport, records.append, ledger.append)
        if target is not None:
            guard(transport, client, target, run)
        try:
            for meta, a in order_controller.trace_events(result):
                client(types.MappingProxyType(dict(base)), order_controller.context_from_trace(meta, a))
        finally:
            transport.close()
        records = json.loads(json.dumps(records, default=lambda v: v.item()))
        write_jsonl(result / 'trace' / 'controller.jsonl', records)
        write_jsonl(result / 'trace' / 'transport.jsonl', ledger)
        receipt = json.loads((run / 'receipt.json').read_text(encoding='utf-8'))
        write_json(run / 'receipt.json', {**receipt, 'manifest': run_wall.manifest(run)})
        reaped = transport.process is None and transport.unreaped is None and not multiprocessing.active_children()
        worker = order_controller.worker_problems(ledger, False)[0]
        decisions = [r for r in records if r.get('decision_point') and not r['source'].startswith('skip')]
        spent = llm_api.spend(full['llm'], [t for r in records for t in r.get('transcripts') or []])
        rows = requests_of(records, llm_api.price(full['llm']))
        failures = [x for x in ledger if x.get('type') == 'request' and x.get('status') != 'ok']
        unknown = max(client.unknown_usage, spent['unknown'] + len(failures))  # the client's count, never less than the records'
        report = {'http_requests_recorded': sum(1 for x in rows if x.get('outcome') in ('reply', 'retried', 'failed')),
                  'max_http_requests': max_requests(full),
                  'decisions': [{'event': r['event'], 'source': r['source'], 'order': r['final']['order'], 'switched': r['final']['switched'],
                                 'calls': len(r.get('transcripts') or [])} for r in decisions],
                  'skips': sorted({r['source'] for r in records if r not in decisions}),
                  'requests': rows, 'tokens_reported': spent['tokens'], 'cost_usd_estimate': spent['cost_usd'],
                  'unknown_usage': unknown, 'client': {'unknown_usage': client.unknown_usage, 'request_failures': client.request_failures,
                                                       'llm_decisions': client.llm_decisions, 'tokens': client.tokens, 'cost_usd': client.cost_usd},
                  'worker_failures': [{k: x.get(k) for k in ('event', 'status', 'error', 'worker_pid')} for x in failures],
                  'workers': {'reaped': reaped, 'ledger_problems': worker}, 'geometry_validation': GEOMETRY}
        report['check'] = offline_check(run, full)
        write_json(files.report, report)
        report['key_scan'] = key_scan(run)  # llm/report.json included
        elsewhere = []
        if target is not None:  # every request of the ledger at the target decision point, nothing else
            at = (target['event'], *(target[k] for k in DECISION_KEYS))
            elsewhere = [x.get('event') for x in ledger if x.get('type') == 'request' and (x.get('event'), *(x.get(k) for k in DECISION_KEYS)) != at]
            report['target'] = {**target, 'requests_elsewhere': elsewhere}
        reasons = [why for why, bad in (('unknown usage', unknown), ('a worker failure', failures), ('a worker not reaped', not reaped),
                                        ('worker lifecycle problems', worker), ('the key in an output file', report['key_scan']),
                                        ('a request outside the target decision point', elsewhere)) if bad]
        report['clean'] = bool(report['check']['ok'] and not reasons)
        report['api_success'] = bool(report['clean'] and any(x.get('outcome') == 'reply' and x.get('billed') == 'reported' for x in rows)
                                     and any(r['source'] == 'llm' for r in decisions))
        write_json(files.report, report)
        state = 'uncertain' if reasons else 'finished'
    except BaseException as error:  # the journal keeps the run blocked for inspection
        state, report, reasons = 'aborted', {'error': order_controller.error_text(error)}, ['an exception: ' + order_controller.error_text(error)]
        raise
    finally:
        if transport is not None:
            transport.close('smoke_end')
        write_json(files.journal, {**journal, 'state': state, 'reasons': reasons, 'ended_at': datetime.datetime.now().isoformat(timespec='seconds'),
                                          'clean': report.get('clean'), 'api_success': report.get('api_success'), 'error': report.get('error')})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('action', choices=['preflight', 'run', 'check'])
    parser.add_argument('--name', required=True)
    parser.add_argument('--decisions', type=int, default=1)
    parser.add_argument('--after', help='the earlier run this one follows (required once any run exists)')
    parser.add_argument('--confirm-paid', action='store_true', help='run: send real, paid requests')
    parser.add_argument('--loopback', action='store_true', help='tests only: OPENAI_BASE_URL is a local mock server')
    parser.add_argument('--observation-version', type=int, choices=[1, 2], default=1)
    parser.add_argument('--target', type=parse_target, metavar='SEED:STEP:EVENT:PHASE:TRIAL:SIDE',
                        help='the one decision point that may call the API (first_step STEP; --decisions 1)')
    args = parser.parse_args(argv)
    target = args.target
    spec = spec_of(args.decisions, args.observation_version, None if target is None else target['step'])
    if args.action == 'check':
        run = all_runs().get(args.name)
        if run is None:
            raise SystemExit(f'check: no smoke run is named {args.name}')
        journal = json.loads(run.journal.read_text(encoding='utf-8'))
        report = offline_check(run.folder, order_controller.full_spec(journal['spec']))
        report['key_scan'] = key_scan(run.folder)
        print(json.dumps(report, indent=1))
        return 0 if report['ok'] and not report['key_scan'] else 1
    if args.action == 'preflight':
        problems = preflight(args.name, spec, after=args.after, loopback=args.loopback, paid=False, target=target)
        full = order_controller.full_spec(spec)
        print(json.dumps({'spec': full, 'observation_version': order_controller.observation_version(full), 'target': target,
                          'plan': plan_of(spec) if not spec_problems(spec) and not wall_problems() else None,
                          'limits': {'transport': 'ProcessTransport', 'max_http_requests': max_requests(spec),
                                     'max_retries': full['llm']['max_retries'], 'max_total_tokens': full['max_total_tokens'],
                                     'max_output_tokens_per_call': full['llm']['max_tokens'], 'max_cost_usd': full['max_cost_usd']},
                          'max_http_requests': max_requests(spec),
                          'problems': problems, 'ready_for_paid_run': not problems and not env_problems(args.loopback, True)}, indent=1))
        return 1 if problems else 0
    if not (args.confirm_paid or args.loopback):
        raise SystemExit('run needs --confirm-paid (real, paid requests)')
    report = run_smoke(args.name, spec, after=args.after, loopback=args.loopback, target=target)
    summary = {'api_success': report['api_success'], 'clean': report['clean'], 'check_verdict': report['check']['verdict'],
               'geometry_validation': report['geometry_validation']}
    print(json.dumps({**summary, **{k: v for k, v in report.items() if k not in summary and k != 'requests'}}, indent=1))
    return 0 if report['api_success'] else 1


if __name__ == '__main__':
    sys.exit(main())
