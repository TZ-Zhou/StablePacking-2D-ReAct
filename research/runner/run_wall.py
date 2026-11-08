"""Run an original test case with a chosen filter policy and candidate tracing.

Usage: python -B -m research.runner.run_wall <case> [--policy code_original|paper] [--order 3,0,2,1,4]
                                      [--tol 1e-9] [--placements N] [--seed S] [--parent WALL]
                                      [--switch STEP=a,b,c,d,e ...] [--controller JSON] [--workers N] [--sync-output]

Staging follows the frozen baseline run_original_case.py (copied inputs, Windows main guard,
shell cleanup refused, spawned workers inherit constants/RNG via windows_case_runtime.py).
research/validation/ holds tracked copies of the two baseline tools used here and the registered
environment manifest; baseline/ itself is never read. Output goes to a new research/cases/<case id>
(research/storage/case_id.py); the research/runs archive is never written. The receipt there records
the definition and outcome, and derived/steps.csv the policy and order of every traced selection.
--parent names the wall a branch or lookahead wall starts from: a case id or a research/runs path.
With --placements N the case is stopped once decisions.txt has N committed placements.
--switch makes a one-step counterfactual branch: same seed and code, so every decision before
STEP should repeat the run without it; verify that with check_run.py --compare.
--controller '{"kind": "null"}' (research/controller/order_controller.py) picks the order at each selection;
verify its records with order_controller.py check <run>.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime

from research.paths import REPO, VALIDATION
from research.storage import case_id as cid
from research.storage import case_index
from research.validation import local_setup  # this computer's paths; the registered ones without a local-setup file
from research.walls import case_definition
ENV_PREFIX = local_setup.env_prefix(json.loads((VALIDATION / 'environment-manifest.json').read_text(encoding='utf-8'))['prefix'])
BAR_CELLS = 24
BAR_LINE_WIDTH = 79


class TerminationFailed(RuntimeError):
    """The case's process tree was not verified stopped: nothing else may be launched. `receipt` is the
    wall's receipt, so a batch can list the wall in its study before it stops."""

    def __init__(self, receipt):
        super().__init__('case process tree not verified stopped; do not launch another case')
        self.receipt = receipt


def prepare_output(case, start, root=None):
    """A new folder named by its case id (case_id.allocate), so neither baseline data nor the research/runs
    archive is ever written. Returns (case id, folder)."""
    case_id, output = cid.allocate(case, start, root=root)
    return case_id, output.resolve()


def committed_placements(output):
    return sum(len(p.read_text(encoding='utf-8').splitlines()) for p in output.glob('result*/decisions.txt'))


def keep_data_only(output):
    """Remove what the finished case no longer needs: the copies of its inputs (they stay in Tests/), the runtime
    copy, matplotlib's cache and Wang's image and record folders once they hold nothing."""
    for name in ('data', 'matplotlib-config'):
        shutil.rmtree(output / name, ignore_errors=True)
    (output / 'windows_case_runtime.py').unlink(missing_ok=True)
    (output / "step_writer.py").unlink(missing_ok=True)
    for result in output.glob('result*'):
        (result / 'img' / 'selection.txt').unlink(missing_ok=True)  # with the records off, only its header line
        for folder in (result / 'img', result / 'record'):
            if folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()


def stones_given(output):
    """The number of stones the case was given, None before its building loop has started."""
    path = output / "events.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True) if path.exists() else []
    enter = next((line for line in lines if '"build_enter"' in line and line.endswith("\n")), None)
    return None if enter is None else json.loads(enter)["input_stones"]


def bar_line(label, placed, total, seconds):
    """One line of progress. A stopped wall can overrun its limit, so the count shown never passes the total."""
    if not total:
        return f"{label}: {placed} stones, {seconds:.0f} s"
    shown = min(placed, total)
    cells = BAR_CELLS * shown // total
    left = f"{seconds * (total - shown) / shown:.0f} s left" if shown else "time left unknown"
    return f"{label}: {shown}/{total} [{'#' * cells}{'.' * (BAR_CELLS - cells)}] {seconds:.0f} s, {left}"


class StoneBar:
    """Stones committed so far, on a terminal only. Reading the case's files for it can never stop the case."""

    def __init__(self, output, limit, label):
        self.output, self.limit, self.label = output, limit, label
        self.start, self.active = time.monotonic(), sys.stderr.isatty()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.refresh()
        if self.active:
            sys.stderr.write("\n")

    def refresh(self):
        if not self.active:
            return
        with contextlib.suppress(OSError, ValueError):  # the case is writing these files
            given = stones_given(self.output)
            total = min(given, self.limit or given) if given else self.limit
            line = bar_line(self.label, committed_placements(self.output), total, time.monotonic() - self.start)
            sys.stderr.write(f"\r{line:<{BAR_LINE_WIDTH}}")
            sys.stderr.flush()


def process_table():
    """{pid: (parent pid, creation ticks)}; creation time guards against PID reuse."""
    out = subprocess.run(['powershell', '-NoProfile', '-Command',
                          'Get-CimInstance Win32_Process | ForEach-Object '
                          '{ "$($_.ProcessId),$($_.ParentProcessId),$($_.CreationDate.Ticks)" }'],
                         capture_output=True, text=True, timeout=120, check=True, env=case_definition.quiet_env()).stdout.split()
    return {int(p): (int(pp), c) for p, pp, c in (line.split(',') for line in out)}


def process_tree(table, root):
    found, todo = set(), [root]
    while todo:
        pid = todo.pop()
        found.add(pid)
        todo += [child for child, (parent, _) in table.items() if parent == pid and child not in found]
    return {pid: table[pid][1] for pid in found if pid in table}


def terminate(pid, settle_seconds=30):
    """taskkill the tree, then verify every member (same PID and creation time) is gone. If the
    process listing fails, taskkill still runs, and the stop is unverified."""
    try:
        members = process_tree(process_table(), pid)
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        members, listing = None, f"{type(error).__name__}: {error}"
    try:
        kill = subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], capture_output=True,
                              text=True, errors='replace', timeout=60, env=case_definition.quiet_env())
        record = {'taskkill_returncode': kill.returncode, 'stdout': kill.stdout, 'stderr': kill.stderr}
    except subprocess.TimeoutExpired:
        record = {'taskkill_returncode': None, 'error': 'taskkill timed out'}
    if members is None:
        record.update(tree=None, alive_after=None, verified=False, listing_error=listing)
        return record
    deadline = time.monotonic() + settle_seconds
    while True:
        try:
            table = process_table()
        except (subprocess.SubprocessError, OSError, ValueError) as error:
            record.update(tree=sorted(members), alive_after=None, verified=False, listing_error=f'{type(error).__name__}: {error}')
            return record
        alive = sorted(p for p, created in members.items() if table.get(p, (None, None))[1] == created)
        if not alive or time.monotonic() > deadline:
            break
        time.sleep(1)
    record.update(tree=sorted(members), alive_after=alive, verified=not alive)
    return record


def check_commits(output, stopped=False):
    """Every result directory's trace chain (check_run.chain_problems): events, commits.jsonl and
    decisions.txt agree selection by selection; a run stopped by the launcher may end in a prefix
    of build_wall's write order."""
    from research.walls import check_run
    from Stonepacker2D.stack.selection import load_trace, published_events
    problems = []
    for result in check_run.result_dirs(output):
        trace = result / 'trace'
        if not (trace / 'commits.jsonl').exists() and not stopped:
            problems.append(f'{trace}: no commits.jsonl')
            continue
        events = [load_trace(p) for p in published_events(trace)]
        problems += [f'{result}: {p}' for p in check_run.chain_problems(result, events, stopped)]
    return problems


def check_trace(output, stopped=False):
    """Trace files must exist, be numbered without gaps and load; otherwise the data is incomplete."""
    from Stonepacker2D.stack.selection import load_trace, published_events
    files = [Path(p) for t in sorted(output.rglob('trace')) if t.is_dir() for p in published_events(t)]
    numbers = [int(f.stem[1:]) for f in files]
    problems = []
    if not files:
        problems.append('no trace files')
    elif numbers != list(range(len(files))):
        problems.append('event numbers are not contiguous from 0')
    for f in files:
        try:
            load_trace(f)
        except Exception as error:  # any unreadable file makes the run incomplete
            problems.append(f'{f.name}: {error!r}')
    if not problems:  # the chain check needs every event loadable
        problems += check_commits(output, stopped)
    return {'files': len(files), 'ok': not problems, 'problems': problems}


def check_controller(output, trace, stopped=False):
    """A controller run is complete only if every result's controller check gives the verdict PASS
    (order_controller.check: records, transcripts and ledger equal the receipt's controller re-executed
    on the trace, every outcome geometry verified, ledger and run manifest present and matching)
    and the ledgers show every worker reaped (order_controller.worker_problems)."""
    from research.walls import check_run
    from research.controller import order_controller
    try:
        results = check_run.result_dirs(output)
        reports = [order_controller.check(r, stopped=stopped) for r in results]
        problems = [p for r in reports for p in r['problems'][:5]] + ([] if reports else ['no result directory'])
        problems += [f"not fully verified: {p}" for r in reports for p in r.get('incomplete', [])]
        entries = [json.loads(l) for r in results if (r / 'trace' / 'transport.jsonl').exists()
                   for l in (r / 'trace' / 'transport.jsonl').read_text(encoding='utf-8').splitlines()]
        workers, counts = order_controller.worker_problems(entries, stopped)
        problems += workers
    except Exception as error:  # an unreadable record or trace leaves the run unverified
        reports, problems, counts = [], [f'controller check failed: {error!r}'], None
    return {**trace, 'ok': trace['ok'] and not problems, 'problems': trace['problems'] + problems, 'workers': counts,
            'controller': [{k: r[k] for k in ('ok', 'verdict', 'records', 'decision_points', 'switches', 'unverified_trailing_records',
                                              'geometry', 'provenance', 'incomplete') if k in r} for r in reports]}


def verified_trace(output, args, process_status, termination):
    """The trace check of a finished case: trace files and commits, and for a controller run that
    ended successfully (complete or stopped after --placements) the controller check as well."""
    if args.no_trace:
        return {'ok': True, 'skipped': 'no-trace control run'}
    trace = check_trace(output, stopped=termination is not None)
    if args.controller is not None and process_status in ('PROCESS_PASS', 'STOPPED_AFTER_PLACEMENTS') and trace['ok']:
        trace = check_controller(output, trace, stopped=process_status == 'STOPPED_AFTER_PLACEMENTS')
    return trace


def write_steps(output):
    """derived/steps.csv of a traced wall, one row per sampling attempt: wall_metrics.step_rows of every result, the
    wall's four metrics after it (partial_quality) and its wall time; returns the number of rows."""
    from research.walls import check_run, wall_metrics

    times = iter(wall_metrics.step_wall_times(output / "events.jsonl"))
    rows = [
        {"result": r.name, **row, **quality, **next(times, {})}
        for r in check_run.result_dirs(output)
        for row, quality in zip(wall_metrics.step_rows(r), wall_metrics.partial_quality(r))
    ]
    path = output / 'derived' / 'steps.csv'
    path.parent.mkdir(exist_ok=True)
    case_index.write_csv(path, list(dict.fromkeys(k for row in rows for k in row)), rows)
    return len(rows)


def final_status(process_status, trace, termination):
    if termination is not None and not termination['verified']:
        return 'TERMINATION_FAILED'
    if process_status in ('PROCESS_PASS', 'STOPPED_AFTER_PLACEMENTS') and not trace['ok']:
        return 'INCOMPLETE_TRACE'
    return process_status


def run(argv=None, root=None, label=None):
    """Build one wall in a new case folder under `root` (default research/cases); returns its receipt (also written
    to receipt.json there). `label` titles the wall's progress bar (default the case name)."""
    parser = argparse.ArgumentParser()
    parser.add_argument('case', choices=case_definition.CASES)
    parser.add_argument('--parent', metavar='WALL',
                        help='the wall this one branches from: a case id or a research/runs path')
    parser.add_argument('--policy', default='code_original', choices=['code_original', 'paper'])
    parser.add_argument('--order', type=lambda s: tuple(int(c) for c in s.split(',')))
    parser.add_argument('--tol', type=float, default=1e-9)
    parser.add_argument('--placements', type=int, default=0, help='stop after N placements (0 = run to the end)')
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--no-trace', action='store_true', help='control run: logging off, outputs only')
    parser.add_argument(
        "--fresh-pool", action="store_true", help="control run: new workers for every batch, as the original code does"
    )
    parser.add_argument("--workers", type=int, help="worker processes instead of the case's own number")
    parser.add_argument(
        "--sync-output", action="store_true", help="write step outputs in the building loop, not a thread"
    )
    parser.add_argument('--seed', type=int, help='seed_everything(seed) instead of the source default (20)')
    parser.add_argument('--switch', action='append', default=[], metavar='STEP=a,b,c,d,e',
                        help='paper policy only: use this order at construction step STEP (every sampling trial of that step), the --order elsewhere; repeatable')
    parser.add_argument('--worker-venv', action='store_true',
                        help='with --controller: the controller worker runs in the registered venv (research/validation/worker-venv.json)')
    parser.add_argument('--controller', type=json.loads, metavar='JSON',
                        help='paper policy only: research/controller/order_controller.py spec, e.g. {"kind": "random", "seed": 7}')
    args = parser.parse_args(argv)
    if args.parent is not None and (problem := case_definition.parent_problem(args.parent)):
        parser.error(f'--parent: {problem}')
    if args.worker_venv and args.controller is None:
        parser.error('--worker-venv needs --controller')
    if args.controller is not None:
        if args.policy != 'paper' or args.switch or args.no_trace:
            parser.error('--controller needs --policy paper, tracing and no --switch')
        if args.worker_venv:
            from research.controller import worker as controller_worker
            problems = controller_worker.venv_problems()
            if problems:
                parser.error('--worker-venv: ' + '; '.join(problems))
        from research.controller import order_controller
        try:
            order_controller.make_controller(args.controller, sink=lambda record: None)
        except (ValueError, TypeError, NotImplementedError) as error:
            parser.error(f'--controller: {error}')
    assert Path(sys.prefix).resolve() == ENV_PREFIX, sys.prefix

    started = datetime.now().astimezone()
    case_id, output = prepare_output(args.case, started, root)
    guard = subprocess.run(
        [sys.executable, "-B", str(VALIDATION / "verify_environment.py")],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**case_definition.quiet_env(), "PYTHONPATH": str(REPO)},
    )
    if guard.returncode:
        (output / 'environment-guard.log').write_text(guard.stdout + guard.stderr, encoding='utf-8')
    assert guard.returncode == 0, guard.stderr

    source_name, replicates = case_definition.CASES[args.case]
    original = REPO / source_name
    source = case_definition.with_workers(case_definition.source_text(args.case), args.workers)
    filter_policy = {'policy': args.policy, 'order': args.order, 'tol': args.tol}
    if args.switch:
        filter_policy['switch'] = case_definition.parse_switch(args.switch)
        if args.policy != 'paper':
            parser.error('--switch needs --policy paper')
    staged = case_definition.stage(source, {'filter': filter_policy, 'run_id': None if args.no_trace else case_id,
                            'controller': args.controller, 'worker_venv': args.worker_venv}, seed=args.seed)
    target = output / original.name
    target.write_text(staged, encoding='utf-8')
    shutil.copy2(VALIDATION / 'windows_case_runtime.py', output / 'windows_case_runtime.py')
    shutil.copy2(VALIDATION / "step_writer.py", output / "step_writer.py")
    shutil.copytree(original.parent / 'data', output / 'data')
    if (original.parent / 'config.json').exists():
        shutil.copy2(original.parent / 'config.json', output / 'config.json')

    sync_output = args.sync_output or args.controller is not None  # a controller's records are written synchronously
    git = lambda *a: subprocess.run(
        ["git", *a], cwd=REPO, capture_output=True, encoding="utf-8", errors="replace", env=case_definition.quiet_env()
    ).stdout.strip()
    receipt = {
        "schema_version": 2,
        "case_id": case_id,
        "scene": cid.scene_code(args.case),
        "parent": args.parent,
        "started": started.isoformat(timespec="seconds"),
        "case": args.case,
        "run_id": case_id,
        "source": source_name,
        "source_sha256": case_definition.digest(original),
        **(
            {"source_edits": [list(e) for e in case_definition.SOURCE_EDITS[args.case]]}
            if args.case in case_definition.SOURCE_EDITS
            else {}
        ),
        "staged_sha256": case_definition.digest(target),
        "filter_policy": filter_policy,
        "controller": args.controller,
        **({"worker_venv": case_definition.worker_venv_identity()} if args.worker_venv else {}),
        "trace": not args.no_trace,
        "worker_pool": "per_batch" if args.fresh_pool else "persistent",
        "step_writer": not sync_output,
        **({"workers": args.workers} if args.workers else {}),
        "seed": "seed_everything() default in source (20)" if args.seed is None else args.seed,
        "python_hash_seed_launcher": 20,
        "placements_limit": args.placements,
        "git_head": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_status": git("status", "--porcelain").splitlines(),
        "git_tracked_diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
        "code_sha256": case_definition.code_fingerprint(),
        "timeout_seconds": args.timeout,
        "status": "RUNNING",
        "command": [sys.executable, "-B", "-u", str(target)],
    }
    receipt['spec_sha256'] = cid.spec_sha256(receipt)
    env = {
        **os.environ,
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(output / "matplotlib-config"),
        "PYTHONPATH": str(REPO),  # the environment's editable install points at one checkout: run this one's code
        "PYTHONHASHSEED": "20",
        "PYTHONDONTWRITEBYTECODE": "1",
        "BASELINE_CASE_EVENTS": str(output / "events.jsonl"),
        "BASELINE_SELECT_FIRST_WALL": "1" if replicates > 1 else "0",
        "BASELINE_BASE_EACH_PLACEMENT": "1" if args.placements else "0",
        "BASELINE_PERSISTENT_POOL": "0" if args.fresh_pool else "1",
        "BASELINE_SYNC_OUTPUT": "1" if sync_output else "0",
    }
    start = time.monotonic()
    termination, process_status = None, 'LAUNCHER_ERROR'
    try:
        bar = StoneBar(output, args.placements, label or args.case)
        with (output / "execution.log").open("w", encoding="utf-8") as log, bar:
            process = subprocess.Popen(receipt['command'], cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT)
            receipt['pid'] = process.pid
            (output / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
            while process.poll() is None:
                bar.refresh()
                limit_hit = args.placements and committed_placements(output) >= args.placements
                if limit_hit or time.monotonic() - start > args.timeout:
                    termination = terminate(process.pid)
                    process_status = 'STOPPED_AFTER_PLACEMENTS' if limit_hit else 'TIMEOUT'
                    break
                time.sleep(2)
            else:
                # A trace write failure raises inside the case and lands here as a non-zero exit.
                process_status = 'PROCESS_PASS' if process.returncode == 0 else 'FAILED'
    finally:
        if 'pid' in receipt and process.poll() is None:  # an exception left the case running (with its paid requests): stop it first
            termination = terminate(process.pid)
        if 'pid' in receipt and not args.no_trace:  # the outputs as the case left them, before any check reads them
            from research.walls import check_run
            receipt['manifest'] = check_run.manifest(output)
            (output / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        trace = verified_trace(output, args, process_status, termination)
        receipt.update(status=final_status(process_status, trace, termination), process_status=process_status,
                       termination=termination, returncode=process.poll() if 'pid' in receipt else None,
                       elapsed_seconds=round(time.monotonic() - start, 3),
                       committed_placements=committed_placements(output), trace_check=trace)
        if not args.no_trace and trace['ok']:
            try:
                receipt['derived'] = {'steps.csv': write_steps(output)}
            except Exception as error:  # a derived table never changes the wall's status
                receipt['derived'] = {'steps.csv': f'not written: {error!r}'}
        keep_data_only(output)
        if receipt['status'] != process_status or process_status == 'FAILED':
            receipt['execution_log_tail'] = (output / 'execution.log').read_text(
                encoding='utf-8', errors='replace').splitlines()[-25:]
        (output / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('command', 'code_sha256', 'manifest')}, indent=2))
    if receipt['status'] == 'TERMINATION_FAILED':
        raise TerminationFailed(receipt)
    return receipt


def main(argv=None):
    return 0 if run(argv)['status'] in case_definition.SUCCESS else 1


if __name__ == '__main__':
    raise SystemExit(main())
