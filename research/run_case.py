"""Run an original test case with a chosen filter policy and candidate tracing.

Usage: python -B research/run_case.py <case> <run name> [--policy code_original|paper]
                                      [--order 3,0,2,1,4] [--tol 1e-9] [--placements N]
                                      [--seed S] [--switch STEP=a,b,c,d,e ...] [--controller JSON]

Staging follows baseline/validation/run_original_case.py (copied inputs, Windows main guard,
shell cleanup refused, spawned workers inherit constants/RNG via windows_case_runtime.py);
those baseline files stay frozen and are only read. Output goes to a new research/runs/<name>.
With --placements N the case is stopped once decisions.txt has N committed placements.
--switch makes a one-step counterfactual branch: same seed and code, so every decision before
STEP should repeat the run without it; verify that with check_case.py --compare.
--controller '{"kind": "null"}' (research/controllers.py) picks the order at each selection;
verify its records with controllers.py check <run>.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / 'research' / 'runs'
VALIDATION = REPO / 'baseline' / 'validation'
ENV_PREFIX = Path(r'C:\Users\user\miniconda3\envs\stablepacking-baseline')
CASES = {  # case -> (entry script, original number of walls in its batch)
    'bricks': ('Tests/Bricks/run.py', 1),
    'almeida-r': ('Tests/Almeida/R/run.py', 1),
    'almeida-pr': ('Tests/Almeida/PR/run.py', 1),
    'almeida-ir': ('Tests/Almeida/IR/run.py', 1),
    'fft-d2-0': ('Tests/FFT_stones/01_run.py', 5),
    'rubble': ('Tests/Rubble_stone_walls/00_run.py', 20),
    'thangavelu-noise80': ('Tests/Thangavelu/00_run.py', 20),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_output(name):
    """New directory under research/runs; refuses anything else so baseline data is never touched."""
    output = (RUNS / name).resolve()
    if not output.is_relative_to(RUNS.resolve()) or output == RUNS.resolve():
        raise ValueError(f'output must be inside {RUNS}: {output}')
    output.mkdir(parents=True, exist_ok=False)
    return output


def stage(source, policy, seed=None):
    """seed=None keeps the source's seed_everything() call (default 20); an int replaces only its
    argument, so every run initialises its own RNG before reading stones and sampling."""
    lines = source.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if 'os.system(' in line and 'rm -rf' in line and not line.lstrip().startswith('#'):
            # 'rm -rf <dir>/*' becomes: proceed only if <dir>/* matches nothing (Bricks cleans the
            # directory it has just created); never delete anything.
            indent, stripped = line[:len(line) - len(line.lstrip())], line.strip()
            assert stripped.startswith("os.system('rm -rf '+") and stripped.endswith(')'), line
            lines[i] = (indent + "if __import__('glob').glob((" + stripped[len('os.system('):-1] + ")[7:]): "
                        "raise FileExistsError('Refusing to clean a non-empty result directory')\n")
    safe = ''.join(lines)
    boundary = safe.index('seed_everything()\n')
    header = ("\nfrom windows_case_runtime import configure\n"
              "if __name__ == '__main__':\n"
              "    configure(globals())\n"
              "    from Stonepacker2D.utils.constant import set_filter_policy, set_trace\n"
              f"    set_filter_policy(**{policy['filter']!r})\n"
              f"    set_trace({policy['run_id']!r})\n")
    if policy.get('controller') is not None:
        header += (f"    __import__('sys').path.insert(0, {str(REPO / 'research')!r})\n"
                   "    from controllers import make_controller\n"
                   "    from Stonepacker2D.utils.constant import set_filter_controller\n"
                   f"    set_filter_controller(make_controller({policy['controller']!r}))\n")
    body = ''.join('    ' + l if l.strip() else l for l in safe[boundary:].splitlines(keepends=True))
    if seed is not None:
        call = '    seed_everything()\n'
        assert body.startswith(call), body[:40]
        body = f'    seed_everything({int(seed)})\n' + body[len(call):]
    return safe[:boundary] + header + body


def parse_switch(texts):
    switch = {}
    for text in texts:
        step, order = text.split('=')
        order = tuple(int(c) for c in order.split(','))
        if sorted(order) != [0, 1, 2, 3, 4] or int(step) in switch:
            raise ValueError(f'bad --switch {text}')
        switch[int(step)] = order
    return switch


def committed_placements(output):
    return sum(len(p.read_text(encoding='utf-8').splitlines()) for p in output.rglob('decisions.txt'))


def code_fingerprint(root=REPO):
    """SHA256 of every executed source file, tracked or not, so a run is tied to its code."""
    files = sorted(Path(root).glob('Stonepacker2D/**/*.py')) + [
        Path(root) / 'research' / name for name in ('run_case.py', 'controllers.py', 'failure_analysis.py', 'campaign.py')] + [
        Path(root) / 'baseline' / 'validation' / 'windows_case_runtime.py']
    return {f.relative_to(root).as_posix(): digest(f) for f in files if f.exists()}


def process_table():
    """{pid: (parent pid, creation ticks)}; creation time guards against PID reuse."""
    out = subprocess.run(['powershell', '-NoProfile', '-Command',
                          'Get-CimInstance Win32_Process | ForEach-Object '
                          '{ "$($_.ProcessId),$($_.ParentProcessId),$($_.CreationDate.Ticks)" }'],
                         capture_output=True, text=True, timeout=120, check=True).stdout.split()
    return {int(p): (int(pp), c) for p, pp, c in (line.split(',') for line in out)}


def process_tree(table, root):
    found, todo = set(), [root]
    while todo:
        pid = todo.pop()
        found.add(pid)
        todo += [child for child, (parent, _) in table.items() if parent == pid and child not in found]
    return {pid: table[pid][1] for pid in found if pid in table}


def terminate(pid, settle_seconds=30):
    """taskkill the tree, then verify every member (same PID and creation time) is gone."""
    members = process_tree(process_table(), pid)
    try:
        kill = subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], capture_output=True,
                              text=True, errors='replace', timeout=60)
        record = {'taskkill_returncode': kill.returncode, 'stdout': kill.stdout, 'stderr': kill.stderr}
    except subprocess.TimeoutExpired:
        record = {'taskkill_returncode': None, 'error': 'taskkill timed out'}
    deadline = time.monotonic() + settle_seconds
    while True:
        table = process_table()
        alive = sorted(p for p, created in members.items() if table.get(p, (None, None))[1] == created)
        if not alive or time.monotonic() > deadline:
            break
        time.sleep(1)
    record.update(tree=sorted(members), alive_after=alive, verified=not alive)
    return record


def check_commits(output, stopped=False):
    """Placed attempts in trace/commits.jsonl must match decisions.txt line for line and point
    at existing filter events. A run stopped by the launcher may lag decisions.txt by one line."""
    problems = []
    for decisions in output.rglob('decisions.txt'):
        commits_file = decisions.parent / 'trace' / 'commits.jsonl'
        if not commits_file.exists():
            problems.append(f'{commits_file.parent}: no commits.jsonl')
            continue
        commits = [json.loads(line) for line in commits_file.read_text(encoding='utf-8').splitlines()]
        rows = [line.split(';') for line in decisions.read_text(encoding='utf-8').splitlines()]
        placed = [[str(c[k]) for k in ('step', 'stone_id', 'x_world', 'y', 'rotation_label')]
                  for c in commits if c['outcome'] == 'placed']
        if placed != rows[:len(placed)] or not 0 <= len(rows) - len(placed) <= int(stopped):
            problems.append(f'{decisions}: {len(placed)} placed commits do not match {len(rows)} decisions')
        events = {int(p.stem[1:]) for p in (decisions.parent / 'trace').glob('e*.npz')
                  if not p.name.endswith('.tmp.npz')}
        missing = {side['event'] for c in commits for side in c['sides'].values()} - events
        if missing:
            problems.append(f'{commits_file}: references missing events {sorted(missing)[:5]}')
    return problems


def check_trace(output, stopped=False):
    """Trace files must exist, be numbered without gaps and load; otherwise the data is incomplete."""
    from Stonepacker2D.stack.selection import load_trace
    files = sorted(output.rglob('trace/e*.npz'))
    files = [f for f in files if not f.name.endswith('.tmp.npz')]
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
    problems += check_commits(output, stopped)
    return {'files': len(files), 'ok': not problems, 'problems': problems}


def check_controller(output, trace):
    """A controller run is complete only if every result's controller records equal the receipt's
    controller re-executed on the trace (controllers.check)."""
    import check_case
    import controllers
    try:
        reports = [controllers.check(r) for r in check_case.result_dirs(output)]
        problems = [p for r in reports for p in r['problems'][:5]] + ([] if reports else ['no result directory'])
    except Exception as error:  # an unreadable record or trace leaves the run unverified
        reports, problems = [], [f'controller check failed: {error!r}']
    return {**trace, 'ok': trace['ok'] and not problems, 'problems': trace['problems'] + problems,
            'controller': [{k: r[k] for k in ('ok', 'records', 'decision_points', 'switches')} for r in reports]}


def final_status(process_status, trace, termination):
    if termination is not None and not termination['verified']:
        return 'TERMINATION_FAILED'
    if process_status in ('PROCESS_PASS', 'STOPPED_AFTER_PLACEMENTS') and not trace['ok']:
        return 'INCOMPLETE_TRACE'
    return process_status


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('case', choices=CASES)
    parser.add_argument('name')
    parser.add_argument('--policy', default='code_original', choices=['code_original', 'paper'])
    parser.add_argument('--order', type=lambda s: tuple(int(c) for c in s.split(',')))
    parser.add_argument('--tol', type=float, default=1e-9)
    parser.add_argument('--placements', type=int, default=0, help='stop after N placements (0 = run to the end)')
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--no-trace', action='store_true', help='control run: logging off, outputs only')
    parser.add_argument('--seed', type=int, help='seed_everything(seed) instead of the source default (20)')
    parser.add_argument('--switch', action='append', default=[], metavar='STEP=a,b,c,d,e',
                        help='paper policy only: use this order at construction step STEP (every sampling trial of that step), the --order elsewhere; repeatable')
    parser.add_argument('--controller', type=json.loads, metavar='JSON',
                        help='paper policy only: research/controllers.py spec, e.g. {"kind": "random", "seed": 7}')
    args = parser.parse_args(argv)
    if args.controller is not None:
        if args.policy != 'paper' or args.switch or args.no_trace:
            parser.error('--controller needs --policy paper, tracing and no --switch')
        import controllers
        try:
            controllers.make_controller(args.controller, sink=lambda record: None)
        except (ValueError, TypeError, NotImplementedError) as error:
            parser.error(f'--controller: {error}')
    assert Path(sys.prefix).resolve() == ENV_PREFIX, sys.prefix

    output = prepare_output(args.name)
    guard = subprocess.run([sys.executable, '-B', str(VALIDATION / 'verify_environment.py')],
                           cwd=REPO, capture_output=True, text=True)
    (output / 'environment-guard.log').write_text(guard.stdout + guard.stderr, encoding='utf-8')
    assert guard.returncode == 0, guard.stderr

    source_name, replicates = CASES[args.case]
    original = REPO / source_name
    filter_policy = {'policy': args.policy, 'order': args.order, 'tol': args.tol}
    if args.switch:
        filter_policy['switch'] = parse_switch(args.switch)
        if args.policy != 'paper':
            parser.error('--switch needs --policy paper')
    staged = stage(original.read_text(encoding='utf-8'), {'filter': filter_policy, 'run_id': None if args.no_trace else args.name,
                                                          'controller': args.controller}, seed=args.seed)
    target = output / original.name
    target.write_text(staged, encoding='utf-8')
    shutil.copy2(VALIDATION / 'windows_case_runtime.py', output / 'windows_case_runtime.py')
    shutil.copytree(original.parent / 'data', output / 'data')
    if (original.parent / 'config.json').exists():
        shutil.copy2(original.parent / 'config.json', output / 'config.json')

    git = lambda *a: subprocess.run(['git', *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    receipt = {
        'case': args.case, 'run_id': args.name, 'source': source_name, 'source_sha256': digest(original),
        'staged_sha256': digest(target), 'filter_policy': filter_policy, 'controller': args.controller,
        'trace': not args.no_trace,
        'seed': 'seed_everything() default in source (20)' if args.seed is None else args.seed,
        'python_hash_seed_launcher': 20, 'placements_limit': args.placements,
        'git_head': git('rev-parse', 'HEAD'), 'git_branch': git('branch', '--show-current'),
        'git_status': git('status', '--porcelain').splitlines(),
        'git_tracked_diff_sha256': hashlib.sha256(git('diff', 'HEAD').encode()).hexdigest(),
        'code_sha256': code_fingerprint(),
        'timeout_seconds': args.timeout, 'status': 'RUNNING',
        'command': [sys.executable, '-B', '-u', str(target)],
    }
    env = {**os.environ, 'MPLBACKEND': 'Agg', 'MPLCONFIGDIR': str(output / 'matplotlib-config'),
           'PYTHONHASHSEED': '20', 'PYTHONDONTWRITEBYTECODE': '1',
           'BASELINE_CASE_EVENTS': str(output / 'events.jsonl'),
           'BASELINE_SELECT_FIRST_WALL': '1' if replicates > 1 else '0'}
    start = time.monotonic()
    termination, process_status = None, 'LAUNCHER_ERROR'
    try:
        with (output / 'execution.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(receipt['command'], cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT)
            receipt['pid'] = process.pid
            (output / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
            while process.poll() is None:
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
        trace = ({'ok': True, 'skipped': 'no-trace control run'} if args.no_trace
                 else check_trace(output, stopped=termination is not None))
        if args.controller is not None and process_status == 'PROCESS_PASS' and trace['ok']:
            trace = check_controller(output, trace)
        receipt.update(status=final_status(process_status, trace, termination), process_status=process_status,
                       termination=termination, returncode=process.poll() if 'pid' in receipt else None,
                       elapsed_seconds=round(time.monotonic() - start, 3),
                       committed_placements=committed_placements(output), trace_check=trace)
        if receipt['status'] != process_status or process_status == 'FAILED':
            receipt['execution_log_tail'] = (output / 'execution.log').read_text(
                encoding='utf-8', errors='replace').splitlines()[-25:]
        (output / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('command', 'code_sha256')}, indent=2))
    if receipt['status'] == 'TERMINATION_FAILED':
        raise RuntimeError('case process tree not verified stopped; do not launch another case')
    return 0 if receipt['status'] in ('PROCESS_PASS', 'STOPPED_AFTER_PLACEMENTS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
