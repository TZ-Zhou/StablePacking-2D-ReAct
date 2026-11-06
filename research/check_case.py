"""Read-only consistency checks of traced runs, and byte comparison of two runs.

Usage: python -B research/check_case.py <run dir> [--compare <other run dir>]

check_run: trace replay, commits vs chosen candidates, placed rocks in base.pkl vs recorded
world poses (unstabilized rocks must match pixel for pixel), right-side rotations against the
same stone's left-side candidates. compare_runs: identical outputs, base.pkl and trace.
differing_files: every file of the result directories byte for byte (time.txt holds timings and
the PDFs embed creation dates, so those are expected to differ between identical runs).
"""
import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Stonepacker2D.stack.selection import footprint, load_trace, select  # noqa: E402


def result_dirs(run):
    return sorted({p.parent for p in Path(run).rglob('decisions.txt')})


def load_result(result):
    trace = Path(result) / 'trace'
    events = {}
    for p in sorted(trace.glob('e*.npz')):
        if not p.name.endswith('.tmp.npz'):
            meta, arrays = load_trace(p)
            events[meta['event']] = (meta, arrays)
    lines = (trace / 'commits.jsonl').read_text(encoding='utf-8').splitlines()
    return events, [json.loads(line) for line in lines]


def check_result(result):
    problems, stats = [], Counter()
    events, commits = load_result(result)
    for meta, a in events.values():
        p = meta['policy']
        s = select(a['features_raw'], a['weights'], p['policy'], p['order'], p['tol'])
        if (-1 if s.index is None else s.index) != int(a['selected']):
            problems.append(f"event {meta['event']}: replay differs from live selection")
    base = pickle.load(open(Path(result) / 'base.pkl', 'rb'))
    rocks = {r.id: r for r in base.placed_rocks}
    placed = [c for c in commits if c['outcome'] == 'placed']
    for c in placed:
        meta, a = events[c['sides'][c['chosen_side']]['event']]
        i = c['sides'][c['chosen_side']]['index']
        pose = (int(a['stone_id'][i]), int(a['x_world'][i]), int(a['y'][i]), int(a['rotation_world'][i]))
        if pose != (c['stone_id'], c['x_world'], c['y'], c['rotation_world']):
            problems.append(f"step {c['step']}: commit pose {c} differs from candidate {pose}")
        rock = rocks.get(c['stone_id'])
        if rock is None:
            problems.append(f"step {c['step']}: stone {c['stone_id']} not in base.pkl")
            continue
        rows, cols = np.nonzero(rock.matrix)
        same_origin = [int(cols.min()), int(rows.min())] == [c['x_world'], c['y']]
        same_pixels = footprint(rock.matrix, 'left') == a['footprint'][i]
        kind = 'stabilized' if c.get('placed_rot_angle') else 'unrotated'
        stats[f"{c['chosen_side']}/{kind}/{'same' if same_origin and same_pixels else 'changed'}"] += 1
        if kind == 'unrotated' and not (same_origin and same_pixels):
            problems.append(f"step {c['step']}: unstabilized rock does not match its recorded pose")
    # right-side candidates against the same stone's left-side candidates of the same attempt
    for c in commits:
        if {'left', 'right'} <= set(c['sides']) and c['sides']['right']['event'] in events:
            left = events[c['sides']['left']['event']][1]
            right = events[c['sides']['right']['event']][1]
            shapes = {(int(s), int(r)): f for s, r, f in zip(left['stone_id'], left['rotation_world'], left['footprint'])}
            for s, r, f in zip(right['stone_id'], right['rotation_world'], right['footprint']):
                stats['right candidates checked'] += 1
                if shapes.get((int(s), int(r)), f) != f:
                    problems.append(f"attempt step {c['step']}: right candidate {s}@{r} differs from left")
    stats['events'], stats['attempts'], stats['placed'] = len(events), len(commits), len(placed)
    return {'result': str(result), 'problems': problems, 'stats': dict(stats)}


def check_run(run):
    receipt = json.loads((Path(run) / 'receipt.json').read_text(encoding='utf-8'))
    results = [check_result(r) for r in result_dirs(run)]
    problems = list(receipt['trace_check'].get('problems', [])) + [p for r in results for p in r['problems']]
    if not results:
        problems.append('no result directory')
    return {'status': receipt['status'], 'ok': not problems, 'problems': problems, 'results': results}


def compare_runs(a, b):
    """Differences between two runs that should be identical (same case, policy, order, seed)."""
    ra, rb = result_dirs(a), result_dirs(b)
    if len(ra) != len(rb):
        return [f'{len(ra)} vs {len(rb)} result directories']
    diffs = []
    for x, y in zip(ra, rb):
        for name in ['decisions.txt', 'filter.txt', 'transformation.txt', 'trace/commits.jsonl']:
            if (x / name).exists() != (y / name).exists():
                diffs.append(f'{name}: present in only one run')
            elif (x / name).exists() and (x / name).read_bytes() != (y / name).read_bytes():
                diffs.append(f'{name} differs')
        bx, by = pickle.load(open(x / 'base.pkl', 'rb')), pickle.load(open(y / 'base.pkl', 'rb'))
        for attr in ['matrix', 'id_matrix', 'cluster_matrix']:
            if not np.array_equal(getattr(bx, attr), getattr(by, attr)):
                diffs.append(f'base.pkl {attr} differs')
        if (x / 'trace').exists() and (y / 'trace').exists():
            ex, ey = load_result(x)[0], load_result(y)[0]
            if sorted(ex) != sorted(ey):
                diffs.append(f'trace events {len(ex)} vs {len(ey)}')
            for e in sorted(set(ex) & set(ey)):
                (mx, ax), (my, ay) = ex[e], ey[e]
                if {k: v for k, v in mx.items() if k != 'run_id'} != {k: v for k, v in my.items() if k != 'run_id'}:
                    diffs.append(f'event {e}: meta differs')
                if sorted(ax) != sorted(ay) or any(not np.array_equal(ax[k], ay[k], equal_nan=ax[k].dtype.kind == 'f')
                                                   for k in ax):
                    diffs.append(f'event {e}: arrays differ')
    return diffs


def differing_files(a, b):
    """Paths relative to the result directory present in only one run or with different bytes."""
    ra, rb = result_dirs(a), result_dirs(b)
    if len(ra) != len(rb):
        return [f'{len(ra)} vs {len(rb)} result directories']
    out = []
    for x, y in zip(ra, rb):
        fx = {p.relative_to(x).as_posix(): p for p in x.rglob('*') if p.is_file()}
        fy = {p.relative_to(y).as_posix(): p for p in y.rglob('*') if p.is_file()}
        out += sorted(set(fx) ^ set(fy))
        out += sorted(k for k in set(fx) & set(fy) if fx[k].read_bytes() != fy[k].read_bytes())
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args(argv)
    report = check_run(args.run)
    if args.compare:
        report['differences_to'] = {'run': str(args.compare), 'differences': compare_runs(args.run, args.compare),
                                    'differing_files': differing_files(args.run, args.compare)}
    print(json.dumps(report, indent=2))
    return 0 if report['ok'] and not report.get('differences_to', {}).get('differences') else 1


if __name__ == '__main__':
    raise SystemExit(main())
