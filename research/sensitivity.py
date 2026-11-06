"""Offline 120-order sensitivity test on traced candidate score matrices.

Usage: python -B research/sensitivity.py <trace dir> <new output dir> [--tol 1e-9]

Reads only trace/*.npz written during construction (no convolution, placement or kinematics
is re-run) and replays every permutation of the five filters with the paper-style cascade on
the identical matrix. Writes orders.csv, events.csv, candidates.csv and summary.json.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from Stonepacker2D.stack.selection import ALL_ORDERS, PAPER_ORDER, load_trace, select


def order_name(order):
    return '-'.join(map(str, order))


def first_divergence(stages, reference):
    """First filter stage (1..5) whose survivor set differs from the reference order's."""
    for k in range(1, 6):
        mine = stages[k].tolist() if k < len(stages) else None
        ref = reference[k].tolist() if k < len(reference) else None
        if mine != ref:
            return k
    return None


def analyze_event(meta, arrays, tol):
    raw = arrays['features_raw']
    raw.setflags(write=False)  # every replay must see the identical, unmodified matrix
    weights = arrays['weights']
    run_id, event, side = meta['run_id'], meta['event'], meta['side']

    def pose(i):
        return f"{arrays['stone_id'][i]}@{arrays['rotation'][i]}:{arrays['x'][i]},{arrays['y'][i]}:{side}"

    live = meta['policy']
    live_replay = select(raw, weights, live['policy'], live['order'], live['tol']).index
    live_selected = int(arrays['selected'])
    live_ok = (live_replay is None and live_selected == -1) or live_replay == live_selected

    paper = select(raw, weights, 'paper', PAPER_ORDER, tol)
    code_original = select(raw, weights, 'code_original')
    # Selections are row indices; None (no feasible candidate) is kept as None throughout and
    # written as '' only in the CSV cells.
    rows, sels, by_tol0 = [], [], set()
    for order in ALL_ORDERS:
        sel = select(raw, weights, 'paper', order, tol)
        sels.append(sel)
        by_tol0.add(select(raw, weights, 'paper', order, 0.0).index)
        i = sel.index
        rows.append({
            'run_id': run_id, 'event': event, 'step': meta['step'], 'side': side, 'phase': meta['phase'],
            'order': order_name(order), 'is_paper_order': order == PAPER_ORDER,
            'selected_index': '' if i is None else int(i),
            'candidate_id': '' if i is None else f'{run_id}/{event}/{i}',
            'pose_id': '' if i is None else pose(i),
            'stone_id': '' if i is None else int(arrays['stone_id'][i]),
            'rotation': '' if i is None else int(arrays['rotation'][i]),
            'branch': sel.branch, 'same_as_paper': bool(i == paper.index),
            'first_divergence_stage': first_divergence(sel.stages, paper.stages) or '',
        })
    votes = Counter(sel.index for sel in sels if sel.index is not None)
    depth_sets = []
    for depth in range(1, 6):
        sets = {tuple(sel.stages[depth]) if depth < len(sel.stages) else ('fallback', sel.branch)
                for sel in sels}
        depth_sets.append(len(sets))
    event_row = {
        'run_id': run_id, 'event': event, 'step': meta['step'], 'side': side, 'phase': meta['phase'],
        'n_poses': raw.shape[0], 'n_feasible': len(paper.stages[0]) if paper.stages else 0,
        'n_distinct_pose': len(votes), 'n_distinct_stone': len({int(arrays['stone_id'][i]) for i in votes}),
        'n_distinct_pose_tol0': len(by_tol0 - {None}),
        'orders_without_selection': sum(sel.index is None for sel in sels),
        'paper_pose': '' if paper.index is None else pose(paper.index),
        'orders_same_as_paper': sum(r['same_as_paper'] for r in rows),
        'code_original_pose': '' if code_original.index is None else pose(code_original.index),
        'code_original_same_as_paper': bool(code_original.index == paper.index),
        **{f'distinct_sets_after_filter_{d}': n for d, n in enumerate(depth_sets, start=1)},
        'live_policy': live['policy'], 'live_matches_replay': live_ok,
    }
    candidate_rows = [{'run_id': run_id, 'event': event, 'candidate_id': f'{run_id}/{event}/{i}',
                       'pose_id': pose(i), 'stone_id': int(arrays['stone_id'][i]),
                       'rotation': int(arrays['rotation'][i]), 'votes': n}
                      for i, n in votes.most_common()]
    return rows, event_row, candidate_rows


def summarize(events, orders):
    n = len(events)
    per_step = defaultdict(list)
    for e in events:
        per_step[e['step']].append(e)
    divergence = Counter(r['first_divergence_stage'] for r in orders if r['first_divergence_stage'] != '')
    return {
        'events': n,
        'events_with_more_than_one_pose': sum(e['n_distinct_pose'] > 1 for e in events),
        'events_with_more_than_one_stone': sum(e['n_distinct_stone'] > 1 for e in events),
        'distinct_pose_histogram': dict(sorted(Counter(e['n_distinct_pose'] for e in events).items())),
        'mean_distinct_pose': float(np.mean([e['n_distinct_pose'] for e in events])) if n else None,
        'mean_orders_same_as_paper': float(np.mean([e['orders_same_as_paper'] for e in events])) if n else None,
        'first_divergence_stage_histogram': dict(sorted(divergence.items())),
        'events_where_tol0_changes_distinct_count': sum(e['n_distinct_pose'] != e['n_distinct_pose_tol0'] for e in events),
        'events_where_code_original_differs_from_paper': sum(not e['code_original_same_as_paper'] for e in events),
        'live_replay_consistent': all(e['live_matches_replay'] for e in events),
        'per_step': {str(step): {'events': len(es),
                                 'max_distinct_pose': max(e['n_distinct_pose'] for e in es),
                                 'min_orders_same_as_paper': min(e['orders_same_as_paper'] for e in es)}
                     for step, es in sorted(per_step.items())},
    }


CANDIDATE_FIELDS = ['run_id', 'event', 'candidate_id', 'pose_id', 'stone_id', 'rotation', 'votes']


def write_csv(path, rows, fields=None):
    """fields is required for tables that can be empty (candidates of all-infeasible events)."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('trace_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--tol', type=float, default=1e-9)
    args = parser.parse_args(argv)
    files = sorted(args.trace_dir.glob('e*.npz'))
    files = [f for f in files if not f.name.endswith('.tmp.npz')]
    if not files:
        raise SystemExit(f'no trace files in {args.trace_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier analysis
    orders, events, candidates = [], [], []
    for path in files:
        o, e, c = analyze_event(*load_trace(path), args.tol)
        orders += o
        events.append(e)
        candidates += c
    write_csv(args.output_dir / 'orders.csv', orders)
    write_csv(args.output_dir / 'events.csv', events)
    write_csv(args.output_dir / 'candidates.csv', candidates, CANDIDATE_FIELDS)
    summary = {'trace_dir': str(args.trace_dir), 'tol': args.tol, **summarize(events, orders)}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    print(json.dumps(main(), indent=2))
