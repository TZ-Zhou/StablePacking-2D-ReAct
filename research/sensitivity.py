"""Offline 120-order sensitivity test on traced candidate score matrices.

Usage: python -B research/sensitivity.py <trace dir> <new output dir> [--tol 1e-9]

Reads only trace/*.npz written during construction (no convolution, placement or kinematics
is re-run) and replays every permutation of the five filters with the paper-style cascade on
the identical matrix. Writes orders.csv, events.csv, candidates.csv and summary.json.
Poses (pose_id, rotation) are in the wall frame (selection.to_world); traces written before
x_world existed fall back to evaluation-frame values suffixed with the side.
final_ties counts candidates sharing the maximum score of the final argmax (>1: index decides).
placement_id = footprint:x_world,y identifies the pixels a pose occupies: different pose ids
(e.g. a rectangle at 0 and 180 degrees, or identical bricks) can be one geometric placement.
Traces without footprints fall back to placement_id = pose_id.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from Stonepacker2D.stack.selection import ALL_ORDERS, PAPER_ORDER, load_trace, select, tie_count, tied_candidates


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

    xw, rw = arrays.get('x_world', arrays['x']), arrays.get('rotation_world', arrays['rotation'])
    suffix = '' if 'x_world' in arrays else f':{side}'

    def pose(i):
        return f"{arrays['stone_id'][i]}@{rw[i]}:{xw[i]},{arrays['y'][i]}{suffix}"

    def placement(i):
        if 'footprint' not in arrays:
            return pose(i)
        return f"{arrays['footprint'][i]}:{xw[i]},{arrays['y'][i]}"

    def tied_placements(sel, policy, order, tol):
        return len({placement(i) for i in tied_candidates(raw, weights, sel, policy, order, tol)})

    live = meta['policy']
    live_sel = select(raw, weights, live['policy'], live['order'], live['tol'])
    live_replay = live_sel.index
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
            'placement_id': '' if i is None else placement(i),
            'stone_id': '' if i is None else int(arrays['stone_id'][i]),
            'rotation': '' if i is None else int(rw[i]),
            'branch': sel.branch, 'final_ties': tie_count(raw, weights, sel, 'paper', order, tol),
            'final_tie_placements': tied_placements(sel, 'paper', order, tol),
            'same_as_paper': bool(i == paper.index),
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
        'n_distinct_placement': len({placement(i) for i in votes}),
        'geometry_available': 'footprint' in arrays,
        'n_distinct_pose_tol0': len(by_tol0 - {None}),
        'orders_without_selection': sum(sel.index is None for sel in sels),
        'paper_pose': '' if paper.index is None else pose(paper.index),
        'orders_same_as_paper': sum(r['same_as_paper'] for r in rows),
        'code_original_pose': '' if code_original.index is None else pose(code_original.index),
        'code_original_same_as_paper': bool(code_original.index == paper.index),
        **{f'distinct_sets_after_filter_{d}': n for d, n in enumerate(depth_sets, start=1)},
        'live_policy': live['policy'], 'live_matches_replay': live_ok,
        'live_final_ties': tie_count(raw, weights, live_sel, live['policy'], live['order'], live['tol']),
        'paper_final_ties': tie_count(raw, weights, paper, 'paper', PAPER_ORDER, tol),
        'orders_with_final_tie': sum(r['final_ties'] > 1 for r in rows),
        'live_final_tie_placements': tied_placements(live_sel, live['policy'], live['order'], live['tol']),
        'paper_final_tie_placements': tied_placements(paper, 'paper', PAPER_ORDER, tol),
        'orders_with_geometric_final_tie': sum(r['final_tie_placements'] > 1 for r in rows),
    }
    candidate_rows = [{'run_id': run_id, 'event': event, 'candidate_id': f'{run_id}/{event}/{i}',
                       'pose_id': pose(i), 'stone_id': int(arrays['stone_id'][i]),
                       'rotation': int(rw[i]), 'votes': n}
                      for i, n in votes.most_common()]
    decision = {'picks': [sel.index for sel in sels], 'live': live_replay,
                'placement': {i: placement(i) for i in set(votes) | {live_replay} - {None}},
                'gds': raw[:, 1] * weights[1], 'stone_id': arrays['stone_id'],
                'cluster': arrays.get('cluster')}
    return rows, event_row, candidate_rows, decision


def analyze_commits(commits, decisions):
    """Step level: per sampling attempt, the wall action each order would take.

    Combines the left/right picks of that order with build_wall's side rule (right only if
    its weighted GDS is strictly greater). Conditional on the stones actually sampled: a
    different probe choice would change later samples, which a replay cannot know.
    """
    rows = []
    for c in commits:
        sides = {s: v['event'] for s, v in c['sides'].items() if v['event'] in decisions}
        if not sides:
            continue

        def action(pick_of):
            picks = {s: pick_of(decisions[e]) for s, e in sides.items()}
            gds = {s: -1.0 if i is None else float(decisions[sides[s]]['gds'][i]) for s, i in picks.items()}
            side = 'right' if 'right' in gds and gds['right'] > gds.get('left', -np.inf) else min(gds)
            i, d = picks[side], decisions[sides[side]]
            if i is None:
                return None
            cluster = None if d['cluster'] is None else int(d['cluster'][i])
            return side, d['placement'][i], int(d['stone_id'][i]), cluster

        live = action(lambda d: d['live'])
        actions = [action(lambda d, k=k: d['picks'][k]) for k in range(len(ALL_ORDERS))]
        real = [a for a in actions if a is not None]
        recorded = (c.get('chosen_side'), c.get('stone_id'))
        rows.append({
            'step': c['step'], 'phase': c['phase'], 'outcome': c['outcome'], 'events': ';'.join(map(str, sides.values())),
            'live_reproduced': live is not None and (live[0], live[2]) == recorded or (live is None and c['outcome'] == 'no_placement'),
            'n_distinct_placement': len({a[1] for a in real}), 'n_distinct_stone': len({a[2] for a in real}),
            'n_distinct_cluster': len({a[3] for a in real}), 'n_distinct_side': len({a[0] for a in real}),
            'orders_without_action': len(actions) - len(real),
            'orders_same_placement_as_live': sum(live is not None and a is not None and a[1] == live[1] for a in actions),
        })
    return rows


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
        'events_with_more_than_one_placement': sum(e['n_distinct_placement'] > 1 for e in events),
        'distinct_placement_histogram': dict(sorted(Counter(e['n_distinct_placement'] for e in events).items())),
        'distinct_stone_histogram': dict(sorted(Counter(e['n_distinct_stone'] for e in events).items())),
        'geometry_available': all(e['geometry_available'] for e in events),
        'mean_distinct_pose': float(np.mean([e['n_distinct_pose'] for e in events])) if n else None,
        'mean_orders_same_as_paper': float(np.mean([e['orders_same_as_paper'] for e in events])) if n else None,
        'first_divergence_stage_histogram': dict(sorted(divergence.items())),
        'events_where_tol0_changes_distinct_count': sum(e['n_distinct_pose'] != e['n_distinct_pose_tol0'] for e in events),
        'events_where_code_original_differs_from_paper': sum(not e['code_original_same_as_paper'] for e in events),
        'live_replay_consistent': all(e['live_matches_replay'] for e in events),
        'events_with_live_final_tie': sum(e['live_final_ties'] > 1 for e in events),
        'events_with_paper_final_tie': sum(e['paper_final_ties'] > 1 for e in events),
        'events_with_any_order_final_tie': sum(e['orders_with_final_tie'] > 0 for e in events),
        'events_with_live_geometric_final_tie': sum(e['live_final_tie_placements'] > 1 for e in events),
        'events_with_paper_geometric_final_tie': sum(e['paper_final_tie_placements'] > 1 for e in events),
        'per_step': {str(step): {'events': len(es),
                                 'max_distinct_pose': max(e['n_distinct_pose'] for e in es),
                                 'max_distinct_placement': max(e['n_distinct_placement'] for e in es),
                                 'max_distinct_stone': max(e['n_distinct_stone'] for e in es),
                                 'min_orders_same_as_paper': min(e['orders_same_as_paper'] for e in es)}
                     for step, es in sorted(per_step.items())},
    }


def summarize_steps(steps):
    hist = lambda key, rows: dict(sorted(Counter(r[key] for r in rows).items()))
    by_phase = defaultdict(list)
    for r in steps:
        by_phase[r['outcome']].append(r)
    return {'attempts': len(steps), 'live_reproduced': all(r['live_reproduced'] for r in steps),
            **{outcome: {'attempts': len(rs),
                         'distinct_placement_histogram': hist('n_distinct_placement', rs),
                         'distinct_stone_histogram': hist('n_distinct_stone', rs),
                         'distinct_cluster_histogram': hist('n_distinct_cluster', rs),
                         'attempts_with_more_than_one_placement': sum(r['n_distinct_placement'] > 1 for r in rs),
                         'attempts_with_more_than_one_stone': sum(r['n_distinct_stone'] > 1 for r in rs),
                         'mean_orders_same_placement_as_live': float(np.mean([r['orders_same_placement_as_live'] for r in rs]))}
               for outcome, rs in by_phase.items()}}


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
    orders, events, candidates, decisions = [], [], [], {}
    for path in files:
        meta, arrays = load_trace(path)
        o, e, c, decisions[meta['event']] = analyze_event(meta, arrays, args.tol)
        orders += o
        events.append(e)
        candidates += c
    write_csv(args.output_dir / 'orders.csv', orders)
    write_csv(args.output_dir / 'events.csv', events)
    write_csv(args.output_dir / 'candidates.csv', candidates, CANDIDATE_FIELDS)
    summary = {'trace_dir': str(args.trace_dir), 'tol': args.tol, **summarize(events, orders)}
    commits_file = args.trace_dir / 'commits.jsonl'
    if commits_file.exists():
        steps = analyze_commits([json.loads(l) for l in commits_file.read_text(encoding='utf-8').splitlines()], decisions)
        if steps:
            write_csv(args.output_dir / 'steps.csv', steps)
        summary['steps'] = summarize_steps(steps)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    print(json.dumps(main(), indent=2))
