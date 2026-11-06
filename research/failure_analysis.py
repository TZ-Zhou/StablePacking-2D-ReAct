"""Why walls end with stones unplaced: per-attempt candidate sets, per-stone filter fate and wall state (offline).

Usage: python -B research/failure_analysis.py <campaign id> <output dir> [--hard 8,9,13]

Reads the recorded trace of every wall in the campaign manifest (campaign.py summarize); no
wall is built. Writes
- attempts.csv: one row per sampling attempt: remaining/sampled stones, candidates, candidates
  with a geometric position (finite distance), feasible candidates (LLC > 0, finite filter
  scores, the selection gate) and the wall state before the decision.
- stones.csv: one row per (attempt, sampled stone): rotations with a position, feasible
  rotations, best LLC, and where the selection dropped it (eliminated_by: 'infeasible', the
  filter that removed its last candidate, 'final_argmax' if it survived every recorded stage,
  'selected').
- walls.csv: per wall: unplaced stones, attempts in which some remaining stone had no feasible
  rotation while all remaining stones were evaluated ('stranded'; the final no_placement
  retries are counted separately), --hard stones still remaining when 4 stones were left,
  feasible shares with 4-2 stones left and the wall height then.
Only the convolution-best position of each rotation is checked for stability (place_one_stone),
so 'infeasible' means that one position is unstable, not that no stable position exists.
Everything here is association; nothing tests what caused a wall to stop.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign  # noqa: E402
import run_case  # noqa: E402

from Stonepacker2D.stack.selection import load_trace  # noqa: E402

FILTERS = {0: 'NHR', 1: 'GDS', 2: 'IWR', 3: 'LLC', 4: 'LFR'}


def wall_state(wall):
    """Wall matrix of a trace event: row 0 is the bottom, rows 0-1 are ground, the top row and
    the two side columns are boundary (255). Heights are in pixels above the ground."""
    occ = wall[2:-1, 1:-1] != 0
    sky = np.where(occ.any(axis=0), occ.shape[0] - np.argmax(occ[::-1], axis=0), 0)
    hmax = int(sky.max())
    under = np.arange(occ.shape[0])[:, None] < sky[None, :]
    return {'h_max': hmax, 'h_min': int(sky.min()), 'h_range': hmax - int(sky.min()), 'h_std': float(sky.std()),
            'fill_under_hmax': float(occ[:hmax].mean()) if hmax else np.nan,
            'void_under_skyline_px': int((~occ & under).sum()),
            'n_skyline_jumps': int((np.abs(np.diff(sky)) > 3).sum())}


def eliminated_by(rows, stages, order, selected):
    """Where the selection dropped one stone's candidate rows (stages as in the trace meta)."""
    if selected in rows:
        return 'selected'
    passed = sum(bool(set(rows) & set(s)) for s in stages)
    if passed == 0:
        return 'infeasible'
    if passed == len(stages):
        return 'final_argmax'
    return FILTERS[order[passed - 1]]


def attempt_rows(result):
    """(attempt rows, stone rows) of one result directory, in commit order."""
    result = Path(result)
    attempts, stones = [], []
    for c in (json.loads(l) for l in (result / 'trace' / 'commits.jsonl').read_text(encoding='utf-8').splitlines()):
        side = c['sides'][c['chosen_side']]
        meta, a = load_trace(result / 'trace' / f"e{side['event']:05d}.npz")
        raw, sid = a['features_raw'], a['stone_id']
        feasible = (raw[:, 3] > 0) & np.isfinite(raw[:, :5]).all(axis=1)
        p = meta['policy']
        order = p['order'] or ((1, 2, 3, 4, 0) if p['policy'] == 'code_original' else (3, 0, 2, 1, 4))
        selected = int(a['selected'])
        key = {'step': c['step'], 'trial': c['sample_trial'], 'event': side['event']}
        attempts.append({**key, 'outcome': c['outcome'], 'chosen': c.get('stone_id'), 'branch': meta['branch'],
                         'n_remaining': len(a['inventory_ids']) + len(a['sampled_ids']), 'n_sampled': len(a['sampled_ids']),
                         'n_candidates': len(sid), 'n_geom': int(np.isfinite(a['distance']).sum()),
                         'n_feasible': int(feasible.sum()), 'n_feasible_stones': len(set(sid[feasible].tolist())),
                         'remaining_ids': ' '.join(map(str, sorted(np.r_[a['inventory_ids'], a['sampled_ids']].astype(int).tolist()))),
                         **wall_state(a['wall'])})
        for s in a['sampled_ids'].tolist():
            rows = np.flatnonzero(sid == s).tolist()
            stones.append({**key, 'stone': s, 'geom_rotations': int(np.isfinite(a['distance'][rows]).sum()),
                           'feasible_rotations': int(feasible[rows].sum()), 'best_llc': float(raw[rows, 3].max()),
                           'eliminated_by': eliminated_by(rows, meta['stages'], order, selected)})
    return attempts, stones


def wall_summary(attempts, stones, hard=()):
    """attempts/stones: DataFrames of one wall."""
    placed = attempts[attempts.outcome == 'placed']
    terminal = attempts[attempts.outcome == 'no_placement']
    evaluated_all = placed[placed.n_sampled == placed.n_remaining]
    stranded = stones[stones.event.isin(evaluated_all.event)].groupby('event').feasible_rotations.min().eq(0)
    last4 = placed[placed.n_remaining == 4]
    late = placed[placed.n_remaining.between(2, 4)]
    remaining_at_end = attempts.remaining_ids.iloc[-1].split() if terminal.size else []
    return {'unplaced': ' '.join(remaining_at_end), 'n_no_placement': len(terminal),
            'stranded_attempts': int(stranded.sum()),
            'hard_left_at_4': len(set(hard) & set(map(int, last4.remaining_ids.iloc[0].split()))) if len(last4) else np.nan,
            'feasible_share_4to2': float((late.n_feasible / late.n_candidates).mean()) if len(late) else np.nan,
            'feasible_stone_share_4to2': float((late.n_feasible_stones / late.n_sampled).mean()) if len(late) else np.nan,
            'h_max_at_4': float(last4.h_max.mean()) if len(last4) else np.nan,
            'h_max_last': int(attempts.h_max.iloc[-1])}


def analyze(manifest, hard=()):
    attempts, stones, walls = [], [], []
    for w in manifest.itertuples():
        result = next((run_case.RUNS / w.run).rglob('trace/commits.jsonl')).parent.parent
        a, s = attempt_rows(result)
        tag = {'arm': w.arm, 'seed': w.seed, 'complete': w.n_placed == w.n_input}
        a, s = pd.DataFrame(a).assign(**tag), pd.DataFrame(s).assign(**tag)
        attempts.append(a)
        stones.append(s)
        walls.append({**tag, 'n_placed': w.n_placed, **wall_summary(a, s, hard)})
    return pd.concat(attempts), pd.concat(stones), pd.DataFrame(walls)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('id')
    parser.add_argument('out', type=Path)
    parser.add_argument('--hard', type=lambda t: tuple(int(x) for x in t.split(',')), default=(),
                        help='stone ids counted in hard_left_at_4')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = pd.read_csv(campaign.CAMPAIGNS / args.id / 'manifest.csv')
    attempts, stones, walls = analyze(manifest[manifest.status == 'PROCESS_PASS'], args.hard)
    attempts.to_csv(args.out / 'attempts.csv', index=False)
    stones.to_csv(args.out / 'stones.csv', index=False)
    walls.to_csv(args.out / 'walls.csv', index=False)
    pd.set_option('display.width', 250)
    print(walls[~walls.complete].to_string(index=False))
    print(walls.groupby('complete')[['stranded_attempts', 'hard_left_at_4', 'feasible_share_4to2',
                                     'feasible_stone_share_4to2', 'h_max_at_4']].mean().round(2).to_string())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
