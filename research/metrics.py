"""Wall-quality and process metrics of one run, and step rows of its decision trajectory.

Three different fill quantities, kept apart on purpose (H x W = base.matrix, world size):
- fill_wang_code: np.count_nonzero(base.matrix) / (W * H), the ratio build_wall uses to stop at
  0.999. It counts ground, boundary columns/row (ignore pixels) and wedges.
- stone_filling: rock pixels / (H*W - 2H - 3W): Eq. 25 of Wang et al. with the target-area
  formula of Tests/Thangavelu/03_evaluate_wall_metrics.py, but on the simulation grid (the
  author's script re-renders walls at the original resolution first).
- wall_density: rock pixels / ((W - 2) * built height): density of the part actually built.
Stability: lambda_min = min(left, right) unrounded load multipliers of the final tilting-table
evaluation (Eq. 28), read from the img/{left,right}<value>_mechanism.png names as the author's
script does; lambda_wang_code is the value evaluate_kine returned (rounded by the code).
Rock pixels exclude the ground and wedges; wedges are reported separately.
"""
import json
import pickle
import re
from pathlib import Path

import numpy as np

from Stonepacker2D.stack.selection import load_trace, select, tie_count
from Stonepacker2D.utils.constant import get_base_id, get_wedge_id

METRICS_VERSION = 1


def mechanism_lambdas(result):
    found = {}
    for side in ('left', 'right'):
        names = [p.name for p in (Path(result) / 'img').glob(f'{side}*_mechanism.png')]
        values = [float(re.match(rf'{side}(.+)_mechanism\.png$', n).group(1)) for n in names]
        found[side] = values[0] if len(values) == 1 else None
    return found


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def wall_metrics(run, result):
    run, result = Path(run), Path(result)
    receipt = json.loads((run / 'receipt.json').read_text(encoding='utf-8'))
    events = read_jsonl(run / 'events.jsonl')
    build = next((e for e in events if e['event'] == 'build_enter'), {})
    final = next((e for e in events if e['event'] == 'final_evaluation_complete'), {})
    base = pickle.load(open(result / 'base.pkl', 'rb'))
    H, W = base.matrix.shape
    rocks = np.zeros((H, W), bool)
    for rock in base.placed_rocks:
        rocks |= rock.matrix != 0
    wedges = np.zeros((H, W), bool)
    for brick in base.placed_bricks:
        if brick.id == get_wedge_id():
            wedges |= brick.matrix != 0
    ground = base.placed_bricks[0]
    assert ground.id == get_base_id(), 'first placed brick must be the ground'
    ground_top = int(np.nonzero(ground.matrix)[0].max()) + 1
    top = int(np.nonzero(rocks)[0].max()) if rocks.any() else ground_top - 1
    height = top - ground_top + 1
    rock_px = int(rocks.sum())
    lam = mechanism_lambdas(result)
    commits = read_jsonl(result / 'trace' / 'commits.jsonl')
    placed = [c for c in commits if c['outcome'] == 'placed']
    return {
        'status': receipt['status'], 'elapsed_seconds': receipt.get('elapsed_seconds'),
        'n_input': build.get('input_stones'), 'n_placed': len(base.placed_rocks),
        'placed_ratio': len(base.placed_rocks) / build['input_stones'] if build.get('input_stones') else None,
        'fill_wang_code': float(np.count_nonzero(base.matrix) / (base._w * base._h)),
        'stone_filling': rock_px / (H * W - 2 * H - 3 * W),
        'wall_density': rock_px / ((W - 2) * height) if height > 0 else None,
        'wall_height': height, 'height_ratio': height / (H - 3),
        'rock_pixels': rock_px, 'wedge_pixels': int(wedges.sum()),
        'n_wedges': sum(b.id == get_wedge_id() for b in base.placed_bricks),
        'lambda_left': lam['left'], 'lambda_right': lam['right'],
        'lambda_min': min(lam['left'], lam['right']) if None not in lam.values() else None,
        'lambda_wang_code': final.get('limit_force'),
        'attempts': len(commits), 'n_probe': sum(c['outcome'] == 'probe' for c in commits),
        'n_no_placement': sum(c['outcome'] == 'no_placement' for c in commits),
        'n_stabilized': sum(bool(c.get('placed_rot_angle')) for c in placed),
        'termination': next((c['termination'] for c in reversed(commits) if c.get('termination')), None),
        'metrics_version': METRICS_VERSION,
    }


def step_rows(result):
    """One row per sampling attempt: the action taken and the candidate set it was taken from."""
    result = Path(result)
    rows = []
    for c in read_jsonl(result / 'trace' / 'commits.jsonl'):
        side = c['sides'].get(c['chosen_side'], {})
        row = {k: c.get(k) for k in ('step', 'phase', 'outcome', 'chosen_side', 'side_tie', 'stone_id', 'cluster',
                                     'rotation_world', 'x_world', 'y', 'termination')}
        row['placement_id'] = ''
        if side.get('event') is not None:
            meta, a = load_trace(result / 'trace' / f"e{side['event']:05d}.npz")
            p = meta['policy']
            sel = select(a['features_raw'], a['weights'], p['policy'], p['order'], p['tol'])
            row.update(event=side['event'], n_candidates=len(a['stone_id']),
                       n_stones=len(set(a['stone_id'].tolist())), n_feasible=int((a['features_raw'][:, 3] > 0).sum()),
                       branch=meta['branch'], final_ties=tie_count(a['features_raw'], a['weights'], sel, p['policy'],
                                                                  p['order'], p['tol']))
            if side.get('index') is not None:
                i = side['index']
                row['placement_id'] = f"{a['footprint'][i]}:{a['x_world'][i]},{a['y'][i]}"
        rows.append(row)
    return rows


def action_key(row):
    return (row['outcome'], row['stone_id'], row['rotation_world'], row['x_world'], row['y'])


def first_divergence(rows, reference):
    """Step of the first sampling attempt whose action differs from the reference run's."""
    for a, b in zip(rows, reference):
        if action_key(a) != action_key(b):
            return a['step']
    if len(rows) != len(reference):
        shorter = rows if len(rows) < len(reference) else reference
        return (shorter[-1]['step'] if shorter else 0) + 1
    return None
