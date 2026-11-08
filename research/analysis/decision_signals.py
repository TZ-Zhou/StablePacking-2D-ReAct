"""Two cheap decision signals, offline, on recorded walls only (never wired into a controller).

  python -B -m research.analysis.decision_signals --out DIR

Walls: the 50 of campaign thangavelu-5arms-10seeds (wang-quality-v1) and the 6 branch points of
lookahead_study.points(). Each signal is computed from one selection event and the events before
it of the same wall; final walls and branch results are read only to evaluate it.

A. Pareto (auxiliary: it removes nothing, the cascade is unchanged). The feasible candidates of
an event (LLC > 0, five finite filter scores: Algorithm 2's set) on the five weighted filter
scores, all higher-is-better as the cascade keeps them; the front is the candidates no feasible
candidate strictly dominates (>= on all five, > on one). Reported: front / feasible; how many of
the event's outcome classes (distinct picks of the 120 paper-cascade orders) are on it; at the
branch points, whether an alternative off the front had a better final value than one on it.

B. Hard stones. Per stone, before the event: area (its region in the original stone label image,
known from the start), n_sampled, n_infeasible (sampled with no feasible candidate), poses (its
feasible candidates when last sampled, None if never sampled). Hardness, fixed in advance:
higher n_infeasible, then fewer poses, then larger area. Evaluated: (1) per wall, when 6 stones
remain, the hardness rank of each stone that ends unplaced; (2) the wall's top hardness then
against its final placed_ratio; (3) at the branch points, the rank of each alternative's stone
and its final placed_ratio. Exploratory: 5 unplaced stones in 4 of the 50 walls.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from research.paths import LEGACY_RUNS, QUALITY_DIR
from research.storage.case_id import wall_dir
from research.analysis import lookahead_study
from research.walls import wall_metrics
from Stonepacker2D.stack.selection import ALL_ORDERS, select  # noqa: E402

QUALITY = QUALITY_DIR / 'wang-quality-v1' / 'quality.csv'
TOL = 1e-9
CHECKPOINT = 6
FINAL = (('placed_ratio', 1), ('F_SF', 1), ('F_AH', -1), ('F_AV', 1), ('F_LR', 1))


def result_of(run):
    return next(Path(run).glob('result_*'))


def events(run):
    for f in sorted((result_of(run) / 'trace').glob('e*.npz')):
        a = dict(np.load(f))
        yield json.loads(str(a['meta'])), a


def feasible(raw):
    return np.flatnonzero((raw[:, 3] > 0) & np.isfinite(raw[:, :5]).all(axis=1))


def front(raw, weights):
    """Indices of the feasible candidates on the Pareto front of the weighted filter scores."""
    idx = feasible(raw)
    x = (raw * weights)[idx, :5]
    ge = (x[:, None, :] >= x[None, :, :]).all(axis=2)
    gt = (x[:, None, :] > x[None, :, :]).any(axis=2)
    dominated = (ge & gt).any(axis=0)  # [j, i]: j dominates i
    return idx[~dominated]


def classes(raw, weights):
    return sorted({int(select(raw, weights, 'paper', o, TOL).index) for o in ALL_ORDERS} - {None}) if feasible(raw).size else []


def areas(run):
    from skimage.measure import label, regionprops
    import cv2
    image = label(cv2.imread(str(wall_metrics.stone_source(run)), cv2.IMREAD_GRAYSCALE).astype(np.uint8))
    return {r.label: int(r.area) for r in regionprops(image) if r.area >= 50}


class Hardness:
    """Per-stone history of one wall, updated event by event (only what is known by then)."""

    def __init__(self, run):
        self.area = areas(run)
        self.stat = {}

    def rank(self, remaining):
        """Remaining stone ids, hardest first."""
        def key(s):
            st = self.stat.get(s, {'n_infeasible': 0, 'poses': None})
            return (-st['n_infeasible'], st['poses'] if st['poses'] is not None else 10 ** 9, -self.area.get(s % 1000, 0), s)
        return sorted(remaining, key=key)

    def update(self, a):
        ok = feasible(a['features_raw'])
        poses = {}
        for s in a['stone_id'][ok].tolist():
            poses[s] = poses.get(s, 0) + 1
        for s in a['sampled_ids'].tolist():
            st = self.stat.setdefault(s, {'n_sampled': 0, 'n_infeasible': 0, 'poses': None})
            st['n_sampled'] += 1
            st['n_infeasible'] += s not in poses
            st['poses'] = poses.get(s, 0)


def remaining(a):
    return a['inventory_ids'].tolist() + a['sampled_ids'].tolist()


def walls(final):
    rows, timing = [], {'pareto': [], 'classes': [], 'hardness': []}
    for run_name, q in final.items():
        run = wall_dir(run_name)
        placed = {r.id for r in __import__('pickle').load(open(result_of(run) / 'base.pkl', 'rb')).placed_rocks}
        hard, checkpoint = Hardness(run), None
        n_ev = n_feas = n_front = n_cls = n_cls_on = 0
        for meta, a in events(run):
            raw, w = a['features_raw'], a['weights']
            t = time.perf_counter()
            f = set(front(raw, w).tolist()) if raw.size else set()
            timing['pareto'].append(time.perf_counter() - t)
            t = time.perf_counter()
            c = classes(raw, w) if raw.size else []
            timing['classes'].append(time.perf_counter() - t)
            n_ev += 1
            n_feas += feasible(raw).size if raw.size else 0
            n_front += len(f)
            n_cls += len(c)
            n_cls_on += sum(i in f for i in c)
            rem = remaining(a)
            if checkpoint is None and len(rem) <= CHECKPOINT:
                t = time.perf_counter()
                order = hard.rank(rem)
                timing['hardness'].append(time.perf_counter() - t)
                checkpoint = {'remaining': order, 'unplaced': [s for s in order if s not in placed],
                              'unplaced_rank': [order.index(s) + 1 for s in order if s not in placed],
                              'top': hard.stat.get(order[0], {}) | {'id': order[0]}}
            t = time.perf_counter()
            hard.update(a)
            timing['hardness'].append(time.perf_counter() - t)
        rows.append({'run': run_name, 'placed_ratio': float(q['placed_ratio']), 'events': n_ev, 'feasible': n_feas,
                     'front': n_front, 'classes': n_cls, 'classes_on_front': n_cls_on, 'checkpoint': checkpoint})
        print(Path(run_name).name, n_ev, flush=True)
    return rows, timing


def branch_points():
    out = []
    for (parent, step), alts in lookahead_study.points().items():
        run = wall_dir(parent)
        commit = next(c for c in lookahead_study.commits(run) if c['step'] == step and c.get('stone_id') is not None)
        event = commit['sides'][commit['chosen_side']]['event']
        hard = Hardness(run)
        for meta, a in events(run):
            if int(meta['event']) == event:
                break
            hard.update(a)
        raw, w = a['features_raw'], a['weights']
        f = set(front(raw, w).tolist())
        order = hard.rank(remaining(a))
        plan = {b['name']: b for pid in lookahead_study.PLANS
                for b in json.loads((LEGACY_RUNS / 'branches' / pid / 'plan.json').read_text(encoding='utf-8'))['branches']}
        rows = []
        for alt in alts:
            if alt['switch'] is None:
                i = int(a['selected'])
            else:
                s, rot, x, y = plan[alt['name']]['pose']
                i = int(np.flatnonzero((a['stone_id'] == s) & (a['rotation_world'] == rot) & (a['x_world'] == x) & (a['y'] == y))[0])
            stone = int(a['stone_id'][i])
            rows.append({'alt': alt['name'], 'stone': stone, 'on_front': i in f, 'hardness_rank': order.index(stone) + 1,
                         'n_remaining': len(order), 'final': lookahead_study.finals(alt)})
        out.append({'point': f'{Path(parent).name}@{step}', 'n_feasible': int(feasible(raw).size), 'n_front': len(f),
                    'classes_on_front': sum(i in f for i in classes(raw, w)), 'n_classes': len(classes(raw, w)), 'alternatives': rows})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', required=True)
    out = Path(ap.parse_args(argv).out)
    if out.exists():
        raise SystemExit(f'{out} exists')
    final = {r['run']: r for r in csv.DictReader(open(QUALITY, encoding='utf-8'))}
    rows, timing = walls(final)
    points = branch_points()
    excluded_better = []
    for p in points:
        on = [r for r in p['alternatives'] if r['on_front']]
        for r in p['alternatives']:
            if not r['on_front']:
                excluded_better += [{'point': p['point'], 'off': r['alt'], 'metric': m} for m, s in FINAL
                                    if on and s * r['final'][m] > max(s * o['final'][m] for o in on)]
    summary = {
        'pareto': {'events': sum(r['events'] for r in rows), 'feasible': sum(r['feasible'] for r in rows),
                   'front': sum(r['front'] for r in rows), 'classes': sum(r['classes'] for r in rows),
                   'classes_on_front': sum(r['classes_on_front'] for r in rows),
                   'front_share_per_wall': sorted(round(r['front'] / r['feasible'], 3) for r in rows if r['feasible']),
                   'branch_alternatives_off_front': sum(not r['on_front'] for p in points for r in p['alternatives']),
                   'excluded_better': excluded_better},
        'hardness_checkpoint': [{'run': r['run'], 'placed_ratio': r['placed_ratio'], **r['checkpoint']} for r in rows],
        'timing_ms_per_event': {k: {'median': round(1e3 * float(np.median(v)), 3), 'max': round(1e3 * float(np.max(v)), 3)}
                                for k, v in timing.items() if v},
        'branch_points': points}
    out.mkdir(parents=True)
    (out / 'signals.json').write_text(json.dumps(summary, indent=1, default=float), encoding='utf-8')
    print(json.dumps({k: summary['pareto'][k] for k in ('events', 'feasible', 'front', 'classes', 'classes_on_front',
                                                       'branch_alternatives_off_front', 'excluded_better')}))
    print(json.dumps(summary['timing_ms_per_event']))


if __name__ == '__main__':
    main()
