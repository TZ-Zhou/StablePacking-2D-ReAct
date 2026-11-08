"""P0 study of the Observation V3 layer A preview (research/analysis/placement_preview.py) on recorded walls only.

  python -B -m research.analysis.placement_preview_study --out DIR

Reads the 50 walls of campaign thangavelu-5arms-10seeds, their wang-quality-v1 values, and the
branch walls of paper-hardstone-01 and smoke-001-cf (neutral branches left out: they rebuild the
recorded wall). Writes DIR/decisions.jsonl (every selection event of the 50 walls: the wall's
quality before, each outcome class's preview and change, the picks of Paper, Rule-based Adaptive
and Greedy-Preview), DIR/branches.json (each branch point's alternatives: preview at the decision,
final values) and DIR/summary.json. The records are only read.

Candidate pixels are not in Wang's trace; each is rebuilt from a library of placed rocks' pixels
(and their quarter turns and mirror images) whose footprint hash equals the trace's footprint of
the candidate, i.e. the same pixels; a candidate without a match has no preview.
"""
import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from research.paths import LEGACY_RUNS, QUALITY_DIR
from research.controller import order_controller
from research.walls import wall_metrics
from research.analysis import placement_preview
from Stonepacker2D.stack.selection import PAPER_ORDER, footprint, select  # noqa: E402
from Stonepacker2D.utils.constant import get_wedge_id  # noqa: E402

CAMPAIGN = 'thangavelu-5arms-10seeds'
QUALITY = QUALITY_DIR / 'wang-quality-v1' / 'quality.csv'
BRANCH_PLANS = ['paper-hardstone-01', 'smoke-001-cf']
TOL = 1e-9
FINAL = ('F_SF', 'F_AH', 'F_AV', 'F_LR', 'placed_ratio')
SENSE = {'F_SF': 1, 'F_AH': -1, 'F_AV': 1, 'F_LR': 1, 'placed_ratio': 1}
CHANGE = ('density', 'ah', 'av', 'ah_band', 'av_band')


def result_of(run):
    return next(Path(run).glob('result_*'))


def shape_library(runs):
    """footprint hash -> the footprint crop, from every placed rock of the given runs."""
    lib = {}
    for run in runs:
        for rock in pickle.load(open(result_of(run) / 'base.pkl', 'rb')).placed_rocks:
            m = rock.matrix != 0
            rows, cols = np.nonzero(m)
            crop = m[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
            for c in (crop, np.fliplr(crop)):
                for k in range(4):
                    turned = np.ascontiguousarray(np.rot90(c, k))
                    lib.setdefault(footprint(turned, 'left'), turned)
    return lib


def events_of(run):
    out = []
    for f in sorted((result_of(run) / 'trace').glob('e*.npz')):
        a = dict(np.load(f))
        out.append((json.loads(str(a['meta'])), a))
    return out


def wedge_values(run):
    base = pickle.load(open(result_of(run) / 'base.pkl', 'rb'))
    return {k for k, b in enumerate(base.placed_bricks, 1) if b.id == get_wedge_id()}


def change(after, now):
    return {k: None if after[k] is None or now[k] is None else after[k] - now[k] for k in CHANGE}


def decisions(run, lib, wedges):
    """One row per selection event, Observer version 2 run over the trace with Paper as the base."""
    observer, rule, rows = order_controller.Observer(version=2), order_controller.RulePolicy(), []
    policy = {'policy': 'paper', 'order': PAPER_ORDER, 'tol': TOL}
    for meta, a in events_of(run):
        obs, info = observer(policy, order_controller.context_from_trace(meta, a))
        wall = np.asarray(a['wall'])
        world = np.fliplr(wall) if meta['side'] == 'right' else wall
        now = placement_preview.quality(world, wedges)
        classes, raw = info['classes'], np.asarray(a['features_raw'])
        outcomes = []
        for k, i in enumerate(classes):
            crop = lib.get(str(a['footprint'][i]))
            px = None if crop is None else placement_preview.candidate_pixels(crop, int(a['x_world'][i]), int(a['y'][i]), world.shape)
            p = None if px is None else placement_preview.preview(world, px, wedges)
            outcomes.append({'id': k, 'index': int(i), 'stone': int(a['stone_id'][i]), 'llc': float(raw[i, 3]),
                             'preview': p, 'preview_change': None if p is None else change(p['after'], now),
                             'n_supports': None if p is None else p['n_supports']})
        paper = classes.index(info['paper']) if info['paper'] in classes else None
        selected = int(a['selected'])
        row = {'run': str(Path(run).relative_to(LEGACY_RUNS).as_posix()), 'event': int(meta['event']), 'step': int(meta['step']),
               'side': meta['side'], 'n_rocks': int(np.count_nonzero(np.unique(placement_preview.rock_labels(world, wedges)))),
               'now': now, 'n_outcomes': len(classes), 'outcomes': outcomes, 'paper': paper,
               'recorded': classes.index(selected) if selected in classes else None}
        if len(classes) >= 2 and paper is not None and all(o['preview'] is not None for o in outcomes):
            order = order_controller.parse_order(rule.decide({'observation': obs, 'base_order': order_controller.key(PAPER_ORDER)})['order'])
            pick = select(raw, np.asarray(a['weights']), 'paper', tuple(order), TOL).index
            row['rule'] = classes.index(int(pick)) if pick is not None and int(pick) in classes else None
            row['greedy_full'] = placement_preview.greedy_preview(outcomes, paper, 'full')
            row['greedy_band'] = placement_preview.greedy_preview(outcomes, paper, 'band')
        rows.append(row)
    return rows


def spearman(x, y):
    keep = [i for i in range(len(x)) if x[i] is not None and y[i] is not None]
    if len(keep) < 5:
        return {'rho': None, 'n': len(keep)}
    r = spearmanr([x[i] for i in keep], [y[i] for i in keep])
    return {'rho': round(float(r.correlation), 3), 'p': round(float(r.pvalue), 4), 'n': len(keep)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', required=True)
    out = Path(ap.parse_args(argv).out)
    if out.exists():
        raise SystemExit(f'{out} exists')
    final = {r['run']: r for r in csv.DictReader(open(QUALITY, encoding='utf-8'))}
    walls = [LEGACY_RUNS / run for run in final]
    plans = [json.loads((LEGACY_RUNS / 'branches' / p / 'plan.json').read_text(encoding='utf-8')) for p in BRANCH_PLANS]
    branches = [(p['id'], b) for p in plans for b in p['branches'] if b['kind'] != 'neutral']
    lib = shape_library(walls + [LEGACY_RUNS / 'branches' / pid / b['name'] for pid, b in branches])

    rows = []
    for run in walls:
        rows += decisions(run, lib, wedge_values(run))
        print(run.name, len(rows), flush=True)
    by_run = {}
    for r in rows:
        by_run.setdefault(r['run'], []).append(r)

    # A: does the partial wall's quality after k rocks say anything about the final values?
    checkpoints = {}
    for k in (6, 9, 12, 15):
        at = {run: next((r['now'] for r in rs if r['n_rocks'] == k), None) for run, rs in by_run.items()}
        checkpoints[k] = {f: {m: spearman([None if at[run] is None else at[run][f] for run in final],
                                          [float(final[run][m]) for run in final]) for m in FINAL} for f in CHANGE}
    end = {run: placement_preview.quality(pickle.load(open(result_of(LEGACY_RUNS / run) / 'base.pkl', 'rb')).matrix, wedge_values(LEGACY_RUNS / run))
           for run in final}
    at_end = {f: {m: spearman([end[run][f] for run in final], [float(final[run][m]) for run in final]) for m in FINAL}
              for f in CHANGE}
    availability = {k: {f: sum(1 for run in final for r in by_run[run] if r['n_rocks'] == k and r['now'][f] is not None)
                        for f in CHANGE} for k in (3, 6, 9, 12, 15)}

    # B: branch points: preview at the decision against the alternatives' final values
    points = {}
    for pid, b in branches:
        run = LEGACY_RUNS / b['run']
        parent = by_run[b['run']]
        commit = next(c for c in map(json.loads, open(result_of(run) / 'trace' / 'commits.jsonl', encoding='utf-8'))
                      if c['step'] == b['step'] and c.get('stone_id') is not None)
        event = commit['sides'][commit['chosen_side']]['event']
        row = next(r for r in parent if r['event'] == event)
        a = dict(np.load(result_of(run) / 'trace' / f'e{event:05d}.npz'))
        stone, rot, x, y = b['pose']
        hit = np.flatnonzero((a['stone_id'] == stone) & (a['rotation_world'] == rot) & (a['x_world'] == x) & (a['y'] == y))
        branch_run = LEGACY_RUNS / 'branches' / pid / b['name']
        q = wall_metrics.wall_quality(branch_run, result_of(branch_run), 'corrected')
        wm = wall_metrics.wall_metrics(branch_run, result_of(branch_run))
        point = points.setdefault(f"{b['run']}@{b['step']}", {'row': row, 'alternatives': {}})
        rec = final[b['run']]
        point['alternatives'].setdefault('recorded', {'class': row['recorded'], **{m: float(rec[m]) for m in FINAL}})
        cls = next(o['id'] for o in row['outcomes'] if o['index'] in set(hit.tolist())) if hit.size else None
        point['alternatives'][b['name']] = {'class': cls, **{m: q[m] for m in FINAL[:4]}, 'placed_ratio': wm['placed_ratio']}
    pairs, picks = Counter(), []
    for name, p in points.items():
        alts = [v for v in p['alternatives'].values() if v['class'] is not None]
        outs = {o['id']: o for o in p['row']['outcomes']}
        for i in range(len(alts)):
            for j in range(i + 1, len(alts)):
                u, v = alts[i], alts[j]
                if u['class'] == v['class']:
                    continue
                pu, pv = outs[u['class']]['preview_change'], outs[v['class']]['preview_change']
                for f in CHANGE + ('n_supports', 'llc'):
                    a = outs[u['class']][f] if f in ('n_supports', 'llc') else pu[f]
                    b = outs[v['class']][f] if f in ('n_supports', 'llc') else pv[f]
                    if a is None or b is None or abs(a - b) < TOL:
                        continue
                    for m in FINAL:
                        d = u[m] - v[m]
                        if abs(d) > TOL:
                            pairs[(f, m, 'same' if (a > b) == (d > 0) else 'opposite')] += 1
        known = {v['class']: v for v in p['alternatives'].values() if v['class'] is not None}
        for policy in ('paper', 'rule', 'greedy_full', 'greedy_band'):
            c = p['row'].get(policy)
            picks.append({'point': name, 'policy': policy, 'class': c, 'known': c in known,
                          **({m: known[c][m] for m in FINAL} if c in known else {})})

    # C: how often each rule leaves Paper at the 50 walls' decision points
    dp = [r for r in rows if 'rule' in r]
    leave = {pol: sum(r[pol] != r['paper'] for r in dp) for pol in ('rule', 'greedy_full', 'greedy_band')}
    coverage = Counter('preview' if o['preview'] is not None else 'no pixels' for r in rows for o in r['outcomes'])

    out.mkdir(parents=True)
    with open(out / 'decisions.jsonl', 'w', encoding='utf-8', newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, default=float) + '\n')
    (out / 'branches.json').write_text(json.dumps({'points': {k: {**v, 'row': {kk: v['row'][kk] for kk in ('run', 'event', 'step', 'paper', 'recorded', 'rule', 'greedy_full', 'greedy_band', 'outcomes') if kk in v['row']}} for k, v in points.items()},
                                                   'pairs': {'|'.join(k): n for k, n in sorted(pairs.items())}, 'picks': picks},
                                                  indent=1, sort_keys=True, default=float), encoding='utf-8')
    summary = {'n_events': len(rows), 'n_decision_points': len(dp), 'outcome_coverage': dict(coverage),
               'availability_by_rocks': availability, 'checkpoints': checkpoints, 'final_wall_sim_grid': at_end,
               'leave_paper': leave, 'n_branch_points': len(points)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, sort_keys=True, default=float), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('n_events', 'n_decision_points', 'outcome_coverage', 'leave_paper', 'n_branch_points')}))


if __name__ == '__main__':
    main()
