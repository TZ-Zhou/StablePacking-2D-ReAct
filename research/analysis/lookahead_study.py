"""P1 (exploratory): limited lookahead at the 6 counterfactual branch points with known final walls.

  python -B -m research.analysis.lookahead_study run [--max-runs N]   builds the lookahead walls
  python -B -m research.analysis.lookahead_study evaluate --out DIR    compares them with the final walls

A branch point is a step of a recorded paper-order wall (3-0-2-1-4) where branches of the plans
paper-hardstone-01 and smoke-001-cf (neutral ones left out: they rebuild the recorded wall) took
another candidate. Its alternatives: the recorded candidate and each branch's. The lookahead of an
alternative is run_wall.py as counterfactual_branches.py runs its branch (same case, seed, recorded order,
--switch for a branch), stopped by its own --placements once the wall holds the decision's P
placements + K + 1 (K = 4): the alternative, then K placements by the Paper order, all with the
experiment's code, RNG, stones and settings. Its inputs are the seed, the order and the switch;
it reads no final wall and no quality. run then requires every trace event of the lookahead to be
the existing wall's (parent or branch) event of the same number: same candidates, choice and
wall; it stops at the first difference.

State after k = 0, 2, 4 placements past the alternative: the wall of the first selection event
of placement P + k + 1 (k = 0: the alternative placed and stabilized, the single-step state);
if the lookahead's wall is finished before that, its own finished wall (base.pkl of the
lookahead run; 'end' marks it: that lookahead is a full rollout).
Lookahead values of a state, each the counterpart of one final value (sim grid, world frame):
  lost       stones neither placed nor left to place (input - placed - inventory - sampled)
             -> placed_ratio (lower better)
  sf         rock px / ((W - 2) * (H - 3)), the whole wall's area -> F_SF (higher better)
  ah, av     03's F_AH / F_AV of the partial wall's rock labels -> F_AH (lower), F_AV (higher)
  lr         Wang's evaluate_kine (tilting table) of the partial wall / 0.58 -> F_LR (higher)
             (the wall matrix alone reproduces the recorded lambda of finished walls exactly)
Wedges (from commits: the bricks added after a placement's stone) are not rocks.

Comparison rules, fixed before any lookahead was run; no weights, no total score:
  per value and k, over the pairs of alternatives of a point whose lookahead and final
  differences are both nonzero: concordant / discordant counts (with the sign of 'better');
  top-1: the alternative best by lookahead is also best by the final value (unique bests only);
  Spearman of lookahead and final values over all alternatives, and within points (both minus
  the point's mean). k = 0 is the single-step reference. 6 points: exploratory only.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from research.paths import QUALITY_DIR
from research.analysis import placement_preview
from research.walls import run_wall

RUNS = run_wall.RUNS
OUT = 'lookahead-p1'
PLANS = ['paper-hardstone-01', 'smoke-001-cf']
K = 4
DEPTHS = (0, 2, 4)
ORDER = '3,0,2,1,4'
# lookahead value: (final value, +1 higher is better / -1 lower, for the lookahead value, for the final)
VALUES = {'lost': ('placed_ratio', -1, 1), 'sf': ('F_SF', 1, 1), 'ah': ('F_AH', -1, -1), 'av': ('F_AV', 1, 1), 'lr': ('F_LR', 1, 1)}


def result_of(run):
    return next(Path(run).glob('result_*'))


def commits(run):
    return [json.loads(line) for line in open(result_of(run) / 'trace' / 'commits.jsonl', encoding='utf-8')]


def points():
    """{(parent run, step): [alternative]}; an alternative: name, its existing wall, switch."""
    out = {}
    for pid in PLANS:
        plan = json.loads((RUNS / 'branches' / pid / 'plan.json').read_text(encoding='utf-8'))
        for b in plan['branches']:
            if b['kind'] == 'neutral':
                continue
            alts = out.setdefault((b['run'], b['step']), [{'name': 'recorded', 'wall': b['run'], 'switch': None, 'seed': b['seed']}])
            alts.append({'name': b['name'], 'wall': f"branches/{pid}/{b['name']}", 'switch': f"{b['step']}={b['order'].replace('-', ',')}",
                         'seed': b['seed']})
    return out


def placed_before(run, step):
    return sum(c['outcome'] == 'placed' for c in commits(run) if c['step'] < step)


def lookahead_name(parent, step, alt):
    return f"{OUT}/{Path(parent).name}-step{step}-{alt['name'] if alt['switch'] else 'recorded'}"


def same_events(look, wall):
    """Event numbers where the lookahead's trace differs from the existing wall's."""
    bad = []
    for f in sorted((result_of(look) / 'trace').glob('e*.npz')):
        a, b = np.load(f), np.load(result_of(wall) / 'trace' / f.name)
        for key in ('features_raw', 'stone_id', 'x_world', 'y', 'rotation_world', 'selected', 'wall'):
            if not np.array_equal(a[key], b[key]):
                bad.append((f.name, key))
                break
    return bad


def run(max_runs=None):
    started = 0
    for (parent, step), alts in points().items():
        n = placed_before(RUNS / parent, step) + 1 + K + 1
        for alt in alts:
            name = lookahead_name(parent, step, alt)
            if (RUNS / name).exists():
                continue
            if max_runs is not None and started >= max_runs:
                return True
            args = ['thangavelu-noise80', name, '--policy', 'paper', '--order', ORDER, '--seed', str(alt['seed']),
                    '--timeout', '2400', '--placements', str(n)] + (['--switch', alt['switch']] if alt['switch'] else [])
            run_wall.main(args)
            started += 1
            status = json.loads((RUNS / name / 'receipt.json').read_text(encoding='utf-8'))['status']
            bad = same_events(RUNS / name, RUNS / alt['wall'])
            print(json.dumps({'run': name, 'status': status, 'differs': bad[:3]}), flush=True)
            if status not in ('STOPPED_AFTER_PLACEMENTS', 'PROCESS_PASS') or bad:
                print('stopped for inspection')
                return False
    return True


def kine_lr(world):
    """Wang's evaluate_kine of the wall matrix alone; placed_bricks need only matrix, width, height."""
    from Stonepacker2D.evaluate.tilting_angle import evaluate_kine
    from Stonepacker2D.utils.constant import set_head_joint_state, set_kine_precision, set_world_size
    set_world_size(world.shape[1], world.shape[0])
    set_head_joint_state(True)
    set_kine_precision(15)
    bricks = []
    for k in range(1, int(world[world != 255].max()) + 1):
        r, c = np.nonzero(world == k)
        bricks.append(SimpleNamespace(matrix=(world == k).astype(np.uint8), width=np.ptp(c) + 1 if r.size else 0,
                                      height=np.ptp(r) + 1 if r.size else 0))
    base = SimpleNamespace(matrix=world, placed_bricks=bricks, cluster_matrix=np.zeros(world.shape))
    return evaluate_kine(base, load='tilting_table') / 0.58


def states(look, p):
    """{k: lookahead values after k placements past the alternative} from the lookahead's trace;
    p: placements up to and including the alternative."""
    cs = commits(look)
    placed = [c for c in cs if c['outcome'] == 'placed']
    wedges, v = set(), 1
    for c in placed:
        v += 1
        wedges |= set(range(v + 1, v + c['bricks_added']))
        v += c['bricks_added'] - 1
    first = lambda step: result_of(look) / 'trace' / f"e{min(s['event'] for c in cs if c['step'] == step for s in c['sides'].values()):05d}.npz"  # noqa: E731
    e0 = np.load(result_of(look) / 'trace' / 'e00000.npz')
    n_input = len(e0['inventory_ids']) + len(e0['sampled_ids'])
    out = {}
    for k in DEPTHS:
        end = p + k >= len(placed)
        if end:
            import pickle
            f = result_of(look) / 'base.pkl'
            world, remaining = np.asarray(pickle.load(open(f, 'rb')).matrix), 0
        else:
            f = first(placed[p + k]['step'])
            a = np.load(f)
            world = np.fliplr(a['wall']) if json.loads(str(a['meta']))['side'] == 'right' else np.asarray(a['wall'])
            remaining = len(a['inventory_ids']) + len(a['sampled_ids'])
        labels = placement_preview.rock_labels(world, wedges)
        H, W = world.shape
        out[k] = {'end': end, 'lost': n_input - len(np.unique(labels[labels > 0])) - remaining,
                  'sf': float(np.count_nonzero(labels) / ((W - 2) * (H - 3))),
                  'ah': placement_preview.paths_metric(labels, False), 'av': placement_preview.paths_metric(labels, True),
                  'lr': float(kine_lr(world)), 'mtime': os.stat(f).st_mtime}
    t0 = os.stat(first(placed[p - 1]['step'])).st_mtime  # the decision's event
    for k in DEPTHS:
        out[k]['seconds_from_decision'] = round(out[k].pop('mtime') - t0, 1)
    return out


def finals(alt):
    """The alternative's final values (evaluation only, never a lookahead input)."""
    from research.walls import wall_metrics
    q = {r['run']: r for r in csv.DictReader(open(QUALITY_DIR / 'wang-quality-v1' / 'quality.csv', encoding='utf-8'))}
    if alt['wall'] in q:
        return {v[0]: float(q[alt['wall']][v[0]]) for v in VALUES.values()}
    wall = RUNS / alt['wall']
    v = wall_metrics.wall_quality(wall, result_of(wall), 'corrected')
    return {**{m: v[m] for m in ('F_SF', 'F_AH', 'F_AV', 'F_LR')}, 'placed_ratio': wall_metrics.wall_metrics(wall, result_of(wall))['placed_ratio']}


def rho(x, y):
    from scipy.stats import spearmanr
    keep = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(keep) < 4:
        return None
    r = spearmanr(*zip(*keep)).correlation
    return None if np.isnan(r) else round(float(r), 3)


def evaluate(out):
    out = Path(out)
    if out.exists():
        raise SystemExit(f'{out} exists')
    rows = []
    for (parent, step), alts in points().items():
        p = placed_before(RUNS / parent, step) + 1  # with the alternative's own placement
        for alt in alts:
            look = RUNS / lookahead_name(parent, step, alt)
            receipt = json.loads((look / 'receipt.json').read_text(encoding='utf-8'))
            rows.append({'point': f'{Path(parent).name}@{step}', 'alt': alt['name'], 'states': states(look, p), 'final': finals(alt),
                         'elapsed_seconds': receipt['elapsed_seconds']})
    table = {}
    for k in DEPTHS:
        for v, (m, ls, fs) in VALUES.items():
            conc = disc = ties = top_hit = top_miss = 0
            for point in sorted({r['point'] for r in rows}):
                alts = [r for r in rows if r['point'] == point]
                look = [r['states'][k][v] for r in alts]
                fin = [r['final'][m] for r in alts]
                for i in range(len(alts)):
                    for j in range(i + 1, len(alts)):
                        if None in (look[i], look[j]):
                            continue
                        dl, df = look[i] - look[j], fin[i] - fin[j]
                        if abs(dl) < 1e-9 or abs(df) < 1e-9:
                            ties += 1
                        elif (ls * dl > 0) == (fs * df > 0):
                            conc += 1
                        else:
                            disc += 1
                ok = [i for i in range(len(alts)) if look[i] is not None]
                if ok:
                    best_l = [i for i in ok if ls * look[i] == max(ls * look[j] for j in ok)]
                    best_f = [i for i in ok if fs * fin[i] == max(fs * fin[j] for j in ok)]
                    if len(best_l) == 1 and len(best_f) == 1:
                        top_hit += best_l == best_f
                        top_miss += best_l != best_f
            x = [r['states'][k][v] for r in rows]
            y = [r['final'][m] for r in rows]
            centred = []
            for point in {r['point'] for r in rows}:
                idx = [i for i, r in enumerate(rows) if r['point'] == point and x[i] is not None]
                if len(idx) >= 2:
                    mx, my = np.mean([x[i] for i in idx]), np.mean([y[i] for i in idx])
                    centred += [(x[i] - mx, y[i] - my) for i in idx]
            table[f'k{k}|{v}->{m}'] = {'concordant': conc, 'discordant': disc, 'ties': ties, 'top1_hit': top_hit, 'top1_miss': top_miss,
                                       'rho_all': rho(x, y), 'rho_within': rho(*zip(*centred)) if centred else None,
                                       'n_undefined': sum(a is None for a in x)}
    out.mkdir(parents=True)
    (out / 'lookahead.json').write_text(json.dumps({'rows': rows, 'table': table}, indent=1, default=float), encoding='utf-8')
    for key, t in table.items():
        print(key.ljust(28), t)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('run').add_argument('--max-runs', type=int)
    sub.add_parser('evaluate').add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'run':
        return 0 if run(a.max_runs) else 1
    evaluate(a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
