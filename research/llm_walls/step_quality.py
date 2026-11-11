"""Step-wise quality of the gpt-001 counterfactual walls, offline from their records (no wall is built).

  python -B -m research.llm_walls.step_quality        # research/reports/gpt-001-cf-steps/, written once

Reads the walls of the research/runs archive: branches/gpt-001-cf and its two source walls.
After each placement n of a recorded wall:
- F_SF (author: 03's right-cropped area), F_SF_progress (stone px / (H*W - 2H - 3W) of the whole 02
  wall: one denominator for every wall and step), F_AH_partial, F_AV_partial: 02 (corrected canvas,
  as wang-quality-v1) on the first n lines of transformation.txt, then 03 unchanged. 03 drops the
  empty rows above the highest stone, so the pockets of an unfinished wall count as joints: partial
  values, never a final quality. At the last placement they are wang-quality-v1's.
- lambda_left, lambda_right: evaluate_kine's two tilting-table solves (solve_force_rigid, unrounded)
  of the simulation wall after n; F_LR = min / mu, the partial wall's limit.
Checks, each fatal: V's series is gpt-001's; at the last placement F_SF, F_AH, F_AV are
wang-quality-v1's (summary.json of the counterfactual plan), and F_LR is wang-quality-v1's and the
lambdas the mechanism files' within LAMBDA_RTOL (the walls predate the speedup 5a6992c; since it a
repeated solve, and so F_LR = min / mu, is an ulp or two off);
at every placement round(lambda_min, 1) / mu is the chosen candidate's LLC; G_k and P_k agree before
the placement of step k.
"""
import csv
import json
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

from research.paths import LEGACY_RUNS, REPO
from research.walls import wall_metrics as metrics
from research.walls.wall_metrics import lambdas
from Stonepacker2D.stack.selection import load_trace

OUT = REPO / 'research' / 'reports' / 'gpt-001-cf-steps'
CF = 'branches/gpt-001-cf'
WALLS = {'GPT': 'gpt-walls/gpt-001', 'Paper': 'campaigns/dev-2seeds/thangavelu-noise80/paper/seed66794912',
         **{w: f'{CF}/{w}' for w in ('V', 'G6', 'G8', 'P8', 'G17', 'P17')}}
PAIRS = {'k6': ('G6', 'Paper', 6), 'k8': ('G8', 'P8', 8), 'k17': ('G17', 'P17', 17), 'GPT-Paper': ('GPT', 'Paper', 6)}
METRICS = ('F_SF', 'F_SF_progress', 'F_AH_partial', 'F_AV_partial', 'lambda_left', 'lambda_right', 'F_LR')
LLC_DECIMALS = 1  # Stonepacker2D.utils.constant.KINEMATIC_RESULT_DECIMAL
LAMBDA_RTOL = 1e-12  # a solve repeated by later code against the recorded one


def result_of(run):
    return next((LEGACY_RUNS / run).glob('result_*'))


def commits(result):
    return [json.loads(l) for l in (result / 'trace' / 'commits.jsonl').read_text(encoding='utf-8').splitlines()]


def wall_steps(name):
    """One row per placement of a recorded wall."""
    run, result = LEGACY_RUNS / WALLS[name], result_of(WALLS[name])
    cs = commits(result)
    lines = (result / 'transformation.txt').read_text().splitlines()
    if len(lines) - 1 != len(cs) or any(c['outcome'] != 'placed' or c['chosen_side'] != 'left' for c in cs):
        raise ValueError(f'{name}: not one left-side placement per commit')
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for n, c in enumerate(cs, 1):
            (tmp / 'transformation.txt').write_text('\n'.join(lines[:n + 1]) + '\n')
            labels, render = metrics.render_original(run, tmp, 'corrected')
            m = metrics.wang_metrics(tmp, labels=labels.astype(int))
            H, W = labels.shape
            if n < len(cs):
                world = np.asarray(load_trace(result / 'trace' / f"e{cs[n]['sides']['left']['event']:05d}.npz")[1]['wall'])
            else:
                world = np.asarray(pickle.load(open(result / 'base.pkl', 'rb')).matrix)
            lam = lambdas(world)
            meta, x = load_trace(result / 'trace' / f"e{c['sides']['left']['event']:05d}.npz")
            rows.append({'wall': name, 'n': n, 'step': c['step'], 'stone_id': c['stone_id'], 'complete': n == len(cs),
                         'F_SF': m['F_SF_author'], 'F_SF_progress': np.count_nonzero(labels) / (H * W - 2 * H - 3 * W),
                         'F_AH_partial': m['F_AH'], 'F_AV_partial': m['F_AV'],
                         'lambda_left': lam['left'], 'lambda_right': lam['right'],
                         'F_LR': min(lam.values()) / metrics.MU, 'min_side': min(lam, key=lam.get),
                         'LLC_chosen': float(x['features_raw'][int(x['selected'])][3]),
                         'stone_px': int(np.count_nonzero(labels)), 'n_left_out': sum(s['left_out'] for s in render['stones'])})
    return rows


def wall_problems(rows, final):
    """The fatal checks of one wall's rows; final: its wang-quality-v1 row and mechanism lambdas."""
    p = []
    for r in rows:
        if not np.isclose(round(min(r['lambda_left'], r['lambda_right']), LLC_DECIMALS) / metrics.MU, r['LLC_chosen'], rtol=0, atol=1e-12):
            p.append(f"n {r['n']}: round(lambda_min) / mu is not the chosen LLC {r['LLC_chosen']}")
    last = rows[-1]
    col = lambda k: k if k in ('F_SF', 'F_LR') else f'{k}_partial'  # noqa: E731
    off = lambda a, b: b is None or not np.isclose(a, b, rtol=LAMBDA_RTOL, atol=0)  # noqa: E731
    p += [f'last placement: {k} {last[col(k)]} is not wang-quality-v1 {final[k]}'
          for k in ('F_SF', 'F_AH', 'F_AV', 'F_LR')
          if (off(last[k], final[k]) if k == 'F_LR' else last[col(k)] != final[k])]
    p += [f'last placement: lambda_{s} {last[f"lambda_{s}"]} is not the mechanism file {final[s]}'
          for s in ('left', 'right') if off(last[f'lambda_{s}'], final[s])]
    return p


def pair_rows(a, b, n_k):
    """D(n) = a - b per metric, and the placed-stone sets' difference."""
    out = []
    for ra, rb in zip(a, b):
        ids_a = {r['stone_id'] for r in a[:ra['n']]}
        ids_b = {r['stone_id'] for r in b[:rb['n']]}
        out.append({'n': ra['n'], 'phase': 'prefix' if ra['n'] < n_k else 'decision' if ra['n'] == n_k else 'after',
                    'step_a': ra['step'], 'step_b': rb['step'], 'stone_a': ra['stone_id'], 'stone_b': rb['stone_id'],
                    'stones_only_a': len(ids_a - ids_b), 'stones_only_b': len(ids_b - ids_a),
                    'stone_px_diff': ra['stone_px'] - rb['stone_px'],
                    **{f'D_{m}': None if ra[m] is None or rb[m] is None else ra[m] - rb[m] for m in METRICS}})
    return out


def pair_summary(d, n_k):
    """Per metric: D at the decision placement, at the end, what came after, and the first placement
    where it is not 0."""
    out = {}
    for m in METRICS:
        series = [(r['n'], r[f'D_{m}']) for r in d]
        at = dict(series)
        onset = next((n for n, v in series if v is None or v != 0), None)
        out[m] = {'at_decision': at[n_k], 'final': at[len(d)],
                  'after_decision': None if None in (at[n_k], at[len(d)]) else at[len(d)] - at[n_k],
                  'onset_n': onset, 'onset_phase': None if onset is None else 'prefix' if onset < n_k else 'decision' if onset == n_k else 'after'}
    return out


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


COLORS = {'a': '#2a78d6', 'b': '#eb6834'}  # dataviz categorical slots 1 and 2, light surface
INK, MUTED, GRID = '#0b0b0b', '#898781', '#e1e0d9'
PANELS = (('F_SF_progress', 'F_SF_progress (higher)'), ('F_SF', 'F_SF author, right-cropped (higher)'),
          ('F_AH_partial', 'F_AH_partial (lower)'), ('F_AV_partial', 'F_AV_partial (higher)'),
          ('lambda_left', 'lambda left'), ('lambda_right', 'lambda right'))


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcdefaults()  # Stonepacker2D's imports set LaTeX-style rcParams, which take '_' for a subscript
    import matplotlib.pyplot as plt
    return plt


def style(ax):
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_facecolor('#fcfcfb')
    ax.grid(color=GRID, linewidth=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#c3c2b7')
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_pair(path, title, a, b, rows_a, rows_b, n_k):
    plt = pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), facecolor='#fcfcfb')
    for ax, (m, label) in zip(axes.flat, PANELS):
        style(ax)
        ax.axvspan(0.5, n_k - 0.5, color='#f0efec', zorder=0)
        ax.axvline(n_k, color=MUTED, linewidth=1, linestyle=':')
        ends = [rows_a[-1][m], rows_b[-1][m]]
        for i, (name, rows, key, marker) in enumerate(((a, rows_a, 'a', 'o'), (b, rows_b, 'b', 's'))):
            pts = [(r['n'], r[m]) for r in rows if r[m] is not None]
            ax.plot(*zip(*pts), color=COLORS[key], linewidth=2, marker=marker, markersize=4, label=name)
            up = ends[i] > ends[1 - i] or (ends[i] == ends[1 - i] and i == 0)  # the higher end's label above
            ax.annotate(name, pts[-1], xytext=(5, 6 if up else -6), textcoords='offset points', color=INK, fontsize=8, va='center')
        ax.set_title(label, color=INK, fontsize=10, loc='left')
        ax.set_xlabel('placement n', color=MUTED, fontsize=8)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f'{title}: shaded = shared prefix, dotted = the decision placement n = {n_k}', color=INK, fontsize=11, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_lr(path, steps, pairs):
    plt = pyplot()
    fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 3.6), facecolor='#fcfcfb', sharey=True)
    for ax, (key, (a, b, _, n_k)) in zip(axes, pairs.items()):
        style(ax)
        ax.axvspan(0.5, n_k - 0.5, color='#f0efec', zorder=0)
        ax.axvline(n_k, color=MUTED, linewidth=1, linestyle=':')
        for name, c, marker in ((a, 'a', 'o'), (b, 'b', 's')):
            ax.plot([r['n'] for r in steps[name]], [r['F_LR'] for r in steps[name]], color=COLORS[c], linewidth=2, marker=marker, markersize=4, label=name)
        ax.set_title(f'{key}: F_LR of the partial wall', color=INK, fontsize=10, loc='left')
        ax.set_xlabel('placement n', color=MUTED, fontsize=8)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    if OUT.exists():
        raise SystemExit(f'{OUT} exists: the analysis is written once')
    summary = {r['wall']: r for r in json.loads((LEGACY_RUNS / CF / 'summary.json').read_text(encoding='utf-8'))['rows']}
    summary['GPT'], summary['Paper'] = summary['V'], summary['P6']  # V == gpt-001 is checked below
    steps, problems = {}, {}
    for name in WALLS:
        steps[name] = wall_steps(name)
        lam = metrics.mechanism_lambdas(result_of(WALLS[name]))
        problems[name] = wall_problems(steps[name], {**summary[name], **lam})
        print(name, 'OK' if not problems[name] else problems[name][:5], flush=True)
    strip = lambda rows: [{k: v for k, v in r.items() if k != 'wall'} for r in rows]  # noqa: E731
    if strip(steps['V']) != strip(steps['GPT']):
        problems['V'].append('the step series of V is not gpt-001\'s')
    pairs = {}
    for key, (a, b, k) in PAIRS.items():
        n_k = next(r['n'] for r in steps[a] if r['step'] == k)
        if next(r['n'] for r in steps[b] if r['step'] == k) != n_k:
            raise SystemExit(f'{key}: step {k} is not the same placement in both walls')
        d = pair_rows(steps[a], steps[b], n_k)
        if key != 'GPT-Paper':
            problems[key] = [f"n {r['n']}: {m} differs in the shared prefix" for r in d if r['phase'] == 'prefix'
                             for m in METRICS if r[f'D_{m}'] != 0]
        pairs[key] = (a, b, d, n_k)
    if any(problems.values()):
        raise SystemExit(f'checks failed, nothing written: {json.dumps({k: v for k, v in problems.items() if v}, indent=1)}')
    OUT.mkdir(parents=True)
    write_csv(OUT / 'steps.csv', [r for name in WALLS for r in steps[name]])
    write_csv(OUT / 'pairs.csv', [{'pair': key, 'a': a, 'b': b, **r} for key, (a, b, d, _) in pairs.items() for r in d])
    for key, (a, b, d, n_k) in pairs.items():
        plot_pair(OUT / f'pair_{key}.png', f'{key} ({a} - {b})', a, b, steps[a], steps[b], n_k)
    plot_lr(OUT / 'f_lr.png', steps, pairs)
    out = {'walls': WALLS, 'pairs': {key: {'a': a, 'b': b, 'decision_n': n_k, 'metrics': pair_summary(d, n_k)}
                                     for key, (a, b, d, n_k) in pairs.items()},
           'checks': {'V_equals_gpt001': True, 'final_equals_wang_quality_v1': True, 'llc_every_placement': True,
                      'prefix_identical': True},
           'definitions': __doc__}
    (OUT / 'summary.json').write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(out['pairs'], indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
