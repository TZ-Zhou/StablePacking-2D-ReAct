"""Exploratory arm x seed comparison of a summarized campaign (read-only).

Usage: python -B -m research.analysis.arm_comparison <campaign id> [--reference code_original]

Reads research/runs/campaigns/<id>/manifest.csv (campaign.py summarize). One wall per
(arm, seed) cell. Per metric, each reported separately (never merged into one score):
mean/SD/min/max per arm; the share of the total sum of squares explained by arm and by seed
(two-way, no interaction, so the residual also holds the arm x seed interaction); Friedman
test across arms with seeds as blocks; paired differences arm - reference on the same seed
with a bootstrap 95% CI of the mean, the number of seeds where the arm is better and a
Wilcoxon signed-rank p. p values are uncorrected for multiple comparisons; with few seeds
treat everything as exploratory. 'complete' = all input stones placed (n_placed == n_input).
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from research.walls import campaign

METRICS = (('wall_density', 'higher'), ('wall_height', 'lower'), ('lambda_min', 'higher'), ('complete', 'higher'),
           ('stone_filling', 'higher'), ('n_placed', 'higher'))


def table(manifest, metric):
    """Seeds x arms, arms in manifest order."""
    arms = list(dict.fromkeys(manifest.arm))
    return manifest.pivot(index='seed', columns='arm', values=metric)[arms]


def variance_shares(t):
    x = t.to_numpy(dtype=float)
    g = x.mean()
    total = ((x - g) ** 2).sum()
    arm = x.shape[0] * ((x.mean(axis=0) - g) ** 2).sum()
    seed = x.shape[1] * ((x.mean(axis=1) - g) ** 2).sum()
    if np.ptp(x) == 0:  # constant metric
        return {'arm': np.nan, 'seed': np.nan, 'residual': np.nan}
    return {'arm': arm / total, 'seed': seed / total, 'residual': (total - arm - seed) / total}


def paired(t, arm, reference, better, rng, n_boot=5000):
    d = (t[arm] - t[reference]).to_numpy(dtype=float)
    boot = rng.choice(d, (n_boot, len(d))).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {'arm': arm, 'mean_diff': d.mean(), 'sd_diff': d.std(ddof=1), 'ci_low': lo, 'ci_high': hi,
            'better_in': int((d > 0).sum() if better == 'higher' else (d < 0).sum()), 'worse_in': int(
                (d < 0).sum() if better == 'higher' else (d > 0).sum()), 'n_seeds': len(d),
            'wilcoxon_p': stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0}


def best_per_seed(t, better):
    """Arm with the best value on each seed; 'tie' when several arms share it."""
    best = t.max(axis=1) if better == 'higher' else t.min(axis=1)
    winners = t.eq(best, axis=0)
    return winners.apply(lambda r: r.idxmax() if r.sum() == 1 else 'tie', axis=1).value_counts()


def compare(manifest, reference='code_original', n_boot=5000, seed=0):
    manifest = manifest.assign(complete=(manifest.n_placed == manifest.n_input).astype(float))
    rng = np.random.default_rng(seed)
    out = {}
    for metric, better in METRICS:
        t = table(manifest, metric)
        others = [a for a in t.columns if a != reference]
        varying = t.nunique().max() > 1
        out[metric] = {
            'better': better,
            'per_arm': pd.DataFrame({'mean': t.mean(), 'sd': t.std(ddof=1), 'min': t.min(), 'max': t.max()}),
            'shares': variance_shares(t),
            'friedman_p': stats.friedmanchisquare(*[t[a] for a in t.columns]).pvalue
            if varying and metric != 'complete' and t.shape[1] >= 3 else np.nan,
            'paired': pd.DataFrame([paired(t, a, reference, better, rng, n_boot) for a in others]),
            'best_per_seed': best_per_seed(t, better).to_dict(),
        }
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('id')
    parser.add_argument('--reference', default='code_original')
    args = parser.parse_args(argv)
    manifest = pd.read_csv(campaign.CAMPAIGNS / args.id / 'manifest.csv')
    pd.set_option('display.width', 250)
    print('walls', len(manifest), '| status', manifest.status.value_counts().to_dict(),
          '| checks_ok', manifest.checks_ok.value_counts().to_dict())
    print('termination', manifest.groupby('arm').termination.value_counts().unstack(fill_value=0).to_dict('index'))
    for metric, r in compare(manifest, args.reference).items():
        s = r['shares']
        print(f"\n=== {metric} ({r['better']} is better)")
        print(r['per_arm'].round(4).T.to_string())
        print(f"share of sum of squares: arm {s['arm']:.2f}, seed {s['seed']:.2f}, residual {s['residual']:.2f}"
              f" | Friedman p {r['friedman_p']:.3f}")
        print(r['paired'].round(4).to_string(index=False))
        print('best arm per seed:', r['best_per_seed'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
