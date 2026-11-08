"""Which filter orders choose the same stone in the same wall state (offline, no wall is built).

Usage: python -B -m research.analysis.order_agreement <output dir> <sensitivity dir> [<sensitivity dir> ...]

Inputs are order_sensitivity.py outputs (orders.csv, events.csv): for every recorded filter event,
the choice each of the 120 paper-cascade orders makes on the identical candidate matrix, plus
the code_original choice. Events where no order can choose (no feasible candidate) are dropped.
Each event counts once; states come from the walls that were built, so they over-represent
the trajectories of the arms that built them.
Writes agreement_stone.csv / agreement_pose.csv (pairwise fraction of events with the same
choice), orders.csv (per order: agreement with the per-event majority, cluster), clusters.csv,
prefix.csv (mean pairwise agreement by number of shared leading filters) and summary.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

FILTERS = {0: 'NHR', 1: 'GDS', 2: 'IWR', 3: 'LLC', 4: 'LFR'}
CUTS = (0.05, 0.10, 0.20)  # average-linkage distance = 1 - stone agreement


def load(dirs):
    orders, events = [], []
    for d in dirs:
        orders.append(pd.read_csv(Path(d) / 'orders.csv', dtype={'order': str}))
        events.append(pd.read_csv(Path(d) / 'events.csv'))
    orders, events = pd.concat(orders), pd.concat(events)
    orders['key'] = orders['run_id'] + '#' + orders['event'].astype(str)
    events['key'] = events['run_id'] + '#' + events['event'].astype(str)
    stone = orders.pivot(index='key', columns='order', values='stone_id')
    pose = orders.pivot(index='key', columns='order', values='pose_id')
    ev = events.set_index('key').loc[stone.index]
    pose['code_original'] = ev['code_original_pose']
    stone['code_original'] = ev['code_original_pose'].map(lambda p: int(p.split('@')[0]) if isinstance(p, str) else np.nan)
    keep = stone.drop(columns='code_original').notna().any(axis=1)
    return stone[keep], pose[keep]


def agreement(choice):
    """Pairwise fraction of events where two columns made the same choice."""
    x = choice.to_numpy(dtype=object)
    n = x.shape[1]
    out = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = out[j, i] = np.mean(x[:, i] == x[:, j])
    return pd.DataFrame(out, index=choice.columns, columns=choice.columns)


def shared_prefix(a, b):
    a, b = a.split('-'), b.split('-')
    return next((k for k in range(5) if a[k] != b[k]), 5)


def analyze(stone, pose):
    paper = [c for c in stone.columns if c != 'code_original']
    a_stone, a_pose = agreement(stone), agreement(pose)
    majority = stone[paper].mode(axis=1)[0]
    per_order = pd.DataFrame({
        'order': stone.columns,
        'names': ['code_original' if c == 'code_original' else '>'.join(FILTERS[int(f)] for f in c.split('-'))
                  for c in stone.columns],
        'same_stone_as_majority': [(stone[c] == majority).mean() for c in stone.columns],
        'same_stone_as_code_original': a_stone['code_original'].to_numpy(),
        'same_stone_as_paper': a_stone['3-0-2-1-4'].to_numpy(),
        'mean_stone_agreement': (a_stone.sum(axis=1).to_numpy() - 1) / (len(stone.columns) - 1),
    })
    z = linkage(squareform(1 - a_stone.loc[paper, paper].to_numpy(), checks=False), 'average')
    for cut in CUTS:
        per_order.loc[per_order.order.isin(paper), f'cluster_{cut}'] = fcluster(z, cut, 'distance')
    prefix = [(shared_prefix(a, b), a_stone.at[a, b], a_pose.at[a, b])
              for i, a in enumerate(paper) for b in paper[i + 1:]]
    prefix = (pd.DataFrame(prefix, columns=['shared_leading_filters', 'stone_agreement', 'pose_agreement'])
              .groupby('shared_leading_filters').agg(['mean', 'min', 'count']))
    contested = stone[paper].nunique(axis=1)
    return a_stone, a_pose, per_order, prefix, contested


def cluster_table(per_order, a_stone, cut):
    rows = []
    for c, g in per_order.dropna(subset=[f'cluster_{cut}']).groupby(f'cluster_{cut}'):
        members = list(g.order)
        sub = a_stone.loc[members, members].to_numpy()
        leads = g.order.str[:3].value_counts()
        rows.append({'cut': cut, 'cluster': int(c), 'size': len(members),
                     'within_min_agreement': sub.min(), 'leading_pairs': dict(leads),
                     'first_filter': dict(g.order.str[0].map(lambda f: FILTERS[int(f)]).value_counts()),
                     'mean_same_as_majority': g.same_stone_as_majority.mean(),
                     'pilot_arms': [o for o in members if o in ('3-0-2-1-4', '2-4-3-1-0', '1-3-4-2-0')],
                     'members': ' '.join(members)})
    return pd.DataFrame(rows).sort_values('size', ascending=False)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    out, dirs = Path(argv[0]), argv[1:]
    out.mkdir(parents=True, exist_ok=False)
    stone, pose = load(dirs)
    a_stone, a_pose, per_order, prefix, contested = analyze(stone, pose)
    clusters = pd.concat([cluster_table(per_order, a_stone, cut) for cut in CUTS])
    a_stone.to_csv(out / 'agreement_stone.csv')
    a_pose.to_csv(out / 'agreement_pose.csv')
    per_order.sort_values('same_stone_as_majority').to_csv(out / 'orders.csv', index=False)
    clusters.to_csv(out / 'clusters.csv', index=False)
    prefix.to_csv(out / 'prefix.csv')
    summary = {'inputs': dirs, 'events': len(stone),
               'events_by_distinct_stones_over_120_orders': contested.value_counts().sort_index().to_dict(),
               'clusters_per_cut': {str(c): int(per_order[f'cluster_{c}'].nunique()) for c in CUTS}}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=int), encoding='utf-8')
    print(json.dumps(summary, indent=2, default=int))


if __name__ == '__main__':
    main()
