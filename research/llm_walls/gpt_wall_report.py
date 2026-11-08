"""Offline report of a live GPT wall beside the development walls of its seed.

  python -B -m research.llm_walls.gpt_wall_report gpt-001

Writes research/reports/<name>/ once (refuses an existing one): decisions.csv, switches.json,
trajectory.csv, quality.csv, summary.json. Reads the run records only: never writes them, builds
no wall, calls no API.

- decisions.csv: per decision point, the candidate GPT's order selects and the one the Paper order
  selects in the same state (Wang's select with PAPER_ORDER on that event's traced scores), real
  stone ids and poses, and the category: 'paper kept' (Paper order), 'order differs, same
  candidate', 'candidate changed'; tools, HTTP attempts, tokens and cost of the decision.
- switches.json: every changed candidate with the wall state GPT saw, the five filter scores of
  both candidates (as the list_order_outcomes tool reported them), the tools and GPT's rationale.
- trajectory.csv: the GPT wall and the Paper wall of the seed step by step (stone, pose, sample, the
  LLC score of the selected candidate);
  states are the same only up to the first changed candidate.
- quality.csv: wang-quality-v1 of the GPT wall (quality_batch.evaluate) beside the dev walls'
  research/quality/dev-2seeds rows, with lambda_left/right (metrics.mechanism_lambdas).
- summary.json: execution summary (journal, receipt, runner report) and the information checks:
  controllers.check (each record re-executed on its own event's trace) and no quality term in
  any request sent.
Every decision row uses its own event only; final quality appears only in quality.csv.
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

from research.controller import llm_api as llm_adapter
from research.controller import order_controller as controllers
from research.paths import REPO
from research.walls import campaign_quality as quality_batch
from research.walls import wall_metrics as metrics
from Stonepacker2D.stack.selection import PAPER_ORDER, load_trace, select

FILTERS = {0: 'NHR', 1: 'GDS', 2: 'IWR', 3: 'LLC', 4: 'LFR'}  # selection.PAPER_ORDER: LLC, NHR, IWR, GDS, LFR
# the final-wall metrics; not 'tilting': the fixed system prompt defines the LLC filter as the tilting-table
# limit load of each candidate's partial wall, a decision-time score of Wang's cascade
QUALITY_TERMS = re.compile(r'F_SF|F_AH|F_AV|F_LR|lambda_min|mechanism|wang-quality|stone_filling|placed_ratio', re.I)
DEV = 'campaigns/dev-2seeds/thangavelu-noise80/{arm}/seed{seed}'
KEYS = ('F_SF', 'F_AH', 'F_AV', 'F_LR', 'placed_ratio', 'n_placed', 'n_input', 'lambda_min', 'wall_height', 'wall_density')


def names(order):
    return '>'.join(FILTERS[int(c)] for c in order.split('-'))


def pose(a, i):
    return {k: int(a[k][i]) for k in ('stone_id', 'rotation', 'x', 'y')}


def decision_rows(result, records, prices):
    rows, switches = [], []
    for r in records:
        if not r['decision_point']:
            continue
        meta, a = load_trace(result / 'trace' / f"e{r['event']:05d}.npz")
        paper = select(a['features_raw'], a['weights'], 'paper', PAPER_ORDER).index
        mine = r['final']['index']
        assert select(a['features_raw'], a['weights'], 'paper', tuple(map(int, r['final']['order'].split('-')))).index == mine, r['event']
        listing = next((s['feedback']['outcomes'] for s in r.get('react') or [] if s.get('tool') == 'list_order_outcomes'), [])
        shown = {o['id']: o for o in listing}
        chosen, base = shown.get(r['final']['outcome']), next((o for o in listing if o['is_paper']), None)
        use = [t['response'].get('usage') for t in r.get('transcripts') or []]
        ok = [u for u in use if llm_adapter.usage_ok(u)]
        commit = next((s for s in r.get('react') or [] if s.get('tool') == 'commit_order'), {})
        category = 'paper kept' if r['final']['order'] == '3-0-2-1-4' else \
            'order differs, same candidate' if mine == paper else 'candidate changed'
        assert (mine != paper) == r['final']['switched'], r['event']  # the record's switched is this comparison
        row = {'step': r['step'], 'event': r['event'], 'side': r['side'], 'n_remaining': r['observation']['n_remaining'],
               'n_outcomes': r['n_outcomes'], 'gpt_order': r['final']['order'], 'gpt_order_filters': names(r['final']['order']),
               'category': category, 'switched': r['final']['switched'],
               **{f'gpt_{k}': v for k, v in pose(a, mine).items()}, **{f'paper_{k}': v for k, v in pose(a, paper).items()},
               'gpt_label': chosen and chosen['stone'], 'paper_label': base and base['stone'],
               'tools': '+'.join(s.get('tool') for s in r.get('react') or []),
               'http': sum(len(t['response'].get('attempts') or []) for t in r.get('transcripts') or []),
               'input_tokens': sum(u['input_tokens'] + u['cache_creation_input_tokens'] + u['cache_read_input_tokens'] for u in ok),
               'cached_tokens': sum(u['cache_read_input_tokens'] for u in ok), 'output_tokens': sum(u['output_tokens'] for u in ok),
               'cost_usd': round(sum(llm_adapter.cost(prices, u) for u in ok), 6), 'unknown_usage': len(use) - len(ok),
               'latency_s': round(sum(t.get('latency_s') or 0 for t in r.get('transcripts') or []), 2)}
        rows.append(row)
        if category == 'candidate changed':
            switches.append({'step': r['step'], 'event': r['event'], 'n_remaining': row['n_remaining'], 'n_outcomes': r['n_outcomes'],
                             'wall': r['observation']['wall'], 'now': r['observation']['now'],
                             'gpt': {'order': r['final']['order'], 'filters': row['gpt_order_filters'], **pose(a, mine),
                                     'label': row['gpt_label'], 'scores': chosen and chosen['scores'], 'geometry': chosen and chosen.get('geometry')},
                             'paper': {'order': '3-0-2-1-4', 'filters': names('3-0-2-1-4'), **pose(a, paper), 'label': row['paper_label'],
                                       'scores': base and base['scores'], 'geometry': base and base.get('geometry')},
                             'tools': [{'tool': s.get('tool'), 'arguments': s.get('arguments'), 'status': s.get('status')} for s in r['react']],
                             'rationale': (commit.get('arguments') or {}).get('rationale')})
    return rows, switches


def trajectory(gpt_result, paper_result):
    def steps(result):  # each commit with the LLC score of its selected candidate (a tilting-table limit of the partial wall)
        out = []
        for c in metrics.read_jsonl(result / 'trace' / 'commits.jsonl'):
            side = c['sides'][c['chosen_side']]
            a = load_trace(result / 'trace' / f"e{side['event']:05d}.npz")[1]
            out.append({**c, 'llc': round(float(a['features_raw'][side['index'], 3]), 6)})
        return out
    rows = []
    for g, p in zip(steps(gpt_result), steps(paper_result)):
        rows.append({'step': g['step'], **{f'gpt_{k}': g.get(k) for k in ('stone_id', 'rotation_world', 'x_world', 'y', 'outcome', 'llc')},
                     'gpt_sampled': ' '.join(map(str, g.get('sampled_ids') or [])),
                     **{f'paper_{k}': p.get(k) for k in ('stone_id', 'rotation_world', 'x_world', 'y', 'outcome', 'llc')},
                     'paper_sampled': ' '.join(map(str, p.get('sampled_ids') or [])),
                     'same_placement': all(g.get(k) == p.get(k) for k in ('stone_id', 'rotation_world', 'x_world', 'y')),
                     'same_sample': g.get('sampled_ids') == p.get('sampled_ids')})
    return rows


def quality(run, seed):
    row, _ = quality_batch.evaluate(run)
    out = [{'arm': 'gpt', **{k: row.get(k) for k in KEYS}, 'status': row['status']}]
    for d in csv.DictReader(open(REPO / 'research' / 'quality' / 'dev-2seeds' / 'quality.csv', encoding='utf-8')):
        if d['seed'] == str(seed):
            out.append({'arm': d['arm'], **{k: d[k] for k in KEYS}, 'status': d['status']})
    for o in out:
        run_dir = run if o['arm'] == 'gpt' else quality_batch.RUNS / DEV.format(arm=o['arm'], seed=seed)
        lam = metrics.mechanism_lambdas(next(run_dir.glob('result_*')))
        o.update(lambda_left=lam['left'], lambda_right=lam['right'])
    return out


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main(name):
    run = quality_batch.RUNS / 'gpt-walls' / name
    out = REPO / 'research' / 'reports' / name
    if out.exists():
        raise SystemExit(f'{out} exists: a report is written once')
    result = next(run.glob('result_*'))
    receipt = json.loads((run / 'receipt.json').read_text(encoding='utf-8'))
    journal = json.loads((run.parent / 'journals' / f'{name}.json').read_text(encoding='utf-8'))
    runner = json.loads((run / 'report.json').read_text(encoding='utf-8'))
    records = [json.loads(l) for l in (result / 'trace' / 'controller.jsonl').read_text(encoding='utf-8').splitlines()]
    seed = journal['seed']
    rows, switches = decision_rows(result, records, llm_adapter.price(receipt['controller']['llm']))
    traj = trajectory(result, next((quality_batch.RUNS / DEV.format(arm='paper', seed=seed)).glob('result_*')))
    check = controllers.check(result)
    leaks = [(r['event'], t['call']) for r in records for t in r.get('transcripts') or []
             if QUALITY_TERMS.search(json.dumps(t.get('request'), ensure_ascii=False))]
    first = next((t['step'] for t in traj if not t['same_placement']), None)
    summary = {
        'run': run.relative_to(REPO).as_posix(), 'seed': seed, 'git_head': journal['git_head'], 'journal_state': journal['state'],
        'journal_reasons': journal['reasons'], 'status': receipt['status'], 'elapsed_seconds': receipt['elapsed_seconds'],
        'started_at': journal['started_at'], 'ended_at': journal['ended_at'], 'model': receipt['controller']['llm']['model'],
        'observation_version': receipt['controller'].get('observation_version'),
        **{k: runner[k] for k in ('http_requests', 'http_requests_exact', 'tokens', 'cost_usd', 'unknown_usage', 'workers_reaped',
                                  'decision_points', 'llm_decisions', 'switches', 'gpt_complete', 'key_scan')},
        'categories': {c: sum(r['category'] == c for r in rows) for c in ('paper kept', 'order differs, same candidate', 'candidate changed')},
        'tool_sequences': {s: sum(r['tools'] == s for r in rows) for s in sorted({r['tools'] for r in rows})},
        'first_different_placement_vs_paper_wall': first,
        'steps_with_same_sample_vs_paper_wall': [t['step'] for t in traj if t['same_sample']],
        'information_checks': {'controller_check': check['verdict'], 'check_problems': check['problems'],
                               'requests_with_quality_terms': leaks,
                               'requests_naming_tilting': sum('tilting' in json.dumps(t.get('request')) for r in records
                                                              for t in r.get('transcripts') or []),
                               'note': 'check re-executes every record on its own event trace (ReplayLLM); a request built '
                                       'from anything else would not replay. Decision rows use one event each. tilting: the '
                                       'fixed system prompt defines LLC (a decision-time candidate score), not a final metric.'}}
    out.mkdir(parents=True)
    write_csv(out / 'decisions.csv', rows)
    write_csv(out / 'trajectory.csv', traj)
    write_csv(out / 'quality.csv', quality(run, seed))
    (out / 'switches.json').write_text(json.dumps(switches, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main(sys.argv[1])
