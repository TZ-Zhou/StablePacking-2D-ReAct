"""Batch wall quality, method wang-quality-v1: for every run of a campaign, the four metrics of
Wang et al. by wall_metrics.wall_quality on the corrected canvas (F_SF, F_AH, F_AV, F_LR: the formal
values) and on the author-compatible canvas (F_*_author: the reference), beside placed_ratio,
wall_density and wall_height of wall_metrics.wall_metrics. Raw values: no weights, no total score.

  python -B -m research.walls.campaign_quality <campaign id> [--method wang-quality-v1] [--out DIR]

--method wang-quality-s1-keep-all (or -earlier) is the sensitivity analysis of 02's overlap rule
(see wall_metrics.render_original leave_out, overlap); its output goes to research/quality/<method>.

Writes <out>/batch.json, quality.csv and quality.jsonl once; refuses an existing <out> or one
inside research/runs or research/cases, which it only reads. The campaign may be a study or one of
the research/runs archive (campaign.load, campaign.manifest_path); the runs are its manifest.csv rows,
which must be its arms x seeds exactly, each run a case id or a research/runs path
(case_id.wall_dir). A run that fails an integrity check, whose evaluation raises
or leaves a metric undefined gets a FAIL row naming why; nothing is skipped, and the exit code
is 1 when any row failed. Refuses to start when a source file differs from git HEAD, and to
write when one changed during the batch. The output holds no time and no absolute path, so a
rerun writes the same bytes.
"""
import argparse
import csv
import hashlib
import io
import json
import pickle
import platform
import re
import subprocess
import sys
from pathlib import Path

from research.paths import CASES_DIR, LEGACY_RUNS, QUALITY_DIR, REPO
from research.storage import case_id as cid
from research.walls import campaign as camp
from research.walls import check_run
from research.walls import wall_metrics

METHOD = 'wang-quality-v1'  # the formal method
# method: wall_quality options. The s1 methods are a sensitivity analysis of 02's overlap rule
# (every stone drawn), never the formal values.
METHODS = {METHOD: {}, 'wang-quality-s1-keep-all': {'leave_out': False},
           'wang-quality-s1-keep-all-earlier': {'leave_out': False, 'overlap': 'earlier'}}
SOURCES = ['research/walls/campaign_quality.py', 'research/walls/wall_metrics.py', 'research/walls/check_run.py',
           'Tests/Thangavelu/02_view_transformation_ori_reso.py', 'Tests/Thangavelu/03_evaluate_wall_metrics.py',
           'Tests/Thangavelu/graph.py', 'Stonepacker2D']
COLUMNS = ['campaign', 'arm', 'policy', 'order', 'seed', 'run', 'status', 'problems',
           'F_SF', 'F_AH', 'F_AV', 'F_LR', 'F_SF_author', 'F_AH_author', 'F_AV_author', 'F_LR_author',
           'lambda_min', 'mu', 'placed_ratio', 'n_placed', 'n_input', 'wall_density', 'wall_height',
           'restored_px', 'n_stones_restored', 'outside_wall_px', 'n_left_out', 'n_left_out_author',
           'method']
FORMAL = ('F_SF', 'F_AH', 'F_AV', 'F_LR')


def git(*args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def sources(paths=SOURCES):
    """Git blob hash at HEAD of every source file (a folder: of its tracked files), and the
    paths whose working copy differs from HEAD or is untracked (the hashes hold for the files
    evaluated only when that list is empty)."""
    hashes = {line.split('\t')[1]: line.split()[2] for line in git('ls-tree', '-r', 'HEAD', '--', *paths).splitlines()}
    dirty = [line[3:] for line in git('status', '--porcelain', '--untracked-files=all', '--', *paths).splitlines()]
    return hashes, dirty


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def problems(run):
    """Integrity of one recorded wall before it is evaluated; [] when it may be."""
    if not (run / 'receipt.json').exists():
        return ['no receipt.json']
    receipt = json.loads((run / 'receipt.json').read_text(encoding='utf-8'))
    p = []
    if receipt.get('case') != 'thangavelu-noise80':
        p.append(f"case {receipt.get('case')}")
    if receipt.get('status') != 'PROCESS_PASS':
        p.append(f"status {receipt.get('status')}")
    results = sorted(run.glob('result_*'))
    if len(results) != 1:
        return p + [f'{len(results)} result directories']
    if not (run / '00_run.py').exists():
        return p + ['no 00_run.py']
    if sha256(run / '00_run.py') != receipt.get('staged_sha256'):
        p.append('00_run.py differs from the staged source the receipt names')
    try:
        source = wall_metrics.stone_source(run)
        if sha256(source) != sha256(REPO / 'Tests' / 'Thangavelu' / 'data' / source.name):
            p.append(f'{source.name} differs from Tests/Thangavelu/data')
    except (ValueError, FileNotFoundError) as e:
        p.append(f'stone source: {e}')
    res = results[0]
    missing = [f for f in ('base.pkl', 'transformation.txt') if not (res / f).exists()]
    p += [f'missing {f}' for f in missing]
    if not missing:
        rocks = [r.id for r in pickle.load(open(res / 'base.pkl', 'rb')).placed_rocks]
        ids = [int(line.split(';')[0]) for line in (res / 'transformation.txt').read_text().splitlines()[1:] if line.strip()]
        if ids != rocks:
            p.append('transformation.txt does not list the placed rocks of base.pkl')
    if (res / 'mechanism.csv').exists():
        p += [f'{side}: no single limit load' for side, value in wall_metrics.mechanism_lambdas(res).items() if value is None]
    else:
        for side in ('left', 'right'):
            names = [x.name for x in (res / 'img').glob(f'{side}*_mechanism.png')]
            if len(names) != 1 or not re.fullmatch(rf'{side}[0-9.eE+-]+_mechanism\.png', names[0]):
                p.append(f'{side} mechanism images: {names}')
    try:
        check = check_run.check_run(run)
        if not check['ok']:
            p.append(f"check_run: {len(check['problems'])} problems, first: {check['problems'][0]}")
    except Exception as e:  # an unreadable record fails, it is not skipped
        p.append(f'check_run raised {type(e).__name__}: {e}')
    return p


def evaluate(run, method=METHOD):
    """(csv fields, jsonl detail) of one wall; status FAIL with its problems when it cannot be."""
    if method not in METHODS:
        raise ValueError(f'unknown method {method}')
    p = problems(run)
    if p:
        return {'status': 'FAIL', 'problems': p}, {}
    result = next(run.glob('result_*'))
    try:
        qc = wall_metrics.wall_quality(run, result, 'corrected', **METHODS[method])
        qa = wall_metrics.wall_quality(run, result, 'author', **METHODS[method])
        wm = wall_metrics.wall_metrics(run, result)
    except Exception as e:
        return {'status': 'FAIL', 'problems': [f'evaluation raised {type(e).__name__}: {e}']}, {}
    stones = qc['render']['stones']
    row = {k: qc[k] for k in FORMAL}
    row.update({f'{k}_author': qa[k] for k in FORMAL})
    row.update({k: wm[k] for k in ('placed_ratio', 'n_placed', 'n_input', 'wall_density', 'wall_height')},
               lambda_min=qc['lambda_min'], mu=qc['mu'],
               restored_px=sum(s['canvas_cut'] for s in stones),
               n_stones_restored=sum(s['canvas_cut'] > 0 for s in stones),
               outside_wall_px=sum(s['outside_wall'] for s in stones),
               n_left_out=sum(s['left_out'] for s in stones),
               n_left_out_author=sum(s['left_out'] for s in qa['render']['stones']))
    p = [f'{k} undefined' for k in COLUMNS[8:23] if row.get(k) is None]
    p += [f"stone {s['id']} touches its canvas after the turn" for s in stones if s['turn_at_edge']]
    row.update(status='FAIL' if p else 'OK', problems=p)
    return row, {'corrected': qc, 'author': qa, 'wall_metrics': wm}


def batch(campaign_id, out, check_sources=True, method=METHOD):
    """Evaluate the campaign and write <out>; returns the number of FAIL rows."""
    if method not in METHODS:
        raise ValueError(f'unknown method {method}')
    out = Path(out).resolve()
    if out.exists():
        raise SystemExit(f'{out} exists: a batch is written once')
    for records in (LEGACY_RUNS.resolve(), CASES_DIR.resolve()):
        if out == records or records in out.parents:
            raise SystemExit(f'{out} is inside {records}: the wall records are only read')
    hashes, dirty = sources()
    if check_sources and dirty:
        raise SystemExit(f'source files differ from HEAD: {dirty}')
    campaign = camp.load(campaign_id)
    with open(camp.manifest_path(campaign_id), encoding='utf-8') as f:
        manifest = list(csv.DictReader(f))
    planned = {(a['name'], str(s)) for a in campaign['arms'] for s in campaign['seeds']}
    listed = [(m['arm'], m['seed']) for m in manifest]
    if sorted(listed) != sorted(planned):
        raise SystemExit(f'manifest.csv does not list the campaign arms x seeds once each: {len(listed)} rows, {len(planned)} planned')
    rows, details = [], []
    for m in manifest:
        row, detail = evaluate(cid.wall_dir(m['run']), method)
        row = {**{k: m[k] for k in ('arm', 'policy', 'order', 'seed', 'run')}, 'campaign': campaign_id, 'method': method, **row}
        rows.append(row)
        details.append({'run': m['run'], 'arm': m['arm'], 'seed': m['seed'], 'status': row['status'],
                        'problems': row['problems'], **detail})
    if sources() != (hashes, dirty):
        raise SystemExit('a source file changed during the batch: nothing written')
    failed = [r for r in rows if r['status'] == 'FAIL']
    info = {
        'method': method, 'options': METHODS[method], 'campaign': campaign_id, 'formal': 'corrected canvas', 'reference': 'author canvas (*_author)',
        'git_head': git('rev-parse', 'HEAD').strip(), 'sources_git_blob': hashes, 'sources_dirty': dirty,
        'resolution': {'rescale': wall_metrics.RESCALE, 'mu': wall_metrics.MU, 'stone_images': sorted({
            f"{source.name} {sha256(source)}"
            for source in (wall_metrics.stone_source(cid.wall_dir(r['run'])) for r in rows if r['status'] == 'OK')})},
        'environment': {'python': sys.version.split()[0], 'platform': platform.platform(),
                        **{m: __import__(m).__version__ for m in ('numpy', 'scipy', 'cv2', 'skimage', 'networkx')}},
        'n_runs': len(rows), 'n_ok': len(rows) - len(failed),
        'failed': [{'run': r['run'], 'arm': r['arm'], 'seed': r['seed'], 'problems': r['problems']} for r in failed],
        'columns': COLUMNS,
    }
    text = io.StringIO(newline='')
    w = csv.DictWriter(text, COLUMNS, lineterminator='\n')
    w.writeheader()
    for r in rows:
        w.writerow({**r, 'problems': '; '.join(r['problems'])})
    out.mkdir(parents=True)
    (out / 'batch.json').write_text(json.dumps(info, indent=1, sort_keys=True) + '\n', encoding='utf-8', newline='\n')
    (out / 'quality.csv').write_text(text.getvalue(), encoding='utf-8', newline='\n')
    (out / 'quality.jsonl').write_text(''.join(json.dumps(d, sort_keys=True) + '\n' for d in details),
                                       encoding='utf-8', newline='\n')
    return len(failed)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('campaign')
    ap.add_argument('--method', choices=list(METHODS), default=METHOD)
    ap.add_argument('--out', help='default research/quality/<method>')
    a = ap.parse_args(argv)
    n = batch(a.campaign, a.out or QUALITY_DIR / a.method, method=a.method)
    print(f'{n} FAIL rows' if n else 'all rows OK')
    return 1 if n else 0


if __name__ == '__main__':
    sys.exit(main())
