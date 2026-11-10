"""Read-only consistency checks of traced runs, and byte comparison of two runs.

Usage: python -B -m research.walls.check_run <run dir> [--compare <other run dir>]

check_run: trace replay, commits vs chosen candidates, placed rocks in base.pkl vs recorded
world poses (unstabilized rocks must match pixel for pixel), right-side rotations against the
same stone's left-side candidates. compare_runs: identical outputs, base.pkl and trace.
differing_files: every file of the result directories byte for byte (time.txt holds timings and
the PDFs embed creation dates, so those are expected to differ between identical runs).
"""
import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from Stonepacker2D.stack.selection import footprint, load_trace, published_events, select, to_world  # noqa: E402


def result_dirs(run):
    return sorted({p.parent for p in Path(run).rglob('decisions.txt')})


def artifact_files(result):
    """The files of one result directory a run manifest covers: its trace, decisions.txt and base.pkl."""
    trace = Path(result) / 'trace'
    files = sorted(p for p in trace.rglob('*') if p.is_file()) if trace.is_dir() else []
    return files + [p for p in (Path(result) / 'decisions.txt', Path(result) / 'base.pkl') if p.is_file()]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_result(result):
    trace = Path(result) / 'trace'
    events = {}
    for p in published_events(trace):
        meta, arrays = load_trace(p)
        events[meta['event']] = (meta, arrays)
    lines = (trace / 'commits.jsonl').read_text(encoding='utf-8').splitlines()
    return events, [json.loads(line) for line in lines]


def selected(meta, a):
    """(index, weighted GDS) the traced policy selects in an event; (None, -1.0) without a feasible
    candidate, as build_wall's construct_one_step returns it."""
    p = meta['policy']
    s = select(a['features_raw'], a['weights'], p['policy'], p['order'], p['tol'], p.get('pick'))
    return (None, -1.0) if s.index is None else (int(s.index), float(s.weighted[s.index, 1]))


def sides_of(construction_order):
    return {'left', 'right'} if construction_order == 'two_sides' else {'right' if construction_order == 'from_right' else 'left'}


def chosen_side(gds, construction_order):
    """(side, tie) by build_wall's rule from {side: weighted GDS}: right only if strictly greater."""
    if construction_order != 'two_sides':
        return next(iter(sides_of(construction_order))), False
    return ('right' if gds['right'] > gds['left'] else 'left'), gds['right'] == gds['left']


def candidate_pose(a, k):
    return int(a['stone_id'][k]), int(a['x_world'][k]), int(a['y'][k]), int(a['rotation_world'][k]), int(a['rotation'][k])


def world_wall(meta, a):
    """Occupied pixels of an event's wall in the world frame (the right side is traced mirrored)."""
    occupied = np.asarray(a['wall']) != 0
    return occupied if meta['side'] == 'left' else np.fliplr(occupied)


def placement_problems(result, by_event, commits, loose, pending):
    """What the placed commits say against what was added to the wall. A placed commit must report
    add_stone_ok, at least one brick added and an int placed_bbox; base.pkl (dumped after every
    placement) must hold exactly the placed stones in commit order (and the pending attempt's
    stone, if its decisions.txt line was written), each rock with the commit's placed_bbox; between
    the first events of consecutive attempts (the last one against base.pkl) the wall must grow by
    exactly that attempt's placed rock (more only if it added wedges) and by nothing after an
    attempt without a placement, and never lose pixels. That is the committed geometry, after
    rock stabilization; a rock that was not stabilized (placed_rot_angle 0) must also have the
    selected candidate's pixels at its pose. Cut attempts add cut cubes and are not compared."""
    problems = []
    placed = [c for c in commits if c['outcome'] == 'placed']
    for c in placed:
        bbox = c.get('placed_bbox')
        if c.get('add_stone_ok') is not True or type(c.get('bricks_added')) is not int or c['bricks_added'] < 1 or \
                not (isinstance(bbox, list) and len(bbox) == 4 and all(type(v) is int for v in bbox)):
            problems.append(f"placed commit step {c['step']}: add_stone_ok {c.get('add_stone_ok')!r}, "
                            f"bricks_added {c.get('bricks_added')!r}, placed_bbox {bbox!r}")
    if not placed and pending is None:
        return problems
    try:
        with (Path(result) / 'base.pkl').open('rb') as f:
            base = pickle.load(f)
        rocks, final = list(base.placed_rocks), np.asarray(base.matrix) != 0
    except Exception as error:  # no wall to compare with: the placements are unverified
        return problems + [f'base.pkl cannot be read: {type(error).__name__}']
    ids = [c['stone_id'] for c in placed] + ([int(pending[1])] if pending else [])
    if [int(r.id) for r in rocks] != ids:
        return problems + [f'base.pkl holds the rocks {[int(r.id) for r in rocks][:10]}, the placed commits {ids[:10]}']
    rock_of = dict(zip(ids, rocks))
    attempts = [(c, min(s['event'] for s in c['sides'].values())) for c in commits if c['sides']]
    if loose:
        attempts.append((None, loose[0][0]['event']))
    if any(e not in by_event for _, e in attempts):  # already a chain problem; no wall to compare with
        return problems
    walls = [world_wall(*by_event[e]) for _, e in attempts] + [final]
    for k, (c, e) in enumerate(attempts):
        where = f"attempt of event {e}"
        before, after = walls[k], walls[k + 1]
        if before.shape != after.shape or (before & ~after).any():
            problems.append(f'{where}: the wall lost pixels or changed size')
            continue
        growth = after & ~before
        stone = c['stone_id'] if c is not None and c['outcome'] == 'placed' else int(pending[1]) if c is None and pending else None
        if stone is None:
            if growth.any() and (c is None or c['outcome'] != 'cut'):
                problems.append(f'{where}: the wall grew by {int(growth.sum())} pixels without a placement')
            continue
        rock = np.asarray(rock_of[stone].matrix) != 0
        wedges = c is not None and c['bricks_added'] > 1
        if rock.shape != before.shape or (rock & before).any() or (rock & ~growth).any() or (not wedges and (growth & ~rock).any()):
            problems.append(f'{where}: the wall did not grow by the placed rock {stone}')
            continue
        if c is None:
            continue
        rows, cols = np.nonzero(rock)
        if [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())] != c['placed_bbox']:
            problems.append(f"{where}: rock {stone} has the bbox {[int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]}, "
                            f"the commit {c['placed_bbox']}")
        if not c.get('placed_rot_angle'):
            side = c['sides'][c['chosen_side']]
            a, i = by_event[side['event']][1], side['index']
            if footprint(rock, 'left') != a['footprint'][i] or [int(cols.min()), int(rows.min())] != [c['x_world'], c['y']]:
                problems.append(f'{where}: the unstabilized rock {stone} is not the selected candidate at its pose')
    return problems


def chain_problems(result, events, stopped=False):
    """The trace of one result directory as build_wall writes it, per selection: controller record,
    event eNNNNN.npz, and per sampling attempt a decisions.txt line (placed only), then a commit.

    events: (meta, arrays) of the published events in order. Every event must be part of a commit
    and every commit side must name its event, carry that event's step/phase/trial/side/sampled
    stones and hold what the traced policy selects there (index, ok, weighted GDS); the chosen side
    follows build_wall's rule (right only with a strictly greater weighted GDS); the committed pose
    and rotation label are the selected candidate's; decisions.txt lists the placed commits. Each
    event's ids are unique ints, nb_stones sampled stones, candidate i is sampled stone i % n at
    rotation i // n * 90, world poses follow to_world, and sampled plus remaining stones are the
    inventory left by earlier placements. A run stopped by the launcher (stopped) may end only in a
    prefix of that write order: events of one attempt without its commit, then (once all its sides
    are traced) one decisions.txt line with that attempt's chosen candidate, and one unpublished
    eNNNNN.tmp.npz.
    Commits are in the order of their events (each attempt after the previous one, its left side
    before its right), events in step order, and what placed commits report was added to the wall
    (placement_problems, against base.pkl).
    """
    problems, numbers = [], [m['event'] for m, _ in events]
    if numbers != list(range(len(events))):
        problems.append(f'trace events are not numbered 0..{len(events) - 1}')
    if [m['step'] for m, _ in events] != sorted(m['step'] for m, _ in events):
        problems.append('trace events are not in step order')
    trace = Path(result) / 'trace'
    unpublished = sorted(p.name for p in trace.glob('e*.tmp.npz'))
    if unpublished and not (stopped and unpublished == [f'e{len(events):05d}.tmp.npz']):
        problems.append(f'unpublished trace events {unpublished[:3]}')
    path = trace / 'commits.jsonl'
    commits = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    if not path.exists() and not stopped:
        problems.append('no trace/commits.jsonl')
    by_event, referenced = dict(zip(numbers, events)), []
    for c in commits:
        where = f"commit step {c['step']} {c['phase']} trial {c['sample_trial']}"
        order = c['construction_order']
        if set(c['sides']) != sides_of(order):
            problems.append(f'{where}: sides {sorted(c["sides"])} for {order}')
            continue
        picks = {}
        for name, s in c['sides'].items():
            referenced.append(s['event'])
            if s['event'] not in by_event:
                problems.append(f"{where}: references missing event {s['event']}")
                continue
            meta, a = by_event[s['event']]
            if (meta['step'], meta['phase'], meta['sample_trial'], meta['side'], meta['construction_order'], meta['vendor_type']) \
                    != (c['step'], c['phase'], c['sample_trial'], name, order, c['vendor_type']) or a['sampled_ids'].tolist() != c['sampled_ids']:
                problems.append(f"{where} {name}: event {s['event']} is a different selection")
            index, gds = selected(meta, a)
            if index != (None if int(a['selected']) == -1 else int(a['selected'])):
                problems.append(f"{where} {name}: traced pick {int(a['selected'])}, the traced policy selects {index}")
            if (s['index'], s['ok'], s['gds_weighted']) != (index, index is not None, gds):
                problems.append(f"{where} {name}: commit holds {s['index']}, the event selects {index}")
            picks[name] = (index, gds, a)
        if len(picks) != len(c['sides']):
            continue
        chosen, tie = chosen_side({k: v[1] for k, v in picks.items()}, order)
        if (c['chosen_side'], c['side_tie']) != (chosen, tie):
            problems.append(f"{where}: chosen side {c['chosen_side']}, build_wall's rule gives {chosen}")
        index, _, a = picks[chosen]
        if 'stone_id' in c:
            pose = None if index is None else candidate_pose(a, index)
            if pose != (c['stone_id'], c['x_world'], c['y'], c['rotation_world'], c['rotation_label']):
                problems.append(f'{where}: committed pose is not the selected candidate {pose}')
    if len(set(referenced)) != len(referenced):
        problems.append('an event is part of more than one commit side')
    flat = [s['event'] for c in commits for _, s in sorted(c['sides'].items())]
    if flat != list(range(len(flat))):
        problems.append('commits are not in the order of their events (attempt after attempt, left side before right)')
    loose = [by_event[e] for e in sorted(set(numbers) - set(referenced))]
    tail = {(m['step'], m['phase'], m['sample_trial'], m['construction_order']) for m, _ in loose}
    if loose and not (stopped and loose[0][0]['event'] > max(referenced, default=-1) and len(tail) == 1
                      and len({m['side'] for m, _ in loose}) == len(loose)):
        problems.append(f"events {[m['event'] for m, _ in loose][:5]} belong to no commit")

    decisions = Path(result) / 'decisions.txt'
    rows = [l.split(';') for l in decisions.read_text(encoding='utf-8').splitlines()] if decisions.exists() else []
    placed = [c for c in commits if c['outcome'] == 'placed']
    if rows[:len(placed)] != [[str(c[k]) for k in ('step', 'stone_id', 'x_world', 'y', 'rotation_label')] for c in placed]:
        problems.append(f'decisions.txt does not list the {len(placed)} placed commits')
    pending = None  # the decisions.txt line of the unfinished attempt, once all its sides are traced
    if len(tail) == 1 and {m['side'] for m, _ in loose} == sides_of(next(iter(tail))[3]):
        picks = {m['side']: (*selected(m, a), a) for m, a in loose}
        index, _, a = picks[chosen_side({k: v[1] for k, v in picks.items()}, next(iter(tail))[3])[0]]
        pose = None if index is None else candidate_pose(a, index)
        pending = None if pose is None else [str(loose[0][0]['step']), *map(str, (pose[0], pose[1], pose[2], pose[4]))]
    extra = rows[len(placed):]
    if extra and not (stopped and len(extra) == 1 and extra[0] == pending):
        problems.append(f'decisions.txt has {len(extra)} placements without a commit and its selected candidate')
    problems += placement_problems(result, by_event, commits, loose if stopped else [], extra[0] if stopped and extra else None)

    gone, start = set(), None
    last_event = {max(s['event'] for s in c['sides'].values()): c for c in placed if c['sides']}
    for meta, a in events:
        where, sampled, remaining = f"event {meta['event']}", a['sampled_ids'], a['inventory_ids']
        n = len(sampled)
        if sampled.dtype.kind not in 'iu' or (remaining.size and remaining.dtype.kind not in 'iu'):
            problems.append(f'{where}: stone ids are not integers')
        s_ids, r_ids = sampled.tolist(), remaining.tolist()
        if len(set(s_ids)) != n or len(set(r_ids)) != len(r_ids) or set(s_ids) & set(r_ids):
            problems.append(f'{where}: sampled and remaining stone ids are not distinct')
        start = set(s_ids) | set(r_ids) if start is None else start
        if set(s_ids) | set(r_ids) != start - gone:
            problems.append(f'{where}: sampled and remaining stones are not the inventory left by earlier placements')
        i = np.arange(len(a['stone_id']))
        if n != meta['nb_stones'] or not n or len(i) % n or not ((a['stone_id'] == sampled[i % n]).all()
                                                               and (a['rotation'] == i // n * 90).all()):
            problems.append(f'{where}: candidates do not follow the {n} sampled stones and their rotations')
        x_world, rotation_world = to_world(meta['side'], a['x'], a['rotation'], a['width'], meta['world_width'])
        if not (np.array_equal(x_world, a['x_world']) and np.array_equal(rotation_world, a['rotation_world'])):
            problems.append(f'{where}: world poses do not follow the evaluation-frame poses')
        if meta['event'] in last_event:
            gone.add(last_event[meta['event']]['stone_id'])
    return problems


def check_result(result):
    problems, stats = [], Counter()
    events, commits = load_result(result)
    for meta, a in events.values():
        p = meta['policy']
        s = select(a['features_raw'], a['weights'], p['policy'], p['order'], p['tol'], p.get('pick'))
        if (-1 if s.index is None else s.index) != int(a['selected']):
            problems.append(f"event {meta['event']}: replay differs from live selection")
    base = pickle.load(open(Path(result) / 'base.pkl', 'rb'))
    rocks = {r.id: r for r in base.placed_rocks}
    placed = [c for c in commits if c['outcome'] == 'placed']
    for c in placed:
        meta, a = events[c['sides'][c['chosen_side']]['event']]
        i = c['sides'][c['chosen_side']]['index']
        pose = (int(a['stone_id'][i]), int(a['x_world'][i]), int(a['y'][i]), int(a['rotation_world'][i]))
        if pose != (c['stone_id'], c['x_world'], c['y'], c['rotation_world']):
            problems.append(f"step {c['step']}: commit pose {c} differs from candidate {pose}")
        rock = rocks.get(c['stone_id'])
        if rock is None:
            problems.append(f"step {c['step']}: stone {c['stone_id']} not in base.pkl")
            continue
        rows, cols = np.nonzero(rock.matrix)
        same_origin = [int(cols.min()), int(rows.min())] == [c['x_world'], c['y']]
        same_pixels = footprint(rock.matrix, 'left') == a['footprint'][i]
        kind = 'stabilized' if c.get('placed_rot_angle') else 'unrotated'
        stats[f"{c['chosen_side']}/{kind}/{'same' if same_origin and same_pixels else 'changed'}"] += 1
        if kind == 'unrotated' and not (same_origin and same_pixels):
            problems.append(f"step {c['step']}: unstabilized rock does not match its recorded pose")
    # right-side candidates against the same stone's left-side candidates of the same attempt
    for c in commits:
        if {'left', 'right'} <= set(c['sides']) and c['sides']['right']['event'] in events:
            left = events[c['sides']['left']['event']][1]
            right = events[c['sides']['right']['event']][1]
            shapes = {(int(s), int(r)): f for s, r, f in zip(left['stone_id'], left['rotation_world'], left['footprint'])}
            for s, r, f in zip(right['stone_id'], right['rotation_world'], right['footprint']):
                stats['right candidates checked'] += 1
                if shapes.get((int(s), int(r)), f) != f:
                    problems.append(f"attempt step {c['step']}: right candidate {s}@{r} differs from left")
    stats['events'], stats['attempts'], stats['placed'] = len(events), len(commits), len(placed)
    return {'result': str(result), 'problems': problems, 'stats': dict(stats)}


def check_run(run):
    receipt = json.loads((Path(run) / 'receipt.json').read_text(encoding='utf-8'))
    results = [check_result(r) for r in result_dirs(run)]
    problems = list(receipt['trace_check'].get('problems', [])) + [p for r in results for p in r['problems']]
    if not results:
        problems.append('no result directory')
    return {'status': receipt['status'], 'ok': not problems, 'problems': problems, 'results': results}


def compare_runs(a, b):
    """Differences between two runs that should be identical (same case, policy, order, seed)."""
    ra, rb = result_dirs(a), result_dirs(b)
    if len(ra) != len(rb):
        return [f'{len(ra)} vs {len(rb)} result directories']
    diffs = []
    for x, y in zip(ra, rb):
        for name in ['decisions.txt', 'filter.txt', 'transformation.txt', 'trace/commits.jsonl']:
            if (x / name).exists() != (y / name).exists():
                diffs.append(f'{name}: present in only one run')
            elif (x / name).exists() and (x / name).read_bytes() != (y / name).read_bytes():
                diffs.append(f'{name} differs')
        bx, by = pickle.load(open(x / 'base.pkl', 'rb')), pickle.load(open(y / 'base.pkl', 'rb'))
        for attr in ['matrix', 'id_matrix', 'cluster_matrix']:
            if not np.array_equal(getattr(bx, attr), getattr(by, attr)):
                diffs.append(f'base.pkl {attr} differs')
        if (x / 'trace').exists() and (y / 'trace').exists():
            ex, ey = load_result(x)[0], load_result(y)[0]
            if sorted(ex) != sorted(ey):
                diffs.append(f'trace events {len(ex)} vs {len(ey)}')
            diffs += event_differences(ex, ey)
    return diffs


def event_differences(ex, ey):
    """Differences between the trace events two runs share; the id of the run is the only field allowed to differ."""
    diffs = []
    for e in sorted(set(ex) & set(ey)):
        (mx, ax), (my, ay) = ex[e], ey[e]
        if {k: v for k, v in mx.items() if k != "run_id"} != {k: v for k, v in my.items() if k != "run_id"}:
            diffs.append(f"event {e}: meta differs")
        if sorted(ax) != sorted(ay) or any(
            not np.array_equal(ax[k], ay[k], equal_nan=ax[k].dtype.kind == "f") for k in ax
        ):
            diffs.append(f"event {e}: arrays differ")
    return diffs


def differing_files(a, b):
    """Paths relative to the result directory present in only one run or with different bytes."""
    ra, rb = result_dirs(a), result_dirs(b)
    if len(ra) != len(rb):
        return [f'{len(ra)} vs {len(rb)} result directories']
    out = []
    for x, y in zip(ra, rb):
        fx = {p.relative_to(x).as_posix(): p for p in x.rglob('*') if p.is_file()}
        fy = {p.relative_to(y).as_posix(): p for p in y.rglob('*') if p.is_file()}
        out += sorted(set(fx) ^ set(fy))
        out += sorted(k for k in set(fx) & set(fy) if fx[k].read_bytes() != fy[k].read_bytes())
    return out


def manifest(output):
    """sha256 of every trace and result file of the run, taken when the case process has ended."""
    return {p.relative_to(output).as_posix(): sha256_file(p) for r in result_dirs(output) for p in artifact_files(r)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args(argv)
    report = check_run(args.run)
    if args.compare:
        report['differences_to'] = {'run': str(args.compare), 'differences': compare_runs(args.run, args.compare),
                                    'differing_files': differing_files(args.run, args.compare)}
    print(json.dumps(report, indent=2))
    return 0 if report['ok'] and not report.get('differences_to', {}).get('differences') else 1


if __name__ == '__main__':
    raise SystemExit(main())
