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

The four wall metrics of Wang et al. Section 5.2 (wang_metrics; not part of wall_metrics yet), as the
author's Tests/Thangavelu/03_evaluate_wall_metrics.py computes them on a stone label image, with
these departures from the paper's Eqs. 25-28:
- F_SF (Eq. 25, higher is better): the script crops empty columns on the right first and divides
  by H*W - 2H - 3W of the cropped image, not by a target wall area.
- F_AH (Eq. 26, lower is better) and F_AV (Eq. 27, higher is better): the paper averages n
  shortest joint paths; the script starts about 10 (every H//10 rows / W//10 columns from 10 px
  in), keeps the 5 smallest (length - span) / span and reports their mean x 100.
- F_LR (Eq. 28, higher is better): min(alpha_left, alpha_right) / mu; the script and lambda_min
  leave out the division by mu.
The author renders the label image at the stones' original resolution
(02_view_transformation_ori_reso.py); rock_labels is the simulation grid, so values differ by
resolution, not by method. The formal metrics are wall_quality: 02's rendering (render_original)
then 03, as 02 does it: a stone moved beyond its own canvas (the stone image's size, 400 rows for
noise 80, under the 559-row wall) is cut silently, and one overlapping the stones before by
200 px or more is left out; render_original reports both per stone. wall_quality(...,
canvas='corrected') moves each stone on a canvas covering the wall instead, so only the wall's
bounds cut it; everything else is 02's. The two modes are separate methods: never mix them in one
comparison.
"""
import json
import pickle
import re
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.measure import regionprops

from Stonepacker2D.stack.selection import CODE_ORIGINAL_ORDER, PAPER_ORDER, load_trace, select, tie_count
from Stonepacker2D.utils.constant import get_base_id, get_ignore_pixel_value, get_wedge_id

METRICS_VERSION = 1
MU = 0.58  # Stonepacker2D.utils.constant.MU; no run here calls set_mu
SCORES = {"01_LLC": 3, "02_NHR": 0, "03_IWR": 2, "04_GDS": 1, "05_LFR": 4}  # column of features_raw
FILTER_NAMES = {column: name[3:] for name, column in SCORES.items()}


def mechanism_lambdas(result):
    """The limit load of each side: mechanism.csv of a new wall, the image names of an archived one."""
    table = Path(result) / 'mechanism.csv'
    rows = [line.split(',') for line in table.read_text(encoding='utf-8').splitlines()[1:]] if table.exists() else None
    found = {}
    for side in ('left', 'right'):
        if rows is None:
            names = [p.name for p in (Path(result) / 'img').glob(f'{side}*_mechanism.png')]
            values = [float(re.match(rf'{side}(.+)_mechanism\.png$', n).group(1)) for n in names]
        else:
            values = [float(value) for name, value in rows if name == side]
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


def rock_labels(base):
    """Placed rocks numbered in placement order (1..n) on the simulation grid, row 0 at the ground
    as in the author's label image; ground, frame and wedges are 0."""
    labels = np.zeros(base.matrix.shape, int)
    for k, rock in enumerate(base.placed_rocks, 1):
        labels[rock.matrix != 0] = k
    return labels


def crop_right(labels):
    """03: drop the empty columns right of the last stone column."""
    return labels[:, :np.flatnonzero(labels.sum(axis=0) != 0).max() + 1]


def filling_author(labels):
    """03's wall_stone_ratio: stone pixels / (H*W - 2H - 3W) of the right-cropped label image."""
    img = crop_right(labels)
    H, W = img.shape
    return np.count_nonzero(img) / (H * W - 2 * H - 3 * W)


def joint_image(labels):
    """03's joint image: right-cropped, every stone eroded alone with a 3x3 kernel (so touching
    stones are parted), empty rows past the last stone row dropped, then zeros padded (1 row
    before, 2 after, 1 column left, 2 right). 255 = stone, 0 = joint or empty."""
    img = crop_right(labels)
    eroded = np.zeros(img.shape, np.uint8)
    for region in regionprops(img):
        stone = np.zeros(img.shape, np.uint8)
        stone[region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]] = region.image
        eroded += cv2.erode(stone, np.ones((3, 3), np.uint8), iterations=1)
    eroded = np.where(eroded != 0, 255, 0)
    eroded = eroded[:np.flatnonzero(eroded.sum(axis=1) != 0).max() + 1]
    return np.pad(eroded, ((1, 2), (1, 2)))


def joint_graph(img):
    """graph.build_graph(img, erosion=False) as a sparse matrix over the joint pixels, with the
    Euclidean weights 03 assigns afterwards (they replace build_graph's 1, 1.44 and x0.3 interface
    weights): neighbours down and right, the down-right diagonal and the up-right diagonal (whose
    lower end is never the last row, as in build_graph). build_graph zips every edge list with a
    weight list as long as the first one (row neighbours), so it keeps only that many of the other
    three, the first in row-major order; so does this."""
    H, W = img.shape
    free = img == 0
    index = np.arange(H * W).reshape(H, W)
    a, b, w = [], [], []
    n = None
    for mask, s, t, weight in (
            (free[1:, :] & free[:-1, :], (slice(0, -1), slice(None)), (slice(1, None), slice(None)), 1.0),
            (free[:, 1:] & free[:, :-1], (slice(None), slice(0, -1)), (slice(None), slice(1, None)), 1.0),
            (free[1:, 1:] & free[:-1, :-1], (slice(0, -1), slice(0, -1)), (slice(1, None), slice(1, None)), np.sqrt(2)),
            (free[:-2, 1:] & free[1:-1, :-1], (slice(1, -1), slice(0, -1)), (slice(0, -2), slice(1, None)), np.sqrt(2))):
        n = mask.sum() if n is None else n  # ponytail: the author's truncation, kept on purpose to match 03
        a.append(index[s][mask][:n])
        b.append(index[t][mask][:n])
        w.append(np.full(min(n, mask.sum()), weight))
    a, b, w = np.concatenate(a), np.concatenate(b), np.concatenate(w)
    return coo_matrix((w, (a, b)), shape=(H * W, H * W)).tocsr()


def joint_paths(img, vertical):
    """03's compute_FAV (vertical) / compute_FAH: shortest joint paths across the image from every
    H//10-th row (from the left edge, if it is a joint) or W//10-th column (from the top), starting
    10 px in; per path (length - span) / span. None if the step is 0 (03 fails there)."""
    H, W = img.shape
    step = (W if vertical else H) // 10
    if step == 0:
        return None
    if vertical:
        pairs = [((0, c), (H - 1, c)) for c in range(10, W - 10, step)]
    else:
        pairs = [((r, 0), (r, W - 1)) for r in range(10, H - 10, step) if img[r, 0] == 0]
    if not pairs:
        return []
    starts = [s[0] * W + s[1] for s, _ in pairs]
    dist = dijkstra(joint_graph(img), directed=False, indices=starts)
    span = H - 1 if vertical else W - 1
    return [{'start': s, 'end': e, 'length': float(d[e[0] * W + e[1]]),
             'factor': float((d[e[0] * W + e[1]] - span) / span)} for (s, e), d in zip(pairs, dist)]


def path_metric(paths):
    """03: the mean of the 5 smallest factors x 100; None without paths (03 returns nan) or with an
    unreachable end (03 raises)."""
    if not paths or not all(np.isfinite(p['factor']) for p in paths):
        return None
    return 100 * float(np.mean(np.sort([p['factor'] for p in paths])[:5]))


def wang_metrics(result, labels=None):
    """The four metrics of Wang et al. Section 5.2 of one wall, raw values; see the module notes.
    labels: a stone label image (e.g. the author's stacked_wall_label.png); by default the
    simulation grid's rock_labels of result/base.pkl."""
    result = Path(result)
    if labels is None:
        base = pickle.load(open(result / 'base.pkl', 'rb'))
        labels = rock_labels(base)
    out = {'F_SF_author': None, 'F_AH': None, 'F_AV': None, 'F_AH_paths': None, 'F_AV_paths': None}
    if np.count_nonzero(labels):  # 03 fails on a wall without stones
        img = joint_image(labels)
        horizontal, vertical = joint_paths(img, False), joint_paths(img, True)
        out.update(F_SF_author=filling_author(labels), F_AH=path_metric(horizontal), F_AV=path_metric(vertical),
                   F_AH_paths=horizontal, F_AV_paths=vertical, joint_image_shape=list(img.shape))
    lam = mechanism_lambdas(result)
    lam_min = min(lam['left'], lam['right']) if None not in lam.values() else None
    out.update(lambda_min=lam_min, F_LR=None if lam_min is None else lam_min / MU, mu=MU)
    return out


def lambdas(world):
    """evaluate_kine(load='tilting_table') of a wall matrix with both directions kept unrounded: its two
    solve_force_rigid calls, 'right' (load +mass) and 'left' (-mass) as its mechanism files name them.
    Bricks from the matrix as research/analysis/lookahead_study.kine_lr builds them."""
    from Stonepacker2D.evaluate.tilting_angle import (
        base_to_elements,
        cal_Aglobal,
        clear_contps,
        pixel_to_mesh,
        solve_force_rigid,
    )
    from Stonepacker2D.utils.constant import set_head_joint_state, set_world_size

    set_world_size(world.shape[1], world.shape[0])
    set_head_joint_state(True)
    bricks = []
    for k in range(1, int(world[world != 255].max()) + 1):
        r, c = np.nonzero(world == k)
        bricks.append(
            SimpleNamespace(
                matrix=(world == k).astype(np.uint8),
                width=np.ptp(c) + 1 if r.size else 0,
                height=np.ptp(r) + 1 if r.size else 0,
            )
        )
    base = SimpleNamespace(matrix=world, placed_bricks=bricks, cluster_matrix=np.zeros(world.shape))
    img = base.matrix.astype(np.uint8)
    img = np.where(img != get_ignore_pixel_value(), img, 0)
    contps = pixel_to_mesh(img)
    elems = base_to_elements(base)
    contps = clear_contps(elems, contps)
    out = {}
    for side, sign in (("right", 1), ("left", -1)):
        for e in elems.values():
            e.dl, e.ll = [0, -e.mass, 0], [sign * e.mass, 0, 0]
        if side == "right":
            Aglobal = cal_Aglobal(elems, contps)
        out[side] = float(solve_force_rigid(elems, contps, Aglobal)["limit_force"])
    return out


def partial_quality(result):
    """F_SF, F_AH, F_AV, F_LR of the wall as it stands after each placement: one dict per line of
    trace/commits.jsonl, empty for an attempt that placed nothing or a wall without base.pkl.
    On the simulation grid, so not the formal metrics of wall_quality: 03's rules on the rock labels of
    base.pkl (F_AH, F_AV None while the wall is too small for its 10 px margins), and F_LR = the smaller
    unrounded tilting-table solve / mu."""
    result = Path(result)
    commits = read_jsonl(result / "trace" / "commits.jsonl")
    if not (result / "base.pkl").exists():
        return [{} for _ in commits]
    with open(result / "base.pkl", "rb") as f:
        base = pickle.load(f)
    labels, rocks, out = rock_labels(base), 0, []
    for c in commits:
        if c["outcome"] != "placed":
            out.append({})
            continue
        rocks += "placed_bbox" in c
        m = wang_metrics(result, labels=np.where(labels <= rocks, labels, 0))
        kept = (base.matrix <= c["step"] + c["bricks_added"]) | (base.matrix == get_ignore_pixel_value())
        out.append(
            {
                "F_SF": m["F_SF_author"],
                "F_AH": m["F_AH"],
                "F_AV": m["F_AV"],
                "F_LR": min(lambdas(np.where(kept, base.matrix, 0)).values()) / MU,
            }
        )
    return out


def step_wall_times(events):
    """Wall-clock seconds of each sampling attempt in commit order, from events.jsonl: the attempt alone
    and since its build began. Empty for a case recorded before the step_commit events."""
    out = []
    for e in read_jsonl(events):
        now = e["elapsed_seconds"]
        if e["event"] == "build_enter":
            begin = previous = now
        elif e["event"] == "step_commit":
            out.append({"wall_time_s": round(now - previous, 3), "wall_time_cum_s": round(now - begin, 3)})
            previous = now
    return out


RESCALE = 0.5  # _rescale of Tests/Thangavelu/02_view_transformation_ori_reso.py


def stone_source(run):
    """The original-resolution stone label image a Thangavelu run read, as its 00_run.py names it;
    refuses a run whose _rescale is not 02's 0.5."""
    text = (Path(run) / '00_run.py').read_text(encoding='utf-8')
    image = re.findall(r"/data/(stones_noise_\d+_labels\.png)", text)
    rescale = re.findall(r"^\s*_rescale = ([0-9.]+)\s*$", text, re.M)
    if len(set(image)) != 1 or [float(r) for r in rescale] != [RESCALE]:
        raise ValueError(f'not a Thangavelu run with _rescale {RESCALE}: {image} {rescale}')
    copied = Path(run) / 'data' / image[0]  # an archived run has a copy, a new case keeps none
    return copied if copied.exists() else Path(__file__).resolve().parents[2] / 'Tests' / 'Thangavelu' / 'data' / image[0]


def render_original(run, result, canvas='author', leave_out=True, overlap='later'):
    """02's stacked_wall_label.png of a wall, at the stones' original resolution: the wall size
    from the stone label image (5 mean bounding boxes, + 2/rescale and 3/rescale), every stone
    (region >= 50 px, holes filled) centered on a canvas of that image's size, then for each line
    of transformation.txt in order turned by its angle about the canvas center, moved so its
    centroid is at (d_x, d_y), cropped to the wall and drawn with its placement number, unless it
    overlaps the stones drawn before by 200 px or more (then 02 leaves it out).
    canvas='author' moves the stone on its own canvas, as 02 does (line 183), which cuts what lands
    beyond it (rows >= 400 for noise 80, inside the 559-row wall); canvas='corrected' moves it on a
    canvas covering both, so only the wall's bounds crop it. Nothing else differs.
    leave_out=False (sensitivity method wang-quality-s1, not 02) draws every stone, however much it
    overlaps; overlap='later' gives the contested pixels to the stone drawn later, as 02 does for
    every stone it keeps, 'earlier' to the one drawn first.
    Returns the uint8 label image and, per stone: pixels drawn, lost_to_crop (all this mode cut),
    canvas_cut (pixels inside the wall that 02's own canvas cuts, the same in both modes; the
    corrected mode keeps them), outside_wall (beyond the wall, cut in both), turn_at_edge (the
    turned stone touches its canvas border, so the turn itself may have cut it), overlap, left_out."""
    if canvas not in ('author', 'corrected') or overlap not in ('later', 'earlier') or (leave_out and overlap != 'later'):
        raise ValueError((canvas, leave_out, overlap))
    contested_to = overlap  # the loop counts each stone's overlap in `overlap`
    from skimage.measure import label
    image = label(cv2.imread(str(stone_source(run)), cv2.IMREAD_GRAYSCALE).astype(np.uint8))
    regions = [r for r in regionprops(image) if r.area >= 50]
    width = sum(r.bbox[3] - r.bbox[1] for r in regions) / len(regions) * 5
    height = sum(r.bbox[2] - r.bbox[0] for r in regions) / len(regions) * 5
    size = (int(width + 2 / RESCALE), int(height + 3 / RESCALE))
    stones = {}
    for r in regions:
        m = np.zeros_like(image)
        m[r.bbox[0]:r.bbox[2], r.bbox[1]:r.bbox[3]] = r.image_filled.astype(np.uint8)
        c = regionprops(m.astype(np.uint8))[0].centroid
        shift = np.float32([[1, 0, m.shape[1] // 2 - c[1]], [0, 1, m.shape[0] // 2 - c[0]]])
        stones[r.label] = cv2.warpAffine(m, shift, (m.shape[1], m.shape[0]), flags=cv2.WARP_FILL_OUTLIERS)
    lines = (Path(result) / 'transformation.txt').read_text().splitlines()[1:]
    placed = [(int(f[0]), float(f[1]), float(f[2]), float(f[3])) for f in (line.split(';') for line in lines)]
    ids = [p[0] for p in placed]
    # ponytail: 02 keys these lines by id and turns stones[id % 1000] in place, so a repeated id or
    # image would differ from this port; no recorded run has one, refuse instead of emulating
    if len({i % 1000 for i in ids}) != len(ids) or not {i % 1000 for i in ids} <= set(stones):
        raise ValueError(f'transformation.txt ids not one per stone image: {ids}')
    drawn = np.zeros((size[1], size[0]), np.uint8)
    labels, stones_out = np.zeros_like(drawn), []
    wide = (max(image.shape[1], size[0]), max(image.shape[0], size[1]))

    def to_wall(s):
        s = s[:drawn.shape[0], :drawn.shape[1]]
        return np.pad(s, ((0, drawn.shape[0] - s.shape[0]), (0, drawn.shape[1] - s.shape[1])))

    for seq, (stone_id, x, y, angle) in enumerate(placed):
        s = stones[stone_id % 1000]
        turn = cv2.getRotationMatrix2D((s.shape[1] // 2, s.shape[0] // 2), angle, 1.0)
        s = cv2.warpAffine(s, turn, (s.shape[1], s.shape[0]), flags=cv2.WARP_FILL_OUTLIERS)
        edge = bool(s[0].any() or s[-1].any() or s[:, 0].any() or s[:, -1].any())
        c = regionprops(s.astype(np.uint8))[0].centroid
        move = np.float32([[1, 0, x - c[1]], [0, 1, y - c[0]]])
        before = np.count_nonzero(s)
        own = cv2.warpAffine(s, move, (s.shape[1], s.shape[0]), flags=cv2.WARP_FILL_OUTLIERS)
        full = cv2.warpAffine(s, move, wide, flags=cv2.WARP_FILL_OUTLIERS)
        if not np.array_equal(own, full[:own.shape[0], :own.shape[1]]):
            raise RuntimeError(f'stone {stone_id}: the wider canvas moved its pixels')  # the corrected mode's premise
        own, full = to_wall(own), to_wall(full)
        s = own if canvas == 'author' else full
        overlap = np.count_nonzero((s != 0) & (drawn != 0))
        stones_out.append({'id': stone_id, 'seq': seq + 1, 'pixels': int(np.count_nonzero(s)),
                           'lost_to_crop': int(before - np.count_nonzero(s)),
                           'canvas_cut': int(np.count_nonzero(full) - np.count_nonzero(own)),
                           'outside_wall': int(before - np.count_nonzero(full)), 'turn_at_edge': edge,
                           'overlap': int(overlap), 'left_out': leave_out and overlap >= 200})
        if stones_out[-1]['left_out']:
            continue
        if contested_to == 'earlier':
            s = np.where(drawn != 0, 0, s)
        drawn = np.where(s != 0, s, drawn)
        labels = np.where(s != 0, seq + 1, labels).astype(np.uint8)
    return labels, {'size': [size[1], size[0]], 'canvas': list(image.shape), 'mode': canvas, 'stones': stones_out}


def wall_quality(run, result, canvas='author', leave_out=True, overlap='later'):
    """The four formal metrics of a wall, by the author's method: 02's rendering at the original
    resolution (canvas: 'author' as 02 does it, or 'corrected', see render_original), then 03
    (wang_metrics); leave_out, overlap: see render_original. Kept apart from wall_metrics; raw
    values, no total score."""
    result = Path(result)
    labels, render = render_original(run, result, canvas, leave_out, overlap)
    rocks = [r.id for r in pickle.load(open(result / 'base.pkl', 'rb')).placed_rocks]
    if rocks != [s['id'] for s in render['stones']]:
        raise ValueError(f'transformation.txt does not list the placed rocks of base.pkl: {rocks}')
    m = wang_metrics(result, labels=labels.astype(int))
    return {'F_SF': m['F_SF_author'], 'F_AH': m['F_AH'], 'F_AV': m['F_AV'], 'F_LR': m['F_LR'],
            'lambda_min': m['lambda_min'], 'mu': m['mu'], 'method': f'Tests/Thangavelu 02 + 03, original resolution, {canvas} canvas'
                      + ('' if leave_out else f', every stone drawn, overlap to the {overlap} stone'),
            'F_AH_factors': [p['factor'] for p in m['F_AH_paths'] or []],
            'F_AV_factors': [p['factor'] for p in m['F_AV_paths'] or []],
            'render': render}


def selection_columns(result, side):
    """The step_rows columns held by the trace event of a sampling attempt's chosen side."""
    meta, a = load_trace(result / "trace" / f"e{side['event']:05d}.npz")
    p, raw = meta["policy"], a["features_raw"]
    sel = select(raw, a["weights"], p["policy"], p["order"], p["tol"])
    used = p["order"] or (PAPER_ORDER if p["policy"] == "paper" else CODE_ORIGINAL_ORDER)
    columns = {
        "event": side["event"],
        "policy": p["policy"],
        "order": "" if p["order"] is None else "-".join(map(str, p["order"])),
        "filter_order": "-".join(FILTER_NAMES[column] for column in used),
        "n_candidates": len(a["stone_id"]),
        "n_stones": len(set(a["stone_id"].tolist())),
        "n_feasible": int((raw[:, 3] > 0).sum()),
        "branch": meta["branch"],
        "final_ties": tie_count(raw, a["weights"], sel, p["policy"], p["order"], p["tol"]),
    }
    i = side.get("index")
    if i is not None:
        columns["placement_id"] = f"{a['footprint'][i]}:{a['x_world'][i]},{a['y'][i]}"
        columns.update({name: float(raw[i, column]) for name, column in SCORES.items()})
    return columns


def step_rows(result):
    """One row per sampling attempt: the action taken, the candidate set it was taken from, the policy
    and order that selection used (a --switch or a controller changes them from step to step) and the
    raw scores of the selected candidate."""
    result = Path(result)
    rows = []
    for c in read_jsonl(result / 'trace' / 'commits.jsonl'):
        side = c['sides'].get(c['chosen_side'], {})
        row = {k: c.get(k) for k in ('step', 'phase', 'outcome', 'chosen_side', 'side_tie', 'stone_id', 'cluster',
                                     'rotation_world', 'x_world', 'y', 'termination')}
        row['placement_id'] = ''
        if side.get('event') is not None:
            row.update(selection_columns(result, side))
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
