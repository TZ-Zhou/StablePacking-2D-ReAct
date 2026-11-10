"""C4's h of every placement of one batch of research/runner/batch.py (offline, read-only on the walls).

Usage: python -B -m research.analysis.placement_h <batch folder> [--out DIR] [--cases 3-10,15]

For each PROCESS_PASS wall and each placed row t of its derived/steps.csv, the wall state right after placement t:
the skyline s' (from the trace wall of the next placement, or base.pkl after the last one; wedges Wang adds under
the stone included), the void sealed under the placed stone, and the poses of every stone still to place
(sampled_ids and inventory_ids of the placing event minus the placed stone; docs/plans/llm-agent-design-questions.md
Q7). On it, surface_fit.h_scores gives H2b, H3b, H4a and H0_unlevel (docs/plans/preregistration.md C4). The
remaining stones are every stone not yet in the wall, so the walls before and after a placement fix its state; each
distinct state is computed once.

Stone shapes are rebuilt from the scene's label image the way its run script does (Thangavelu: regionprops of
00_run.py; Almeida: Stone.from_labeled_img of run.py). iou is the best overlap of the placed stone's pixels with one
of its four right-angle poses: on 6 walls of each S1 block and 30 of AL_IR the median is 1.0, 8% of AL_IR's
placements are below 0.9 and the lowest is 0.61 (a 33 px stone). Why some placed stones differ was not checked;
the h of the remaining stones uses the four right-angle poses either way.

Writes once, to <out> (default research/datasets/<name>-h): h.csv (one row per wall and placement: wall, t, event,
stone, n_remaining, ref_width, iou, H2b, H3b, H4a, H0_unlevel) and h.json (source batch, code's git head and source
blobs, counts). Refuses an existing <out> and source files that differ from git HEAD.
"""

import argparse
import contextlib
import hashlib
import io
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops

from research.analysis import surface_fit as sf
from research.analysis.batch_tables import BATCH_FILES, case_records
from research.paths import DATASETS_DIR, REPO
from research.walls import batch_plan
from research.walls import campaign_quality as cq
from Stonepacker2D import Stone, get_world_size_from_label_img
from Stonepacker2D.stack.selection import load_trace
from Stonepacker2D.utils.constant import set_world_size

SOURCES = ["research/analysis/placement_h.py", "research/analysis/surface_fit.py", "research/analysis/batch_tables.py",
           "Stonepacker2D"]
TH_SCALE = 0.5  # Tests/Thangavelu/00_run.py: _rescale, regions under 50 px skipped, wall 5 average boxes wide
TH_MIN_AREA = 50
TH_BOXES = 5
AL_MIN_SIDE = 3  # Tests/Almeida/*/run.py keeps stones taller and wider than 3 px
QUARTER_TURNS = 4


def crop(mask):
    """A boolean mask cropped to its bounding box, plus the box's lowest row and leftmost column."""
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1], int(rows[0]), int(cols[0])


def thangavelu_stones(image):
    """{stone id: mask} as Tests/Thangavelu/00_run.py builds them: ids are the region labels."""
    regions = [r for r in regionprops(label(cv2.imread(str(image), cv2.IMREAD_GRAYSCALE).astype(np.uint8)))
               if r.area >= TH_MIN_AREA]
    width = sum(r.bbox[3] - r.bbox[1] for r in regions) / len(regions) * TH_BOXES
    height = sum(r.bbox[2] - r.bbox[0] for r in regions) / len(regions) * TH_BOXES
    set_world_size(int(width * TH_SCALE) + 2, int(height * TH_SCALE) + 3)
    stones = {}
    for region in regions:
        stone = Stone()
        if stone.from_matrix(region.image_filled.astype(np.uint8), TH_SCALE):
            stone.id = region.label
            stones[region.label] = stone.rotate_min_shape_factor()["stone"].matrix
    return stones


def almeida_stones(image, config):
    """{stone id: mask} as Tests/Almeida/*/run.py builds them: ids 0.. in label order (Vendor renews them)."""
    width, height = get_world_size_from_label_img(str(image), scale=config["scale"])
    set_world_size(int(width * config["wall_width_times"] + config["wall_width_plus"]),
                   int(height * config["wall_height_times"] + config["wall_height_plus"]))
    stones = []
    for value in range(1, 256):
        stone = Stone()
        if stone.from_labeled_img(value, str(image), config["scale"]):
            if stone.height > AL_MIN_SIDE and stone.width > AL_MIN_SIDE:
                stones.append(stone)
    rotated = [stone.rotate_min_shape_factor() for stone in stones]
    return dict(enumerate(r["stone"].matrix for r in rotated if r["success"]))


def scene_stones(case):
    """{stone id: cropped boolean mask} of the scene a case was built from (Wang's code prints; that is muted)."""
    receipt = json.loads((case / "receipt.json").read_text(encoding="utf-8"))
    source = REPO / receipt["source"]
    with contextlib.redirect_stdout(io.StringIO()):
        if source.name == "00_run.py":
            name = next((new for old, new in receipt.get("source_edits", []) if old.endswith(".png")),
                        "/data/stones_noise_80_labels.png")
            stones = thangavelu_stones(source.parent / name.lstrip("/"))
        else:
            config = json.loads((source.parent / "config.json").read_text(encoding="utf-8"))
            stones = almeida_stones(source.parent / "data" / f"{config['input_wall_name']}.png", config)
    return {stone: crop(matrix > 0)[0] for stone, matrix in stones.items()}


def best_iou(mask, placed):
    """Largest overlap / union of placed with one of mask's four poses, both put at the same corner."""
    best = 0.0
    for k in range(QUARTER_TURNS):
        pose = np.rot90(mask, k)
        shape = np.maximum(pose.shape, placed.shape)
        a, b = np.zeros(shape, bool), np.zeros(shape, bool)
        a[: pose.shape[0], : pose.shape[1]], b[: placed.shape[0], : placed.shape[1]] = pose, placed
        best = max(best, (a & b).sum() / (a | b).sum())
    return float(best)


def world_wall(meta, arrays):
    """The trace event's wall in the wall frame: a trace stores it in the evaluation frame, mirrored (np.fliplr)
    when the course is built from the right (Stonepacker2D/stack/build_wall.py trace_event)."""
    return np.fliplr(arrays["wall"]) if meta["side"] == "right" else arrays["wall"]


def placements(case):
    """(t, event, wall before, wall after, trace arrays) of each placed row of the case's derived/steps.csv; both
    walls in the wall frame, so a state does not depend on the side its next placement is evaluated from."""
    result = next(case.glob("result*"))
    events = sorted(result.glob("trace/e*.npz"))
    placed = pd.read_csv(case / "derived" / "steps.csv").query("outcome == 'placed'")["event"].astype(int).tolist()
    with open(result / "base.pkl", "rb") as file:
        final = pickle.load(file).matrix
    traces = [load_trace(events[event]) for event in placed]
    walls = [world_wall(meta, arrays) for meta, arrays in traces]
    for t, (event, (_, arrays)) in enumerate(zip(placed, traces), start=1):
        yield t, event, walls[t - 1], walls[t] if t < len(walls) else final, arrays


def placement_state(before, after, arrays, poses_of, masks):
    """The h.csv columns of one placement but wall, t and event."""
    new = np.setdiff1d(np.unique(after), np.unique(before))
    if not len(new):
        raise ValueError("no new label after a placement")
    # Wang may add wedges under the stone (Stonepacker2D/entity/wedge.py): the stone is the largest new label.
    label_of_stone = max(new, key=lambda value: int((after == value).sum()))
    placed, row, col = crop(after == label_of_stone)
    stone = int(arrays["stone_id"][int(arrays["selected"])])
    remaining = sorted((set(arrays["sampled_ids"].tolist()) | set(arrays["inventory_ids"].tolist())) - {stone})
    pose = sf.make_pose(placed)
    under = sf.skyline(np.where(after == label_of_stone, 0, after))  # wedges included: they fill part of the void
    x, y = col - 1, row - sf.GROUND_ROWS  # inner columns start after the frame column
    sealed = int((y + pose.bottom - under[x : x + len(pose.bottom)]).clip(0).sum())
    poses = [poses_of[r] for r in remaining]
    return dict(stone=stone, n_remaining=len(remaining), ref_width=sf.reference_width(poses),
                iou=best_iou(masks[stone], placed), **sf.h_scores(sf.skyline(after), sealed, poses))


def h_table(folder, numbers=None):
    """(h.csv table, number of distinct states) of a batch folder; each distinct state is computed once."""
    rows, seen, masks, poses_of = [], {}, None, None
    for _, record in case_records(folder, numbers):
        if record["status"] != "PROCESS_PASS":
            continue
        case = folder / "cases" / record["case_id"]
        if masks is None:  # one batch is one scene and seed
            masks = scene_stones(case)
            poses_of = {stone: sf.stone_poses(mask) for stone, mask in masks.items()}
        for t, event, before, after, arrays in placements(case):
            # The remaining stones are every stone not in the wall, so the two walls fix the whole state.
            key = (hashlib.sha256(before.tobytes()).digest(), hashlib.sha256(after.tobytes()).digest())
            if key not in seen:
                seen[key] = placement_state(before, after, arrays, poses_of, masks)
            rows.append(dict(wall=record["case_id"], t=t, event=event, **seen[key]))
    return pd.DataFrame(rows), len(seen)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("batch", type=Path, help="runs/batch/<name> folder")
    parser.add_argument("--out", type=Path, help="default research/datasets/<name>-h")
    parser.add_argument("--cases", help="case numbers, e.g. 3-10,15 (default: all)")
    args = parser.parse_args(argv)
    folder = args.batch.resolve()
    out = (args.out or DATASETS_DIR / f"{folder.name}-h").resolve()
    if out.exists():
        raise SystemExit(f"{out} exists: a dataset is written once")
    if REPO.resolve() not in out.parents:
        raise SystemExit(f"{out} is outside the repository")
    hashes, dirty = cq.sources(SOURCES)
    if dirty:
        raise SystemExit(f"source files differ from HEAD: {dirty}")
    total = len(batch_plan.read_plan(folder / "plan.csv"))
    numbers = set(batch_plan.chosen_numbers(total, args.cases)) if args.cases else None
    table, states = h_table(folder, numbers)
    if cq.sources(SOURCES) != (hashes, dirty):
        raise SystemExit("a source file changed while h was computed: nothing written")
    out.mkdir(parents=True)
    table.to_csv(out / "h.csv", index=False, lineterminator="\n")
    info = {
        "batch": folder.name,
        "batch_files_sha256": {name: cq.sha256(folder / name) for name in BATCH_FILES if (folder / name).exists()},
        "git_head": cq.git("rev-parse", "HEAD").strip(),
        "sources_git_blob": hashes,
        "n_walls": int(table["wall"].nunique()) if len(table) else 0,
        "n_rows": len(table),
        "n_states": states,
        "iou_min": float(table["iou"].min()) if len(table) else None,
    }
    (out / "h.json").write_text(json.dumps(info, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{info['n_walls']} walls, {len(table)} placements, {states} distinct states written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
