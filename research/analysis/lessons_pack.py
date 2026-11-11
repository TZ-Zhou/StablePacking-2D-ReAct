"""The pack of finished walls an LLM draws lessons from (docs/plans/llm-lessons.md).

Usage: python -B -m research.analysis.lessons_pack [--without SEED] [--out FILE] [--write]

The source is the 10 TH factorial blocks (SEEDS x SCENES; research/datasets/factorial-s<seed>-<scene>, the walls'
traces in runs/batch). Per block, walls built alike count once (walls.csv unique_id; of a group the smallest filter
order names it) and 4 are taken by formula (select): the 2 of highest mean z among those placing the block's most
stones, the one of lowest mean z among them, and the wall placing the fewest stones (then the lowest mean z); ties by
the filter order string. z is llm_endgame_pick's, in the block's 120 walls.

A wall's steps are read from its trace through the whole-wall pick's own Observer (research/controller/pick.py,
'pick-4'), so every quantity is one the builder sees at a placement: n_remaining, the number of feasible candidates,
the stones on offer without one, the wall summary, and the placed candidate's rotation, bounding box and five scores,
each with the share of that placement's candidates it is at least as good as. Not in a pack: seeds, wall ids, filter
orders, stone ids, coordinates; blocks are stone sets A, B, ... and walls A1..A4.

fold(seed) is the source of the lessons for a test seed: every block of the other seeds (--without SEED); fold(None)
is all 10. --write stores each fold's wall list and pack sha256 in PACKS (committed before a paid induction, which
refuses a pack that differs: research/llm_walls/lessons.py).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.analysis import llm_endgame_pick as step3
from research.analysis.endgame_arena import METRICS
from research.controller import order_controller as oc
from research.controller import pick
from research.paths import BATCH_RUNS, DATASETS_DIR, REPO

SEEDS = (313, 484, 717, 720, 761)
SCENES = ("TH_T80", "TH_T20")
LABELS = ("good", "good", "bad: lower quality", "bad: stones lost")
WALL_KEYS = ("h_max", "h_range", "h_std", "void_under_skyline_px", "n_skyline_jumps")
SCORES = ("LLC", "NHR", "IWR", "GDS", "LFR")
COLUMNS = ["placement", "n_remaining", "n_candidates", "n_unplaceable", *WALL_KEYS, "rotation", "width", "height",
           *(c for s in SCORES for c in (s, f"{s}_share"))]
PACKS = REPO / "research" / "llm_walls" / "lessons_packs.json"


def fold(seed=None):
    """The source blocks of the lessons tested on `seed`: no block of that seed (None: all 10)."""
    return [(s, scene) for s in SEEDS if s != seed for scene in SCENES]


def select(distinct):
    """The 4 rows of LABELS from a block's distinct walls (columns n_placed, mean_z, filter_policy)."""
    top = distinct[distinct.n_placed == distinct.n_placed.max()].sort_values(
        ["mean_z", "filter_policy"], ascending=[False, True])
    fewest = distinct.sort_values(["n_placed", "mean_z", "filter_policy"]).iloc[0]
    if len(top) < 3 or fewest.n_placed == distinct.n_placed.max():
        # ponytail: the plan's fill from the next stone count is not written; no source block needs it
        raise ValueError("a block needs 3 distinct walls at its most stones and a wall placing fewer")
    return [top.iloc[0], top.iloc[1], top.sort_values(["mean_z", "filter_policy"]).iloc[0], fewest]


def chosen(seed, scene):
    """A block's 4 walls: rows of walls.csv with z scores and the rank of the mean z among the block's walls."""
    walls = pd.read_csv(DATASETS_DIR / f"factorial-s{seed}-{scene}" / "walls.csv", dtype={"filter_policy": str})
    walls = pd.concat([walls, step3.zscores(walls, walls)], axis=1)
    walls["rank"] = walls.mean_z.rank(ascending=False, method="min").astype(int)
    assert walls.unique_id.notna().all() and (walls.status == "PROCESS_PASS").all()
    return select(walls.sort_values("filter_policy").drop_duplicates("unique_id")), len(walls)


def steps_of(case):
    """(a wall's placement rows of COLUMNS, their poses (stone, rotation, x, y), the end of a wall that lost
    stones or None) from the trace of its case folder."""
    observer, rows, poses, end = pick.Observer(pick.OBSERVATION_VERSION_WHOLE), [], [], None
    for meta, a in oc.trace_events(next(Path(case).glob("result*"))):
        obs, feasible, _ = observer(None, oc.context_from_trace(meta, a))
        state = [round(obs["wall"][k], 1) for k in WALL_KEYS]
        if not len(feasible):  # an offer with no feasible candidate: a new sample, or the end of the wall
            end = {"n_remaining": obs["n_remaining"], **dict(zip(WALL_KEYS, state))}
            continue
        end, k = None, list(feasible).index(int(a["selected"]))
        c, n = obs["candidates"][k], len(a["sampled_ids"])
        marks = []
        for s in SCORES:
            values = np.array([x["scores"][s] for x in obs["candidates"]])
            marks += [round(c["scores"][s], 3), round(float((values <= c["scores"][s]).mean()), 2)]
        rows.append([len(rows) + 1, obs["n_remaining"], len(feasible), len(obs["unplaceable"]), *state,
                     c["rotation"], c["width"], c["height"], *marks])
        poses.append((int(a["sampled_ids"][feasible[k] % n]), c["rotation"], c["x"], c["y"]))
    return rows, poses, end


def pack(blocks):
    """(the pack of the blocks [(seed, scene)], its walls [{wall, seed, scene, case, filter_policy, label}])."""
    sets, sources = [], []
    for letter, (seed, scene) in zip("ABCDEFGHIJ", blocks):
        picked, n_walls = chosen(seed, scene)
        walls, first = [], None
        for i, (w, label) in enumerate(zip(picked, LABELS), 1):
            rows, poses, end = steps_of(BATCH_RUNS / f"factorial-s{seed}-{scene}" / "cases" / w.wall)
            assert len(rows) == w.n_placed and (end is None) == (w.n_placed == w.n_input), w.wall
            first = first or poses
            differs = next((k + 1 for k, (p, q) in enumerate(zip(poses, first)) if p != q), None)
            walls.append({"wall": f"{letter}{i}", "label": label, "placed": int(w.n_placed),
                          "lost": int(w.n_input - w.n_placed),
                          "metrics": {m: {"value": round(float(w[m]), 4), "z": round(float(w[f"z_{m}"]), 2)}
                                      for m in METRICS},
                          "mean_z": round(float(w.mean_z), 2), "rank": int(w["rank"]),
                          **({} if i == 1 else {"first_placement_differing_from_wall_1": differs}),
                          "placements": rows, **({} if end is None else {"stopped_with": end})})
            sources.append({"wall": f"{letter}{i}", "seed": seed, "scene": scene, "case": w.wall,
                            "filter_policy": w.filter_policy, "label": label})
        sets.append({"stone_set": letter, "n_stones": int(picked[0].n_input), "n_walls_ranked": n_walls,
                     "walls": walls})
    return {"placement_columns": COLUMNS, "stone_sets": sets}, sources


def sha256(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def registered(seed):
    """The committed {'walls', 'pack_sha256'} of a fold (PACKS), None if there is none."""
    return json.loads(PACKS.read_text(encoding="utf-8")).get(str(seed)) if PACKS.exists() else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--without", type=int, choices=SEEDS, help="the test seed whose blocks are left out")
    parser.add_argument("--out", type=Path, help="the pack as JSON")
    parser.add_argument("--write", action="store_true", help=f"store every fold's walls and sha256 in {PACKS.name}")
    args = parser.parse_args(argv)
    if args.write:
        folds = {str(s): dict(zip(("pack_sha256", "walls"), (sha256(p), w)))
                 for s in (*SEEDS, None) for p, w in [pack(fold(s))]}
        PACKS.write_text(json.dumps(folds, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        return 0
    data, sources = pack(fold(args.without))
    if args.out:
        args.out.write_text(json.dumps(data), encoding="utf-8")
    print(pd.DataFrame(sources).to_string(index=False))
    print(f"{len(sources)} walls, {sum(len(w['placements']) for s in data['stone_sets'] for w in s['walls'])} "
          f"placements, {len(json.dumps(data).encode())} bytes, sha256 {sha256(data)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
