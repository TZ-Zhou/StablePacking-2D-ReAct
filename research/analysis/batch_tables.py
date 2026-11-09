"""Per-wall, per-step and deduplication tables of one batch of research/runner/batch.py (offline, A1).

Usage: python -B -m research.analysis.batch_tables <batch folder> [--out DIR] [--cases 3-10,15] [--no-quality]

<batch folder> is runs/batch/<name> of the checkout that built it (it is only read; no wall is built). Writes once,
to <out> (default research/datasets/<name>):
- walls.csv: one row per case of status.jsonl: its plan fields (scene, seed, filter_policy, controller_seed), status,
  git_head, wall_metrics values, the formal quality F_SF, F_AH, F_AV, F_LR of wall_quality(..., 'corrected') for
  PROCESS_PASS walls (quality_err when it cannot be computed, e.g. a scene without an original-resolution label
  image) and unique_id.
- steps.csv: one row per traced sampling attempt, the columns of myopia_tables.step_row (candidate and feasible
  counts, what the 120 filter orders reach, the wall state, the chosen candidate and its scores), keyed by wall.
- unique.csv: one row per distinct wall, keyed by scene, seed and placement sequence (stone, rot, x, y): the walls
  and filter policies that built it. Which analysis deduplicates is decided per analysis
  (docs/plans/llm-agent-design-questions.md Q5).
- dataset.json: the source batch (sha256 of status.jsonl, plan.csv, config.yaml and the receipts), the code's git
  head and source blobs, and counts. No time and no absolute path, so a rerun writes the same bytes.
Refuses an existing <out> and source files that differ from git HEAD. A wall that cannot be read is reported on
stderr and kept with empty columns.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from research.analysis import myopia_tables as mt
from research.paths import DATASETS_DIR, REPO
from research.runner.batch import statuses
from research.walls import batch_plan
from research.walls import campaign_quality as cq

SOURCES = ["research/analysis/batch_tables.py", "research/analysis/myopia_tables.py", "research/walls/wall_metrics.py",
           "Tests/Thangavelu/02_view_transformation_ori_reso.py", "Tests/Thangavelu/03_evaluate_wall_metrics.py",
           "Tests/Thangavelu/graph.py", "Stonepacker2D"]
PLAN_COLUMNS = ("case", "wall_type", "filter_policy", "seed", "controller_seed")
BATCH_FILES = ("status.jsonl", "plan.csv", "config.yaml")


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def case_records(folder, numbers=None):
    """(plan row, status record) of every case with a finished record, in case order; numbers limits the cases."""
    records = statuses(folder)
    chosen = [row for row in batch_plan.read_plan(folder / "plan.csv") if numbers is None or row["case"] in numbers]
    return [(row, records[row["case"]]) for row in chosen if records.get(row["case"], {}).get("case_id")]


def wall_row(folder, row, record, result, n_steps, metrics):
    """One walls.csv row before the quality and unique_id columns."""
    receipt = json.loads((folder / "cases" / record["case_id"] / "receipt.json").read_text(encoding="utf-8"))
    return {
        **{key: row[key] for key in PLAN_COLUMNS},
        "wall": record["case_id"],
        "status": record["status"],
        "git_head": receipt.get("git_head"),
        "result": result.relative_to(folder).as_posix() if result else "",
        "n_steps": n_steps,
        **metrics,
    }


def extract_case(folder, row, record):
    """(step rows, wall row) of one case; a wall without a readable trace keeps its row with no steps."""
    wall_id, run = record["case_id"], folder / "cases" / record["case_id"]
    result = mt.traced_result(run)
    if result is None:
        mt.warn(wall_id, "steps", "no trace")
        return [], wall_row(folder, row, record, None, 0, {})
    steps = mt.readable_steps(wall_id, result) or []
    return steps, wall_row(folder, row, record, result, len(steps), mt.readable_metrics(wall_id, run, result))


def quality_columns(folder, walls):
    """Quality columns of the PROCESS_PASS walls (the slow step); quality_err holds why a wall has none."""
    finished = walls[(walls["status"] == mt.FINISHED_STATUS) & (walls["result"] != "")]
    rows = [mt.quality_row(wall, folder / "cases" / wall, folder / result)
            for wall, result in zip(finished["wall"], finished["result"])]
    table = pd.DataFrame(rows, columns=["wall", *mt.QUALITY_COLUMNS, "n_left_out", "err"])
    table["err"] = table["err"].fillna("")  # all NaN when every wall has quality, and .str needs strings
    for absolute in (str(folder), repr(str(folder))[1:-1]):  # no absolute path in a dataset, also as an error's repr
        table["err"] = table["err"].str.replace(absolute, folder.name, regex=False)
    return table.rename(columns={"err": "quality_err"})


def unique_table(walls, steps):
    """(walls with unique_id, unique.csv): one id per (scene, seed, placement sequence), the first wall by case.

    A wall without steps has no known placement sequence and is its own unique wall.
    """
    walls = walls.sort_values("case").reset_index(drop=True)
    signatures = mt.placement_signatures(steps) if len(steps) else {}
    keys = [(scene, seed, signatures[wall]) if wall in signatures else (wall,)
            for scene, seed, wall in zip(walls["wall_type"], walls["seed"], walls["wall"])]
    first = {}
    for key, wall in zip(keys, walls["wall"]):
        first.setdefault(key, wall)
    walls = walls.assign(unique_id=[first[key] for key in keys])
    unique = (
        walls.groupby("unique_id", sort=False)
        .agg(scene=("wall_type", "first"), seed=("seed", "first"), n_walls=("wall", "size"),
             cases=("case", lambda c: " ".join(map(str, c))), filter_policies=("filter_policy", " ".join))
        .reset_index()
    )
    return walls, unique


def build_tables(folder, numbers=None, quality=True):
    """(walls, steps, unique) DataFrames of a batch folder."""
    extracted = [extract_case(folder, row, record) for row, record in case_records(folder, numbers)]
    steps = pd.DataFrame([step for rows, _ in extracted for step in rows])
    walls = pd.DataFrame([wall for _, wall in extracted])
    if quality:
        walls = walls.merge(quality_columns(folder, walls), on="wall", how="left")
    walls, unique = unique_table(walls, steps)
    return walls, steps, unique


def provenance(folder, walls, steps, unique, quality):
    """dataset.json: what the tables were made from and by which code."""
    hashes, dirty = cq.sources(SOURCES)
    receipts = "".join(f"{wall} {cq.sha256(folder / 'cases' / wall / 'receipt.json')}\n" for wall in walls["wall"])
    return {
        "batch": folder.name,
        "batch_files_sha256": {name: cq.sha256(folder / name) for name in BATCH_FILES if (folder / name).exists()},
        "receipts_sha256": sha256_text(receipts),
        "cases": walls["case"].tolist(),
        "git_head": cq.git("rev-parse", "HEAD").strip(),
        "sources_git_blob": hashes,
        "sources_dirty": dirty,
        "quality": f"wall_quality {mt.QUALITY_CANVAS} canvas" if quality else None,
        "n_walls": len(walls),
        "n_steps": len(steps),
        "n_unique": len(unique),
        "n_quality": int(walls["F_LR"].notna().sum()) if quality else 0,
        "columns": {"walls": list(walls.columns), "steps": list(steps.columns), "unique": list(unique.columns)},
    }


def write_tables(out, walls, steps, unique, info):
    out.mkdir(parents=True)
    for name, table in (("walls", walls), ("steps", steps), ("unique", unique)):
        table.to_csv(out / f"{name}.csv", index=False, lineterminator="\n")
    (out / "dataset.json").write_text(json.dumps(info, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("batch", type=Path, help="runs/batch/<name> folder")
    parser.add_argument("--out", type=Path, help="default research/datasets/<name>")
    parser.add_argument("--cases", help="case numbers, e.g. 3-10,15 (default: all)")
    parser.add_argument("--no-quality", action="store_true", help="leave out the slow quality columns")
    args = parser.parse_args(argv)
    folder = args.batch.resolve()
    out = (args.out or DATASETS_DIR / folder.name).resolve()
    if out.exists():
        raise SystemExit(f"{out} exists: a dataset is written once")
    if REPO.resolve() not in out.parents:
        raise SystemExit(f"{out} is outside the repository")
    hashes, dirty = cq.sources(SOURCES)
    if dirty:
        raise SystemExit(f"source files differ from HEAD: {dirty}")
    total = len(batch_plan.read_plan(folder / "plan.csv"))
    numbers = set(batch_plan.chosen_numbers(total, args.cases)) if args.cases else None
    walls, steps, unique = build_tables(folder, numbers, quality=not args.no_quality)
    info = provenance(folder, walls, steps, unique, not args.no_quality)
    if (info["sources_git_blob"], info["sources_dirty"]) != (hashes, dirty):
        raise SystemExit("a source file changed while the tables were made: nothing written")
    write_tables(out, walls, steps, unique, info)
    print(f"{len(walls)} walls, {len(steps)} steps, {len(unique)} unique walls written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
