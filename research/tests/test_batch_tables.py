"""research/analysis/batch_tables.py on a hand-made batch folder: which status records become walls, the
deduplication key (scene, seed, placement sequence) and the shapes of the three tables."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from research.analysis import batch_tables

PLAN = "case,wall_type,filter_policy,seed,controller_seed\n1,TH_T80,30124,7,\n2,TH_T80,01234,7,\n3,AL_R,random,7,99\n"
FIRST = [(1, 0, 10, 2), (2, 90, 40, 2)]
SECOND = [(2, 90, 40, 2), (1, 0, 10, 2)]


def batch_folder(root, records):
    """A batch folder with PLAN, the status records in order, and a receipt for each case id."""
    folder = Path(root) / "demo"
    (folder / "cases").mkdir(parents=True)
    (folder / "plan.csv").write_text(PLAN, encoding="utf-8")
    (folder / "status.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records) + "{cut", encoding="utf-8")
    for record in records:
        if record.get("case_id"):
            (folder / "cases" / record["case_id"]).mkdir(exist_ok=True)
            receipt = {"git_head": "abc", "status": record["status"]}
            (folder / "cases" / record["case_id"] / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return folder


def walls_table(rows):
    """walls.csv columns unique_table reads, from (case, scene, seed, wall) tuples."""
    return pd.DataFrame(
        [
            dict(case=case, wall_type=scene, seed=seed, wall=wall, filter_policy=f"p{case}")
            for case, scene, seed, wall in rows
        ]
    )


def steps_table(sequences):
    """steps.csv columns placement_signatures reads: every (stone, rot, x, y) of {wall: placements} as placed."""
    return pd.DataFrame(
        [
            dict(wall=wall, outcome="placed", stone=s, rot=r, x=x, y=y)
            for wall, poses in sequences.items()
            for s, r, x, y in poses
        ]
    )


class CaseRecords(unittest.TestCase):
    def test_the_latest_record_with_a_case_id_counts_in_plan_order(self):
        records = [
            {"case": 2, "status": "RAISED", "error": "boom"},
            {"case": 1, "status": "PROCESS_PASS", "case_id": "w1"},
            {"case": 2, "status": "PROCESS_PASS", "case_id": "w2"},
        ]
        with tempfile.TemporaryDirectory() as root:
            found = batch_tables.case_records(batch_folder(root, records))
            self.assertEqual([(row["case"], record["case_id"]) for row, record in found], [(1, "w1"), (2, "w2")])
            limited = batch_tables.case_records(batch_folder(Path(root) / "x", records), numbers={2})
            self.assertEqual([record["case_id"] for _, record in limited], ["w2"])


class UniqueTable(unittest.TestCase):
    def test_same_scene_seed_and_sequence_share_the_first_wall_by_case(self):
        walls = walls_table([(2, "TH_T80", 7, "b"), (1, "TH_T80", 7, "a"), (3, "TH_T80", 7, "c")])
        walls, unique = batch_tables.unique_table(walls, steps_table({"a": FIRST, "b": FIRST, "c": SECOND}))
        self.assertEqual(dict(zip(walls["wall"], walls["unique_id"])), {"a": "a", "b": "a", "c": "c"})
        self.assertEqual(unique[["unique_id", "n_walls", "cases", "filter_policies"]].values.tolist(),
                         [["a", 2, "1 2", "p1 p2"], ["c", 1, "3", "p3"]])

    def test_scene_and_seed_are_part_of_the_key(self):
        walls = walls_table([(1, "TH_T80", 7, "a"), (2, "TH_T20", 7, "b"), (3, "TH_T80", 8, "c")])
        walls, _ = batch_tables.unique_table(walls, steps_table({"a": FIRST, "b": FIRST, "c": FIRST}))
        self.assertEqual(walls["unique_id"].tolist(), ["a", "b", "c"])

    def test_a_wall_without_steps_is_its_own_unique_wall(self):
        walls = walls_table([(1, "TH_T80", 7, "a"), (2, "TH_T80", 7, "b")])
        walls, unique = batch_tables.unique_table(walls, pd.DataFrame())
        self.assertEqual(walls["unique_id"].tolist(), ["a", "b"])
        self.assertEqual(len(unique), 2)


class BuildTables(unittest.TestCase):
    def test_walls_without_trace_keep_a_row(self):
        records = [{"case": c, "status": "PROCESS_PASS", "case_id": f"w{c}"} for c in (1, 2, 3)]
        with tempfile.TemporaryDirectory() as root:
            walls, steps, unique = batch_tables.build_tables(batch_folder(root, records))
        self.assertEqual(walls["wall"].tolist(), ["w1", "w2", "w3"])
        self.assertEqual(walls["n_steps"].tolist(), [0, 0, 0])
        self.assertEqual(walls.loc[2, ["filter_policy", "controller_seed", "git_head"]].tolist(), ["random", 99, "abc"])
        self.assertTrue({"F_SF", "F_AH", "F_AV", "F_LR", "quality_err", "unique_id"} <= set(walls.columns))
        self.assertEqual((len(steps), len(unique)), (0, 3))

    def test_quality_errors_name_no_absolute_path(self):
        with tempfile.TemporaryDirectory() as root:
            folder = batch_folder(root, [{"case": 1, "status": "PROCESS_PASS", "case_id": "w1"}])
            walls = pd.DataFrame([dict(wall="w1", status="PROCESS_PASS", result="cases/w1/result")])
            error = batch_tables.quality_columns(folder, walls).loc[0, "quality_err"]
        self.assertIn("demo", error)
        self.assertNotIn(root, error)
        self.assertNotIn(repr(root)[1:-1], error)


if __name__ == "__main__":
    unittest.main()
