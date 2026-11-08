"""A new case keeps data only: figures are skipped, limit loads are in mechanism.csv, leftovers are removed."""

import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from research.validation import windows_case_runtime as runtime
from research.runner import run_wall
from research.walls import wall_metrics
from Stonepacker2D.stack.selection import load_trace

REPO = Path(__file__).resolve().parents[2]
RECORDED = (
    REPO
    / "research"
    / "runs"
    / "campaigns"
    / "tool-validation-fft"
    / "fft-d2-0"
    / "paper_3-0-2-1-4"
    / "seed20"
    / "result_20261106-205510_wall0"
)


class Figures(unittest.TestCase):
    def test_no_figure_reaches_the_disk(self):
        base = importlib.import_module("Stonepacker2D.entity.base").Base
        figure = matplotlib.figure.Figure
        restored = [
            (plt, "savefig"),
            (figure, "savefig"),
            (cv2, "imwrite"),
            (base, "draw_assembly"),
            (base, "draw_last_action"),
        ]
        patches = [mock.patch.object(owner, name, getattr(owner, name)) for owner, name in restored]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        runtime._skip_figures()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            drawn = plt.figure()
            self.addCleanup(plt.close, drawn)
            plt.savefig(folder / "a.png")
            drawn.savefig(folder / "b.pdf")
            self.assertTrue(cv2.imwrite(str(folder / "c.png"), np.zeros((2, 2), np.uint8)))
            base.draw_assembly(None, file_name="d.pdf")
            self.assertEqual(list(folder.iterdir()), [])


class LimitLoads(unittest.TestCase):
    def test_the_table_of_a_new_wall_is_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = "side,limit_load\nleft,0.40642658684885175\nright,0.5\n"
            (Path(tmp) / "mechanism.csv").write_text(table, encoding="utf-8")
            self.assertEqual(wall_metrics.mechanism_lambdas(tmp), {"left": 0.40642658684885175, "right": 0.5})

    def test_a_side_without_a_single_value_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "mechanism.csv").write_text("side,limit_load\nleft,0.1\nleft,0.2\n", encoding="utf-8")
            self.assertEqual(wall_metrics.mechanism_lambdas(tmp), {"left": None, "right": None})


class Leftovers(unittest.TestCase):
    def case(self, tmp):
        output = Path(tmp)
        for folder in ("data", "matplotlib-config", "result_x/img", "result_x/record", "result_x/trace", "result/img"):
            (output / folder).mkdir(parents=True)
        for file in (
            "data/a.png",
            "windows_case_runtime.py",
            "01_run.py",
            "receipt.json",
            "result_x/img/selection.txt",
            "result_x/trace/commits.jsonl",
            "result_x/base.pkl",
            "result/img/selection.txt",
        ):
            (output / file).write_text("x", encoding="utf-8")
        return output

    def test_only_data_remains(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = self.case(tmp)
            run_wall.keep_data_only(output)
            kept = sorted(p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file())
            folders = sorted(p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_dir())
            self.assertEqual(kept, ["01_run.py", "receipt.json", "result_x/base.pkl", "result_x/trace/commits.jsonl"])
            self.assertEqual(folders, ["result", "result_x", "result_x/trace"])

    def test_a_folder_still_holding_data_stays(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = self.case(tmp)
            (output / "result_x" / "img" / "kept.csv").write_text("x", encoding="utf-8")
            run_wall.keep_data_only(output)
            self.assertTrue((output / "result_x" / "img" / "kept.csv").is_file())

    def test_placements_are_counted_in_either_result_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = self.case(tmp)
            (output / "data" / "decisions.txt").write_text("no\n", encoding="utf-8")
            (output / "result_x" / "decisions.txt").write_text("1\n2\n", encoding="utf-8")
            (output / "result" / "decisions.txt").write_text("3\n", encoding="utf-8")
            self.assertEqual(run_wall.committed_placements(output), 3)


class StepTimes(unittest.TestCase):
    def test_every_attempt_gets_its_own_and_its_cumulative_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = Path(tmp) / "events.jsonl"
            lines = [("build_enter", 10.0), ("step_commit", 12.5), ("other", 13.0), ("step_commit", 16.0)]
            events.write_text(
                "".join(json.dumps({"event": e, "elapsed_seconds": t}) + "\n" for e, t in lines), encoding="utf-8"
            )
            expected = [{"wall_time_s": 2.5, "wall_time_cum_s": 2.5}, {"wall_time_s": 3.5, "wall_time_cum_s": 6.0}]
            self.assertEqual(wall_metrics.step_wall_times(events), expected)

    def test_a_commit_leaves_an_event_after_its_own_line(self):
        build_wall = importlib.import_module("Stonepacker2D.stack.build_wall")
        written = []
        namespace = {"build_wall_vendor": None, "evaluate_kine": None}
        with tempfile.TemporaryDirectory() as tmp:
            events = Path(tmp) / "events.jsonl"
            with (
                mock.patch.object(build_wall, "trace_commit", written.append),
                mock.patch.object(build_wall, "Pool", build_wall.Pool),
                mock.patch.object(runtime, "_skip_figures"),
                mock.patch.dict(os.environ, {"BASELINE_CASE_EVENTS": str(events)}),
            ):
                runtime.configure(namespace)
                build_wall.trace_commit({"step": 4, "outcome": "placed"})
            record = json.loads(events.read_text(encoding="utf-8"))
            self.assertEqual(written, [{"step": 4, "outcome": "placed"}])
            self.assertEqual((record["event"], record["step"], record["outcome"]), ("step_commit", 4, "placed"))


@unittest.skipUnless(RECORDED.exists(), "records not present")
class RecordedWall(unittest.TestCase):
    def commits(self):
        return wall_metrics.read_jsonl(RECORDED / "trace" / "commits.jsonl")

    def test_scores_and_order_are_the_selected_candidates(self):
        rows = wall_metrics.step_rows(RECORDED)
        placed = [(r, c) for r, c in zip(rows, self.commits()) if c["outcome"] == "placed"]
        self.assertTrue(placed)
        for row, commit in placed:
            event = commit["sides"][commit["chosen_side"]]["event"]
            trace = load_trace(RECORDED / "trace" / f"e{event:05d}.npz")[1]
            raw = trace["features_raw"][int(trace["selected"])]
            self.assertEqual(
                [row[f"0{k}_{n}"] for k, n in enumerate(("LLC", "NHR", "IWR", "GDS", "LFR"), 1)],
                [raw[3], raw[0], raw[2], raw[1], raw[4]],
            )
            self.assertEqual(row["filter_order"], "LLC-NHR-IWR-GDS-LFR")

    def test_each_placement_is_measured_on_the_wall_of_the_next_snapshot(self):
        walls = []

        def solve(wall):
            walls.append(wall)
            return {"left": 0.29, "right": 0.58}

        with mock.patch.object(wall_metrics, "lambdas", solve):
            quality = wall_metrics.partial_quality(RECORDED)
        commits = self.commits()
        self.assertEqual([bool(q) for q in quality], [c["outcome"] == "placed" for c in commits])
        placed = [q for q in quality if q]
        self.assertAlmostEqual(placed[-1]["F_LR"], 0.29 / wall_metrics.MU)
        whole = wall_metrics.wang_metrics(RECORDED)
        self.assertEqual(
            (placed[-1]["F_SF"], placed[-1]["F_AH"], placed[-1]["F_AV"]),
            (whole["F_SF_author"], whole["F_AH"], whole["F_AV"]),
        )
        for wall, i in zip(walls, [i for i, c in enumerate(commits[:-1]) if c["outcome"] == "placed"]):
            event = commits[i + 1]["sides"][commits[i + 1]["chosen_side"]]["event"]
            np.testing.assert_array_equal(
                wall, load_trace(RECORDED / "trace" / f"e{event:05d}.npz")[1]["wall"], str(commits[i]["step"])
            )


class Inputs(unittest.TestCase):
    def test_a_case_without_a_copy_reads_the_stones_from_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = "wall_image = _root_dir+'/data/stones_noise_80_labels.png'\n_rescale = 0.5\n"
            (Path(tmp) / "00_run.py").write_text(script, encoding="utf-8")
            expected = REPO / "Tests" / "Thangavelu" / "data" / "stones_noise_80_labels.png"
            self.assertEqual(wall_metrics.stone_source(tmp), expected)
            self.assertTrue(expected.is_file())


if __name__ == "__main__":
    unittest.main()
