"""The progress bar of a wall and the number of workers of research/runner/run_wall.py."""

import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from research.runner import run_wall
from research.walls import case_definition

ENTER = {"event": "build_enter", "elapsed_seconds": 1.0, "input_stones": 40}


class Terminal(io.StringIO):
    def isatty(self):
        return True


class StonesGiven(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.output = Path(tmp.name)

    def events(self, text):
        (self.output / "events.jsonl").write_text(text, encoding="utf-8")

    def test_before_the_case_has_written_anything(self):
        self.assertIsNone(run_wall.stones_given(self.output))

    def test_before_the_building_loop(self):
        self.events('{"event": "configured"}\n')
        self.assertIsNone(run_wall.stones_given(self.output))

    def test_from_the_event(self):
        self.events('{"event": "configured"}\n' + json.dumps(ENTER) + "\n")
        self.assertEqual(run_wall.stones_given(self.output), 40)

    def test_a_line_still_being_written_is_not_read(self):
        self.events('{"event": "build_enter", "input_st')
        self.assertIsNone(run_wall.stones_given(self.output))


class Line(unittest.TestCase):
    def test_the_count_share_and_time(self):
        self.assertEqual(
            run_wall.bar_line("[3/12] stones", 10, 40, 20.0),
            "[3/12] stones: 10/40 [######..................] 20 s, 60 s left",
        )

    def test_before_the_first_stone(self):
        self.assertEqual(run_wall.bar_line("x", 0, 4, 9.0), "x: 0/4 [........................] 9 s, time left unknown")

    def test_a_total_not_known_yet(self):
        self.assertEqual(run_wall.bar_line("x", 2, None, 5.4), "x: 2 stones, 5 s")

    def test_an_overrun_stops_at_the_total(self):
        self.assertEqual(run_wall.bar_line("x", 17, 3, 44.0), "x: 3/3 [########################] 44 s, 0 s left")


class Bar(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.output = Path(tmp.name)
        self.result = self.output / "result_x"
        self.result.mkdir()
        self.terminal = Terminal()
        patcher = mock.patch.object(sys, "stderr", self.terminal)
        patcher.start()
        self.addCleanup(patcher.stop)

    def bar(self, limit=0):
        return run_wall.StoneBar(self.output, limit, "label")

    def committed(self, stones):
        (self.result / "decisions.txt").write_text("row\n" * stones, encoding="utf-8")

    def enter(self):
        (self.output / "events.jsonl").write_text(json.dumps(ENTER) + "\n", encoding="utf-8")

    def last_line(self):
        return self.terminal.getvalue().rsplit("\r", 1)[-1]

    def test_the_stones_so_far_out_of_the_stones_given(self):
        stones = self.bar()
        self.enter()
        self.committed(3)
        stones.refresh()
        self.assertIn("label: 3/40 ", self.last_line())

    def test_a_limit_below_the_stones_given_is_the_total(self):
        stones = self.bar(limit=5)
        self.enter()
        stones.refresh()
        self.assertIn("/5 ", self.last_line())

    def test_a_limit_above_the_stones_given_is_not(self):
        stones = self.bar(limit=500)
        self.enter()
        stones.refresh()
        self.assertIn("/40 ", self.last_line())

    def test_the_total_stays_unknown_until_the_building_loop(self):
        stones = self.bar()
        self.committed(2)
        stones.refresh()
        self.assertIn("label: 2 stones", self.last_line())

    def test_a_shorter_line_overwrites_a_longer_one(self):
        stones = self.bar()
        stones.refresh()
        self.assertGreaterEqual(len(self.last_line()), run_wall.BAR_LINE_WIDTH)

    def test_a_file_that_cannot_be_read_does_not_stop_the_case(self):
        stones = self.bar()
        for error in (PermissionError("in use"), ValueError("half a line")):
            with self.subTest(error), mock.patch.object(run_wall, "stones_given", side_effect=error):
                stones.refresh()

    def test_leaving_shows_the_last_count_and_ends_the_line(self):
        with self.bar():
            self.committed(7)
        self.assertIn("label: 7 stones", self.last_line())
        self.assertTrue(self.terminal.getvalue().endswith("\n"))

    def test_without_a_terminal_nothing_is_read_or_written(self):
        piped = io.StringIO()
        with mock.patch.object(sys, "stderr", piped), mock.patch.object(run_wall, "committed_placements") as counted:
            with self.bar():
                pass
        counted.assert_not_called()
        self.assertEqual(piped.getvalue(), "")


class RegisteredEnvironment(unittest.TestCase):
    def test_a_case_process_can_import_run_wall_without_the_user_site(self):
        """A controller wall imports run_wall in the case process, which has PYTHONNOUSERSITE=1."""
        code = "import research.runner.run_wall, research.runner.batch"
        done = subprocess.run([sys.executable, "-s", "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr[-500:])


class Workers(unittest.TestCase):
    def test_no_number_keeps_the_script(self):
        self.assertEqual(case_definition.with_workers("set_number_cpu(10)\n", None), "set_number_cpu(10)\n")

    def test_the_number_replaces_the_scripts_own(self):
        self.assertEqual(case_definition.with_workers("a\nset_number_cpu(10)\nb\n", 4), "a\nset_number_cpu(4)\nb\n")

    def test_every_case_has_its_number_once(self):
        for case in case_definition.CASES:
            with self.subTest(case):
                self.assertIn("set_number_cpu(3)", case_definition.with_workers(case_definition.source_text(case), 3))

    def test_a_script_without_the_call_is_refused(self):
        with self.assertRaises(AssertionError):
            case_definition.with_workers("nothing here\n", 4)


if __name__ == "__main__":
    unittest.main()
