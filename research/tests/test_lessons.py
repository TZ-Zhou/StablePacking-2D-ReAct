"""docs/plans/llm-lessons.md: the pack (research/analysis/lessons_pack.py), the induction (research/llm_walls/
lessons.py), the lessons in the whole-wall prompt (research/controller/pick.py), the lessons walls
(research/llm_walls/lessons_wall.py) and the reading (research/analysis/llm_lessons.py)."""

import json
import re
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from research.analysis import lessons_pack, llm_lessons
from research.controller import order_controller as oc
from research.controller import pick
from research.llm_walls import gpt_smoke, lessons, lessons_wall, whole_wall

PICK_4 = "c9b8128adf32529b074a7e22ae1013acb47759829bcb2c787527281f939c4242"  # the paid whole-wall prompt
TEXTS = ["Prefer a candidate that keeps the top of the wall level.", "Check what a candidate leaves."]


class Scripted:
    """An LLM that answers commit_lessons with each of `firsts` in turn."""
    model, params = "scripted", {}

    def __init__(self, *firsts):
        self.firsts = list(firsts)

    def __call__(self, messages, tools):
        args = {f"lesson_{i}": "" for i in range(1, 6)}
        return {"thought": "", "tool": "commit_lessons", "arguments": {**args, "lesson_1": self.firsts.pop(0)}}


class Pack(unittest.TestCase):
    def test_two_good_walls_the_worst_complete_one_and_the_one_with_the_fewest_stones(self):
        walls = pd.DataFrame({"filter_policy": list("abcdef"), "n_placed": [9, 9, 9, 9, 8, 7],
                              "mean_z": [0.1, 0.9, -0.5, 0.4, 2.0, 1.0]})
        self.assertEqual([w.filter_policy for w in lessons_pack.select(walls)], ["b", "d", "c", "f"])
        with self.assertRaises(ValueError):  # no wall lost a stone
            lessons_pack.select(walls[walls.n_placed == 9])

    def test_a_fold_has_no_wall_of_its_seed_and_names_nothing(self):
        pack, walls = lessons_pack.pack(lessons_pack.fold(313))
        text = json.dumps(pack)
        self.assertEqual(({w["seed"] for w in walls}, len(walls)), ({484, 717, 720, 761}, 32))
        self.assertEqual(lessons_pack.sha256(pack), lessons_pack.registered(313)["pack_sha256"])
        self.assertIsNone(re.search(r"S\d|TH_T|seed|case|filter_policy|2026", text))
        self.assertEqual([len(s["walls"]) for s in pack["stone_sets"]], [4] * 8)
        self.assertEqual({len(row) for s in pack["stone_sets"] for w in s["walls"] for row in w["placements"]},
                         {len(lessons_pack.COLUMNS)})
        self.assertEqual(set(json.loads(lessons_pack.PACKS.read_text())), {"313", "484", "717", "720", "761", "None"})


class Prompt(unittest.TestCase):
    def test_lessons_are_one_to_five_short_lines_that_name_nothing(self):
        self.assertIsNone(oc.lessons_problem(TEXTS))
        for bad in ([], TEXTS * 3, "a lesson", [""], [" padded"], ["two\nlines"], [" ".join(["word"] * 61)],
                    ["Place S7 early."], ["As in wall B3."], [7]):
            with self.subTest(bad):
                self.assertIsNotNone(oc.lessons_problem(bad))

    def test_the_lessons_follow_the_whole_wall_prompt_and_without_them_it_is_unchanged(self):
        llm = oc.MockLLM("pick_safe")
        plain = pick.PickPolicy(llm, max_tool_calls=4, lookahead=True, whole_wall=True)
        taught = pick.PickPolicy(llm, max_tool_calls=4, lookahead=True, whole_wall=True, lessons=TEXTS)
        self.assertEqual((plain.prompt_sha256, plain.system), (PICK_4, pick.SYSTEM_WHOLE))
        self.assertEqual(taught.system, pick.SYSTEM_WHOLE + pick.LESSONS + f"\n1. {TEXTS[0]}\n2. {TEXTS[1]}")
        self.assertEqual((taught.tools, taught.prompt_sha256 != PICK_4), (plain.tools, True))
        whole = {"kind": "pick", "llm": {"kind": "mock", "mode": "pick_safe"}, "lookahead": True, "whole_wall": True}
        oc.validate_spec({**whole, "lessons": TEXTS})
        for spec in ({**whole, "lessons": []}, {**whole, "whole_wall": False, "lessons": TEXTS}):
            with self.subTest(spec), self.assertRaises(ValueError):
                oc.validate_spec(spec)


class Induction(unittest.TestCase):
    def test_a_lesson_that_breaks_the_format_is_rejected_once_with_the_reason(self):
        long = " ".join(["word"] * 61)
        reply = lessons.LessonsPolicy(Scripted(long, " " + TEXTS[0])).decide({"observation": {"stone_sets": []}})
        self.assertEqual((reply["source"], reply["lessons"]), ("llm", TEXTS[:1]))
        self.assertIn("61 words", reply["react"][0]["feedback"]["error"])
        reply = lessons.LessonsPolicy(Scripted(long, "Place S7 early.")).decide({"observation": {}})
        self.assertEqual((reply["source"], reply["lessons"]), ("fallback:no_valid_commit", None))
        short = {"tool": "commit_lessons", "arguments": {"lesson_1": "x"}}  # a strict schema: all five texts
        self.assertEqual(oc.reply_problem(short, [lessons.COMMIT])[:7], "missing")

    def test_the_command_line_runs_the_module_the_worker_can_import(self):
        with mock.patch.object(lessons, "main", return_value=0) as main, self.assertRaises(SystemExit) as stop:
            runpy.run_module("research.llm_walls.lessons", run_name="__main__")
        self.assertEqual((main.call_count, stop.exception.code, lessons.serve.__module__),
                         (1, 0, "research.llm_walls.lessons"))

    def test_a_wall_gets_the_lessons_of_the_fold_without_its_seed(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(lessons, "STUDIES_DIR", Path(tmp)):
            with self.assertRaises(SystemExit):  # no induction yet
                lessons_wall.spec_of(313)
            folder = lessons.folder_of(313)
            folder.mkdir(parents=True)
            gpt_smoke.write_json(folder / "lessons.json", {"lessons": TEXTS})
            for journal, ok in (({"state": "finished", "walls": [{"seed": 484}]}, True),
                                ({"state": "uncertain", "walls": [{"seed": 484}]}, False),
                                ({"state": "finished", "walls": [{"seed": 313}]}, False)):
                gpt_smoke.write_json(folder / "journal.json", journal)
                with self.subTest(journal):
                    if ok:
                        self.assertEqual(lessons_wall.spec_of(313), {**whole_wall.SPEC, "lessons": TEXTS})
                    else:
                        self.assertRaises(SystemExit, lessons_wall.spec_of, 313)
        self.assertEqual((lessons_wall.spec_problems(), len(set(lessons_wall.STATES))), ([], 10))
        self.assertEqual(lessons_wall.name_of(313, "TH_T80"), "lessons-s313-th-t80")


class Reading(unittest.TestCase):
    def test_a_win_is_not_below_on_both_and_above_on_one(self):
        def wall(z, n, complete=True):
            return {"mean_z": z, "n_placed": n, "complete": complete}
        for a, b, out in ((wall(1, 18), wall(0, 18), "win"), (wall(0, 18), wall(0, 17), "win"),
                          (wall(0, 18), wall(0, 18), "tie"), (wall(1, 17), wall(0, 18), "tie"),
                          (wall(0, 18), wall(1, 18), "loss"), (wall(5, 18, False), wall(0, 18), "loss")):
            with self.subTest(a=a, b=b):
                self.assertEqual(llm_lessons.outcome(a, b), out)

    def test_the_sign_test_screens_at_one_in_five(self):
        self.assertEqual((llm_lessons.sign_p(3, 0), llm_lessons.sign_p(7, 3), llm_lessons.sign_p(0, 0)),
                         (0.125, 176 / 1024, 1.0))
        for wins, losses, out in ((7, 3, "a sign of help, worth confirming"), (6, 3, "no difference seen"),
                                  (2, 0, "no difference seen"), (1, 4, "possibly harmful")):
            with self.subTest(wins=wins, losses=losses):
                self.assertEqual(llm_lessons.reading(["win"] * wins + ["loss"] * losses + ["tie"]), out)


if __name__ == "__main__":
    unittest.main()
