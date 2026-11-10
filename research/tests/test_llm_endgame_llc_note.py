"""research/analysis/llm_endgame_llc_note.py: the rule of docs/plans/llm-endgame-llc-note.md."""

import unittest

import pandas as pd

from research.analysis import llm_endgame_llc_note as ln


def table(llm, n=18, pick_safe=0.2, gave_up=1):
    return pd.DataFrame({"llm_mean_z": llm, "cascade_mean_z": 0.0, "low_top_mean_z": 0.2, "random_median": -0.3,
                         "llm_n_placed": n, "cascade_n_placed": 18, "complete": True, "switches": 2,
                         "llm_decisions": 7, "low_top_safe_mean_z": 0.2, "low_top_safe_n_placed": 18,
                         "pick_safe_mean_z": pick_safe, "pick_safe_n_placed": 18, "gave_up_llc": gave_up})


class Rule(unittest.TestCase):
    def test_f_comes_after_the_lookahead_rule(self):
        self.assertEqual(ln.reading(table([0.5] * 9)), "go")
        self.assertEqual(ln.reading(table([0.5] * 9, pick_safe=[0.6] * 5 + [0.4] * 4)),
                         "no endgame contribution shown")
        self.assertEqual(ln.reading(table([0.5] * 9, pick_safe=[0.6] * 4 + [0.5] * 5)), "go")  # a tie counts
        self.assertEqual(ln.reading(table([-0.5] * 5 + [0.5] * 4)), "stop direct pick")

    def test_the_note_against_the_control(self):
        control = table([0.3] * 9, gave_up=2)
        self.assertEqual(ln.note_effect(table([0.4] * 7 + [0.3] * 2), control), "the note helps")
        self.assertEqual(ln.note_effect(table([0.4] * 6 + [0.3] * 3), control),  # a tie does not count in (g)
                         "the LLM picks differently, the wall not shown better")
        self.assertEqual(ln.note_effect(table([0.4] * 9, n=[17] * 3 + [18] * 6), control),
                         "the LLM picks differently, the wall not shown better")
        self.assertEqual(ln.note_effect(table([0.4] * 9, gave_up=2), control), "no effect of the note shown")


if __name__ == "__main__":
    unittest.main()
