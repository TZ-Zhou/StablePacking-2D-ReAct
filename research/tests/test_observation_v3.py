"""Observation version 3 (research/controller/order_controller.py Observer.version_3): the shapes of every stone not
in the wall, as build_wall passes them (selection.remaining_shapes) and check_run tolerates them on older walls."""

import json
import types
import unittest

import numpy as np

from research.controller import order_controller
from research.walls import check_run
from Stonepacker2D.stack.selection import REMAINING_SHAPE_KEYS, remaining_shapes

PAPER = {"policy": "paper", "order": (3, 0, 2, 1, 4), "tol": 1e-9}
V3_SPEC = {"kind": "react", "llm": {"kind": "mock", "mode": "keep"}, "observation_version": 3}


def context(rename=lambda s: s, shapes=True):
    """Stones 5 and 6 sampled (one candidate each), 7 and 8 never sampled; rename maps the stone ids."""
    wall = np.zeros((10, 8))
    wall[:2] = 1
    wall[-1], wall[:, 0], wall[:, -1] = 255, 255, 255
    ids = [rename(s) for s in (5, 6, 7, 8)]
    ctx = {"step": 1, "side": "left", "phase": "single", "sample_trial": 0, "wall": wall, "all_bricks": None,
           "sampled_ids": ids[:2], "inventory_ids": ids[2:], "x": [2, 4], "y": [2, 2], "distance": [0.0, 0.0],
           "features_raw": np.array([[1, 0, 0, 1, 0, 0], [0, 1, 0, 1, 0, 0]], float), "weights": [1] * 6,
           "width": np.array([3, 2]), "height": np.array([2, 2])}
    if shapes:
        ctx.update(zip(REMAINING_SHAPE_KEYS, (np.array(ids), np.array([3, 2, 4, 5]), np.array([2, 2, 3, 1]),
                                              np.array([6, 4, 9, 5]))))
    return ctx


class Version3(unittest.TestCase):
    def test_spec(self):
        spec = order_controller.validate_spec(V3_SPEC)
        self.assertEqual(order_controller.observation_version(spec), 3)
        self.assertIn("remaining.shapes", order_controller.PROMPTS[3][0])

    def test_shapes_list_seen_stones_by_label_then_the_rest_largest_first(self):
        obs, _ = order_controller.Observer(version=3)(PAPER, context())
        self.assertEqual(obs["observation_version"], 3)
        self.assertEqual(obs["remaining"]["shapes"], [
            {"stone": "S1", "width": 3, "height": 2, "area": 6}, {"stone": "S2", "width": 2, "height": 2, "area": 4},
            {"stone": None, "width": 4, "height": 3, "area": 9}, {"stone": None, "width": 5, "height": 1, "area": 5}])
        self.assertEqual(obs["remaining"]["total_area"], 24)
        self.assertIn("sampled_sizes", obs["remaining"])  # version 2 fields stay

    def test_observation_is_the_same_under_renamed_stone_ids(self):
        a, _ = order_controller.Observer(version=3)(PAPER, context())
        b, _ = order_controller.Observer(version=3)(PAPER, context(lambda s: 100 - s))
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_a_trace_without_shapes_gives_none(self):
        obs, _ = order_controller.Observer(version=3)(PAPER, context(shapes=False))
        self.assertEqual((obs["remaining"]["shapes"], obs["remaining"]["total_area"]), (None, None))

    def test_remaining_shapes_reads_the_stone_attributes(self):
        stones = [types.SimpleNamespace(id=4, width=3, height=2, area=5),
                  types.SimpleNamespace(id=9, width=1, height=7, area=6)]
        shapes = remaining_shapes(stones)
        self.assertEqual({k: v.tolist() for k, v in shapes.items()},
                         {"remaining_ids": [4, 9], "remaining_width": [3, 1], "remaining_height": [2, 7],
                          "remaining_area": [5, 6]})


class TraceCompare(unittest.TestCase):
    def test_only_the_version_3_arrays_may_be_missing_from_an_older_wall(self):
        old = {"wall": np.zeros(2)}
        new = {**old, **{k: np.arange(2) for k in REMAINING_SHAPE_KEYS}}
        def event(arrays):
            return {0: ({"run_id": "a"}, arrays)}

        self.assertEqual(check_run.event_differences(event(new), event(old)), [])
        self.assertEqual(check_run.event_differences(event({**old, "x": np.zeros(1)}), event(old)),
                         ["event 0: arrays differ"])
        self.assertEqual(check_run.event_differences(event(new), event({**new, "remaining_area": np.ones(2)})),
                         ["event 0: arrays differ"])


if __name__ == "__main__":
    unittest.main()
