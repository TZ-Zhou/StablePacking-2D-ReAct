"""research/runner/batch.py and research/walls/batch_plan.py: config checks, the Latin hypercube plan, choosing cases, resuming, the walls' arguments."""

import collections
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from research.paths import REPO
from research.runner import batch, run_wall
from research.walls import case_definition
from research.walls import batch_plan

CONFIG = """
name: demo
samples: 30
wall_type: {BR, AL_IR, AL_PR}
filter_policy: {"random", "01234", "12340"}
seed: {min: 1, max: 1000}
"""


def config_with(**changes):
    """The checked CONFIG with some settings replaced by YAML text."""
    lines = [line for line in CONFIG.splitlines() if line.split(":")[0] not in changes]
    return batch_plan.read_config("\n".join(lines + [f"{key}: {value}" for key, value in changes.items()]))


class Config(unittest.TestCase):
    def test_the_levels_of_a_set_and_of_a_list(self):
        config = config_with(wall_type="[BR, RB]")
        self.assertEqual(config["wall_type"], ["BR", "RB"])
        self.assertEqual(config["filter_policy"], ["random", "01234", "12340"])

    def test_defaults(self):
        config = batch_plan.read_config("name: a\nsamples: 2\nwall_type: BR\nfilter_policy: code_original\n")
        self.assertEqual(
            (config["sampler_seed"], config["seed"], config["placements"], config["timeout"], config["workers"]),
            (20, [20], 0, 1800, 8),
        )

    def test_refused(self):
        cases = (
            {"wall_type": "{XX}"},
            {"filter_policy": '{"0123"}'},
            {"filter_policy": '{"01233"}'},
            {"seed": "{min: 5, max: 1}"},
            {"seed": "[-1]"},
            {"samples": "0"},
            {"workers": "two"},
            {"colour": "red"},
            {"name": "a/b"},
        )
        for changes in cases:
            with self.subTest(changes), self.assertRaises(ValueError):
                config_with(**changes)

    def test_every_order_runs_from_01234_to_43210(self):
        orders = batch_plan.ALL_ORDERS
        self.assertEqual((len(orders), len(set(orders)), orders[0], orders[-1]), (120, 120, "01234", "43210"))

    def test_the_example_file_writes_out_random_and_every_order(self):
        text = (REPO / "research" / "walls" / "batch_example.yaml").read_text(encoding="utf-8")
        self.assertEqual(batch_plan.read_config(text)["filter_policy"], ["random", *batch_plan.ALL_ORDERS])

    def test_all_is_not_a_filter_policy(self):
        with self.assertRaises(ValueError):
            config_with(filter_policy="all")

    def test_without_a_filter_policy_random_and_every_order(self):
        config = batch_plan.read_config("name: a\nsamples: 2\nwall_type: BR\n")
        self.assertEqual(config["filter_policy"], ["random", *batch_plan.ALL_ORDERS])

    def test_a_missing_key_is_refused(self):
        with self.assertRaisesRegex(ValueError, "missing keys .'samples'"):
            batch_plan.read_config("name: a\nwall_type: BR\nfilter_policy: code_original\n")

    def test_an_unquoted_order_gets_a_hint(self):
        with self.assertRaisesRegex(ValueError, "in quotes"):
            config_with(filter_policy="[01234]")


class Plan(unittest.TestCase):
    def test_the_same_config_gives_the_same_plan(self):
        self.assertEqual(batch_plan.make_plan(config_with()), batch_plan.make_plan(config_with()))

    def test_another_sampler_seed_gives_another_plan(self):
        self.assertNotEqual(batch_plan.make_plan(config_with()), batch_plan.make_plan(config_with(sampler_seed="21")))

    def test_every_level_comes_up_equally_often(self):
        plan = batch_plan.make_plan(config_with())
        self.assertEqual(len(plan), 30)
        for name in ("wall_type", "filter_policy"):
            self.assertEqual(set(collections.Counter(case[name] for case in plan).values()), {10}, name)

    def test_as_many_samples_as_policies_draw_each_policy_once(self):
        plan = batch_plan.make_plan(batch_plan.read_config("name: a\nsamples: 121\nwall_type: BR\n"))
        self.assertEqual(len({case["filter_policy"] for case in plan}), 121)
        self.assertEqual(len(plan), 121)

    def test_a_seed_range_stays_inside_and_is_spread_out(self):
        seeds = sorted(case["seed"] for case in batch_plan.make_plan(config_with()))
        self.assertTrue(1 <= seeds[0] and seeds[-1] <= 1000)
        for tenth in range(10):
            self.assertEqual(sum(tenth * 100 < seed <= (tenth + 1) * 100 for seed in seeds), 3, tenth)

    def test_cases_are_numbered_from_one(self):
        self.assertEqual([case["case"] for case in batch_plan.make_plan(config_with())], list(range(1, 31)))

    def test_a_draw_that_would_repeat_a_wall_is_left_out(self):
        plan = batch_plan.make_plan(config_with(samples="30", wall_type="BR", filter_policy='{"01234"}', seed="[20, 21]"))
        self.assertEqual(sorted(case["seed"] for case in plan), [20, 21])

    def test_random_walls_never_repeat(self):
        plan = batch_plan.make_plan(config_with(samples="5", wall_type="BR", filter_policy="random", seed="[20]"))
        self.assertEqual(len({case["controller_seed"] for case in plan}), 5)

    def test_only_random_walls_have_a_stream_of_orders(self):
        for case in batch_plan.make_plan(config_with()):
            self.assertEqual(case["filter_policy"] == "random", case["controller_seed"] != "")


class Files(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        patcher = mock.patch.object(batch_plan, "BATCH_RUNS", self.root / "batch")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.config = self.root / "config.yaml"
        self.config.write_text(CONFIG, encoding="utf-8")

    def test_the_plan_reads_back_as_it_was_made(self):
        made = batch_plan.make_plan(batch_plan.read_config(CONFIG))
        read = batch_plan.create_batch(self.config).plan
        self.assertEqual(json.loads(json.dumps(read)), json.loads(json.dumps(made)))

    def test_planning_again_keeps_the_plan(self):
        first = batch_plan.create_batch(self.config)
        (first.folder / "plan.csv").write_text("case,wall_type,filter_policy,seed,controller_seed\n", encoding="utf-8")
        self.assertEqual(batch_plan.create_batch(self.config).plan, [])

    def test_another_config_under_the_same_name_is_refused(self):
        batch_plan.create_batch(self.config)
        self.config.write_text(CONFIG.replace("samples: 30", "samples: 31"), encoding="utf-8")
        with self.assertRaises(FileExistsError):
            batch_plan.create_batch(self.config)

    def test_a_batch_that_was_never_planned(self):
        with self.assertRaises(FileNotFoundError):
            batch_plan.load_batch("nothing")


class Choosing(unittest.TestCase):
    def test_nothing_named_means_every_case(self):
        self.assertEqual(batch_plan.chosen_numbers(4), [1, 2, 3, 4])

    def test_a_range_and_single_numbers(self):
        self.assertEqual(batch_plan.chosen_numbers(20, cases="3-5,10"), [3, 4, 5, 10])

    def test_first_and_last(self):
        self.assertEqual(batch_plan.chosen_numbers(10, first=2), [1, 2])
        self.assertEqual(batch_plan.chosen_numbers(10, last=3), [8, 9, 10])

    def test_every_option_counts(self):
        self.assertEqual(batch_plan.chosen_numbers(10, cases="5", first=1, last=1), [1, 5, 10])

    def test_more_than_the_plan_has_means_all_of_it(self):
        self.assertEqual(batch_plan.chosen_numbers(3, first=9), [1, 2, 3])
        self.assertEqual(batch_plan.chosen_numbers(3, last=9), [1, 2, 3])

    def test_a_case_outside_the_plan_is_refused(self):
        with self.assertRaisesRegex(ValueError, r"cannot choose \[11\]"):
            batch_plan.chosen_numbers(10, cases="9-11")

    def test_choosing_nothing_is_refused(self):
        with self.assertRaises(ValueError):
            batch_plan.chosen_numbers(10, first=0)


class WallArguments(unittest.TestCase):
    CONFIG = {"placements": 3, "timeout": 60, "workers": 4}

    def argv(self, policy, controller_seed=""):
        case = {"wall_type": "AL_PR", "seed": 7, "filter_policy": policy, "controller_seed": controller_seed}
        return batch_plan.wall_argv(case, self.CONFIG)

    def test_an_order(self):
        self.assertEqual(self.argv("30214")[-4:], ["--policy", "paper", "--order", "3,0,2,1,4"])

    def test_random_picks_an_order_at_every_decision_from_its_own_stream(self):
        argv = self.argv("random", 42)
        self.assertEqual(argv[-4:-1], ["--policy", "paper", "--controller"])
        self.assertEqual(json.loads(argv[-1]), {"kind": "random", "seed": 42})

    def test_code_original(self):
        self.assertEqual(self.argv("code_original")[-2:], ["--policy", "code_original"])

    def test_the_wall_and_its_limits(self):
        self.assertEqual(
            self.argv("code_original")[:9],
            ["almeida-pr", "--seed", "7", "--placements", "3", "--timeout", "60", "--workers", "4"],
        )

    def test_run_wall_accepts_every_scene_of_the_config(self):
        self.assertEqual(set(batch_plan.WALL_CASES.values()), set(case_definition.CASES))


class Records(unittest.TestCase):
    @staticmethod
    def record(status="INCOMPLETE_TRACE", **trace_check):
        receipt = {"status": status, "case_id": "id", "committed_placements": 3, "elapsed_seconds": 2.0}
        return batch.receipt_record(1, {**receipt, "trace_check": trace_check})

    def test_a_controller_wall_without_a_choice_is_a_finished_wall(self):
        record = self.record(controller=[{"verdict": "PARTIAL", "decision_points": 0}])
        self.assertTrue(batch.succeeded(record))
        self.assertIn("no decision points", batch.outcome_text(record))

    def test_an_incomplete_trace_is_a_failure_otherwise(self):
        for reports in (
            [{"verdict": "PARTIAL", "decision_points": 2}],
            [{"verdict": "FAIL", "decision_points": 0}],
            [],
            None,
        ):
            with self.subTest(reports):
                self.assertFalse(batch.succeeded(self.record(controller=reports)))

    def test_a_wall_of_another_status_keeps_it(self):
        self.assertTrue(batch.succeeded(self.record("PROCESS_PASS")))
        self.assertFalse(
            batch.succeeded(self.record("TIMEOUT", controller=[{"verdict": "PARTIAL", "decision_points": 0}]))
        )


class Running(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = Path(tmp.name)
        self.plan = [
            {"case": n, "wall_type": "BR", "filter_policy": "01234", "seed": n, "controller_seed": ""}
            for n in range(1, 5)
        ]
        self.batch = batch_plan.Batch(self.folder, {"name": "x", "placements": 0, "timeout": 9, "workers": 2}, self.plan)
        self.outcomes = {}
        self.calls = []

    def launch(self, argv, root, label):
        self.calls.append((int(argv[2]), label, root))
        outcome = self.outcomes.get(int(argv[2]), "PROCESS_PASS")
        if isinstance(outcome, BaseException):
            raise outcome
        return self.receipt(int(argv[2]), outcome)

    @staticmethod
    def receipt(seed, status):
        return {"status": status, "case_id": f"id-{seed}", "committed_placements": 3, "elapsed_seconds": 1.5}

    def run_all(self, cases=None, retry_failed=False):
        with mock.patch("builtins.print"):
            return batch.run_cases(self.batch, cases or self.plan, retry_failed, self.launch)

    def built(self):
        return [call[0] for call in self.calls]

    def test_every_case_is_built_in_order_into_the_batch_folder(self):
        self.assertTrue(self.run_all())
        self.assertEqual(self.built(), [1, 2, 3, 4])
        self.assertEqual({call[2] for call in self.calls}, {self.folder / "cases"})

    def test_the_label_counts_the_chosen_cases(self):
        self.run_all(self.plan[1:3])
        self.assertEqual([call[1] for call in self.calls], ["[1/2] stones", "[2/2] stones"])

    def test_a_record_for_each_wall(self):
        self.run_all()
        record = batch.statuses(self.folder)[2]
        self.assertEqual((record["status"], record["case_id"], record["placed"]), ("PROCESS_PASS", "id-2", 3))

    def test_resuming_skips_the_finished_walls_and_keeps_counting(self):
        self.run_all(self.plan[:2])
        self.calls.clear()
        self.assertTrue(self.run_all())
        self.assertEqual(self.built(), [3, 4])
        self.assertEqual([call[1] for call in self.calls], ["[3/4] stones", "[4/4] stones"])

    def test_an_interrupted_wall_is_built_again_on_resume(self):
        self.outcomes[2] = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.run_all()
        self.assertEqual(list(batch.statuses(self.folder)), [1])
        self.outcomes.clear()
        self.calls.clear()
        self.assertTrue(self.run_all())
        self.assertEqual(self.built(), [2, 3, 4])

    def test_a_failed_wall_does_not_stop_the_batch_and_fails_it(self):
        self.outcomes[2] = "FAILED"
        self.assertFalse(self.run_all())
        self.assertEqual(self.built(), [1, 2, 3, 4])

    def test_a_failed_wall_stays_failed_unless_retried(self):
        self.outcomes[2] = "TIMEOUT"
        self.run_all()
        self.calls.clear()
        self.assertFalse(self.run_all())
        self.assertEqual(self.built(), [])
        self.outcomes.clear()
        self.assertTrue(self.run_all(retry_failed=True))
        self.assertEqual(self.built(), [2])

    def test_a_wall_that_raises_is_recorded_and_the_batch_goes_on(self):
        self.outcomes[1] = RuntimeError("no license")
        self.assertFalse(self.run_all())
        self.assertEqual(self.built(), [1, 2, 3, 4])
        self.assertIn("no license", batch.statuses(self.folder)[1]["error"])

    def test_an_argv_refused_by_run_wall_is_recorded(self):
        self.outcomes[1] = SystemExit(2)
        self.run_all()
        self.assertEqual(batch.statuses(self.folder)[1]["status"], batch.RAISED)

    def test_a_process_tree_not_stopped_is_recorded_and_ends_the_batch(self):
        self.outcomes[2] = run_wall.TerminationFailed(self.receipt(2, "TERMINATION_FAILED"))
        with self.assertRaises(run_wall.TerminationFailed):
            self.run_all()
        self.assertEqual(self.built(), [1, 2])
        self.assertEqual(batch.statuses(self.folder)[2]["status"], "TERMINATION_FAILED")

    def test_the_receipt_printed_by_a_wall_goes_to_the_log(self):
        def printing(argv, root, label):
            print("the receipt")
            return self.receipt(int(argv[2]), "PROCESS_PASS")

        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            batch.run_cases(self.batch, self.plan[:1], False, printing)
        self.assertNotIn("the receipt", shown.getvalue())
        self.assertIn("the receipt", (self.folder / "launcher.log").read_text(encoding="utf-8"))

    def test_a_line_cut_short_by_a_crash_is_ignored(self):
        self.run_all(self.plan[:1])
        with (self.folder / "status.jsonl").open("a", encoding="utf-8") as file:
            file.write('{"case": 2, "sta')
        self.assertEqual(list(batch.statuses(self.folder)), [1])


class Command(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        patcher = mock.patch.object(batch_plan, "BATCH_RUNS", self.root / "batch")
        patcher.start()
        self.addCleanup(patcher.stop)
        config = self.root / "c.yaml"
        config.write_text(CONFIG, encoding="utf-8")
        with mock.patch("builtins.print"):
            batch.main(["plan", str(config)])

    def test_a_dry_run_builds_nothing_and_needs_no_license(self):
        with mock.patch.object(batch, "run_cases") as running, mock.patch("builtins.print") as shown:
            code = batch.main(["run", "demo", "--first", "2", "--dry-run"])
        self.assertEqual((code, running.call_count, shown.call_count), (0, 0, 2))

    def test_the_chosen_cases_reach_the_runner(self):
        with mock.patch.object(batch, "run_cases", return_value=True) as running, mock.patch.object(
            batch, "prepare_environment"
        ):
            code = batch.main(["run", "demo", "--cases", "3-4", "--last", "1"])
        self.assertEqual(code, 0)
        self.assertEqual([case["case"] for case in running.call_args.args[1]], [3, 4, 30])

    def test_exit_codes(self):
        outcomes = ((False, 1), (KeyboardInterrupt(), 130), (run_wall.TerminationFailed({}), 2))
        for outcome, expected in outcomes:
            with self.subTest(outcome), mock.patch.object(batch, "prepare_environment"), mock.patch.object(
                batch, "run_cases", side_effect=[outcome]
            ), mock.patch("builtins.print"):
                self.assertEqual(batch.main(["run", "demo"]), expected)

    def test_show_lists_every_case(self):
        with mock.patch("builtins.print") as shown:
            batch.main(["show", "demo"])
        self.assertEqual(shown.call_count, 31)
        self.assertIn("30 cases, 0 succeeded, 0 failed", shown.call_args_list[0].args[0])


if __name__ == "__main__":
    unittest.main()
