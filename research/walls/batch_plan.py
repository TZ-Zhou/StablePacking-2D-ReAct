"""What a batch of walls is: its config file, the Latin hypercube of its settings and each case's wall arguments.

research/runner/batch.py plans and builds batches. The config file is YAML:

  name: demo                 # the batch's folder: runs/batch/demo
  samples: 30                # Latin hypercube draws
  sampler_seed: 20           # the same config always gives the same cases (default 20)
  wall_type: {BR, AL_IR, AL_PR}                 # scene codes of research/storage/case_id.SCENES
  filter_policy: {"random", "01234", "12340"}   # see below; write an order in quotes
  seed: {min: 1, max: 1000}  # a list of seeds, or an integer range (default 20)
  placements: 0              # stop each wall after N stones; 0 builds the whole wall (default)
  timeout: 1800              # seconds allowed per wall (default)
  workers: 10                # worker processes per wall (default)

A filter policy is an order of the five filters ("30214" is the paper's), "random" (a new order at every decision
point, from a stream of orders that is the case's own) or "code_original" (Wang's code). Without the key the levels
are "random" and every order from "01234" to "43210"; batch_example.yaml writes all of them out. An order followed by
"+<mode>" ("30214+llc", "30214+random:7") builds with that order and research/controller/endgame.py's controller of
that mode (a random mode names its seed), which picks directly once the inventory is empty; "+pick_safe" builds
with the pick controller of research/controller/pick.py, its mock LLM pick_safe and the check_after tool (lookahead).
"+whole_<mode>" ("30214+whole_borda", "30214+whole_pick_safe"; docs/plans/whole-wall-rules.md) is that controller
picking at every selection, from the first stone.

The plan is drawn once and written to plan.csv, numbered from 1; a draw equal to an earlier one would build the same
wall again and is left out, so a plan can have fewer cases than `samples`.
"""

import contextlib
import csv
import itertools
import json
import re
from collections import namedtuple
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import qmc

from research.controller import endgame, order_controller
from research.paths import BATCH_RUNS
from research.storage.case_id import SCENES

WALL_CASES = {code: case for case, code in SCENES.items()}
ALL_ORDERS = ["".join(order) for order in itertools.permutations("01234")]  # 01234, 01243, ..., 43210
NAMED_POLICIES = ("random", "code_original")
FACTORS = ("wall_type", "filter_policy", "seed")
REQUIRED = ("name", "samples", "wall_type")
DEFAULTS = {
    "filter_policy": ["random", *ALL_ORDERS],
    "sampler_seed": 20,
    "seed": [20],
    "placements": 0,
    "timeout": 1800,
    "workers": 8,
}
LOWEST_VALUES = {"samples": 1, "sampler_seed": 0, "placements": 0, "timeout": 1, "workers": 1}
PLAN_FIELDS = ("case", *FACTORS, "controller_seed")
Batch = namedtuple("Batch", "folder config plan")


def wall_type_problem(level):
    return None if level in WALL_CASES else f"choose from {sorted(WALL_CASES)}"


def filter_policy_problem(level):
    """An unquoted 01234 arrives as a number: YAML reads it as octal."""
    if not isinstance(level, str):
        return 'write an order in quotes, e.g. "01234"'
    if "+" in level:
        try:
            spec = endgame_spec(level)
            endgame.full_spec(spec) if spec["kind"] == "endgame" else order_controller.validate_spec(spec)
        except ValueError as error:
            return f"an order with an endgame controller, e.g. 30214+llc or 30214+random:7: {error}"
        level = level.partition("+")[0]
    is_order = sorted(level) == list("01234")
    return None if is_order or level in NAMED_POLICIES else f"an order of 01234 or one of {NAMED_POLICIES}"


def endgame_spec(level):
    """The endgame controller spec of a level "<order>+<mode>[:<seed>]"."""
    mode, _, seed = level.partition("+")[2].partition(":")
    whole, mode = mode.startswith("whole_"), mode.removeprefix("whole_")
    if mode == "pick_safe":  # whole: the budget of research/llm_walls/whole_wall.SPEC
        return {"kind": "pick", "llm": {"kind": "mock", "mode": "pick_safe"}, "lookahead": True,
                **({"whole_wall": True, "max_tool_calls": 4, "max_decisions": 25} if whole else {})}
    return {"kind": "endgame", "mode": mode, **({"seed": int(seed)} if seed.isdigit() else {}),
            **({"whole": True} if whole else {})}


def seed_problem(level):
    return None if type(level) is int and level >= 0 else "a non-negative integer"


def levels_of(name, value, problem):
    """A YAML list, a YAML `{a, b}` (a mapping without values) or one value, as checked levels without repeats."""
    levels = list(value) if isinstance(value, (dict, list)) else [value]
    bad = {repr(level): problem(level) for level in levels if problem(level)}
    if bad or not levels:
        raise ValueError(f"{name}: no usable levels {bad}")
    return list(dict.fromkeys(levels))


def seed_factor(value):
    """The seeds to draw from: a list of levels, or every integer of {min: a, max: b}."""
    if not (isinstance(value, dict) and set(value) == {"min", "max"}):
        return levels_of("seed", value, seed_problem)
    if seed_problem(value["min"]) or seed_problem(value["max"]) or value["min"] > value["max"]:
        raise ValueError(f"seed: bad range {value}")
    return value


def check_whole_numbers(config):
    for key, lowest in LOWEST_VALUES.items():
        if type(config[key]) is not int or config[key] < lowest:
            raise ValueError(f"{key}: an integer of at least {lowest}, got {config[key]!r}")


def read_config(text):
    """The checked settings of a config file: a bad value stops here, before anything is sampled or built."""
    raw = yaml.safe_load(text)
    unknown, missing = set(raw) - {*REQUIRED, *DEFAULTS}, set(REQUIRED) - set(raw)
    if unknown or missing:
        raise ValueError(f"config: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    config = {**DEFAULTS, **raw}
    check_whole_numbers(config)
    if not re.fullmatch(r"[\w.-]+", str(config["name"])):
        raise ValueError(f"name: {config['name']!r} cannot name a folder")
    config["wall_type"] = levels_of("wall_type", config["wall_type"], wall_type_problem)
    config["filter_policy"] = levels_of("filter_policy", config["filter_policy"], filter_policy_problem)
    config["seed"] = seed_factor(config["seed"])
    return config


def level_at(factor, point):
    """The level of a factor at a point of [0, 1): every level or integer has an equal share of the interval."""
    if isinstance(factor, dict):
        return factor["min"] + int(point * (factor["max"] - factor["min"] + 1))
    return factor[int(point * len(factor))]


def drawn_settings(config):
    """`samples` Latin hypercube draws: over the batch, each factor's levels come up equally often."""
    points = qmc.LatinHypercube(d=len(FACTORS), seed=config["sampler_seed"]).random(config["samples"])
    return [{name: level_at(config[name], x) for name, x in zip(FACTORS, point)} for point in points]


def controller_seed(sampler_seed, draw):
    """The seed of a random wall's stream of orders: its own, and the same every time the plan is made."""
    return int(np.random.default_rng([sampler_seed, draw]).integers(2**31 - 1))


def make_plan(config):
    """The numbered cases; a draw equal to an earlier one is left out, a random wall never equals another."""
    plan, seen = [], set()
    for draw, settings in enumerate(drawn_settings(config)):
        stream = controller_seed(config["sampler_seed"], draw) if settings["filter_policy"] == "random" else ""
        key = (*settings.values(), stream)
        if key not in seen:
            seen.add(key)
            plan.append({"case": len(plan) + 1, **settings, "controller_seed": stream})
    return plan


def write_plan(path, plan):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, PLAN_FIELDS)
        writer.writeheader()
        writer.writerows(plan)


def plan_row(row):
    """A row of plan.csv with its numbers; only a random wall has a controller seed."""
    stream = int(row["controller_seed"]) if row["controller_seed"] else ""
    return {**row, "case": int(row["case"]), "seed": int(row["seed"]), "controller_seed": stream}


def read_plan(path):
    with path.open(newline="", encoding="utf-8") as file:
        return [plan_row(row) for row in csv.DictReader(file)]


def create_batch(config_path):
    """Write the batch's folder once and return its plan; the same config again changes nothing, another is refused."""
    text = Path(config_path).read_text(encoding="utf-8")
    config = read_config(text)
    folder = BATCH_RUNS / config["name"]
    saved = folder / "config.yaml"
    if saved.exists() and saved.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"{folder} holds a batch with another config: choose another name")
    if not saved.exists():
        folder.mkdir(parents=True, exist_ok=True)
        write_plan(folder / "plan.csv", make_plan(config))
        saved.write_text(text, encoding="utf-8")
    return load_batch(config["name"])


def load_batch(name):
    """The Batch of a planned batch."""
    folder = BATCH_RUNS / name
    if not (folder / "config.yaml").exists():
        raise FileNotFoundError(f"no batch {name}: run `plan` with its config first")
    config = read_config((folder / "config.yaml").read_text(encoding="utf-8"))
    return Batch(folder, config, read_plan(folder / "plan.csv"))


def statuses(folder):
    """{case number: its latest record}; a line cut short by a crash names no finished wall."""
    path = folder / "status.jsonl"
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else ():
        with contextlib.suppress(json.JSONDecodeError):
            record = json.loads(line)
            records[record["case"]] = record
    return records


def parse_ranges(text):
    """The numbers of `3-10,15`."""
    numbers = set()
    for part in text.split(","):
        first, _, last = part.partition("-")
        numbers.update(range(int(first), int(last or first) + 1))
    return numbers


def chosen_numbers(total, cases=None, first=None, last=None):
    """The case numbers named by any of the options, in order; no option at all means every case."""
    if cases is None and first is None and last is None:
        return list(range(1, total + 1))
    numbers = parse_ranges(cases) if cases else set()
    numbers |= set(range(1, min(first or 0, total) + 1))
    numbers |= set(range(max(total - (last or 0), 0) + 1, total + 1))
    outside = sorted(number for number in numbers if not 1 <= number <= total)
    if outside or not numbers:
        raise ValueError(f"the plan has cases 1 to {total}: cannot choose {outside or 'nothing'}")
    return sorted(numbers)


def policy_args(case):
    """The run_wall arguments of a case's filter policy."""
    policy = case["filter_policy"]
    if policy == "code_original":
        return ["--policy", "code_original"]
    if policy == "random":
        return [
            "--policy",
            "paper",
            "--controller",
            json.dumps({"kind": "random", "seed": case["controller_seed"]}),
        ]
    if "+" in policy:
        order = policy.partition("+")[0]
        return ["--policy", "paper", "--order", ",".join(order), "--controller", json.dumps(endgame_spec(policy))]
    return ["--policy", "paper", "--order", ",".join(policy)]


def wall_argv(case, config):
    """The run_wall.run argv of one case."""
    limits = ["--placements", str(config["placements"]), "--timeout", str(config["timeout"])]
    return [
        WALL_CASES[case["wall_type"]],
        "--seed",
        str(case["seed"]),
        *limits,
        "--workers",
        str(config["workers"]),
        *policy_args(case),
    ]
