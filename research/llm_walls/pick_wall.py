"""The paid LLM endgame pick of docs/plans/llm-endgame-pick.md: one TH_T80 wall per state of the endgame arena whose
last placements an OpenAI model picks directly (controller kind 'pick', research/controller/pick.py), every other part
of the wall Wang's under the state's filter order. research/runner/pick_wall.py builds them (usage there).

The walls run under every guarantee of research/llm_walls/gpt_wall.py (that module's functions with this one as the
experiment, gpt_wall.experiment): the spec below checked against WALL_LIMITS, the experiment budget EXPERIMENT_LIMITS
(US$2 for the batch), the key only in the venv worker, no connection setting, the paid gate and --confirm-paid, a
clean committed tree, a price checked within PRICE_MAX_AGE days, one wall at a time under the study's lock, a name
used once, every earlier wall finished and named with --after, and a wall finished only if its billing is certain.
Studies: pick-walls (paid) and pick-walls-loopback (a local mock server); no archive.
"""
import sys

from research.controller import llm_api, order_controller
from research.llm_walls import gpt_smoke, gpt_wall
from research.paths import LEGACY_RUNS

STUDIES = {False: "pick-walls", True: "pick-walls-loopback"}
STUDY_KIND = "pick-walls"
LEGACY = {False: LEGACY_RUNS / "pick-walls", True: LEGACY_RUNS / "pick-walls-loopback"}  # none exist
WALL_ARGS = ["thangavelu-noise80", "--policy", "paper", "--worker-venv"]  # the order is the state's (wall_args)
PRICE = gpt_smoke.PRICE
PRICE_CHECKED = gpt_wall.PRICE_CHECKED
PRICE_MAX_AGE = gpt_wall.PRICE_MAX_AGE
LOOPBACK_KEY = gpt_wall.LOOPBACK_KEY
FORMAL = gpt_wall.FORMAL
SPEC = {"kind": "pick",
        "llm": {"kind": "api", "provider": "openai", "model": gpt_smoke.MODEL, "max_tokens": 4096, "effort": "low",
                "request_timeout_s": 60.0, "max_retries": 0, "retry_backoff_s": 0.0, "price_per_mtok": PRICE},
        "llm_timeout": 90.0, "max_tool_calls": 3, "max_decisions": 10, "max_total_tokens": 100_000,
        "max_cost_usd": 0.20, "max_llm_failures": 1}
WALL_LIMITS = {"http_requests": 30, "tokens": 100_000, "cost_usd": 0.20, "seconds": 2400}
EXPERIMENT_LIMITS = {"walls": 9, "http_requests": 270, "tokens": 900_000, "cost_usd": 2.00}
EXECUTED = (*gpt_wall.EXECUTED, "research/controller/pick.py", "research/llm_walls/pick_wall.py",
            "research/runner/pick_wall.py")
# (seed, order): the 9 states of docs/plans/endgame-arena.md, in the order they are built
STATES = [(313, "30214"), (313, "40321"), (313, "34021"), (484, "30214"), (484, "21043"), (484, "13204"),
          (717, "30214"), (717, "40123"), (717, "14032")]


def name_of(seed, order):
    return f"pick-s{seed}-{order}"


def wall_args(order):
    return ["--order", ",".join(order)]


def spec_problems(spec=SPEC, x=None):
    """Why SPEC does not fit WALL_LIMITS (the structural bounds a wall cannot exceed); x: the experiment module whose
    limits apply (this one by default; research/llm_walls/lookahead_wall.py)."""
    x = x or sys.modules[__name__]
    wall, batch = x.WALL_LIMITS, x.EXPERIMENT_LIMITS
    full, llm = order_controller.full_spec(spec), llm_api.full(spec["llm"])
    rules = {"kind pick": full["kind"] == "pick", "max_retries 0": llm["max_retries"] == 0,
             f"max_decisions x max_tool_calls <= {wall['http_requests']}":
                 gpt_wall.max_requests(full) <= wall["http_requests"],
             f"max_total_tokens <= {wall['tokens']}": full["max_total_tokens"] <= wall["tokens"],
             f"max_cost_usd <= {wall['cost_usd']}": full["max_cost_usd"] <= wall["cost_usd"],
             "call_bound <= llm_timeout": llm_api.call_bound(llm) <= full["llm_timeout"],
             "max_llm_failures 1": full["max_llm_failures"] == 1,
             f"price {PRICE}": list(llm["price_per_mtok"] or []) == PRICE,
             f"{batch['walls']} walls x the wall cost <= the batch cost":
                 batch["walls"] * wall["cost_usd"] <= batch["cost_usd"]}
    return [f"spec: needs {rule}" for rule, ok in rules.items() if not ok]
