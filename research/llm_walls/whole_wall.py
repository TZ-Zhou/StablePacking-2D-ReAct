"""The paid whole-wall LLM pick of docs/plans/llm-whole-wall-pick.md: one wall per development seed and TH scene whose
every placement an OpenAI model picks among the candidates Wang's sampling offers there, with the check_after tool
(controller kind 'pick' with lookahead and whole_wall, observation 'pick-4', research/controller/pick.py); the base
order is the Paper's, so a skipped decision is Wang's cascade. research/runner/pick_wall.py --whole builds them.

Every guarantee of lookahead_wall.py, with this module as the experiment and its own limits (US$3 for the batch).
Studies: whole-walls (paid) and whole-walls-loopback (a local mock server); no archive.
"""
import sys

from research.llm_walls import lookahead_wall, pick_wall
from research.paths import LEGACY_RUNS

STUDIES = {False: "whole-walls", True: "whole-walls-loopback"}
STUDY_KIND = "whole-walls"
LEGACY = {k: LEGACY_RUNS / v for k, v in STUDIES.items()}  # none exist
WALL_ARGS = ["--policy", "paper", "--order", "3,0,2,1,4", "--worker-venv"]  # the case is the state's (wall_args)
PRICE, PRICE_CHECKED, PRICE_MAX_AGE = pick_wall.PRICE, pick_wall.PRICE_CHECKED, pick_wall.PRICE_MAX_AGE
LOOPBACK_KEY, FORMAL = pick_wall.LOOPBACK_KEY, pick_wall.FORMAL
SPEC = {**lookahead_wall.SPEC, "whole_wall": True, "max_decisions": 25, "max_total_tokens": 250_000,
        "max_cost_usd": 0.50}
WALL_LIMITS = {"http_requests": 100, "tokens": 250_000, "cost_usd": 0.50, "seconds": 3600}
EXPERIMENT_LIMITS = {"walls": 6, "http_requests": 600, "tokens": 1_500_000, "cost_usd": 3.00}
EXECUTED = (*lookahead_wall.EXECUTED, "research/llm_walls/whole_wall.py")
CASES = {"TH_T80": "thangavelu-noise80", "TH_T20": "thangavelu-noise20"}
# (seed, scene): the 6 walls of docs/plans/llm-whole-wall-pick.md, in the order they are built
STATES = [(313, "TH_T80"), (313, "TH_T20"), (484, "TH_T80"), (484, "TH_T20"), (717, "TH_T80"), (717, "TH_T20")]


def name_of(seed, scene):
    return f"whole-s{seed}-{scene.lower().replace('_', '-')}"  # a run name is [a-z0-9-]


def wall_args(scene):
    return [CASES[scene]]


def spec_problems(spec=SPEC):
    """lookahead_wall's rules under this module's limits, and the whole wall."""
    return pick_wall.spec_problems(spec, sys.modules[__name__]) + (
        [] if spec.get("lookahead") is True and spec.get("whole_wall") is True and not spec.get("llc_note") else
        ["spec: needs lookahead true, whole_wall true and no llc_note"])
