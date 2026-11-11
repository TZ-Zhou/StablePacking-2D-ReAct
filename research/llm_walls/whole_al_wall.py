"""The whole-wall LLM pick of whole_wall.py on the Almeida walls (docs/plans/llm-whole-wall-al.md): prompt, tools,
model and settings unchanged (observation 'pick-4'), on the development seeds, one wall per AL scene.
research/runner/pick_wall.py --whole --al builds them, a stage at a time (--limit).

Only a wall's budget differs from whole_wall.SPEC: an AL wall has four selections a stone (Wang's sample-twice probe
and place, each from the left and from the right) and up to 59 stones, so up to 236 decisions before any re-sampling.
Every guarantee of whole_wall.py, with this module as the experiment and its own limits.
Studies: whole-al-walls (paid) and whole-al-walls-loopback (a local mock server); no archive.
"""
import sys

from research.llm_walls import pick_wall, whole_wall
from research.paths import LEGACY_RUNS

STUDIES = {False: "whole-al-walls", True: "whole-al-walls-loopback"}
STUDY_KIND = "whole-al-walls"
LEGACY = {k: LEGACY_RUNS / v for k, v in STUDIES.items()}  # none exist
WALL_ARGS = whole_wall.WALL_ARGS
PRICE, PRICE_CHECKED, PRICE_MAX_AGE = pick_wall.PRICE, pick_wall.PRICE_CHECKED, pick_wall.PRICE_MAX_AGE
LOOPBACK_KEY, FORMAL = pick_wall.LOOPBACK_KEY, pick_wall.FORMAL
SPEC = {**whole_wall.SPEC, "max_decisions": 260, "max_total_tokens": 3_500_000, "max_cost_usd": 5.00}
WALL_LIMITS = {"http_requests": 1040, "tokens": 3_500_000, "cost_usd": 5.00, "seconds": 10800}
EXPERIMENT_LIMITS = {"walls": 9, "http_requests": 9360, "tokens": 31_500_000, "cost_usd": 45.00}
EXECUTED = (*whole_wall.EXECUTED, "research/llm_walls/whole_al_wall.py")
CASES = {"AL_R": "almeida-r", "AL_PR": "almeida-pr", "AL_IR": "almeida-ir"}
# (seed, scene): the 9 walls of docs/plans/llm-whole-wall-al.md, in the order they are built
STATES = [(seed, scene) for seed in (313, 484, 717) for scene in CASES]
name_of = whole_wall.name_of


def wall_args(scene):
    return [CASES[scene]]


def spec_problems(spec=SPEC):
    """whole_wall's rules under this module's limits."""
    return pick_wall.spec_problems(spec, sys.modules[__name__]) + (
        [] if spec.get("lookahead") is True and spec.get("whole_wall") is True and not spec.get("llc_note") else
        ["spec: needs lookahead true, whole_wall true and no llc_note"])
