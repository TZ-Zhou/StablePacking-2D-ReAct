"""The confirmation of docs/plans/llm-whole-wall-confirm.md: the whole-wall LLM pick of whole_wall.py, unchanged
(observation 'pick-4', the same spec and wall limits), on the held-out seeds 720 and 761, one wall per TH scene.
research/runner/pick_wall.py --whole --confirm builds them.

Every guarantee of whole_wall.py, with this module as the experiment and its own limits (US$2 for the batch).
Studies: whole-confirm-walls (paid) and whole-confirm-walls-loopback (a local mock server); no archive.
"""
import sys

from research.llm_walls import pick_wall, whole_wall
from research.paths import LEGACY_RUNS

STUDIES = {False: "whole-confirm-walls", True: "whole-confirm-walls-loopback"}
STUDY_KIND = "whole-confirm-walls"
LEGACY = {k: LEGACY_RUNS / v for k, v in STUDIES.items()}  # none exist
WALL_ARGS = whole_wall.WALL_ARGS
PRICE, PRICE_CHECKED, PRICE_MAX_AGE = pick_wall.PRICE, pick_wall.PRICE_CHECKED, pick_wall.PRICE_MAX_AGE
LOOPBACK_KEY, FORMAL = pick_wall.LOOPBACK_KEY, pick_wall.FORMAL
SPEC = whole_wall.SPEC
WALL_LIMITS = whole_wall.WALL_LIMITS
EXPERIMENT_LIMITS = {"walls": 4, "http_requests": 400, "tokens": 1_000_000, "cost_usd": 2.00}
EXECUTED = (*whole_wall.EXECUTED, "research/llm_walls/whole_confirm_wall.py")
# (seed, scene): the 4 walls of docs/plans/llm-whole-wall-confirm.md, in the order they are built
STATES = [(720, "TH_T80"), (720, "TH_T20"), (761, "TH_T80"), (761, "TH_T20")]
name_of, wall_args = whole_wall.name_of, whole_wall.wall_args


def spec_problems(spec=SPEC):
    """whole_wall's rules under this module's limits."""
    return pick_wall.spec_problems(spec, sys.modules[__name__]) + (
        [] if spec.get("lookahead") is True and spec.get("whole_wall") is True and not spec.get("llc_note") else
        ["spec: needs lookahead true, whole_wall true and no llc_note"])
