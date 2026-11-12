"""The safe top-LLC hint walls of docs/plans/llm-safe-top-hint.md: the whole-wall LLM pick of whole_wall.py with
"safe_top_llc": true (observation 'pick-5': the list of the safe candidates at the highest LLC among the safe ones,
and one paragraph on it in the prompt; the same model, tools and wall limits), on the states of the pick-4 repeats
(repeat_wall.STATES: 4 rounds over the 10 TH blocks, a round at a time), to be read against those repeats.
research/runner/pick_wall.py --safe-top builds them.

Every guarantee of whole_wall.py, with this module as the experiment and its own limits (US$20 for the batch).
Studies: safe-top-walls (paid) and safe-top-walls-loopback (a local mock server); no archive.
"""
import sys

from research.llm_walls import pick_wall, repeat_wall, whole_wall
from research.paths import LEGACY_RUNS

STUDIES = {False: "safe-top-walls", True: "safe-top-walls-loopback"}
STUDY_KIND = "safe-top-walls"
LEGACY = {k: LEGACY_RUNS / v for k, v in STUDIES.items()}  # none exist
WALL_ARGS = whole_wall.WALL_ARGS
PRICE, PRICE_CHECKED, PRICE_MAX_AGE = pick_wall.PRICE, pick_wall.PRICE_CHECKED, pick_wall.PRICE_MAX_AGE
LOOPBACK_KEY, FORMAL = pick_wall.LOOPBACK_KEY, pick_wall.FORMAL
SPEC = {**whole_wall.SPEC, "safe_top_llc": True}
WALL_LIMITS = whole_wall.WALL_LIMITS
EXPERIMENT_LIMITS = dict(repeat_wall.EXPERIMENT_LIMITS)
EXECUTED = (*repeat_wall.EXECUTED, "research/llm_walls/safe_top_wall.py")
STATES = repeat_wall.STATES  # (seed, (scene, round)), a round at a time
wall_args = repeat_wall.wall_args


def name_of(seed, state):
    scene, r = state
    return f"safetop-s{seed}-{scene.lower().replace('_', '-')}-r{r}"


def spec_problems(spec=SPEC):
    """whole_wall's rules under this module's limits, and the list."""
    return pick_wall.spec_problems(spec, sys.modules[__name__]) + (
        [] if spec.get("lookahead") is True and spec.get("whole_wall") is True and not spec.get("llc_note")
        and spec.get("safe_top_llc") is True else
        ["spec: needs lookahead true, whole_wall true, safe_top_llc true and no llc_note"])
