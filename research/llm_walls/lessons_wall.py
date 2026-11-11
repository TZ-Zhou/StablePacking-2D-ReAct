"""The lessons walls of docs/plans/llm-lessons.md, stage A: the whole-wall LLM pick of whole_wall.py (observation
'pick-4', the same model, tools and wall limits) with the lessons of an induction (lessons.py) appended to its
prompt, one wall per TH block of the 5 factorial seeds. A wall of seed s gets the lessons of the fold without s
(spec_of), so no wall of its stones was in the pack they were drawn from. research/runner/pick_wall.py --lessons
builds them.

Every guarantee of whole_wall.py, with this module as the experiment and its own limits (US$5 for the batch); the
paid walls also need the paid inductions' records committed (EXECUTED). SPEC is the base every wall shares; a wall's
journal and receipt hold its own spec, lessons included.
Studies: lessons-walls (paid) and lessons-walls-loopback (a local mock server); no archive.
"""
import sys

from research.llm_walls import lessons, pick_wall, whole_confirm_wall, whole_wall
from research.paths import LEGACY_RUNS

STUDIES = {False: "lessons-walls", True: "lessons-walls-loopback"}
STUDY_KIND = "lessons-walls"
LEGACY = {k: LEGACY_RUNS / v for k, v in STUDIES.items()}  # none exist
WALL_ARGS = whole_wall.WALL_ARGS
PRICE, PRICE_CHECKED, PRICE_MAX_AGE = pick_wall.PRICE, pick_wall.PRICE_CHECKED, pick_wall.PRICE_MAX_AGE
LOOPBACK_KEY, FORMAL = pick_wall.LOOPBACK_KEY, pick_wall.FORMAL
SPEC = whole_wall.SPEC
WALL_LIMITS = whole_wall.WALL_LIMITS
EXPERIMENT_LIMITS = {"walls": 10, "http_requests": 1000, "tokens": 2_500_000, "cost_usd": 5.00}
EXECUTED = (*lessons.EXECUTED, "research/llm_walls/lessons_wall.py", f"research/studies/{lessons.STUDIES[False]}")
# (seed, scene): the 10 blocks of the whole-wall pick and its confirmation, in the order they are built
STATES = [*whole_wall.STATES, *whole_confirm_wall.STATES]
wall_args = whole_wall.wall_args


def name_of(seed, scene):
    return f"lessons-s{seed}-{scene.lower().replace('_', '-')}"


def spec_of(seed, loopback=False):
    """The spec of the walls of `seed`: SPEC with the lessons of the fold without that seed."""
    return {**SPEC, "lessons": lessons.committed(seed, loopback)}


def spec_problems(spec=SPEC):
    """whole_wall's rules under this module's limits."""
    return pick_wall.spec_problems(spec, sys.modules[__name__]) + (
        [] if spec.get("lookahead") is True and spec.get("whole_wall") is True and not spec.get("llc_note") else
        ["spec: needs lookahead true, whole_wall true and no llc_note"])
