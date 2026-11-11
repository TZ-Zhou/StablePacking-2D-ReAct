"""The note batch of docs/plans/llm-endgame-llc-note.md: the control batch (research/llm_walls/llc_control_wall.py) with
"llc_note": true, so the system prompt also says LLC and F_LR are the same kind of quantity (observation 'pick-3',
research/controller/pick.py). research/runner/pick_wall.py --llc builds both batches interleaved (usage there).

Every guarantee of lookahead_wall.py, with this module as the experiment and its own budget (US$2).
Studies: llc-note-walls (paid) and llc-note-walls-loopback (a local mock server); no archive.
"""
import sys

from research.llm_walls import llc_control_wall, pick_wall
from research.paths import LEGACY_RUNS

STUDIES = {False: "llc-note-walls", True: "llc-note-walls-loopback"}
STUDY_KIND = "llc-note-walls"
LEGACY = {k: LEGACY_RUNS / v for k, v in STUDIES.items()}  # none exist
WALL_ARGS = pick_wall.WALL_ARGS
PRICE, PRICE_CHECKED, PRICE_MAX_AGE = pick_wall.PRICE, pick_wall.PRICE_CHECKED, pick_wall.PRICE_MAX_AGE
LOOPBACK_KEY, FORMAL = pick_wall.LOOPBACK_KEY, pick_wall.FORMAL
SPEC = {**llc_control_wall.SPEC, "llc_note": True}
WALL_LIMITS = llc_control_wall.WALL_LIMITS
EXPERIMENT_LIMITS = llc_control_wall.EXPERIMENT_LIMITS
EXECUTED = (*llc_control_wall.EXECUTED, "research/llm_walls/llc_note_wall.py")
STATES = llc_control_wall.STATES
wall_args = pick_wall.wall_args


def name_of(seed, order):
    return f"llcn-s{seed}-{order}"


def spec_problems(spec=SPEC):
    """lookahead_wall's rules under this module's limits, and the LLC note."""
    return pick_wall.spec_problems(spec, sys.modules[__name__]) + (
        [] if spec.get("lookahead") is True and spec.get("llc_note") is True else
        ["spec: needs lookahead true and llc_note true"])
