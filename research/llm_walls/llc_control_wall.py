"""The control batch of docs/plans/llm-endgame-llc-note.md: the LLM endgame pick with check_after exactly as in
research/llm_walls/lookahead_wall.py (observation 'pick-2'), on the 9 states of that plan. research/runner/pick_wall.py
--llc builds it interleaved with research/llm_walls/llc_note_wall.py (usage there).

Every guarantee of lookahead_wall.py, with this module as the experiment and its own budget (US$2).
Studies: llc-control-walls (paid) and llc-control-walls-loopback (a local mock server); no archive.
"""
import sys

from research.llm_walls import lookahead_wall, pick_wall
from research.paths import LEGACY_RUNS

STUDIES = {False: "llc-control-walls", True: "llc-control-walls-loopback"}
STUDY_KIND = "llc-control-walls"
LEGACY = {k: LEGACY_RUNS / v for k, v in STUDIES.items()}  # none exist
WALL_ARGS = pick_wall.WALL_ARGS
PRICE, PRICE_CHECKED, PRICE_MAX_AGE = pick_wall.PRICE, pick_wall.PRICE_CHECKED, pick_wall.PRICE_MAX_AGE
LOOPBACK_KEY, FORMAL = pick_wall.LOOPBACK_KEY, pick_wall.FORMAL
SPEC = lookahead_wall.SPEC
WALL_LIMITS = lookahead_wall.WALL_LIMITS
EXPERIMENT_LIMITS = lookahead_wall.EXPERIMENT_LIMITS
EXECUTED = (*lookahead_wall.EXECUTED, "research/llm_walls/llc_control_wall.py")
# (seed, order): the 9 states of docs/plans/llm-endgame-llc-note.md, in the order they are built
STATES = [(313, "12403"), (313, "41023"), (313, "04231"), (484, "43102"), (484, "40321"), (484, "21430"),
          (717, "14302"), (717, "13204"), (717, "41203")]
wall_args = pick_wall.wall_args


def name_of(seed, order):
    return f"llcc-s{seed}-{order}"


def spec_problems(spec=SPEC):
    """lookahead_wall's rules under this module's limits, and no LLC note."""
    return pick_wall.spec_problems(spec, sys.modules[__name__]) + (
        [] if spec.get("lookahead") is True else ["spec: needs lookahead true"]) + (
        [] if not spec.get("llc_note") else ["spec: the control has no llc_note"])
