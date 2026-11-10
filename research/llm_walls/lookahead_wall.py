"""The paid LLM endgame pick with the check_after tool of docs/plans/llm-endgame-lookahead.md: one TH_T80 wall per new
seven-left state, its last placements picked by an OpenAI model that can ask, per candidate, how many feasible poses
each other remaining stone keeps at the next placement (controller kind 'pick' with lookahead, research/controller/
pick.py). research/runner/pick_wall.py --lookahead builds them (usage there).

Every guarantee of research/llm_walls/pick_wall.py, with this module as the experiment: the spec below checked
against WALL_LIMITS, the batch budget EXPERIMENT_LIMITS (US$2), the paid gate and --confirm-paid, a clean committed
tree, one wall at a time under the study's lock, a name used once, a wall finished only if its billing is certain.
Studies: lookahead-walls (paid) and lookahead-walls-loopback (a local mock server); no archive.
"""
import sys

from research.llm_walls import pick_wall
from research.paths import LEGACY_RUNS

STUDIES = {False: "lookahead-walls", True: "lookahead-walls-loopback"}
STUDY_KIND = "lookahead-walls"
LEGACY = {False: LEGACY_RUNS / "lookahead-walls", True: LEGACY_RUNS / "lookahead-walls-loopback"}  # none exist
WALL_ARGS = pick_wall.WALL_ARGS
PRICE, PRICE_CHECKED, PRICE_MAX_AGE = pick_wall.PRICE, pick_wall.PRICE_CHECKED, pick_wall.PRICE_MAX_AGE
LOOPBACK_KEY, FORMAL = pick_wall.LOOPBACK_KEY, pick_wall.FORMAL
SPEC = {**pick_wall.SPEC, "max_tool_calls": 4, "lookahead": True}
WALL_LIMITS = {"http_requests": 40, "tokens": 100_000, "cost_usd": 0.20, "seconds": 2400}
EXPERIMENT_LIMITS = {"walls": 9, "http_requests": 360, "tokens": 900_000, "cost_usd": 2.00}
EXECUTED = (*pick_wall.EXECUTED, "research/llm_walls/lookahead_wall.py", "research/controller/endgame.py",
            "Stonepacker2D/stack/build_wall.py")
# (seed, order): the 9 new states of docs/plans/llm-endgame-lookahead.md, in the order they are built
STATES = [(313, "13420"), (313, "14302"), (313, "03241"), (484, "02413"), (484, "10243"), (484, "01324"),
          (717, "04123"), (717, "23410"), (717, "20341")]
wall_args = pick_wall.wall_args


def name_of(seed, order):
    return f"look-s{seed}-{order}"


def spec_problems(spec=SPEC):
    """Why SPEC does not fit WALL_LIMITS: pick_wall's rules under this module's limits, and the tool."""
    return pick_wall.spec_problems(spec, sys.modules[__name__]) + (
        [] if spec.get("lookahead") is True else ["spec: needs lookahead true"])
