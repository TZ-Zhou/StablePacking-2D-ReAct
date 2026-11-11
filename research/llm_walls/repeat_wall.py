"""The pick-4 repeats of docs/plans/llm-pick-repeats.md: the whole-wall LLM pick of whole_wall.py with nothing changed
(SPEC, so the prompt, the model, the tools and the wall limits), built ROUNDS times on each TH block of the 5
factorial seeds, to measure how much the LLM's walls differ from run to run on the same stones. A state is
(seed, (scene, round)); the walls are built a round at a time, so a batch that stops after a round has every block
equally often. research/runner/pick_wall.py --repeats builds them.

Every guarantee of whole_wall.py, with this module as the experiment and its own limits (US$20 for the batch).
Studies: pick-repeats (paid) and pick-repeats-loopback (a local mock server); no archive.
"""
import sys

from research.llm_walls import pick_wall, whole_confirm_wall, whole_wall
from research.paths import LEGACY_RUNS

STUDIES = {False: "pick-repeats", True: "pick-repeats-loopback"}
STUDY_KIND = "pick-repeats"
LEGACY = {k: LEGACY_RUNS / v for k, v in STUDIES.items()}  # none exist
WALL_ARGS = whole_wall.WALL_ARGS
PRICE, PRICE_CHECKED, PRICE_MAX_AGE = pick_wall.PRICE, pick_wall.PRICE_CHECKED, pick_wall.PRICE_MAX_AGE
LOOPBACK_KEY, FORMAL = pick_wall.LOOPBACK_KEY, pick_wall.FORMAL
SPEC = whole_wall.SPEC
WALL_LIMITS = whole_wall.WALL_LIMITS
EXPERIMENT_LIMITS = {"walls": 40, "http_requests": 4000, "tokens": 10_000_000, "cost_usd": 20.00}
EXECUTED = (*whole_wall.EXECUTED, "research/llm_walls/whole_confirm_wall.py", "research/llm_walls/repeat_wall.py")
ROUNDS = 4
# (seed, scene): the 10 blocks of the whole-wall pick and its confirmation
BLOCKS = [*whole_wall.STATES, *whole_confirm_wall.STATES]
# (seed, (scene, round)), in the order they are built: round 1 of every block, then round 2, ...
STATES = [(seed, (scene, r)) for r in range(1, ROUNDS + 1) for seed, scene in BLOCKS]


def name_of(seed, state):
    scene, r = state
    return f"repeat-s{seed}-{scene.lower().replace('_', '-')}-r{r}"


def wall_args(state):
    return whole_wall.wall_args(state[0])


def spec_problems(spec=SPEC):
    """whole_wall's rules under this module's limits."""
    return pick_wall.spec_problems(spec, sys.modules[__name__]) + (
        [] if spec.get("lookahead") is True and spec.get("whole_wall") is True and not spec.get("llc_note") else
        ["spec: needs lookahead true, whole_wall true and no llc_note"])
