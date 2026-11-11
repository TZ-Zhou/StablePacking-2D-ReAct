"""The induction of docs/plans/llm-lessons.md: one decision in which the model of the whole-wall pick reads a pack of
finished factorial walls (research/analysis/lessons_pack.py) and commits 1 to 5 lessons. The lessons walls
(lessons_wall.py) append them to the pick's prompt word for word.

  python -B -m research.llm_walls.lessons preflight --without SEED [--loopback]
  STONEPACKING_ALLOW_PAID_API=1 python -B -m research.llm_walls.lessons run --without SEED --confirm-paid
  python -B -m research.llm_walls.lessons run --without SEED --loopback      # a local mock server

--without SEED is the fold: the lessons for the walls of SEED, from a pack with no wall of that seed; 'none' is the
pack of all 10 blocks (stage B).

Guarantees, as for a paid wall (gpt_wall.py): the gate, the key and the connection settings (gpt_wall.env_problems);
the policy runs in the registered worker venv; paid needs the executed files committed and a fresh price; the pack
must be the registered one (lessons_pack.PACKS, committed before the first paid induction). A fold is induced once:
its folder (research/studies/lessons-induction[-loopback]/fold-<seed>) is made by the run and a run refuses one that
exists. journal.json is written before the request; pack.json is what the model read; records.json holds the
worker's reply with every transcript; lessons.json is written only if the run ends 'finished': the model committed
lessons that pass order_controller.lessons_problem, within LIMITS, with known usage. A reply that breaks the format
is rejected with the reason once (the policy's second and last call); after that there are no lessons and the
state is 'uncertain'. The lessons are never edited, chosen among or induced again.
"""
import argparse
import datetime
import hashlib
import json
import multiprocessing

from research.controller import llm_api
from research.controller import order_controller as oc
from research.controller import worker as controller_worker
from research.llm_walls import gpt_smoke, gpt_wall, whole_wall
from research.paths import STUDIES_DIR

STUDIES = {False: "lessons-induction", True: "lessons-induction-loopback"}
# the wall's model, thinking harder and with room for a long answer; kind 'pick' gives the worker the key (worker_env)
SPEC = {"kind": "pick", "max_tool_calls": 2, "llm_timeout": 630.0,
        "llm": {**whole_wall.SPEC["llm"], "effort": "high", "max_tokens": 16_000, "request_timeout_s": 600.0}}
LIMITS = {"http_requests": 2, "tokens": 300_000, "cost_usd": 0.50}  # of one induction
EXECUTED = (*whole_wall.EXECUTED, "research/llm_walls/lessons.py", "research/analysis/lessons_pack.py",
            "research/llm_walls/lessons_packs.json")
SYSTEM = (
    "You study finished dry-stone walls and write lessons for the builder of later walls.\n"
    "The builder. A wall is built course by course with Wang's method. At each placement some of the remaining "
    "stones are on offer (all of them once only a few are left), and the builder chooses one candidate: a stone on "
    "offer in one of four rotations, at the position Wang's image convolution found for it. Every candidate is "
    "feasible. Of each candidate the builder sees the rotation, the width and height of its bounding box in pixels "
    "and Wang's five filter scores (higher is better for each): NHR course height (0 = level with its left "
    "neighbour), GDS global density, IWR interlocking (its vertical joint far from the joints below), LLC the "
    "kinematic limit load of the wall with the stone added and stabilized, divided by the friction coefficient, LFR "
    "local void (0 = flush on the left and on the bottom). The builder also sees n_remaining, the stones not yet "
    "placed; a summary of the current wall, where the skyline is the height of the wall's top above the ground at "
    "each column, in pixels (h_max its maximum, h_range maximum minus minimum, h_std its standard deviation, "
    "void_under_skyline_px the empty pixels below it, n_skyline_jumps its steps of more than 3 pixels between "
    "neighbouring columns); and the stones on offer that have no feasible candidate now. Before choosing, the "
    "builder may call check_after for some candidates: how many feasible poses every other stone on offer would "
    "have at the next placement if that candidate were placed (0: that stone could not be placed next). The builder "
    "knows nothing about the finished wall.\n"
    "The goal. A finished wall is judged on four metrics: F_SF, stone filling (the share of the wall area that is "
    "stone, higher is better); F_AH, the horizontal joint path index (lower is better: no long continuous "
    "horizontal joints); F_AV, the vertical joint path index (higher is better: vertical joints are staggered); "
    "F_LR, the smaller of the left and right kinematic limit loads of the wall divided by the friction coefficient "
    "(higher is better). All four must be good, and a stone that cannot be placed is lost to the wall: losing "
    "stones is worse than a lower metric.\n"
    "The data. stone_sets lists sets of stones. The walls of one set were built from the same stones in the same "
    "place and differ only in the choices made at the placements; never compare walls of different sets. They were "
    "built by fixed rules (the five scores applied in a fixed order of priority), without check_after. Of each "
    "set's n_walls_ranked walls you get 4: two labelled good (the highest mean_z among the walls that placed the "
    "most stones), one of lower quality (the lowest mean_z among those) and one that lost stones (the fewest "
    "placed). Per wall: placed and lost count stones; metrics gives each metric's value and z, its z-score among "
    "the set's walls with the sign turned so that higher is always better; mean_z is the mean of the four z; rank "
    "is the wall's place by mean_z among the set's walls (1 is the best); "
    "first_placement_differing_from_wall_1 is the first placement at which the wall and the set's first wall "
    "differ (null: none); stopped_with, on a wall that lost stones, is n_remaining and the wall summary when no "
    "remaining stone could be placed. placements has one row per stone placed, in the order of placement_columns: "
    "placement (from 1), n_remaining, n_candidates (the feasible candidates there), n_unplaceable (the stones on "
    "offer without one), the wall summary before the placement, then the rotation, width and height of the "
    "candidate that was placed and its five scores, each followed by its share: the share of that placement's "
    "candidates whose score is not higher (1.0: no candidate scored higher; a low share: the choice gave that "
    "score up).\n"
    "Your task. Within each set compare the good walls with the bad ones, look for what recurs across sets, and "
    "write at most 5 lessons for the builder. A lesson is one line of at most 60 words that the builder can apply "
    "at a placement from what it sees there: the candidates' scores and bounding boxes, the wall summary, "
    "n_remaining, the unplaceable stones, check_after. Do not mention particular stones, sets, walls or "
    "coordinates. Prefer what several sets support to what one wall suggests; write fewer than 5 lessons if the "
    "data support fewer. Finish with commit_lessons, with an empty text for each lesson you do not write.")
COMMIT = {"name": "commit_lessons", "description": "Final action: the lessons, 1 to 5.",
          "parameters": {"type": "object", "additionalProperties": False,
                         "required": [f"lesson_{i}" for i in range(1, 6)],
                         "properties": {f"lesson_{i}": {"type": "string", "description":
                                                        "One lesson of at most 60 words, or an empty text."}
                                        for i in range(1, 6)}}}


def lessons_of(args):
    return [args[k].strip() for k in sorted(args) if args[k].strip()]


class LessonsPolicy(oc.ReActPolicy):
    """ReActPolicy with commit_lessons as its only tool and the pack as the first message."""

    def __init__(self, llm, max_tool_calls=2, llm_timeout=630.0):
        self.llm, self.max_tool_calls, self.llm_timeout = llm, max_tool_calls, llm_timeout
        self.system, self.tools = SYSTEM, [COMMIT]
        self.prompt_sha256 = hashlib.sha256(json.dumps([self.system, self.tools]).encode()).hexdigest()

    def view(self, obs):
        return obs

    def answer(self, request, choice):
        return {"lessons": choice}

    def problem(self, reply, request):
        return oc.reply_problem(reply, self.tools) or oc.lessons_problem(lessons_of(reply["arguments"]))

    def tool(self, name, args, request):
        return {"ok": True}, lessons_of(args)


def policy_of(spec):
    return LessonsPolicy(llm_api.APILLM(spec["llm"]), spec["max_tool_calls"], spec["llm_timeout"])


def serve(conn, spec):
    """The worker of an induction (controller_worker.ProcessTransport's target)."""
    controller_worker.serve(conn, spec, policy_of)


def folder_of(seed, loopback=False):
    return STUDIES_DIR / STUDIES[loopback] / f"fold-{'all' if seed is None else seed}"


def committed(seed, loopback=False):
    """The lessons for the walls of `seed`: those of its fold's finished induction, whose pack had no wall of that
    seed; refuses otherwise."""
    folder = folder_of(seed, loopback)
    if not (folder / "lessons.json").exists():
        gpt_smoke.refuse([f"lessons: no finished induction in {folder}"])
    journal = json.loads((folder / "journal.json").read_text(encoding="utf-8"))
    lessons = json.loads((folder / "lessons.json").read_text(encoding="utf-8"))["lessons"]
    seeds = {w["seed"] for w in journal["walls"]}
    problems = [why for why, bad in (("the induction is not finished", journal["state"] != "finished"),
                                     ("the pack has a wall of this seed", seed in seeds),
                                     (oc.lessons_problem(lessons), oc.lessons_problem(lessons))) if bad]
    if problems:
        gpt_smoke.refuse([f"lessons: {folder}: {why}" for why in problems])
    return lessons


def preflight(seed, loopback=False, today=None):
    """(why the fold's induction may not run, its pack, the pack's walls)."""
    from research.analysis import lessons_pack  # pandas: not in the worker's venv
    pack, walls = lessons_pack.pack(lessons_pack.fold(seed))
    llm = llm_api.full(SPEC["llm"])
    problems = gpt_wall.env_problems(loopback) + controller_worker.venv_problems()
    if seed is not None and seed not in lessons_pack.SEEDS:
        problems.append(f"fold: {seed} is not one of the seeds {lessons_pack.SEEDS}")
    if folder_of(seed, loopback).exists():
        problems.append(f"fold: {folder_of(seed, loopback)} exists: an induction is never repeated")
    if (lessons_pack.registered(seed) or {}).get("pack_sha256") != lessons_pack.sha256(pack):
        problems.append(f"pack: not the one in {lessons_pack.PACKS.name} (lessons_pack --write, then commit)")
    if llm_api.call_bound(llm) > SPEC["llm_timeout"] or list(llm["price_per_mtok"] or []) != whole_wall.PRICE:
        problems.append("spec: needs call_bound <= llm_timeout and the wall's price")
    if not loopback:
        if gpt_wall.git("status", "--porcelain", "--", *EXECUTED):
            problems.append("code: the executed files differ from the commit: commit them before a paid induction")
        if ((today or datetime.date.today()) - whole_wall.PRICE_CHECKED).days > whole_wall.PRICE_MAX_AGE:
            problems.append(f"price: PRICE was checked {whole_wall.PRICE_CHECKED}; check it again")
    return problems, pack, walls


def run(seed, loopback=False):
    """One fold's induction (see the module docstring); returns its journal."""
    from research.analysis import lessons_pack
    problems, pack, walls = preflight(seed, loopback)
    if problems:
        gpt_smoke.refuse(problems)
    folder = folder_of(seed, loopback)
    policy = LessonsPolicy(None)
    journal = {"state": "started", "without": seed, "mode": "loopback mock" if loopback else "paid", "spec": SPEC,
               "limits": LIMITS, "prompt_sha256": policy.prompt_sha256, "pack_sha256": lessons_pack.sha256(pack),
               "walls": walls, "git_head": gpt_wall.head(), "code_dirty": bool(gpt_wall.git("status", "--porcelain")),
               "started_at": datetime.datetime.now().isoformat(timespec="seconds")}
    folder.parent.mkdir(parents=True, exist_ok=True)
    try:
        folder.mkdir()  # the lock: two starts of one fold cannot both pass
    except FileExistsError:
        gpt_smoke.refuse([f"fold: {folder} appeared: another run is starting"])
    gpt_smoke.write_json(folder / "journal.json", journal, exclusive=True)  # before the request
    gpt_smoke.write_json(folder / "pack.json", pack)
    transport, state, reasons, spend, lessons, ledger = None, "aborted", [], {}, None, []
    try:
        transport = controller_worker.ProcessTransport(SPEC, target=serve, python=controller_worker.worker_python())
        transport.log = ledger.append
        budget = {"tokens_left": LIMITS["tokens"], "usd_left": LIMITS["cost_usd"]}
        status, reply = transport.decide({"observation": pack, "budget": budget})
        transport.close("induction_end")
        reaped = transport.process is None and transport.unreaped is None and not multiprocessing.active_children()
        gpt_smoke.write_json(folder / "records.json", {"status": status, "reply": reply, "ledger": ledger})
        reply = reply if status == "ok" and isinstance(reply, dict) else {}
        transcripts = reply.get("transcripts") or []
        use = llm_api.spend(SPEC["llm"], transcripts)
        spend = {"http_requests": sum(len((t.get("response") or {}).get("attempts") or []) for t in transcripts),
                 "tokens": use["tokens"] + use["estimated_tokens"], "cost_usd": use["cost_usd"],
                 "unknown_usage": use["unknown"] + (status != "ok")}
        lessons = reply.get("lessons") if reply.get("source") == "llm" else None
        over = [k for k in LIMITS if spend[k] > LIMITS[k]]
        broken = lessons is not None and oc.lessons_problem(lessons)
        reasons = [why for why, bad in (
            (f"worker status {status}", status != "ok"),
            (f"no lessons committed: source {reply.get('source')}", status == "ok" and lessons is None),
            (f"lessons that fail the check: {broken}", broken),
            ("another prompt than this code's", status == "ok" and reply.get("prompt_sha256") != policy.prompt_sha256),
            ("unknown usage", spend["unknown_usage"]),
            ("a worker not reaped", not reaped),
            (f"over the limits: {over}", over),
            ("the key in an output file", gpt_smoke.key_scan(folder))) if bad]
        if not reasons:
            gpt_smoke.write_json(folder / "lessons.json", {"without": seed, "lessons": lessons,
                                                           "pack_sha256": journal["pack_sha256"],
                                                           "prompt_sha256": policy.prompt_sha256})
        state = "uncertain" if reasons else "finished"
    except BaseException as error:  # the journal keeps the fold blocked for inspection
        reasons = ["an exception: " + oc.error_text(error)]
        raise
    finally:
        if transport is not None:
            transport.close("induction_end")
        journal = {**journal, "state": state, "reasons": reasons, "spend": spend, "lessons": lessons,
                   "ended_at": datetime.datetime.now().isoformat(timespec="seconds")}
        gpt_smoke.write_json(folder / "journal.json", journal)
    return journal


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("preflight", "run"))
    ap.add_argument("--without", required=True, help="the test seed left out of the pack, or 'none'")
    ap.add_argument("--loopback", action="store_true")
    ap.add_argument("--confirm-paid", action="store_true")
    a = ap.parse_args(argv)
    seed = None if a.without == "none" else int(a.without)
    if a.cmd == "preflight":
        problems = preflight(seed, a.loopback)[0]
        print(json.dumps({"ok": not problems, "problems": problems, "fold": str(folder_of(seed, a.loopback)),
                          "limits": LIMITS}, indent=1))
        return 0 if not problems else 1
    if not (a.loopback or a.confirm_paid):
        gpt_smoke.refuse(["run: a paid induction needs --confirm-paid (or --loopback for a local mock server)"])
    journal = run(seed, a.loopback)
    print(json.dumps({k: journal[k] for k in ("state", "reasons", "spend", "lessons")}, indent=1))
    return 0 if journal["state"] == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
