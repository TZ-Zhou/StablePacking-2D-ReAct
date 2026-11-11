"""Build one live GPT wall under the guarantees of research/llm_walls/gpt_wall.py.

  python -B -m research.runner.gpt_wall preflight --name gpt-002 --seed S [--after NAME] [--loopback]
  STONEPACKING_ALLOW_PAID_API=1 python -B -m research.runner.gpt_wall run --name gpt-002 --seed S --after gpt-001 \
      --confirm-paid
  python -B -m research.runner.gpt_wall run --name lb-001 --seed S --loopback [--placements N]   # a local mock server
  python -B -m research.runner.gpt_wall ledger [--loopback]                                     # what the walls spent

Run it in the stablepacking-baseline environment (run_wall builds the wall there).
"""

import argparse
import datetime
import json
import os

from research.controller import order_controller
from research.llm_walls import gpt_smoke, gpt_wall
from research.runner import run_wall as wall_runner
from research.storage import case_id as cid
from research.walls import case_definition


def run_wall(name, seed, after=None, loopback=False, placements=0, launch=wall_runner.run, x=None, wall_args=(),
             spec=None):
    """One GPT wall (see the module docstring) under the study's lock; returns its report (<case>/llm/report.json).
    x: the experiment (gpt_wall.experiment; default the GPT walls); wall_args: run_wall arguments after x.WALL_ARGS."""
    if placements and not loopback:
        raise SystemExit("refused: --placements only for a loopback wall")
    x = gpt_wall.experiment(x)
    spec = x.SPEC if spec is None else spec  # this wall's own (lessons_wall.spec_of)
    problems = gpt_wall.preflight(name, seed, after, loopback, x=x)
    if problems:
        gpt_smoke.refuse(problems)
    study, lock = x.STUDIES[loopback], gpt_wall.lock_of(loopback, x)
    # built before the lock: a git failure here leaves nothing behind
    journal = {
        "state": "started",
        "after": after,
        "seed": seed,
        "study": study,
        "mode": "loopback mock" if loopback else "paid",
        "spec": order_controller.full_spec(spec),
        "limits": x.WALL_LIMITS,
        "experiment_limits": x.EXPERIMENT_LIMITS,
        "max_http_requests": gpt_wall.max_requests(order_controller.full_spec(spec)),
        "placements": placements,
        **({"wall_args": list(wall_args)} if wall_args else {}),
        "code_dirty": bool(gpt_wall.code_dirty(x)),
        "git_head": gpt_wall.head(),
        "worker_venv": case_definition.worker_venv_identity(),
    }
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        gpt_wall.write_json(lock, {"run": name, "pid": os.getpid()}, exclusive=True)
    except FileExistsError:
        gpt_smoke.refuse([f"run: {lock} appeared: another run is starting"])
    problems = gpt_wall.journal_problems(name, after, loopback, holding=True, x=x)  # again, under the lock
    if problems:
        lock.unlink()
        gpt_smoke.refuse(problems)
    journal["started_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    report, reasons, state, written = {}, [], "aborted", False
    try:
        # the journal before the wall starts: in the lock, as run_wall only then allocates the case
        gpt_wall.write_json(lock, {"run": name, "pid": os.getpid(), "journal": journal})
        returned = launch(
            [
                *x.WALL_ARGS,
                *wall_args,
                "--seed",
                str(seed),
                "--timeout",
                str(x.WALL_LIMITS["seconds"]),
                "--controller",
                json.dumps(spec),
            ]
            + (["--placements", str(placements)] if placements else [])
        )
        journal["case_id"] = returned["case_id"]
        run = cid.wall_dir(returned["case_id"])
        files = gpt_smoke.case_records(run)
        # the study lists the case first: from here on a case without a journal blocks too
        gpt_smoke.join_study(
            study, x.STUDY_KIND, gpt_wall.study_design(loopback, x), {"case_id": returned["case_id"], "name": name}
        )
        files.journal.parent.mkdir()
        gpt_wall.write_json(files.journal, journal, exclusive=True)
        written = True
        receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
        ok_status = ("PROCESS_PASS", "STOPPED_AFTER_PLACEMENTS") if placements else ("PROCESS_PASS",)
        report = gpt_wall.accounting(run, spec, receipt.get("process_status") == "STOPPED_AFTER_PLACEMENTS")
        report["check"] = {
            k: v
            for k, v in order_controller.check(next(run.glob("result*"))).items()
            if k
            in (
                "ok",
                "verdict",
                "problems",
                "incomplete",
                "records",
                "decision_points",
                "switches",
                "geometry",
                "provenance",
            )
        }
        report.update(status=receipt["status"], elapsed_seconds=receipt.get("elapsed_seconds"))
        gpt_wall.write_json(files.report, report)
        report["key_scan"] = gpt_smoke.key_scan(run)  # llm/report.json included
        over = [k for k in ("http_requests", "tokens", "cost_usd") if report[k] > x.WALL_LIMITS[k]]
        reasons = [
            why
            for why, bad in (
                (f"case status {receipt['status']}", receipt["status"] not in ok_status),
                (f"check verdict {report['check'].get('verdict')}", report["check"].get("verdict") != "PASS"),
                ("unknown usage", report["unknown_usage"]),
                ("a worker failure", report["worker_failures"]),
                ("requests that may have gone out unrecorded", not report["http_requests_exact"]),
                ("a worker not reaped", not report["workers_reaped"]),
                (f"over the wall limits: {over}", over),
                ("over the time limit", (receipt.get("elapsed_seconds") or 1e9) > x.WALL_LIMITS["seconds"]),
                ("the key in an output file", report["key_scan"]),
            )
            if bad
        ]
        gpt_wall.write_json(files.report, report)
        state = "uncertain" if reasons else "finished"
    except BaseException as error:  # the journal keeps the experiment blocked for inspection
        reasons = ["an exception: " + order_controller.error_text(error)]
        raise
    finally:
        final = {
            **journal,
            "state": state,
            "reasons": reasons,
            "ended_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "spend": {k: report.get(k) for k in ("http_requests", "tokens", "cost_usd", "unknown_usage")},
            "gpt_complete": report.get("gpt_complete"),
        }
        if written:
            gpt_wall.write_json(files.journal, final)
            lock.unlink()
        else:  # no journal in a case of the study: the lock keeps the final state and blocks every later wall
            gpt_wall.write_json(lock, {"run": name, "pid": os.getpid(), "journal": final})
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("preflight", "run"):
        p = sub.add_parser(cmd)
        p.add_argument("--name", required=True)
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--after")
        p.add_argument("--loopback", action="store_true")
        if cmd == "run":
            p.add_argument("--confirm-paid", action="store_true")
            p.add_argument("--placements", type=int, default=0)
    sub.add_parser("ledger").add_argument("--loopback", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "ledger":
        shown = ("state", "reasons", "spend", "gpt_complete", "case_id")
        listed = {
            n: {k: (j or {"state": "no journal"}).get(k) for k in shown}
            for n, j in gpt_wall.journals(a.loopback).items()
        }
        print(
            json.dumps(
                {
                    "spent": gpt_wall.spent(a.loopback),
                    "limits": gpt_wall.EXPERIMENT_LIMITS,
                    "lock": gpt_smoke.read_journal(gpt_wall.lock_of(a.loopback)),
                    "walls": listed,
                },
                indent=1,
            )
        )
        return 0
    if a.cmd == "preflight":
        problems = gpt_wall.preflight(a.name, a.seed, a.after, a.loopback)
        print(
            json.dumps(
                {
                    "ok": not problems,
                    "problems": problems,
                    "max_http_requests": gpt_wall.max_requests(order_controller.full_spec(gpt_wall.SPEC)),
                },
                indent=1,
            )
        )
        return 0 if not problems else 1
    if not (a.loopback or a.confirm_paid):
        gpt_smoke.refuse(["run: a paid wall needs --confirm-paid (or --loopback for a local mock server)"])
    report = run_wall(a.name, a.seed, a.after, a.loopback, a.placements)
    print(json.dumps({k: v for k, v in report.items() if k != "requests"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
