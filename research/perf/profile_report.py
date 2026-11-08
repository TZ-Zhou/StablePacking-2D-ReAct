"""Time report of a profile_case run: phases, main-process timers, worker sections, busy workers (read-only).

Usage: python -B -m research.perf.profile_report show <run name or directory> [--top 18]

Reads perf.json, events.jsonl and perf/ of a run written by research/perf/profile_case.py and prints
- phase wall-clock from launch to exit (events of the registered runtime plus the launch and configure times),
- the main-process timers, sorted by time (inclusive: nested timers overlap),
- the worker sections summed over every place_one_stone call and the share of the worker busy time,
- the average number of busy workers: worker busy time per starmap wall second,
- with a cProfile run (profile_case --cprofile): the functions with the most own time over all workers.
Nothing is written.
"""

import argparse
import glob
import json
import pstats
from pathlib import Path

from research.paths import LEGACY_RUNS, PERF_RUNS

CASE_MARKS = (
    ("build_enter", "stone loading, shape factors, clustering"),
    ("build_return", "build_wall_vendor"),
    ("final_evaluation_complete", "final evaluate_kine and failure plot"),
)


def resolve_run(run):
    """A run's folder: a path, a name in research/perf/runs, or a name in the research/runs archive."""
    for path in (Path(run), PERF_RUNS / run):
        if path.is_dir():
            return path
    return LEGACY_RUNS / run


def load_events(run_dir):
    """{event name: seconds after the runtime was configured} from events.jsonl (first occurrence)."""
    events = {}
    for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        events.setdefault(record["event"], record["elapsed_seconds"])
    return events


def load_worker_records(perf_dir):
    lines = (
        line
        for path in sorted(glob.glob(str(perf_dir / "place_*.jsonl")))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
    )
    return sorted((json.loads(line) for line in lines), key=lambda record: record["t_end"])


def phase_rows(events, meta, launch_epoch, wall_seconds):
    """[(label, seconds)] from launch to exit; a phase whose closing event is missing is left out."""
    configured = meta["configure_epoch"] - launch_epoch
    marks = [("interpreter and imports", configured)]  # (phase that ends at this time, time since launch)
    marks += [(label, configured + events[name]) for name, label in CASE_MARKS if name in events]
    rows, previous = [], 0.0
    for label, end in marks:
        rows.append((label, end - previous))
        previous = end
    rows.append(("exit", wall_seconds - previous))
    return rows


def main_rows(stats):
    """[(label, seconds, calls)] of the main-process timers, largest first; the task counter is not a timer."""
    return sorted(
        (
            (label, seconds, int(calls))
            for label, (seconds, calls) in stats.items()
            if label.startswith("main.") and label != "main.pool.tasks"
        ),
        key=lambda row: -row[1],
    )


def worker_sections(records):
    """{label: [seconds, calls]} summed over all calls, from the per-call section deltas."""
    totals = {}
    for record in records:
        for label, (seconds, calls) in record["sections"].items():
            total = totals.setdefault(label, [0.0, 0])
            total[0] += seconds
            total[1] += calls
    return totals


def average_busy_workers(records, starmap_seconds):
    """Worker busy seconds per starmap wall second: how many workers were computing on average."""
    return sum(record["total"] for record in records) / starmap_seconds if starmap_seconds else float("nan")


def cprofile_top(perf_dir, count):
    paths = sorted(glob.glob(str(perf_dir / "cprof_*.pstats")))
    if not paths:
        return []
    merged = pstats.Stats(paths[0])
    merged.add(*paths[1:])
    rows = [
        (own, cumulative, calls, f"{Path(file).name}:{line}({name})")
        for (file, line, name), (_, calls, own, cumulative, _) in merged.stats.items()
    ]
    return sorted(rows, reverse=True)[:count]


def phase_lines(run_dir, record, stats_file):
    wall = record["wall_seconds"]
    rows = phase_rows(load_events(run_dir), stats_file["meta"], record["launch_epoch"], wall)
    header = "phase                                              seconds   share"
    return [header] + [f"{label:48s} {seconds:9.2f} {100 * seconds / wall:6.1f}%" for label, seconds in rows]


def main_lines(stats, top):
    header = "main process timers (inclusive)                    seconds  calls   ms/call"
    rows = main_rows(stats)[:top]
    return [header] + [
        f"{label:48s} {seconds:9.2f} {calls:6d} {1000 * seconds / max(calls, 1):9.2f}" for label, seconds, calls in rows
    ]


def worker_lines(records, starmap_seconds):
    busy = sum(record["total"] for record in records)
    processes = len({record["pid"] for record in records})
    header = f"workers: {len(records)} place_one_stone calls in {processes} processes, busy {busy:.1f}s"
    sections = sorted(worker_sections(records).items(), key=lambda item: -item[1][0])
    return (
        [header]
        + [
            f"{label:48s} {seconds:9.2f} {100 * seconds / busy:6.1f}% {calls:6d}"
            for label, (seconds, calls) in sections
        ]
        + [f"average busy workers during starmap: {average_busy_workers(records, starmap_seconds):.1f}"]
    )


def cprofile_lines(perf_dir, top):
    rows = cprofile_top(perf_dir, top)
    header = "cProfile, all workers                              own_s    cum_s     calls"
    return (
        [header] + [f"{name[:48]:48s} {own:9.2f} {cumulative:8.2f} {calls:9d}" for own, cumulative, calls, name in rows]
        if rows
        else []
    )


def format_report(run_dir, top):
    perf_dir = run_dir / "perf"
    record = json.loads((run_dir / "perf.json").read_text(encoding="utf-8"))
    stats_file = json.loads((perf_dir / "main_stats.json").read_text(encoding="utf-8"))
    stats = stats_file["stats"]
    title = f"{run_dir.name}: {record['case']} wall {record['wall_seconds']:.1f}s"
    sections = [
        [f"{title} persistent_pool={record['persistent_pool']}"],
        phase_lines(run_dir, record, stats_file),
        main_lines(stats, top),
        worker_lines(load_worker_records(perf_dir), stats.get("main.pool.starmap", [0.0, 0])[0]),
        cprofile_lines(perf_dir, top),
    ]
    return "\n\n".join("\n".join(lines) for lines in sections if lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser("show")
    show.add_argument("run")
    show.add_argument("--top", type=int, default=18)
    args = parser.parse_args(argv)
    print(format_report(resolve_run(args.run), args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
