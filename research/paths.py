"""Repository paths of the research package, defined once for every module."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEGACY_RUNS = REPO / 'research' / 'runs'  # the archive of walls built before research/cases; read only
CASES_DIR = REPO / 'research' / 'cases'
STUDIES_DIR = REPO / 'research' / 'studies'
PERF_RUNS = REPO / "research" / "perf" / "runs"  # timing runs of research/perf: not registered walls
SPEEDUP_RUNS = REPO / "runs" / "speedup"  # acceptance walls of research/perf/verify_speedup.py: not registered walls
BATCH_RUNS = REPO / "runs" / "batch"  # batches of research/runner/batch.py: each with its plan, status and walls
VALIDATION = REPO / 'research' / 'validation'
QUALITY_DIR = REPO / 'research' / 'quality'
