"""Repository paths of the research package, defined once for every module."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEGACY_RUNS = REPO / 'research' / 'runs'  # the archive of walls built before research/cases; read only
CASES_DIR = REPO / 'research' / 'cases'
STUDIES_DIR = REPO / 'research' / 'studies'
PERF_RUNS = REPO / "research" / "perf" / "runs"  # timing runs of research/perf: not registered walls
VALIDATION = REPO / 'research' / 'validation'
QUALITY_DIR = REPO / 'research' / 'quality'
