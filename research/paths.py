"""Repository paths of the research package, defined once for every module."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / 'research' / 'runs'
LEGACY_RUNS = RUNS  # the archive of walls built before research/cases; read only
CASES_DIR = REPO / 'research' / 'cases'
STUDIES_DIR = REPO / 'research' / 'studies'
VALIDATION = REPO / 'research' / 'validation'
QUALITY_DIR = REPO / 'research' / 'quality'
