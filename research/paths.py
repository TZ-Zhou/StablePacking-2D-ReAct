"""Repository paths of the research package, defined once for every module."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / 'research' / 'runs'
VALIDATION = REPO / 'research' / 'validation'
QUALITY_DIR = REPO / 'research' / 'quality'
