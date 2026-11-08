"""Case ids of the walls in research/cases: make, parse and check them, and hash a wall's definition.

A case id is {YYYYMMDD-hhmm}{suffix}-{scene code}-{pc code}, e.g. 20261108-1254-TH_T80-pc1: the start
minute, a suffix b, c, ... when one computer starts the same scene twice in one minute, the scene and
the computer. The policy, filter order, switch, controller, seed and placements limit are left out on
purpose (the order can change while a wall is built): they live in the case's receipt.json, and
spec_sha256 hashes them so that reruns of one definition can be grouped.

Imported by research/walls/run_wall.py (allocate) and by every reader of walls (wall_dir). Walls in the
research/runs archive have no case id: their path there is their id.
"""
import hashlib
import json
import re
import socket
import string
from collections import namedtuple
from datetime import datetime
from pathlib import Path

from research.paths import CASES_DIR, LEGACY_RUNS

SCENES = {  # run_wall case -> scene code
    'almeida-ir': 'AL_IR',
    'almeida-pr': 'AL_PR',
    'almeida-r': 'AL_R',
    'thangavelu-noise20': 'TH_T20',
    'thangavelu-noise80': 'TH_T80',
    'bricks': 'BR',
    'fft-d2-0': 'FF_D20',
    'rubble': 'RB',
}
PCS = {  # host name (upper case) -> pc code; pc2 is added on that computer before it builds a wall
    'GIAAIL-STRIX': 'pc1',
}
SUFFIXES = ('',) + tuple(string.ascii_lowercase[1:])  # '', 'b', ..., 'z'
PATTERN = re.compile(r'(?P<minute>\d{8}-\d{4})(?P<suffix>[b-z]?)-(?P<scene>[A-Z][A-Z0-9_]*)-(?P<pc>pc\d+)')
CaseId = namedtuple('CaseId', 'minute suffix scene pc')


def scene_code(case):
    """The scene code of a run_wall case name."""
    if case not in SCENES:
        raise ValueError(f'no scene code for case {case!r}: add it to research/storage/case_id.SCENES')
    return SCENES[case]


def pc_code(hostname=None):
    """This computer's pc code (or that of `hostname`); a computer missing from PCS may not build walls."""
    host = (hostname or socket.gethostname()).upper()
    if host not in PCS:
        raise ValueError(f'computer {host!r} has no pc code: add it to research/storage/case_id.PCS first')
    return PCS[host]


def make(start, scene, pc, taken=()):
    """The first id not in `taken` for a wall of `scene` started on `pc` at `start` (a datetime)."""
    minute = start.strftime('%Y%m%d-%H%M')
    for suffix in SUFFIXES:
        case_id = f'{minute}{suffix}-{scene}-{pc}'
        if case_id not in taken:
            return case_id
    raise ValueError(f'more than {len(SUFFIXES)} walls of {scene} started on {pc} at {minute}')


def is_case_id(text):
    return PATTERN.fullmatch(str(text)) is not None


def parse(case_id):
    match = PATTERN.fullmatch(str(case_id))
    if match is None:
        raise ValueError(f'not a case id: {case_id!r}')
    return CaseId(**match.groupdict())


def problems(case_id):
    """What is wrong with an id ([] when nothing): its format, its start minute or its scene code."""
    if not is_case_id(case_id):
        return [f'not a case id: {case_id!r}']
    parsed, found = parse(case_id), []
    try:
        datetime.strptime(parsed.minute, '%Y%m%d-%H%M')
    except ValueError:
        found.append(f'{case_id}: no such minute {parsed.minute}')
    if parsed.scene not in SCENES.values():
        found.append(f'{case_id}: unknown scene code {parsed.scene}')
    return found


def spec(receipt):
    """The fields of a receipt that decide its wall; timeout, tracing, venv and paths do not."""
    return {'scene': receipt.get('scene') or scene_code(receipt['case']), 'seed': receipt.get('seed'),
            'filter_policy': receipt.get('filter_policy'), 'controller': receipt.get('controller'),
            'placements_limit': receipt.get('placements_limit')}


def spec_sha256(receipt):
    """SHA256 of spec() as canonical JSON: equal for reruns of one definition, from a receipt in
    memory or one read back from disk (switch steps are int keys in one and str keys in the other)."""
    plain = json.loads(json.dumps(spec(receipt)))
    text = json.dumps(plain, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def allocate(case, start=None, hostname=None, root=CASES_DIR):
    """Reserve the folder of a new wall: its id from the start time, the case's scene and this computer,
    claimed by an exclusive mkdir so that two launches never share one. Returns (case_id, folder)."""
    root = Path(root)
    if root.resolve().is_relative_to(LEGACY_RUNS.resolve()):
        raise ValueError(f'{LEGACY_RUNS} is an archive: new walls go to {CASES_DIR}')
    start = start or datetime.now().astimezone()
    scene, pc, taken = scene_code(case), pc_code(hostname), set()
    root.mkdir(parents=True, exist_ok=True)
    while True:
        case_id = make(start, scene, pc, taken)
        try:
            (root / case_id).mkdir()
        except FileExistsError:
            taken.add(case_id)
            continue
        return case_id, root / case_id


def wall_dir(ref):
    """The folder of a wall: a case id names a folder of research/cases, anything else a path in the
    research/runs archive."""
    return CASES_DIR / ref if is_case_id(ref) else LEGACY_RUNS / ref
