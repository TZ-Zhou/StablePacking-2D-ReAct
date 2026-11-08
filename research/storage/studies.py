"""Studies in research/studies: designed groups of cases, such as a campaign, a branch plan or a lookahead study.

research/studies/<study id>/study.json holds the design, written once when the study is created, and the
cases it has used so far: one entry per case, with whatever the study needs to find it again (arm, seed,
branch name). A case may belong to several studies; only the studies record that, never the case folder.
"""
import json
import os
import re
from datetime import datetime
from pathlib import Path

from research.paths import STUDIES_DIR
from research.storage import case_id as cid

SCHEMA_VERSION = 1
STUDY_ID = re.compile(r'[a-z0-9][a-z0-9-]*')


def path_of(study_id, root=None):
    if not STUDY_ID.fullmatch(study_id):
        raise ValueError(f'study ids are lower-case letters, digits and hyphens: {study_id!r}')
    return Path(STUDIES_DIR if root is None else root) / study_id / 'study.json'


def write_whole(path, study):
    """Replace the file in one step, so a crash never leaves half a study behind."""
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(study, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def create(study_id, kind, design, root=None):
    """Write a new study with no cases yet; refuses an existing one, so a design is never overwritten."""
    path = path_of(study_id, root)
    if path.exists():
        raise FileExistsError(f'study {study_id} exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    study = {'schema_version': SCHEMA_VERSION, 'id': study_id, 'kind': kind,
             'created': datetime.now().astimezone().isoformat(timespec='seconds'), 'design': design, 'cases': []}
    write_whole(path, study)
    return study


def load(study_id, root=None):
    return json.loads(path_of(study_id, root).read_text(encoding='utf-8'))


def add_case(study_id, entry, root=None):
    """Append one entry ({'case_id': ..., and the study's own keys}); a case is listed once per study."""
    problems = cid.problems(entry.get('case_id', ''))
    if problems:
        raise ValueError('; '.join(problems))
    study = load(study_id, root)
    if any(e['case_id'] == entry['case_id'] for e in study['cases']):
        raise ValueError(f"{entry['case_id']} is already in study {study_id}")
    study['cases'].append(entry)
    write_whole(path_of(study_id, root), study)
    return study


def find(study_id, root=None, **keys):
    """The study's entries whose fields equal `keys`, in the order they were added."""
    return [e for e in load(study_id, root)['cases'] if all(e.get(k) == v for k, v in keys.items())]
