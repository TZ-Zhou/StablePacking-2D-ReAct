"""The research/runs archive: walls built before research/cases, kept in place and never written again.

  python -B -m research.storage.legacy_runs index     write research/runs/legacy-index.csv
  python -B -m research.storage.legacy_runs freeze    make every file under research/runs read-only

Old walls keep their path under research/runs as their id; they get no case id. A receipt whose run_id
is not its own path is a copy of another wall (the API smoke tests copy the wall they read): the index
names the original in copy_of.
"""
import argparse
import json
import os
import stat
from pathlib import Path

from research.paths import LEGACY_RUNS
from research.storage import case_index

INDEX = 'legacy-index.csv'
SKIPPED = ('case_id', 'pc', 'started', 'parent')  # case-only columns: an old wall has none of them
COLUMNS = ('path', 'copy_of') + tuple(c for c in case_index.COLUMNS if c not in SKIPPED)
WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


def receipts(root=LEGACY_RUNS):
    """(path relative to root, receipt) of every wall in the archive, sorted by path."""
    for p in sorted(Path(root).rglob('receipt.json')):
        yield p.parent.relative_to(root).as_posix(), json.loads(p.read_text(encoding='utf-8'))


def row(path, receipt):
    columns = case_index.row(receipt)
    run_id = receipt.get('run_id', path)
    return {'path': path, 'copy_of': '' if run_id == path else run_id, **{k: columns[k] for k in COLUMNS[2:]}}


def write_index(root=LEGACY_RUNS):
    """Rewrite root/legacy-index.csv (the one file a frozen archive lets change); returns (path, rows)."""
    out = Path(root) / INDEX
    rows = [row(path, receipt) for path, receipt in receipts(root)]
    if out.exists():
        out.chmod(out.stat().st_mode | stat.S_IWUSR)
    case_index.write_csv(out, COLUMNS, rows)
    return out, len(rows)


def freeze(root=LEGACY_RUNS):
    """Clear the write permission of every file under root (on Windows: the read-only attribute);
    returns the number of files."""
    files = [p for p in Path(root).rglob('*') if p.is_file()]
    for p in files:
        p.chmod(p.stat().st_mode & ~WRITE_BITS)
    return len(files)


def writable(root=LEGACY_RUNS):
    """The files under root that can still be written ([] once frozen)."""
    return [p for p in Path(root).rglob('*') if p.is_file() and os.access(p, os.W_OK)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('command', choices=('index', 'freeze'))
    args = parser.parse_args(argv)
    if args.command == 'index':
        out, n = write_index()
        print(f'{n} walls in {out}')
    else:
        print(f'{freeze()} files under {LEGACY_RUNS} are read-only; still writable: {len(writable())}')


if __name__ == '__main__':
    main()
