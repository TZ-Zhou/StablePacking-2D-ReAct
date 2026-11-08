"""research/cases/index.csv: one row per case, rebuilt from the receipts and never edited by hand.

  python -B -m research.storage.case_index        rewrite the index from every receipt in research/cases

row() turns one receipt into the index columns and write_csv() writes them; research/storage/legacy_runs.py
uses both for the archive's index, so the two indexes read a receipt the same way.
"""
import csv
import json
from pathlib import Path

from research.paths import CASES_DIR
from research.storage import case_id as cid

INDEX = 'index.csv'
COLUMNS = ('case_id', 'scene', 'pc', 'started', 'seed', 'policy', 'order', 'switch', 'controller', 'llm',
           'placements_limit', 'parent', 'status', 'committed_placements', 'elapsed_seconds', 'spec_sha256',
           'git_head', 'case')


def joined(order):
    return '' if order is None else '-'.join(str(i) for i in order)


def switch_text(switch):
    """'10:0-3-1-4-2 12:...' in step order; step keys may be int (in memory) or str (read from JSON)."""
    steps = sorted((int(k), v) for k, v in (switch or {}).items())
    return ' '.join(f'{step}:{joined(order)}' for step, order in steps)


def llm_text(controller):
    """'<provider>:<model>' of an API LLM, 'mock:<mode>' of a mock one, '' for a controller without an LLM."""
    llm = (controller or {}).get('llm') or {}
    if 'kind' not in llm:
        return ''
    return f"{llm.get('provider') or llm['kind']}:{llm.get('model') or llm.get('mode') or ''}"


def row(receipt):
    """The index columns of one receipt, blank where the receipt has no such field."""
    policy, controller = receipt.get('filter_policy') or {}, receipt.get('controller') or {}
    own_id = receipt.get('case_id', '')
    return {
        'case_id': own_id, 'scene': receipt.get('scene') or cid.SCENES.get(receipt.get('case'), ''),
        'pc': cid.parse(own_id).pc if cid.is_case_id(own_id) else '', 'started': receipt.get('started', ''),
        'seed': receipt.get('seed', ''), 'policy': policy.get('policy', ''), 'order': joined(policy.get('order')),
        'switch': switch_text(policy.get('switch')), 'controller': controller.get('kind', ''),
        'llm': llm_text(controller), 'placements_limit': receipt.get('placements_limit', ''),
        'parent': receipt.get('parent') or '', 'status': receipt.get('status', ''),
        'committed_placements': receipt.get('committed_placements', ''),
        'elapsed_seconds': receipt.get('elapsed_seconds', ''),
        'spec_sha256': receipt.get('spec_sha256') or cid.spec_sha256(receipt),
        'git_head': receipt.get('git_head', ''), 'case': receipt.get('case', ''),
    }


def write_csv(path, columns, rows):
    """UTF-8 CSV with a header row; replaces the file whole."""
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def rebuild(root=None):
    """Rewrite root/index.csv from root/*/receipt.json, sorted by case id; returns the number of rows."""
    root = Path(CASES_DIR if root is None else root)
    receipts = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(root.glob('*/receipt.json'))]
    write_csv(root / INDEX, COLUMNS, [row(r) for r in receipts])
    return len(receipts)


def main():
    print(f'{rebuild()} cases in {CASES_DIR / INDEX}')


if __name__ == '__main__':
    main()
