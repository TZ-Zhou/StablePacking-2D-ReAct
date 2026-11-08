"""research/llm_walls/step_quality.py: the pair split on known series, and on gpt-001's records the step series
ends at wang-quality-v1 with the chosen LLC at every placement."""
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO)]
from research.llm_walls import step_quality as sq  # noqa: E402


def rows(values, stones):
    return [{'n': n, 'step': n, 'stone_id': s, 'stone_px': 10 * n, **{m: v for m in sq.METRICS}}
            for n, (v, s) in enumerate(zip(values, stones), 1)]


class Split(unittest.TestCase):
    def test_decision_and_after(self):
        a, b = rows([1, 1, 2, 5], [1, 2, 3, 4]), rows([1, 1, 3, 3], [1, 2, 4, 3])
        d = sq.pair_rows(a, b, 3)
        self.assertEqual([r['phase'] for r in d], ['prefix', 'prefix', 'decision', 'after'])
        self.assertEqual([r['stones_only_a'] for r in d], [0, 0, 1, 0])  # the same stones again at n = 4
        s = sq.pair_summary(d, 3)['F_LR']
        self.assertEqual((s['at_decision'], s['final'], s['after_decision'], s['onset_n'], s['onset_phase']), (-1, 2, 3, 3, 'decision'))

    def test_a_difference_that_starts_later(self):
        d = sq.pair_rows(rows([1, 1, 1, 4], [1, 2, 3, 4]), rows([1, 1, 1, 2], [1, 2, 5, 4]), 3)
        s = sq.pair_summary(d, 3)['F_SF']
        self.assertEqual((s['at_decision'], s['onset_n'], s['onset_phase'], s['after_decision']), (0, 4, 'after', 2))


@unittest.skipUnless((sq.LEGACY_RUNS / sq.WALLS['GPT']).exists() and (sq.LEGACY_RUNS / sq.CF / 'summary.json').exists(), 'records not present')
class Gpt001(unittest.TestCase):
    def test_the_series_ends_at_wang_quality_v1(self):
        steps = sq.wall_steps('GPT')
        final = {r['wall']: r for r in json.loads((sq.LEGACY_RUNS / sq.CF / 'summary.json').read_text(encoding='utf-8'))['rows']}['V']
        self.assertEqual(sq.wall_problems(steps, {**final, **sq.metrics.mechanism_lambdas(sq.result_of(sq.WALLS['GPT']))}), [])
        self.assertEqual(len(steps), 18)
        broken = [dict(steps[-1], F_AH_partial=steps[-1]['F_AH_partial'] + 1e-9)]
        self.assertTrue(sq.wall_problems(steps[:-1] + broken, {**final, 'left': final['lambda_left'], 'right': final['lambda_right']}))


if __name__ == '__main__':
    unittest.main()
