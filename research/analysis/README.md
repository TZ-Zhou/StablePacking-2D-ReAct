# analysis：濾波順序的離線分析與 P0–P2 探索研究

只讀已記錄的追蹤與牆；`counterfactual_branches.py`、`lookahead_study.py` 的 `run` 子指令另外經 `research/walls/run_wall.py` 建分支牆。

## 內容
| 項目 | 說明 |
| --- | --- |
| `order_sensitivity.py` | 在已追蹤的候選分數矩陣上離線重放 120 種濾波順序 |
| `order_agreement.py` | 比較哪些濾波順序在同一牆況下選同一塊石頭 |
| `arm_comparison.py` | 已彙整 campaign 的 arm × 種子探索性比較（唯讀） |
| `unplaced_stones.py` | 牆為何剩下石頭沒放：每次取樣的候選、每塊石頭在各濾波的去留與牆況 |
| `counterfactual_branches.py` | 已記錄 Paper 牆的單步換順序分支：列出選項、建分支牆、彙整 |
| `lookahead_study.py` | P1：在反事實分支點做有限前瞻 |
| `decision_signals.py` | P2：Pareto 前緣與難放石頭兩種簡易決策訊號 |
| `placement_preview.py` | P0：候選擺放的局部品質預覽與 Greedy-Preview 規則（觀測版本 3 A 層草稿，未接進控制器） |
| `placement_preview_study.py` | P0：在已記錄牆上研究上述預覽 |

## 使用
`python -B -m research.analysis.<模組> ...`，子指令與參數見各檔開頭的 docstring；結論寫在 `research/findings.md`。
