# analysis：濾波順序的離線分析與 P0–P2 探索研究

只讀已記錄的追蹤與牆；`counterfactual_branches.py`、`lookahead_study.py` 只定義分支牆與前瞻牆，建牆由 `research/runner/` 的同名模組執行。

## 內容
| 項目 | 說明 |
| --- | --- |
| `order_sensitivity.py` | 在已追蹤的候選分數矩陣上離線重放 120 種濾波順序 |
| `order_agreement.py` | 比較哪些濾波順序在同一牆況下選同一塊石頭 |
| `arm_comparison.py` | 已彙整 campaign 的 arm × 種子探索性比較（唯讀） |
| `unplaced_stones.py` | 牆為何剩下石頭沒放：每次取樣的候選、每塊石頭在各濾波的去留與牆況 |
| `counterfactual_branches.py` | 已記錄 Paper 牆的單步換順序分支：列出選項、規劃分支、檢查與彙整 |
| `lookahead_study.py` | P1：反事實分支點上的有限前瞻：前瞻牆的定義與評估 |
| `decision_signals.py` | P2：Pareto 前緣與難放石頭兩種簡易決策訊號 |
| `placement_preview.py` | P0：候選擺放的局部品質預覽與 Greedy-Preview 規則（觀測版本 3 A 層草稿，未接進控制器） |
| `placement_preview_study.py` | P0：在已記錄牆上研究上述預覽 |
| `myopia_tables.py` | 從封存的 TH_T80 牆建每步、每牆與去重後的表 |
| `batch_tables.py` | 從一個 batch 資料夾（`runs/batch/<名稱>/`）建每牆、每步與去重表，寫到 `research/datasets/<名稱>/` |
| `branch_concordance.py` | C2、C4：每個種子 × 場景的前綴樹上，分支點各子節點的分數（當下分數或 h）和最終牆的一致率與區組符號翻轉檢定 |
| `myopia_report.py` | 由上述表重算研究結論裡的 Temporal myopia 數字 |
| `surface_fit.py` | 「剩餘石頭和新頂面還配不配」的候選 heuristic（純函式，未接進控制器） |
| `surface_fit_check.py` | 垂直落下模型與實際放置的比對，以及各 heuristic 的耗時 |

## 使用
`python -B -m research.analysis.<模組> ...`，建牆的兩個研究改用 `python -B -m research.runner.<模組> ...`；子指令與參數見各檔開頭的 docstring；結論寫在 `research/findings.md`。
