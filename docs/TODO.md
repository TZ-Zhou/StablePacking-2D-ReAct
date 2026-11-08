# TODO：尚未完成的項目

## 待驗證

- `research/analysis/myopia_tables.py`：建出的表還沒和原本的分析表逐欄比對。
- `research/analysis/myopia_report.py`：還沒逐行對照 `research/findings.md` 的 Temporal myopia 一節；對不上的要先查原因，再改模組或改結論。
- `research/analysis/surface_fit_check.py`：還沒確認它重現 82.6%／17.4%／94.6% 與各 heuristic 耗時。
- 上面三支與 `surface_fit.py` 只跑過各自的單元測試（小型手造輸入，全數通過），還沒在封存牆上完整跑過。
- `research/tests/test_worker_pool.py` 目前失敗：`PermissionError: [WinError 5]`，出在 `_winapi.DuplicateHandle`。還沒判定是執行環境或程式造成。
- 完整測試集還沒跑過。`research/runner/` 這次改動只跑了 `test_runner_layout`、`test_batch`、`test_step_writer`、`test_stone_bar`、`test_verify_speedup`，都通過。

## surface-fit heuristic（`docs/plans/surface-fit-heuristic.md`）

待決：
- 第一版預測目標：完成度＋牆高，或納入 `F_LR`。
- 控制器可否直接選候選，或只能換濾波順序。
- 可否跑離線建牆批次。
- 第二個驗證場景選哪一個。

待執行：
- 階段 0–6。
- 驗證 A：74 個分支點上的離線篩選。
- 驗證 B：因果分支。
- 驗證 C：牆的比較，保留 holdout seeds。
- 終局枚舉的校準。
- 常駐 Pool 的相依處理。

文件：
- 計畫裡有和 `research/findings.md` 重複的數字；依 `.claude/DOC_STYLE.md` 決定是否改成只引用。

## 分支

- `feat/case-output`（`d55755f`）尚未併入 `main`。
- 砌牆加速與 runner 重整在 `pref/case-speedup`，完整牆比較通過前不併入 `main`。
- 待清理：`feat/case-storage`（含 `-cases` worktree）、`chore/project-setup`、本地 `feat/filter-cascade`、`zhou/feat-filter-cascade`、本地 `chore/machine-local-setup`。

## 砌牆加速（`docs/plans/case-speedup-2s-per-stone.md`）

待驗證：
- 完整比較：T80、FFT、bricks、PR 完整牆，2 政策，`--fresh-pool` 與持久 worker 各一次，含封存牆比對。階段 3–6 只有前段短牆與單元比對的證據，整面牆是否相同要靠這次比較：

  ```bash
  python -B -m research.runner.verify_speedup final --scenes thangavelu-noise80,fft-d2-0,bricks,almeida-pr --timeout 3600
  ```

- `thangavelu-noise80` 在階段 3–6 改動後未驗證；改動前每步約 12.9 秒，150 秒內放 9 顆就逾時。
- `test_lookahead` 要 MOSEK 授權，先載 `.env.local` 再跑。
- 階段 6 少一個測試：寫檔停在第 k 項時，`check_commits(stopped=True)` 無問題。
- 階段 3 的序列化相等只用合成陣列比過，沒有真實步驟的 payload；協定 5 下唯讀陣列仍是唯讀，協定 4 會變可寫。

未做：

| 項 | 要做什麼 |
| --- | --- |
| 4a 重用 `mosek.Env` | 載授權後比對重用與新建 Env 的求解結果 |
| 4g `num_threads=1` | 計畫限定先量到超訂才做，且會改變求解路徑 |
| 階段 5 worker 預熱 | `configure` 時 `NUMBER_CPU` 還是預設 8，`set_number_cpu(10)` 在其後才跑；Python 3.10 的 executor 要 submit 才建 worker，需要暖機 task |
| 階段 5 完成後評估 | `tilting_angle.plot_failure_mec` 把 `mechanism.csv` 的寫入移到函式開頭，略過圖時提早返回 |

`rotate_min_shape_factor` 的平行分支要等 worker 預熱完成才會啟用，目前走序列分支。

待決定：
- D4：PR 是否一定要 < 2 秒/顆；階段 3–6 後前 11 顆約 2.5 秒/顆，尚未達標。
- D5：workers 是否超過 10。
- 是否清理 `research/runs/verify-perf-*`。
- 編輯舊單引號 Python 檔時 clean-hook 會整檔排版，是否接受。
- `run_wall.run`、`configure` 過長，是否拆分。
