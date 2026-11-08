# TODO：尚未完成的項目

## batch

- 在 pc1 跑 `al-th-all-orders`（`research/walls/batch_al_th.yaml`：`AL_R`、`AL_PR`、`AL_IR`、`TH_T20`、`TH_T80`，121 面）。pc1 的 hostname 要先加進 `research/storage/case_id.py` 的 `PCS`，代碼 `pc1`。
- 輸出在 pc1 的 `runs/batch/al-th-all-orders/`，不進 git。
- 濾波順序的完整交叉批次：規模由 pc1 決定，見 `docs/plans/factorial-filter-order-batch.md`。

## 待驗證

- fresh 與持久 worker 只比過 `thangavelu-noise80`、`paper` 的前 5 塊石頭。持久 worker 的 16 面整牆與基準牆零差異，見 `docs/plans/case-speedup-2s-per-stone.md`。比較出差異時從 `6886b87` 查起：

  ```bash
  python -B -m research.runner.verify_speedup final --scenes thangavelu-noise80,fft-d2-0,bricks,almeida-pr --timeout 3600
  ```

- `gpt_wall` loopback 在建牆入口搬到 `research/runner` 之後沒跑過（`lb-001` 是搬移前的牆）。repo 沒有可獨立啟動的 mock 伺服器，只有 `test_gpt_smoke` 內建的測試伺服器。
- case-v1（`feat/case-output`）從未端到端跑過。

## surface-fit heuristic（`docs/plans/surface-fit-heuristic.md`）

- 階段 0–6。
- 驗證 A：74 個分支點上的離線篩選。
- 驗證 B：因果分支。
- 驗證 C：牆的比較，保留 holdout seeds。
- 終局枚舉的校準。
- 常駐 Pool 的相依處理。
- 計畫裡有和 `research/findings.md` 重複的數字，改成只引用。

## 砌牆加速（`docs/plans/case-speedup-2s-per-stone.md`）

第二顆起每顆 < 2 秒已達成，結果與決定都在該計畫，已合併 `main`。

| 項 | 要做什麼 |
| --- | --- |
| worker 預熱（C1） | `configure` 時 `NUMBER_CPU` 還是預設 8，`set_number_cpu(10)` 在其後才跑；Python 3.10 的 executor 要 submit 才建 worker，需要暖機 task |
| 完成後評估 | `tilting_angle.plot_failure_mec` 把 `mechanism.csv` 的寫入移到函式開頭，略過圖時提早返回 |

- `rotate_min_shape_factor` 的平行分支要等 worker 預熱完成才會啟用，目前走序列分支。
- batch 跑完後拆 `run_wall.run`、`configure`。

## 分支與清理（只在 pc2 `GIAAIL-STRIX`）

- 待清理：`feat/case-storage`（含 `../StablePacking-2D-ReAct-cases` worktree）、`chore/project-setup`、本地 `feat/filter-cascade`、`zhou/feat-filter-cascade`、本地 `chore/machine-local-setup`。
- 待刪：`research/runs/verify-perf-*`（21 個）。
- `feat/case-output`（`d55755f`，tag `backup/case-output-d55755f`，worktree 在 `.claude/worktrees/case-output`）尚未併入 `main`：
  - 把 `main` 併進它，在獨立 worktree 解衝突，不要 stash 主工作樹，解完再快轉 `main`。
  - `research/validation/windows_case_runtime.py` 的 `configure()` 兩邊都改：`SharedPool`／`BASELINE_PERSISTENT_POOL` 與 `_keep_records_only`／`BASELINE_CASE_RECORDS` 都要留，自動合併成功也要人工看過。
  - `run_wall.py` 跟著搬到 `research/runner/`，環境變數區塊要同時有兩個變數。`check_run.py` 在試合併時也有衝突。
  - 合併後把 `research/perf/verify_speedup.py` 改用 case-v1，並在 `research/walls/README.md` 加一列 `post_case.py`。

## 已決定

| 項目 | 決定 |
| --- | --- |
| surface-fit 第一版預測對象 | 完成度＋牆高，`F_LR` 當次要檢查 |
| 控制器可否直接選候選 | 可以 |
| 離線建牆批次 | 可以 |
| 第二個驗證場景 | `TH_T20` |
| 砌牆加速的目標 | 第二顆起每顆 < 2 秒，努力到 0.5 秒，結果完全不變 |
| MOSEK `num_threads=1` | 量到超訂才做 |
| 舊單引號 Python 檔被 clean-hook 整檔排版 | 接受，排版變動獨立 commit |
