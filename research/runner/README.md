# runner：所有建牆的啟動點與執行邏輯

一面牆或一批牆都從這裡開始。這裡只放「怎麼跑」：命令列、啟動與監看行程、中斷續跑、進度、收據。「跑什麼」的定義留在各研究資料夾，定義檔不得 import `research.runner`，由 `research/tests/test_runner_layout.py` 檢查。

從 repo 根目錄執行 `python -B -m research.runner.<模組> ...`，子指令與參數見各檔開頭的 docstring。

## 單面牆

| 模組 | 建什麼 | 定義所在的資料夾與模組 |
| --- | --- | --- |
| `run_wall.py` | 一面牆：暫存原始案例、監看放置數與逾時、驗證追蹤、寫收據。其他模組都經由它建牆 | walls 的 case_definition |
| `gpt_wall.py` | 一面 GPT 實牆，含鎖、日誌與花費帳本；僅 `run`（不含 `--loopback`）呼叫付費 API | llm_walls 的 gpt_wall |
| `profile_case.py` | 一面牆加階段計時；不追蹤、不是登記牆 | walls 的 case_definition |

## 多面牆

| 模組 | 建什麼 | 定義所在的資料夾與模組 |
| --- | --- | --- |
| `batch.py` | 由設定檔抽樣的批次，可挑區間、看進度、中斷續跑；用法見 `docs/specs/batch-runner.md` | walls 的 batch_plan |
| `campaign.py` | 單一場景的濾波順序 × 種子實驗 | walls 的 campaign |
| `dev_walls.py` | 開發種子上的 Paper 牆與各控制器牆 | llm_walls 的 dev_walls |
| `counterfactual_branches.py` | 已記錄 Paper 牆的單步換順序分支 | analysis 的 counterfactual_branches |
| `gpt_counterfactual.py` | GPT 牆的反事實分支，不呼叫 API | llm_walls 的 gpt_counterfactual |
| `lookahead_study.py` | P1 有限前瞻牆與評估 | analysis 的 lookahead_study |
| `verify_speedup.py` | 持久 worker 與每批新 Pool 各建一次，比對結果並量時間 | perf 的 verify_speedup |

## 共用

| 模組 | 說明 |
| --- | --- |
| `environment.py` | 建牆環境：移除 API 金鑰，`.env.local` 只取 MOSEK 授權與 `PYTHONNOUSERSITE` |

只讀牆的工具（check_run、離線分析與各種報告）不在這裡。
