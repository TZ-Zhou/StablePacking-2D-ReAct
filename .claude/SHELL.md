# Shell 執行規則 — StablePacking-2D-ReAct

## 環境

- **建牆只在登記的 baseline conda 環境（`stablepacking-baseline`）裡跑。** 路徑、Python 版本與套件清單只記在 `research/validation/environment-manifest.json`；`run_wall.py` 每面牆開跑前都會用 `research/validation/verify_environment.py` 比對，不符就停。
- **控制器 worker 跑在登記的 venv**（`research/validation/worker-venv.json`，含檔案 SHA256），由 `run_wall.py --worker-venv` 啟動。
- 本機找不到登記的環境時，先回報使用者；不得自行重建環境或改寫 manifest。
- 環境比對只在 `run_wall.py` 建牆時執行；純離線分析（`order_sensitivity.py`、`order_agreement.py`、`arm_comparison.py` 等）不建牆，也不做這項比對。

## 執行方式

- 一律從 repo 根目錄執行：`python -B -m research.<資料夾>.<模組> <子指令與參數>`，例如 `python -B -m research.walls.run_wall ...`（`-B` 不寫 bytecode；`research` 是套件，不能用檔案路徑跑）。各模組的子指令寫在檔案開頭的 docstring。
- 新牆由 `run_wall.py` 寫進 `research/cases/<case_id>/`（不進版控），一組牆的設計與 case 清單在 `research/studies/<id>/study.json`；`research/runs/` 是唯讀封存，不再寫入。規則見 `docs/plans/runs-archive-and-cases.md`。評估結果用 `campaign_quality.py` 寫到 `research/quality/<方法>/`。
- 測試：`python -m pytest research/tests`（unittest 寫法）；缺 baseline 環境、openai SDK 或記錄牆的測試會自動 skip。

## 付費 API

- **付費呼叫需要使用者明確同意。** 預設不送：要設 `STONEPACKING_ALLOW_PAID_API=1` 並加 `--confirm-paid` 才會送出。
- 先跑 `preflight`（只檢查，不送）；花費看 `gpt_wall.py ledger`。
- 不得把 API 金鑰寫進指令、檔案或 log。

## 長時間指令

**整批建牆（`campaign.py run`、`counterfactual_branches.py run`、`lookahead_study.py run`、`dev_walls.py run`、`gpt_wall.py run`）一律開獨立視窗，除非使用者說「背景執行」。**
python 加 `-u` 關掉輸出緩衝，並用 `Tee-Object` 同時寫 log，才讀得到進度。

```powershell
$pyExe = "<environment-manifest.json 的 prefix>\python.exe"
$workDir = "C:\Users\GAI\Desktop\Scott\StablePacking-2D-ReAct"
$log = "$workDir\research\cases\<name>.log"
$cmd = "[Console]::OutputEncoding = [Text.Encoding]::UTF8; `$env:PYTHONIOENCODING = 'utf-8'; Set-Location '$workDir'; & '$pyExe' -u -B -m research.<資料夾>.<模組> <args> 2>&1 | Tee-Object -FilePath '$log'; pause"
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $cmd
```

前兩句讓 python 與新視窗都用 UTF-8：新視窗預設以 cp950 解碼，python 以 UTF-8 輸出時中文會變亂碼，log 也一樣。`.claude/hooks/ps_utf8_guard.py` 會擋缺這兩句的新視窗指令。

- 新 PowerShell 視窗預設 `ExecutionPolicy = Restricted`，所以要加 `-ExecutionPolicy Bypass`；python 用絕對路徑呼叫，不靠 activate。
- 開跑後讀 log 確認有實際輸出，再回報。
- 失敗（`FAILED`、`TIMEOUT`、`INCOMPLETE_TRACE`）照實記錄，不重跑掩蓋；campaign 本身就會記下每面牆的狀態。
