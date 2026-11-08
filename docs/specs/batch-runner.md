# 批次執行器

`research/runner/batch.py`：先由設定檔抽樣產生案例，再挑要跑的。設定檔與抽樣的定義在 `research/walls/batch_plan.py`。輸出全在 `runs/batch/<name>/`。

```powershell
& $PY -B -m research.runner.batch plan research\walls\batch_example.yaml   # 抽樣，寫 plan.csv
& $PY -B -m research.runner.batch show example                             # 每個案例的設定與狀態
& $PY -B -m research.runner.batch run example --dry-run                    # 只列出每個案例的 run_wall 參數
& $PY -B -m research.runner.batch run example                              # 全部
```

不必先載入 `.env.local`：`run` 自己只取 MOSEK 授權，並移除 API 金鑰。

## 設定檔

| 欄位 | 內容 | 預設 |
| --- | --- | --- |
| `name` | 批次名稱，也是資料夾名稱 | 必填 |
| `samples` | 抽樣數 | 必填 |
| `wall_type` | 場景代碼：`BR`、`AL_R`、`AL_PR`、`AL_IR`、`FF_D20`、`RB`、`TH_T20`、`TH_T80` | 必填 |
| `filter_policy` | 見下表 | `"random"` 加全部 120 種順序 |
| `seed` | 種子清單，或整數範圍 `{min: 1, max: 1000}` | `20` |
| `sampler_seed` | 抽樣的種子；設定相同，案例就相同 | `20` |
| `placements` | 每面牆放到 N 塊就停止，停止前通常會多放幾塊；`0` 建到完 | `0` |
| `timeout` | 每面牆的秒數上限 | `1800` |
| `workers` | 每面牆的 worker 數 | `8` |

`filter_policy` 的值：

| 值 | 意思 |
| --- | --- |
| `"30214"` | 五個濾波器的順序，整面牆固定；要加引號，否則 YAML 會把 `01234` 讀成八進位數字 |
| `"random"` | 每個決策點重新隨機挑順序，每個案例有自己的隨機序列（`run_wall --controller` 的 `random`） |
| `"code_original"` | Wang 程式碼的原始順序 |

多值欄位可寫 `{a, b}` 或 `[a, b]`。

`batch_example.yaml` 逐一列出 `"random"` 和 `"01234"` 到 `"43210"` 全部 120 種順序，要少跑就刪掉不要的。每個值佔相同比例：全部列出時 `samples` 是 121 才每種各抽一次，少於 121 時會有順序抽不到，`random` 也可能抽不到。

## 抽樣

`wall_type`、`filter_policy`、`seed` 三個因子一起做拉丁超立方抽樣，每個因子的各水準出現次數相等（相差不超過 1）。會建出同一面牆的重複抽樣會被去掉，所以案例數可能少於 `samples`；`random` 的案例不會重複。

同一個名稱再 `plan` 一次：設定相同時沿用原案例，不同時拒絕，要換名稱。

## 挑選

| 選項 | 意思 |
| --- | --- |
| `--cases 3-10,15` | 第 3 到 10 個與第 15 個 |
| `--first N` | 前 N 個 |
| `--last M` | 後 M 個 |

編號是 `plan.csv` 的 `case`，從 1 開始。選項可以並用，取聯集；都不給就是全部。

## 進度與續跑

每面牆開始時印 `[目前/總數]`，總數是這次挑選的案例數；在終端機上另有該面牆的進度條。

同一個指令再跑一次就是續跑：

- 已成功的牆跳過。
- Ctrl+C 或當機時正在建的牆，從第一塊石頭重建到新資料夾；中斷的資料夾留著，但不算數。
- 失敗的牆維持失敗，加 `--retry-failed` 才重建。一面牆失敗不會停下整批；只有行程樹無法確認停止（`TERMINATION_FAILED`）才會停。

結束碼：挑選的案例全部成功為 0，有失敗為 1，Ctrl+C 為 130，`TERMINATION_FAILED` 為 2。

`BR` 的磚全部相同，`random` 沒有決策點，順序不影響牆。`run_wall` 的收據會是 `INCOMPLETE_TRACE`，`batch` 視為成功，顯示 `(no decision points)`。

## 輸出

| 項目 | 說明 |
| --- | --- |
| `config.yaml` | 設定檔原文 |
| `plan.csv` | 全部案例：編號、場景、濾波政策、種子、`random` 的控制器種子 |
| `status.jsonl` | 每面牆結束時一行：狀態、`case_id`、放置數、秒數 |
| `cases/<case_id>/` | 每面牆的 case 資料夾 |
| `launcher.log` | 每面牆結束時 `run_wall` 印出的收據 |
| `tmp/` | 暫存檔 |
