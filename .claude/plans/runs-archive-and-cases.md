# 計畫：封存 research/runs，新牆依 case_id 存放

`research/runs/` 封存、不再寫入；之後每一面牆是一個 case，存到 `research/cases/<case_id>/`。case_id 的格式仍在討論（見最後一節），其餘已定案。

## 名詞

| 名詞 | 意思 |
| --- | --- |
| case | 一次建牆執行。一面牆＝一個 case＝一個資料夾 |
| scene | Wang 的測試場景（`thangavelu-noise80`、`almeida-pr`、`bricks`、`fft-d2-0`）。現行程式與 receipt 的 `case` 欄位就是 scene；receipt 保留 `case` 欄位以相容舊紀錄，新欄位與 index 一律用 `scene` |
| study | 一組事先設計好的 case（campaign、分支計畫、前瞻研究）。一個 case 可以屬於多個 study，關係只記在 study，不靠資料夾巢狀 |
| parent | 分支與前瞻牆的母牆，以 case_id 記在 receipt |

## 資料夾

| 路徑 | 內容 | git |
| --- | --- | --- |
| `research/runs/` | 封存的舊紀錄，唯讀 | 忽略 |
| `research/cases/<case_id>/` | 新牆，一 case 一資料夾，全部同一層 | 忽略 |
| `research/cases/index.csv` | 一 case 一列，由各 receipt 重建，不手改 | 忽略 |
| `research/studies/<study_id>/study.json` | study 的設計（arm × 種子、分支計畫）與用到的 case_id | 追蹤 |

## 一個 case 資料夾

```
research/cases/<case_id>/
  receipt.json               定義、來源、結果摘要；建牆結束寫一次，之後不改
  run.py、config.json、windows_case_runtime.py
  execution.log、environment-guard.log、events.jsonl
  result_<時間>_wall0/        Wang 的輸出原樣保留（base.pkl、img/、record/、trace/…）
  llm/                       只有 LLM 控制器才有：journal.json、report.json
  derived/<方法版本>/         可重算的分析，例如 wang-quality-v1/、order-sensitivity-v2/
```

receipt.json 在現有欄位之外新增：`schema_version`、`case_id`、`scene`、`parent`（母牆 case_id，沒有則為 null）、`spec_sha256`（定義的雜湊，同一定義的重跑以它分組）。

## 存檔格式

| 資料 | 格式 |
| --- | --- |
| 定義與摘要 | JSON，UTF-8，帶 `schema_version` |
| 逐步事件 | JSONL，只追加 |
| 表格（index、指標、敏感度） | CSV，UTF-8，有表頭 |
| 陣列（trace） | NPZ |
| 牆的完整狀態 | `base.pkl`；版本相依以 receipt 的環境 manifest sha 對應 |
| 時間 | ISO 8601，帶時區（`+08:00`） |

## 舊 research/runs 封存

- 原地凍結，不搬檔案：程式、測試、`research/quality/*/quality.csv` 與 `research/findings.md` 都以現有路徑引用這些紀錄。
- 新增 `research/runs/ARCHIVE.md`（封存說明）與 `research/runs/legacy-index.csv`（每面舊牆一列：舊路徑、scene、seed、政策、控制器、狀態、git_head、spec_sha256）。
- 舊牆不補 case_id，一律以舊路徑識別。
- 資料夾設為唯讀；`run_wall` 拒絕寫入 `research/runs/`。
- 測試的暫存牆改用系統暫存目錄，不再寫 `research/runs/tmp-tests/`。
- `base.pkl` 不壓縮：`research/tests/test_wall_quality.py` 以雜湊釘住紀錄樹。

## 程式

一律依 `.claude/PYTHON_CODING.md`：英文 docstring、套件內絕對匯入、路徑只從 `research/paths.py` 取、一個概念一份實作、核心計算寫成純函式、IO 集中在邊界。

新程式放 `research/storage/`（案例的存放與索引）：

| 模組 | 職責 |
| --- | --- |
| `storage/case_id.py` | 由定義產生、解析、檢查 case_id；scene 代碼表只在這裡 |
| `storage/case_index.py` | 掃描 receipt，重建 `research/cases/index.csv` |
| `storage/legacy_runs.py` | 產生 `research/runs/legacy-index.csv`、設唯讀 |
| `storage/studies.py` | 讀寫 `study.json` |

測試對應放 `research/tests/`：`test_case_id.py`、`test_case_index.py`、`test_legacy_runs.py`、`test_studies.py`。

既有模組的改動：

| 模組 | 改動 |
| --- | --- |
| `paths.py` | `RUNS` 改為唯讀的 `LEGACY_RUNS`；新增 `CASES`、`STUDIES` |
| `walls/run_wall.py` | 配 case_id、寫到 `cases/`、receipt 新欄位、拒寫舊資料夾 |
| `walls/campaign.py`、`llm_walls/dev_walls.py` | 以 study 記錄 arm × 種子，不再巢狀命名 |
| `analysis/counterfactual_branches.py`、`analysis/lookahead_study.py` | 母牆以 `parent` 記錄 |
| `llm_walls/gpt_wall.py`、`llm_walls/gpt_smoke.py` | 輸出改到 `cases/`；LLM 紀錄放 `llm/` |
| `walls/campaign_quality.py`、`analysis/order_sensitivity.py` | 輸出改到 `derived/<方法版本>/` |

## 實作順序

在新分支上依序進行，每步一個 commit：

1. `storage/` 與 `paths.py`，含測試。
2. 封存舊 runs：`ARCHIVE.md`、`legacy-index.csv`、唯讀、拒寫。
3. 寫入端改用 `cases/`、`parent` 與 `studies/`。
4. 衍生分析改存 `derived/`。
5. 測試路徑、`research/README.md`、`.claude/SHELL.md`、`.claude/PYTHON_CODING.md` 同步。
6. `.gitignore` 加入 `research/cases/`（該檔目前有他人未提交的修改，提交前先協調）。

## case_id

格式：`{YYYYMMDD-hhmm}-{場景代碼}-{pc1|pc2}`，例如 `20261108-1254-TH_T80-pc1`；段與段之間用 `-`，段內用 `_`。

| 場景代碼 | 場景 | run_wall 的 case |
| --- | --- | --- |
| `AL_IR` | Tests/Almeida/IR | `almeida-ir` |
| `AL_PR` | Tests/Almeida/PR | `almeida-pr` |
| `AL_R` | Tests/Almeida/R | `almeida-r` |
| `TH_T20` | Tests/Thangavelu，noise 20 | 尚無，需新增 |
| `TH_T80` | Tests/Thangavelu，noise 80 | `thangavelu-noise80` |

id 只表達何時、哪個場景、哪台機器。政策與順序在建牆中途可能改變，所以和種子、控制器、`placements_limit` 一樣，一律記在 metadata：

| 資訊 | 記在哪 |
| --- | --- |
| 起始政策與順序、switch、控制器、種子、placements_limit | receipt.json 的 `filter_policy`、`controller`、`seed`、`placements_limit` |
| 每一步實際用的政策與順序 | 有追蹤的牆：trace 每個事件的 `meta.policy`；建牆結束時另由 trace 產生 `derived/steps.csv`，一列一步 |
| 同一定義的重跑 | receipt 的 `spec_sha256`，index 依它分組 |

完整定義在 receipt.json 與 index.csv，兩者不符時以 receipt 為準。

討論中：同一分鐘撞名；pc1／pc2 對應哪台機器；`bricks`、`fft-d2-0`、`rubble` 的代碼；`spec_sha256` 涵蓋的欄位。
