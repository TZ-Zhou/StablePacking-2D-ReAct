# research：以追蹤式濾波順序政策跑 Wang 的案例、離線分析濾波順序，並以 ReAct／LLM 控制器挑選順序

整個資料夾是 `research` 套件，從 repo 根目錄以 `python -B -m research.<資料夾>.<模組>` 執行。

## 內容
| 項目 | 說明 |
| --- | --- |
| `paths.py` | repo、`runs`（封存）、`cases`、`studies`、`validation/`、`quality/` 的路徑，全套件只在這裡定義 |
| `walls/` | 建牆、檢查、批次實驗與品質評分（見下表） |
| `controller/` | 濾波順序控制器與 LLM（見下表） |
| `analysis/` | 離線分析與 P0–P2 探索研究；見該處 README |
| `llm_walls/` | LLM 實牆與付費 API（見下表） |
| `perf/` | 單次案例的階段耗時與可選的持久 Pool；見該處 README |
| `storage/` | 新牆的 case_id、索引與 study 紀錄，以及封存的 `runs`（見下表） |
| `load_env.ps1` | 把 repo 根目錄 `.env.local`（授權與 API 金鑰，已被 git 忽略）載入目前 PowerShell 視窗，只印名稱不印值 |
| `findings.md` | 研究結論 |
| `quality/` | 各品質方法一個子資料夾，放批次彙總 json 與品質 csv、jsonl 輸出 |
| `reports/` | 已提交的離線報告，每面 GPT 實牆一個子資料夾（目前 `gpt-001`） |
| `tests/` | 測試（unittest 寫法，pytest 也可跑） |
| `validation/` | 登記的環境清單、環境驗證、Windows 執行期相容層與 worker venv 登記，以及每台電腦自己的本機路徑設定（`local_setup.py`，詳見該處 README） |

### walls/
| 檔案 | 說明 |
| --- | --- |
| `run_wall.py` | 以指定濾波政策與候選追蹤建牆（跑一個原始測試案例） |
| `check_run.py` | 唯讀檢查追蹤過的 run，並可逐位元組比對兩個 run |
| `campaign.py` | 預先登記濾波順序 × 種子的 campaign，一次建一面牆並彙整 |
| `wall_metrics.py` | 單一 run 的牆品質與過程指標，以及決策軌跡的步驟列 |
| `campaign_quality.py` | 對 campaign 的每個 run 批次計算 Wang 的四項牆品質指標 |

### controller/
| 檔案 | 說明 |
| --- | --- |
| `order_controller.py` | 每次選石前決定濾波順序的控制器（Null、Random、Rule、ReAct 等）與其紀錄檢查 |
| `worker.py` | 控制器政策所在的獨立子行程，以 JSON 位元組溝通 |
| `llm_api.py` | ReAct 控制器的 LLM API 轉接層，預設不啟用付費呼叫 |

### llm_walls/
| 檔案 | 說明 |
| --- | --- |
| `gpt_smoke.py` | 在既有牆的決策軌跡上做少量真實 GPT 決策的煙霧測試 |
| `gpt_wall.py` | 以 OpenAI 模型選濾波順序跑完整面牆的安全執行器與花費帳本 |
| `dev_walls.py` | 在 GPT 牆之前，以開發種子建 Paper 牆與各控制器牆並檢查 |
| `gpt_wall_report.py` | 唯讀產生一面 GPT 實牆的離線報告（決策、切換、軌跡、品質），寫到 `reports/<名稱>/` |
| `gpt_counterfactual.py` | gpt-001 的反事實分支：以記錄的順序重建，不呼叫 API；新牆是 study `gpt-001-cf` 的 case |
| `step_quality.py` | 由封存紀錄離線算反事實牆的逐步品質，寫到 `reports/gpt-001-cf-steps/` |

### storage/
| 檔案 | 說明 |
| --- | --- |
| `case_id.py` | case_id 的產生、解析與檢查，以及定義雜湊 `spec_sha256`；場景代碼表與電腦代碼表只在這裡 |
| `case_index.py` | 由各 receipt 重建 cases 資料夾的 index.csv |
| `legacy_runs.py` | 封存的 runs 資料夾：產生 legacy-index.csv、設唯讀 |
| `studies.py` | 讀寫各 study 的 study.json（一組事先設計的 case） |

## 使用
- 從 repo 根目錄以模組執行，例如 `python -B -m research.walls.run_wall ...`；各檔開頭的 docstring 有子指令與參數。
- 付費 API 呼叫需設 `STONEPACKING_ALLOW_PAID_API=1` 並加 `--confirm-paid`；先用 `preflight` 檢查。
- 測試：`python -m unittest discover -s research/tests`（或 `python -m pytest research/tests`）。
- 新牆在 `cases`（未追蹤，一 case 一資料夾），一組牆的設計與 case 清單在 `studies`；`runs` 是唯讀封存，說明在該處的 ARCHIVE.md。
