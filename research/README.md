# research：以追蹤式濾波順序政策跑 Wang 的案例、離線分析濾波順序，並以 ReAct／LLM 控制器挑選順序

整個資料夾是 `research` 套件，從 repo 根目錄以 `python -B -m research.<資料夾>.<模組>` 執行。

## 內容
| 項目 | 說明 |
| --- | --- |
| `paths.py` | repo、`runs/`、`validation/`、`quality/` 的路徑，全套件只在這裡定義 |
| `walls/` | 建牆、檢查、批次實驗與品質評分（見下表） |
| `controller/` | 濾波順序控制器與 LLM（見下表） |
| `analysis/` | 離線分析與 P0–P2 探索研究；見該處 README |
| `llm_walls/` | LLM 實牆與付費 API（見下表） |
| `findings.md` | 研究結論 |
| `quality/` | 各品質方法一個子資料夾，放批次彙總 json 與品質 csv、jsonl 輸出 |
| `tests/` | 測試（unittest 寫法，pytest 也可跑） |
| `validation/` | 登記的環境清單、環境驗證、Windows 執行期相容層與 worker venv 登記 |

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

## 使用
- 從 repo 根目錄以模組執行，例如 `python -B -m research.walls.run_wall ...`；各檔開頭的 docstring 有子指令與參數。
- 付費 API 呼叫需設 `STONEPACKING_ALLOW_PAID_API=1` 並加 `--confirm-paid`；先用 `preflight` 檢查。
- 測試：`python -m unittest discover -s research/tests`（或 `python -m pytest research/tests`）。
- `runs/` 是未追蹤的輸出資料夾。
