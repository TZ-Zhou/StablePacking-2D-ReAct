# research：以追蹤式濾波順序政策跑 Wang 的案例、離線分析濾波順序，並以 ReAct／LLM 控制器挑選順序

## 內容
| 項目 | 說明 |
| --- | --- |
| `run_case.py` | 以指定濾波政策與候選追蹤跑一個原始測試案例 |
| `check_case.py` | 唯讀檢查追蹤過的 run，並可逐位元組比對兩個 run |
| `campaign.py` | 預先登記濾波順序 x 種子的 campaign，一次建一面牆並彙整 |
| `metrics.py` | 單一 run 的牆品質與過程指標，以及決策軌跡的步驟列 |
| `quality_batch.py` | 對 campaign 的每個 run 批次計算 Wang 的四項牆品質指標 |
| `sensitivity.py` | 在已追蹤的候選分數矩陣上離線重放 120 種濾波順序 |
| `order_agreement.py` | 離線比較哪些濾波順序在同一牆況下選同一塊石頭 |
| `arm_comparison.py` | 已彙整 campaign 的 arm x 種子探索性比較（唯讀） |
| `failure_analysis.py` | 離線分析牆為何以石頭未放置收尾 |
| `counterfactual.py` | 對已記錄牆的單步濾波順序分支做離線選項與建牆比較 |
| `lookahead_study.py` | 在反事實分支點做有限前瞻的探索性研究 |
| `decision_signals.py` | 在已記錄牆上離線計算兩種簡易決策訊號 |
| `preview.py` | 觀測版本 3 的 A 層局部品質預覽與 Greedy-Preview 規則 |
| `preview_study.py` | 在已記錄牆上離線研究上述預覽 |
| `controllers.py` | 論文政策的動態濾波順序控制器（Null、Random、Rule、ReAct 等） |
| `controller_worker.py` | 控制器政策所在的獨立子行程，以 JSON 位元組溝通 |
| `llm_adapter.py` | ReAct 控制器的 LLM API 轉接層，預設不啟用付費呼叫 |
| `api_smoke.py` | 在既有牆的決策軌跡上做少量真實 API 決策的煙霧測試 |
| `gpt_wall.py` | 以 OpenAI 模型選濾波順序跑完整面牆的安全執行器與花費帳本 |
| `dev_walls.py` | 在 GPT 牆之前，以開發種子建 Paper 牆與各控制器牆並檢查 |
| `findings.md` | 研究結論 |
| `quality/` | 各品質方法一個子資料夾，放批次彙總 json 與品質 csv、jsonl 輸出 |
| `tests/` | pytest 測試 |
| `validation/` | 登記的環境清單、環境驗證、Windows 執行期相容層與 worker venv 登記 |

## 使用
- 腳本從 repo 根目錄執行：`python -B research/<script>.py <子指令與參數>`，各檔開頭的 docstring 有子指令與參數。
- 付費 API 呼叫需設 `STONEPACKING_ALLOW_PAID_API=1` 並加 `--confirm-paid`；先用 `preflight` 檢查。
- 測試：`python -m pytest research/tests`。
- `runs/` 是未追蹤的輸出資料夾。
