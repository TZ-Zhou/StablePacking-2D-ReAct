# llm_walls：用 GPT 與反事實分支建 LLM 實牆、檢查並離線出報告

本資料夾放 LLM 控制器實牆的定義與檢查：API 煙霧測試、完整 GPT 牆（含花費帳本與安全閘）、開發牆、反事實分支，以及這些牆的離線報告。建牆的啟動點在 `research/runner/`。

## 內容
| 項目 | 說明 |
| --- | --- |
| `gpt_smoke.py` | 在既有牆的決策軌跡上做少量真實 GPT 決策的煙霧測試；僅 `run` 呼叫付費 API，`preflight`、`check` 不呼叫 |
| `gpt_wall.py` | 以 OpenAI 模型選濾波順序的完整牆：規格、限額、預檢與花費帳本；僅 `research.runner.gpt_wall run`（不含 `--loopback`）呼叫付費 API，`preflight`、`ledger`、`run --loopback`（本機模擬伺服器）不呼叫 |
| `dev_walls.py` | 開發種子上的 Paper 牆與各控制器牆（含假供應商 ReAct）的定義與檢查，供 GPT 牆對照；不呼叫付費 API |
| `gpt_wall_report.py` | 唯讀產生一面 GPT 實牆的離線報告（決策、切換、軌跡、品質），寫到 `research/reports/<名稱>/`；不呼叫付費 API |
| `gpt_counterfactual.py` | gpt-001 反事實分支的定義與檢查：以記錄的順序重建，不呼叫任何 API；新牆是 study `gpt-001-cf` 的 case |
| `step_quality.py` | 由封存紀錄離線算反事實牆的逐步品質，寫到 `research/reports/gpt-001-cf-steps/`；不呼叫付費 API |

新牆是 `research/cases/` 底下的 case；各模組開頭的 docstring 是完整的保證與限制說明。

## 使用
指令在 repo 根目錄執行，建牆用 `python -B -m research.runner.<模組> ...`，其餘用 `python -B -m research.llm_walls.<模組> ...`。付費的 `run` 需要先在 PowerShell 載入授權與金鑰（`. .\research\load_env.ps1`，只印名稱不印值），並設 `STONEPACKING_ALLOW_PAID_API=1`、加 `--confirm-paid`：

```
python -B -m research.llm_walls.gpt_smoke preflight --name smoke-001
python -B -m research.llm_walls.gpt_smoke check --name smoke-001
python -B -m research.runner.gpt_wall preflight --name gpt-002 --seed S [--after NAME]
python -B -m research.runner.gpt_wall run --name gpt-002 --seed S --after gpt-001 --confirm-paid   # 付費
python -B -m research.runner.gpt_wall run --name lb-001 --seed S --loopback                        # 本機模擬，不付費
python -B -m research.runner.gpt_wall ledger
python -B -m research.runner.dev_walls init|run|check <id>
python -B -m research.llm_walls.gpt_wall_report gpt-001
python -B -m research.runner.gpt_counterfactual init|run|check|summarize
python -B -m research.llm_walls.step_quality
```
