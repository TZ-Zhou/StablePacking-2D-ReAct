# reports：已提交的離線報告，每個子資料夾一份

## 內容
| 項目 | 說明 |
| --- | --- |
| `gpt-001/` | 第一面即時 GPT 牆（gpt-001）的離線分析：執行紀錄、各決策點的候選比較，以及與同種子開發牆的品質對照。 |
| `gpt-001-cf-steps/` | gpt-001 反事實牆（GPT 與 Paper 順序混合重放）逐放置步驟的品質指標與成對差異分析。 |

兩份報告各自的說明在其子資料夾的 `README.md`。

## 使用
- `gpt-001/` 的資料表由 `python -B -m research.llm_walls.gpt_wall_report gpt-001` 產生。
- `gpt-001-cf-steps/` 由 `python -B -m research.llm_walls.step_quality` 產生。
- 兩者都只讀既有紀錄，不建牆、不呼叫 API；輸出資料夾已存在時拒絕覆寫。
