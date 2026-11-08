# walls：牆的定義與檢查

這裡定義一面牆與一批牆是什麼，以及建好之後怎麼檢查。建牆的啟動與執行都在 `research/runner/`。

| 檔案 | 說明 |
| --- | --- |
| `case_definition.py` | 場景表、入口腳本的暫存改寫、種子與 worker 數的替換、程式指紋、不含 API 金鑰的環境 |
| `batch_plan.py` | 批次的設定檔檢查、拉丁超立方抽樣、案例清單與每個案例的 `run_wall` 參數。用法見 `docs/specs/batch-runner.md` |
| `batch_example.yaml` | 批次設定檔範例，逐一列出 `"random"` 與全部 120 種濾波順序 |
| `campaign.py` | 單一場景的預先登記實驗：arm、種子、登記在 `research/studies`、彙整表 |
| `campaign_quality.py` | campaign 每面牆的四項品質指標 |
| `check_run.py` | 唯讀檢查一面牆的追蹤，逐位元組比對兩面牆 |
| `wall_metrics.py` | 單面牆的品質與過程指標、每一步的紀錄 |
