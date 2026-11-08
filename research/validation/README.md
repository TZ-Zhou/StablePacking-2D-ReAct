# validation：登記的環境清單、環境驗證、案例執行期相容層，以及每台電腦自己的本機路徑設定

## 內容
| 項目 | 說明 |
| --- | --- |
| `environment-manifest.json` | 凍結的基線環境清單（套件版本、conda 記錄）；內含登記時那台電腦的 `prefix` |
| `worker-venv.json` | 登記的 controller worker venv（Python 與 SDK 版本、檔案雜湊）；內含登記時那台電腦的路徑 |
| `verify_environment.py` | 實驗前檢查目前環境與清單完全一致 |
| `windows_case_runtime.py` | 案例行程內的 Windows 相容與觀測層：略過繪圖輸出、換掉 `build_wall` 的 Pool、把進度事件寫進 `events.jsonl`；建牆時複製進案例資料夾再匯入，所以不能 import `research` 套件 |
| `local_setup.py` | 讀取本機覆寫：讓每台電腦用自己的環境路徑，不必改上面兩個已追蹤的清單 |
| `local-setup.example.json` | 本機覆寫檔的範本 |

## 使用
- 環境驗證：用基線環境的 Python 執行 `python -B research/validation/verify_environment.py`（無參數），全部相符印 PASS，任一項不符以 AssertionError 結束；`research/runner/run_wall.py` 每次開跑前會自動執行，不符就停。
- 兩台電腦共用同一份已追蹤的清單，位元組一致。run 記下的 worker venv `manifest_sha256` 取自這台電腦實際使用的登記檔，有本機檔時各機不同；跨電腦比對 worker venv 用 `files_sha256`。
- 本機路徑放在不進 git 的兩個本機檔（名稱 local-setup、worker-venv.local，副檔名 .json，在本資料夾內產生）；建立方式：
  - 在基線環境執行 `python research/validation/local_setup.py init`。
  - `init` 之後，用該電腦的 worker venv 根目錄呼叫 `research.controller.worker.register_venv(根目錄)`，它會寫到本機登記檔，不改已追蹤的 `worker-venv.json`（本機檔已存在時不會覆蓋）。
- local-setup 有兩個鍵：`env_prefix`（這台電腦基線環境的 prefix）、`worker_venv_manifest`（本資料夾內本機登記檔的檔名）。沒有此檔、缺鍵或值為空時，採用清單內登記的值。
