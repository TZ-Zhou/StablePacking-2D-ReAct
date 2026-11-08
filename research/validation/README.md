# validation：登記的環境清單、環境驗證，以及每台電腦自己的本機路徑設定

## 內容
| 項目 | 說明 |
| --- | --- |
| `environment-manifest.json` | 凍結的基線環境清單（套件版本、conda 記錄）；內含登記時那台電腦的 `prefix` |
| `worker-venv.json` | 登記的 controller worker venv（套件版本與檔案雜湊）；內含登記時那台電腦的路徑 |
| `verify_environment.py` | 實驗前檢查目前環境與清單完全一致 |
| `windows_case_runtime.py` | 原始案例在 Windows 上的執行期相容層 |
| `local_setup.py` | 讀取本機覆寫：讓每台電腦用自己的環境路徑，不必改上面兩個已追蹤的清單 |
| `local-setup.example.json` | 本機覆寫檔的範本 |

## 使用
- 兩台電腦共用同一份已追蹤的清單，所以清單位元組一致，紀錄在 run 裡的雜湊也一致。
- 本機路徑放在不進 git 的兩個本機檔（名稱 local-setup、worker-venv.local，副檔名 .json，在本資料夾內產生）；建立方式：
  - 在基線環境執行 `python research/validation/local_setup.py init`。
  - `init` 之後，用該電腦的 worker venv 根目錄呼叫 `research.controller.worker.register_venv(根目錄)`，它會寫到本機登記檔，不改已追蹤的 `worker-venv.json`（本機檔已存在時不會覆蓋）。
- 沒有 local-setup.json 時，直接採用清單內登記的路徑。
