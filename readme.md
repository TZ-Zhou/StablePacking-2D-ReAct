# Stable packing of irregular stones in 2D masonry wall construction
![fft](./figs/build_process.gif)

![rubble_wall](./figs/rubble_wall.gif)


# News
We have another implementation of the algorithm online! The [new implementation](https://github.com/qianqing-wanggg/StablePacking2D/tree/master) is easy to read with a few lines of code!

# 這個 repo 是什麼

這是 Wang 等人（EJOR 2024）〈An image convolution-based method for the irregular stone packing problem〉演算法的 fork，另加一層研究程式（`research/`）：

| 路線 | 誰決定「這一步放哪塊石頭」 | 要不要付費 |
| --- | --- | --- |
| **A. Wang 原版** | Wang 的固定規則：五個濾波器依固定順序一層層篩選候選位置 | 不用 |
| **B. ReAct／LLM** | 每次選石前，控制器從 120 種濾波順序中挑一種，其餘（候選、卷積、穩定性分析）完全不變 | 離線控制器與 mock 不用；真 GPT 要付費 |

兩條路線共用同一個建牆程式 `research/runner/run_wall.py`，差別只在命令列參數。建牆的力學分析需要 **MOSEK 授權**。

# 一、前置條件（每台電腦做一次）

| 項目 | 說明 |
| --- | --- |
| 作業系統 | Windows（建牆流程依賴 Windows 的執行期相容層） |
| 基線環境 | conda 環境 `stablepacking-baseline`，Python 3.10.22，套件版本凍結在 `research/validation/environment-manifest.json`，不可升級任何套件 |
| MOSEK | 套件 11.2.8 已含在基線環境；另需**自己的授權檔** `mosek.lic`（授權綁電腦，不能共用） |
| worker venv | 只有路線 B 用 API 時需要：一個獨立 venv，裝 openai SDK；基底是基線環境 |
| API 金鑰 | 只有付費 GPT 才需要：`OPENAI_API_KEY` |

## 1. 本機設定檔（不進 git，兩台電腦各自一份）

| 檔案 | 用途 |
| --- | --- |
| `.env.local` | 授權路徑與 API 金鑰（範本：`.env.example`） |
| `secrets\` | 放 `mosek.lic` |
| `research\validation\local-setup.json` | 這台電腦的基線環境路徑 |
| `research\validation\worker-venv.local.json` | 這台電腦的 worker venv 登記 |

已追蹤的清單（`environment-manifest.json`、`worker-venv.json`）兩台電腦位元組完全相同，不要改它們；路徑差異全放在上面四個本機檔。所有文字檔以 LF 簽出（`.gitattributes`），程式碼雜湊才會一致。

## 2. 設定步驟（PowerShell，在 repo 根目錄）

```powershell
# (a) 建立本機設定
Copy-Item .env.example .env.local          # 已有就不要覆蓋
# 把 mosek.lic 放進 secrets\，並在 .env.local 填：
#   MOSEKLM_LICENSE_FILE=C:\你的路徑\StablePacking-2D-ReAct\secrets\mosek.lic

# (b) 指定這台電腦的 Python（基線環境）
$PY = "$env:USERPROFILE\miniconda3\envs\stablepacking-baseline\python.exe"

# (c) 每開一個 PowerShell 視窗都要做：載入 .env.local（只印變數名稱，不印值）
. .\research\load_env.ps1
$env:PYTHONUTF8 = '1'                      # 中文版 Windows 預設 cp950，不設會讀不了 git 輸出

# (d) 一次性：寫入這台電腦的環境路徑
& $PY -B research\validation\local_setup.py init

# (e) 驗證環境，必須印出 PASS
& $PY -B research\validation\verify_environment.py
```

`PYTHONNOUSERSITE=1`（已在 `.env.example`）一定要在，否則 `C:\Users\<你>\AppData\Roaming\Python` 的套件會漏進凍結環境，驗證失敗，pip 也可能動到那邊的套件。

## 3. 驗證 MOSEK 授權

```powershell
& $PY -B -c "import mosek; e=mosek.Env(); t=e.Task(); t.appendvars(1); t.putvarbound(0, mosek.boundkey.lo, 1.0, 0.0); t.putcj(0, 1.0); t.putobjsense(mosek.objsense.minimize); t.optimize(); print('MOSEK OK', t.getsolsta(mosek.soltype.bas))"
```

印出 `MOSEK OK solsta.optimal` 才算成功。出現 `err_missing_license_file(1008)` 表示找不到授權：檢查 `MOSEKLM_LICENSE_FILE` 是否已載入，預設搜尋位置是 `C:\Users\<你>\mosek\mosek.lic`。

## 4. worker venv（只有路線 B 的 API 需要）

venv 位置記在 `research/validation/worker-venv.json` 的 `path`（原為 `C:\StonePacking-2D-ReAct\venvs\controller-worker`），套件版本如下，必須完全一致：openai 3.46.0、httpx2 2.14.0、httpcore2 2.14.0、jiter 0.18.0、sniffio 1.3.1、truststore 0.10.5、pip 23.0.1、setuptools 79.0.1。

```powershell
# 用基線環境建 venv（要能看到基線環境的 numpy，所以加 --system-site-packages）
& $PY -m venv --system-site-packages C:\StonePacking-2D-ReAct\venvs\controller-worker
# 以固定版本安裝（--no-deps，避免 pip 另外解相依）
C:\StonePacking-2D-ReAct\venvs\controller-worker\Scripts\python.exe -m pip install --no-deps openai==3.46.0 httpx2==2.14.0 httpcore2==2.14.0 jiter==0.18.0 sniffio==1.3.1 truststore==0.10.5 setuptools==79.0.1
# 登記並檢查（寫入本機登記檔，不改已追蹤的 worker-venv.json）
& $PY -B -c "from research.controller import worker as w; w.register_venv(r'C:\StonePacking-2D-ReAct\venvs\controller-worker'); print(w.venv_problems())"
```

最後一行必須印出 `[]`。如果公司或學校網路有 TLS 攔截而讓 pip 憑證驗證失敗，可改用基線環境的 `pip download` 取得 wheel，再用 `--no-index --find-links` 離線安裝。

# 二、路線 A：Wang 原版

## 命令

`run_wall.py` 的格式是 `<案例>`。每面牆自動取得一個 case_id（開始時間、場景代碼、電腦，例如 `20261108-1254-TH_T80-pc1`），輸出到新資料夾 `research/cases/<case_id>/`；case_id 印在結束時的收據裡。舊牆封存在 `research/runs/`，只讀不寫。`python -B -m research.storage.case_index` 依各收據重建總表 `research/cases/index.csv`。

```powershell
# A1. 單面牆、Wang 的原始行為（--policy code_original）
& $PY -B -m research.runner.run_wall bricks --policy code_original

# A2. 論文的濾波級聯，預設順序 3,0,2,1,4
& $PY -B -m research.runner.run_wall thangavelu-noise80 --policy paper --seed 20

# A3. 指定濾波順序（五個濾波器 0–4 的排列）
& $PY -B -m research.runner.run_wall thangavelu-noise80 --policy paper --order 2,4,3,1,0 --seed 20

# A4. 先試跑：只放 3 塊就停（確認環境、授權、流程都通）
& $PY -B -m research.runner.run_wall bricks --policy code_original --placements 3
```

`code_original` 是重現 Wang 程式碼的濾波順序（含它的實作細節）；`paper` 是論文描述的順序，各濾波器只套用一次。兩者差別見 `Stonepacker2D/stack/selection.py`。

選項：`--parent 牆`（這面牆分支自哪面牆：case_id 或 `research/runs` 內的路徑）、`--seed S`（預設 20）、`--timeout 秒`（預設 1800）、`--no-trace`（不寫候選追蹤，只留輸出）、`--placements N`（放 N 塊後停）、`--tol`（數值容差，預設 1e-9）、`--fresh-pool`（對照用：像原始程式一樣每批重開 worker；預設整面牆共用一組 worker）、`--workers N`（worker 數，預設用案例自己的數字；不影響牆的結果）。

在終端機執行時，每面牆會顯示進度條：已放石頭數／給定石頭數（牆放不完所有石頭時，結束時不會到 100%）。

## 案例

| 案例名稱 | 對應論文 | 每批牆數 | 備註 |
| --- | --- | --- | --- |
| `bricks` | Figure B.34 | 1 | 固定尺寸的磚；所有磚相同，控制器不會有決策點 |
| `almeida-r`、`almeida-pr`、`almeida-ir` | Figure 18、21、B.30–B.32 | 1 | 真實石頭，從不規則到規則；`almeida-pr` 本機約 23 分鐘 |
| `fft-d2-0` | Figures 26–27 | 5 | 傅立葉形狀產生的石頭 |
| `rubble` | Figure B.33 | 20 | 92 個真實石頭 |
| `thangavelu-noise80` | Figures 22–25 | 20 | 多邊形、雜訊 80；研究層的主要案例 |
| `thangavelu-noise20` | 同一組資料的雜訊 20 | 20 | 與 noise80 共用建牆程式，只換石頭影像 |

`run_wall` 一次只建批次中的第一面牆。實驗要建多面牆請用 `campaign`（見第四節），或用 `batch` 抽樣多種場景與濾波順序（見 [docs/specs/batch-runner.md](docs/specs/batch-runner.md)）。

## 看結果

`research/cases/<case_id>/` 內容：

| 項目 | 說明 |
| --- | --- |
| `receipt.json` | 整次執行的收據：狀態、種子、git 版本、程式雜湊、耗時、放置數、追蹤檢查 |
| `result_<時間>/` | 牆本身的資料：`decisions.txt`、`filter.txt`、`transformation.txt`、`mechanism.csv`（左右極限載重，牆完成才有）、`base.pkl`（牆完成時寫一次；有 `--placements` 時每放一顆更新）、`trace/` |
| 暫存的案例腳本、`execution.log`、`events.jsonl` | 實際執行的腳本與過程紀錄；`environment-guard.log` 只在環境檢查失敗時才有 |
| `derived/steps.csv` | 有追蹤的牆才有，每次抽樣一列：動作、實際用的政策與順序（`filter_order`）、入選候選的五個原始分數（`01_LLC`～`05_LFR`）、放完後的 `F_SF`、`F_AH`、`F_AV`、`F_LR`、耗時（`wall_time_s` 為該次，`wall_time_cum_s` 為從建牆開始累計） |

`steps.csv` 的四項指標是當下未完成牆在模擬格點上的值，與 `research/quality` 的正式指標（原解析度）不能直接比較；牆還太小時 `F_AH`、`F_AV` 為空。計算 `F_LR` 需要 MOSEK，要先載入 `.env.local`。

新的 case 只存資料：不存圖片與 PDF，輸入資料不留複本（在 `Tests/`）。`research/runs` 的舊牆維持原樣，讀取程式兩種都支援。

`receipt.json` 的 `status`：

| 狀態 | 意思 |
| --- | --- |
| `PROCESS_PASS` | 完整建完，追蹤檢查通過 |
| `STOPPED_AFTER_PLACEMENTS` | 用 `--placements N` 到數後被刻意停止（`returncode: 1` 是預期的） |
| `INCOMPLETE_TRACE` | 牆建完或停止了，但追蹤驗證不完整（例如沒有決策點，見第三節說明） |
| `TIMEOUT` | 超過 `--timeout`，已記錄 |
| `FAILED`、`TERMINATION_FAILED`、`LAUNCHER_ERROR` | 失敗；看 `execution.log` |

## 檢查與比對

```powershell
# 唯讀檢查一個 run（參數是含 receipt.json 的 run 資料夾，不是 result_<時間>）：
# 追蹤重放、提交與候選一致、基底矩陣與位姿一致
& $PY -B -m research.walls.check_run research\cases\<case_id>

# 逐位元組比較兩個 run（例如換電腦重跑同一個種子）
& $PY -B -m research.walls.check_run research\cases\<case_id> --compare research\cases\<另一個 case_id>
```

`time.txt` 記錄耗時、PDF 內嵌建立日期，這兩者每次都不同，是預期的差異。

# 三、路線 B：ReAct 控制器

控制器在每次選石前，從 120 種濾波順序挑一種；它只能挑「某個濾波順序會選到的結果」，不能改候選、卷積、LLC > 0 的篩選與穩定性分析。以下按**由免費到付費**的順序做。每一步通過再進下一步。

## B0. 離線控制器（免費，不需金鑰）

用 `--controller` 傳 JSON，必須搭配 `--policy paper` 且保持追蹤開啟。控制器種類：

| JSON | 說明 |
| --- | --- |
| `{"kind": "null"}` | 永遠採用論文順序；結果必須與 Paper 牆逐位元組相同 |
| `{"kind": "random", "seed": 7}` | 隨機挑順序 |
| `{"kind": "rule"}` | 規則式 |
| `{"kind": "react", "llm": {"kind": "mock", "mode": "max_llc"}}` | ReAct 流程，但 LLM 是假的（`max_llc`、`blind`、`invalid` 三種模式） |

PowerShell 5.1 會吃掉傳給外部程式的雙引號，所以 JSON 內的 `"` 要寫成 `\"`：

```powershell
# Null 控制器（約 10 個放置；Thangavelu 有真正的決策點）
& $PY -B -m research.runner.run_wall thangavelu-noise80 --policy paper --seed 20 --controller '{\"kind\": \"null\"}' --placements 10

# 假 LLM 的 ReAct，控制器在獨立 worker venv 的子行程中執行
& $PY -B -m research.runner.run_wall thangavelu-noise80 --policy paper --seed 20 --controller '{\"kind\": \"react\", \"llm\": {\"kind\": \"mock\", \"mode\": \"max_llc\"}}' --worker-venv --placements 10
```

實測（本機，MOSEK 授權就位）：Thangavelu 放 10 塊，Null 控制器約 9 分鐘、mock LLM 的 ReAct（經 worker venv）約 11 分鐘，兩者都是 7 個決策點、檢查 `verdict: PASS`，ReAct 的傳輸帳本 `ledger: ok`。控制器牆比 Wang 原版慢。

**決策點**：只有當不同濾波順序會選到不同候選時才有決策點。`bricks` 的磚全部相同，所以控制器從不被呼叫，收據會是 `INCOMPLETE_TRACE`、`verdict: PARTIAL`、`decision_points: 0`；這不是錯誤，要測 ReAct 請用 `thangavelu-noise80`。

重放驗證控制器紀錄（每筆紀錄在它自己事件的追蹤上重新執行並比對）：

```powershell
& $PY -B -m research.controller.order_controller check research\cases\<case_id>
```

## B1. 開發牆（免費）

在 GPT 牆之前，以 2 個開發種子建 Paper 牆與各控制器牆，並檢查 Null 牆等於 Paper 牆：

```powershell
& $PY -B -m research.runner.dev_walls init my-dev     # 寫 research/studies/my-dev/study.json（只寫一次）
& $PY -B -m research.runner.dev_walls run my-dev      # 建尚未建的牆，然後檢查（要建很久）
& $PY -B -m research.runner.dev_walls check my-dev    # 檢查每面牆，寫 manifest.csv
& $PY -B -m research.walls.campaign_quality my-dev --out research\quality\my-dev
```

現成的一份在封存的 `research/runs/campaigns/dev-2seeds`（同事 zip 帶來），通常不必重跑；`dev_walls check`（不加 id）檢查它，只讀不寫。

## B2. 真實 GPT 呼叫（**付費**）

### 付費前的檢查表（全部成立才會送出請求）

| 條件 | 做法 |
| --- | --- |
| `OPENAI_API_KEY` | 填在 `.env.local`；只有 worker 子行程讀得到，Wang 的行程會丟掉它 |
| 付費閘門 | 當次視窗設 `$env:STONEPACKING_ALLOW_PAID_API = '1'`，指令加 `--confirm-paid`；用完請關閉該視窗或移除變數 |
| `OPENAI_BASE_URL` | 必須**留空**（只給本機模擬伺服器用，付費時會被拒絕） |
| git 工作樹乾淨 | 所有修改都已 commit，且程式雜湊未變 |
| 價格新鮮 | `research/llm_walls/gpt_wall.py` 的 `PRICE_CHECKED` 距今不超過 7 天，目前是 2026-11-07，**2026-11-14 後要先重新核對價格再改這個日期** |
| worker venv | 已登記且 `venv_problems()` 為 `[]`（第一節第 4 項） |
| 一次一面牆 | 一次只能有一個；名稱只能用一次；沒有 `finished` 的牆會擋住後面所有執行，要人看過才解 |
| 預算上限 | 每面牆最多 60 次請求、300,000 token、US$0.60、2,400 秒；整個實驗最多 2 面牆、120 次請求、600,000 token、US$1.20 |

目前帳本（`gpt_wall ledger`）：已完成 1 面（`gpt-001`），共 37 次請求、145,293 token、約 US$0.14，所以**還剩 1 面牆、約 US$1.06**，下一面牆用新名字（例如 `gpt-002`）並用 `--after gpt-001`。

### 指令

```powershell
# 1. 先預檢（不送任何請求）
& $PY -B -m research.runner.gpt_wall preflight --name gpt-002 --seed <種子> --after gpt-001

# 2. 真的送出（付費）
$env:STONEPACKING_ALLOW_PAID_API = '1'
& $PY -B -m research.runner.gpt_wall run --name gpt-002 --seed <種子> --after gpt-001 --confirm-paid
Remove-Item Env:STONEPACKING_ALLOW_PAID_API

# 3. 看花費
& $PY -B -m research.runner.gpt_wall ledger

# 4. 離線報告（唯讀，不呼叫 API）：寫到 research/reports/<名稱>/
& $PY -B -m research.llm_walls.gpt_wall_report gpt-002
```

一面 GPT 牆是 Thangavelu noise80 的完整一面牆，每個決策點最多 3 次工具呼叫，最多 20 個決策。如果任何決策點沒有由 LLM 決定（略過或退回預設），這面牆記為「GPT incomplete」，不納入品質比較。

少量決策的煙霧測試（只重放既有牆的前幾個決策點，不建新牆）：

```powershell
& $PY -B -m research.llm_walls.gpt_smoke preflight --name smoke-002          # 只檢查
& $PY -B -m research.llm_walls.gpt_smoke run --name smoke-002 --confirm-paid  # 付費
& $PY -B -m research.llm_walls.gpt_smoke check --name smoke-002              # 離線檢查
```

本機模擬伺服器（免費，驗證流程而不花錢）：`gpt_wall run --name lb-001 --seed S --loopback [--placements N]`，需要 `OPENAI_BASE_URL` 指向 loopback 位址、金鑰以 `sk-loopback-` 開頭（絕不能用真的金鑰）。

> 注意：本 README 的付費步驟在這台電腦**尚未實際執行過**（尚未送出任何付費請求）；`gpt-001` 是同事電腦上完成的。

# 四、批次實驗（多面牆、多個種子）

預先登記「濾波順序 × 種子」，一次建一面牆：

```powershell
# 登記（只寫一次，不能覆蓋）：arms 可為 code_original、paper=順序、或 random
& $PY -B -m research.runner.campaign init my-camp --case thangavelu-noise80 --arms code_original paper=3,0,2,1,4 alt=2,4,3,1,0 random --n-seeds 3 --timeout 1200

# 先看會跑什麼
& $PY -B -m research.runner.campaign run my-camp --dry-run

# 真的跑；--max-runs 可限制這次最多建幾面
& $PY -B -m research.runner.campaign run my-camp --max-runs 2

# 重新彙整 manifest.csv 與 steps.csv
& $PY -B -m research.runner.campaign summarize my-camp

# 四項牆品質指標（Wang 的 F_SF、F_AH、F_AV、F_LR），寫到 research\quality\<id>
& $PY -B -m research.walls.campaign_quality my-camp --out research\quality\my-camp
```

規則：程式碼改了就拒絕繼續（保證各組只差在濾波順序）；已存在的 run 不會重跑或覆蓋；`FAILED` 會停下整個 campaign，`TIMEOUT` 與 `INCOMPLETE_TRACE` 只記錄後繼續。一面 Thangavelu 牆通常 5–20 分鐘，所以一個 campaign 可能要跑很多小時，建議分次用 `--max-runs`。

新 campaign 的設計、case 清單與彙整表在 `research/studies/<id>/`，每面牆在 `research/cases/<case_id>/`。已存在的 campaign（來自同事的 zip）封存在 `research/runs/campaigns/`，例如 `thangavelu-5arms-10seeds`（5 組 × 10 個種子），只讀不寫；可用 `--extends` 接續的前提是執行程式碼未變。

# 五、離線分析（不建牆，只讀已記錄的追蹤；不需要 MOSEK）

放在 `research/analysis/`，每支檔案開頭有說明，例如：

```powershell
# 在已追蹤的候選分數矩陣上重放 120 種濾波順序
& $PY -B -m research.analysis.order_sensitivity <追蹤資料夾> <新的輸出資料夾>
```

其餘（`order_agreement`、`counterfactual_branches`、`decision_signals`、`unplaced_stones` 等）見 `research/analysis/README.md`。

# 六、測試

```powershell
$env:PYTHONUTF8 = '1'
& $PY -B -m unittest discover -s research\tests
```

約 20 分鐘（有授權時會真的建牆）。預期：全部通過，只略過 15 項「openai SDK 只裝在隔離的 venv」。測試不會呼叫付費 API：`OPENAI_API_KEY` 有設也一樣。為保險，跑測試的視窗不要載入 `.env.local`。

# 七、常見問題

| 症狀 | 原因與處理 |
| --- | --- |
| `err_missing_license_file(1008)` | MOSEK 授權找不到。載入 `.env.local`，或把 `mosek.lic` 放到 `C:\Users\<你>\mosek\` |
| `verify_environment` 因多出套件而失敗 | 使用者 site-packages 漏進來。設 `PYTHONNOUSERSITE=1` |
| 測試出現 `'NoneType' object has no attribute 'strip'` | 中文版 Windows 的 cp950 讀不了 git 輸出。設 `PYTHONUTF8=1` |
| `assert sys.prefix == ENV_PREFIX` | 用錯 Python。要用基線環境的 `python.exe`，並跑過 `local_setup.py init` |
| 報告表格逐位元組比對失敗，差在 `\r\n` | 工作樹是 CRLF。`.gitattributes` 已規定 LF；對已簽出的檔案重新簽出一次 |
| `conda` 或 `pip` 出現 `CERTIFICATE_VERIFY_FAILED` | 網路上有 TLS 攔截（如防毒軟體）。conda 用 `CONDA_SSL_VERIFY=truststore`；pip 用基線環境自己的 pip |
| Python 卡死不動 | 有時是環境變數 `SSLKEYLOGFILE` 指向失效檔案；自己寫呼叫網路的 Python 時，指令前加 `SSLKEYLOGFILE=` |
| `INCOMPLETE_TRACE` | 該牆沒有決策點（例如 `bricks`）或沒有傳輸帳本；見第三節 |

# 八、兩台電腦要得到一樣的結果

- **同一份已追蹤內容**：程式碼、`environment-manifest.json`、`worker-venv.json`、案例資料。
- **各自一份本機檔**：`.env.local`、`secrets\`、`local-setup.json`、`worker-venv.local.json`。
- **驗證**：兩台各自 `verify_environment.py` 通過；同一案例、同一種子、同一政策各跑一次，用 `check_run --compare` 比對，輸出應逐位元組相同（`time.txt` 與 PDF 的建立日期除外）。

# 九、目錄

| 項目 | 說明 |
| --- | --- |
| `Stonepacker2D/` | 堆石演算法函式庫；見該處 README |
| `Tests/` | 論文範例案例與資料；見該處 README |
| `research/` | 本 fork 的研究層：追蹤濾器順序建牆、離線分析、ReAct 控制器；見該處 README |
| `docs/` | 計畫、規格、決策、觀念與查證文件；分類與存放規則見該處 README |
| `figs/` | 本 readme 用的動畫 |
| `pyproject.toml`、`setup.py` | 套件與相依設定、ruff／black／pytest 設定 |
| `LICENSE` | MIT 授權 |

# 附：原作者的範例用法

原作者的方式是進入範例資料夾執行 `python run.py`，結果寫在該資料夾的 `result` 子資料夾。這個 fork 在 Windows 上建牆請用上面的 `research.walls.run_wall`，它處理了主程式保護、輸出隔離與追蹤；下面的對照表僅供參考。

| Name | Corresponding figures in paper | Description |
| --- | --- | --- |
| Almeida | Figure 18, Figure 21, Figure B.30-B.32 | real stone shapes from irregular to regular |
| Bricks | Figure B.34 | rectangular shapes of fixed sizes |
| FFT-stones | Figures 26-27 | Stones shapes generated from a Fourier-based shape generator |
| Rubble stone walls | Figures B.33 | 92 real stone shapes |
| Thangavelu | Figures 22-25 | polygonal stones of different noise levels |
