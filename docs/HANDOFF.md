# 交接：StablePacking-2D-ReAct

給接手的另一台電腦（人或 agent）。內容是 2026-11-08 21:15 在來源電腦 `GIAAIL-STRIX`（`pc1`）的狀態。未完成項目的完整清單在 [`docs/TODO.md`](TODO.md)，這份只說怎麼接手、下一步做什麼，以及只存在來源電腦的資訊。

## 現況

- 所有程式碼與本檔都在 `main`，已 push，可以直接從 GitHub 拿。
- 新分析模組只過單元測試，還沒在封存牆上跑完過。下一步就是驗證它們（見「下一步」）。
- `main` 上的砌牆加速（階段 3–6 與 `Stonepacker2D` 改動）只驗過前段短牆，完整牆比較與全套測試都沒跑。比較出差異時從 `6886b87` 查起。

## 搬到另一台電腦

### 來源電腦要先做

1. 複製 `research/runs/`（1.7 GB，不進 git，含 `legacy-index.csv`）到目標電腦同一個相對路徑。新分析全部讀這裡，沒有它無法驗證。
2. 想做表格逐欄比對時，複製參考表（見「只存在來源電腦的東西」）。不複製也能驗證，只是少一層比對。
3. 要接續 case-v1 時，一併 push `feat/case-output`，或用 `git bundle` 帶過去；這條分支只在本機。

### 目標電腦要做

1. clone 後留在 `main`。
2. 照 `readme.md`「一、前置條件」建環境：conda 環境 `stablepacking-baseline`（Python 3.10.22，版本凍結，不可升級任何套件）、`.env.local`、`secrets\mosek.lic`，跑到驗證環境印出 PASS。
3. MOSEK 授權綁電腦，要用目標電腦自己的授權。`.env.local` 與 `secrets\` 不經 git 傳，也不要貼進任何文件。
4. 要在目標電腦建牆時，先把它的 hostname 加進 `research/storage/case_id.py` 的 `PCS`（代碼 `pc2`），否則會被拒絕。離線分析不需要這步。

## 分支與 worktree

| 分支 | commit | 狀態 |
| --- | --- | --- |
| `main` | 最新 | 目前工作分支，已 push |
| `pref/case-speedup` | `6a573a9` | 已 push，內容都在 `main`，只少交接文件 |
| `feat/case-output` | `d55755f` | case-v1 六檔輸出，未併入 `main`，只在本機；備份 tag `backup/case-output-d55755f`；worktree 在 `.claude/worktrees/case-output` |
| `feat/case-storage` | `ff5ef0d` | 已併入 `main`，待清理；worktree 在 `../StablePacking-2D-ReAct-cases` |

`main` 上交接文件之前的 6 個 commit：

| Commit | 內容 |
| --- | --- |
| `de11970` | surface-fit、myopia 分析模組與測試 |
| `3a94436` | heuristic 計畫、findings、analysis README |
| `ff91d75` | batch runner（LHS 取樣、續跑、進度條） |
| `406df80` | 建牆入口搬到 `research/runner`，定義留在各研究資料夾 |
| `6886b87` | 砌牆加速：重用 worker、共用步驟資料、向量化 |
| `6a573a9` | `docs/TODO.md` |

### case-v1 合併注意

`feat/case-output` 含 `research/storage/case_records.py`、`research/walls/post_case.py`，規格 `docs/specs/case-output.md` 已在 `main`。這份程式從未端到端跑過。

- 建牆入口的搬移已在 `main`。把 `main` 併進 `feat/case-output`，在獨立 worktree 解衝突，不要 stash 主工作樹，解完再快轉 `main`。
- `research/validation/windows_case_runtime.py` 的 `configure()` 兩邊都改：`SharedPool`／`BASELINE_PERSISTENT_POOL` 與 `_keep_records_only`／`BASELINE_CASE_RECORDS` 都要留，就算 git 自動合併成功也要人工看過。
- `run_wall.py` 跟著搬到 `research/runner/`，環境變數區塊要同時有兩個變數。`check_run.py` 在試合併時也有衝突。
- 合併後把 `research/runner/verify_speedup.py` 改用 case-v1，並在 `research/walls/README.md` 加一列 `post_case.py`。

## 研究主線

### 問題

使用者的核心假設：Wang 五濾波級聯（LLC、NHR、IWR、GDS、LFR）逐步取當下最佳，卡在時間軸上的局部解，短期最佳達不到長期最佳。假設決策當下算不起後續回合，要找一個類似 A\* 的「感覺」，估計這一步對完工牆的影響。使用者認可的問法：

> 放下這塊之後，剩下的石頭和新頂面還配不配？

使用者的切入點是用包絡線或頂面平整度定義 h，也就是放下後最能填平目前牆體。

### 目前結論

數字與細節在 `research/findings.md` 的「Temporal myopia on TH_T80」，設計在 `docs/plans/surface-fit-heuristic.md`，這裡不重複。

- 統計一定要先去重：`research/runs` 的 101 面 TH_T80 完成牆只有 72 面不同（seed＋放置序列），`legacy-index.csv` 沒標這些重複。不去重時 LLC→`F_LR` 的一致率會虛高。真實樣本是 72 面牆、13 個種子。
- 問題核心是決策缺少未來資訊，而且多數可行選項選不到。當下分數不是系統性誤導，只是對長期結果幾乎沒有資訊。
- 只換濾波順序空間太小：120 種順序的中位數只有 3 個不同結果，h 要發揮就得讓控制器能直接選候選。
- 所有結論都是探索性：單一場景、18 塊石頭、檢定未校正。

### h 的候選

只用來排序同一牆況的候選，不保證 A\* 的最佳性。全部假設石頭垂直落下（實際 17.4% 的放置不符合）。

| 代號 | 作法 |
| --- | --- |
| H0 | 基準：GDS、unlevel |
| H1 | 流體水位 |
| H2 | 頂面包絡：開放面積＋比石頭窄的死角＋封死空隙 |
| H3 | 單石試坐：每塊剩餘石頭最好的座位 |
| H4 | 骨架推演：不算力學，把剩餘石頭落完 |
| H5 | 依已放面積比例分階段混合 |
| H6 | 學習式（需要更多資料） |

建議先做 H2、H3、H4。純函式在 `research/analysis/surface_fit.py`，未接進控制器，預測力尚未驗證。

不建議現在做：繼續調濾波順序或權重、短步數前瞻、直接訓練深度價值網路、只給 LLM 更多資訊而不放寬動作。

### 等使用者決定

1. 第一版預測對象：完成度＋牆高，或納入 `F_LR`。
2. 控制器能否直接選候選。
3. 能否跑離線建牆批次。
4. 第二個驗證場景。

## 下一步：驗證新分析模組

尚未開始。之前的建表與報告在跑完前就中斷，沒有留下輸出；`myopia_report.py`、`surface_fit_check.py` 的撰寫 agent 也是中途停掉的，內容可能不完整，要當成未經審查的程式看待。

在 PowerShell、repo 根目錄執行。輸出放 repo 外，`research/runs` 只讀。耗時沒有實測過，建表要讀全部 trace，建議背景跑。

```powershell
. .\research\load_env.ps1          # F_LR 的計算需要 MOSEK 授權
$PY = "$env:USERPROFILE\miniconda3\envs\stablepacking-baseline\python.exe"
$T  = "<repo 外的暫存資料夾>"

foreach ($t in 'test_surface_fit','test_myopia_tables','test_myopia_report') { & $PY -B -m "research.tests.$t" }

& $PY -B -m research.analysis.myopia_tables $T
& $PY -B -m research.analysis.myopia_report $T | Out-File -Encoding utf8 "$T\report.txt"
& $PY -B -m research.analysis.surface_fit_check "$T\unique.csv" | Out-File -Encoding utf8 "$T\check.txt"
```

基線環境沒有 pytest 和 ruff，測試用 `python -B -m research.tests.<模組>` 跑，ruff 用其他 Python 的。

判定方式：

| 模組 | 對照 | 怎樣算過 |
| --- | --- | --- |
| `myopia_tables` | 參考表 `steps.csv`、`walls.csv`、`unique.csv` | 列數與數值欄一致；`result` 這類絕對路徑欄換電腦後必定不同，不算差異 |
| `myopia_report` | findings 的 Temporal myopia 各小節 | 逐項相同；best-of-N bootstrap 依 `walls.csv` 的列順序抽樣，順序要一致 |
| `surface_fit_check` | findings 的 Vertical-drop model 小節 | 比例（82.6%／17.4%／94.6%、0.17、0.585）相同；耗時隨電腦變動，只看量級 |

`myopia_report` 不涵蓋 Vertical-drop 小節。

數字對不上時，先查原因，再決定改模組或改結論，不要直接改 findings。全部對上後，把 `research/findings.md` 開頭「whether they reproduce every number below is not yet checked」那句拿掉，並更新 `docs/TODO.md`。

驗證通過後，依計畫進驗證 A：用既有 74 個分支點離線篩選 h，不必新跑牆。

## 其他未完成

見 [`docs/TODO.md`](TODO.md)。最需要留意：

- `research/tests/test_worker_pool.py` 在來源電腦的 agent 工具環境失敗：`PermissionError: [WinError 5]`，出在 `_winapi.DuplicateHandle`，原因未判定。到目標電腦先在一般終端機重跑。
- 完整測試集沒跑過。
- 真實 `gpt_wall` loopback 牆沒實際跑過。

## 專案規則與使用習慣

這些原本只記在來源電腦的 Claude 記憶，不會跟著 repo 走。

- 規則檔：`.claude/GIT.md`、`DOC_STYLE.md`、`PYTHON_CODING.md`、`SHELL.md`。repo 根目錄沒有 `CLAUDE.md`。
- commit：英文、`<type>: <描述>`、一個 commit 一件事、不用 `--no-verify`、`--amend` 或 force push、不掛 AI 署名。文件只寫現在成立的事，沿革留給 git。
- `research/runs/` 是唯讀封存，程式經 `paths.LEGACY_RUNS` 讀取。新牆寫到 `research/cases/<case_id>/`，格式 `{YYYYMMDD-hhmm}-{場景代碼}-{pc1|pc2}`。策略與濾波順序刻意不放進 id（牆中途可能變），同定義的重跑用 `spec_sha256` 歸組。批次放 `research/studies/<id>/study.json`。
- 付費 GPT 要同時有 `STONEPACKING_ALLOW_PAID_API=1` 與 `--confirm-paid`，沒有使用者明確同意不要跑。
- 使用者以繁體中文討論，偏好先定義問題、討論後才動手；解釋時用學術語氣但領域知識當作零，先講大觀念，有結構就附圖。
- 來源電腦的硬體調度（借用 cfd-lbm-4rhgh 的 hwqueue，只管標題以「[周周]」開頭的 session）是那台電腦的設定，目標電腦沒有。

## 只存在來源電腦的東西

| 項目 | 位置 | 需要時 |
| --- | --- | --- |
| 封存牆 | `research/runs/`（1.7 GB） | 驗證必需 |
| 其他建牆輸出 | `runs/` | 視需要 |
| 授權與金鑰 | `.env.local`、`secrets\` | 目標電腦各自建立，不要複製金鑰到文件或 git |
| 參考表（1.5 MB）與重算輸出（56 KB） | `C:\Users\GAI\AppData\Local\Temp\claude\f5dff38df4192af5\fb6a8f72-5312-4f03-83e3-8284c0b6afbd\scratchpad\` 的 `out\`、`rerun\` | 逐欄比對用；在暫存區，可能被清掉 |
| 比對腳本 | 同一資料夾的 `compare_repo_tables.py` | 參數是含 `repo_tables\`（新表）與 `out\`（參考表）的資料夾 |
| 本機環境登記 | `research/validation/local-setup.json`、`worker-venv.local.json` | 目標電腦重新產生 |
| case-v1 worktree | `.claude/worktrees/case-output` | 分支帶過去後重建 |
