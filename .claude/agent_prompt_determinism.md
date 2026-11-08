# 給執行 agent 的 prompt：隨機性驗證與方法整理

> 工作目錄：`C:\Users\GAI\Desktop\Scott\StablePacking-2D-ReAct`（Windows 11，branch `refactor/research-package`）。
> 任務分兩部分：**任務 1 驗證隨機性**（要實際跑），**任務 2 整理 Wang 與本專案的所有方法**（讀碼寫文件）。
> 兩部分互相獨立，可分給不同 agent；各自照下面的「共同規則」與「回報格式」。

---

## 背景

這個 repo 有兩層：

- **Wang 的原始方法**：`Stonepacker2D/`（石材堆疊、濾波選石、運動學穩定度評估），入口是 `Tests/*/00_run.py`、`Tests/Bricks/run.py`、`Tests/Almeida/*/run.py` 等。
- **本專案（使用者的研究）**：`research/`（以追蹤式濾波順序政策跑 Wang 的案例、離線分析濾波順序、ReAct／LLM 控制器挑選順序）。入口是 `python -B -m research.walls.run_wall ...`，說明見 `research/README.md`。

使用者發現「all runs 結果都不一樣」，懷疑是 random seed 沒控制好（LLM agent 本身不可控，這部分不用管）。上一輪只靠讀碼整理出下列**假說**，**全部尚未實跑驗證**，請你驗證或推翻，不要照單全收。

### 假說清單（待驗證）

**A. 真正的隨機來源（全域 `np.random`，主程序）**

- `Stonepacker2D/stack/vendor.py:15` import 時 `np.random.seed(1307)`。
- `vendor.py:89` `get_random_stones` 的 `np.random.choice`。
- `vendor.py:316` `get_variant_stones` 的 `np.random.permutation`（Thangavelu、Rubble、FFT 用 `vendor_type='variant'`）。
- `vendor.py:412 / 431 / 447` `get_same_stones` 的 `randint` 與補石 `choice`。
- `vendor.py:46` `reorder_dict` 的 `shuffle`（`Tests` 腳本似乎沒呼叫）。
- `Stonepacker2D/utils/kine_2d.py:10` import 時用 Python `random.choice` 產生顏色。
- 種子來源：各 `Tests/*/run.py` 開頭的 `seed_everything()`（預設 20）；`research/walls/run_wall.py:53` 的 `stage()` 用 `--seed` 改掉參數；`research/walls/campaign.py:54` 用 `default_rng(master_seed)` 產生各 seed，第一個固定 20。

**B. 自帶獨立 RNG 的部分**

- `research/controller/order_controller.py:545` `RandomPolicy` 用 `default_rng(seed)`。
- `research/analysis/arm_comparison.py:62` bootstrap 預設 `seed=0`。
- `research/llm_walls/dev_walls.py:45` 固定 `default_rng(MASTER_SEED)`。

**C. 沒有 seed 但可能讓結果不同的因素（重點驗證）**

1. **sampling 隨決策分岔**：不同 arm 第一個不同決策之後，剩餘庫存與 `return_stones` 的 dict 順序改變，同 seed 後面抽到的石頭也不同。
2. **同程序連跑 20 面牆共用一個全域 RNG**：`Tests/Thangavelu/00_run.py:131` 的 `for wall_i in range(20)` 只在最上面 seed 一次；`run_wall` 靠 `BASELINE_SELECT_FIRST_WALL` 只取第 1 面牆。
3. **牆鐘時間截斷 run**：`research/walls/run_wall.py:345` 每 2 秒輪詢 `committed_placements >= placements` 再 `taskkill`，停在第 N 顆或之後取決於時機；`--timeout` 也是牆鐘時間。
4. **MOSEK 求解器**（`Stonepacker2D/utils/kine_2d.py:290` `solve_force_rigid`）：LLC 值進入濾波的 `kinematic>0` 判斷，執行緒數或 CPU 不同時浮點尾數可能不同。
5. **`PYTHONHASHSEED`**：`seed_everything` 在程序內設 `os.environ` 無效，`run_wall` 靠啟動時設 `20`；直接跑腳本沒設。
6. **`place_one_stone` 內的 `seed_everything()`**（`Stonepacker2D/stack/place_stone.py:60`）在 worker 內重設；`build_wall.py:180` 的 `defined_sequence` 路徑會在主程序呼叫它而重設主程序 RNG。

**D. 無關結果**：`time.time`、`datetime.now`（資料夾名稱）、`time.monotonic`（只計時）。

---

## 共同規則

1. **先讀規範再動手**：`CLAUDE.md`（若有）、`.claude/PYTHON_CODING.md`、`.claude/SHELL.md`、`.claude/GIT.md`、`.claude/DOC_STYLE.md`、`research/README.md`。文件與 commit 的語言、格式以這些為準。
2. **不得修改既有原始碼**（`Stonepacker2D/`、`Tests/`、`research/` 底下現有 `.py`）。需要的暫時腳本放在你自己的暫存目錄，不放進 repo。
3. **不得動 `baseline/`、`research/runs/` 底下既有資料夾、`research/quality/`**。新實驗只能產生新的 `research/runs/<新名稱>`（`run_wall` 會拒絕其他位置，且不覆蓋）。
4. **不得呼叫付費 API**：不設 `STONEPACKING_ALLOW_PAID_API`，不跑 `llm_walls/`、不用 `--controller` 的 `api` 類型；`.env.local`、`secrets/` 不讀、不印。
5. **不 commit、不 push**，除非使用者明說。
6. **跑實驗一律循序，不要平行**：run 有牆鐘 timeout，平行跑會干擾結果，也會讓 CPU 競爭造成人為差異。每次跑前確認沒有殘留的 `python` 子程序在佔 CPU。
7. 環境用 `research/validation/environment-manifest.json` 登記的那個 Python（`run_wall` 會呼叫 environment guard）；不要自行 `pip install`。
8. 不確定的結論要標「未驗證」，不要寫成事實。引用程式用 `路徑:行號`。

---

## 任務 1：隨機性驗證

目標：判斷「同 arm、同 seed 重跑，結果是否位元組相同」；若不同，找出差異從哪個檔案、哪一步開始，並對應到上面 A–D 哪一項。

### 步驟

**1.0 靜態稽核（不用跑）**

- 全 repo 搜尋 A–D 沒列到的隨機或非決定性來源：`random`、`np.random`、`default_rng`、`RandomState`、`shuffle`、`permutation`、`choice`、`sample`、`set(` 的迭代、`dict` 順序依賴、`os.listdir`／`glob` 未排序、`time`、`uuid`、多執行緒／多程序的結果合併順序、`hash(`。範圍包含 `Stonepacker2D/entity/`、`Stonepacker2D/evaluate/`、`Stonepacker2D/utils/`、`Tests/`、`research/`（`tests/` 除外）。
- 逐項核對假說 A–D 的行號是否正確、描述是否成立；錯的要更正。
- 讀 `kine_2d.py` 的 MOSEK 設定：有沒有設 `intpnt_num_threads`、`num_threads`、`optimizer` 等參數；沒設就是預設（用全部執行緒）。
- 確認 `Pool` 的結果收集是否依序（`starmap` 保序），以及每個 worker 是否因 `seed_everything()` 與 `windows_case_runtime._restore_worker` 而對全域 RNG 狀態無依賴。

**1.1 重現性實驗（同 arm、同 seed 跑兩次）**

選最小、最快的案例（先看 `research/runs/` 既有 run 的耗時，優先 `bricks`；`CASES` 見 `research/walls/run_wall.py:29`）。名稱用 `det-<case>-a1`、`det-<case>-a2`（先確認不存在）。

```powershell
python -B -m research.walls.run_wall bricks det-bricks-a1 --policy paper --order 3,0,2,1,4 --seed 20
python -B -m research.walls.run_wall bricks det-bricks-a2 --policy paper --order 3,0,2,1,4 --seed 20
python -B -m research.walls.check_run research/runs/det-bricks-a1 --compare research/runs/det-bricks-a2
```

- 若 `bricks` 一次要超過 `--timeout` 預設，請先讀 `--help` 調整，**不要靠 `--placements` 截斷**（那會引入假說 C3）。
- 記錄 `compare_runs` 與 `differing_files` 的完整輸出。`time.txt` 與 PDF 內建立日期預期會不同，其他檔案不同才算問題。
- 同樣再對 `thangavelu-noise80`（或 `fft-d2-0`）做一組，但這組**先不要截斷**；若太慢，改用步驟 1.3。

**1.2 對照組**

- 同 arm、**不同 seed**（`--seed 20` vs `--seed 7`）跑一次，預期不同；用來確認 `--seed` 確實進到 `seed_everything(seed)`（讀 run 目錄裡 staged 的 `<case>.py` 與 `receipt.json` 的 `seed`、`staged_sha256`）。
- 不同 arm、同 seed（`--order 3,0,2,1,4` vs `--order 1,2,3,4,0`），找出**第一個不同的決策步**（比對 `decisions.txt`、trace 的 `commits.jsonl`），確認它之前完全相同、之後分岔，驗證假說 C1。

**1.3 牆鐘截斷效應（假說 C3）**

- 對同一 arm、同 seed 用 `--placements 10` 跑 3 次（名稱 `det-<case>-p1..p3`），比對每次 `decisions.txt` 實際行數（是否都剛好 10，還是 ≥10 且不一致）、`receipt.json` 的 `status`（`STOPPED_AFTER_PLACEMENTS`）。
- 檢查前 10 步的決策是否與 1.1 完整 run 的前 10 步相同（只比 `decisions.txt` 前 10 行即可）。

**1.4 若 1.1 出現不同**

- 用 `check_run.differing_files` 找最早分歧檔；用 trace 找第一個分歧的 `event`／`step`，比對該步的 `features_raw`（6 欄）與 `x`、`y`、`distance`。
- 判斷分歧的是 sampling（`sampled_ids` 不同 → RNG 問題）、還是 feature 值極小差異（浮點／MOSEK → 假說 C4）、還是 selection 平手（見 `Stonepacker2D/stack/selection.py` 的 tol 與 tie 規則）。
- 只回報證據，不要改程式修。

### 任務 1 的產出

在 `research/` 之外的暫存目錄（或回報訊息內）給出：

1. 假說 A–D 逐項「成立／不成立／未驗證」表，附證據（檔案:行號、或實驗名稱與輸出）。
2. 稽核新發現的隨機或非決定性來源清單（若有）。
3. 實驗表：實驗名、指令、結果（identical／differs）、第一個分歧點（若有）。
4. 結論：使用者看到的「all runs 不一樣」主要由哪幾項造成（排序），以及**建議的修法**（只寫建議，不要實作）。例如：每個 run 前重 seed、改用獨立 `default_rng` 傳入 `Vendor`、固定 MOSEK 執行緒、用完成事件而非輪詢截斷。

---

## 任務 2：整理 Wang 與本專案的所有方法

目標：產出一份可交接的方法總覽文件，**依據程式碼，不是只抄 README**。

### 要寫進文件

**Part 1 — Wang 的方法（`Stonepacker2D/`）**，每一項附 `路徑:行號`：

- 石材與牆的表示：`entity/`（Brick、Stone、Base、Wedge、transform）、矩陣／影像表示、`get_world_size`。
- 庫存 `Vendor`：`cluster_stones`（特徵、AgglomerativeClustering 的參數）、各 `vendor_type`（random、similar、variant、same、full、full_replace、sample_twice）的抽樣規則與是否用 RNG。
- 建牆主迴圈 `build_wall_vendor`：抽樣、四向旋轉、`construction_order`（from_left、from_right、two_sides）、重試 `max_sample_trial`、cutting、終止條件。
- 單顆石頭擺放 `place_one_stone`：`image_convolve` 候選位置、`evaluate_convolve`、穩定化（`set_stabalize_method`）、wedge。
- 六個特徵：course_height、global_density、interlocking、kinematic（LLC）、local_void、local_void_center，各自公式或來源檔案。
- 階層式濾波 `stack/selection.py`：`code_original` 與 `paper` 政策、`order`、`tol`、`switch`、`ALL_ORDERS`（120 種）、平手規則。
- 運動學評估 `evaluate/tilting_angle.py`、`evaluate/ksteps.py`、`utils/kine_2d.py`（MOSEK、摩擦係數 mu、limit force）。
- 並行與 RNG：`Pool`、`seed_everything`、各腳本的 `set_*` 設定。
- 各測試案例（`Tests/*`）的差異：石材來源、牆尺寸、`vendor_type`、`variant_sample`、旋轉間隔、輸出內容。
- 牆品質指標（Wang 的四項）：位置在 `research/walls/wall_metrics.py`、`campaign_quality.py`，標明指標名稱與公式。

**Part 2 — 本專案的方法（`research/`）**，同樣附 `路徑:行號`：

- 追蹤與重播：`run_wall.py` 的 staging 機制、trace 格式（`selection.write_trace`、`load_trace`）、`check_run.py` 的檢查與逐位元組比較。
- 濾波順序政策：`paper`／`code_original`、`--order`、`--switch`（單步反事實）。
- 控制器：`controller/order_controller.py` 的 Null、Random、Matched-Random、Rule、ReAct；`worker.py` 的獨立子行程；`llm_api.py`（預設不啟用付費）。
- Campaign：`walls/campaign.py`（arm × seed、master_seed、planned_runs、彙整）、`campaign_quality.py`。
- 離線分析：`analysis/` 底下每支腳本一句話說明「問什麼問題、輸入、輸出」。
- LLM 實牆：`llm_walls/`（smoke、gpt_wall、dev_walls）與花費帳本、安全限制。
- 驗證與環境：`validation/`（環境清單、Windows 相容層、worker venv）。
- 目前的研究結論：摘自 `research/findings.md`，標明哪些是已驗證、哪些是假設。

**橫切章節 — 隨機性地圖**：以任務 1 的驗證結果為準，畫出（文字或 mermaid）RNG 從哪裡來、誰消耗、哪些地方會因決策不同而分岔。

### 格式

- 繁體中文，格式依 `.claude/DOC_STYLE.md`；表格優先，少用長段落。
- 放在 `research/methods.md`（若與現有檔案重名，改 `research/methods_overview.md`）。
- 文件內每個結論必須能對到程式碼；沒讀過的檔案不要寫它的細節，列在「未涵蓋」。
- 檔案很多，可以用 subagent 平行讀碼（只讀、不改），但最後由你統一整合與核對行號。
- 完成後跑 `.claude/DOC_STYLE.md` 與 `.claude/readme_policy.json` 規定的檢查（若有）。

---

## 回報格式（兩個任務都適用）

1. 一段話結論（先講最重要的）。
2. 做了什麼、沒做什麼（含任何被規則擋下或跳過的步驟）。
3. 產出的檔案路徑與新建的 `research/runs/` 資料夾名稱。
4. 未驗證項目與原因。
5. 需要使用者決定的事（若有）。

測試或實驗失敗時，貼出實際輸出，不要只寫「失敗」。
