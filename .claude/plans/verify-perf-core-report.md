# 驗證報告：`research/perf` 持久 worker pool 是否改變 Wang 核心結果

> 狀態：靜態稽核完成；fft-d2-0 三組 tb/pp 配對完成；almeida-pr 的持久 pool run 與既有登記 run 比較完成（無同條件 tb 對照）；bricks、thangavelu、pool-size、並行測試**未跑**（見「未驗證」）。

## 1. 結論

| 項目 | 結果 |
| --- | --- |
| 核心檔案是否被改 | 未改。`git diff HEAD --stat -- Stonepacker2D Tests research/validation research/walls research/controller` 為空。跑完後重算 210 個檔案的 sha256，與跑前相比只有 `research/perf/perf_runtime.py`、`research/perf/profile_report.py` 不同（主 session 修 `get_similar_stones` 的編輯，非本次驗證所改） |
| 持久 pool 結果是否與每批新 Pool 相同 | fft-d2-0 三組（`paper`、順序 A／B、seed 20／7）`decisions.txt`、`filter.txt`、`transformation.txt`、`base.pkl` 三個矩陣與其餘確定性檔案逐位元組相同；almeida-pr（`two_sides`＋`sample_twice`，code_original，seed 20，pool 8）與 11 月既有登記 run 相同（`compare_runs` 無差異，31 步）。**bricks、thangavelu、code_original 的 fft、其他 pool 大小、並行 run 未測** |
| 靜態稽核 | worker 呼叫路徑上沒有跨 task 保留且會影響回傳值的狀態（見 §5），前提是所有大寫常數在每個 task 前同步（`run_task` 已做） |
| 比較能抓出差異 | 能。換 seed、換順序都被偵測（§7） |
| 耗時 | 同組配對 305→55 s、389→65 s、263→57 s（同時另有其他 session 的負載，僅供參考） |

## 2. 等價矩陣

名稱前綴 `verify-perf-`；順序 A = `3,0,2,1,4`，B = `1,2,3,4,0`；tb = 每批新 Pool（`TimedPool`，不加 `--persistent-pool`），pp = `--persistent-pool`。比較工具為 `check_run.compare_runs`（decisions／filter／transformation／base.pkl 三矩陣）加上結果目錄所有檔案的 sha256；`time.txt`、PDF 忽略；`img/selection.txt` 見 §3。

| 案例 | 政策／順序 | seed | 池模式 | 基準 run | 比較 run | 結果 |
| --- | --- | --- | --- | --- | --- | --- |
| fft-d2-0 | paper／A | 20 | tb vs pp | `fft-d2-0-paperA-s20-tb` | `fft-d2-0-paperA-s20-pp` | identical（215 檔相同） |
| fft-d2-0 | paper／A | 7 | tb vs pp | `fft-d2-0-paperA-s7-tb` | `fft-d2-0-paperA-s7-pp` | identical（231 檔相同） |
| fft-d2-0 | paper／B | 20 | tb vs pp | `fft-d2-0-paperB-s20-tb` | `fft-d2-0-paperB-s20-pp` | identical（167 檔相同） |
| fft-d2-0 | code_original | 20 | tb vs pp | `fft-d2-0-tbase-a` | （無有效 pp run） | **未測**：`pers-a`、`pers-b` 啟動即失敗（rc 1，`get_similar_stones`），`code-s7-*` 被中斷 |
| almeida-pr（`two_sides`、`sample_twice`） | code_original | 20 | pp（pool 8）vs 既有登記 run | `pr-code-original-full-01`（追蹤式、每批新 Pool，預設 seed 20） | `ap-code-s20-pp` | `compare_runs` 無差異；核心輸出相同，31 步；僅 `img/left0.28864445493914576_mechanism.png` 與 `...91457_...` 檔名差浮點末位（相對差 1.9e-16）。登記 run 的結果目錄只有 5 個非 PDF 檔，因此「逐檔相同」只涵蓋這 5 個檔 |
| bricks | — | — | — | — | — | **未測** |
| thangavelu-noise80 | — | — | — | — | — | **未測** |

與既有登記 run 的獨立比較（不依賴本分支新寫的程式）：

| 比較 | 結果 |
| --- | --- |
| `fft-d2-0-tbase-a`（code_original，預設 seed 20）vs `fft-d2-0-code-original-notrace-01`（11/6） | 核心輸出相同；僅 `img/left0.06452246103746809_mechanism.png` 與 `...807_...` 檔名差一個浮點末位（相對差 2.2e-16），PNG 內容未逐一比對 |
| `ap-code-s20-pp`（持久 pool）vs `pr-code-original-full-01` | 核心輸出相同（見上表 almeida-pr 列） |
| 持久 pool vs `bricks-code-original-trace-02`、`thangavelu-noise80-code-original-full-01`、fft 登記 run | **未測** |

## 3. 基準的確定性（baseline vs baseline）

| 比較 | 結果 |
| --- | --- |
| `tbase-a` vs `tbase-b`（同為 `profile_case` TimedPool，code_original，預設 seed） | 核心輸出與其餘檔案相同 |
| `tbase-a` vs `rw-a`（`research.walls.run_wall` 追蹤式登記路徑，`check_run` 自檢 `ok: true`，34 步，PROCESS_PASS） | 相同；證明計時包裝不改結果 |
| `tbase-a` vs `notrace-01`（11/6） | 見 §2：僅 mechanism PNG 檔名浮點末位差 |

`img/selection.txt` 由多個 worker 以 `a+` 同時附加，每次跑行數都不同（同組基準 149、157、157、159 行），部分行被截斷或遺失。這在 tb、pp 兩種模式都發生，所以不能直接 sha256 比較。`cmp` 腳本改為只取欄位完整的行，以前 5 欄為鍵，檢查「兩邊都只有 1 行的鍵，其值是否相同」；所有配對都沒有矛盾值。多行同鍵（旋轉姿態共用前 5 欄）的情況只確認一邊是另一邊的子集。這是推論，不是逐行相等。

浮點末位差（MOSEK）：`kine_2d.py:290` 的 `solve_force_rigid` 沒有設定 `optimizer` 參數，使用 MOSEK 預設值；`kine_2d.py:736` 的第二個求解器才指定 `dual_simplex`。這是基準本身的不確定來源，與 pool 模式無關，只出現在未四捨五入的 LLC 值（PNG 檔名）。

## 4. 本次使用的 `perf_runtime.py` 版本

| 項目 | 內容 |
| --- | --- |
| 有效 run 的 staged `perf_runtime.py` | 與目前 repo 版本逐位元組相同（`diff` 無差異）。主 session 的修正與我當時的暫存副本相同，只移除 `MAIN_TARGETS` 的 `get_similar_stones` 一項 |
| 失敗 run（rc 1） | `base-a`、`base-b`、`pers-a`、`pers-b`：staged 版本含 `get_similar_stones`，啟動時 `AttributeError: type object 'Vendor' has no attribute 'get_similar_stones'`，沒有產生結果，不作證據 |
| 被中斷 run | `almeida-pr-code-tb`、`almeida-pr-paperA-s20-tb`、`fft-d2-0-code-s7-tb` 有部分輸出；`almeida-pr-code-pp`、`almeida-pr-paperA-s20-pp`、`fft-d2-0-code-s7-pp`、`bricks-code-tb` 無輸出。原因：使用者中斷我的等待指令時，背景行程一併結束。這些 run 無 `perf.json`，不作證據 |

## 5. 持久 worker 狀態稽核

稽核範圍：`place_stone.py` → `evaluate/local_void.py`、`evaluate/tilting_angle.py`、`utils/kine_2d.py`、`entity/*`、`utils/plot.py`、`utils/logger.py`、`utils/constant.py`。我自己讀完後，另請一個唯讀 agent 獨立稽核，結論一致。

| 位置 | 內容 | 能否影響回傳值 |
| --- | --- | --- |
| `place_stone.py:58-60` | 第 60 行 `seed_everything()` 重設 `random`、`np.random`；其前只有 `get_record_detail()`，不碰 RNG；`perf_runtime.py` 的包裝在它之前也不碰 RNG | 否 |
| `place_stone.py:32-35` | `os.environ['PYTHONHASHSEED']` 在 worker 內改寫；已啟動的直譯器 hash 不受影響，兩種模式 worker 都從啟動環境繼承 `PYTHONHASHSEED=20` | 否 |
| `utils/kine_2d.py:10` | import 時用 `random.choice` 產生顏色，只用於畫圖；消耗的 RNG 狀態被第 60 行重設 | 否 |
| `evaluate/tilting_angle.py:39-41` | 模組層 `Parameter` 物件；`parameters.mu` 在 242、417 行、`parameters_head.mu` 在 245-247 行每次使用前從 `get_mu()` 重新指派；`ContType`（`kine_2d.py:222-238`）按值複製 `mu`；`parameters_mortar` 未使用 | 否（過期值不會留存） |
| `utils/kine_2d.py:290、514、833` | 每次求解 `with mosek.Env()`，用完釋放，沒有模組層求解器狀態；`print_detail` 不被重設 | 否 |
| `evaluate/tilting_angle.py:131,190,224` | `plot_failure_mec` 只在 `save_failure=True`（主程序最終評估）時呼叫；worker 路徑不畫圖 | 否 |
| `place_stone.py:95,790`、`local_void.py:69,75,161` | 寫 `record/place_ite*.txt`（`w+`／`a+`）與 `img/selection.txt`（`a+`）；整個 `Stonepacker2D` 沒有任何讀回 | 否（只影響檔案內容或行序） |
| `utils/plot.py:49-73` | `plt.clf()` 後畫在當前 figure；圖會留在持久 worker，可能記憶體緩增；案例腳本設 `RECORD_PLACEMENT_IMG=False`，worker 不走這段 | 否 |
| `entity/base.py:148`、`entity/transform.py:267,301,340,348` | 可變預設參數 `[0,0]`，只被索引讀取，沒有被修改 | 否 |
| `entity/stone.py:321`、`transform.py:298,332,381-382` | `displacement_for_ka` 與原石共用同一個 list 並被寫入；只在 `stabilize='all'`（`base.py:178-181`）時執行，案例使用 `none`／`rotation`；整個 repo 沒有讀它 | 否（只寫不讀） |
| `entity/transform.py:12-205`、`entity/wedge.py:152-186` | 修改的是 `deepcopy(base)` 得到的 `future_base`，task 內部且確定性 | 否 |
| `utils/constant.py`（大寫全域） | `Stonepacker2D` 內沒有任何 `set_*()` 呼叫（在 `constant.py` 之外），worker 路徑不修改常數；`SELECTION_WEIGHTS`、`RECORD_SWITCHS`、`FILTER_POLICY` 是用 `{**a, **b}` 重綁，不是就地修改 | 否 |
| `utils/constant.py:275-282` | `_filter_controller` 為小寫，不在同步集合內；只有主程序 `build_wall.py:419` 讀 | 否 |
| `utils/logger.py:5,28,40-54` | `log_level`、`main_logger` 為小寫，不同步；只影響日誌或 `plt.show()` 分支 | 否（僅日誌） |
| 整條路徑 | 沒有 `lru_cache`、計數器、`id()`／`hash()` 排序、`default_rng`、`cv2` RNG、sklearn | 否 |
| `starmap` 分塊 | `len(inputs) ≤ 4×pool` 時 chunksize 為 1；pool 小於批次時（如 `--pool-size 2`）同塊內的 task 共用 unpickle 後的 `base`、`all_bricks`、`current_bricks`；`place_one_stone` 不修改這些參數，所以不影響結果 | 否，但**未用實測驗證** |

跨 wall 與常數同步：`PersistentPool.starmap` 每批呼叫 `settings()`（`perf_runtime.py:203`）取主程序當下所有大寫常數，`run_task` 在每個 task 前 `update` 進 worker 的 `constant` 模組（第 166 行），所以案例腳本在 pool 建立之後呼叫的 `set_dump`、`set_selection_weights`、多面牆迴圈中的 `set_dump` 都會到達 worker。實測支持：fft-d2-0 的 pp run 輸出到正確的 `result_*_wall0` 目錄。**多面牆（`BASELINE_SELECT_FIRST_WALL=0`）連續跑未實測**。

## 6. 包裝層靜態審查（`perf_runtime.py`）

| 檢查項 | 結論 |
| --- | --- |
| `timed`（88-94 行） | 回傳 callee 結果；例外經 `finally` 計時後原樣拋出；不改參數；不碰 `np.random`、`random`、全域設定 |
| `install_timers`（97-104 行） | 被包裝的皆為一般函式或實例方法，`Stone`、`Base` 的方法都沒有 `staticmethod`／`classmethod`，換成函式後綁定行為不變；屬性不存在時 `AttributeError` 大聲失敗（`get_similar_stones` 事件），不會靜默改變行為 |
| `restore_worker`（152-155 行） | 先呼叫登記的 `registered._restore_worker`（常數、numpy 狀態、random 狀態），再安裝計時器 |
| `wrap_place_one_stone`（121-149 行） | 只在 `finally` 內讀 `args[3].placed_bricks` 並寫 `perf/place_<pid>.jsonl`，不改回傳值；主程序端的 `place_one_stone` 沒被包裝，所以 pickle 以名稱解析時兩種模式一致 |
| `TimedPool`（170-188 行） | `__exit__` 呼叫 `terminate()`，與原 `with Pool(...) as p` 相同；`starmap` 直接回傳；初始化參數與登記 `_pool` 相同 |
| `PersistentPool.starmap`（202-208 行） | 對 `inputs` 逐項包成 `(constants, function, arguments)`，`multiprocessing.Pool.starmap` 保序；`function(*arguments)` 參數與原相同；`__exit__` 回傳 `None`、不終止 pool；結果由 `atexit` 的 `terminate` 清理，不影響已回傳的值 |
| 未受 RNG 狀態重送影響的原因 | worker 每次任務在 `place_stone.py:60` 重設 RNG；實測（paper A／B、seed 20／7 的三組 identical）與此一致 |
| pool 大小 | `PersistentPool.__init__` 忽略 `processes` 參數，大小取 `--pool-size` 或首次建立時的 `NUMBER_CPU`；批次大於 pool 時任務排隊，順序不變 |
| 失敗案例 | 啟動時 `get_similar_stones` 不存在導致 `AttributeError`（rc 1），已由主 session 修正；錯誤是大聲的，不會產生錯結果 |

## 7. 敏感度檢查（比較會失敗）

| 比較 | `compare_runs` 回報 |
| --- | --- |
| fft paper A seed 20 vs seed 7（tb） | `decisions.txt`、`filter.txt`、`transformation.txt`、`base.pkl matrix`、`base.pkl id_matrix` 全不同；0 個其餘檔案相同 |
| fft paper A seed 20 vs seed 7（pp） | 同上 |
| fft paper A vs B（seed 20，tb） | 同上；mechanism PNG 值也不同 |

因此上述 identical 結果不是比較工具永遠回報相同。

## 8. 未驗證與待決定

| 項目 | 狀態 | 原因／需要 |
| --- | --- | --- |
| almeida-pr 同條件 tb（`TimedPool`）對照 | 未跑；票 `1108-155503-1ca9` 已取消（狀態 cancelled） | tb 需 10 執行緒，白天 cpu 道上限 8 執行緒，調度員要等夜間才放行；改以 2026 年 11 月既有登記 run `pr-code-original-full-01` 當基準，其 `git_head`（97a2365）與目前 HEAD 之間 `Stonepacker2D/` 有 4 個檔案變動（`git diff 97a2365 HEAD --stat -- Stonepacker2D`，136 行新增、9 行刪除，內容未逐行審） |
| bricks、thangavelu-noise80 tb vs pp | 未跑 | 依優先序在 almeida-pr 之後，使用者要求先收尾；調度員白天不放行 10 執行緒（tb 需要） |
| fft-d2-0 code_original 的持久 pool run | 未跑 | 原有的 `pers-a/b` 啟動失敗 |
| 與登記 run 的持久 pool 比較（bricks-trace-02、thangavelu-full-01、pr-full-01） | 未跑 | 同上 |
| `--pool-size 2/4/10`、4 個並行 pp run | 未跑 | 同上；`--pool-size 2` 時的 chunk 共用參數只有靜態推論 |
| 多面牆（`BASELINE_SELECT_FIRST_WALL=0`）連續跑 | 未驗證 | 需要完整跑多面牆，耗時過長 |
| `stabilize='all'` 路徑與 pyMetaheuristic RNG 來源 | 未驗證 | 套件不在預設環境、案例不使用此模式 |
| mechanism PNG 內容 | 只比檔名浮點差；PNG 位元組在檔名相同時相同，檔名不同者未比對 | — |
| `img/selection.txt` | 只能以鍵比對，不是逐行相等 | 附加競爭在兩種模式都會截斷或遺失行 |
