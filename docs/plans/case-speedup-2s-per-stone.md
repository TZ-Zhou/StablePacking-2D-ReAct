# 計畫：砌牆迴圈壓到每顆石頭 2 秒以內

讓 `run_wall` 的砌牆迴圈平均每放一顆石頭少於 2 秒，且每一面牆的決策與現在逐位元相同。

## 目標

**指標**　秒/顆 = (`events.jsonl` 的 `build_return` − `build_enter`) ÷ `placed_rocks`。

開機（匯入、載入石頭、形狀因子，約 15–25 秒）與牆完成後的評估不算在內，另外記錄總時間。

| 等級 | 場景 | 條件 |
| --- | --- | --- |
| 必達 | `thangavelu-noise80`、`fft-d2-0`、`bricks` | 完整牆、兩種政策、10 workers，兩次的較大值 < 2.0 秒/顆 |
| 爭取 | `almeida-pr` | 同上；階段 2 結束時至少 ≤ 3.5 秒/顆 |
| 記錄 | `almeida-r`、`almeida-ir`、`rubble`、`thangavelu-noise20` | 量出秒/顆，不設門檻 |

**不變條件**　同場景、政策、順序、種子的牆，`check_run.compare_runs` 回報零差異（決策、篩選值、變換、`commits.jsonl`、每個事件 npz、`base.pkl` 三個矩陣）。

## 現況

### 秒/顆

| 場景 | 石數 | 每批新 Pool | 持久 Pool | 紀錄（`research/runs/` 下） |
| --- | --- | --- | --- | --- |
| `bricks` code_original | 98 | 3.2 | 未量 | `bricks-code-original-trace-02` |
| `fft-d2-0` code_original | 34 | 2.9 | 未量 | `fft-d2-0-code-original-notrace-01` |
| `fft-d2-0` paper | 41 | 6.6（當時主機有其他負載） | 0.59 | `verify-perf-fft-d2-0-paperA-s20-tb`、`-pp` |
| `thangavelu-noise80` code_original | 18 | 16.4 | 未量 | `thangavelu-noise80-code-original-full-01` |
| `almeida-pr` code_original | 31 | 43.3 | 8.0（8 workers） | `pr-code-original-full-01`、`verify-perf-ap-code-s20-pp` |

`almeida-r`、`almeida-ir`、`rubble`、`thangavelu-noise20` 沒有完整牆的時間紀錄。

### 時間花在哪

**重開 worker 是最大成本。**`build_wall.py:371-378` 每批開一個 `Pool`，Windows 的 spawn 要重新匯入 numpy、scipy、cv2、mosek。

| `fft-d2-0` paper，每批 | 每批新 Pool | 持久 Pool |
| --- | --- | --- |
| 建 Pool | 0.56 秒 | 整面牆一次 1.46 秒 |
| `starmap`（含計算） | 4.55 秒 | 0.35 秒 |
| 整面牆的迴圈 | 272 秒 | 24 秒 |
| worker 合計忙碌 | 50.8 秒（156 個行程） | 36.7 秒（10 個行程） |

**`almeida-pr` 卡在批次等待與 worker 計算。**持久 Pool 下：

- 迴圈 248 秒，其中 `starmap` 241 秒，分 382 批，每顆 12.3 批。
- 3200 次 `place_one_stone`，worker 合計 770 秒，每顆 24.8 worker-秒。
- 平均只有 3.2 個 worker 在忙（共 8 個）：每批要等最慢的一個 pose。

| `almeida-pr` worker 區段 | 秒 | 佔比 |
| --- | --- | --- |
| `evaluate_kine` | 321 | 41.7% |
| 　`solve_force_rigid`（每次評估解兩個 LP） | 121 | 15.7% |
| 　`clear_contps` | 83 | 10.8% |
| 　`pixel_to_mesh` | 57 | 7.4% |
| 　`cal_Aglobal` | 46 | 6.0% |
| `evaluate_convolve` | 238 | 30.8% |
| 未掛計時器（主要是 `place_stone.py:64-80` 的兩次卷積） | 至少約 170 | 22% |

`fft-d2-0` 的 worker 時間有 87.6% 在 `evaluate_kine`，但每顆只有 4 個 pose，持久 Pool 後已低於目標。

**主行程不是瓶頸。**`deepcopy`、`select`、`add_stone` 整面牆合計不到 1 秒。每步寫檔也小：把封存的事件 npz 重存一次，每顆 6–28 毫秒（`almeida-pr` 最多，每顆 4.4 個事件），文字檔每次追加 0.24 毫秒。

### 傳輸量

`place_one_stone` 的每個 task 都帶整份 `current_bricks, all_bricks, base, region`。下表由封存的 `base.pkl` 與單顆石頭的序列化大小估算，是牆尾的上限，平均約一半。

| 場景 | `base.pkl` | 單顆石頭 | 每 pose | 每顆石頭的傳輸量 |
| --- | --- | --- | --- | --- |
| `thangavelu-noise80` | 9.6 MB | 0.11 MB | 約 14 MB | 約 330 MB（23 pose） |
| `bricks` | 36.9 MB | 0.36 MB | 約 39 MB | 約 165 MB（4.2 pose） |
| `almeida-pr` | 5.1 MB | 0.19 MB | 約 11 MB | 約 1.1 GB（103 pose） |
| `fft-d2-0` | 1.5 MB | 0.06 MB | 約 2 MB | 約 8 MB（4 pose） |

## 原則

- 一個階段一個 commit；每階段過驗收才開始下一階段。
- 每個加速都能關掉：`run_wall --fresh-pool` 回到每批新 Pool，供 A/B 量測與退路。
- `Stonepacker2D/` 依 `.claude/PYTHON_CODING.md:9` 只為追蹤或控制而改。能從 `windows_case_runtime` 做到的就不動上游；非動不可的項目集中在階段 4，要使用者同意。
- 同一件事一份程式：持久 Pool 只實作在 `windows_case_runtime.py`，`perf_runtime.py` 引用它再加計時。
- 時間一律成對量（見「驗收」），單次差距不算證據。

## 不做

- 不跳過任何 pose 的 kinematic 評估：`filter.txt` 與 trace 的值會變。
- 不讓同一批的 task 共用解封後的物件：`place_one_stone` 不修改參數只有靜態稽核，沒有實測。
- 不做石頭或牆的磁碟快取：多一層會過期的狀態，收益也還沒量到。
- 不把 `place_one_stone` 改成執行緒：理由見階段 6。
- 不改候選生成、篩選順序、卷積語意、LLC 閘門。
- 不改 `research/runs/`。不推送、不合併到 `main`。

## 階段總覽

| 階段 | 內容 | 動到 | 進入條件 | 預期結果（估計） |
| --- | --- | --- | --- | --- |
| 0 | 補基準量測、抓 task 樣本、修測試環境 | `research/perf`、`research/tests` | 案例 I/O 改動已提交 | 上表「未量」全部補齊 |
| 1 | 持久 worker 進 `run_wall` | 外圍 | 階段 0 完成 | T80 1–2、FFT 0.5、bricks < 1、PR 8 |
| 2 | 一步一次送出全部 pose | 外圍 | 階段 1 通過 | PR 約 3–3.6（8 workers）；T80 再降 |
| 3 | 每步只序列化一次 | 外圍 | 階段 0 量到傳輸佔迴圈 ≥ 10% | T80、bricks 牆尾變快 |
| 4 | worker 熱點 | 上游為主 | 使用者同意、PR 仍未達標 | PR < 2 |
| 5 | 開機固定成本 | 視項目 | 使用者要求總時間 | 總時間少 15–20 秒 |
| 6 | 步驟輸出移到寫檔執行緒 | 外圍 | 使用者決定（D6） | 每顆少 6–28 毫秒（≤ 1.5%） |

## 階段 0：補量測

本階段不改 `Stonepacker2D/` 與登記的執行路徑。

**前置**

1. 提交目前工作樹的案例 I/O 改動，之後所有基準以該 commit 為準。
2. 修測試環境：`test_order_controller`、`test_campaign_quality` 從唯讀封存複製後，複本仍是唯讀而報 `PermissionError`。測試要自己把複本改成可寫（`runs/_smoke/rerun_tests.py` 的 `make_writable` 做法）。完成後全套測試才有判讀價值。

**工具補強（`research/perf/`）**

| 項目 | 位置 | 內容 |
| --- | --- | --- |
| 秒/顆 | `profile_report.py` | 由 `events.jsonl` 算秒/顆並列在報告最上面 |
| 傳輸量 | `perf_runtime.py` `TimedPool.starmap` | 記每批第一個 task 的 `len(pickle.dumps(task, 5))` 與耗時，標籤 `main.payload.bytes`、`main.payload.pickle` |
| 記憶體 | `perf_runtime.py` `wrap_place_one_stone` | 每筆紀錄加 `psutil.Process().memory_info().peak_wset` |
| task 樣本 | `perf_runtime.py`，環境變數 `PERF_DUMP_TASKS=N` | 把 N 組真實輸入存成 `perf/tasks/*.pkl`，供階段 4 不跑牆就能比對 |

**量測矩陣**（`profile_case`，輸出到 `research/perf/runs/`，10 workers，主機無其他 CPU 票）

| 組 | 場景 | 模式 | 取得 |
| --- | --- | --- | --- |
| A | T80、bricks、rubble、PR、FFT | 每批新 Pool、持久 Pool 各一 | 秒/顆、傳輸量、busy worker |
| B | 同上 | 持久 Pool + `--cprofile` | 函式層級熱點，尤其 PR 那 22% 未計時區段 |
| C | T80、PR | 持久 Pool，`--pool-size` 8、10、14 | workers 擴展性（主機 16 邏輯核心） |
| D | T80、PR | 用 `run_wall` 開、關追蹤 | 追蹤寫檔佔迴圈多少 |

組 A 兩種模式也跑 `compare_runs`，順便補上 T80、bricks、rubble 持久 Pool 的等價證據。

**微觀量測**（用 task 樣本，單行程）

- `mosek.Env()` 建立耗時；`solve_force_rigid` 內建模與 `optimize` 各佔多少。
- MOSEK `num_threads` 預設與 1 的耗時，以及 `round(min(...), get_kine_precision())` 後結果是否相同。
- 每次 `ndimage.convolve`、`minimum_filter` 的耗時。

**產出**　更新本檔「現況」的表格，並依數字定案階段 3、4 要做哪些項目。

## 階段 1：持久 worker 進 run_wall

**改動**

| 檔案 | 改動 |
| --- | --- |
| `research/validation/windows_case_runtime.py` | 新增 `SharedPool`、`_run_task`；`configure` 依 `BASELINE_PERSISTENT_POOL` 選 `SharedPool` 或 `_pool` |
| `research/runner/run_wall.py` | 新增 `--fresh-pool`；環境變數預設 `1`；receipt 加 `worker_pool`（不進 `case_id.spec`，因為不決定牆） |
| `research/perf/perf_runtime.py` | `PersistentPool` 改成繼承 `SharedPool` 再加計時 |
| `research/perf/README.md`、`readme.md` | 更新說明 |
| `research/tests/test_worker_pool.py` | 新增 |

**設計**

```python
class SharedPool:
    """build_wall's `with Pool(n) as p: p.starmap(...)` over workers that live until the case process exits."""

    executor = None

    def __init__(self, processes):
        if SharedPool.executor is None:
            state = _worker_state()
            SharedPool.executor = ProcessPoolExecutor(
                state[0]["NUMBER_CPU"], initializer=_start_worker, initargs=(_restore_worker, *state))

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    def starmap(self, function, inputs):
        constants = _constants()
        futures = [SharedPool.executor.submit(_run_task, constants, function, task) for task in inputs]
        return [future.result() for future in futures]
```

| 議題 | 決定 | 依據 |
| --- | --- | --- |
| Pool 實作 | `concurrent.futures.ProcessPoolExecutor` | worker 異常死亡時丟 `BrokenProcessPool`；`multiprocessing.Pool` 會補新 worker 但遺失 task，`starmap` 永遠等不到 |
| 常數 | 每個 task 前同步大寫常數 | `set_dump` 等會在 Pool 建立後改變 |
| RNG | 不重送 | `place_stone.py:60` 每個 task 都先 `seed_everything()` |
| Pool 大小 | 首次建立時的 `NUMBER_CPU` | `build_wall` 傳入的 `min(cpu, nb_poses)` 只影響批次大小 |
| 結束 | 交給 `concurrent.futures` 的退出流程；`starmap` 出錯時先取消同一步未開始的 task | 正常結束時沒有未完成的 task，中途錯誤最多等正在跑的 pose |
| 父行程被單獨砍掉 | worker 內一條執行緒等父行程結束後 `os._exit` | executor 的 worker 自己也握著佇列的寫入端，讀不到 EOF，不會自行結束 |

worker 跨 task 的狀態已靜態稽核，結果在 `.claude/plans/verify-perf-core-report.md` §5。

**測試（`test_worker_pool.py`，不跑牆）**

- 兩次 `with Pool(n)` 用同一組 PID。
- Pool 建立後改常數，下一批 worker 看得到。
- 結果順序等於輸入順序；task 內的例外傳回主行程。
- worker 用 `os._exit(1)` 死亡時，`starmap` 在 30 秒內丟 `BrokenProcessPool`。
- `run_wall.terminate` 能收掉整棵行程樹，`terminate` 紀錄顯示全部 PID 已消失。

**驗收**　L1、L2、L3、L4（見「驗收」），另加：

- `--controller` 迴路測試（loopback）與 `--worker-venv` 牆都能跑完。
- `--placements 3` 停牆後沒有殘留行程。全套測試出現過 `TerminationFailed`（10 個行程在 `taskkill` 加 30 秒後仍在），持久 worker 不得使它更常發生。
- 完整牆 worker 的 `peak_wset`，牆尾不超過第一步的 1.5 倍。

## 階段 2：一步一次送出全部 pose

`build_wall.py:371-378` 把一步的 pose 切成多批，每批等最慢的一個。`place_one_stone` 每次呼叫都自己重設 RNG，互相獨立，所以整步一起排程不影響結果。

**做法（預設）**　只改 `SharedPool.starmap`，上游不動。收到一步的第一批（`j == 0`）時送出該步全部 pose，之後各批直接取結果：

```python
def starmap(self, function, inputs):
    first, common = inputs[0][0], inputs[0][1:]
    if first == 0:
        constants = _constants()
        SharedPool.common = common
        SharedPool.step = [SharedPool.executor.submit(_run_task, constants, function, (j, *common))
                           for j in range(len(common[1]))]
    assert all(a is b for task in inputs for a, b in zip(task[1:], SharedPool.common))
    return [SharedPool.step[task[0]].result() for task in inputs]
```

主行程在全部 `result()` 拿到前都停在迴圈內，不會修改 `base`，因此背景序列化看到的內容不變。`assert` 保證上游呼叫形式改變時大聲失敗。

**替代做法**　直接改 `build_wall.py:371-385` 成一次 `starmap`，約 8 行，意圖更清楚，但動到上游（見「待決定」D2）。

**測試**

- 同步驟多批的結果與逐批相同。
- 第 2 批的物件不是同一份時 `assert` 失敗。
- 下一步第一批（`j == 0`）會換掉上一步的 future。

**驗收**　L1、L2、L3、L4；PR ≤ 3.5 秒/顆，平均 busy worker 明顯高於 3.2。

## 階段 3：每步只序列化一次

**進入條件**　階段 0 量到必達場景的傳輸（序列化＋管線）≥ 迴圈時間 10%；否則跳過。

**做法**（只改 `windows_case_runtime.py`）

1. 主行程把 `common` 用 `pickle.dumps(common, 5)` 序列化一次，寫入 `multiprocessing.shared_memory.SharedMemory`。
2. task 只帶 `(name, size, j)`。
3. worker 連上共享記憶體，對每個 task 重新 `pickle.loads`，得到獨立物件，語意與現在相同。
4. 一步結束後主行程 `close()` 並 `unlink()`；行程被砍時 Windows 在最後一個 handle 關閉後自動回收。

**測試**　解封結果與直接傳遞相等；同一步兩個 task 拿到不同物件；一步結束後共享記憶體已釋放。

**驗收**　L1、L2、L3、L4；`main.payload.pickle` 降為約 1/pose 數。

## 階段 4：worker 熱點

**進入條件**　PR 在階段 2 後仍 > 2 秒/顆，且使用者同意改上游（D3）。

**預算**　目標 62 秒迴圈（31 顆）需要 worker 總量 ≤ 62 × workers × 效率。

| workers | 效率 | 上限 | 相對現在的 770 秒 |
| --- | --- | --- | --- |
| 10 | 100% | 620 秒 | −20% |
| 10 | 85% | 527 秒 | −32% |
| 8 | 85% | 422 秒 | −45% |

項目依序做，達標即停：

| 項 | 位置 | 做法 | 潛在節省（PR） | 風險 |
| --- | --- | --- | --- | --- |
| 4a | `kine_2d.py:290` | 每個 worker 重用一個 `mosek.Env`；在 `_restore_worker` 把 `mosek.Env` 換成不關閉的單例，上游不動 | 階段 0 微觀量測決定 | 低 |
| 4b | `kine_2d.py:78` `cal_Aglobal` | 每個接觸點只影響兩個元素，改成向量化填值，免掉 elements × contacts 迴圈與 `np.matrix` | 約 40 秒 | 低，運算順序相同 |
| 4c | `tilting_angle.py:418` `clear_contps` | 只走有接觸的元素對；每對的法向量與座標一次取陣列；`(cand, anta)` 順序與編號維持原雙迴圈順序 | 約 60 秒 | 中，接觸點順序決定 LP 欄位 |
| 4d | `place_stone.py:69,75`、`local_void.py:261,267` | 這些卷積只看 `== 0` 或 `!= 0`，輸入非負，改用二值化＋ `cv2.filter2D` 或 FFT 卷積，閾值 0.5；`cval`、`origin` 用補邊重現 | 最多約 300 秒（待 cProfile 拆分） | 中，邊界語意 |
| 4e | `tilting_angle.py:241` `pixel_to_mesh` | 向量化索引，最後才建 `ContPoint` | 約 30 秒 | 中 |
| 4f | `kine_2d.py:398-417` | `putcj`、`putvarbound`、`putacol` 迴圈改批次 API，索引順序不變 | 視建模佔比 | 低 |
| 4g | `kine_2d.py:292` 之後 | `num_threads=1`，只在階段 0 量到超訂時做 | 視量測 | 中，求解路徑改變，靠 `round` 吸收末位差 |

**每項的驗證**

1. 用階段 0 的 task 樣本比對：中間陣列（遮罩、`Aglobal`、接觸點序列）`np.array_equal`，`place_one_stone` 回傳值完全相同。
2. 過了才跑 L2、L3。

**驗收**　L1、L2、L3、L4；PR < 2.0 秒/顆。

## 階段 5：開機固定成本

只影響總時間，不影響秒/顆。使用者要求時才做。

| 項 | 目前 | 做法 |
| --- | --- | --- |
| `rotate_min_shape_factor`（`stone.py:401`） | FFT 18 秒、PR 11 秒，每顆算 92 次形狀因子 | 用持久 worker 平行計算，10 workers 預期降到 2–3 秒；不做磁碟快取 |
| worker 啟動 | 第一批多等約 4–5 秒（估計） | `configure` 時就建立 executor，與主行程載入石頭同時進行 |
| 完成後評估 | 約 4.5 秒 | 量過 `plot_failure_mec` 的繪圖耗時後，只保留 `mechanism.csv` 所需的計算 |

## 階段 6：步驟輸出移到寫檔執行緒

主迴圈不等磁碟：每步的輸出送進佇列，由一條執行緒依序寫出。

**分工**

| 工作 | 跑在 | 內容 |
| --- | --- | --- |
| 核心計算 | worker 行程，已與主行程分開 | `place_one_stone` |
| 主迴圈 | 主行程的主執行緒 | 等結果、`select`、更新 `base`，把輸出送進佇列 |
| 寫檔 | 主行程的一條執行緒 | trace npz、`commits.jsonl`、`filter.txt`、`decisions.txt`、`base.pkl` |

**核心計算維持多行程，不改執行緒**

- `evaluate_kine` 大多是 Python 迴圈，受 GIL 限制，執行緒不會平行。
- `place_stone.py:60` 每個 task 重設全域 `random`、`np.random`，`tilting_angle.py:39-41` 是模組層物件。執行緒共用這些狀態，決策會變。
- 主迴圈等 worker 時不佔 CPU。下一步要看這一步全部 pose 的結果，這段等待省不掉。

**預期效益**（封存事件重存到暫存目錄量得）

| 場景 | 事件/顆 | 存一個事件 | 每顆 | 佔 2 秒 |
| --- | --- | --- | --- | --- |
| `thangavelu-noise80` | 1.00 | 10.1 毫秒 | 10 毫秒 | 0.5% |
| `bricks` | 1.06 | 8.9 毫秒 | 9 毫秒 | 0.5% |
| `almeida-pr` | 4.39 | 6.5 毫秒 | 28 毫秒 | 1.4% |
| `fft-d2-0` | 1.18 | 5.2 毫秒 | 6 毫秒 | 0.3% |

這一階段不是達成 2 秒的手段，作用是磁碟忙碌時不拖慢迴圈。

**設計**

```python
class StepWriter:
    """One thread writes every step output in call order; the building loop never waits for the disk."""

    def __init__(self, depth=32):
        self.jobs = queue.Queue(depth)
        self.error = None
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, job):
        self._raise_if_failed()
        self.jobs.put(job)

    def flush(self):
        self.jobs.join()
        self._raise_if_failed()

    def _run(self):
        while True:
            job = self.jobs.get()
            try:
                if self.error is None:
                    job()
            except BaseException as error:
                self.error = error
            finally:
                self.jobs.task_done()
```

| 規則 | 原因 |
| --- | --- |
| 全部步驟輸出走同一條佇列、一條執行緒 | `run_wall.check_commits` 要求被砍的牆是 `build_wall` 寫入順序的前綴；只移走 npz，`decisions.txt` 會跑到事件前面 |
| 送出時快照：陣列與 bytes 當場複製，路徑當場算好 | 主迴圈會繼續更新 `base` |
| 寫檔時用原本的 mode 重開 | Windows 文字模式的換行與編碼逐位元不變 |
| 佇列上限 32 項，滿了主迴圈等待 | 記憶體有上限 |
| 寫檔錯誤在下一次送出或 flush 時丟出 | 案例非零結束，receipt 為 `FAILED`，不完整的資料不會被當成成功 |
| `build()` 在 `build_return` 前 flush，`atexit` 再保險 | `build_return` 之後牆的輸出必須完整 |
| 有 `--controller` 的牆維持同步 | `order_controller.append_trace` 同步寫 `controller.jsonl`，順序要一致；LLM 等待遠大於寫檔，也沒有省的必要 |

**上游不動**　`windows_case_runtime.configure` 換掉 `build_wall` 模組裡的兩個名字：

- `write_trace`：複製陣列後送進佇列。
- 注入模組全域 `open`：`build_wall.py:58,435,652,656,726` 都會先解析到它。模式 `a`、`a+`、`wb` 先收進記憶體，關閉時送進佇列；其他模式交給內建 `open`。

**開關**　`run_wall --sync-output` 關閉；receipt 加 `step_writer`（不進 `case_id.spec`）。

**測試（`research/tests/test_step_writer.py`，不跑牆）**

- npz、文字、pkl 混合寫入，落地順序等於呼叫順序。
- 送出後改陣列，檔案內容是改之前的。
- 寫檔失敗時，下一次送出或 flush 丟出例外。
- 佇列滿時主執行緒等待，資料不遺失。
- 寫檔停在第 k 項時，磁碟上是前 k 項，`check_commits(stopped=True)` 無問題。
- 文字模式輸出與直接寫檔逐位元相同。

**驗收**　L1、L2、L3。L2 的 `--placements 3` 由父行程砍牆，正好驗前綴規則。效益小於時間量測的雜訊，不做 L4，改量主迴圈每次送出 < 2 毫秒。

## 驗收

| 層級 | 內容 | 通過標準 | 機器時間（估計） |
| --- | --- | --- | --- |
| L1 | `research.tests` 全套 | 全過 | 約 25 分鐘 |
| L2 | 8 場景 × 2 政策，`--placements 3`，與同 commit 的 `--fresh-pool` 比 | 共同的前幾顆石頭零差異（被停的牆可能多放幾顆，比 `decisions.txt`、`filter.txt`、`trace/commits.jsonl` 的共同前綴）；`check_run` 為 ok | 約 15 分鐘（基準只跑一次，約 25 分鐘） |
| L3 | T80、FFT、bricks、PR 完整牆，2 政策，與同 commit 的 `--fresh-pool` 比；code_original 再與封存牆比 | 同上；`mechanism.csv` 相對差 < 1e-12 | 每階段約 30 分鐘（基準只跑一次，約 1.5 小時） |
| L4 | 時間成對量 | 每場景依序「舊、新、舊、新」；報告每一次；主機無其他 CPU 票；10 workers | 約 1 小時 |

- `mechanism.csv` 不要求逐位元：`kine_2d.py:290` 的 MOSEK 預設求解路徑在重跑間有末位差，基準對基準也會出現。
- `--placements 3` 的牆只驗決策，不驗速度：牆矮時成本低，`bricks` 的成本隨牆變大而增加。
- 驗證牆輸出到 `runs/speedup/<階段>/`（已被排除版控）；由 `research/runner/verify_speedup.py` 跑矩陣、寫結果 JSON，比對的定義在 `research/perf/verify_speedup.py`，邏輯取自 `runs/_smoke/smoke_matrix.py` 與 `verify_io.py`。階段 0 新增。
- 階段 0–2 的驗收不分 L1–L4，只跑一次比較：T80、FFT、bricks、PR 完整牆，2 政策，每批新 Pool 與持久 worker 各建一次，10 workers，輸出到 `runs/speedup/final/`。硬體調度員下線，直接執行，不送票。
- 階段 3、4b–4f、`rotate_min_shape_factor` 與階段 6 已實作，完整牆比較尚未跑；未做的項目與待驗證事項見 `docs/TODO.md`。

## 送票

CPU 工作一律經硬體調度員（`cfd-lbm-4rhgh/roso/runner/hwqueue.py`），`--who` 用本 session 標題，只管理「[周周]」開頭的 session。

- 白天（08:30–18:00）CPU 道上限 8 執行緒。L2、L3 用 8 workers 即可：T80 完整牆以 4 workers 與 10 workers 跑出的 `decisions.txt`、`transformation.txt` 逐位元相同。
- L4 與階段 0 組 A–C 的時間要 10 workers，排夜間。
- `--threads`、`--ram-gb`、`--est-min` 要填實際值。

## 風險

| 風險 | 偵測 | 處理 |
| --- | --- | --- |
| worker 跨 task 留下狀態 | L2、L3 | 關掉持久模式回到 `--fresh-pool`，定位後再開 |
| worker 記憶體增長 | `peak_wset` | 每 N 步重建 executor（Python 3.10 的 executor 沒有 `max_tasks_per_child`） |
| worker 崩潰卡住 | `test_worker_pool` | `BrokenProcessPool` 使案例失敗，receipt 記失敗 |
| 停牆後殘留行程 | `terminate` 紀錄、L2 | 先修 `TerminationFailed` 的根因，再開持久模式 |
| 上游呼叫形式改變 | 階段 2 的 `assert` | 大聲失敗，改用替代做法 |
| MOSEK 授權被長時間占用 | 同時跑多面牆時觀察 | 階段 4a 才有此風險；受影響時回到每次建 Env |
| 共享記憶體外洩 | 階段 3 測試 | 依賴 Windows 在 handle 關閉後回收 |
| 寫檔執行緒失敗或落後 | 階段 6 測試 | 錯誤在下一次送出或 flush 時丟出，案例失敗；佇列滿時主迴圈等待 |

## 待決定

| 編號 | 問題 | 建議 |
| --- | --- | --- |
| D1 | 目前的案例 I/O 改動是否提交，作為階段 0 的基準 | 提交 |
| D2 | 階段 2 改在外圍還是直接改上游 8 行 | 外圍，符合 `.claude/PYTHON_CODING.md:9`，可用 `--fresh-pool` 關閉 |
| D3 | 階段 4 能否改 `Stonepacker2D/` | PR 要達標就需要；階段 2 後看數字再決定 |
| D4 | PR 是否一定要 < 2 秒 | 先做到階段 2，用實測決定 |
| D5 | workers 是否超過 10（主機 16 邏輯核心） | 看階段 0 組 C；worker 數不影響決策 |
| D6 | 階段 6 要不要做 | 目前估計省不到 1.5%；等階段 0 組 D 的實測，佔迴圈 ≥ 5% 再做 |

## 提交順序

| 階段 | commit message |
| --- | --- |
| 前置 | `feat(walls): keep only data files in a case folder` |
| 0 | `test: make the copies of the read-only archive writable` |
| 0 | `feat(perf): record seconds per stone, payload size, peak memory and task samples` |
| 1 | `feat(walls): reuse one set of workers for a whole case` |
| 2 | `feat(walls): schedule every pose of a step at once` |
| 3 | `feat(walls): serialize the shared task arguments once per step` |
| 4 | 每項一個，例如 `perf(core): vectorize cal_Aglobal without changing its values` |
| 6 | `feat(walls): write the step outputs on a background thread` |

## 完成後

- 仍成立的規格（`worker_pool`、`step_writer` 欄位，`--fresh-pool`、`--sync-output` 語意）搬進 `docs/specs/`，本檔刪除。
- 最終秒/顆寫進 `research/findings.md`，其他文件只引用。
- `research/runs/verify-perf-*` 與 `.claude/plans/verify-perf-core-report.md` 的處理由使用者決定。
