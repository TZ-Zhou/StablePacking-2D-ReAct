# perf：單次案例的階段耗時、傳輸量、記憶體與 task 樣本

`research/runner/profile_case.py` 只用來量時間，不是登記牆的入口。`run_wall` 本身就用持久 worker（`--fresh-pool` 改回每批新 Pool），要牆一律走 `run_wall`。

這裡只動外圍：計時、取樣與 worker 行程的建立方式。Wang 的核心（`Stonepacker2D/`）與已登記的執行期相容層都不改；`run_wall` 的登記執行不經過 `perf_runtime`。

## 內容
| 檔案 | 說明 |
| --- | --- |
| `perf_runtime.py` | 暫存進案例資料夾的執行期：先套用登記的相容層，再換掉 `build_wall.Pool`，加上計時、傳輸量、記憶體與 task 樣本；不可匯入 `research` 套件（spawn 的 worker 在案例資料夾執行） |
| `profile_report.py` | 唯讀報告：秒/顆、階段耗時、主行程計時、傳輸量、worker 區段、平均忙碌 worker 數、worker 記憶體，以及 `--cprofile` 的函式層級資料 |
| `task_results.py` | 不建牆比較兩份程式：對 `perf/tasks` 存下的真實 task 跑 `place_one_stone`，比每個回傳值的位元，`PYTHONPATH` 決定跑哪一份程式 |
| `verify_speedup.py` | 加速比較的定義：要建哪些牆、兩面牆是否相同、報告內容。建牆由 `research/runner/verify_speedup.py` 執行，輸出到 `runs/speedup/<標籤>/` |

`profile_case` 的輸出在 `research/perf/runs/<名稱>/`（不進版控）。

## Pool 模式
| 模式 | 指令旗標 | 行為 |
| --- | --- | --- |
| 每批新 Pool | 預設 | `build_wall` 每批建一個新的 `multiprocessing.Pool`，只加計時，當作基準 |
| 持久 Pool | `--persistent-pool` [`--pool-size N`] | 繼承 `run_wall` 用的 `SharedPool`，只多計時：worker 建一次、整次執行共用，每步的第一批就送出該步全部 pose；每個 task 帶常數快照，RNG 狀態不重送（`place_one_stone` 開頭自己重設 `random` 與 `np.random`） |

持久 Pool 的 `main.pool.create` 只含建立 executor。worker 行程在送出 task 時才啟動，這段時間算在 `main.pool.starmap` 內。

## 報告內容
| 項目 | 怎麼算 |
| --- | --- |
| `seconds per stone` | 標題下第一行，(`build_return` − `build_enter`) ÷ `placed_rocks`，事件取自 `events.jsonl` |
| `payload` | 主行程計時之後的一行：每個 task 的 MB、總 GB、序列化秒數。每次送出時序列化一個 task，再乘上實際送出的 task 數，所以是估計值；每批新 Pool 算該批的 task 數，持久 Pool 在每步第一批算整步的 task 數，之後各批不算 |
| `worker peak working set` | worker 段落最後一行：各 worker 第一筆與最後一筆紀錄的 `peak_wset`，各取所有 worker 的最大值 |

- 舊的執行資料缺少某項量測時，該行印 `not available`。
- `main.payload.bytes` 記的是位元組，不列入主行程計時；`main.payload.pickle` 是估計的秒數，會列入。
- 量測用的那次序列化與 task 樣本的寫檔發生在 `main.pool.starmap` 計時開始之前，但仍在砌牆迴圈內，所以算在秒/顆裡。

## task 樣本
`--dump-tasks N` 把真實輸入存成 `perf/tasks/task_<i>.pkl`：主行程每第 10 次 `starmap` 取該批中間的 task，存滿 N 個為止。每個檔案是 `{"constants": <大寫常數>, "arguments": <place_one_stone 的參數>}`，可以不跑牆直接重現單次呼叫。序列化失敗時案例直接中止。取樣不抽亂數，也不改參數。

## 使用
```powershell
. .\research\load_env.ps1   # MOSEK 授權；金鑰在案例環境中被移除
python -B -m research.runner.profile_case fft-d2-0 perf-fft-pp --policy paper --order 3,0,2,1,4 --persistent-pool
python -B -m research.perf.profile_report show perf-fft-pp
```
- 兩種模式的結果是否相同，用同案例、政策、順序、種子各跑一次後，用 `research.walls.check_run.compare_runs(<資料夾 a>, <資料夾 b>)` 比對。
- 這是大量耗時的行程；在共用主機上先走硬體調度員送票，不要直接開跑。
- 測試：`python -B -m unittest research.tests.test_perf`（不跑案例，也不開 worker 行程）。
