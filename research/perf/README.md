# perf：單次案例的階段耗時與可選的持久 Pool

只動外圍：計時與 worker 行程的建立方式。Wang 的核心（`Stonepacker2D/`）與已登記的執行期相容層都不改；`run_wall` 的登記執行也不經過這裡。

## 內容
| 檔案 | 說明 |
| --- | --- |
| `profile_case.py` | 以 `run_wall.CASES` 與 `run_wall.stage` 暫存並執行一個原始案例（不追蹤、無控制器），寫到新的 `research/perf/runs/<名稱>/`（不進版控，不是登記牆） |
| `perf_runtime.py` | 暫存進案例資料夾的執行期：先套用登記的相容層，再換掉 `build_wall.Pool` 並加上計時；不可匯入 `research` 套件（spawn 的 worker 在案例資料夾執行） |
| `profile_report.py` | 唯讀報告：階段耗時、主行程計時、worker 區段、平均忙碌 worker 數，以及 `--cprofile` 的函式層級資料 |

## Pool 模式
| 模式 | 指令旗標 | 行為 |
| --- | --- | --- |
| 每批新 Pool | 預設 | `build_wall` 每批建一個新 Pool，只加計時 |
| 持久 Pool | `--persistent-pool` [`--pool-size N`] | worker 建一次、整次執行共用；每批重送常數快照，RNG 狀態不重送（`place_one_stone` 開頭自己重設 `random` 與 `np.random`） |

## 使用
```powershell
. .\research\load_env.ps1   # MOSEK 授權；金鑰在案例環境中被移除
python -B -m research.perf.profile_case fft-d2-0 perf-fft-pp --policy paper --order 3,0,2,1,4 --persistent-pool
python -B -m research.perf.profile_report show perf-fft-pp
```
- 兩種模式的結果是否相同，用同案例、政策、順序、種子各跑一次後，用 `research.walls.check_run.compare_runs(<資料夾 a>, <資料夾 b>)` 比對。
- `img/selection.txt` 由多個 worker 附加寫入，列的順序不固定，比對時要排序。
- 這是大量耗時的行程；在共用主機上先走硬體調度員送票，不要直接開跑。
- 測試：`python -B -m unittest research.tests.test_perf`（不跑案例）。
