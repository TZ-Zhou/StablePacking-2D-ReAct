# Case 輸出規格（case-v1）

有追蹤、沒有 `--controller` 的牆，結束後 `research/cases/<case_id>/` 只留六個檔案，`receipt.json` 的 `output_format` 為 `case-v1`。建牆迴圈只寫事後算不出來的東西；其餘由 `research/walls/post_case.py` 事後計算。

格式的唯一定義與讀取程式是 `research/storage/case_records.py`。

## 資料夾

```text
research/cases/<case_id>/
├── receipt.json          這次執行的定義與結果
├── steps.csv             每次抽樣一列的決策
├── candidates.npz        每次選石的全部候選與分數
├── wall.npz              最後的牆
├── transformation.txt    作者格式的石頭位移與旋轉
├── run.log               建牆行程的輸出
└── derived/case-post-v1/ 跑過 post_case 才有
    ├── quality.csv       每次抽樣後的四項指標
    ├── mechanism.csv     左右極限載重
    ├── filter.txt        作者格式的濾波紀錄
    └── check.json        一致性檢查與資料檔雜湊
```

各種結束方式留下的檔案：

| 結束方式 | `steps.csv`、`candidates.npz` | `wall.npz` | `transformation.txt` |
| --- | --- | --- | --- |
| 完整建完 | 有 | 最後的牆 | 完整 |
| `--placements N` 到數停止 | 有 | 最後一次放置後的牆 | 只有標題 |
| `TIMEOUT`、`FAILED` | 有事件才有 | 建牆函式走完才有 | 只有標題 |
| 一個事件都沒有 | 沒有 | 沒有 | 只有標題 |

`--placements` 停止的牆可能多出未完成的最後一次嘗試：`candidates.npz` 有該次事件但 `steps.csv` 沒有對應列，`wall.npz` 也可能比 `placed` 列多一顆石頭。牆的寫入在提交之前，所以不會反過來。

## 產生流程

建牆行程只寫暫存：

| 暫存 | 內容 |
| --- | --- |
| `result_*/trace/eNNNNN.npz` | 每次選石一個事件檔，先寫暫存名再改名 |
| `result_*/trace/commits.jsonl` | 每次抽樣一行 |
| `result_*/wall.npz` | 有 `--placements` 時每放一顆更新，建牆函式結束時再寫一次，先寫暫存名再改名 |
| `result_*/transformation.txt` | 標題在建牆前寫，內容在建牆完成後寫 |
| `events.jsonl`、`execution.log` | 時鐘事件、行程輸出 |

行程結束後 `case_records.pack` 依序：

1. 寫 `candidates.npz`，複製 `wall.npz` 與 `transformation.txt`。
2. 最後寫 `steps.csv`（先寫暫存名再改名）。`steps.csv` 與 `candidates.npz` 都在，資料夾才算打包完成。
3. `execution.log` 改名為 `run.log`，並把 `events.jsonl` 的建牆資訊存進 `receipt.json`。
4. 刪除 `result*`、`data`、`matplotlib-config`、`config.json`、`events.jsonl` 與頂層的 `*.py`。

打包失敗時 `status` 為 `PACK_FAILED`，原因在 `pack_error`，暫存留在原處。行程樹未確認停止（`TERMINATION_FAILED`）時不打包。

## 座標與編號

| 名稱 | 定義 |
| --- | --- |
| 世界座標 | `matrix` 的列由地面算起（列 0 在地面側），行由左算起 |
| 評估座標 | 左側事件同世界座標；右側事件在 `np.fliplr(wall)` 上用鏡射石頭評估 |
| `x_world` | 左側等於 `x`；右側為 `W - x - width`；不可行候選維持 `x = -1` |
| `rotation_world` | 左側等於 `rotation`；右側為 `(360 - rotation) % 360`，即 90 與 270 互換 |
| `y` | 兩個座標系相同 |
| `(x, y)` | 姿態外框的最小行、最小列 |
| `rotation` `r` | 對原始石頭做 `r / 90` 次 `rotated_by_90` |
| 事件編號 | 從 0 起連續；`candidates.npz` 的第 k 個事件就是事件 k |
| 候選順序 | 事件抽到 `n` 顆石頭時，候選 `i` 是第 `i % n` 顆石頭、旋轉 `i // n * 90`；候選數是 `n` 的整數倍，四個旋轉時為 `4n` |

本文件的符號：

| 符號 | 意思 |
| --- | --- |
| `E` | 選石事件數；兩側建造時，一次抽樣有左右兩個事件 |
| `C` | 全部事件的候選總數 |
| `H`、`W` | 牆的模擬格點列數、行數 |
| `I`、`S` | 全部事件的剩餘石頭 id 總數、抽到的石頭 id 總數 |
| `B`、`R` | 牆上的塊數（地面、石頭、楔塊、切塊）、石頭數 |

## receipt.json

| 欄位 | 型別 | 內容 |
| --- | --- | --- |
| `schema_version` | int | receipt 結構版本 |
| `case_id`、`run_id` | str | 資料夾名稱 |
| `scene`、`case` | str | 場景代碼、`run_wall` 的案例名 |
| `parent` | str 或 null | 分支起點的牆 |
| `started` | str | ISO 8601 開始時間 |
| `seed` | int 或 str | 指定的種子；未指定時是說明來源預設值的文字 |
| `filter_policy` | object | `policy`、`order`、`tol`，有 `--switch` 時多 `switch` |
| `controller` | null | 打包的牆固定為 null |
| `trace` | bool | 打包的牆固定為 true |
| `placements_limit`、`timeout_seconds` | int | 停止條件；`placements_limit` 為 0 表示建到底 |
| `python_hash_seed_launcher` | int | 啟動時的 `PYTHONHASHSEED` |
| `source`、`source_sha256`、`source_edits` | str、str、array | `Tests/` 的建牆腳本、雜湊、套用的文字替換；只有一個腳本服務多場景時才有 `source_edits` |
| `staged_sha256` | str | 實際執行腳本的雜湊 |
| `spec_sha256` | str | 這面牆定義的雜湊，同一定義重跑時相同 |
| `git_head`、`git_branch`、`git_status`、`git_tracked_diff_sha256` | str、str、array、str | 版本與工作區狀態 |
| `code_sha256` | object | `{相對路徑: sha256}`，每個被執行的原始檔 |
| `command`、`pid` | array、int | 建牆行程的指令與 PID |
| `status` | str | 最終狀態，見下表 |
| `process_status`、`returncode` | str、int | 行程本身的結束情形 |
| `termination` | object 或 null | 被啟動器停止時的 `taskkill` 結果與 `verified`；自行結束時為 null |
| `elapsed_seconds` | float | 從啟動行程算起的秒數 |
| `committed_placements` | int | 放置數 |
| `output_format` | str | `case-v1` |
| `attempts`、`selection_events` | int | `steps.csv` 列數、`candidates.npz` 的 `E` |
| `build` | object | 見下表 |
| `execution_log_tail` | array | 只在失敗時出現：行程輸出最後 25 行 |
| `pack_error` | str | 只在 `PACK_FAILED` 時出現 |

`build`：

| 欄位 | 內容 |
| --- | --- |
| `input_stones` | 輸入石頭數 |
| `candidates_per_iteration` | 每次抽樣的石頭數設定 |
| `build_options` | 傳給 `build_wall_vendor` 的關鍵字參數 |
| `constants` | 建牆開始時 `Stonepacker2D.utils.constant` 的全部大寫常數 |
| `enter_seconds`、`return_seconds` | 建牆函式進入、返回的時鐘秒數；被停止時 `return_seconds` 為 null |

`status`：

| 值 | 意思 |
| --- | --- |
| `PROCESS_PASS` | 完整建完 |
| `STOPPED_AFTER_PLACEMENTS` | `--placements N` 到數後停止 |
| `TIMEOUT` | 超過 `--timeout` |
| `FAILED` | 行程非零結束 |
| `TERMINATION_FAILED` | 行程樹未確認停止，未打包 |
| `LAUNCHER_ERROR` | 啟動器本身出錯 |
| `PACK_FAILED` | 建牆結束但打包失敗，暫存保留 |

## steps.csv

UTF-8、有標題列、每次抽樣一列，列序即決策順序。編碼規則：

- 空白代表該次沒有這個值。
- 布林值寫成 `True`、`False`。
- 清單以空白分隔。
- 整數與浮點數都用 Python 的 `str`，可無損讀回。
- 用 `case_records.commits(case)` 讀，會還原成每次抽樣一個 dict。

| 欄位 | 型別 | 內容 | 空白的時候 |
| --- | --- | --- | --- |
| `step` | int | 建造步，第一步為 1；一步可有多次抽樣 | 不會 |
| `outcome` | str | 見下表 | 不會 |
| `chosen_side` | str | `left` 或 `right`；兩側建造時右側加權 GDS 嚴格較大才選右 | 不會 |
| `stone_id` | int | 選中的石頭 | 沒有可行候選 |
| `rotation_world` | int | 世界座標旋轉角 | 同上 |
| `x_world`、`y` | int | 世界座標位置 | 同上 |
| `filter_order` | str | 實際的濾波順序，例如 `LLC-NHR-IWR-GDS-LFR` | 事件檔遺失 |
| `01_LLC`、`02_NHR`、`03_IWR`、`04_GDS`、`05_LFR` | float | 入選候選的五個原始分數 | 沒有入選候選 |
| `wall_time_s` | float | 這次抽樣的秒數 | 行程在記錄前被停止 |
| `wall_time_cum_s` | float | 從建牆開始累計的秒數 | 同上 |
| `phase` | str | `single`；`sample_twice` 時為 `probe` 或 `place` | 不會 |
| `sample_trial` | int | 同一步內第幾次抽樣，從 0 起 | 不會 |
| `vendor_type` | str | `random`、`similar`、`variant`、`same`、`full`、`full_replace`、`sample_twice` | 不會 |
| `construction_order` | str | `from_left`、`from_right`、`two_sides` | 不會 |
| `sampled_ids` | int 清單 | 抽到的石頭 | 不會 |
| `side_tie` | bool | 兩側加權 GDS 是否相同 | 不會 |
| `cluster` | int | 選中石頭的群集 | 沒有可行候選 |
| `rotation_label` | int | 評估座標系旋轉角 | 同上 |
| `add_stone_ok` | bool | 石頭是否成功加到牆上 | 非 `placed` |
| `bricks_added` | int | 這次加上牆的塊數，含楔塊 | 非 `placed` |
| `placed_bbox` | int 清單 | `最小行 最小列 最大行 最大列`，穩定化之後 | 非 `placed` |
| `placed_rot_angle` | number | 穩定化的旋轉角（度），0 表示未旋轉 | 非 `placed` |
| `placed_rotate_from_ori` | number | 相對原始石頭的旋轉角（度） | 非 `placed` |
| `probe_cluster` | int | `probe` 得到的群集 | 非 `probe` |
| `termination` | str | 這次結束了建牆的原因 | 建牆繼續 |
| `left_event`、`right_event` | int | 該側的事件編號 | 該側未評估 |
| `left_ok`、`right_ok` | bool | 該側有沒有可行候選 | 該側未評估 |
| `left_index`、`right_index` | int | 入選候選在事件內的序號 | 該側未評估或沒有入選 |
| `left_gds_weighted`、`right_gds_weighted` | float | 入選候選的加權 GDS，沒有入選時為 `-1.0` | 該側未評估 |

`outcome`：

| 值 | 意思 |
| --- | --- |
| `placed` | 石頭放上牆 |
| `no_placement` | 這次抽樣沒有可行候選 |
| `probe` | `sample_twice` 的探測抽樣，只決定群集，不放石頭 |
| `cut` | 這次改放切割塊，沒有放選中的石頭 |

`termination`：`sample_trials_exhausted`（抽樣次數用完）、`inventory_empty`（石頭放完）、`filled`（牆填滿率超過 0.999）。

`wall_time_s` 是這次提交與上一次提交（第一次為建牆函式進入）的時鐘差。

## candidates.npz

`np.savez_compressed` 的壓縮檔，內容與事件順序固定。檔內不存 `run_id`，所以相同的兩面牆這個檔逐位元組相同；讀取時由資料夾名稱補上。

### 事件層

| 陣列 | shape | dtype | 內容 |
| --- | --- | --- | --- |
| `event_start` | `(E+1,)` | int32 | 事件 k 的候選是 `[event_start[k]:event_start[k+1]]` |
| `step` | `(E,)` | int32 | 建造步 |
| `side` | `(E,)` | str | `left`、`right`；`right` 的 `x`、`rotation`、`wall` 是鏡射的評估座標 |
| `phase`、`vendor_type`、`construction_order` | `(E,)` | str | 值同 `steps.csv` |
| `sample_trial` | `(E,)` | int32 | 同一步內第幾次抽樣 |
| `nb_stones` | `(E,)` | int32 | 這次要抽的石頭數 |
| `policy` | `(E,)` | str | `paper` 或 `code_original` |
| `order` | `(E,5)` | int8 | 濾波順序，值是 `features_raw` 的欄號；沒有指定順序時整列 `-1` |
| `tol` | `(E,)` | float64 | 濾波比較平均值的容差 |
| `branch` | `(E,)` | str | `full`：五個濾波都有倖存者；`fallback_k`：第 k 個濾波清空，改用加權和；`no_feasible`：沒有可行候選 |
| `n_stages` | `(E,)` | int8 | 這個事件留下的濾波階段數 |
| `weights` | `(E,6)` | float64 | 每個分數的權重，欄序同 `features_raw` |
| `selected` | `(E,)` | int32 | 入選候選在事件內的序號，沒有時 `-1` |
| `wall` | `(E,H,W)` | int16 | 選石當下的牆，評估座標；值的意義同 `wall.npz` 的 `matrix`；右側事件要 `np.fliplr` 才回到世界座標 |
| `world_width` | `()` | int32 | 牆寬 |
| `sampled_start` | `(E+1,)` | int32 | `sampled_ids` 的起點 |
| `inventory_start` | `(E+1,)` | int32 | `inventory_ids` 的起點 |

### 候選層

事件內的候選順序見「座標與編號」。

| 陣列 | shape | dtype | 內容 |
| --- | --- | --- | --- |
| `features_raw` | `(C,6)` | float64 | 五個濾波分數與一個未使用的分數，見下表 |
| `stone_id` | `(C,)` | int32 | 石頭 id |
| `rotation` | `(C,)` | int32 | 評估座標系旋轉角：0、90、180、270 |
| `width`、`height` | `(C,)` | int32 | 這個旋轉下的外框寬、高 |
| `footprint` | `(C,)` | str（16 字元） | 世界座標下形狀的雜湊，不含位置；同雜湊且同 `(x_world, y)` 表示佔用相同格點；沒有格點時為空字串 |
| `cluster` | `(C,)` | int64 | 群集，未分群為 `-1` |
| `x` | `(C,)` | int64 | 評估座標的行；不可行為 `-1` |
| `y` | `(C,)` | int64 | 列 |
| `x_world` | `(C,)` | int64 | 世界座標的行 |
| `rotation_world` | `(C,)` | int32 | 世界座標旋轉角 |
| `distance` | `(C,)` | float64 | `place_one_stone` 回傳的最佳位置距離分數，找不到位置時為 `inf` |
| `stage` | `(C,)` | int8 | 通過的最後一個濾波階段，沒通過為 `-1` |

`features_raw` 的欄：

| 欄 | 程式名稱 | 濾波 |
| --- | --- | --- |
| 0 | `course_height` | NHR |
| 1 | `global_density` | GDS |
| 2 | `interlocking` | IWR |
| 3 | `kinematic` | LLC |
| 4 | `local_void` | LFR |
| 5 | `local_void_center` | 不參與濾波 |

分數越大越好。`paper` 政策的可行候選是 `LLC > 0` 且五個濾波分數都是有限值，保留不低於當層平均的候選。

`stage` 的讀法：階段 `s` 的倖存者是 `np.flatnonzero(stage[first:last] >= s)`，`s < n_stages`。`paper` 政策的階段 0 是可行候選，階段 1 到 5 是通過第 1 到 5 個濾波。階段是巢狀的，所以一個數字就夠。

### id 清單

| 陣列 | shape | dtype | 內容 |
| --- | --- | --- | --- |
| `sampled_ids` | `(S,)` | int32 | 每個事件抽到的石頭 |
| `inventory_ids` | `(I,)` | int32 | 每個事件當下庫存裡沒被抽到的石頭 |

### 直接讀取

```python
import numpy as np

with np.load('candidates.npz') as c:
    first, last = c['event_start'][k:k + 2]
    raw = c['features_raw'][first:last]
    survivors = np.flatnonzero(c['stage'][first:last] >= s)
    sampled = c['sampled_ids'][c['sampled_start'][k]:c['sampled_start'][k + 1]]
```

## wall.npz

| 陣列 | shape | dtype | 內容 |
| --- | --- | --- | --- |
| `matrix` | `(H,W)` | int16 | `0` 空；`k+1` 是第 `k` 塊；`255` 是邊框 |
| `id_matrix` | `(H,W)` | int32 | 每格的 id：石頭 id；地面 256、楔塊 257、切塊 258；空格 0 |
| `cluster_matrix` | `(H,W)` | int16 | 每格的群集；地面為 `-1`；空格是 0，與群集 0 要配合 `matrix` 區分 |
| `ground` | `(H,W)` | bool | 地面遮罩 |
| `brick_id` | `(B,)` | int32 | 每塊的 id，依加上牆的順序，第 0 塊是地面 |
| `rock_index` | `(R,)` | int32 | 每顆石頭在 `brick_id` 的位置，依放置順序 |
| `rock_rot_angle` | `(R,)` | float64 | 穩定化的旋轉角（度） |
| `rock_rotate_from_ori` | `(R,)` | float64 | 相對原始石頭的旋轉角（度） |
| `rock_cluster` | `(R,)` | int32 | 群集 |

第 `k` 塊（`k > 0`）的遮罩是 `matrix == k + 1`。地面用 `ground`，因為邊框蓋過的地面格點在 `matrix` 裡是 255。

`case_records.load_wall(case)` 還原成與 `Base` 相同的屬性：`.matrix`、`.id_matrix`、`.cluster_matrix`（float64）、`._w`、`._h`、`.placed_bricks`、`.placed_rocks`；每塊有 `.id` 與世界大小的 `.matrix` 遮罩，石頭另有 `.rot_angle`、`.rotate_from_ori`、`.cluster`。

## transformation.txt

分號分隔，第一行是標題 `id;d_x;d_y;angle`，之後每顆石頭一行，依放置順序。由 `Tests/` 的建牆腳本寫入，重繪原解析度影像時需要，因為初始旋轉角只存在該腳本的執行過程。

`Tests/Thangavelu/00_run.py` 的定義：

| 欄位 | 內容 |
| --- | --- |
| `id` | 石頭 id |
| `d_x`、`d_y` | 石頭質心的行、列，除以 `_rescale` 換回原始解析度 |
| `angle` | 初始旋轉角、放置旋轉角與穩定化旋轉角的和（度） |

## run.log

建牆行程的標準輸出與標準錯誤，UTF-8 文字。

## derived/case-post-v1/

```powershell
. .\research\load_env.ps1
python -B -m research.walls.post_case <case id 或資料夾> [...]
```

極限載重需要 MOSEK，要先載入環境。`post_case` 拒絕 `research/runs` 封存、未打包的資料夾與已有此輸出的 case。檔案先寫到暫存資料夾，全部完成才改名。

`quality.csv`，每次抽樣一列，與 `steps.csv` 同列序：

| 欄位 | 內容 | 空白的時候 |
| --- | --- | --- |
| `step`、`phase`、`sample_trial`、`outcome` | 同 `steps.csv` | 不會 |
| `F_SF` | 石頭填充率 | 非 `placed` |
| `F_AH`、`F_AV` | 水平、垂直接縫路徑指標 | 非 `placed`，或牆還太小 |
| `F_LR` | 較小的極限載重除以摩擦係數 | 非 `placed` |

四項指標是當下未完成牆在模擬格點上的值，與 `research/quality` 的正式指標（原解析度）不能直接比較。

`mechanism.csv`：標題 `side,limit_load`，`right`、`left` 各一列，未四捨五入。沒有石頭的牆只有標題。

`filter.txt`：作者格式，每個事件依序寫 `step;result<階段>;<倖存石頭 id>;...;` 各一行，有入選時再一行 `step;select;<石頭 id>`。

`check.json`：

| 欄位 | 內容 |
| --- | --- |
| `status` | receipt 的 `status` |
| `ok` | 是否沒有任何問題 |
| `problems` | 問題清單；讀取失敗也列在這裡 |
| `results` | 每個結果的 `result`、`problems`、`stats` |
| `data_sha256` | `steps.csv`、`candidates.npz`、`wall.npz`、`transformation.txt` 的雜湊 |

## 讀取程式

`research/storage/case_records.py` 的函式同時讀打包的 case 與 `result_*` 資料夾：

| 函式 | 回傳 |
| --- | --- |
| `is_packed(folder)` | 是否為打包完成的 case |
| `result_dirs(run)` | 紀錄所在的資料夾清單：打包時是 case 本身 |
| `result_of(run)`、`traced_result(run)` | 單一紀錄資料夾；後者用於有追蹤、位在任意深度的封存牆 |
| `commits(result)` | 每次抽樣一個 dict，`sides` 為 `{left: {event, ok, index, gds_weighted}, ...}` |
| `events(result)` | 全部事件的 `(meta, arrays)` |
| `event(result, k)` | 事件 k 的 `(meta, arrays)` |
| `load_wall(result)`、`has_wall(result)` | 牆物件、有沒有牆 |
| `decisions(result)` | `step;stone_id;x_world;y;rotation_label`，每顆放置一行 |
| `filter_lines(result)` | 作者格式的濾波紀錄 |
| `event_clock(result, k)` | 事件的時鐘秒數，只有同一面牆內的差值有意義 |

`meta` 的鍵：`step`、`side`、`phase`、`sample_trial`、`vendor_type`、`construction_order`、`nb_stones`、`run_id`、`event`、`policy`（`policy`、`order`、`tol`）、`frame`（`world` 或 `mirrored`）、`world_width`、`branch`、`stages`（各階段倖存者序號的清單）。

`arrays` 的鍵：候選層的十二個陣列，以及 `weights`、`wall`（float64）、`selected`、`sampled_ids`、`inventory_ids`。

## 一致性規則

`research/walls/check_run.py` 對打包的 case 檢查：

- 事件編號從 0 起連續，事件依建造步排序。
- 每個事件都屬於一次抽樣，每次抽樣的側邊都指向存在的事件，且建造步、抽樣次、側邊、抽到的石頭一致。
- 依 `policy`、`order`、`tol` 重跑濾波，得到與 `selected`、`*_index`、`*_gds_weighted` 相同的結果。
- `chosen_side` 符合右側嚴格較大才選右的規則。
- 提交的位姿是入選候選的位姿。
- 每個事件的石頭 id 互不重複，抽到與剩餘的石頭等於先前放置後的庫存。
- 候選依抽到的石頭與旋轉排列，世界位姿符合座標換算。
- `placed` 列必須 `add_stone_ok`、`bricks_added ≥ 1`、`placed_bbox` 為四個整數。
- `wall.npz` 的石頭依序等於 `placed` 列的石頭，外框等於 `placed_bbox`。
- 相鄰事件之間，牆只增不減，且只增加該次放置的格點（有楔塊時可多）；未放置時不變。
- 未穩定化的石頭（`placed_rot_angle` 為 0）與入選候選逐格相同。

被啟動器停止的牆，尾端可以是一次未完成的嘗試。

## 與 result 資料夾的對應

`--controller`、`--no-trace` 的牆與 `research/runs` 封存維持 `result_*` 資料夾，內容對應如下：

| `result_*` | case-v1 |
| --- | --- |
| `trace/eNNNNN.npz` | `candidates.npz` 的事件 N |
| `trace/commits.jsonl` | `steps.csv` |
| `decisions.txt` | `steps.csv` 的 `placed` 列 |
| `filter.txt` | `candidates.npz` 的 `stage`、`selected`；事後重建於 `derived/case-post-v1/filter.txt` |
| `base.pkl` | `wall.npz` |
| `mechanism.csv` | `derived/case-post-v1/mechanism.csv` |
| `events.jsonl` | `steps.csv` 的耗時欄、`receipt.json` 的 `build` |
| `execution.log` | `run.log` |
| 事件檔的 `np_rng`、`py_rng`、`np_rng_keys`、`py_rng_state` | 不保存；`case_records` 讀 `result_*` 時略過 |
