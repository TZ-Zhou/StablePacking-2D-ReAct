# TH_T80 殘局 LLM 直接選，加上一步可行性查詢（修法 2，草案）

第 3 步（`docs/plans/llm-endgame-pick.md`）的 9 面牆中，484/13204 和 717/14032 少放了石頭。修法 1 把兩面牆在第 17 步換成其他候選、窮舉後面所有接續（16 面整牆重蓋，`runs/batch/llm-pick-step17-tree.csv`），結果：

- 兩面都是在第 17 步（最後一個有得選的決策）鎖死：484 的 4 個候選只有 S9 r180 能放完（4/4 接續），LLM 選了 S17 r180；717 的 4 個候選有 3 個能放完，LLM 選了唯一會丟石頭的 S14 r90。
- 兩面牆裡，一個候選會不會丟石頭，完全由「放下它之後，下一步是否有某塊石頭的可行姿勢變成 0」決定，沒有例外。

所以給 LLM 一個只查下一步可行性的工具。V3 暫停的 lookahead 只為這個工具開例外；觀察的其他欄位不變。為了公平，對手也拿到同樣的資訊：免費的對手牆多兩種「安全」規則（見「對手」），判定時 LLM 也要和它們比。本文在任何付費請求之前 commit；之後要改，另開 commit，報告時列出差異。

## 工具

`check_after`，參數 `candidates`：一個或多個候選 id。回傳每個 id：如果現在放這個候選，**下一次放置時**，其他每塊剩下的石頭各有幾個可行姿勢（0–4；Wang 的方法每個旋轉只找一個位置；可行 = LLC > 0 且五項分數有限，與 `unplaceable` 同一個定義）。只有個數，不給分數、位置或任何指標。

```json
{"after": {"2": {"S16": 2, "S17": 2}, "1": {"S16": 0, "S9": 0}}}
```

一次可以查多個 id，這樣每個決策通常只要 2 個請求（查詢 + commit_pick）。

## 怎麼算

- 只有 spec 寫 `"lookahead": true` 的 pick 控制器，或 endgame 控制器的兩種安全模式（`low_top_safe`、`random_safe`）會用到：庫存為空、可行候選 ≥ 2、單邊建造（TH_T80 從左）。其他情況（含所有舊牆）這段程式不會執行。
- 在 `Stonepacker2D/stack/build_wall.py` 的選擇步驟、呼叫控制器之前：對每個可行列，複製 base（deepcopy），把該候選用與實際放置相同的 `add_stone` 放上去，其餘石頭用與實際步驟相同的 `place_one_stone`（四個旋轉、同一個 worker pool）產生候選，數每塊石頭的可行姿勢。結果是一張表 `after`（列 × 石頭），隨其他輸入一起交給控制器，也寫進 trace 的 npz。
- 算完要求 numpy 與 Python 的亂數狀態不變（和 `controlled_policy` 的檢查相同），否則中止。
- 表在 LLM 呼叫之前就算好，工具只是查表（和 order 控制器的 `list_order_outcomes` 同一種做法）。所以重播檢查不需要 base，直接用 trace 裡的表。
- 成本：一般殘局步驟約 0.25 秒，每個決策多花「可行候選數 × 0.25 秒」，最多約 5 秒。

## 提示的改動（只有這些）

觀察版本 `pick-2`：欄位與 `pick-1` 相同，差在多一個工具。SYSTEM 在 "Compare the main candidates ..." 之前加一段：

> Before committing you may call check_after with one or more candidate ids. For each, it returns how many feasible poses (0 to 4, at most one per rotation) every other remaining stone would have at the very next placement if that candidate were placed now. 0 means that stone could not be placed next; the counts can change again after later placements. It gives no scores.

工具清單從 `[commit_pick]` 變成 `[check_after, commit_pick]`。`max_tool_calls` 3 → 4（一次查詢、一次 commit，兩次給被拒絕的回覆）。其他文字不變。

## 免費檢查（付費之前全部要過）

1. 單元測試與假 LLM：新增 mock 模式 `pick_safe`（查全部候選，在安全候選中選 LLC 最高者；沒有安全候選時在全部候選中選 LLC 最高者）。fake provider 與 loopback 牆走完整流程。
2. 控制器檢查加一項「工具說的是真的」：每個決策，實際放下的候選，它那一列必須等於下一個 trace 事件裡每塊石頭的可行姿勢數（pick 與 endgame 兩種控制器都檢查）。沒被選的列，用修法 1 的強制重蓋抽查（每面 loopback 牆至少 2 個決策的全部列）。
3. 舊牆不變：重蓋 factorial、arena 與第 3 步的 pick 牆各幾面，和原本完全相同；第 3 步 9 面牆的控制器檢查仍是 PASS。
4. 天花板：`low_top_safe` 在舊的 9 個狀態各蓋 1 面，加上新狀態的對手牆，看「只靠這一步資訊」能不能避免丟石頭。如果安全規則也常丟石頭，就先不付費，回來討論。

## 付費測試（另外批准）

**狀態**　9 個新的 7 塊狀態，不用第 3 步那 9 個。720、761 是內部測試用的種子，不能拿來開發，所以只能從開發種子 313、484、717 取。選法照 `docs/plans/endgame-arena.md` 的同一次洗牌往下取：每個種子跳過已用過的 3 個，再取 3 個狀態不同的。重算後前 3 個與 arena 的 9 個完全一致。

| 種子 | 順序 | factorial 牆（cascade 結局） |
|---|---|---|
| 313 | 13420 | 20261109-1211-TH_T80-pc1 |
| 313 | 14302 | 20261109-1215c-TH_T80-pc1 |
| 313 | 03241 | 20261109-1205-TH_T80-pc1 |
| 484 | 02413 | 20261109-1852b-TH_T80-pc1 |
| 484 | 10243 | 20261109-1835-TH_T80-pc1 |
| 484 | 01324 | 20261109-1848c-TH_T80-pc1 |
| 717 | 04123 | 20261110-0049-TH_T80-pc1 |
| 717 | 23410 | 20261110-0039b-TH_T80-pc1 |
| 717 | 20341 | 20261110-0038-TH_T80-pc1 |

**對手**　先免費替新狀態蓋 arena 結局，每個狀態 46 面，共 414 面：

- 第 1 步的 24 面：`none`、`random` 1–20、`llc`、`borda`、`low_top`；
- 同樣資訊的對手 22 面：`low_top_safe` 1 面、`random_safe` 種子 1–20 共 20 面，以及 `pick_safe`（假 LLM，走完整的 pick 控制器與工具）1 面。

**安全候選**：放下後，其他每塊剩下的石頭在下一步都至少有 1 個可行姿勢（`check_after` 那一列沒有 0；只剩最後一塊時一定安全）。`low_top_safe` 在安全候選中取 `low_top` 的選擇，`random_safe` 在安全候選中均勻抽一個（numpy `default_rng(種子)`，每次出手抽一次）；沒有安全候選時，兩者都退回在全部可行候選中照原規則選（`low_top`、`random`）。

另外在舊的 9 個狀態各蓋 1 面 `low_top_safe`（9 面），只當天花板檢查，不進判定。

**LLM**　與第 3 步相同（`gpt-6.1-terra`，effort low，不重試），只改 `lookahead: true` 與 `max_tool_calls` 4。

**上限**　每面牆最多 40 個請求、100k tokens、US$0.20；整批最多 9 面、360 個請求、900k tokens、**US$2**。第 3 步每面約 15k tokens、US$0.04，加工具後估計每面約 US$0.10，整批約 US$0.9。

**判定**（依序，只判定一次）

- 停止、繼續 (a)(b)(c)、「沒有顯示殘局貢獻」：與第 3 步相同。
- 另加兩項，與 (a)–(c) 一起都成立才算繼續；(d) 或 (e) 不成立，就算 (a)–(c) 都成立也不算繼續：
  - (d) 工具：LLM 結局丟石頭的狀態數 ≤ cascade 丟石頭的狀態數；
  - (e) 同資訊對手：至少 5 個狀態中，LLM 結局的四項 z 平均不低於 `low_top_safe` 的結局，而且放下的石頭不少於 `low_top_safe`。

**另外報告**　每個決策 LLM 有沒有呼叫工具、查了幾個候選、選的候選會不會讓某塊石頭變 0、和 `pick_safe` 選的是否相同；每個狀態 LLM 結局在 20 個 `random_safe` 結局中的百分位（和 `random` 的百分位並列）。

結果只當探索：同一組石頭、只有 TH_T80，而且只能看一步。

## 紀錄

開始、結束時間與檢查結果寫進 `runs/batch/factorial-log.csv`；每狀態的表寫到 `runs/batch/llm-endgame-lookahead-th80.csv`。
