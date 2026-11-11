# LLM 整面牆直接選：四項指標都要好，和 Wang 的 Paper 排序比

教授的標準：跟 LLM 說各項指標都要好，看它能不能做出好成績；跟 Wang 一樣好就算成功。

到目前為止 LLM 只選最後 7 顆（`docs/plans/llm-endgame-pick.md`、`llm-endgame-lookahead.md`、`llm-endgame-llc-note.md`），比較的對象是同一個殘局的各種結局。這次改成：

1. 提示明確寫四項指標都要好，不為了一項犧牲其他項；不再有只談 LLC 的那一段。
2. LLM 從第一顆到最後一顆都自己選，保留 `check_after` 工具。
3. 和 Wang 原本的做法（Paper 排序 3,0,2,1,4 的 cascade）在同樣的石頭上比整面牆。

本文在任何付費請求之前 commit；之後要改，另開 commit，報告時列出差異。

## 做法

- **每一步的候選和 Wang 一樣。** 每次放石頭，Wang 的抽樣先抽出一部分還沒放的石頭（`TH_T80` 每次 7 顆、`TH_T20` 每次 3 顆，剩下不夠時就是全部），每顆石頭四個旋轉各有一個位置。LLM 看到的就是 Wang 的 cascade 在這一步拿到的那一批候選（可行的那些），不多也不少；紀錄裡同時留下 cascade 在這一步會選哪一個。LLM 看不到沒被抽到的石頭，Wang 也看不到。
- LLM 一旦選得和 cascade 不同，牆就和 Wang 的牆分開，之後抽到的石頭也可能不同。所以「候選一樣」指的是 LLM 這面牆的每一步，不是和 Wang 那面牆逐步相同；兩面牆相同的是石頭（同一個種子、同一種牆）和第一步的候選。
- **決策點**：每一次有 2 個以上可行候選的選擇（以前只有庫存空了以後才是）。只有 0 或 1 個可行候選時不問 LLM。
- LLM 沒回答或回答無效時，這一步照 Paper 排序的 cascade（和以前相同）。
- 每個決策是獨立的一次對話，沒有前幾步的記憶（和以前相同）。
- **`check_after`** 每個決策點都能用，定義不變：對指定的候選，回答「放下它之後，這一步其他被抽到的石頭在下一步各有幾個可行姿勢」。庫存還有石頭時，下一步會重新抽樣，現在被抽到的石頭下一步不一定在；提示裡寫明這一點。表只在控制器要求時計算，不動亂數。

## 提示（觀察版本 `pick-4`）

觀察的欄位與 `pick-2` 相同。工具是 `check_after` 和 `commit_pick`，`max_tool_calls` 4。SYSTEM 全文：

> You build a dry-stone wall from the first stone to the last. The wall is built course by course with Wang's method. At each placement Wang's sampling puts some of the remaining stones on offer (all of them once only a few are left), and you choose which candidate to place: a stone on offer in one of four rotations, at the position Wang's image convolution found for it. Every listed candidate is feasible (LLC > 0). Your goal is the finished wall, judged on four metrics: F_SF, stone filling (the share of the wall area that is stone, higher is better); F_AH, the horizontal joint path index (lower is better: no long continuous horizontal joints); F_AV, the vertical joint path index (higher is better: vertical joints are staggered); F_LR, the smaller of the left and right kinematic limit loads of the wall divided by the friction coefficient (higher is better). All four metrics must be good: F_SF, F_AH, F_AV and F_LR. Do not sacrifice the others for one of them. A stone that cannot be placed is lost to the wall.
>
> Data. Stones are labelled S1, S2, ... in the order they were first sampled. Coordinates are pixels in the evaluation frame (mirrored when building from the right, so 'left' is the side the course is built from); rows count upward from the ground. Each candidate: id, stone, rotation, its pose x and y, the width and height of its bounding box, and Wang's five filter scores (higher is better for each): NHR course height (0 = level with its left neighbour), GDS global density, IWR interlocking (its vertical joint far from the joints below), LLC the kinematic limit load of the wall with the stone added and stabilized, divided by the friction coefficient, LFR local void (0 = flush on the left and on the bottom). n_remaining counts the stones not yet placed, on offer or not; wall summarizes the current wall; unplaceable lists stones on offer with no feasible candidate now. Nothing about the final wall is known.
>
> Before committing you may call check_after with one or more candidate ids (comma-separated). For each, it returns how many feasible poses (0 to 4, at most one per rotation) every other stone on offer now would have at the very next placement if that candidate were placed now. 0 means that stone could not be placed next; the counts can change again after later placements. While more stones remain than are on offer, the next placement's offer is a new sample, so a stone on offer now may not be on offer next. It gives no scores.
>
> Compare the main candidates on all five scores and on what they leave for the remaining stones; in the rationale state each one's main advantages and costs. Finish with commit_pick.

和 `pick-2` 的差別只有四處：開頭（整面牆、每步抽樣）、「All four metrics must be good ... Do not sacrifice the others for one of them.」、`n_remaining` 的說明、`check_after` 改成「其他被抽到的石頭」並加上重新抽樣的說明。四項指標和五個分數的定義一字不改，沒有 `pick-3` 那段 LLC 與 `F_LR` 的說明。`check_after` 工具的描述相應改成 "each other stone on offer now"。

`pick-1`、`pick-2`、`pick-3` 的提示文字不變（測試比對 SHA-256），舊紀錄照舊重播。

**實作**（付費前做）：pick 控制器的 spec 加布林值 `whole_wall`，只能和 `lookahead: true` 一起用，不能和 `llc_note` 一起用；為 true 時觀察版本是 `pick-4`，每個選擇都是決策點。`build_wall.py` 只改「何時計算 `check_after` 表」：這種控制器要求時，庫存不空也算；其他控制器照舊只在殘局算。控制器檢查在庫存不空時，只比對下一步又被抽到的石頭。

## 牆（先做 6 面）

開發種子 313、484、717，`TH_T80`（18 顆）和 `TH_T20`（23 顆）各一面。保留種子 720、761 這次不用，留給之後的確認。`AL` 三種牆這次不做（`check_after` 只定義在從左邊砌的牆）。

Wang 的牆是 factorial 區塊裡 Paper 排序（`30214`）的那一面；z 用同一個種子、同一種牆的 factorial 區塊（120 個排序）算，四項平均，`F_AH` 取負號。

| 種子 | 牆 | Wang 的牆 | 石頭（放下 / 全部） | 四項平均 z | 119 個其他排序中達到下面標準的面數 |
|---|---|---|---|---|---|
| 313 | TH_T80 | 20261109-1224c-TH_T80-pc1 | 18 / 18 | −0.047 | 64 |
| 313 | TH_T20 | 20261109-1117-TH_T20-pc1 | 22 / 23 | −0.149 | 26 |
| 484 | TH_T80 | 20261109-1841c-TH_T80-pc1 | 18 / 18 | −0.720 | 97 |
| 484 | TH_T20 | 20261109-1749b-TH_T20-pc1 | 19 / 23 | −0.479 | 91 |
| 717 | TH_T80 | 20261110-0031b-TH_T80-pc1 | 18 / 18 | 0.966 | 3 |
| 717 | TH_T20 | 20261109-2343c-TH_T20-pc1 | 20 / 23 | −0.667 | 108 |

建牆順序就是表的順序。

## 判定規則（只判定一次）

每面牆「不輸 Wang」的定義，三項都要成立：

- 這面牆走完，而且每個決策點都是 LLM 回答的（沒有退回 cascade）；
- LLM 的牆四項平均 z **不低於** Wang 的牆（相等也算）；
- LLM 放下的石頭數**不少於** Wang 的牆。

**過關：6 面中至少 4 面不輸 Wang。** 結果寫成「過關：整面牆不輸 Wang」或「沒有顯示追上 Wang」。6 面沒有全部走完就不判定，只報告。

這個門檻有多嚴，先講清楚（用上表最後一欄算）：隨便換一個過濾排序（不是 LLM），6 面中至少 4 面達標的機率是 0.41，至少 5 面是 0.075，6 面全部是 0.002。也就是說「至少 4 面」只代表和 Wang 同一個水準（Paper 排序本來就大約在 120 個排序的中間），不代表比隨便一個排序好；5 面以上才是隨便換排序不太做得到的。報告會把這一點和面數一起寫出來。

另外，LLM 如果每一步都選和 cascade 一樣的候選，牆會和 Wang 的完全相同，6 面都算不輸。所以報告一定同時列出 LLM 有多少次決策選得和 cascade 不同。

**另外報告**（不進判定）

- 每面牆 LLM − Wang 的四項 z 差（看有沒有為了一項犧牲其他項）、平均 z 差與 6 面的平均。
- 每面牆 LLM 的平均 z 在該區塊 120 個排序中的百分位。
- 選得和 cascade 不同的決策數、`check_after` 的使用次數、選的候選有沒有讓其他石頭變 0。
- 每面牆的請求數、tokens、金額、時間。

## 免費演練（付費之前全部要過）

1. 單元測試：`pick-1`、`pick-2`、`pick-3` 的提示 SHA-256 不變；`pick-4` 的 SYSTEM 等於上面的全文；`whole_wall` 沒有 `lookahead: true` 或同時有 `llc_note` 時 spec 被拒絕；`whole_wall` 時庫存不空的選擇也是決策點；判定規則的測試。
2. 舊牆不變：重蓋 factorial、arena、lookahead-arena 各 1 面，和原本完全相同；已有的 36 面付費牆，控制器檢查仍是 PASS。
3. loopback：6 面牆在本機假伺服器上走完整個 harness（`PROCESS_PASS`、控制器檢查 PASS，包含每一步 `check_after` 的那一列和下一步實際可行數相符），判讀程式從頭到尾跑通。
4. 同樣的石頭：每面 loopback 牆第一步的候選（石頭 id 和分數）與 Wang 的牆第一步完全相同。
5. 用 loopback 的請求大小估計付費牆的 tokens 與金額，和下面的上限對照。

## 付費測試（另外批准）

設定和修法 2 相同（`gpt-6.1-terra`，effort low，不重試，`lookahead: true`，`max_tool_calls` 4），加 `whole_wall: true`，`max_decisions` 25。研究名稱 `whole-walls`（loopback 是 `whole-walls-loopback`）。6 面只跑一次，不重跑。中途有牆沒完成就停下來報告。

**上限**　每面牆最多 100 個請求、250k tokens、US$0.50；整批最多 6 面、600 個請求、1.5M tokens、**US$3**。

**估計**　殘局批次每個決策約 4.3k tokens、US$0.008（7 個決策一面牆 US$0.05）。整面牆 `TH_T80` 約 18 個決策，前 11 步候選比較多（最多 28 個）；`TH_T20` 約 23 個決策，每步最多 12 個候選。估計 `TH_T80` 每面約 US$0.20、`TH_T20` 每面約 US$0.12，6 面合計約 **US$1**。loopback 之後用實際請求大小再核對一次，報告時一起給。

**限制**　每個種子、每種牆只有一面 LLM 牆，同樣的牆重跑時 LLM 也可能選得不同；Wang 那邊也只有一面。6 面只能回答「有沒有顯示追上 Wang」，不能回答穩不穩定。只有 `TH` 兩種牆、只有開發種子。

## 紀錄

開始、結束時間與檢查結果寫進 `runs/batch/factorial-log.csv`；每面牆的表寫到 `runs/batch/llm-whole-wall.csv`。
