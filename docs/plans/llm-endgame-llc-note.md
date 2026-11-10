# TH_T80 殘局 LLM 直接選：在提示中說明 LLC 與 F_LR 的關係

修法 2（`docs/plans/llm-endgame-lookahead.md`）的 9 面付費牆判定為 go，但平均比 `pick_safe` 差（`research/findings.md` 的 "LLM against pick_safe"）：

- LLM 輸給 `pick_safe` 的 5 個狀態，`F_LR` 全都比較低。
- 有 14 次決策 LLM 沒選 LLC 最高的安全候選，其中 11 次選的候選 NHR 比較高。
- 提示要 LLM 比較全部五個分數，但沒有說 LLC 和 `F_LR` 是同一種量。

程式裡兩者確實是同一種量：選中候選的 LLC 是放下它之後整面牆的 λ_min（四捨五入到小數一位）除以摩擦係數；`F_LR` 是完成的牆的 λ_min 除以摩擦係數。

這次只在提示中補上這件事，其他都不變。本文在任何付費請求之前 commit；之後要改，另開 commit，報告時列出差異。

## 提示的改動（只有這些）

觀察版本 `pick-3`：欄位、工具（`check_after`、`commit_pick`）、`max_tool_calls` 4 都與 `pick-2` 相同，只差 SYSTEM 多一段。這段插在 `pick-2` 的 check_after 說明之後、"Compare the main candidates ..." 之前：

> LLC and F_LR are the same kind of quantity, a kinematic limit load of the wall divided by the friction coefficient: LLC is that of the wall right after this placement, F_LR that of the finished wall. Giving up a higher LLC now can therefore leave a lower F_LR at the end. Weigh this together with everything else; it is not a rule.

寫這段的原則：

1. 只說 LLC 和 `F_LR` 的關係（程式裡成立），不給任何數值。
2. 其他四個分數（NHR、GDS、IWR、LFR）和最終指標的關係，資料沒有證明（`research/findings.md` 的 C2：只有 NHR→`F_AH` 顯著，而且效果很小），所以不寫。
3. 不寫「一定要選 LLC 最高的」，由 LLM 自己權衡。

`pick-1`、`pick-2` 的提示文字不變（測試比對 SHA-256），舊紀錄照舊重播。

**實作**（付費前做，不動 `build_wall.py`）：pick 控制器的 spec 加一個布林值 `llc_note`，只能和 `lookahead: true` 一起用；為 true 時觀察版本是 `pick-3`。預設是 false，舊紀錄的 spec 不變。批次等級 `+pick_safe` 不受影響。harness 另開一個研究名稱 `llc-note-walls`（loopback 用 `llc-note-walls-loopback`）。

## 狀態

9 個新的 7 塊狀態，不能和第 3 步、修法 2 用過的 18 個重複。只能從開發種子 313、484、717 取（720、761 是保留種子）。選法照 `docs/plans/endgame-arena.md` 的同一次洗牌往下取：每個種子跳過已用過的 6 個，再取 3 個狀態不同的。重算後，前 6 個和用過的 18 個完全一致。

| 種子 | 順序 | factorial 牆（cascade 結局） |
|---|---|---|
| 313 | 12403 | 20261109-1237b-TH_T80-pc1 |
| 313 | 41023 | 20261109-1244c-TH_T80-pc1 |
| 313 | 04231 | 20261109-1158-TH_T80-pc1 |
| 484 | 43102 | 20261109-1842-TH_T80-pc1 |
| 484 | 40321 | 20261109-1819-TH_T80-pc1 |
| 484 | 21430 | 20261109-1900b-TH_T80-pc1 |
| 717 | 14302 | 20261110-0022b-TH_T80-pc1 |
| 717 | 13204 | 20261110-0031-TH_T80-pc1 |
| 717 | 41203 | 20261110-0026b-TH_T80-pc1 |

## 對手（免費）

與修法 2 相同，每個狀態 46 面，共 414 面：

- `none`、`random` 1–20、`llc`、`borda`、`low_top`；
- 拿到同樣可行性資訊的對手：`low_top_safe`、`random_safe` 1–20、`pick_safe`。

安全候選與這些規則的定義見修法 2。

## 免費檢查（付費之前全部要過）

1. 單元測試：`pick-1`、`pick-2` 的提示 SHA-256 不變；`pick-3` 只比 `pick-2` 多上面那一段；`llc_note` 不和 `lookahead: true` 一起時，spec 會被拒絕。
2. 舊牆不變：重蓋 factorial、arena、lookahead-arena 各 1 面，和原本完全相同；第 3 步與修法 2 的 18 面付費牆，控制器檢查仍是 PASS。
3. 對手牆：414 面都是 `PROCESS_PASS`，閘門 G1 和 G2 都通過（定義見 `docs/plans/endgame-arena.md`）。
4. loopback：9 面牆在本機假伺服器上走完整個 harness，判讀程式從頭到尾跑通。

## 付費測試（另外批准）

**LLM**　與修法 2 相同（`gpt-6.1-terra`，effort low，不重試，`lookahead: true`，`max_tool_calls` 4），只加 `llc_note: true`。

**上限**　每面牆最多 40 個請求、100k tokens、US$0.20；整批最多 9 面、360 個請求、900k tokens、**US$2**。修法 2 整批花了 US$0.47，這次提示只多一段，估計差不多。

**判定**（依序，只判定一次）

- 停止條件，以及 (a)–(e)：與修法 2 相同（`docs/plans/llm-endgame-lookahead.md`）。
- 另加 (f)，與 (a)–(e) 都成立才算繼續；(f) 不成立，就算 (a)–(e) 都成立也不算繼續：
  - (f) 和 `pick_safe` 比：至少 5 個狀態中，LLM 結局的四項 z 平均不低於 `pick_safe` 的結局，而且放下的石頭不少於 `pick_safe`。
- 結果寫成「繼續」、「停止直接選」或「沒有顯示殘局貢獻」三者之一。

**另外報告**（探索性，不進判定）

- 每個狀態 LLM 和 `pick_safe` 的四項 z 差，特別是 `F_LR`。
- 沒選 LLC 最高安全候選的決策數，以及其中所選候選 NHR 比較高的次數，並與修法 2 的 14/62、11/14 並列。
- 最高 LLC 並列的決策中，LLM 選在並列之內的比例，以及和 `pick_safe` 選得不同的次數。
- 修法 2 報告的各項（工具使用次數、選的候選會不會讓某塊石頭變 0，以及在 `random`、`random_safe` 結局中的百分位）。

**限制**　這批牆和修法 2 用的是不同狀態，所以「加了這段提示有沒有幫助」只能和 `pick_safe` 對照，不能直接和修法 2 的 LLM 比。結果只當探索：仍然只有 `TH_T80`、只有開發種子，而且只能看一步。

## 紀錄

開始、結束時間與檢查結果寫進 `runs/batch/factorial-log.csv`；每個狀態的表寫到 `runs/batch/llm-endgame-llc-note-th80.csv`。
