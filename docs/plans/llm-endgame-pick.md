# TH_T80 殘局的 LLM 直接選（新計畫第 3 步）

讓 LLM 在第 1 步（`docs/plans/endgame-arena.md`）的 9 個殘局狀態裡直接選石頭，每個狀態蓋 1 面牆，和同一狀態的 cascade、20 個隨機結局與 `low_top` 規則比。這是第一個付費步驟。本文在送出任何付費請求之前 commit；之後要改，另開 commit，報告時列出差異。

## 牆

**狀態**　與第 1 步相同的 9 個狀態（種子 313、484、717，各 3 個順序），每個狀態 1 面牆，共 9 面。每面牆從第 1 塊重蓋，用該狀態的順序，接上 `research/controller/pick.py` 的控制器（kind `pick`）。控制器只在庫存為空、可行候選至少 2 個時出手，所以前 11 塊和 factorial 牆相同。

**LLM**　OpenAI `gpt-6.1-terra`，effort low，不重試；規格與每面牆、整批的上限寫在 `research/llm_walls/pick_wall.py`（`SPEC`、`WALL_LIMITS`、`EXPERIMENT_LIMITS`）。整批花費上限 **US$2**。

**一面牆算數**　只有在：case `PROCESS_PASS`、控制器檢查 PASS、沒有用量不明的請求、worker 都已回收、在每面牆的上限內、輸出檔裡沒有金鑰。任何一面不是 finished（uncertain 或 aborted），整批停止，先檢查，不做下面的判定。

**完整**　殘局每個決策點都由 LLM 決定（沒有 skip、沒有 fallback）的牆是完整的。不完整的牆保留並報告，但在下面 (a)、(b) 兩項中，該狀態算未達成。

## 指標

與第 1 步相同：模擬網格上的 F_SF、F_AH、F_AV、F_LR，換成該種子 factorial TH_T80 區組 120 面牆的 z（F_AH 取負，+ 代表較好），每個結局的分數是四項 z 的平均；另報各項 z 與放下的石頭數。cascade、20 個隨機結局和三條規則的結局直接用第 1 步的牆（`runs/batch/endgame-arena-s<種子>-TH_T80`），不重蓋。

## 判定（依序，只判定一次）

**停止直接選**（先判斷）：至少 5 個狀態中 LLM 結局低於同狀態 20 個隨機結局的中位數，或至少 3 個狀態中 LLM 放下的石頭比 cascade 少。→ 停止讓 LLM 直接選；「和 Wang 一樣好」改用受限選單或只選順序的 agent。

**繼續**：下面三項都成立。

- (a) 至少 7 個狀態中，LLM 結局的四項 z 平均嚴格高於 cascade，而且放下的石頭不少於 cascade（單尾二項 p ≈ 0.09）；
- (b) 至少 5 個狀態中，LLM 結局的四項 z 平均不低於 `low_top` 規則的結局；
- (c) 9 面牆所有由 LLM 決定的決策中，至少 1/4 選的候選和 cascade 不同。

**介於兩者之間**：不屬於上面兩種，記為「沒有顯示殘局貢獻」。

## 另外報告

每個狀態：LLM 結局在 20 個隨機結局中的百分位、與 cascade 和三條規則的差、各項 z 與石頭數。全部：和 cascade 不同的決策比例、fallback 與 skip 的比例、每次決策實測的 token、費用與秒數。

9 個狀態都來自同一組石頭、只有 TH_T80，結果在任何情況下都只當探索。

## 紀錄

開始、結束時間與檢查結果寫進 `runs/batch/factorial-log.csv`，每狀態的表寫到 `runs/batch/llm-endgame-pick-th80.csv`。
