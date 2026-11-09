# 計畫：用頂面配合度補上長期估計

現行砌牆每一步只用當下分數選石頭，看不到這一步對完工牆的影響。假設決策當下算不起後續回合，本計畫改用一個便宜的估計函數 h 回答一個問題：

> 放下這塊之後，剩下的石頭和新頂面還配不配？

本文件整理問題觀察、解決提案與 h 的候選作法。h 的預測力尚未驗證；文件裡的耗時是實測，預期效果都是假設。

## 結論

- 問題的核心是決策缺少未來資訊，而且多數可行選項根本選不到。當下分數並沒有系統性地誤導，它對長期結果只是接近沒有資訊。
- 建議先做三種 h，由便宜到貴：頂面包絡（H2）、單石試坐（H3）、骨架推演（H4）。任一種的每步成本都低於現行姿態評估的 10%。
- h 要有發揮空間，控制器必須能直接選候選。只挑濾波順序時，120 種順序的中位數只有 3 個不同結果。
- 用現有 74 個分支點就能先篩選 h，不必新跑牆。

## 範圍與資料

| 項目 | 內容 |
| --- | --- |
| 場景 | `TH_T80`（18 塊石頭，內部寬 403 px，理想滿密高度約 111 px） |
| 樣本 | `research/runs` 的 101 面完成牆，依種子與放置序列去重後 72 面、13 個種子 |
| 平衡設計 | `thangavelu-5arms-10seeds` 的 50 面牆（5 種順序 × 10 種子） |
| 因果分支 | 6 個人工挑選的分支點，11 個有實際差異的分支 |
| 分支點 | 72 面牆中 74 個不重複的分歧點，195 對牆 |
| 排除 | Almeida、bricks、FFT 共 17 面牆，以及 verify-perf 牆：沒有順序 × 種子設計 |

未去重時，null 控制器與 Paper 相同、前瞻 recorded、確定性重複等 29 面牆會製造假訊號。例如 LLC 對 `F_LR` 的一致率會從 22/34（65%）膨脹到 84%。下文所有統計都已去重。

### 名詞

| 名詞 | 意思 |
| --- | --- |
| 頂面 `s(x)` | 每一欄最高已佔用像素離地的高度 |
| 候選 | 一個姿態（石頭＋旋轉）在現行定位規則下得到的唯一位置 |
| 可行候選 | LLC>0 且五個分數都是有限值的候選 |
| `s'` | 放下某候選後的頂面 |
| 分支點 | 兩面牆歷史相同，到某一步選了不同候選的位置 |
| 頂面下空隙 | 頂面以下未被石頭佔用的面積 |
| 一致率 | 兩個分支中，某分數較高的一方最後結果也較好的比例 |

`F_SF`、`F_AH`（越低越好）、`F_AV`、`F_LR` 的定義見 `research/walls/wall_metrics.py`。此場景總石面積只佔目標面積 40.9%，放完 18 塊的 `F_SF` 約 0.405，所以 `F_SF` 在這裡只反映有沒有放完。

高度與寬度以模擬格像素計。石塊面積以原圖像素計，是模擬格面積的 4 倍。

### 現行決策

```text
每群抽 1 塊共 7 塊 × 4 旋轉 = 28 個姿態
  → 每個姿態用卷積定出 1 個位置
  → 算五個分數（LLC 最貴）
  → LLC>0 過濾，再依順序做五次平均切（保留 ≥ 平均者）
  → 放下 1 塊，不回頭
  → 無處可放時重抽，最多 5 次，否則結束
```

| 代號 | 程式名稱 | 看什麼 |
| --- | --- | --- |
| LLC | `kinematic` | 放下後整面牆的傾斜承載，四捨五入後只有 6 階 |
| NHR | `course_height` | 石頭頂部和左鄰頂部的高差 |
| IWR | `interlocking` | 與下層接縫的水平距離 |
| GDS | `global_density` | 石頭落在目前最高點以下的比例，低於 0.7 歸零 |
| LFR | `local_void` | 石頭左側的空隙 |

## 問題觀察

本節與 h 耗時的數字，正式紀錄在 `research/findings.md` 的 Temporal myopia 一節，兩處不一致時以該節為準。

### 目標沒有單一方向

| 指標（72 面牆） | 範圍 | 平均 | 標準差 |
| --- | --- | --- | --- |
| `F_LR` | 0.087–0.87 | 0.311 | 0.161 |
| 牆高 px | 161–258 | 199.6 | 23.75 |
| 放滿 18 塊 | 62/72 面 | | |

- `F_LR` 與牆高幾乎無關（相關 0.02），`F_AV` 與牆高強負相關（−0.71）。50 面平衡牆中有 17 面位於 `F_SF`、`F_AH`、`F_AV`、`F_LR` 的 Pareto 前緣。
- 11 個因果分支中，`F_LR` 變好 8 個，完成數 +1 有 7 個，但牆高變差也有 8 個，`F_AV` 變差 7 個，`F_AH` 變差 8 個。同一次改動常常無法單純判定好壞。

### 濾波順序影響小，種子與隨機互動影響大

50 面平衡牆的變異數分解：順序主效果佔 4–17%，種子佔 14–41%，順序 × 種子殘差佔 53–74%。沒有順序在 `F_LR` 上贏過 Paper。`code_original` 的 `F_AV` 高 4.4（8/10 種子，p=0.04，未校正）。

### 當下分數對長期結果接近沒有資訊

74 個分支點的一致率（非因果，分歧後兩面牆的政策也可能不同）：

| 分數 | 對 `F_LR` | 對牆高 |
| --- | --- | --- |
| LLC | 22/34（65%，p=0.12） | 16/34 |
| NHR | 34/73 | 42/73 |
| IWR | 22/49 | 24/49 |
| GDS | 30/57 | 28/57 |
| LFR | 34/74 | 37/74 |
| 五項總和 | 36/74 | 35/74 |

- 對完成數，五項總和較高反而較差（7/27，p=0.019），LFR 較高則較好（21/27，p=0.006）。這是 30 個探索性檢定中的結果，未校正。
- LLC 同分的 40 個分支點，其他分數的一致率介於 35% 到 63%，沒有穩定方向。
- 74 點中 51 點在第 6 步（含）以前，只有 5 點在第 13 步（含）以後。

這些數字表示反轉或調整當下分數救不了問題，需要新資訊。

### 多數可行選項選不到

每步可行候選中位數 14（平均 13.4），120 種順序只能到達中位數 3 個（平均 3.24）。15.8% 的步驟只有 1 個可達結果，71.8% 的選擇落在 LLC 最大值。

| 可行石塊被至少一種順序選到的比例 | 第 1–6 步 | 第 7–12 步 | 第 13–18 步 |
| --- | --- | --- | --- |
| 面積最大 5 塊 | 22.5% | 26.9% | 59.7% |
| 其餘 13 塊 | 40.4% | 55.6% | 75.2% |
| 石 13 | 22.9% | 21.2% | 52.2% |

平均切要求候選在每一關都高於平均。第 1–12 步，石 13 可行 398 次，實際被選只佔約 5%。

### 大石被延後

面積與平均放置順序 ρ=0.61（p=0.007），最大 5 塊平均正是最後 5 塊，83% 的牆呈現大石靠後。每步平均放置面積呈 U 形：第 1–2 步約 11.4k px，第 6 步 6.5k，第 18 步 15.2k。

因果未證實。最大 5 塊放得越晚，牆高越高（ρ=+0.24）、`F_AH` 越差（+0.30），但完成數也越多（+0.25）。不能直接推出「大石先放」。

### 最終品質在尾盤結算

- 完工牆比理想高度多 89 px，其中頂面不平 65 px、頂面下空隙 24 px。
- 最高點由最後一塊決定的有 45/72，由倒數第二塊決定的有 19/72，合計 89%。
- 第 13 步時（62 面完成牆）最高點 114.6 px，平均高 75.0 px；完工牆高 200.3 px。最後 6 塊貢獻 85.7 px，佔 43%。
- 可行姿態數在中段約 16–19 個，第 13–18 步依序為 12.7、9.8、8.0、5.6、3.6、1.9。
- 可行比例由第 4 步的 0.67 降到第 18 步的 0.46。選中候選的 LLC 由第 2 步的 0.98 降到第 17–18 步的 0.33。
- 10 面未完成牆留下 12 塊未放：石 13 六次、石 8 三次、石 16 兩次、石 9 一次。

自由度最少的時候，決策的影響最大，而且無法回頭。

### 後果延遲且不穩定

單次分歧造成的最終差異（中位數）已接近整體標準差：

| 分歧步 | \|Δ`F_LR`\| | \|Δ牆高\| px |
| --- | --- | --- |
| 1–3 | 0.10 | 14.9 |
| 4–6 | 0.14 | 13.5 |
| 7–12 | 0.12 | 22 |
| 13–18 | 0.17 | 15 |

這些差異包含分歧後的政策差異。具體案例：

- 種子 762733368 第 7 步放小石 8（面積 1448）而非石 4（18866）。當下分數差為 LLC −0.517、IWR −0.854、GDS +1.0，當下看來明顯較差。完工牆高 234→184，`F_LR` 0.39→0.54，`F_AV` 13.3→23.6，由 17 塊變成放滿 18 塊。
- 種子 184947496 第 10 步，三個改放石 19 的分支使牆高 +36、+52、+61，`F_AV` −1.9、−5.6、−10.1。
- `gpt-001` 的反事實中，`k17` 的 `F_LR` 差值到最後一塊才由負轉正，`k8` 的差值翻號 5 次（`research/reports/gpt-001-cf-steps/README.md`）。

短前瞻在 `research/findings.md`（P1）沒有穩定勝過不前瞻，符合這個現象：差值會中途翻號，停在第 2 或第 4 步看方向不可靠。此處的差異有多少是真實結構、多少是混沌，目前不確定。

### 頂面特徵有訊號，但來得晚

同種子內的 Spearman 相關，特徵取自第 T 步，對象是完工結果（n=72，探索性，共 350 個相關，未校正）：

| 特徵 | 對象 | T=3 | 6 | 9 | 12 | 15 |
| --- | --- | --- | --- | --- | --- | --- |
| 頂面不平（最高點−平均高） | 牆高 | −0.17 | +0.12 | 0.00 | +0.22 | +0.26 |
| 頂面不平 | `F_AV` | −0.11 | −0.34 | −0.35 | −0.28 | −0.36 |
| 頂面不平 | `F_AH` | −0.03 | +0.05 | +0.17 | +0.36 | +0.38 |
| 頂面不平 | `F_LR` | −0.16 | −0.07 | −0.11 | −0.08 | −0.04 |
| 頂面下空隙 | `F_AV` | −0.10 | −0.09 | −0.32 | −0.33 | −0.34 |
| 最高點 | 牆高 | −0.24 | +0.05 | +0.04 | +0.26 | +0.32 |
| GDS 累積平均 | 牆高 | +0.08 | −0.18 | −0.25 | −0.40 | −0.42 |
| GDS 累積平均 | `F_AV` | +0.22 | +0.35 | +0.37 | +0.34 | +0.35 |
| LLC 累積平均 | `F_LR` | +0.20 | +0.26 | +0.27 | +0.32 | +0.38 |
| IWR 累積平均 | `F_LR` | −0.21 | +0.05 | −0.31 | −0.30 | −0.34 |

- 這些是不同牆之間的比較。h 要排序同一狀態的候選，條件更嚴格。
- 頂面特徵與 `F_LR` 沒有穩定關聯。
- 所有特徵都只描述牆，沒有描述牆與剩餘石頭的匹配。`research/findings.md`（P0）的單步預覽也是只看牆，決策層級的方向一致率接近各半。

### 完整預演的成本

以記錄路徑的可行數當分支數，窮舉尾盤的姿態評估次數：

| 搜尋最後幾塊 | 葉節點中位數 | 評估次數中位數 | 90 百分位 | 10 核估計時間 |
| --- | --- | --- | --- | --- |
| 3 | 34 | 142 | 228 | 數秒 |
| 4 | 284 | 1,168 | 2,284 | 12–23 秒 |
| 5 | 2,918 | 10,900 | 27,542 | 2–4 分鐘 |
| 6 | 32,130 | 148,266 | 402,603 | 25–50 分鐘 |

估計假設每姿態 0.1–0.2 秒 CPU（`time.txt` 已計時部分平均 0.097 秒，卷積未計時），未含狀態複製。`B=4`、每步 5 候選的全程 beam search 約 78,000 次評估，每面牆約 13–26 分鐘。

### 困難點總結

| 困難 | 主要證據 | h 必須做到 |
| --- | --- | --- |
| 目標有四項且互相取捨 | 因果分支中 `F_LR` 變好時 `F_AV`、`F_AH` 常變差 | 四項分開預測、分開檢定 |
| 動作範圍窄 | 中位數 14 個候選只到得了 3 個 | 搭配可直接選候選的策略 |
| 沒有未來訊號 | 當下分數一致率約 50% | 使用剩餘石頭的資訊 |
| 尾盤結算 | 89% 的最高點由最後兩塊決定 | 尾盤權重要提高 |
| 延遲且不穩定 | 單次分歧差異接近整體標準差 | 可能只在中後段有效 |
| 證據薄 | 單一場景、13 個種子，尾盤分支點 5 個 | 事先定檢定，保留新種子 |

## 解決提案

### 從 A* 借什麼

| A* 概念 | 這裡的對應 | 可用性 |
| --- | --- | --- |
| 放寬問題得到 h | 拿掉形狀、石頭互動或力學 | 直接採用，是所有作法的來源 |
| `g`：已花成本 | 目前最高點 | 只對牆高成立 |
| `h`：剩餘成本 | 還會再長多少 | 只對牆高成立；`F_LR` 是完工牆的最小值，沒有累加結構 |
| 可採納：`h` 不高估 | 下界 | 只在垂直落下假設下成立，17.4% 的實際放置低於垂直落下高度 |
| 展開搜尋 | 每展開一個節點要評估 28 個姿態 | 第一版不展開，h 只看一步 |

不需要 A* 的最佳解保證。h 只要在同一狀態的候選間排對順序，共同誤差會抵消，絕對值不準也無妨。

### h 的規格

**輸入**　`s'`、全部剩餘石塊的 4 個姿態、候選新增的頂面下空隙、已放面積比例。18 塊石頭都已知，每步抽 7 塊只影響當步能選誰。

**輸出**　一個數字，越大越好，只在同一狀態的候選之間可比較。

**不可做**　不呼叫 `evaluate_kine`、不讀或改 RNG、不改牆物件。

**預算**　14 個候選合計不超過 0.27 秒，約為現行每步姿態評估 CPU 的 10%（28 × 0.097 秒）。

**預測對象**　主要是 `F_SF`、`F_AH`、`F_AV`、`F_LR` 四項，與 LLM agent 的最後評估一致（`docs/plans/simulation-to-llm-agent.md` E3）；次要是牆高與完成數。H2–H4 是為頂面幾何設計的，它們與四項的關係目前只有間接證據：`F_AV` 與牆高相關 −0.71，頂面不平在第 12、15 步與 `F_AH` 相關 +0.36、+0.38，`F_SF` 在此場景只反映有沒有放完，頂面特徵與 `F_LR` 沒有穩定關聯。所以 `F_LR` 最可能是 h 預測不到的一項，LLC 門檻繼續負責當步的穩定度。

### 併入決策的方式

| 方式 | 作法 | 可選範圍 | 評估 |
| --- | --- | --- | --- |
| A 挑順序 | 對 120 種順序的結果算 h，選最高者 | 中位數 3 | 不必改 Wang 的程式，現有控制器介面就能做，但上限低；當第一個實牆對照 |
| B 第六個濾波 | 加一道平均切 | 再縮小 | 不採用，會加重選項被砍的問題 |
| C 門檻後排序 | LLC>0 的候選全部按 h 排序 | 中位數 14 | 建議的主要作法 |
| D 混合 | 當下分數標準化後加 `β(φ)·h` | 中位數 14 | C 的延伸，保留當下品質 |
| E 預篩 | 先按 h 排序，依序算 LLC 直到得到 m 個可行候選 | 中位數 14 | 可省下 LLC 計算（佔已計時部分約 63%）；決策與舊牆不同 |

`φ` 是已放面積比例，取代步數，換場景時不必重設。

### 路線

| 階段 | 內容 | 通過條件 |
| --- | --- | --- |
| 0 定裁判 | 預測對象見「h 的規格」；比較一律用相同的姿態評估次數 | 已決定 |
| 1 基準 | 多次重啟：同一種子不同隨機流各建 N 面，取最佳 | 建立對手 |
| 2 放寬動作 | 新增可直接選候選的策略；既有策略逐位元相同 | `check_run.compare_runs` 零差異 |
| 3 離線篩選 | 用既有分支點檢驗 h | 見驗證計畫 A |
| 4 因果驗證 | 從同一狀態強制走高 h 與低 h 候選 | 見驗證計畫 B |
| 5 建牆比較 | h 策略對基準與 Paper | 見驗證計畫 C |
| 6 學習 | 累積結果後學權重或價值函數（H6） | 留一種子交叉驗證勝過手工 h |

多次重啟的歷史牆抽樣估計（62 面完成牆，5000 次 bootstrap，以 `F_LR` 與牆高 z 分數和選牆）：

| N | `F_LR` | 牆高 px |
| --- | --- | --- |
| 1 | 0.328 | 200.6 |
| 5 | 0.474 | 182.6 |
| 10 | 0.551 | 180.8 |

這是在共用歷史樣本上的估計，不同種子的難度也混在其中，可能偏樂觀。尾盤窮舉（評估次數見上表）只當少量樣本的校準工具，用來知道 h 離真實最佳有多遠，不進入決策迴圈。

## h 的作法

### 比較

耗時為單一行程、一面記錄牆上的實測，每次呼叫是一個候選。三個數字對應剩餘 17／11／5 塊。

| 作法 | 需要剩餘石頭的什麼 | 每候選 ms | 14 候選合計 | 主要盲點 |
| --- | --- | --- | --- | --- |
| H1 流體水位 | 總面積 | 0.02／0.02／0.02 | <1 ms | 看不到形狀 |
| H2 頂面包絡 | 寬度 | 0.12／0.11／0.11 | <2 ms | 不知道尾盤要放誰 |
| H3 單石試坐 | 全部姿態 | 11.2／7.4／3.4 | 0.05–0.16 秒 | 石頭互相搶位 |
| H4a 骨架推演（大石優先） | 全部姿態 | 11.9／7.3／3.6 | 0.05–0.17 秒 | 穩定度 |
| H4b 骨架推演（最佳配合） | 全部姿態 | 105／43／12 | 0.17–1.5 秒 | 太貴，只當參考 |
| 二維卷積版試坐 | 全部姿態 | 277／202／103 | 1.4–3.9 秒 | 超出預算，不採用 |

H5（分段組合）與 H6（學習）建立在上述作法之上。凸包不採用：整面斜坡沒有凹陷，會被判定為平。

### 共用元件

一維模型假設石頭垂直落下。對 72 面牆 1,198 次實際放置的檢查：

- 82.6% 的實際高度與垂直落下完全相同。
- 17.4% 低於垂直落下高度（第 5 百分位 −50 px，最低 −174 px），也就是石頭鑽到了懸挑下方。因此頂面下空隙不全是封死的，下界性質也不嚴格成立。
- 94.6% 的實際放置通過下面的 `supported` 檢查（重心落在最外側兩個接觸欄之間）。

實作時，頂面計算併入 `research/analysis/placement_preview.py` 與 `research/analysis/unplaced_stones.py` 已有的邏輯，一個概念只留一份。

```python
from typing import NamedTuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import grey_closing

GROUND_ROWS = 2
IGNORE_VALUE = 255
CONTACT_TOLERANCE = 2
GOOD_VOID_RATIO = 0.25
VOID_WEIGHT = 1.0


class Pose(NamedTuple):
    """Column profiles of one stone pose, rows counted upward from the box bottom."""

    bottom: np.ndarray
    top: np.ndarray
    center: float
    area: int


def skyline(wall):
    """Height above the ground of the highest occupied pixel in each inner column."""
    inner = (wall[:, 1:-1] != 0) & (wall[:, 1:-1] != IGNORE_VALUE)
    return inner.shape[0] - np.argmax(inner[::-1], axis=0) - GROUND_ROWS


def make_pose(mask):
    """Column profiles of a stone mask cropped to its box, rows counted upward."""
    columns = np.arange(mask.shape[1])
    center = float((mask.sum(axis=0) * columns).sum() / mask.sum())
    return Pose(mask.argmax(axis=0), mask.shape[0] - mask[::-1].argmax(axis=0), center, int(mask.sum()))


def stone_poses(mask):
    """The four right-angle poses of one stone."""
    return [make_pose(np.rot90(mask, k)) for k in range(4)]


def place(s, x, y, pose):
    """Skyline after a stone box lands at column x, height y; also the void it seals."""
    end = x + len(pose.bottom)
    sealed = int((y + pose.bottom - s[x:end]).clip(0).sum())
    new = s.copy()
    new[x:end] = np.maximum(new[x:end], y + pose.top)
    return new, sealed


def seats(s, pose):
    """All vertical-drop seats of one pose: rest height, sealed void, top, supported flag."""
    window = sliding_window_view(s, len(pose.bottom))
    rest = (window - pose.bottom).max(axis=1)
    gap = rest[:, None] + pose.bottom - window
    contact = gap <= CONTACT_TOLERANCE
    left = contact.argmax(axis=1)
    right = contact.shape[1] - 1 - contact[:, ::-1].argmax(axis=1)
    return rest, gap.sum(axis=1), rest + pose.top.max(), (left <= pose.center) & (pose.center <= right)
```

### H0 基準：現有訊號

任何新 h 都要贏過這兩個便宜對手，否則只是重寫現有分數：

- 候選的 GDS（已計算）。
- 放下後的頂面不平度：`s'` 的最高點減平均高。

### H1 流體水位

**想法**　把剩餘石塊當液體倒在 `s'` 上，水位越低越好。

```python
def fluid_level(s, area):
    """Water level reached by pouring area onto skyline s."""
    rising = np.sort(s)
    level = (area + rising.cumsum()) / np.arange(1, len(s) + 1)
    return float(level[np.argmax(level <= np.append(rising[1:], np.inf))])
```

**分數**　`h1 = -max(s'.max(), fluid_level(s', A_rem))`，`A_rem` 是剩餘面積。

**看得到**　頂面下空隙、已經超過水位的尖點。

**看不到**　完工牆超高 89 px 中，頂面不平的 65 px。H1 最多只反映空隙的 24 px，所以只當其他作法的附加項。

### H2 頂面包絡

**想法**　放下後，頂面離平整還差多少，以及那些空著的地方石頭進不進得去。

```text
peak ................................#######
     ................................#######
     ................#######.........#######
     ........######xx#######.........#######
     ##############xx#######.........#######
     ###########################################

# 已建牆體　. 最高點以下的開放面積 G　x 窄於參考寬度的死角 Dead（也算在 G 內）
```

```python
def surface_terms(s, width):
    """Open area below the current peak, and the part of it in pockets narrower than width."""
    peak = int(s.max())
    padded = np.pad(s, width, constant_values=peak)
    closed = grey_closing(padded, size=width)[width:-width]
    return int((peak - s).sum()), int((closed - s).clip(0).sum())


def envelope_h(s_new, sealed, width, dead_weight=1.0):
    """Larger is better: little open area, few dead pockets, little sealed void."""
    open_area, dead = surface_terms(s_new, width)
    return -(open_area + dead_weight * (dead + sealed))
```

`h2 = -[G(s') + a·(Dead(s') + ΔV)]`，`ΔV` 是候選在自己下方新增的頂面下空隙。

**為什麼這樣定義**

- 候選改變 `G` 的量恰為 `W·δ − Σ(s'−s)`，`W` 是牆寬，`δ` 是最高點上升量。它自動懲罰長高，並獎勵在最高點以下填入面積。不用另設「不要長高」的項。
- 量標準差或粗糙度時，大石造成新台階會被扣分，選擇偏向小石，可能加重大石延後。量面積則大石只要不超過最高點就得高分。
- 石頭跨過凹陷時 `Σ(s'−s)` 會包含被蓋住的空隙，等於把空隙算成填平。`a≥1` 才能抵銷。

**變體**

| 變體 | 定義 | 用途 |
| --- | --- | --- |
| H2a 水平封頂 | `a=0`，只用 `G` | 最簡單的對照 |
| H2b 石寬包絡 | 參考寬度取剩餘石頭窄邊寬的 25% 分位數 | 建議的主要版本 |
| H2c 雙尺度 | 再加一個以寬邊寬為尺度的死角項，權重較小 | 區分「只能直立放進去」與「平放得進去」 |

此場景的窄邊寬為 9–62 px（25% 分位數 32，中位數 41.5），寬邊寬為 39–129 px（25% 分位數 61.5，中位數 88）。不用最小值，因為石 8 的窄邊只有 9 px，死角幾乎永遠是零。參考寬度隨剩餘石頭改變，H2 因此間接知道還剩哪些石頭。

**看不到**　最後幾塊的具體形狀；頂面太平是否減少咬合（未測）。

### H3 單石試坐

**想法**　每塊剩餘石頭各自在 `s'` 上試所有姿態與位置，看它還有沒有好位置。對應限制滿足問題的兩個經典原則：選擇留給別人最多選項的值，並優先關注選項最少的變數。

```python
def stone_fit(s, poses):
    """Good-seat count, least sealed void and lowest top over the supported seats of one stone."""
    count, void, low_top = 0, np.inf, np.inf
    for pose in poses:
        _, sealed, top, supported = seats(s, pose)
        if supported.any():
            count += int((sealed[supported] <= GOOD_VOID_RATIO * pose.area).sum())
            void = min(void, sealed[supported].min())
            low_top = min(low_top, top[supported].min())
    return count, void, low_top


def fit_scores(s, remaining):
    """Three aggregations of stone_fit over the remaining stones; larger is better."""
    if not remaining:
        return dict(worst_top=-float(s.max()), scarcity=0.0, void=0.0)
    count, void, low_top = np.array([stone_fit(s, poses) for poses in remaining], dtype=float).T
    areas = np.array([poses[0].area for poses in remaining], dtype=float)
    tallest = np.array([max(pose.top.max() for pose in poses) for poses in remaining], dtype=float)
    low_top = np.where(np.isfinite(low_top), low_top, s.max() + tallest)
    void = np.where(np.isfinite(void), void, areas)
    return dict(
        worst_top=-float(low_top.max()),
        scarcity=-float((areas / (1 + count)).sum() / areas.sum()),
        void=-float(void.sum()),
    )
```

| 變體 | 分數 | 含義 |
| --- | --- | --- |
| H3a 最難石 | `worst_top` | 最難放那塊的最低可能頂高 |
| H3b 稀缺加權 | `scarcity` | 面積大又沒位置的石頭扣最多；建議的主要版本 |
| H3c 空隙成本 | `void` | 全部石頭最佳坐位的空隙和 |

`GOOD_VOID_RATIO` 初值 0.25，實際放置的空隙／面積中位數 0.17、90 百分位 0.585，要用資料校準。

**看不到**　石頭之間互相搶位；坐下後頂面改變。

**注意**　`research/findings.md`（P2）的難放石排序在分支點沒有選中真正有幫助的石頭。該排序用的是過去的失敗與當下姿態數，H3 量的是放下候選之後的未來坐位，兩者不同，但這個前例說明稀缺訊號不一定有效。

### H4 骨架推演

**想法**　把剩餘石頭依序丟到一維頂面上，不算力學，直接得到預估牆高與放不放得完。它仍是往後看，但不做真實姿態評估。

```python
def best_seat(s, poses):
    """Cheapest supported drop of one stone: (cost, column, height, pose), or None."""
    best = None
    for pose in poses:
        rest, sealed, top, supported = seats(s, pose)
        if not supported.any():
            continue
        cost = np.where(supported, top + VOID_WEIGHT * sealed / len(pose.bottom), np.inf)
        k = int(cost.argmin())
        if best is None or cost[k] < best[0]:
            best = (float(cost[k]), k, int(rest[k]), pose)
    return best


def rollout(s, remaining):
    """Drop stones largest first without mechanics: (final height, unplaced stones, sealed void)."""
    unplaced, sealed_total = 0, 0
    for poses in sorted(remaining, key=lambda p: -p[0].area):
        seat = best_seat(s, poses)
        if seat is None:
            unplaced += 1
            continue
        s, sealed = place(s, seat[1], seat[2], seat[3])
        sealed_total += sealed
    return int(s.max()), unplaced, sealed_total
```

**分數**　依序比較：未放數較少、預估牆高較低、空隙較少。

| 變體 | 下一塊怎麼選 | 成本 |
| --- | --- | --- |
| H4a 大石優先 | 固定由面積大到小 | 與 H3 同級 |
| H4b 最佳配合 | 每次在所有剩餘石頭中選最便宜的坐位 | 剩 17 塊時每候選約 105 ms，超出預算 |

**看不到**　穩定度。實際放置還有 17.4% 會低於垂直落下。推演的策略也不是真實的濾波級聯，所以它估的是「用這個簡單規則接著建」的結果，不是現行策略的結果。

### H5 分段組合

前中段看頂面，尾盤看剩餘石頭：

```text
w(φ)  = clip((φ − φ0) / (φ1 − φ0), 0, 1)
h5    = (1 − w(φ)) · z(H2b) + w(φ) · z(H3b 或 H4a)
```

`z` 是同一狀態候選間的標準化，`φ` 是已放面積比例。`φ0`、`φ1` 的初值由 TH_T80 第 6 步與第 13 步的平均 `φ` 換算，要驗證後才定。依據是第 3 步幾乎沒有訊號，第 13 步起可行姿態快速減少。

### H6 學習權重或價值函數

**想法**　決策當下算不起未來，但離線可以一次付清，之後只讀模型。

**目標**　從此狀態起照現行規則建完的結果。若估計準確，照它選不會比現行規則差；有誤差時沒有這個保證。

**特徵**　H1–H4 的輸出、`φ`、候選當下五個分數。

**模型順序**　先用少量權重的排序模型，以分支對的方向為標籤；資料足夠後才考慮樹模型或卷積網路。

**資料**　既有 195 對牆（非因果）、驗證計畫 B 的因果分支、尾盤窮舉的葉節點。

**驗證**　留一種子交叉驗證，同一種子不可同時出現在訓練與測試。72 面牆不足以訓練深度模型。`F_LR` 能否學到未知。

## 驗證計畫

### A 離線篩選（不新跑牆）

**資料**　74 個分支點。每個分支放下後的 `s'` 取自下一個事件的 `wall`，剩餘石頭取自 `sampled_ids` 與 `inventory_ids`，只讀 trace。

**主要檢定**　事先固定 H2b、H3b、H4a 對 `F_SF`、`F_AH`、`F_AV`、`F_LR` 共 12 個檢定，以 Holm 校正。每個檢定只用該指標有差的分支點（雙尾二項檢定）。

| 有差的分支點 | α=0.05 需要一致 | α=0.0042（12 個檢定中最嚴格一關） |
| --- | --- | --- |
| 74 | 46（62%） | 50（68%） |
| 27 | 20（74%） | 22（81%） |

**通過條件**　某一項至少有一個 h 通過 Holm 校正，且一致率高於 H0 的兩個基準。通過的 h 只對通過的那一項算數。

**次要分析**　牆高、完成數（同樣三個 h，不校正，只報告）；依階段（第 1–6、7–12、13–18 步）分開；檢查種子 762733368 第 7 步 h 是否偏好石 8。

**限制**　這些分支點不是因果設計，第 13 步以後只有 5 點，尾盤結論要靠 B。

### B 因果分支

選 10 個種子，在第 4、8、12、14、16 步各取一個狀態，強制走最高 h、最低 h 兩個候選，之後都用 Paper 政策建完。共 100 條分支，平均每條約 7 塊，按 2–16 秒／塊估計約 24 分鐘至 3.2 小時（2 秒／塊需持久 Pool，見 `docs/plans/case-speedup-2s-per-stone.md`）。

**通過條件**　在 A 通過的那一項上，最高 h 勝過最低 h 的分支比例達到與 A 相同的校正門檻，且效果隨階段的變化可解釋。

### C 建牆比較

| 項目 | 設定 |
| --- | --- |
| 組別 | Paper、`code_original`、多次重啟（以四項標準化後的和挑最佳）、h 配方式 A、h 配方式 C 或 D |
| 種子 | `docs/plans/simulation-to-llm-agent.md` S0 的保留種子 |
| 預算 | 相同的姿態評估次數；h 的時間另記 |
| 指標 | 主要 `F_SF`、`F_AH`、`F_AV`、`F_LR`；次要牆高、完成數 |
| 統計 | 與 `docs/plans/simulation-to-llm-agent.md` E3 相同：四項都不輸 δ = 0.5 SD 才算不輸；另報 Pareto 支配與 hypervolume |

**通過條件**　在保留種子上，h 策略四項都不輸 Paper，且至少一項贏過同預算的多次重啟。

### 停止條件

| 情況 | 判斷 |
| --- | --- |
| A 全部不過，且一致率都不超過 55% | 頂面與剩餘石頭的匹配資訊不足，回頭檢查資料與定義，不進 B |
| A 過、B 不過 | 相關不是因果，先做尾盤窮舉校準 |
| B 只在第 12 步以後有效 | 早期有混沌上限，h 只用於中後段 |
| C 贏不過多次重啟 | 不推進 h；改以多次重啟為主，把 h 當重啟的初篩 |
| 一項變好，但另一項（最可能是 `F_LR`）不輸檢定失敗 | 把該項的訊號（`F_LR` 是 LLC）納入 h 的分數 |

## 實作位置

| 項目 | 位置 |
| --- | --- |
| h 的純函式 | `research/analysis/surface_fit.py`，共用既有的 `candidate_pixels`、`paste` 與頂面邏輯 |
| 離線篩選 | `research/analysis/` 的分析腳本，只讀 trace |
| 方式 A | `research/controller/order_controller.py`，走現有 `controlled_policy` 介面；控制器只收到庫存的 id，石塊形狀要另外載入 |
| 方式 C、D、E | 新增獨立策略路徑，修改 `Stonepacker2D/stack/selection.py` 與 `build_wall.py:419-425`，只動必要的行 |
| 不變條件 | 既有策略、順序、種子的牆，`check_run.compare_runs` 回報零差異 |
| 紀錄 | 每個候選的 h 各項寫入 trace |
| 新牆 | 經 `run_wall.run` 寫到 `research/cases/` |
| 測試 | `research/tests/`，涵蓋本文件列出的恆等式與邊界情況 |

## 風險

- **早期排序有上限。** 單次分歧差異接近整體標準差，若主要是混沌，h 只在中後段有效。
- **分布偏移。** h 在現行規則走過的狀態上驗證，實際用 h 選之後會走到沒見過的狀態。階段 B 與 C 用來偵測。
- **穩定性盲點。** h 可能偏好幾何漂亮、實際不穩的位置。LLC 門檻只擋得住當步。
- **垂直落下假設。** 17.4% 的實際放置不符合，鑽到懸挑下的機會會被低估。
- **目標取捨。** 四項都要不輸，h 若只改善頂面幾何，可能傷 `F_LR`，因果分支已出現這種取捨。
- **場景依賴。** 所有數字來自單一場景的 18 塊石頭，參數換場景時需重新校準。

## 不建議現在做

- 繼續調濾波順序或權重：順序主效果只佔 4–17%。
- 短步數前瞻：`research/findings.md`（P1）沒有穩定勝過不前瞻，成本又高。
- 直接訓練深度價值網路：72 面牆遠遠不足。
- 只給 LLM 更多資訊而不放寬動作：可選結果仍只有中位數 3 個。

## 決策

已決定，見 `docs/TODO.md`「已決定」。

## 參考

- Wang et al., *European Journal of Operational Research*, 2024：本專案的砌牆方法。
- Hart, Nilsson, Raphael, 1968：A* 與可採納的 heuristic。
- Pearl, *Heuristics*, 1984：由放寬問題產生 heuristic。
- Bertsekas, Tsitsiklis, Wu, 1997：組合最佳化的 rollout 演算法。
- Burke, Kendall, Whitwell, 2004：矩形排樣的 best-fit 與 skyline 類啟發式。
- Russell, Norvig, *Artificial Intelligence: A Modern Approach*：限制滿足問題的選值與選變數原則。
