---
name: concept-map
description: 黑盒層級的專案架構地圖：討論架構時停在 L0/L1，點名才展開 L2/L3，不鑽進細節；討論中確定的變動先改地圖再繼續；產生可縮放的 viewer.html 與現況對照。所有產物放專案的 .claude/concept-map/。使用者說「討論架構」「黑盒」「概念地圖」「架構圖」「展開 X」「對一下現況」「/concept-map」時使用。
argument-hint: [init [專案路徑] | 討論主題 | 展開 <盒子> | 縮回 | 畫 | 對一下現況 | 建立]
---

# /concept-map

用一份人寫的 Markdown 地圖（盒子＋契約＋連線）當唯一來源，討論時控制深度，並即時畫成可縮放的圖。
不依賴 Graphify、NetworkX；只用 `scripts/build.py`（Python 標準函式庫）。

## 產出規則（最重要）

- 所有產物只放 `{專案根}/.claude/concept-map/`：`model.json`、`viewer.html`、`drift.md`。不寫到別處。
- 地圖本身（人寫的 Markdown）是專案文件，位置由專案決定，不放進上述資料夾。
- 第一次產生後，確認專案 `.gitignore` 有 `.claude/concept-map/`；沒有就告訴使用者，不擅自改共用的 `.gitignore`。
- 不發佈成 Artifact（會放到 claude.ai）；使用者明說才做。

## 步驟 0：找地圖

1. 專案 `CLAUDE.md` 有 `概念地圖：<路徑>` 這行就用它。
2. 否則在專案根找 `architecture-map.md`、`ARCHITECTURE.md`、`docs/**/architecture*.md`。
3. 都沒有：提議建立。掃頂層資料夾、README、入口檔，照 `template.md` 起草 L0 和 L1，看不出來的標「待確認」；**先給使用者看草稿，同意才存檔**。
4. 使用者下 `/concept-map:init`（或 `/concept-map init`）：不走第 3 點的逐步確認，改照 `INIT.md` 一次做完（盤點 `scripts/survey.py`、起草、存檔、build、開圖、回報待確認）。

## 討論時的深度規則

- 預設停在 L1。每次回答開頭標一行：`目前層：L1 / 盒子：—`。
- 使用者說「展開 X」才進 X 的 L2；其他盒子維持黑盒。一次只下一層。「縮回」退一層。
- 碰到公式、參數、kernel、函式細節就停，問：「這屬於 L3，要往下嗎？」
- L3 先用 viewer 或 `model.json` 自動列出的實際檔案，不先讀原始碼；使用者要看某檔再讀。

## 更新規則

- 使用者在討論中確定盒子、契約或連線有變：**先改地圖，再繼續討論**。改完更新「最後確認」日期，然後跑 build 重畫。
- 改之前先看專案自己的文件規則（例如 `CLAUDE.md` 指向的文件風格、編號、待確認清單），規則優先於本 skill。
- 每個盒子只寫「Input、Output、保證、內部、程式」；數字與細節留在出處。

## 指令

```bash
python .claude/skills/concept-map/scripts/build.py --root <專案根> [--map <地圖.md>] [--out <資料夾>]
```

- 解析地圖 → 檢查「程式」欄路徑是否存在、列出尚未登記的鄰近目錄、檢查連線與盒子是否對得上 → 寫 `model.json`、`drift.md`、`viewer.html`。
- 「對一下現況」：跑 build，只摘要 `drift.md` 的差異給使用者決定，不自動修。需要使用者決定的事照專案規則放進它的待確認清單。

## 看圖（讓使用者邊討論邊看）

- 用內建瀏覽器面板開 `file:///{專案根}/.claude/concept-map/viewer.html`（`navigate` 或 `preview_start` 帶 `url`）。地圖改過、重跑 build 後再 `navigate` 一次即重新載入。
- 面板載入本機檔時網址的 `#hash` 會被丟掉，要帶焦點就用 `javascript_tool` 呼叫頁面函式：
  `select('C')` 選取盒子 C、`drill('C')` 進入 C（有 L2 子盒就進 L2，否則列 L3 檔案）、`go('')` 回 L1、`go('L0')` 看 L0、`up()` 上一層。
  用瀏覽器自己開檔時，`viewer.html#C`、`#scope=C`、`#L0` 也可以。
- 討論到哪個盒子就選取哪個，並在回答裡標明目前層與盒子。
- viewer 的顏色、分組、圖例、連線全部來自地圖的 mermaid 區塊；要改圖就改地圖，不要改 viewer。

## 圖的畫法（沿用使用者偏好，細節見 `template.md` 檔尾）

`flowchart LR`；`classDef` 用 600 級色當填色、同色系深一級當邊框、白字、2px，一種顏色一種角色（每張最多 4 種）；`subgraph` 分組且標題寫「誰擁有它」；節點兩行（「ID 名稱」`<br/>`一句話）；只在跨分組的連線標檔案或資料，必要用實線、選用用虛線；回饋線最後寫；`class` 套色寫在圖尾。

## 邊界

- 地圖與程式可能脫節；平常不主動掃，只在使用者要求或改地圖後 build 時檢查。
- build 只檢查路徑與格式，不判斷程式依賴是否真的符合連線。
