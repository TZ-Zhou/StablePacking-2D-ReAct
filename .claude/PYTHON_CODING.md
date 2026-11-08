# Python 程式碼原則 — StablePacking-2D-ReAct

本檔只規範**怎麼寫**。實驗設定、指標定義、研究結論以 `research/` 的程式與 `research/findings.md` 為準，本檔不另立定義。

---

## 0. 上游程式碼與新程式碼

- `Stonepacker2D/` 與 `Tests/` 是 Wang et al. 的上游程式。**不做整檔重排或改風格**；為了追蹤或控制而修改時只動必要的行，且不改演算法行為（候選生成、卷積、LLC 閘門、穩定性分析）。
- 本檔的規則適用於 `research/` 的程式，以及上游檔案中被整段重寫的部分。

## 1. Pythonic

- `research/` 是套件：模組之間一律絕對匯入（`from research.runner import run_wall`），不改 `sys.path`；路徑常數只從 `research/paths.py` 取。例外只有兩處：`controller/worker.py`（在 venv 以 `python -I` 直接執行）與 staged case 的開頭（`case_definition.stage`）。
- 用語言既有機制：`with` 管資源、`enumerate`／`zip` 取代索引算術、comprehension 取代累加迴圈、`pathlib` 取代字串拼路徑。
- numpy 能向量化就不寫 Python 迴圈。
- 標準庫先於第三方，第三方先於自己重寫。
- 不用 `*` import，不改動可變的預設參數，函式庫模組不在模組層級做事（進入點腳本例外）。
- 讀寫文字檔一律指定 `encoding='utf-8'`。

## 2. Clean code

- 函式做一件事，一眼看得完。
- **函式本體縮排最多 3 層**。要到第 4 層時，就把內層抽成函式，或改寫邏輯：提早 return、`continue`、comprehension、numpy 向量化。
- 名字讀得出意圖，不靠註解補救。
- 魔術數字要具名；可調參數放在模組頂端的大寫常數。
- 死碼直接刪，不留註解掉的版本（git 會記得）。
- 副作用集中在邊界（IO、子行程、寫檔），核心計算保持純函式，才測得動。

## 3. DRY

- 同一個概念只能有一份實作。看到第二份就抽共用。
- **牆品質指標、篩選順序、追蹤格式尤其嚴格：一個定義一份程式碼**（指標在 `research/walls/wall_metrics.py`，篩選在 `Stonepacker2D/stack/selection.py`）。兩份實作＝數字遲早對不起來。
- 抽象等到真的重複才做。

## 4. 職責分離

- 控制器只拿唯讀副本、只回 JSON（`research/controller/worker.py`），不碰 Wang 的物件與 RNG。
- 所有建牆的啟動點與執行邏輯都在 `research/runner/`；一面牆或一批牆「是什麼」的定義放在各研究資料夾，定義檔不 import `research.runner`，也沒有命令列。
- 離線分析只讀追蹤與記錄，不重建牆；要建牆就經 `research/runner/run_wall.py`。
- 新牆只由 `run_wall.run` 寫到 `research/cases/<case_id>/`；一組牆用 `research/storage/studies.py` 記錄。讀牆一律用 `case_id.wall_dir(ref)`（case_id 或封存路徑皆可）或 `paths.LEGACY_RUNS`；`research/runs/` 只讀不寫。一面牆的衍生分析寫到該 case 的 `derived/<方法版本>/`。
- 有狀態的才用 class；無狀態計算寫成函式。繼承深度以一層為度，其餘用組合。

## 5. docstring 與註解

- **一律英文。**
- 模組 docstring：這個檔案做什麼、用法（`python -B -m research.<資料夾>.<模組> ...`）、輸入與輸出寫到哪裡。讀完就知道要不要往下看。
- 函式 docstring：一句或一小段，寫做什麼與不明顯的約束（座標系、單位、哪些資料只讀）。
- 註解寫**為什麼**，不寫**做什麼**；只在非顯而易見處出現。不寫沿革——那是 commit message 的事。

## 6. 格式與工具

- `ruff` 負責 lint，line-length 120（`pyproject.toml`）；上游檔案不整檔 format。
- 字串沿用既有檔案的引號風格。
- 測試放 `research/tests/`，用 pytest。
