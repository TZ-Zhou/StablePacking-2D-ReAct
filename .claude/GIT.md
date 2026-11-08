# Git 原則 — StablePacking-2D-ReAct

## 語言

- **commit message 一律英文**，和 repo 既有歷史一致。

## 格式

```
<type>: <one-line description, imperative, no period>

<why, and what it affects; bullets allowed>
```

- 主旨 ≤ 100 字元，不加句號；scope 可省略，要寫就寫被改動的模組或資料夾（`feat(research): ...`）。
- 主旨與 body 之間空一行。body 只在「為什麼」不明顯時才寫，可用 `- ` 條列，長的條目可縮排續行。
- 以上格式由 `.claude/hooks/git_commit_guard.py` 檢查，不符會被擋下。

**type**

| type | 用途 |
| --- | --- |
| `feat` | 新功能、新分析或新工具 |
| `fix` | 修 bug |
| `perf` | 效能優化，body 附實測數字 |
| `test` | 測試 |
| `data` | 研究資料（例如 `research/quality/` 的評估輸出），不得含程式改動 |
| `docs` | 文件 |
| `refactor` | 不改行為的重構 |
| `chore` | 雜項 |
| `deps` | 升依賴，單獨一個 commit |

- **不得用自訂的階段或代號當 scope**（`p0`、`P1.6`、`gate1`）。階段編號寫進主旨或 body，例如 `feat: add the P1 limited-lookahead study ...`。

## 粒度

- 一個 commit 一件可描述的事，跨主題的改動要拆開。
- 同一批作業、相同 `type(scope)` 的改動合併成一個 commit。hook 只擋同一指令裡有多個同前綴 commit。
- `data:` 與 `deps:` 單獨成 commit，讓 `git bisect` 能單獨找到數字變動的來源。
- 文件與程式碼盡量分開 commit，除非該文件就是在描述這次程式碼改動。

## 與實驗的關係

- `campaign.py init` 會記下 git head 與每個執行原始檔的 SHA256；程式改了，`run` 就拒絕繼續。要引用的 campaign 先 commit 再 init。
- `research/cases/` 與封存的 `research/runs/` 不進版控；`research/studies/` 的 study 紀錄與 `research/quality/<方法>/` 的評估結果進版控，用 `data:` commit。

## 沿革寫在這裡

文件只寫現況（見 `DOC_STYLE.md`）。**被取代的數字、作廢的結論、改動的理由，全部寫進 commit message。**

## 不進版控

- `research/runs/`、`research/cases/`、`Tests/*/result*/`、`.venv/`、`__pycache__/`、`.env`
- API 金鑰與 token
- 超過 10 MB 的檔案一律先問過再加

## 署名

**commit 只掛人類作者。** 訊息不得出現 `Co-Authored-By:` 或 `Generated with` 這類 AI 署名行。
`.claude/settings.json` 的 `attribution` 設為空字串，hook 再擋一層。

## 禁止

| 禁止 | 理由 |
| --- | --- |
| 中文 commit message | 本專案規定英文 |
| commit message 掛 AI 署名 | commit 只掛人類作者 |
| 用自訂階段或代號當 scope（`feat(p0)`） | 脫離計畫書就讀不懂 |
| 資料更新混在程式改動裡 | 無法分辨數字變動來自哪裡 |
| `--no-verify` 跳過 hook | 有問題就修，不要繞過 |
| `--amend` 既有 commit、force push | 除非明確授權 |
| commit 密鑰、token | — |
| 用 commit 當備份（「wip」） | 訊息無法回答「為什麼」 |
