---
name: readme
description: 檢查目前專案哪些資料夾需要、缺少或違規的 README.md，先列出要新增／修正／可刪的清單，使用者確認後再並行套用。使用者說「/readme」「檢查 readme」「哪些資料夾要 readme」時使用。
argument-hint: [apply | 路徑]
---

# /readme

規則全文在 `.claude/skills/readme/RULES.md`（先讀）。這個 skill 只做兩階段：**先列清單，使用者確認後才動檔案**。

## 階段 1：列清單（唯讀）

```bash
python ~/.claude/hooks/readme_guard.py scan .
```

把輸出整理成表給使用者看，分四段：要新增、要修正、可考慮刪除（僅供參考，不自動刪）、README 被 .gitignore 排除。每一列附一句理由。對「要新增」的每個資料夾，先簡短讀過內容再決定該寫什麼，不要憑檔名猜。

這個階段不寫任何檔案。結束後問使用者：哪些照做、哪些不要、有沒有要改政策（門檻、`exempt_globs`、`skip_check_globs` 寫在 `<repo>/.claude/readme_policy.json`）。

## 階段 2：並行套用（使用者確認後）

- 每個要新增或修正的 README 派一個 subagent，**同一則訊息一次派完**，互相不改對方的檔。
- subagent 的 brief 要寫：該資料夾路徑、`RULES.md` 的 L1–L6、沿用的格式（見下）、只許寫這一個 README、不得改其他檔。
- 根目錄 README 可以寫詳細：專案簡介、每個頂層資料夾一行、指向各自 README。其他 README 每份 60 行內（或政策檔的 `max_lines`），只講自己的範圍。
- 全部完成後再跑一次 `scan` 驗證，回報剩下的問題。**不自動 commit**；commit 時 hook 會再檢查一次。
- README 在 `.gitignore` 內時，提議在 `.gitignore` 加 `!路徑`，經使用者同意才改。

## README 格式

```markdown
# 資料夾名：一句用途

## 內容
| 項目 | 說明 |
| --- | --- |
| `a.py` | 一句話 |
| `sub/` | 一句話；有自己的 README 見該處 |

## 使用（有的話）
一兩行指令或規則。
```

只寫現況，不寫沿革；不重述其他文件的數字與結論。

## 模式

政策檔 `mode` 預設 `warn`（commit 只提醒）。使用者要擋 commit 時改成 `deny`。
