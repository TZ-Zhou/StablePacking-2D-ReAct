# README 規則（使用者全域）

目的：資料夾內容多到無法一眼讀完一個觀念時，由該資料夾自己的 README 說明這個範圍。

## 何時需要

只算 git 看得見的檔案（含未追蹤但未被 .gitignore 排除者），不含 `__pycache__`、`__init__.py`、`conftest.py`、`.gitkeep`。

| 判準 | 門檻（預設，可被政策檔覆寫） |
| --- | --- |
| 直屬項目數（檔案＋子資料夾，不含 README 自己） | ≥ 6 |
| 或遞迴檔案數 | ≥ 15 |
| 根目錄 | 一律需要 |

豁免：`archive`、`code`、`_ref_repo`、`tests`、`.claude`、`.venv`、`node_modules` 等（`exempt_dirs`）；只含圖檔／媒體的資料夾；政策檔 `exempt_globs` 列的路徑。

## 怎麼寫（非根目錄）

| 編號 | 規則 |
| --- | --- |
| L1 | 一眼讀完：不超過 `max_lines`（預設 60）行 |
| L2 | 第一行是 `# 標題`，前 3 行內有一句用途 |
| L3 | 只講自己的範圍：每個直屬項目都出現一次（`bench_*.py` 這類萬用字元可涵蓋）；不提不存在的直屬檔 |
| L4 | 不往上層延伸：不寫 `..`；指向本資料夾以外的路徑最多 3 條，且只是一行指標，不重述對方內容 |
| L5 | 不過度深入：子資料夾有自己的 README 時只一行帶過；沒有的最多往下寫一層 |
| L6 | 只寫現況：出現「原本」「改成」「已作廢」等沿革用語時提醒（只警告） |

根目錄 README 例外：只檢查 L2、L3，可寫詳細、可列所有頂層資料夾。

數字與結論指向它們的家，不在 README 重述（一件事只有一個家）。

## 政策檔

預設值 < `~/.claude/readme_policy.json` < `<repo>/.claude/readme_policy.json`（`exempt_dirs`、`ignore_names`、`media_ext`、`exempt_globs`、`skip_check_globs` 取聯集）。

| 鍵 | 意思 |
| --- | --- |
| `mode` | `warn`：commit 只提醒；`deny`：違規擋住 commit |
| `min_direct_entries`、`min_recursive_files`、`max_lines`、`max_outward_pointers`、`max_scan_depth` | 門檻 |
| `exempt_globs` | 符合的資料夾不要求 README（fnmatch，`*` 可跨層） |
| `skip_check_globs` | 符合的資料夾若已有 README，不檢查 L1–L6，也不列為可刪 |

## Hook 範圍（scope only）

| Hook | 時機 | 檢查範圍 |
| --- | --- | --- |
| post | 寫檔後 | 只看被寫檔案所在的那一個資料夾 |
| commit | `git commit` 前 | 只看暫存區裡直屬項目有增減的資料夾與被動到的 README；沒碰的舊違規不管 |
| scan | `/readme` | 整個 repo，列出清單，不動檔案 |

任何讀取錯誤、不在 git repo、被豁免：一律放行。
