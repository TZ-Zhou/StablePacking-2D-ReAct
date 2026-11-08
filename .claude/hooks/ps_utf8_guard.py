"""PreToolUse hook：開新 PowerShell 視窗跑 python 時，兩邊都要設成 UTF-8，否則中文輸出與 log 會變亂碼。

成因：Claude Code 的環境帶 `PYTHONIOENCODING=utf-8:surrogateescape`，`Start-Process` 開的子視窗會繼承，
python 因此以 UTF-8 輸出；但新視窗的主控台預設 cp950（Big5），`Tee-Object` 依主控台編碼解碼，
畫面與 log 的中文都變亂碼。只設其中一邊不夠：沒有繼承環境時 python 會改用 cp950 輸出。

行為：
- 指令沒有「`Start-Process` 開 powershell／pwsh」或沒有呼叫 python：不輸出，直接放行。
- 有開新視窗又跑 python，但新視窗的指令缺 `[Console]::OutputEncoding = UTF8`
  或缺 `PYTHONIOENCODING=utf-8`（或 `PYTHONUTF8=1`）：deny，附上 .claude/SHELL.md 的標準寫法。

輸入為 stdin 的 hook JSON，輸出為 PreToolUse 的 JSON 決策；任何讀取錯誤都放行（不擋其他指令）。
"""

from __future__ import annotations

import json
import re
import sys

# Start-Process（含別名 start、saps）開 powershell 或 pwsh 的新視窗
LAUNCH = re.compile(r"\b(?:start-process|saps|start)\s+(?:-filepath\s+)?['\"]?(?:powershell|pwsh)\b", re.I)

# python 直譯器：python、python.exe、SHELL.md 範本的 $pyExe；不比對 Python313 這類路徑片段
PYTHON = re.compile(r"\bpython(?:\.exe)?\b|\$pyexe\b", re.I)

# 新視窗的主控台改成 UTF-8
CONSOLE_UTF8 = re.compile(
    r"\[console\]::outputencoding\s*=\s*(?:"
    r"\[(?:system\.)?text\.encoding\]::utf8"
    r"|\[(?:system\.)?text\.utf8encoding\]::new\("
    r"|new-object\s+(?:system\.)?text\.utf8encoding)"
    r"|\bchcp(?:\.com)?\s+65001\b",
    re.I,
)

# python 的輸出固定為 UTF-8
PYTHON_UTF8 = re.compile(r"pythonioencoding\s*=\s*['\"]?utf-?8|pythonutf8\s*=\s*['\"]?1", re.I)

TEMPLATE = (
    "$cmd = \"[Console]::OutputEncoding = [Text.Encoding]::UTF8; `$env:PYTHONIOENCODING = 'utf-8'; "
    "Set-Location '$workDir'; & '$pyExe' -u <script> <args> 2>&1 | Tee-Object -FilePath '$log'; pause\""
)


def deny(reason: str) -> None:
    """輸出 PreToolUse 的 deny 決策後結束。"""
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 主控台預設 cp950
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            },
            ensure_ascii=False,
        )
    )
    sys.exit(0)


def missing_settings(command: str) -> list[str]:
    """開新視窗跑 python 卻缺的 UTF-8 設定；不是這種指令時回傳空串列。"""
    if not (LAUNCH.search(command) and PYTHON.search(command)):
        return []
    missing = []
    if not CONSOLE_UTF8.search(command):
        missing.append("新視窗的 `[Console]::OutputEncoding = [Text.Encoding]::UTF8`")
    if not PYTHON_UTF8.search(command):
        missing.append("新視窗的 `$env:PYTHONIOENCODING = 'utf-8'`（或 `PYTHONUTF8=1`）")
    return missing


def main() -> None:
    try:
        # stdin 預設用系統編碼（cp950）解碼，中文會被拆成亂碼
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)  # 讀不到輸入就不干預

    command = str(payload.get("tool_input", {}).get("command", ""))
    if missing := missing_settings(command):
        deny(
            "開新 PowerShell 視窗跑 python 時，中文輸出與 log 會變亂碼：缺 " + "、".join(missing) + "。"
            "子視窗繼承 PYTHONIOENCODING=utf-8，python 以 UTF-8 輸出，但新視窗主控台預設 cp950。"
            "照 .claude/SHELL.md 的寫法，在 -Command 字串的最前面加上兩個設定，例如：" + TEMPLATE
        )
    sys.exit(0)


if __name__ == "__main__":
    main()
