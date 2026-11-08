"""PreToolUse hook：檢查 git commit 訊息的語言、格式與署名，並注入專案 git 原則。

行為：
- 非 `git commit` 的 Bash 指令：不輸出，直接放行。
- 訊息帶 Co-Authored-By 或 Generated with 署名：deny。
- 訊息含中日韓字元：deny（本專案 commit 一律英文）。
- scope 是自訂階段或代號（p0、phase2、gate1…）：deny，階段編號寫進主旨或 body。
- 主旨不符 `<type>: <說明>` 或 `<type>(<scope>): <說明>`、超過 100 字元或以句號結尾：deny。
- 主旨與 body 之間沒有空行：deny。
- 同一指令內多個 commit 的 `type(scope)` 相同：deny，要求合併成一個。
- 其餘 `git commit`：允許，並把 .claude/GIT.md 的重點注入 context。

輸入為 stdin 的 hook JSON，輸出為 PreToolUse 的 JSON 決策。
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path

# 中日韓統一表意文字 + 擴充區 A + 相容表意文字
CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")

# AI 署名：Co-Authored-By trailer、Generated with 廣告行
ATTRIBUTION = re.compile(
    r"^\s*(?:Co-Authored-By\s*:|(?:🤖\s*)?Generated with\b)", re.M | re.I
)

TYPES = "feat|fix|perf|test|data|docs|refactor|chore|deps"
SUBJECT = re.compile(rf"^(?:{TYPES})(?:\([a-z0-9-]+\))?: \S")

# 主旨前綴 `type(scope)`，用來判斷是否屬於同一批作業
PREFIX = re.compile(rf"^((?:{TYPES})(?:\([a-z0-9-]+\))?):")

# 計畫階段或代號（p0、phase2、gate1、stage-3）脫離計畫書就讀不懂，不得當 scope
SCOPE = re.compile(rf"^(?:{TYPES})\(([^)]*)\)")
CODENAME_SCOPE = re.compile(r"^(?:p|phase|stage|gate|step|m|milestone)-?\d+(?:[.-]\d+)*$", re.I)

SUBJECT_MAX = 100

RULES = (
    "專案 git 原則（.claude/GIT.md）："
    "commit message 一律英文；"
    f"主旨 `<type>: <description>`（scope 可省略），≤ {SUBJECT_MAX} 字元，不加句號，"
    f"type 為 {TYPES.replace('|', '、')}；"
    "scope 若寫，寫被改動的模組或資料夾，不得用 p0、gate1 這類自訂階段或代號；"
    "body 寫為什麼與影響範圍，可用條列；"
    "一個 commit 一件事，跨主題要拆開；同一批作業、相同 type(scope) 合併成一個 commit；"
    "資料（research/quality 等）與程式盡量分開 commit，用 data type；"
    "被取代的數字與改動理由寫進 commit message，不寫進文件；"
    "不得用 --no-verify 繞過 hook；"
    "commit message 不得掛任何 AI 署名（Co-Authored-By、Generated with）。"
)

DENY_ATTRIBUTION = (
    "commit message 不得掛 AI 署名。"
    "偵測到 Co-Authored-By 或 Generated with 行。"
    "本專案的 commit 只掛人類作者，請移除該行後重試。"
)

DENY_LANGUAGE = "commit message 必須是英文。偵測到訊息裡有中文字。請改寫成 `<type>: <English description>`。"


def emit(decision: str, reason: str) -> None:
    """輸出 PreToolUse 決策 JSON 後結束。"""
    # Windows 主控台預設 cp950，會把中文輸出成亂碼
    sys.stdout.reconfigure(encoding="utf-8")
    suffix = "" if decision == "allow" else "詳見 .claude/GIT.md。"
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": decision,
                    "permissionDecisionReason": reason + suffix,
                    "additionalContext": RULES,
                }
            },
            ensure_ascii=False,
        )
    )
    sys.exit(0)


def parse_flags(tokens: list[str]) -> str | None:
    """從單一 `git commit` 之後的 tokens 組出 -m／-F 訊息；沒有訊息參數回傳 None。"""
    parts: list[str] = []
    for flag, value in zip(tokens, tokens[1:]):
        if flag in ("-m", "--message"):
            parts.append(value)
        elif flag in ("-F", "--file") and value != "-":
            try:
                parts.append(Path(value).read_text(encoding="utf-8"))
            except OSError:
                return " ".join(tokens)
    parts += [t.split("=", 1)[1] for t in tokens if t.startswith("--message=")]
    return "\n\n".join(parts) if parts else None


def split_commands(command: str) -> list[list[str]]:
    """把指令切成以 ; && || | 分隔的段落 tokens；引號內的符號不切。"""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
    lexer.whitespace_split = True
    lexer.commenters = ""  # 訊息裡的 `#12` 不是註解
    segments: list[list[str]] = [[]]
    for token in lexer:
        if set(token) <= set(";&|"):
            segments.append([])
        else:
            segments[-1].append(token)
    return segments


def extract_messages(command: str) -> list[str]:
    """取出指令中每個 git commit 的訊息；開編輯器的 commit 不在其內。"""
    heredocs = re.findall(r"<<-?\s*['\"]?(\w+)['\"]?\n(.*?)\n\1\s*(?:\n|$)", command, re.S)
    if heredocs:
        return [body for _, body in heredocs]

    try:
        segments = split_commands(command)
    except ValueError:
        return [command]  # 引號不成對：整段當訊息檢查，交給格式規則擋下

    # 每段各自解析，避免只檢查到第一個 commit
    messages = []
    for seg in segments:
        starts = [i for i in range(len(seg) - 1) if seg[i : i + 2] == ["git", "commit"]]
        if starts and (msg := parse_flags(seg[starts[0] + 2 :])) is not None:
            messages.append(msg)
    return messages


def check_format(message: str) -> str | None:
    """檢查主旨格式與主旨後的空行；違規時回傳 deny 理由。"""
    lines = [line.rstrip() for line in message.strip().splitlines()]
    subject = lines[0]

    scope = SCOPE.match(subject)
    if scope and CODENAME_SCOPE.match(scope.group(1)):
        return (
            f"scope `{scope.group(1)}` 是自訂階段或代號，日後讀不懂。"
            "scope 要寫被改動的模組或資料夾，或省略；階段編號寫進主旨或 body。"
        )
    if not SUBJECT.match(subject):
        return (
            "主旨格式須為 `<type>: <description>` 或 `<type>(<scope>): <description>`，"
            f"type 限 {TYPES}。目前：`{subject}`。"
        )
    if len(subject) > SUBJECT_MAX:
        return f"主旨 {len(subject)} 字元，超過 {SUBJECT_MAX}。請再精簡，細節寫進 body。"
    if subject.endswith(("。", ".")):
        return "主旨句尾不加句號。"
    if len(lines) > 1 and lines[1].strip():
        return "主旨與 body 之間要空一行。"
    return None


def prefix_of(message: str) -> str | None:
    """取出主旨的 `type(scope)` 前綴。"""
    match = PREFIX.match(message.strip().splitlines()[0]) if message.strip() else None
    return match.group(1) if match else None


def check_batch(messages: list[str]) -> str | None:
    """同一指令內相同 type(scope) 的 commit 必須合併；違規時回傳 deny 理由。"""
    prefixes = [p for m in messages if (p := prefix_of(m))]
    duplicated = sorted({p for p in prefixes if prefixes.count(p) > 1})
    if duplicated:
        return f"同一指令內的 `{duplicated[0]}` commit 要合併成一個，不要分多次 commit。"
    return None


def main() -> None:
    try:
        # stdin 預設用系統編碼（cp950）解碼，中文會被拆成亂碼、字數算錯
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)  # 讀不到輸入就不干預

    command = str(payload.get("tool_input", {}).get("command", ""))
    if not re.search(r"\bgit\s+commit\b", command):
        sys.exit(0)

    # 沒有 -m／-F／heredoc 的 commit 交給編輯器，不干預
    messages = extract_messages(command)
    for message in messages:
        if ATTRIBUTION.search(message):
            emit("deny", DENY_ATTRIBUTION)
        if CJK.search(message):
            emit("deny", DENY_LANGUAGE)
        if reason := check_format(message):
            emit("deny", reason)
    if reason := check_batch(messages):
        emit("deny", reason)

    emit("allow", "commit message 符合專案規則。")


if __name__ == "__main__":
    main()
