"""專案盤點：給 /concept-map init 用，唯讀，只用標準庫。

用法：python survey.py <專案根>
輸出一份 Markdown 摘要：清單檔、README 開頭、文件、入口檔、頂層目錄的
檔案組成與代表檔。目的是讓起草 L0／L1 的人少讀檔。
"""

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

SKIP = {
    ".git", "node_modules", "__pycache__",
    ".venv", "venv", "env",
    "dist", "build", "target",
    ".idea", ".vscode", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".tox",
    ".next", ".claude",
    "site-packages", ".cache",
}
MANIFESTS = (
    "pyproject.toml", "setup.py", "setup.cfg",
    "requirements.txt", "package.json", "Cargo.toml",
    "go.mod", "pom.xml", "build.gradle",
    "CMakeLists.txt", "Makefile", "Dockerfile",
    "docker-compose.yml", "environment.yml", "tsconfig.json",
)
ENTRY_NAMES = {
    "main.py", "__main__.py", "cli.py",
    "app.py", "manage.py", "server.py",
    "run.py", "index.js", "index.ts",
    "main.ts", "main.js", "main.go",
    "main.rs", "lib.rs",
    "Program.cs", "Main.java",
}
CODE_EXTS = (
    ".py", ".js", ".ts", ".go",
    ".rs", ".cs", ".java", ".cpp",
    ".c", ".h", ".md",
)
KEY_STEMS = ("__init__", "main", "index", "app", "cli", "core", "api")
MAX_FILES = 20000


def rel(root: Path, p: Path) -> str:
    """相對專案根的 POSIX 路徑。"""
    return p.relative_to(root).as_posix()


def list_dir(cur: Path) -> list[Path]:
    """列目錄，讀不了回空。"""
    try:
        return sorted(cur.iterdir())
    except OSError:
        return []


def walk(root: Path) -> list[Path]:
    """列出所有檔案（略過雜訊目錄），超過上限就截斷。"""
    out: list[Path] = []
    stack = [root]
    while stack and len(out) < MAX_FILES:
        for p in list_dir(stack.pop()):
            if not p.is_dir():
                out.append(p)
            elif p.name not in SKIP and not p.is_symlink():
                stack.append(p)
    return out


def head_lines(path: Path, n: int) -> list[str]:
    """讀檔案前 n 行；讀不了回空。"""
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            return [next(f).rstrip() for _ in range(n)]
    except (OSError, StopIteration, UnicodeError):
        return []


def first_doc(path: Path) -> str:
    """取檔案的第一行說明：Python 看 docstring，其餘看開頭註解。"""
    if path.suffix == ".py":
        try:
            doc = ast.get_docstring(ast.parse(path.read_text("utf-8")))
        except (SyntaxError, OSError, UnicodeError):
            return ""
        return doc.strip().splitlines()[0][:80] if doc else ""
    for line in head_lines(path, 6):
        m = re.match(r"\s*(?://+|#|/\*+|\*)\s*(\S.*)", line)
        if m and not m.group(1).startswith(("!", "-*-")):
            return m.group(1)[:80]
    return ""


def manifest_info(path: Path) -> str:
    """清單檔的一行摘要。"""
    if path.name == "package.json":
        try:
            data = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            return ""
        scripts = ",".join(list(data.get("scripts", {}))[:6])
        return (
            f"name={data.get('name', '')} {data.get('description', '')} "
            f"scripts=[{scripts}]"
        )
    if path.suffix in (".toml", ".cfg", ".txt", ".mod", ".yml"):
        pat = r"\s*(name|description|module|version)\s*[=:]"
        keep = [ln.strip() for ln in head_lines(path, 40) if re.match(pat, ln)]
        return "; ".join(keep[:3])
    return ""


def readme_head(root: Path, n: int = 30) -> list[str]:
    """README 開頭。"""
    for p in list_dir(root):
        if p.name.lower().startswith("readme") and p.is_file():
            return [f"({p.name})"] + head_lines(p, n)
    return []


def pick_files(mine: list[Path]) -> list[Path]:
    """挑代表檔：先挑關鍵檔名，再挑最大的程式與文件檔，最多 6 個。"""
    code = [p for p in mine if p.suffix in CODE_EXTS]
    keyed = [p for p in code if p.stem in KEY_STEMS]
    biggest = sorted(code, key=lambda p: -p.stat().st_size)
    shown: list[Path] = []
    for p in keyed + biggest:
        if p not in shown:
            shown.append(p)
    return shown[:6]


def dir_summary(root: Path, files: list[Path], top: str) -> list[str]:
    """一個頂層目錄的檔案組成、子目錄與代表檔。"""
    # 屬於這個目錄的檔案，與副檔名、子目錄的統計
    mine = [p for p in files if p.relative_to(root).parts[0] == top]
    exts = Counter(p.suffix or "(無副檔名)" for p in mine)
    subs = Counter(
        p.relative_to(root).parts[1]
        for p in mine
        if len(p.relative_to(root).parts) > 2
    )

    # 標題與子目錄行
    kinds = "、".join(f"{k}×{v}" for k, v in exts.most_common(3))
    lines = [f"### {top}/  （{len(mine)} 檔：{kinds}）"]
    if subs:
        names = "、".join(f"{k}({v})" for k, v in subs.most_common(10))
        lines.append(f"- 子目錄：{names}")

    # 代表檔與它們的第一行說明
    lines += [f"- `{rel(root, p)}`：{first_doc(p)}" for p in pick_files(mine)]
    return lines


def print_manifests(root: Path, files: list[Path]) -> None:
    """印清單檔與 README 開頭。"""
    print("## 清單檔")
    for f in files:
        if f.name in MANIFESTS and len(f.relative_to(root).parts) <= 2:
            print(f"- `{rel(root, f)}` {manifest_info(f)}")
    print("\n## README 開頭")
    print("\n".join(readme_head(root)) or "（沒有）")


def print_docs_entries(root: Path, files: list[Path]) -> None:
    """印文件清單與入口檔。"""
    docs = [
        f
        for f in files
        if f.suffix == ".md"
        and f.relative_to(root).parts[0].lower() in ("docs", "doc")
    ]
    print(f"\n## 文件（{len(docs)} 份，列前 25）")
    for f in docs[:25]:
        print(f"- `{rel(root, f)}`")
    print("\n## 入口檔")
    for f in files:
        if f.name in ENTRY_NAMES:
            print(f"- `{rel(root, f)}`：{first_doc(f)}")


def print_dirs(root: Path, files: list[Path]) -> None:
    """印頂層目錄摘要與根目錄檔案。"""
    print("\n## 頂層目錄")
    tops = sorted(
        {
            f.relative_to(root).parts[0]
            for f in files
            if len(f.relative_to(root).parts) > 1
        }
    )
    for top in tops[:40]:
        print("\n".join(dir_summary(root, files, top)))
    loose = [f.name for f in files if len(f.relative_to(root).parts) == 1]
    print(f"\n## 根目錄檔案\n{'、'.join(loose[:40]) or '（沒有）'}")


def main() -> int:
    """印出盤點結果。"""
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    root = Path(ap.parse_args().root).resolve()
    files = walk(root)
    cut = "，已截斷" if len(files) >= MAX_FILES else ""
    print(f"# 專案盤點：{root.name}（{len(files)} 檔{cut}）\n")
    print_manifests(root, files)
    print_docs_entries(root, files)
    print_dirs(root, files)
    return 0


if __name__ == "__main__":
    sys.exit(main())
