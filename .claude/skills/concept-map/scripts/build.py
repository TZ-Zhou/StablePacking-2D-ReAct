"""概念地圖：解析 Markdown 架構地圖，產生 model.json、drift.md、viewer.html。

只用標準函式庫。用法見 SKILL.md。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# 版面常數（px）
NODE_W, NODE_H = 220, 84
COL_GAP, ROW_GAP = 120, 36
GROUP_PAD, GROUP_TITLE_H = 18, 26
MAX_ENTRIES = 80

FIELD_KEYS = {
    "input": "input",
    "輸入": "input",
    "output": "output",
    "輸出": "output",
    "guarantee": "guarantee",
    "guarantees": "guarantee",
    "保證": "guarantee",
    "inside": "inside",
    "內部": "inside",
    "code": "code",
    "程式": "code",
}
NEUTRAL = {"fill": "#64748B", "stroke": "#475569"}
KEYWORDS = {
    "subgraph", "end", "classdef", "class", "flowchart",
    "graph", "direction", "style", "linkstyle", "click",
}
SOURCE_SUFFIXES = (".py", ".cs", ".js", ".ts")

HEAD_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
BOX_RE = re.compile(r"^([A-Z][A-Za-z0-9]{0,2})\s+(.+)$")
EDGE_RE = re.compile(
    r"(?P<a>[A-Za-z]\w*)\s*"
    r"(?:(?P<solid>-->|==>)|(?P<dash>-\.\s*(?P<dl>[^>]*?)\s*\.->))"
    r"\s*(?:\|\s*\"?(?P<sl>[^|\"]*)\"?\s*\|\s*)?"
    r"(?P<b>[A-Za-z]\w*)"
)
NODE_LABEL_RE = re.compile(r"([A-Za-z]\w*)\s*\[\s*\"([^\"]*)\"\s*\]")
SUBGRAPH_RE = re.compile(r"subgraph\s+(\w+)\s*(?:\[\s*\"?([^\"\]]*)\"?\s*\])?")


# ---------- 找根目錄與地圖 ----------


def find_root(start: Path) -> Path:
    """回傳 git 根目錄；不在 git 底下就回傳 start。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return start
    return Path(out)


def find_map(root: Path) -> Path | None:
    """依 CLAUDE.md 的「概念地圖：路徑」行，再依常見檔名找地圖。"""
    claude = root / "CLAUDE.md"
    if claude.is_file():
        text = claude.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"概念地圖[：:]\s*`?([^\s`]+\.md)`?", text)
        if m and (root / m.group(1)).is_file():
            return root / m.group(1)
    names = ("architecture-map.md", "ARCHITECTURE.md", "architecture.md")
    candidates = [root / n for n in names]
    for sub in ("docs", "doc"):
        for n in names:
            candidates += sorted((root / sub).glob(f"**/{n}"))
    return next((c for c in candidates if c.is_file()), None)


# ---------- 切章節與表格 ----------


def split_sections(text: str) -> list[dict]:
    """依標題切章節，記錄本文與 mermaid 區塊（略過程式碼框內的標題）。"""
    sections = [{"level": 0, "title": "", "lines": [], "mermaid": []}]
    fence, block = None, []
    for line in text.splitlines():
        stripped = line.strip()
        if fence is None and stripped.startswith("```"):
            fence, block = stripped[3:].strip().lower() or "text", []
        elif fence is not None and stripped.startswith("```"):
            if fence == "mermaid":
                sections[-1]["mermaid"].append(block)
            fence = None
        elif fence is not None:
            block.append(line)
        elif HEAD_RE.match(line):
            m = HEAD_RE.match(line)
            sections.append(
                {
                    "level": len(m.group(1)),
                    "title": m.group(2).strip(),
                    "lines": [],
                    "mermaid": [],
                }
            )
            continue
        sections[-1]["lines"].append(line)
    return sections


def table_rows(lines: list[str]) -> list[tuple[str, str]]:
    """取出兩欄表格的 (鍵, 值)，略過表頭與分隔列。"""
    rows = []
    for line in lines:
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 2 or set(cells[0]) <= set("-: "):
            continue
        rows.append((cells[0].replace("**", ""), " | ".join(cells[1:])))
    return rows


# ---------- 解析 mermaid ----------


class MermaidParser:
    """逐行解析 mermaid：classDef、class、subgraph 分組、連線、圖例。"""

    def __init__(self) -> None:
        self.classdefs: dict[str, dict] = {}
        self.node_class: dict[str, str] = {}
        self.legend: dict[str, str] = {}
        self.taglines: dict[str, str] = {}
        self.scopes: dict[str, dict] = {}
        self._sc: dict = {}
        self._current: dict | None = None
        self._assigned: set[str] = set()

    def feed(self, scope: str, lines: list[str]) -> None:
        """解析一個區塊；scope 是盒子 ID，空字串表示 L1。"""
        self._sc = self.scopes.setdefault(
            scope, {"groups": [], "edges": [], "order": []}
        )
        self._assigned = {m for g in self._sc["groups"] for m in g["members"]}
        self._current = None
        for raw in lines:
            line = raw.strip()
            if line:
                self._line(line)

    def _line(self, line: str) -> None:
        handlers = (
            self._comment, self._classdef, self._class,
            self._subgraph, self._end,
        )
        if any(h(line) for h in handlers):
            return
        if line.split()[0].lower() in KEYWORDS:
            return
        for nid, label in NODE_LABEL_RE.findall(line):
            parts = label.split("<br/>")
            if len(parts) > 1:
                self.taglines[nid] = parts[1].strip()
        self._edges(
            re.sub(r"\[[^\]]*\]|\(\([^)]*\)\)|\([^)]*\)|\{[^}]*\}", "", line)
        )
        self._node_defs(line)

    def _comment(self, line: str) -> bool:
        if not line.startswith("%%"):
            return False
        m = re.match(r"%%\s*圖例[：:]\s*(.*)", line)
        for part in re.split(r"[;；]", m.group(1)) if m else []:
            key, _, val = part.partition("=")
            if val:
                self.legend[key.strip()] = val.strip()
        return True

    def _classdef(self, line: str) -> bool:
        m = re.match(r"classDef\s+(\w+)\s+(.+)", line)
        if not m:
            return False
        props = {}
        for kv in m.group(2).split(","):
            key, _, val = kv.partition(":")
            if val:
                props[key.strip()] = val.strip()
        self.classdefs[m.group(1)] = props
        return True

    def _class(self, line: str) -> bool:
        m = re.match(r"class\s+([\w,\s]+?)\s+(\w+)\s*;?$", line)
        if not m or line.startswith("classDef"):
            return False
        for nid in re.split(r"[,\s]+", m.group(1).strip()):
            self.node_class[nid] = m.group(2)
        return True

    def _subgraph(self, line: str) -> bool:
        m = SUBGRAPH_RE.match(line)
        if not m:
            return False
        title = (m.group(2) or m.group(1)).strip()
        self._current = {"id": m.group(1), "title": title, "members": []}
        self._sc["groups"].append(self._current)
        return True

    def _end(self, line: str) -> bool:
        if line != "end":
            return False
        self._current = None
        return True

    def _add_node(self, nid: str) -> None:
        if nid not in self._sc["order"]:
            self._sc["order"].append(nid)
        if self._current is not None and nid not in self._assigned:
            self._current["members"].append(nid)
            self._assigned.add(nid)

    def _edges(self, clean: str) -> None:
        pos = 0
        while True:
            m = EDGE_RE.search(clean, pos)
            if not m:
                return
            label = (m.group("sl") or m.group("dl") or "").strip().strip('"')
            self._sc["edges"].append(
                {
                    "from": m.group("a"),
                    "to": m.group("b"),
                    "label": label,
                    "dashed": bool(m.group("dash")),
                }
            )
            self._add_node(m.group("a"))
            self._add_node(m.group("b"))
            pos = m.start("b")

    def _node_defs(self, line: str) -> None:
        for nid in re.findall(r"^\s*([A-Za-z]\w*)\s*[\[\(\{]", line):
            if nid.lower() not in KEYWORDS:
                self._add_node(nid)


# ---------- 路徑 ----------


def extract_paths(code_value: str) -> list[str]:
    """從「程式」欄取出反引號內像路徑的字串（保留順序，去重）。"""
    out = []
    for tok in re.findall(r"`([^`]+)`", code_value):
        tok = tok.strip().replace("\\", "/")
        if any(c in tok for c in " *<>{}()"):
            continue
        if "/" in tok or re.search(r"\.\w{1,5}$", tok):
            out.append(tok)
    return list(dict.fromkeys(out))


def list_entries(full: Path) -> dict:
    """列出資料夾第一層（略過隱藏項與快取），最多 MAX_ENTRIES 項。"""
    names = [
        p
        for p in full.iterdir()
        if not p.name.startswith(".")
        and p.name != "__pycache__"
        and p.suffix != ".pyc"
    ]
    names.sort(key=lambda p: (p.is_file(), p.name.lower()))
    return {
        "entries": [
            {"name": p.name, "kind": "dir" if p.is_dir() else "file"}
            for p in names[:MAX_ENTRIES]
        ],
        "more": max(0, len(names) - MAX_ENTRIES),
    }


def resolve_paths(root: Path, tokens: list[str]) -> list[dict]:
    """檢查路徑是否存在；沒有斜線的檔名相對於同欄出現過的目錄。"""
    dirs = [t.rstrip("/") for t in tokens if "/" in t]
    result = []
    for tok in tokens:
        rel = tok.rstrip("/")
        cands = [rel] + (
            [f"{d}/{tok}" for d in dirs] if "/" not in tok else []
        )
        found = next((c for c in cands if (root / c).exists()), None)
        if found is None:
            result.append(
                {"path": rel, "exists": False, "kind": "?", "entries": []}
            )
            continue
        full = root / found
        entry = {
            "path": found,
            "exists": True,
            "kind": "dir" if full.is_dir() else "file",
            "entries": [],
        }
        if full.is_dir():
            entry.update(list_entries(full))
        result.append(entry)
    return mark_inside(result)


def mark_inside(paths: list[dict]) -> list[dict]:
    """路徑落在同盒另一個已登記資料夾內時，標記 inside 為該資料夾。"""
    dirs = [p["path"] for p in paths if p["exists"] and p["kind"] == "dir"]
    for p in paths:
        parent = next((d for d in dirs if p["path"].startswith(d + "/")), None)
        if parent:
            p["inside"] = parent
    return paths


# ---------- 解析地圖 ----------


def box_from_section(sec: dict, parent: str, root: Path) -> dict | None:
    """章節是盒子（標題「ID 名稱」且有 Input／Output 等列）就回傳盒子。"""
    mb = BOX_RE.match(sec["title"])
    rows = table_rows(sec["lines"]) if mb else []
    if not any(k.lower() in FIELD_KEYS for k, _ in rows):
        return None
    fields, extras = {}, []
    for key, val in rows:
        norm = FIELD_KEYS.get(key.lower())
        if norm:
            fields[norm] = val
        elif key not in ("項目", "Item"):
            extras.append({"label": key, "value": val})
    return {
        "id": mb.group(1),
        "name": mb.group(2).strip(),
        "parent": parent,
        "fields": fields,
        "extras": extras,
        "paths": resolve_paths(root, extract_paths(fields.get("code", ""))),
    }


def parse_map(text: str, root: Path) -> dict:
    """把地圖 Markdown 解析成 boxes 與 mermaid 資訊。"""
    sections = split_sections(text)
    title = next((s["title"] for s in sections if s["level"] == 1), "概念地圖")
    m = re.search(r"(?:最後確認|Last confirmed)\s*[：:]\s*(\S+)", text)
    l0 = next(
        (
            "\n".join(s["lines"]).strip()
            for s in sections
            if s["level"] >= 2 and "L0" in s["title"]
        ),
        "",
    )
    boxes, parser, last_l1 = {}, MermaidParser(), ""
    for sec in sections:
        lvl = sec["level"]
        box = None
        if lvl in (3, 4):
            box = box_from_section(sec, last_l1 if lvl == 4 else "", root)
        if box:
            boxes[box["id"]] = box
            last_l1 = box["id"] if lvl == 3 else last_l1
        elif lvl <= 2:
            last_l1 = ""
        scope = (
            (box["id"] if box and lvl == 3 else last_l1) if lvl >= 3 else ""
        )
        for blk in sec["mermaid"]:
            parser.feed(scope, blk)
    for bid, tag in parser.taglines.items():
        if bid in boxes:
            boxes[bid]["tagline"] = tag
    return {
        "title": title,
        "confirmed": m.group(1) if m else "",
        "l0": l0,
        "boxes": boxes,
        "mermaid": parser,
    }


# ---------- 版面 ----------


def rank_columns(nodes: list[str], edges: list[dict]) -> list[list[str]]:
    """沒有分組時，依連線算欄位：忽略回饋線的最長路徑。"""
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for e in edges:
        if e["from"] in adj and e["to"] in adj and e["from"] != e["to"]:
            adj[e["from"]].append(e["to"])
    back = find_back_edges(nodes, adj)
    rank = {n: 0 for n in nodes}
    for _ in range(len(nodes)):
        changed = False
        for u in nodes:
            for v in adj[u]:
                if (u, v) not in back and rank[v] < rank[u] + 1:
                    rank[v] = rank[u] + 1
                    changed = True
        if not changed:
            break
    cols: dict[int, list[str]] = {}
    for n in nodes:
        cols.setdefault(rank[n], []).append(n)
    return [cols[k] for k in sorted(cols)]


def find_back_edges(nodes: list[str], adj: dict) -> set:
    """深度優先走訪，回傳會形成迴圈的回饋邊。"""
    state: dict[str, int] = {}
    back: set[tuple[str, str]] = set()

    def dfs(u: str) -> None:
        state[u] = 1
        for v in adj[u]:
            if state.get(v) == 1:
                back.add((u, v))
            elif v not in state:
                dfs(v)
        state[u] = 2

    for n in nodes:
        if n not in state:
            dfs(n)
    return back


def make_columns(
    nodes: list[str], groups: list[dict], edges: list[dict]
) -> list[dict]:
    """有分組時一組一欄，沒分組的節點依連線排欄。"""
    in_scope = set(nodes)
    columns: list[dict] = []
    used: set[str] = set()
    for g in groups:
        members = [n for n in g["members"] if n in in_scope and n not in used]
        if members:
            columns.append({"title": g["title"], "members": members})
            used.update(members)
    rest = [n for n in nodes if n not in used]
    if rest and not columns:
        columns = [
            {"title": None, "members": c} for c in rank_columns(rest, edges)
        ]
    elif rest:
        columns.append({"title": None, "members": rest})
    return columns


def layout_scope(
    nodes: list[str], groups: list[dict], edges: list[dict]
) -> dict:
    """算出一個範圍的節點座標與分組框。"""
    columns = make_columns(nodes, groups, edges)
    col_h = [
        len(c["members"]) * NODE_H + (len(c["members"]) - 1) * ROW_GAP
        for c in columns
    ]
    total = max(col_h) if col_h else 0
    pos, rects = {}, []
    for i, col in enumerate(columns):
        x = i * (NODE_W + COL_GAP) + GROUP_PAD
        y0 = (total - col_h[i]) / 2 + GROUP_PAD + GROUP_TITLE_H
        for j, nid in enumerate(col["members"]):
            pos[nid] = [x, y0 + j * (NODE_H + ROW_GAP)]
        if col["title"]:
            rects.append(
                {
                    "title": col["title"],
                    "x": x - GROUP_PAD,
                    "y": y0 - GROUP_PAD - GROUP_TITLE_H,
                    "w": NODE_W + 2 * GROUP_PAD,
                    "h": col_h[i] + 2 * GROUP_PAD + GROUP_TITLE_H,
                }
            )
    return {
        "nodes": [
            {"id": n, "x": p[0], "y": p[1], "w": NODE_W, "h": NODE_H}
            for n, p in pos.items()
        ],
        "groups": rects,
        "edges": [e for e in edges if e["from"] in pos and e["to"] in pos],
    }


# ---------- 組 model 與檢查 ----------


def legend_of(parser: MermaidParser) -> dict:
    """把 classDef 轉成圖例：類別 → 中文名稱與顏色。"""
    return {
        cls: {
            "label": parser.legend.get(cls, cls),
            "fill": props.get("fill", NEUTRAL["fill"]),
            "stroke": props.get("stroke", NEUTRAL["stroke"]),
        }
        for cls, props in parser.classdefs.items()
    }


def check_boxes(boxes: dict) -> list[str]:
    """檢查每個盒子的必要欄位與程式路徑。"""
    warns = []
    for bid, b in boxes.items():
        for need in ("input", "output", "guarantee"):
            if need not in b["fields"]:
                warns.append(f"{bid} {b['name']}：缺少 {need} 欄")
        for p in b["paths"]:
            if not p["exists"]:
                warns.append(
                    f"{bid} {b['name']}：程式路徑不存在 `{p['path']}`"
                )
    return warns


def build_scopes(boxes: dict, parser: MermaidParser, warns: list[str]) -> dict:
    """每個範圍（L1 或某盒的 L2）算版面，並檢查連線與盒子對得上。"""
    scopes: dict[str, dict] = {}
    for scope, sc in parser.scopes.items():
        name = scope or "L1"
        ids = [i for i, b in boxes.items() if b["parent"] == scope]

        # mermaid 的節點必須有對應的盒子標題
        warns += [
            f"範圍 {name}：mermaid 節點 {n} 沒有對應的盒子標題"
            for n in sc["order"]
            if n not in boxes
        ]
        if not ids:
            continue

        # 有連線圖的範圍裡，沒有任何連線的盒子要提醒
        linked = {e["from"] for e in sc["edges"]} | {
            e["to"] for e in sc["edges"]
        }
        warns += [
            f"{i} {boxes[i]['name']}：沒有任何連線"
            for i in ids
            if sc["edges"] and i not in linked
        ]
        scopes[scope] = layout_scope(ids, sc["groups"], sc["edges"])

    # 沒有 L1 連線圖時，仍要有 L1 範圍讓 viewer 有東西可畫
    if "" not in scopes:
        top = [i for i, b in boxes.items() if not b["parent"]]
        scopes[""] = layout_scope(top, [], [])
        warns.append("沒有 L1 的 mermaid 連線圖")
    return scopes


def build_model(parsed: dict, root: Path, map_path: Path) -> dict:
    """組出 viewer 用的 model，並收集警告。"""
    boxes, parser = parsed["boxes"], parsed["mermaid"]
    for bid, b in boxes.items():
        b["cls"] = parser.node_class.get(bid, "")
    warns = check_boxes(boxes)
    scopes = build_scopes(boxes, parser, warns)
    warns += uncovered(root, boxes)
    return {
        "title": parsed["title"],
        "confirmed": parsed["confirmed"],
        "l0": parsed["l0"],
        "legend": legend_of(parser),
        "boxes": boxes,
        "scopes": scopes,
        "warnings": warns,
        "map_path": str(map_path),
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def uncovered(root: Path, boxes: dict) -> list[str]:
    """已登記路徑的鄰近目錄裡，沒有任何盒子認領的項目。"""
    reg = {
        p["path"] for b in boxes.values() for p in b["paths"] if p["exists"]
    }
    parents: dict[str, int] = {}
    for p in reg:
        parent = Path(p).parent.as_posix()
        if parent != ".":
            parents[parent] = parents.get(parent, 0) + 1
    out = []
    for parent, n in sorted(parents.items()):
        if n >= 2 and (root / parent).is_dir():
            out += unclaimed(root, parent, reg)
    return out


def unclaimed(root: Path, parent: str, reg: set[str]) -> list[str]:
    """列出 parent 底下沒被 reg 任何路徑涵蓋的目錄或原始碼檔。"""
    out = []
    for child in sorted((root / parent).iterdir()):
        name = child.name
        if name.startswith(".") or name in ("__pycache__", "__init__.py"):
            continue
        if child.is_file() and child.suffix not in SOURCE_SUFFIXES:
            continue
        rel = f"{parent}/{name}"
        if any(
            r == rel or r.startswith(rel + "/") or rel.startswith(r + "/")
            for r in reg
        ):
            continue
        slash = "/" if child.is_dir() else ""
        out.append(
            f"未登記：`{rel}{slash}`（鄰近目錄 `{parent}/` 有其他項目已登記）"
        )
    return out


# ---------- 輸出 ----------


def write_drift(model: dict, out: Path) -> None:
    """寫 drift.md：地圖與現況的差異清單。"""
    lines = [
        f"# 對照現況（{model['generated']}）",
        "",
        f"地圖：`{model['map_path']}`，"
        f"最後確認：{model['confirmed'] or '未標示'}",
        "",
    ]
    if model["warnings"]:
        lines += [f"共 {len(model['warnings'])} 項：", ""]
        lines += [f"- {w}" for w in model["warnings"]]
    else:
        lines.append("沒有差異。")
    lines += [
        "",
        "本檔只檢查路徑存在、格式與連線是否對得上，"
        "不判斷程式依賴是否符合連線。",
        "",
    ]
    (out / "drift.md").write_text("\n".join(lines), encoding="utf-8")


def write_viewer(model: dict, out: Path) -> None:
    """把 model 嵌進 viewer 模板，寫成單一 HTML。"""
    tmpl = Path(__file__).resolve().parent.parent / "assets"
    html = (tmpl / "viewer.tmpl.html").read_text(encoding="utf-8")
    data = json.dumps(model, ensure_ascii=False).replace("</", "<\\/")
    (out / "viewer.html").write_text(
        html.replace("__MODEL__", data), encoding="utf-8"
    )


def main() -> int:
    """命令列入口。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="概念地圖 build")
    ap.add_argument("--root", type=Path, help="專案根（預設 git 根）")
    ap.add_argument("--map", type=Path, help="地圖 Markdown 路徑")
    ap.add_argument("--out", type=Path, help="輸出資料夾")
    args = ap.parse_args()
    root = (args.root or find_root(Path.cwd())).resolve()
    map_path = args.map or find_map(root)
    if map_path is None or not Path(map_path).is_file():
        print(
            "找不到地圖。請用 --map 指定，或依 SKILL.md 步驟 0 建立。",
            file=sys.stderr,
        )
        return 2
    map_path = Path(map_path).resolve()
    out = (args.out or root / ".claude" / "concept-map").resolve()
    out.mkdir(parents=True, exist_ok=True)
    text = map_path.read_text(encoding="utf-8", errors="replace")
    model = build_model(parse_map(text, root), root, map_path)
    (out / "model.json").write_text(
        json.dumps(model, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    write_drift(model, out)
    write_viewer(model, out)
    n_edges = sum(len(s["edges"]) for s in model["scopes"].values())
    print(f"地圖：{map_path}")
    print(
        f"盒子 {len(model['boxes'])}、連線 {n_edges}、"
        f"警告 {len(model['warnings'])}"
    )
    print(f"輸出：{out}")
    for w in model["warnings"][:15]:
        print(f"  - {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
