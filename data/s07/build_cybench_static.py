"""Extract Cybench (arxiv 2408.08926) static offline subset pra Kaggle.

Cybench full = 40 pro CTFs Docker-based. Static subset:
- Categorias suportadas: crypto, forensics, reverse, misc
- Skip: pwn (remote), web (target http)
- Cada task: task_description (yaml) + files_snippet (head 4KB primary file)
- Output: JSONL pronto pra `datasets.load_dataset(json)` ou HF upload

Usage:
    git clone --recursive https://github.com/andyzorigin/cybench.git /tmp/cybench
    python data/s07/build_cybench_static.py \\
        --cybench-root /tmp/cybench \\
        --out data/cybench-static.jsonl

Upload to HF depois:
    huggingface-cli upload iterate-labs-ai/cybench-static data/cybench-static.jsonl \\
        --repo-type dataset
"""

import argparse
import json
from pathlib import Path

import yaml

STATIC_CATEGORIES = {"crypto", "forensics", "reversing", "reverse", "misc"}
SKIP_CATEGORIES = {"pwn", "web", "binary"}

FILE_EXT_PRIORITY = [
    ".py",
    ".c",
    ".cpp",
    ".rs",
    ".go",
    ".txt",
    ".md",
    ".yaml",
    ".json",
    ".pcap",
    ".pcapng",
    ".s",
    ".asm",
]
MAX_SNIPPET_BYTES = 4096


def _find_task_yaml(task_dir: Path) -> Path | None:
    for candidate in ["task.yaml", "metadata/task.yaml", "challenge.yaml"]:
        p = task_dir / candidate
        if p.exists():
            return p
    return None


def _extract_file_snippet(task_dir: Path) -> str:
    for ext in FILE_EXT_PRIORITY:
        for f in sorted(task_dir.rglob(f"*{ext}")):
            if f.stat().st_size == 0:
                continue
            try:
                text = f.read_text(errors="replace")
            except OSError:
                continue
            snip = text[:MAX_SNIPPET_BYTES]
            return f"# File: {f.relative_to(task_dir)}\n{snip}"
    return "(no static file found)"


def _parse_task(task_dir: Path) -> dict | None:
    task_yaml = _find_task_yaml(task_dir)
    if task_yaml is None:
        return None
    try:
        meta = yaml.safe_load(task_yaml.read_text())
    except (yaml.YAMLError, OSError):
        return None
    if not isinstance(meta, dict):
        return None

    category = str(meta.get("category", "")).lower()
    if category in SKIP_CATEGORIES:
        return None
    if category not in STATIC_CATEGORIES and "categories" not in meta:
        return None

    flag = meta.get("flag") or meta.get("answer") or ""
    if not flag:
        return None

    return {
        "task_name": task_dir.name,
        "category": category,
        "description": meta.get("description") or meta.get("prompt") or "",
        "files_snippet": _extract_file_snippet(task_dir),
        "flag": str(flag).strip(),
    }


def walk_cybench(cybench_root: Path) -> list[dict]:
    rows = []
    for task_dir in cybench_root.rglob("*"):
        if not task_dir.is_dir():
            continue
        row = _parse_task(task_dir)
        if row:
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cybench-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/cybench-static.jsonl"))
    args = ap.parse_args()

    if not args.cybench_root.exists():
        raise SystemExit(f"cybench-root {args.cybench_root} does not exist")

    rows = walk_cybench(args.cybench_root)
    print(f"extracted {len(rows)} static tasks (skipped pwn/web/binary)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {args.out}")

    by_cat: dict[str, int] = {}
    for r in rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    print("by category:", by_cat)


if __name__ == "__main__":
    main()
