"""Extract NYU CTF Bench (arxiv 2406.05590) static offline subset pra Kaggle.

NYU CTF full = 200 CTFs Docker-based. Static subset:
- Categorias suportadas: crypto, forensics, rev
- Skip: pwn, web
- Cada task: challenge.json meta + files_snippet primary artifact
- Output: JSONL pra HF upload

Usage:
    git clone https://github.com/NYU-LLM-CTF/NYU_CTF_Bench.git /tmp/nyu_ctf
    python data/s07/build_nyu_ctf_static.py \\
        --nyu-root /tmp/nyu_ctf \\
        --out data/nyu_ctf-static.jsonl
"""

import argparse
import json
from pathlib import Path

STATIC_CATEGORIES = {"crypto", "forensics", "rev", "reverse", "reversing", "misc"}
SKIP_CATEGORIES = {"pwn", "web"}

FILE_EXT_PRIORITY = [
    ".py",
    ".c",
    ".cpp",
    ".rs",
    ".txt",
    ".md",
    ".json",
    ".pcap",
    ".pcapng",
    ".s",
    ".asm",
]
MAX_SNIPPET_BYTES = 4096


def _extract_file_snippet(task_dir: Path) -> str:
    for ext in FILE_EXT_PRIORITY:
        for f in sorted(task_dir.rglob(f"*{ext}")):
            if f.name in {"challenge.json", "solution.json"}:
                continue
            if f.stat().st_size == 0:
                continue
            try:
                text = f.read_text(errors="replace")
            except OSError:
                continue
            return f"# File: {f.relative_to(task_dir)}\n{text[:MAX_SNIPPET_BYTES]}"
    return "(no static file found)"


def _parse_task(task_dir: Path) -> dict | None:
    meta_path = task_dir / "challenge.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    category = str(meta.get("category", "")).lower()
    if category in SKIP_CATEGORIES:
        return None
    if category not in STATIC_CATEGORIES:
        return None

    flag = meta.get("flag") or ""
    if not flag:
        return None

    return {
        "task_name": task_dir.name,
        "category": category,
        "description": meta.get("description") or "",
        "files_snippet": _extract_file_snippet(task_dir),
        "flag": str(flag).strip(),
    }


def walk_nyu(nyu_root: Path) -> list[dict]:
    rows = []
    for task_dir in nyu_root.rglob("*"):
        if not task_dir.is_dir():
            continue
        row = _parse_task(task_dir)
        if row:
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nyu-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/nyu_ctf-static.jsonl"))
    args = ap.parse_args()

    if not args.nyu_root.exists():
        raise SystemExit(f"nyu-root {args.nyu_root} does not exist")

    rows = walk_nyu(args.nyu_root)
    print(f"extracted {len(rows)} static tasks (skipped pwn/web)")
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
