"""Build MathArena live rolling bench JSONL (arxiv 2505.23281 UNVERIFIED).

Source: MathArena live tracks AIME/HMMT/USAMO/IMO fresh contests.
Zero-contamination via rolling ingestion.

Fallback: manual JSONL of AIME 2025 + HMMT 2026 problems + integer answers
from official releases when HF dataset absent.

Usage:
    python data/ignite/build_matharena.py --out data/ignite/matharena.jsonl

Output: {id, prompt, gold, contest, year, source}
"""

import argparse
import json
from pathlib import Path

MANUAL_FALLBACK_URL = (
    "https://raw.githubusercontent.com/mathematicalarena/matharena/main/problems.jsonl"
)


def build(out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset

        ds = load_dataset("MathArena/matharena-live", split="test")
        rows = list(ds)
    except (FileNotFoundError, ValueError, ConnectionError):
        import urllib.request

        try:
            with urllib.request.urlopen(MANUAL_FALLBACK_URL, timeout=30) as r:
                rows = [json.loads(line) for line in r.read().decode().splitlines() if line.strip()]
        except (OSError, ValueError) as e:
            print(f"matharena fallback failed: {e}")
            return 0

    n = 0
    with out_path.open("w") as f:
        for i, row in enumerate(rows):
            prompt = row.get("problem") or row.get("question") or ""
            gold = row.get("answer") or ""
            if not prompt or not gold:
                continue
            f.write(
                json.dumps(
                    {
                        "id": row.get("id") or f"matharena_{i}",
                        "prompt": prompt,
                        "gold": str(gold).strip(),
                        "contest": row.get("contest"),
                        "year": row.get("year"),
                        "source": "matharena-live",
                    }
                )
                + "\n"
            )
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/ignite/matharena.jsonl"))
    args = ap.parse_args()
    n = build(args.out)
    print(f"wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
