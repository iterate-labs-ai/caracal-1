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


def build(out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset
    from datasets.exceptions import DatasetNotFoundError

    rows = []
    for repo, split, tag in [
        ("MathArena/aime_2026", "train", "AIME 2026"),
        ("MathArena/hmmt_feb_2026", "train", "HMMT Feb 2026"),
        ("MathArena/aime_2025_I", "train", "AIME 2025 I"),
        ("MathArena/aime_2025_II", "train", "AIME 2025 II"),
        ("MathArena/hmmt_feb_2025", "train", "HMMT Feb 2025"),
    ]:
        try:
            ds = load_dataset(repo, split=split)
            for r in ds:
                r["contest"] = tag
                rows.append(r)
            print(f"loaded {repo}/{split}: {len(list(ds))} rows")
        except (FileNotFoundError, ValueError, ConnectionError, DatasetNotFoundError) as e:
            print(f"skip {repo}/{split}: {e}")
    if not rows:
        print("matharena: no source available")
        return 0

    n = 0
    with out_path.open("w") as f:
        for i, row in enumerate(rows):
            prompt = row.get("problem") or row.get("question") or row.get("statement") or ""
            gold = row.get("answer") or row.get("gold") or ""
            if not prompt or gold == "":
                continue
            f.write(
                json.dumps(
                    {
                        "id": row.get("id") or f"matharena_{i}",
                        "prompt": prompt,
                        "gold": str(gold).strip(),
                        "contest": row.get("contest"),
                        "year": row.get("year"),
                        "source": "matharena",
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
