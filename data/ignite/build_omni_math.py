"""Build canonical OMNI-MATH JSONL (arxiv 2410.07985).

Source: KbsdJames/Omni-MATH on HF (4428 Olympiad problems, exact-answer verifiable).

Usage:
    python data/ignite/build_omni_math.py --out data/ignite/omni_math.jsonl
    huggingface-cli upload iterate-labs-ai/ignite-omni-math data/ignite/omni_math.jsonl --repo-type dataset

Output schema: {id, prompt, gold, difficulty, subject, source}
"""

import argparse
import json
from pathlib import Path


def build(out_path: Path, split: str = "test") -> int:
    from datasets import load_dataset

    ds = load_dataset("KbsdJames/Omni-MATH", split=split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as f:
        for i, row in enumerate(ds):
            prompt = row.get("problem") or row.get("question") or ""
            gold = row.get("answer") or row.get("solution") or ""
            if not prompt or not gold:
                continue
            f.write(
                json.dumps(
                    {
                        "id": f"omni_math_{i}",
                        "prompt": prompt,
                        "gold": str(gold).strip(),
                        "difficulty": row.get("difficulty"),
                        "subject": row.get("domain") or row.get("subject"),
                        "source": row.get("source", "omni-math"),
                    }
                )
                + "\n"
            )
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/ignite/omni_math.jsonl"))
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    n = build(args.out, args.split)
    print(f"wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
