"""Build BigCodeBench-Hard JSONL (arxiv 2406.15877).

Source: bigcode/bigcodebench-hard on HF. 148 challenging function calls
with complex library dependencies. Distinct dist from LiveCodeBench.

Usage:
    python data/ignite/build_bigcodebench.py --out data/ignite/bcb_hard.jsonl

Output schema: {id, prompt, tests, entry_point, libs, source}
"""

import argparse
import json
from pathlib import Path


def build(out_path: Path, split: str = "v0.1.4") -> int:
    from datasets import load_dataset

    ds = load_dataset("bigcode/bigcodebench-hard", split=split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as f:
        for i, row in enumerate(ds):
            prompt = row.get("complete_prompt") or row.get("instruct_prompt") or ""
            tests = row.get("test") or ""
            entry = row.get("entry_point") or ""
            if not prompt or not tests:
                continue
            f.write(
                json.dumps(
                    {
                        "id": row.get("task_id") or f"bcb_{i}",
                        "prompt": prompt,
                        "tests": [tests],
                        "entry_point": entry,
                        "libs": row.get("libs", []),
                        "source": "bigcodebench-hard",
                    }
                )
                + "\n"
            )
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/ignite/bcb_hard.jsonl"))
    ap.add_argument("--split", default="v0.1.4")
    args = ap.parse_args()
    n = build(args.out, args.split)
    print(f"wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
