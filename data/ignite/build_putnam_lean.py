"""Build PutnamBench Lean 4 JSONL (arxiv 2407.11214).

Source: trishullab/PutnamBench on HF (640 Putnam undergrad hard theorems).
Each has Lean 4 signature; proof is what model must generate.

Usage:
    python data/ignite/build_putnam_lean.py --out data/ignite/putnam.jsonl

Output: {id, prompt, preamble, theorem, informal, year, source}
"""

import argparse
import json
from pathlib import Path


def build(out_path: Path) -> int:
    from datasets import load_dataset

    from datasets.exceptions import DatasetNotFoundError

    for repo, split in [("amitayusht/PutnamBench", "lean4"), ("amitayusht/PutnamBench", "train")]:
        try:
            ds = load_dataset(repo, split=split)
            print(f"loaded {repo} split={split}")
            break
        except (FileNotFoundError, ValueError, ConnectionError, DatasetNotFoundError) as e:
            print(f"skip {repo}/{split}: {e}")
    else:
        print("no PutnamBench source available")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as f:
        for i, row in enumerate(ds):
            theorem = row.get("theorem_statement") or row.get("formal_statement") or ""
            preamble = row.get("preamble") or row.get("imports") or "import Mathlib\n"
            informal = row.get("informal_statement") or row.get("problem") or ""
            if not theorem:
                continue
            prompt = (
                f"Prove the following Putnam theorem in Lean 4. "
                f"Output the tactic proof inside a ```lean code fence.\n\n"
                f"Informal: {informal}\n\n"
                f"Formal:\n```lean\n{theorem}\n```\n"
            )
            f.write(
                json.dumps(
                    {
                        "id": row.get("name") or f"putnam_{i}",
                        "prompt": prompt,
                        "preamble": preamble,
                        "theorem": theorem,
                        "informal": informal,
                        "year": row.get("year"),
                        "source": "putnambench-lean4",
                    }
                )
                + "\n"
            )
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/ignite/putnam.jsonl"))
    args = ap.parse_args()
    n = build(args.out)
    print(f"wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
