"""Build AIME 2024/2025 JSONL - held-in / held-out temporal split.

Sources:
- AIME 2024: AI-MO/aimo-validation-aime or Maxwell-Jia/AIME_2024 on HF
- AIME 2025: opencompass/AIME2025 on HF (or manual list post-cutoff)

Usage:
    python data/ignite/build_aime.py --out data/ignite/aime.jsonl

Output: {id, prompt, gold, year, source}
"""

import argparse
import json
from pathlib import Path

YEAR_SOURCES = {
    2024: [("Maxwell-Jia/AIME_2024", None, "train")],
    2025: [
        ("yentinglin/aime_2025", None, "train"),
        ("opencompass/AIME2025", "AIME2025-I", "test"),
        ("opencompass/AIME2025", "AIME2025-II", "test"),
    ],
}


def build(out_path: Path, years: list[int]) -> int:
    from datasets import load_dataset
    from datasets.exceptions import DatasetNotFoundError

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as f:
        for year in years:
            if year not in YEAR_SOURCES:
                continue
            ds = None
            for repo, config, split in YEAR_SOURCES[year]:
                try:
                    ds = (
                        load_dataset(repo, config, split=split)
                        if config
                        else load_dataset(repo, split=split)
                    )
                    print(f"aime {year}: loaded {repo}/{config or 'default'}/{split}")
                    break
                except (FileNotFoundError, ValueError, ConnectionError, DatasetNotFoundError) as e:
                    print(f"aime {year} skip {repo}: {e}")
            if ds is None:
                continue
            for i, row in enumerate(ds):
                prompt = row.get("Problem") or row.get("problem") or row.get("question") or ""
                gold = row.get("Answer") or row.get("answer") or ""
                if not prompt or gold == "":
                    continue
                f.write(
                    json.dumps(
                        {
                            "id": f"aime_{year}_{i}",
                            "prompt": prompt,
                            "gold": str(gold).strip(),
                            "year": year,
                            "source": repo,
                        }
                    )
                    + "\n"
                )
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/ignite/aime.jsonl"))
    ap.add_argument("--years", type=int, nargs="+", default=[2024, 2025])
    args = ap.parse_args()
    n = build(args.out, args.years)
    print(f"wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
