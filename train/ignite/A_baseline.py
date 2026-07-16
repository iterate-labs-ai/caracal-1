"""Cond A - static baseline (no recursion).

Eval v_0 (Qwen2.5-3B-Instruct base) on all held-out benches. No training.
Used as reference for L1 delta measurement.
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    ap.add_argument("--benches", nargs="+", default=["omni_math", "livecodebench", "matharena"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("results/cond_A.json"))
    args = ap.parse_args()

    from unsloth import FastLanguageModel

    from eval.ignite.benches import BENCH_REGISTRY

    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.base, max_seq_length=4096, load_in_4bit=True, fast_inference=True
    )

    results: dict = {"cond": "A", "model": args.base}
    for name in args.benches:
        if name not in BENCH_REGISTRY:
            print(f"skip unknown: {name}")
            continue
        print(f"=== {name} ===")
        results[name] = BENCH_REGISTRY[name](model, tok, n=args.n)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
