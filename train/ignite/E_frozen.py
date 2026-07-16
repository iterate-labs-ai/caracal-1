"""Cond E - ablation: inner frozen, only outer trains.

Tests whether outer-loop-only recursion suffices (contrast Cond C which trains
both). Inner v_0 frozen throughout, only mutation scaffolding evolves.
"""

import argparse
from pathlib import Path

from .C_rsi_outer import outer_loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    ap.add_argument("--dataset-train", type=Path, required=True)
    ap.add_argument("--dataset-dev", type=Path, required=True)
    ap.add_argument("--dataset-val", type=Path, required=True)
    ap.add_argument("--bench", choices=["math", "code", "lean"], default="math")
    ap.add_argument("--bench-name", default="omni_math")
    ap.add_argument("--gens", type=int, default=4)
    ap.add_argument("--cands", type=int, default=4)
    ap.add_argument("--steps", type=int, default=0, help="0 = no inner training (frozen)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    outer_loop(
        base=args.base,
        v0_adapter=None,
        dataset_train=args.dataset_train,
        dataset_dev=args.dataset_dev,
        dataset_val=args.dataset_val,
        bench=args.bench,
        bench_name=args.bench_name,
        gens=args.gens,
        cands=args.cands,
        steps=args.steps,
        out_root=args.out,
    )


if __name__ == "__main__":
    main()
