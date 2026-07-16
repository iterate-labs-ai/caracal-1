"""Ignite bench CLI. Runs subset of BENCH_REGISTRY, dumps JSON + McNemar vs baseline.

Usage:
    python -m eval.ignite.run_all_ignite \
        --base unsloth/Qwen2.5-3B-Instruct-bnb-4bit \
        --adapter path/to/adapter \
        --benches omni_math livecodebench matharena \
        --out results/ignite-eval.json
"""

import argparse
import json
from pathlib import Path

DEFAULT_BASE = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"


def load_model(base: str, adapter: str | None = None):
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        model_name=base, max_seq_length=4096, load_in_4bit=True, fast_inference=True
    )
    if adapter:
        model.load_adapter(adapter)
    return model, tok


def main():
    from .benches import BENCH_REGISTRY
    from .stats import mcnemar_test

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--benches", nargs="+", default=list(BENCH_REGISTRY.keys()))
    ap.add_argument("--n-omni-math", type=int, default=200)
    ap.add_argument("--n-livecodebench", type=int, default=100)
    ap.add_argument("--n-bigcodebench", type=int, default=100)
    ap.add_argument("--n-matharena", type=int, default=50)
    ap.add_argument("--n-aime", type=int, default=30)
    ap.add_argument("--n-putnam", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("results/ignite-eval.json"))
    ap.add_argument("--compare-with", type=Path, default=None)
    args = ap.parse_args()

    print(f"[ignite] loading base={args.base} adapter={args.adapter}")
    model, tok = load_model(args.base, args.adapter)

    n_map = {
        "omni_math": args.n_omni_math,
        "livecodebench": args.n_livecodebench,
        "bigcodebench": args.n_bigcodebench,
        "matharena": args.n_matharena,
        "aime": args.n_aime,
        "putnam_lean": args.n_putnam,
    }

    results: dict = {"base": args.base, "adapter": args.adapter}
    for name in args.benches:
        if name not in BENCH_REGISTRY:
            print(f"skip unknown: {name}")
            continue
        print(f"\n=== {name} ===")
        results[name] = BENCH_REGISTRY[name](model, tok, n=n_map.get(name, 100))

    if args.compare_with and args.compare_with.exists():
        prev = json.loads(args.compare_with.read_text())
        cmp_out = {}
        for bench, res in results.items():
            if not isinstance(res, dict) or "per_sample" not in res:
                continue
            prev_bench = prev.get(bench)
            if not isinstance(prev_bench, dict) or "per_sample" not in prev_bench:
                continue
            curr_c = [s["correct"] for s in res["per_sample"]]
            prev_c = [s["correct"] for s in prev_bench["per_sample"]]
            if len(curr_c) == len(prev_c):
                cmp_out[bench] = mcnemar_test(prev_c, curr_c)
        if cmp_out:
            results["mcnemar_vs_prev"] = cmp_out

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[ignite] wrote {args.out}")
    _print_summary(results)


def _print_summary(results: dict):
    print("\n=== SUMMARY ===")
    for bench, res in results.items():
        if bench in ("base", "adapter", "mcnemar_vs_prev") or not isinstance(res, dict):
            continue
        if "accuracy" in res:
            print(f"  {bench:20s}: {res['accuracy'] * 100:5.1f}%  n={res.get('n', 0):4d}")
    if "mcnemar_vs_prev" in results:
        print("\n  McNemar vs previous:")
        for k, v in results["mcnemar_vs_prev"].items():
            sig = " *" if v["p_value"] < 0.05 else ""
            print(f"    {k}: chi2={v['chi2']:.2f} p={v['p_value']:.4f} g={v['cohens_g']:.3f}{sig}")


if __name__ == "__main__":
    main()
