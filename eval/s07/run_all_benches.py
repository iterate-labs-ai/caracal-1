"""Roda toda suite de benchmarks Caracal s07.

Usage:
    python eval/s07/run_all_benches.py \\
        --model pedroafonso2/caracal-s07-rl \\
        --benches cti_bench cybermetric secqa mmlu_security cwe_prediction \\
        --out results/s07-full-eval.json

    # Comparar com resultados anteriores (McNemar paired):
    python eval/s07/run_all_benches.py \\
        --model ... \\
        --compare-with results/s05-baseline.json
"""

import argparse
import json
from pathlib import Path

from .benches import BENCH_REGISTRY


def mcnemar_test(a_correct: list[int], b_correct: list[int]) -> dict:
    from scipy.stats.contingency import mcnemar

    b_only = sum(1 for x, y in zip(a_correct, b_correct, strict=False) if x == 0 and y == 1)
    a_only = sum(1 for x, y in zip(a_correct, b_correct, strict=False) if x == 1 and y == 0)
    both = sum(1 for x, y in zip(a_correct, b_correct, strict=False) if x == 1 and y == 1)
    none = sum(1 for x, y in zip(a_correct, b_correct, strict=False) if x == 0 and y == 0)
    table = [[both, a_only], [b_only, none]]
    result = mcnemar(table, exact=(b_only + a_only) < 25, correction=True)
    cohens_g = (b_only - a_only) / max(b_only + a_only, 1)
    return {
        "chi2": float(result.statistic),
        "p_value": float(result.pvalue),
        "cohens_g": float(cohens_g),
        "b_only": b_only,
        "a_only": a_only,
    }


DEFAULT_BASE = "Qwen/Qwen2.5-3B-Instruct"


def load_model(name: str, adapter_path: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, device_map={"": "cuda:0"}
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        tok = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    model.eval()
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_BASE)
    ap.add_argument("--adapter", default=None, help="Local path to LoRA adapter (Kaggle input)")
    ap.add_argument("--benches", nargs="+", default=list(BENCH_REGISTRY.keys()))
    ap.add_argument("--n-cti-rcm", type=int, default=1000)
    ap.add_argument("--n-cti-mcq", type=int, default=500)
    ap.add_argument("--cybermetric-tier", type=int, default=500)
    ap.add_argument("--n-secbench", type=int, default=1000)
    ap.add_argument("--n-mmlu", type=int, default=100)
    ap.add_argument("--n-cwe-pred", type=int, default=500)
    ap.add_argument("--n-seceval", type=int, default=500)
    ap.add_argument("--n-cybersoceval", type=int, default=500)
    ap.add_argument("--n-cs-eval", type=int, default=500)
    ap.add_argument("--n-cybercert", type=int, default=300)
    ap.add_argument("--n-primevul", type=int, default=500)
    ap.add_argument("--n-cybench", type=int, default=40)
    ap.add_argument("--n-nyu-ctf", type=int, default=60)
    ap.add_argument("--out", type=Path, default=Path("results/s07-eval.json"))
    ap.add_argument("--compare-with", type=Path, default=None)
    args = ap.parse_args()

    print(f"[bench] loading base={args.model} adapter={args.adapter}")
    model, tok = load_model(args.model, args.adapter)

    bench_kwargs = {
        "cti_bench": {"n_rcm": args.n_cti_rcm, "n_mcq": args.n_cti_mcq},
        "cybermetric": {"tier": args.cybermetric_tier},
        "secqa": {},
        "secbench": {"n_mcq": args.n_secbench},
        "mmlu_security": {"n": args.n_mmlu},
        "cwe_prediction": {"n": args.n_cwe_pred},
        "seceval": {"n": args.n_seceval},
        "cybersoceval": {"n": args.n_cybersoceval},
        "cs_eval": {"n": args.n_cs_eval},
        "cybercert": {"n": args.n_cybercert},
        "primevul": {"n": args.n_primevul},
        "cybench_kaggle": {"n": args.n_cybench},
        "nyu_ctf_kaggle": {"n": args.n_nyu_ctf},
    }

    results: dict = {"model": args.model, "adapter": args.adapter}
    for name in args.benches:
        if name not in BENCH_REGISTRY:
            print(f"[bench] unknown bench {name}, skip")
            continue
        print(f"\n=== {name} ===")
        results[name] = BENCH_REGISTRY[name](model, tok, **bench_kwargs.get(name, {}))

    if args.compare_with and args.compare_with.exists():
        prev = json.loads(args.compare_with.read_text())
        cmp_out = {}
        for bench, res in results.items():
            if not isinstance(res, dict):
                continue
            prev_bench = prev.get(bench)
            if not prev_bench:
                continue
            for sub in res:
                curr = res[sub]
                prev_sub = prev_bench.get(sub) if isinstance(prev_bench, dict) else None
                if not (
                    isinstance(curr, dict)
                    and "per_sample" in curr
                    and isinstance(prev_sub, dict)
                    and "per_sample" in prev_sub
                ):
                    continue
                curr_c = [s["correct"] for s in curr["per_sample"]]
                prev_c = [s["correct"] for s in prev_sub["per_sample"]]
                if len(curr_c) == len(prev_c):
                    cmp_out[f"{bench}.{sub}"] = mcnemar_test(prev_c, curr_c)
        if cmp_out:
            results["mcnemar_vs_prev"] = cmp_out

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n[bench] wrote {args.out}")
    _print_summary(results)


def _print_summary(results: dict):
    print("\n=== SUMMARY ===")
    for bench, res in results.items():
        if bench in ("model", "mcnemar_vs_prev") or not isinstance(res, dict):
            continue
        if "accuracy" in res:
            print(f"  {bench}: {res['accuracy']:.3f} n={res['n']}")
        else:
            for sub, sub_res in res.items():
                if isinstance(sub_res, dict) and "accuracy" in sub_res:
                    print(f"  {bench}.{sub}: {sub_res['accuracy']:.3f} n={sub_res['n']}")
    if "mcnemar_vs_prev" in results:
        print("\n  McNemar vs previous:")
        for k, v in results["mcnemar_vs_prev"].items():
            sig = " *" if v["p_value"] < 0.05 else ""
            print(f"    {k}: chi2={v['chi2']:.2f} p={v['p_value']:.4f} g={v['cohens_g']:.3f}{sig}")


if __name__ == "__main__":
    main()
