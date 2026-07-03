"""Full benchmark runner Caracal s07.

Roda CTI-Bench RCM + CyberMetric + SecQA + estatísticas.

Usage:
    python eval/s07/bench_runner.py \\
        --model pedroafonso2/caracal-s07-rl \\
        --benches cti_bench cybermetric secqa \\
        --out results/s07-v1.json
"""

import argparse
import csv
import json
import re
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

CTI_BENCH_HF_BASE = "https://huggingface.co/datasets/AI4Sec/cti-bench/resolve/main"
CWE_RE = re.compile(r"CWE-?(\d{1,4})", re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{(CWE-?\d{1,4})\}", re.IGNORECASE)


def normalize_cwe(text: str) -> str | None:
    m = BOXED_RE.search(text)
    if m:
        inner = CWE_RE.search(m.group(1))
        if inner:
            return f"CWE-{int(inner.group(1))}"
    for line in reversed(text.splitlines()):
        m = CWE_RE.search(line)
        if m:
            return f"CWE-{int(m.group(1))}"
    return None


def load_cti_rcm(subset: str = "cti-rcm") -> list[dict]:
    url = f"{CTI_BENCH_HF_BASE}/{subset}.tsv"
    fd, tmp = tempfile.mkstemp(suffix=".tsv")
    import os

    os.close(fd)
    try:
        urllib.request.urlretrieve(url, tmp)
        with open(tmp) as f:
            return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    finally:
        os.unlink(tmp)


def bootstrap_ci(
    scores: list[int], ci: float = 0.95, n_resamples: int = 10_000
) -> tuple[float, float]:
    arr = np.array(scores)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    boots = [arr[np.random.randint(0, n, n)].mean() for _ in range(n_resamples)]
    alpha = 1 - ci
    return float(np.percentile(boots, 100 * alpha / 2)), float(
        np.percentile(boots, 100 * (1 - alpha / 2))
    )


def mcnemar_test(a_correct: list[int], b_correct: list[int]) -> dict:
    """McNemar paired test - a and b are 0/1 arrays same length."""
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


def run_generate(model, tokenizer, prompt: str, max_new: int = 256) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)


def eval_cti_rcm(model, tokenizer, n: int) -> dict:
    rows = load_cti_rcm("cti-rcm")[:n]
    correct = []
    per_sample = []
    for i, r in enumerate(rows):
        prompt = r["Prompt"]
        resp = run_generate(model, tokenizer, prompt, max_new=256)
        pred = normalize_cwe(resp)
        gold = normalize_cwe(r["GT"])
        ok = int(pred is not None and pred == gold)
        correct.append(ok)
        per_sample.append({"i": i, "pred": pred, "gold": gold, "correct": ok})
        if (i + 1) % 100 == 0:
            print(f"[cti_rcm {i + 1}/{len(rows)}] acc = {sum(correct) / (i + 1):.3f}")
    acc = sum(correct) / len(correct) if correct else 0.0
    ci_lo, ci_hi = bootstrap_ci(correct)
    return {
        "n": len(correct),
        "accuracy": acc,
        "ci_95_low": ci_lo,
        "ci_95_high": ci_hi,
        "per_sample": per_sample,
    }


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-cti-rcm", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("results/s07-eval.json"))
    ap.add_argument(
        "--compare-with", default=None, help="Path to previous results JSON for McNemar"
    )
    args = ap.parse_args()

    print(f"[bench] loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": "cuda:0"}
    )
    model.eval()

    results = {"model": args.model}
    results["cti_rcm"] = eval_cti_rcm(model, tok, args.n_cti_rcm)
    print(
        f"[bench] CTI-RCM = {results['cti_rcm']['accuracy']:.3f} [CI {results['cti_rcm']['ci_95_low']:.3f}-{results['cti_rcm']['ci_95_high']:.3f}]"
    )

    if args.compare_with and Path(args.compare_with).exists():
        prev = json.loads(Path(args.compare_with).read_text())
        curr_correct = [s["correct"] for s in results["cti_rcm"]["per_sample"]]
        prev_cti = prev.get("cti_rcm")
        prev_correct = [s["correct"] for s in prev_cti["per_sample"]] if prev_cti else []
        if prev_correct and len(curr_correct) == len(prev_correct):
            results["mcnemar_vs_prev"] = mcnemar_test(prev_correct, curr_correct)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"[bench] wrote {args.out}")


if __name__ == "__main__":
    main()
