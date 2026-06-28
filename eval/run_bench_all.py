"""Orchestrator - roda probe + cybergym + humaneval + mmlu_security + baseline.

Modelo carregado uma vez via load_model() (caro), reutilizado entre evals.
Cada eval gera JSON proprio em out_dir. No fim, agrega summary.json.

Usage:
    python eval/run_bench_all.py --adapter ./ckpt --out-dir ./bench-s01
    python eval/run_bench_all.py --no-adapter --out-dir ./bench-base
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval._common import load_model  # noqa: E402

logger = logging.getLogger(__name__)


def _step(name, fn, summary):
    t0 = time.time()
    logger.info(f"=== {name} START ===")
    try:
        out = fn()
        elapsed = time.time() - t0
        summary[name] = {"status": "ok", "elapsed_s": round(elapsed, 1), "metrics": out}
        logger.info(f"=== {name} OK ({elapsed:.0f}s) ===")
    except Exception as e:
        elapsed = time.time() - t0
        summary[name] = {"status": "fail", "elapsed_s": round(elapsed, 1), "error": str(e)}
        logger.exception(f"=== {name} FAIL ({elapsed:.0f}s): {e} ===")


def run_probe(model, tokenizer, device, out_dir):
    from eval.run_probe import compute_perplexity, generate, load_probes, score_cwe

    probes = load_probes()
    ppls, cwe_hits, cwe_total = [], 0, 0
    for p in probes:
        response = generate(model, tokenizer, p["prompt"], device, max_new=64)
        ppl = compute_perplexity(model, tokenizer, (p["prompt"] + response)[:2048], device)
        ppls.append(ppl)
        if p.get("expected_cwe"):
            cwe_hits += score_cwe(p["expected_cwe"], response)
            cwe_total += 1
    out = {
        "mean_ppl": sum(ppls) / len(ppls),
        "cwe_hit_rate": cwe_hits / cwe_total if cwe_total else None,
        "n_probes": len(probes),
    }
    (out_dir / "probe.json").write_text(json.dumps(out, indent=2))
    return out


def run_mmlu(model, tokenizer, device, out_dir):
    from eval.run_mmlu_security import PROMPT_TEMPLATE, load_mmlu_security, pick_answer

    letter_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in ["A", "B", "C", "D"]]
    qs = load_mmlu_security()
    n_correct = 0
    for q in qs:
        prompt = PROMPT_TEMPLATE.format(
            question=q["question"],
            a=q["choices"][0],
            b=q["choices"][1],
            c=q["choices"][2],
            d=q["choices"][3],
        )
        if pick_answer(model, tokenizer, prompt, device, letter_ids) == q["answer"]:
            n_correct += 1
    out = {"accuracy": n_correct / len(qs), "n_total": len(qs), "n_correct": n_correct}
    (out_dir / "mmlu_security.json").write_text(json.dumps(out, indent=2))
    return out


def run_humaneval(model, tokenizer, device, out_dir, limit):
    from eval.run_humaneval import check_solution, generate_completion, load_humaneval

    probs = load_humaneval()
    if limit:
        probs = probs[:limit]
    n_pass = 0
    for p in probs:
        comp = generate_completion(model, tokenizer, p["prompt"], device, max_new=384)
        passed, _ = check_solution(p, comp)
        n_pass += int(passed)
    out = {"pass_at_1": n_pass / len(probs), "n_total": len(probs), "n_pass": n_pass}
    (out_dir / "humaneval.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--skip-humaneval", action="store_true", help="HumanEval pesa, skip pra TPU rapido"
    )
    parser.add_argument(
        "--humaneval-limit", type=int, default=None, help="limit HumanEval n problems"
    )
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--skip-mmlu", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter = None if args.no_adapter else args.adapter
    t_load = time.time()
    model, tokenizer, device = load_model(adapter)
    logger.info(f"loaded model device={device} adapter={adapter} in {time.time() - t_load:.0f}s")

    summary = {"adapter": adapter, "device": str(device), "started_unix": int(time.time())}

    if not args.skip_probe:
        _step("probe", lambda: run_probe(model, tokenizer, device, out_dir), summary)
    if not args.skip_mmlu:
        _step("mmlu_security", lambda: run_mmlu(model, tokenizer, device, out_dir), summary)
    if not args.skip_humaneval:
        _step(
            "humaneval",
            lambda: run_humaneval(model, tokenizer, device, out_dir, args.humaneval_limit),
            summary,
        )

    summary["finished_unix"] = int(time.time())
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"DONE -> {out_dir}/summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
