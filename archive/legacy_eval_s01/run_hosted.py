"""Roda benches contra hosted LLMs via Kaggle AI quota ($10/dia, $100/mes).

Reusa loaders + prompt templates dos standalones. Em vez de model.generate(),
chama kbench.llms[tag].prompt(prompt). Score logic igual.

Models suportados (de kbench.llms.keys()):
- google/gemini-2.5-flash, google/gemini-2.5-pro, google/gemini-3-flash-preview
- anthropic/claude-sonnet-4
- meta/llama-3.1-70b
- qwen/*, deepseek/*, gemma/* (sem structured output)

Usage (Kaggle kernel):
    python eval/run_hosted.py --bench mmlu_security --llm google/gemini-2.5-flash --out hosted.json
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

LETTER_RE = re.compile(r"\b([A-D])\b")


def parse_letter(text):
    m = LETTER_RE.search(text.strip().upper())
    return m.group(1) if m else None


def prompt_hosted(llm, prompt, max_chars=512):
    """Wrapper unificado. Hosted models de kbench retornam .text na response."""
    out = llm.prompt(prompt)
    text = out if isinstance(out, str) else getattr(out, "text", str(out))
    return text[:max_chars]


LETTERS = ["A", "B", "C", "D"]


def _mcq_run(llm, qs, prompt_fn, gold_fn):
    n_correct, results = 0, []
    for i, q in enumerate(qs):
        prompt = prompt_fn(q)
        text = prompt_hosted(llm, prompt)
        pred_letter = parse_letter(text)
        pred = LETTERS.index(pred_letter) if pred_letter in LETTERS else -1
        gold = gold_fn(q)
        correct = pred == gold
        n_correct += int(correct)
        results.append({"i": i, "pred": pred, "gold": gold, "correct": correct})
        if (i + 1) % 50 == 0:
            logger.info(f"[{i + 1}/{len(qs)}] acc = {n_correct / (i + 1):.3f}")
    return {
        "n_total": len(qs),
        "n_correct": n_correct,
        "accuracy": n_correct / len(qs) if qs else 0.0,
        "results": results,
    }


def run_mmlu_security(llm):
    from eval.run_mmlu_security import PROMPT_TEMPLATE, load_mmlu_security

    qs = load_mmlu_security()

    def fmt(q):
        return PROMPT_TEMPLATE.format(
            question=q["question"],
            a=q["choices"][0],
            b=q["choices"][1],
            c=q["choices"][2],
            d=q["choices"][3],
        )

    return _mcq_run(llm, qs, fmt, lambda q: q["answer"])


def run_secqa(llm):
    from eval.run_mmlu_security import PROMPT_TEMPLATE
    from eval.run_secqa import load_secqa

    qs = load_secqa()

    def fmt(q):
        return PROMPT_TEMPLATE.format(
            question=q["question"],
            a=q["choices"][0],
            b=q["choices"][1],
            c=q["choices"][2],
            d=q["choices"][3],
        )

    return _mcq_run(llm, qs, fmt, lambda q: q["answer"])


def run_cybermetric(llm, tier=500):
    from eval.run_cybermetric import PROMPT_TEMPLATE, load_cybermetric

    qs = load_cybermetric(tier)

    def fmt(q):
        return PROMPT_TEMPLATE.format(
            question=q["question"],
            a=q["choices"][0],
            b=q["choices"][1],
            c=q["choices"][2],
            d=q["choices"][3],
        )

    return _mcq_run(llm, qs, fmt, lambda q: q["answer"])


def run_cti_bench(llm, max_n=500):
    from eval.run_cti_bench import CWE_RE, load_subset

    rows = load_subset("cti-rcm")[:max_n]
    n_correct, results = 0, []

    def norm(s):
        m = CWE_RE.search(s)
        return f"CWE-{int(m.group(1))}" if m else None

    for i, r in enumerate(rows):
        # Prompt oficial CTI-Bench ja vem na coluna - sem wrap
        text = prompt_hosted(llm, r["Prompt"], max_chars=256)
        pred = norm(text)
        gold = norm(r["GT"])
        correct = pred is not None and pred == gold
        n_correct += int(correct)
        results.append({"i": i, "pred": pred, "gold": gold, "correct": correct})
        if (i + 1) % 50 == 0:
            logger.info(f"[{i + 1}/{len(rows)}] acc = {n_correct / (i + 1):.3f}")
    return {
        "n_total": len(rows),
        "n_correct": n_correct,
        "accuracy": n_correct / len(rows) if rows else 0.0,
        "results": results,
    }


def run_probe(llm):
    from eval.run_probe import load_probes, score_cwe

    probes = load_probes()
    cwe_hits, cwe_total, results = 0, 0, []
    for i, p in enumerate(probes):
        text = prompt_hosted(llm, p["prompt"], max_chars=512)
        score = None
        if p.get("expected_cwe"):
            score = score_cwe(p["expected_cwe"], text)
            cwe_hits += score
            cwe_total += 1
        results.append({"i": i, "id": p["id"], "response": text[:200], "cwe_hit": score})
    return {
        "n_probes": len(probes),
        "cwe_hit_rate": cwe_hits / cwe_total if cwe_total else None,
        "cwe_total": cwe_total,
        "results": results,
    }


def run_humaneval(llm, limit=None):
    from eval.run_humaneval import GEN_PREFIX, check_solution, load_humaneval

    probs = load_humaneval()
    if limit:
        probs = probs[:limit]
    n_pass, results = 0, []
    for i, p in enumerate(probs):
        text = prompt_hosted(llm, GEN_PREFIX + p["prompt"], max_chars=2000)
        passed, reason = check_solution(p, text)
        n_pass += int(passed)
        results.append(
            {"task_id": p["task_id"], "passed": passed, "reason": None if passed else reason}
        )
        if (i + 1) % 25 == 0:
            logger.info(f"[{i + 1}/{len(probs)}] pass@1 = {n_pass / (i + 1):.3f}")
    return {
        "n_total": len(probs),
        "n_pass": n_pass,
        "pass_at_1": n_pass / len(probs) if probs else 0.0,
        "results": results,
    }


BENCHES = {
    "probe": run_probe,
    "mmlu_security": run_mmlu_security,
    "secqa": run_secqa,
    "cybermetric": run_cybermetric,
    "cti_bench": run_cti_bench,
    "humaneval": run_humaneval,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True, choices=list(BENCHES.keys()))
    parser.add_argument("--llm", required=True, help="kbench llm tag e.g. google/gemini-2.5-flash")
    parser.add_argument("--out", required=True, help="JSON output path")
    parser.add_argument(
        "--max-n", type=int, default=None, help="Limit n samples (cti_bench, humaneval)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import kaggle_benchmarks as kbench  # type: ignore[import-not-found]

    if args.llm not in kbench.llms:
        raise SystemExit(f"unknown llm {args.llm}. available: {sorted(kbench.llms.keys())}")
    llm = kbench.llms[args.llm]

    runner = BENCHES[args.bench]
    kw = {}
    if args.bench == "cti_bench" and args.max_n:
        kw["max_n"] = args.max_n
    elif args.bench == "humaneval" and args.max_n:
        kw["limit"] = args.max_n
    summary = runner(llm, **kw)
    summary["llm"] = args.llm
    summary["bench"] = args.bench
    Path(args.out).write_text(json.dumps(summary, indent=2))
    logger.info(f"[{args.llm}] {args.bench} acc={summary.get('accuracy', '?'):.3f} -> {args.out}")


if __name__ == "__main__":
    main()
