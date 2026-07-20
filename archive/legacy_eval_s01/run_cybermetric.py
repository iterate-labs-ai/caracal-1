"""CyberMetric MCQ - knowledge cyber em 9 dominios (pentest, net-sec, crypto, IR, etc).

20x mais signal que MMLU computer_security. Tiers: 80 / 500 / 2000 / 10000.
Default 2000 - bom trade-off entre signal e tempo (~25min T4).

Forward pass unico (shape-fixo paddable), TPU-friendly.

Dataset: github.com/cybermetric/CyberMetric (CyberMetric-{N}-v1.json)

Usage:
    python eval/run_cybermetric.py --adapter ./ckpt --out cybermetric.json --tier 2000
"""

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval._common import load_model  # noqa: E402
from eval.run_mmlu_security import letter_token_ids, pick_answer  # noqa: E402

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "The following is a multiple choice question about cyber security. "
    "Output ONLY the letter of the correct answer (A, B, C, or D).\n\n"
    "Question: {question}\n"
    "A. {a}\n"
    "B. {b}\n"
    "C. {c}\n"
    "D. {d}\n"
    "Answer:"
)

CACHE_DIR = Path("/tmp/cybermetric-cache")
SOURCE_BASE = "https://raw.githubusercontent.com/cybermetric/CyberMetric/main"


def load_cybermetric(tier=2000):
    """Baixa CyberMetric-{tier}-v1.json se nao tiver cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"CyberMetric-{tier}-v1.json"
    cache = CACHE_DIR / fname
    if not cache.exists():
        url = f"{SOURCE_BASE}/{fname}"
        logger.info(f"downloading {url}")
        urllib.request.urlretrieve(url, cache)
    raw = json.loads(cache.read_text())
    qs = raw.get("questions", raw) if isinstance(raw, dict) else raw
    return [
        {
            "question": q["question"],
            "choices": [q["answers"]["A"], q["answers"]["B"], q["answers"]["C"], q["answers"]["D"]],
            "answer": ord(q["solution"].strip().upper()) - ord("A"),
        }
        for q in qs
    ]


def run(model, tokenizer, device, tier=2000):
    """Retorna dict de metrics."""
    ids = letter_token_ids(tokenizer)
    qs = load_cybermetric(tier)
    logger.info(f"loaded {len(qs)} CyberMetric questions (tier={tier})")
    results, n_correct = [], 0
    for i, q in enumerate(qs):
        prompt = PROMPT_TEMPLATE.format(
            question=q["question"],
            a=q["choices"][0],
            b=q["choices"][1],
            c=q["choices"][2],
            d=q["choices"][3],
        )
        pred = pick_answer(model, tokenizer, prompt, device, ids)
        correct = pred == q["answer"]
        n_correct += int(correct)
        results.append({"i": i, "pred": pred, "gold": q["answer"], "correct": correct})
        if (i + 1) % 100 == 0:
            logger.info(f"[{i + 1}/{len(qs)}] running acc = {n_correct / (i + 1):.3f}")
    return {
        "tier": tier,
        "n_total": len(qs),
        "n_correct": n_correct,
        "accuracy": n_correct / len(qs) if qs else 0.0,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir (default base puro)")
    parser.add_argument("--out", required=True, help="JSON output path")
    parser.add_argument("--tier", type=int, default=2000, choices=[80, 500, 2000, 10000])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model, tokenizer, device = load_model(args.adapter)
    summary = run(model, tokenizer, device, args.tier)
    summary["adapter"] = args.adapter
    Path(args.out).write_text(json.dumps(summary, indent=2))
    logger.info(
        f"accuracy = {summary['accuracy']:.3f} ({summary['n_correct']}/{summary['n_total']}) -> {args.out}"
    )


if __name__ == "__main__":
    main()
