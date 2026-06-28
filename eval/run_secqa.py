"""SecQA v1+v2 MCQ - cyber comprehension intro + advanced.

Sanity-check rapido (~5min T4). 242 MCQ no total (110 v1 + 132 v2).
Forward pass unico, shape-fixo paddable.

Dataset: zefang-liu/secqa no HuggingFace.

Usage:
    python eval/run_secqa.py --adapter ./ckpt --out secqa.json
"""

import argparse
import json
import logging
import sys
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


def load_secqa():
    """Concat secqa_v1 + secqa_v2."""
    from datasets import load_dataset

    rows = []
    for cfg in ["secqa_v1", "secqa_v2"]:
        ds = load_dataset("zefang-liu/secqa", cfg, split="test")
        for ex in ds:
            rows.append(
                {
                    "version": cfg,
                    "question": ex["Question"],
                    "choices": [ex["A"], ex["B"], ex["C"], ex["D"]],
                    "answer": ord(ex["Answer"].strip().upper()) - ord("A"),
                }
            )
    return rows


def run(model, tokenizer, device):
    """Retorna dict de metrics."""
    ids = letter_token_ids(tokenizer)
    qs = load_secqa()
    logger.info(f"loaded {len(qs)} SecQA questions (v1+v2)")
    results, n_correct = [], 0
    by_version = {}
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
        v = by_version.setdefault(q["version"], {"n": 0, "n_correct": 0})
        v["n"] += 1
        v["n_correct"] += int(correct)
        results.append(
            {"i": i, "version": q["version"], "pred": pred, "gold": q["answer"], "correct": correct}
        )
        if (i + 1) % 50 == 0:
            logger.info(f"[{i + 1}/{len(qs)}] running acc = {n_correct / (i + 1):.3f}")
    for v in by_version.values():
        v["accuracy"] = v["n_correct"] / v["n"] if v["n"] else 0.0
    return {
        "n_total": len(qs),
        "n_correct": n_correct,
        "accuracy": n_correct / len(qs) if qs else 0.0,
        "by_version": by_version,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir (default base puro)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model, tokenizer, device = load_model(args.adapter)
    summary = run(model, tokenizer, device)
    summary["adapter"] = args.adapter
    Path(args.out).write_text(json.dumps(summary, indent=2))
    logger.info(
        f"accuracy = {summary['accuracy']:.3f} ({summary['n_correct']}/{summary['n_total']}) -> {args.out}"
    )


if __name__ == "__main__":
    main()
