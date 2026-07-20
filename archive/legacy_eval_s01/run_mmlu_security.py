"""MMLU computer_security subset - ancora de conhecimento cyber sem judge LLM.

Cada questao tem 4 opcoes (A/B/C/D). Pra cada uma, computa logprob da letra
no token seguinte ao prompt, escolhe argmax. Forward pass unico, shape fixo
- TPU-friendly, sem recompile.

Usage:
    python eval/run_mmlu_security.py --adapter ./ckpt --out mmlu-sec.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval._common import load_model  # noqa: E402

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "The following is a multiple choice question about computer security. "
    "Output ONLY the letter of the correct answer (A, B, C, or D).\n\n"
    "Question: {question}\n"
    "A. {a}\n"
    "B. {b}\n"
    "C. {c}\n"
    "D. {d}\n"
    "Answer:"
)


def load_mmlu_security():
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "computer_security", split="test")
    return [
        {
            "question": ex["question"],
            "choices": ex["choices"],
            "answer": ex["answer"],
        }
        for ex in ds
    ]


def pick_answer(model, tokenizer, prompt, device, letter_token_ids):
    import torch

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model(**enc)
    last_logits = out.logits[0, -1, :]
    letter_logits = torch.stack([last_logits[tid] for tid in letter_token_ids])
    return int(torch.argmax(letter_logits).item())


def letter_token_ids(tokenizer):
    return [tokenizer.encode(c, add_special_tokens=False)[0] for c in ["A", "B", "C", "D"]]


def run(model, tokenizer, device):
    """Roda MMLU computer_security. Retorna dict de metrics. Reutilizado em run_bench_all."""
    ids = letter_token_ids(tokenizer)
    qs = load_mmlu_security()
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
        if (i + 1) % 20 == 0:
            logger.info(f"[{i + 1}/{len(qs)}] running acc = {n_correct / (i + 1):.3f}")
    return {
        "n_total": len(qs),
        "n_correct": n_correct,
        "accuracy": n_correct / len(qs) if qs else 0.0,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir (default base puro)")
    parser.add_argument("--out", required=True, help="JSON output path")
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
