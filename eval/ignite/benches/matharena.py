"""MathArena live bench - rolling contest ingestion, zero contamination.

L3 gate. AIME/HMMT/USAMO/IMO fresh problems.
"""

import os
from pathlib import Path

from eval.ignite.reward import math_reward

from ._common import bootstrap_ci, generate, read_jsonl

SYSTEM = "You are a math olympiad expert. Solve the problem step by step. Output the final answer inside \\boxed{}."


def _load(n: int, dataset_path: str) -> list[dict]:
    path = Path(os.environ.get("MATHARENA_JSONL", dataset_path))
    if path.exists():
        rows = read_jsonl(path)
        return rows[:n]
    return [{"error": f"matharena_jsonl_missing: {path}"}]


def eval_matharena(
    model, tok, n: int = 100, dataset_path: str = "data/ignite/matharena.jsonl"
) -> dict:
    rows = _load(n, dataset_path)
    if rows and "error" in rows[0]:
        return rows[0]

    per, correct = [], []
    for i, r in enumerate(rows):
        prompt = tok.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": r["prompt"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        resp = generate(model, tok, prompt, max_new=2048)
        r_score = math_reward(resp, r["gold"], format_bonus=False)
        ok = int(r_score >= 1.0)
        correct.append(ok)
        per.append(
            {
                "id": r["id"],
                "pred": resp[-500:],
                "gold": r["gold"],
                "correct": ok,
                "contest": r.get("contest"),
                "year": r.get("year"),
            }
        )
        if (i + 1) % 10 == 0:
            print(f"[matharena {i + 1}/{len(rows)}] acc={sum(correct) / (i + 1):.3f}")

    ci_lo, ci_hi = bootstrap_ci(correct)
    return {
        "n": len(correct),
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
        "ci_95_low": ci_lo,
        "ci_95_high": ci_hi,
        "per_sample": per,
    }
