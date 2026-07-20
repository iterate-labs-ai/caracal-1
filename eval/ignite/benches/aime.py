"""AIME 2024/2025 bench - held-in/held-out temporal split.

Sanity reference only, não gate primário. Integer exact match RLVR.
"""

import os
from pathlib import Path

from eval.ignite.reward import math_reward

from ._common import bootstrap_ci, generate, read_jsonl

SYSTEM = "You are a math olympiad expert. Solve the AIME problem step by step. The answer is an integer 0-999. Output the final answer inside \\boxed{}."


def _load(n: int, dataset_path: str, year: int | None) -> list[dict]:
    path = Path(os.environ.get("AIME_JSONL", dataset_path))
    if not path.exists():
        return [{"error": f"aime_jsonl_missing: {path}"}]
    rows = read_jsonl(path)
    if year is not None:
        rows = [r for r in rows if r.get("year") == year]
    return rows[:n]


def eval_aime(
    model,
    tok,
    n: int = 30,
    year: int | None = 2025,
    dataset_path: str = "data/ignite/aime.jsonl",
) -> dict:
    rows = _load(n, dataset_path, year)
    if rows and "error" in rows[0]:
        return rows[0]

    per, correct = [], []
    for r in rows:
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
                "year": r.get("year"),
            }
        )

    ci_lo, ci_hi = bootstrap_ci(correct)
    return {
        "year": year,
        "n": len(correct),
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
        "ci_95_low": ci_lo,
        "ci_95_high": ci_hi,
        "per_sample": per,
    }
