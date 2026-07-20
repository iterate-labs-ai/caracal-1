"""OMNI-MATH bench (arxiv 2410.07985) - 4428 Olympiad problems, RLVR.

Uses math_verify tool for verification. First functional bench for Ignite-3B.
"""

import os
from pathlib import Path

from eval.ignite.reward import math_reward

from ._common import bootstrap_ci, generate, read_jsonl


def _load(n: int, dataset_path: str) -> list[dict]:
    path = Path(os.environ.get("OMNI_MATH_JSONL", dataset_path))
    if path.exists():
        rows = read_jsonl(path)
        return rows[:n]
    try:
        from datasets import load_dataset

        ds = load_dataset("iterate-labs-ai/ignite-omni-math", split="test").select(range(n))
        return list(ds)
    except (FileNotFoundError, ValueError, ConnectionError) as e:
        return [{"error": f"omni_math_load_failed: {e}"}]


SYSTEM = "You are a math olympiad expert. Solve the problem and output your final answer inside \\boxed{}."


def eval_omni_math(
    model, tok, n: int = 500, dataset_path: str = "data/ignite/omni_math.jsonl"
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
        resp = generate(model, tok, prompt, max_new=1024)
        gold = r["gold"]
        r_score = math_reward(resp, gold, format_bonus=False)
        ok = int(r_score >= 1.0)
        correct.append(ok)
        per.append(
            {
                "id": r.get("id", f"omni_math_{i}"),
                "pred": resp[-500:],
                "gold": gold,
                "correct": ok,
                "reward": r_score,
            }
        )
        if (i + 1) % 25 == 0:
            print(f"[omni_math {i + 1}/{len(rows)}] acc={sum(correct) / (i + 1):.3f}")

    ci_lo, ci_hi = bootstrap_ci(correct)
    return {
        "n": len(correct),
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
        "ci_95_low": ci_lo,
        "ci_95_high": ci_hi,
        "per_sample": per,
    }
