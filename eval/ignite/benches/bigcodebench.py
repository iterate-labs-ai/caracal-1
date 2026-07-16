"""BigCodeBench-Hard bench (arxiv 2406.15877).

L2 asymptotic gate. Complex function calls, distinct dist from LiveCodeBench.
"""

import json
import os
from pathlib import Path

from eval.ignite.reward import code_task_reward

from ._common import bootstrap_ci, generate

SYSTEM = "You are an expert Python programmer. Complete the function. Output only the completed function inside a python code fence."


def _load(n: int, dataset_path: str) -> list[dict]:
    path = Path(os.environ.get("BCB_JSONL", dataset_path))
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return rows[:n]
    return [{"error": f"bcb_jsonl_missing: {path}"}]


def eval_bigcodebench(
    model, tok, n: int = 148, dataset_path: str = "data/ignite/bcb_hard.jsonl"
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
        tests = r.get("tests") or []
        r_score = code_task_reward(resp, tests, timeout_s=15.0)
        ok = int(r_score >= 1.0)
        correct.append(ok)
        per.append({"id": r["id"], "pred": resp[-500:], "correct": ok, "reward": r_score})
        if (i + 1) % 10 == 0:
            print(f"[bcb {i + 1}/{len(rows)}] acc={sum(correct) / (i + 1):.3f}")

    ci_lo, ci_hi = bootstrap_ci(correct)
    return {
        "n": len(correct),
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
        "ci_95_low": ci_lo,
        "ci_95_high": ci_hi,
        "per_sample": per,
    }
