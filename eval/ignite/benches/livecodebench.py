"""LiveCodeBench bench (arxiv 2403.07974) - rolling monthly code bench.

Uses code_exec sandbox for RLVR verification. Temporal decontamination via
monthly buckets (release_v6 in 2026).
"""

import json
import os
from pathlib import Path

from eval.ignite.reward import code_task_reward

from ._common import bootstrap_ci, generate

SYSTEM = "You are an expert programmer. Solve the coding problem. Output the code inside a python code fence."


def _load(n: int, dataset_path: str) -> list[dict]:
    path = Path(os.environ.get("LCB_JSONL", dataset_path))
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return rows[:n]
    return [{"error": f"lcb_jsonl_missing: {path}"}]


def eval_livecodebench(
    model, tok, n: int = 300, dataset_path: str = "data/ignite/lcb.jsonl"
) -> dict:
    rows = _load(n, dataset_path)
    if rows and "error" in rows[0]:
        return rows[0]

    per, correct = [], []
    for i, r in enumerate(rows):
        prompt = tok.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": r["prompt"] + "\n\n" + (r.get("starter_code") or "")},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        resp = generate(model, tok, prompt, max_new=2048)
        tests = r.get("tests") or []
        r_score = code_task_reward(resp, tests, timeout_s=10.0)
        ok = int(r_score >= 1.0)
        correct.append(ok)
        per.append(
            {
                "id": r["id"],
                "pred": resp[-500:],
                "n_tests": len(tests),
                "correct": ok,
                "reward": r_score,
                "difficulty": r.get("difficulty"),
            }
        )
        if (i + 1) % 10 == 0:
            print(f"[lcb {i + 1}/{len(rows)}] acc={sum(correct) / (i + 1):.3f}")

    ci_lo, ci_hi = bootstrap_ci(correct)
    by_diff: dict[str, list[int]] = {}
    for p in per:
        by_diff.setdefault(str(p.get("difficulty")), []).append(p["correct"])
    per_difficulty = {d: {"n": len(v), "accuracy": sum(v) / len(v)} for d, v in by_diff.items()}
    return {
        "n": len(correct),
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
        "ci_95_low": ci_lo,
        "ci_95_high": ci_hi,
        "per_difficulty": per_difficulty,
        "per_sample": per,
    }
