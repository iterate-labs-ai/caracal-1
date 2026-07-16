"""PutnamBench Lean bench (arxiv 2407.11214).

L2+ dual-purpose: eval + Lean RL reward source. Kernel-verifiable proofs.
"""

import json
import os
from pathlib import Path

from eval.ignite.tools.lean_tool import LeanDaemonPool, extract_lean_proof

from ._common import bootstrap_ci, generate

SYSTEM = "You are a Lean 4 theorem prover. Prove the theorem using Mathlib tactics. Output the proof inside a ```lean code fence."


def _load(n: int, dataset_path: str) -> list[dict]:
    path = Path(os.environ.get("PUTNAM_JSONL", dataset_path))
    if not path.exists():
        return [{"error": f"putnam_jsonl_missing: {path}"}]
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[:n]


def eval_putnam_lean(
    model,
    tok,
    n: int = 30,
    dataset_path: str = "data/ignite/putnam.jsonl",
    workers: int = 2,
) -> dict:
    rows = _load(n, dataset_path)
    if rows and "error" in rows[0]:
        return rows[0]

    pool = LeanDaemonPool(workers=workers)
    try:
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
            proof = extract_lean_proof(resp)
            if proof is None:
                ok = 0
                out_msg = "no proof fence"
            else:
                source = f"{r['preamble']}\n\n{r['theorem']}\n{proof}\n"
                result = pool.verify_proof(source)
                ok = int(result.proved)
                out_msg = result.error or "ok"
            correct.append(ok)
            per.append({"id": r["id"], "pred": resp[-500:], "correct": ok, "msg": out_msg})
            if (i + 1) % 5 == 0:
                print(f"[putnam {i + 1}/{len(rows)}] proved={sum(correct) / (i + 1):.3f}")
    finally:
        pool.shutdown()

    ci_lo, ci_hi = bootstrap_ci(correct)
    return {
        "n": len(correct),
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
        "proved_rate": sum(correct) / len(correct) if correct else 0.0,
        "ci_95_low": ci_lo,
        "ci_95_high": ci_hi,
        "per_sample": per,
    }
