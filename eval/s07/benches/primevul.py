"""PrimeVul (arxiv 2403.18624) - realistic vulnerability detection C/C++.

Binary classification: vuln vs benign. 140+ CWE types.
"""

import re

from ._common import bootstrap_ci, generate

VULN_YES_RE = re.compile(r"\b(yes|vulnerable|true)\b", re.IGNORECASE)
VULN_NO_RE = re.compile(r"\b(no|benign|safe|false)\b", re.IGNORECASE)


def _parse_vuln_label(text: str) -> int | None:
    if VULN_YES_RE.search(text):
        return 1
    if VULN_NO_RE.search(text):
        return 0
    return None


def eval_primevul(model, tok, n: int = 500) -> dict:
    try:
        from datasets import load_dataset

        ds = load_dataset("ASSERT-KTH/PrimeVul", split="test_paired").select(range(n))
    except (FileNotFoundError, ValueError, ConnectionError) as e:
        return {"error": f"primevul_load_failed: {e}"}
    per, correct = [], []
    system = "You are a code security analyst. Answer YES or NO whether the code is vulnerable."

    for i, r in enumerate(ds):
        code = r.get("func") or r.get("code") or ""
        gold = int(r.get("is_vulnerable", r.get("target", 0)))
        prompt = tok.apply_chat_template(
            [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"Is this code vulnerable?\n\n```c\n{code[:2000]}\n```\n\nAnswer YES or NO.",
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        resp = generate(model, tok, prompt, max_new=16)
        pred = _parse_vuln_label(resp)
        ok = int(pred == gold)
        correct.append(ok)
        per.append({"i": i, "pred": pred, "gold": gold, "correct": ok})
    ci_lo, ci_hi = bootstrap_ci(correct)
    return {
        "n": len(correct),
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
        "ci_95_low": ci_lo,
        "ci_95_high": ci_hi,
    }
