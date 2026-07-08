"""CyberMetric benchmark - 500 / 2000 / 10000 tier MCQ.

Dataset: cybermetric.ai (rag-verified security MCQ 9 domains).
"""

import json
import urllib.request

from ._common import bootstrap_ci, generate, mcq_chat_prompt, normalize_mcq_letter

CYBERMETRIC_TIERS = {
    500: "https://raw.githubusercontent.com/cybermetric/CyberMetric/main/CyberMetric-500-v1.json",
    2000: "https://raw.githubusercontent.com/cybermetric/CyberMetric/main/CyberMetric-2000-v1.json",
    10000: "https://raw.githubusercontent.com/cybermetric/CyberMetric/main/CyberMetric-10000-v1.json",
}


def _load(tier: int) -> list[dict]:
    url = CYBERMETRIC_TIERS[tier]
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.load(r)
    return data.get("questions", data)


def eval_cybermetric(model, tok, tier: int = 500) -> dict:
    rows = _load(tier)
    per, correct = [], []
    system = "You are a cybersecurity expert. Answer the multiple choice question."

    for i, r in enumerate(rows):
        q = r["question"]
        choices = r["answers"]  # dict A/B/C/D -> text
        prompt = mcq_chat_prompt(tok, system, q, choices)
        resp = generate(model, tok, prompt, max_new=64)
        pred = normalize_mcq_letter(resp)
        gold = r["solution"].strip().upper()
        ok = int(pred == gold)
        correct.append(ok)
        per.append({"i": i, "pred": pred, "gold": gold, "correct": ok})
        if (i + 1) % 100 == 0:
            print(f"[cybermetric-{tier} {i + 1}/{len(rows)}] acc={sum(correct) / (i + 1):.3f}")
    ci_lo, ci_hi = bootstrap_ci(correct)
    return {
        "tier": tier,
        "n": len(correct),
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
        "ci_95_low": ci_lo,
        "ci_95_high": ci_hi,
    }
