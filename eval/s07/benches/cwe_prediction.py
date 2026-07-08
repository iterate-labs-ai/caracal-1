"""CWE prediction - CWE class multi-classification em xamxte held-out."""

from ._common import bootstrap_ci, generate, normalize_cwe


def eval_cwe_prediction(model, tok, n: int = 500) -> dict:
    """Test on xamxte agreement-filtered eval split."""
    from datasets import load_dataset

    ds = load_dataset("xamxte/cve-to-cwe", split="test").select(range(n))
    per, correct = [], []
    for i, r in enumerate(ds):
        prompt = tok.apply_chat_template(
            [
                {"role": "system", "content": "Classify CVE by CWE. Output \\boxed{CWE-NNN}."},
                {"role": "user", "content": f"CVE Description: {r['description']}"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        resp = generate(model, tok, prompt, max_new=256)
        pred = normalize_cwe(resp)
        gold = r["cwe_id"] if r["cwe_id"].startswith("CWE-") else f"CWE-{r['cwe_id']}"
        ok = int(pred == gold)
        correct.append(ok)
        per.append({"i": i, "pred": pred, "gold": gold, "correct": ok})
        if (i + 1) % 100 == 0:
            print(f"[cwe_pred {i + 1}/{len(ds)}] acc={sum(correct) / (i + 1):.3f}")
    ci_lo, ci_hi = bootstrap_ci(correct)
    return {
        "n": len(correct),
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
        "ci_95_low": ci_lo,
        "ci_95_high": ci_hi,
    }
