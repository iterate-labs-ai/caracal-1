"""SecEval - 2000+ security exam questions (XuanwuAI/SecEval)."""

from ._common import bootstrap_ci, generate, mcq_chat_prompt, normalize_mcq_letter


def eval_seceval(model, tok, n: int = 500) -> dict:
    from datasets import load_dataset

    try:
        ds = load_dataset("XuanwuAI/SecEval", split="train").select(range(n))
    except (FileNotFoundError, ValueError) as e:
        return {"error": f"secval_load_failed: {e}"}
    per, correct = [], []
    system = "You are a cybersecurity expert. Answer the multiple choice question."

    for i, r in enumerate(ds):
        raw = r.get("choices") or {}
        if isinstance(raw, list):
            choices = {}
            for idx, letter in enumerate("ABCD"):
                if idx >= len(raw):
                    break
                text = raw[idx]
                for prefix in (f"{letter}: ", f"{letter}. ", f"{letter}) "):
                    if text.startswith(prefix):
                        text = text[len(prefix) :]
                        break
                choices[letter] = text
        else:
            choices = raw
        q = r.get("question") or ""
        if not choices or not q:
            continue
        prompt = mcq_chat_prompt(tok, system, q, choices)
        resp = generate(model, tok, prompt, max_new=64)
        pred = normalize_mcq_letter(resp)
        raw_gold = r.get("answer") or r.get("label") or ""
        gold = str(raw_gold).strip().upper() if not isinstance(raw_gold, int) else "ABCD"[raw_gold]
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
