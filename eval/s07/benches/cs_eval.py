"""CS-Eval (arxiv 2411.16239) - bilingual comprehensive cyber eval.

42 categories, ZH + EN, MCQ + short answer.
"""

from ._common import bootstrap_ci, generate, mcq_chat_prompt, normalize_mcq_letter


def eval_cs_eval(model, tok, n: int = 500, split: str = "en") -> dict:
    try:
        from datasets import load_dataset

        ds = load_dataset("CS-EVAL/CS-Eval", split, split="test").select(range(n))
    except (FileNotFoundError, ValueError, ConnectionError) as e:
        return {"error": f"cs_eval_load_failed: {e}"}
    per, correct = [], []
    system = "You are a cybersecurity expert. Answer the multiple choice question."

    for i, r in enumerate(ds):
        q = r.get("question") or r.get("query") or ""
        raw_choices = r.get("choices") or r.get("options") or []
        if isinstance(raw_choices, list):
            choices = {
                letter: raw_choices[idx]
                for idx, letter in enumerate("ABCD")
                if idx < len(raw_choices)
            }
        else:
            choices = raw_choices
        if not choices or not q:
            continue
        prompt = mcq_chat_prompt(tok, system, q, choices)
        resp = generate(model, tok, prompt, max_new=64)
        pred = normalize_mcq_letter(resp)
        raw_gold = r.get("answer") or ""
        gold = str(raw_gold).strip().upper() if not isinstance(raw_gold, int) else "ABCD"[raw_gold]
        ok = int(pred == gold)
        correct.append(ok)
        per.append({"i": i, "pred": pred, "gold": gold, "correct": ok})
    ci_lo, ci_hi = bootstrap_ci(correct)
    return {
        "split": split,
        "n": len(correct),
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
        "ci_95_low": ci_lo,
        "ci_95_high": ci_hi,
    }
