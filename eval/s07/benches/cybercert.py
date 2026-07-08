"""CyberCertBench (arxiv 2604.20389) - certification-style MCQ (CISSP/OSCP).

Security certification exam knowledge eval.
"""

from ._common import bootstrap_ci, generate, mcq_chat_prompt, normalize_mcq_letter


def eval_cybercert(model, tok, n: int = 300) -> dict:
    try:
        from datasets import load_dataset

        ds = load_dataset("cybercertbench/CyberCertBench", split="test").select(range(n))
    except (FileNotFoundError, ValueError, ConnectionError) as e:
        return {"error": f"cybercert_load_failed: {e}"}
    per, correct = [], []
    system = "You are a certified cybersecurity professional. Answer the exam question."

    for i, r in enumerate(ds):
        q = r.get("question") or ""
        raw_choices = r.get("choices") or []
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
        "n": len(correct),
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
        "ci_95_low": ci_lo,
        "ci_95_high": ci_hi,
    }
