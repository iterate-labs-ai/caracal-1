"""MMLU computer_security subset (100 MCQ)."""

from ._common import bootstrap_ci, generate, mcq_chat_prompt, normalize_mcq_letter


def eval_mmlu_security(model, tok, n: int = 100) -> dict:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "computer_security", split="test").select(range(n))
    per, correct = [], []
    system = "You are a computer security expert. Answer the multiple choice question."

    for i, r in enumerate(ds):
        choices = {letter: r["choices"][idx] for idx, letter in enumerate("ABCD")}
        prompt = mcq_chat_prompt(tok, system, r["question"], choices)
        resp = generate(model, tok, prompt, max_new=64)
        pred = normalize_mcq_letter(resp)
        gold = "ABCD"[r["answer"]]
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
