"""SecQA (arxiv 2312.15838) v1 (110) + v2 (100) MCQs foundational security."""

from ._common import bootstrap_ci, generate, mcq_chat_prompt, normalize_mcq_letter


def _load_split(version: str) -> list[dict]:
    from datasets import load_dataset

    return list(load_dataset("zefang-liu/secqa", version, split="test"))


def eval_secqa(model, tok, versions: list[str] | None = None) -> dict:
    versions = versions or ["secqa_v1", "secqa_v2"]
    out = {}
    system = "You are a cybersecurity expert. Answer the multiple choice question."

    for v in versions:
        rows = _load_split(v)
        per, correct = [], []
        for i, r in enumerate(rows):
            choices = {"A": r["A"], "B": r["B"], "C": r["C"], "D": r["D"]}
            prompt = mcq_chat_prompt(tok, system, r["Question"], choices)
            resp = generate(model, tok, prompt, max_new=64)
            pred = normalize_mcq_letter(resp)
            gold = r["Answer"].strip().upper()
            ok = int(pred == gold)
            correct.append(ok)
            per.append({"i": i, "pred": pred, "gold": gold, "correct": ok})
        ci_lo, ci_hi = bootstrap_ci(correct)
        out[v] = {
            "n": len(correct),
            "accuracy": sum(correct) / len(correct) if correct else 0.0,
            "ci_95_low": ci_lo,
            "ci_95_high": ci_hi,
        }
        print(f"[{v}] acc={out[v]['accuracy']:.3f} ci=[{ci_lo:.3f}-{ci_hi:.3f}]")
    return out
