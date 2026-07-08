"""CyberSOCEval (arxiv 2509.20166) - SOC threat-intel + malware MCQ.

CrowdStrike + Meta 2025. ~1.5K MCQ (malware analysis + threat-intel reasoning).
"""

from ._common import bootstrap_ci, generate, mcq_chat_prompt, normalize_mcq_letter


SPLITS = ["malware_analysis", "threat_intel_reasoning"]


def eval_cybersoceval(model, tok, n: int = 500) -> dict:
    from datasets import load_dataset

    system = "You are a SOC analyst. Answer the multiple choice question about threat intelligence."
    out = {}
    for split in SPLITS:
        try:
            ds = load_dataset("kyleavery/cybersoceval-questions", split=split).select(
                range(min(n, 999_999))
            )
        except (FileNotFoundError, ValueError, ConnectionError) as e:
            out[split] = {"error": f"cybersoceval_load_failed: {e}"}
            continue
        per, correct = [], []
        for i, r in enumerate(ds):
            q = r.get("question") or ""
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
            raw_gold = r.get("answer") or r.get("label", "")
            gold = (
                str(raw_gold).strip().upper() if not isinstance(raw_gold, int) else "ABCD"[raw_gold]
            )
            ok = int(pred == gold)
            correct.append(ok)
            per.append({"i": i, "pred": pred, "gold": gold, "correct": ok})
        ci_lo, ci_hi = bootstrap_ci(correct)
        out[split] = {
            "n": len(correct),
            "accuracy": sum(correct) / len(correct) if correct else 0.0,
            "ci_95_low": ci_lo,
            "ci_95_high": ci_hi,
        }
        print(f"[cybersoceval.{split}] acc={out[split]['accuracy']:.3f}")
    return out
