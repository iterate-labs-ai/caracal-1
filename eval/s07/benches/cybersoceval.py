"""CyberSOCEval (arxiv 2509.20166) - SOC threat-intel + malware MCQ.

CrowdStrike + Meta 2025. ~1.5K MCQ (malware analysis + threat-intel reasoning).
"""

from ._common import bootstrap_ci, generate, mcq_chat_prompt, normalize_mcq_letters

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
        per, correct, partial = [], [], []
        for i, r in enumerate(ds):
            q = r.get("question") or ""
            raw_choices = r.get("options") or r.get("choices") or []
            if isinstance(raw_choices, list):
                # Opcoes ja vem prefixadas ("A. texto"); tira pra nao duplicar no prompt.
                choices = {}
                for idx, letter in enumerate("ABCD"):
                    if idx >= len(raw_choices):
                        break
                    text = str(raw_choices[idx])
                    for prefix in (f"{letter}. ", f"{letter}: ", f"{letter}) "):
                        if text.startswith(prefix):
                            text = text[len(prefix) :]
                            break
                    choices[letter] = text
            else:
                choices = raw_choices

            # Gold e `answers`, uma LISTA (multi-resposta). Ler `answer`/`label`
            # devolvia "" e nada batia nunca - era a causa do 0.0% com n=50.
            raw_gold = r.get("answers")
            if isinstance(raw_gold, list):
                gold_set = {str(g).strip().upper() for g in raw_gold if str(g).strip()}
            elif isinstance(raw_gold, int):
                gold_set = {"ABCD"[raw_gold]}
            else:
                gold_set = {str(raw_gold).strip().upper()} if raw_gold else set()

            if not choices or not q or not gold_set:
                continue

            multi = len(gold_set) > 1
            prompt = mcq_chat_prompt(tok, system, q, choices, multi=multi)
            resp = generate(model, tok, prompt, max_new=64)
            pred_set = normalize_mcq_letters(resp)

            ok = int(pred_set == gold_set)
            jac = len(pred_set & gold_set) / len(pred_set | gold_set) if pred_set | gold_set else 0.0
            correct.append(ok)
            partial.append(jac)
            per.append(
                {
                    "i": i,
                    "pred": sorted(pred_set),
                    "gold": sorted(gold_set),
                    "correct": ok,
                    "jaccard": jac,
                    "multi": multi,
                }
            )
        ci_lo, ci_hi = bootstrap_ci(correct)
        out[split] = {
            "n": len(correct),
            "accuracy": sum(correct) / len(correct) if correct else 0.0,
            "partial_credit": sum(partial) / len(partial) if partial else 0.0,
            "multi_answer_frac": sum(1 for p in per if p["multi"]) / len(per) if per else 0.0,
            "ci_95_low": ci_lo,
            "ci_95_high": ci_hi,
            "per_sample": per,
        }
        print(f"[cybersoceval.{split}] acc={out[split]['accuracy']:.3f}")
    return out
