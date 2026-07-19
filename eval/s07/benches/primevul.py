"""PrimeVul (arxiv 2403.18624) - realistic vulnerability detection C/C++.

Binary classification: vuln vs benign. test_paired e 50/50 balanceado por
construcao (cada par tem uma versao vulneravel e a corrigida), entao 50% de
acuracia = baseline de chutar sempre o mesmo rotulo. Por isso reportamos
tambem a distribuicao das predicoes e F1: sem elas, 50% parece sinal quando
na verdade e ausencia de sinal.
"""

import re
from collections import Counter

from ._common import bootstrap_ci, generate

# So conta a PRIMEIRA polaridade que aparece na resposta, e ignora a palavra
# "vulnerable" que vem do proprio prompt sendo ecoada.
DECISION_RE = re.compile(r"\b(yes|no|vulnerable|not\s+vulnerable|benign|safe)\b", re.IGNORECASE)


def _parse_vuln_label(text: str) -> int | None:
    m = DECISION_RE.search(text)
    if not m:
        return None
    token = m.group(1).lower()
    if token in ("no", "not vulnerable", "benign", "safe") or token.startswith("not"):
        return 0
    return 1


def eval_primevul(model, tok, n: int = 500) -> dict:
    try:
        from datasets import load_dataset

        ds = load_dataset("ASSERT-KTH/PrimeVul", split="test_paired").select(
            range(min(n, 1128))
        )
    except (FileNotFoundError, ValueError, ConnectionError) as e:
        return {"error": f"primevul_load_failed: {e}"}
    per, correct = [], []
    tp = fp = tn = fn = 0
    system = "You are a code security analyst. Answer YES or NO whether the code is vulnerable."

    for i, r in enumerate(ds):
        code = r.get("func") or r.get("code") or ""
        gold = int(r.get("is_vulnerable", r.get("target", 0)))
        prompt = tok.apply_chat_template(
            [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"Does this C function contain a security vulnerability? "
                    f"Reply with a single word: YES or NO.\n\n```c\n{code[:2000]}\n```",
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        resp = generate(model, tok, prompt, max_new=16)
        pred = _parse_vuln_label(resp)
        ok = int(pred == gold)
        correct.append(ok)
        if pred == 1 and gold == 1:
            tp += 1
        elif pred == 1 and gold == 0:
            fp += 1
        elif pred == 0 and gold == 0:
            tn += 1
        elif pred == 0 and gold == 1:
            fn += 1
        per.append({"i": i, "pred": pred, "gold": gold, "correct": ok})

    ci_lo, ci_hi = bootstrap_ci(correct)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    preds = [p["pred"] for p in per]
    return {
        "n": len(correct),
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
        "ci_95_low": ci_lo,
        "ci_95_high": ci_hi,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        # Se pred_distribution mostrar quase tudo num rotulo so, 50% = colapso,
        # nao acerto. E a diferenca entre "modelo fraco" e "sem sinal nenhum".
        "pred_distribution": dict(Counter(preds)),
        "per_sample": per,
    }
