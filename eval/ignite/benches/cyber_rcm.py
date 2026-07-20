"""Bench CVE->CWE (CTI-Bench RCM) pro loop RSI+RL cyber do Caracal.

Reporta `accuracy` (match exato, o numero que compara com Foundation-Sec-8B
RCM 72-75) e `hier_score` (credito parcial na arvore CWE, sinal mais denso pra
enxergar progresso cedo quando o exato ainda nao mexeu).
"""

import json
import os
from pathlib import Path

from eval.ignite.reward import cyber_rcm_reward
from eval.s07.benches._common import normalize_cwe

from ._common import bootstrap_ci, generate

# Espelha BENCH_GEN_BUDGET["cyber_rcm"][1] do inner_grpo. Nao importamos de la
# pra nao inverter a camada (eval nao depende de train); test_rewards.py trava
# os dois no mesmo valor.
MAX_NEW = 160


def _load(n: int, dataset_path: str) -> list[dict]:
    path = Path(os.environ.get("CYBER_RCM_JSONL", dataset_path))
    if not path.exists():
        return [{"error": f"cyber_rcm dataset ausente: {path}. Rode data/ignite/build_cyber_rcm.py"}]
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[:n]


def eval_cyber_rcm(
    model, tok, n: int = 200, dataset_path: str = "data/ignite/cyber_rcm_dev.jsonl"
) -> dict:
    rows = _load(n, dataset_path)
    if rows and "error" in rows[0]:
        return rows[0]

    per, correct, hier = [], [], []
    unparsed = 0
    for i, r in enumerate(rows):
        resp = generate(model, tok, r["prompt"], max_new=MAX_NEW)
        pred = normalize_cwe(resp)
        gold = r["gold"]
        ok = int(pred is not None and pred == gold)
        if pred is None:
            unparsed += 1
        correct.append(ok)
        hier.append(max(0.0, cyber_rcm_reward(resp, gold)))
        per.append({"i": i, "pred": pred, "gold": gold, "correct": ok})
        if (i + 1) % 50 == 0:
            print(f"[cyber_rcm {i + 1}/{len(rows)}] acc={sum(correct) / (i + 1):.3f}", flush=True)

    ci_lo, ci_hi = bootstrap_ci(correct)
    n = len(correct)  # um append por row: mesmo denominador pras tres metricas
    return {
        "n": n,
        "accuracy": sum(correct) / n if n else 0.0,
        "hier_score": sum(hier) / n if n else 0.0,
        # unparsed alto = modelo parou de emitir CWE (colapso de formato), nao
        # "ficou ruim". Sem isso os dois viram a mesma queda de accuracy.
        "unparsed_frac": unparsed / n if n else 0.0,
        "ci_95_low": ci_lo,
        "ci_95_high": ci_hi,
        "per_sample": per,
    }
