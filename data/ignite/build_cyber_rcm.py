"""Constroi o dataset CVE->CWE (CTI-Bench RCM) pro RSI+RL cyber do Caracal.

Saida no formato que o inner_grpo espera: {"prompt": ..., "gold": "CWE-79"}.
O prompt oficial ja vem embutido na coluna `Prompt` do dataset, entao nao
inventamos formatacao propria - assim o numero compara com o baseline publicado
do Foundation-Sec-8B (RCM 72-75).

Split disjunto train/dev/val: dev guia a selecao de candidato dentro da geracao,
val so decide retencao (contamination gate do outer loop).
"""

import json
from pathlib import Path

from eval.s07.benches._common import normalize_cwe
from eval.s07.benches.cti_bench import _load_tsv

OUT_DIR = Path("data/ignite")
SPLITS = {"train": 0.7, "dev": 0.15, "val": 0.15}


def build(seed: int = 0) -> dict[str, int]:
    import random

    rows = _load_tsv("cti-rcm")
    items = []
    for r in rows:
        prompt, gold_raw = r.get("Prompt"), r.get("GT")
        if not prompt or not gold_raw:
            continue
        gold = normalize_cwe(gold_raw) or str(gold_raw).strip()
        if not gold.startswith("CWE-"):
            continue
        items.append({"prompt": prompt, "gold": gold})

    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(n * SPLITS["train"])
    n_dev = int(n * SPLITS["dev"])
    parts = {
        "train": items[:n_train],
        "dev": items[n_train : n_train + n_dev],
        "val": items[n_train + n_dev :],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, part in parts.items():
        path = OUT_DIR / f"cyber_rcm_{name}.jsonl"
        path.write_text("\n".join(json.dumps(x) for x in part) + "\n")
        counts[name] = len(part)
    return counts


if __name__ == "__main__":
    counts = build()
    print("cyber_rcm:", counts)
    total = sum(counts.values())
    assert total > 0, "dataset vazio - checar o loader do CTI-Bench"
    print(f"total {total} pares CVE->CWE em {OUT_DIR}/cyber_rcm_*.jsonl")
