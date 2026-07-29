"""Prova o CONTINUAL: um datastore que cresce no stream sem re-treino mantem o
velho (sem esquecer) e pega o novo, batendo o re-treino caro?

Decisor ja mostrou o estatico (probe/kNN sobre MiniLM bate o 3B zero-shot +12pp).
Aqui a pergunta que importa: no stream, append-only (sem tocar peso) >= re-treino,
e sem esquecimento catastrofico?

Protocolo (sem precisar de datas): stream = train dividido em K chunks sequenciais.
Apos cada chunk, avalia no DEV inteiro tres arms:
  GROW   : kNN datastore que so APENDA os chunks vistos (zero re-treino)
  CUR    : probe treinado SO no chunk atual (esquece o anterior — o baseline ruim)
  RETRAIN: probe re-treinado em TUDO visto ate agora (upper bound caro)

Se GROW ~ RETRAIN e GROW >> CUR: continual sem re-treino e sem esquecer, de graca.
Roda em CPU (MiniLM minusculo).
"""

import argparse
import json
from pathlib import Path

from eval.exp.decider import _load, _score, arm_knn, arm_probe, embed
from eval.s07.benches._common import normalize_cwe
from eval.s07.cwe_tree_parser import get_cwe_tree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/ignite/cyber_rcm_train.jsonl")
    ap.add_argument("--test", default="data/ignite/cyber_rcm_dev.jsonl")
    ap.add_argument("--embedder", default="st:sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--out", default="/kaggle/working/continual.json")
    a = ap.parse_args()

    tr, te = _load(a.train), _load(a.test)
    tr_txt = [r["prompt"] for r in tr]
    tr_gold = [normalize_cwe(r["gold"]) or r["gold"] for r in tr]
    te_txt = [r["prompt"] for r in te]
    te_gold = [normalize_cwe(r["gold"]) or r["gold"] for r in te]
    tree = get_cwe_tree()

    print(f"embedding {len(tr)} train + {len(te)} test ...", flush=True)
    tr_emb = embed(tr_txt, a.embedder)
    te_emb = embed(te_txt, a.embedder)

    n = len(tr)
    bounds = [round(i * n / a.windows) for i in range(a.windows + 1)]
    traj = []
    for w in range(a.windows):
        seen = bounds[w + 1]  # chunks 0..w vistos (apenda)
        cs, ce = bounds[w], bounds[w + 1]  # chunk atual

        grow = _score(arm_knn(tr_emb[:seen], tr_gold[:seen], te_emb, a.k), te_gold, tree)
        retr = _score(arm_probe(tr_emb[:seen], tr_gold[:seen], te_emb), te_gold, tree)
        try:
            cur = _score(arm_probe(tr_emb[cs:ce], tr_gold[cs:ce], te_emb), te_gold, tree)
        except ValueError:
            cur = {"acc": 0.0, "hier": 0.0, "unparsed": 0.0, "n": len(te)}  # chunk 1-classe

        row = {
            "window": w + 1,
            "seen": seen,
            "GROW_noretrain": round(grow["acc"], 4),
            "RETRAIN_upper": round(retr["acc"], 4),
            "CUR_forgets": round(cur["acc"], 4),
        }
        traj.append(row)
        print(
            f"win {w + 1}/{a.windows} seen={seen:4d} | GROW {row['GROW_noretrain']:.3f} "
            f"| RETRAIN {row['RETRAIN_upper']:.3f} | CUR {row['CUR_forgets']:.3f}",
            flush=True,
        )

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(traj, indent=2))
    last = traj[-1]
    print("\n=== CONTINUAL (dev acc por janela) ===")
    print(f"final: GROW(sem retreino)={last['GROW_noretrain']:.3f}  "
          f"RETRAIN(caro)={last['RETRAIN_upper']:.3f}  CUR(esquece)={last['CUR_forgets']:.3f}")
    gap = last["RETRAIN_upper"] - last["GROW_noretrain"]
    print(f"gap GROW vs RETRAIN = {gap:+.3f}  (perto de 0 = nao precisa re-treinar)")
    print(f"ganho GROW vs CUR   = {last['GROW_noretrain'] - last['CUR_forgets']:+.3f}  (alto = CUR esquece, GROW nao)")


if __name__ == "__main__":
    main()
