"""Experimento decisor: a arquitetura Ontology-Gated Continual Memory tem sinal,
ou colapsa em kNN sobre embeddings congelados?

Motivado pela revisao adversarial (2026-07). Verificado na web: kNN-sobre-
embedding e baseline forte publicado pra CVE->CWE; Modern Hopfield = attention
= kNN no limite; probe linear sobre congelado (G-MAP/LONGMEM) e o que de fato
levanta modelo congelado. Este script roda os braços e deixa os numeros decidirem.

Braços (todos sobre cyber_rcm: CVE description -> gold CWE):
  A  zero-shot  : o Qwen gera o CWE (a abordagem atual). Precisa GPU.
  B  kNN        : embedding congelado -> FAISS -> CWE do vizinho majoritario.
  D  probe      : regressao logistica treinada sobre o MESMO embedding congelado.
Dois embedders: MiniLM (barato) e o proprio Qwen mean-pooled (testa se a rep do
backbone congelado e o gargalo — o Flaw 2 do crítico).

Condicoes de morte:
  B/D ~ A            -> memoria sobre congelado nao supera a geracao do proprio backbone.
  D >> B             -> kNN puro e fraco; a arquitetura honesta precisa de modulo treinado.
  MiniLM+D >> Qwen+D -> a escolha do backbone importa mais que a memoria.
"""

import argparse
import json
from pathlib import Path

from eval.s07.benches._common import normalize_cwe
from eval.s07.cwe_tree_parser import get_cwe_tree
from eval.s07.hier_reward import hier_cwe_reward


def _load(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def _cve_text(prompt: str) -> str:
    """O prompt do CTI-RCM embute a descricao; pra embedding usamos o texto cru."""
    return prompt


def _score(preds, golds, tree):
    """acc exato + hier (credito parcial na arvore) + fracao sem CWE."""
    exact, hier, unparsed = [], [], 0
    for p, g in zip(preds, golds, strict=False):
        if p is None:
            unparsed += 1
            exact.append(0)
            hier.append(0.0)
            continue
        exact.append(int(p == g))
        hier.append(max(0.0, hier_cwe_reward(p, g, tree)) if tree else int(p == g))
    n = len(golds)
    return {
        "acc": sum(exact) / n if n else 0.0,
        "hier": sum(hier) / n if n else 0.0,
        "unparsed": unparsed / n if n else 0.0,
        "n": n,
    }


def embed(texts, model_name):
    """Embedder congelado. sentence-transformer OU mean-pool via transformers."""
    import numpy as np

    hf_name = model_name
    if model_name.startswith("st:"):
        hf_name = model_name[3:]
        try:
            from sentence_transformers import SentenceTransformer

            m = SentenceTransformer(hf_name)
            v = m.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
            return np.asarray(v, dtype="float32")
        except Exception as e:
            # sentence-transformers 5+ puxa torchcodec/libavutil e quebra com torch 2.6;
            # cai pro mean-pool via transformers, mesmo resultado sem a dep fragil.
            print(f"[embed] sentence-transformers falhou ({type(e).__name__}), mean-pool via transformers", flush=True)

    # mean-pool do ultimo hidden (o backbone congelado de verdade)
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda").eval()
    out = []
    for i in range(0, len(texts), 16):
        batch = texts[i : i + 16]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
            "cuda"
        )
        with torch.no_grad():
            h = mdl(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
        out.append(pooled.cpu().numpy())
    del mdl
    torch.cuda.empty_cache()
    return np.concatenate(out).astype("float32")


def arm_knn(tr_emb, tr_gold, te_emb, k):
    from collections import Counter

    try:
        import faiss

        index = faiss.IndexFlatIP(tr_emb.shape[1])
        index.add(tr_emb)
        _, idx = index.search(te_emb, k)
    except ImportError:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(tr_emb)
        idx = nn.kneighbors(te_emb, return_distance=False)
    preds = []
    for row in idx:
        votes = Counter(tr_gold[j] for j in row)
        preds.append(votes.most_common(1)[0][0])
    return preds


def arm_probe(tr_emb, tr_gold, te_emb):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=2000, C=10.0)
    clf.fit(tr_emb, tr_gold)
    return list(clf.predict(te_emb))


def arm_zeroshot(test_rows, base):
    """Qwen gera o CWE (a abordagem atual). Reusa o bench cyber_rcm."""
    from eval.ignite.benches.run import bench_adapter

    # bench_adapter roda o registry cyber_rcm e devolve acc/hier/unparsed
    tmp = Path("/tmp/_decider_test.jsonl")
    tmp.write_text("\n".join(json.dumps(r) for r in test_rows))
    r = bench_adapter(base, None, "cyber_rcm", str(tmp), len(test_rows))
    return {
        "acc": r["accuracy"],
        "hier": r["hier_score"],
        "unparsed": r["unparsed_frac"],
        "n": r["n"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/ignite/cyber_rcm_train.jsonl")
    ap.add_argument("--test", default="data/ignite/cyber_rcm_dev.jsonl")
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--embedders", nargs="+", default=["st:sentence-transformers/all-MiniLM-L6-v2"])
    ap.add_argument("--zeroshot", action="store_true", help="tambem roda o braço A (lento, GPU)")
    ap.add_argument("--out", default="/kaggle/working/decider.json")
    a = ap.parse_args()

    tr, te = _load(a.train), _load(a.test)
    tr_txt = [_cve_text(r["prompt"]) for r in tr]
    te_txt = [_cve_text(r["prompt"]) for r in te]
    tr_gold = [normalize_cwe(r["gold"]) or r["gold"] for r in tr]
    te_gold = [normalize_cwe(r["gold"]) or r["gold"] for r in te]
    tree = get_cwe_tree()
    results = {"k": a.k, "n_train": len(tr), "n_test": len(te)}

    if a.zeroshot:
        results["A_zeroshot"] = arm_zeroshot(te, a.base)
        print("A zero-shot:", results["A_zeroshot"], flush=True)

    for emb_name in a.embedders:
        print(f"\n== embedder {emb_name} ==", flush=True)
        tr_emb = embed(tr_txt, emb_name)
        te_emb = embed(te_txt, emb_name)
        b = _score(arm_knn(tr_emb, tr_gold, te_emb, a.k), te_gold, tree)
        d = _score(arm_probe(tr_emb, tr_gold, te_emb), te_gold, tree)
        tag = emb_name.replace("st:", "").split("/")[-1]
        results[f"B_knn/{tag}"] = b
        results[f"D_probe/{tag}"] = d
        print(f"B kNN   ({tag}): {b}", flush=True)
        print(f"D probe ({tag}): {d}", flush=True)

    Path(a.out).write_text(json.dumps(results, indent=2))
    print("\n=== DECISOR ===")
    print(f"{'arm':28s} {'acc':>7s} {'hier':>7s} {'unparsed':>9s}")
    for kx, v in results.items():
        if isinstance(v, dict) and "acc" in v:
            print(f"{kx:28s} {v['acc']:7.3f} {v['hier']:7.3f} {v['unparsed']:9.3f}")
    print("\nMorte: B/D ~ A -> memoria nao supera geracao | D >> B -> precisa modulo treinado")


if __name__ == "__main__":
    main()
