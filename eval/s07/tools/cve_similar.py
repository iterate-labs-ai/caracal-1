"""Vector search top-K similar past CVEs.

Local index via sentence-transformers all-MiniLM-L6-v2 (T4-friendly).
"""

import json
from pathlib import Path

_INDEX = None
_METADATA: list[dict] = []


def _build_index(cve_jsonl: Path):
    """Build FAISS-like flat index from CVE dataset."""
    global _INDEX, _METADATA
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    with cve_jsonl.open() as f:
        rows = [json.loads(line) for line in f]
    _METADATA = rows
    texts = [r["description"] for r in rows]
    _INDEX = model.encode(texts, normalize_embeddings=True, batch_size=64)
    return model


def cve_similar(description: str, top_k: int = 5) -> dict:
    """Returns top-K historically similar CVEs + their CWEs."""
    import os

    global _INDEX
    if _INDEX is None:
        idx_path = Path(os.environ.get("CVE_INDEX_JSONL", "data/xamxte-train.jsonl"))
        if not idx_path.exists():
            return {"error": f"index_missing_{idx_path}"}
        _build_index(idx_path)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query = model.encode([description], normalize_embeddings=True)[0]
    scores = _INDEX @ query
    top_idx = scores.argsort()[::-1][:top_k]
    return {
        "results": [
            {
                "cve_id": _METADATA[i]["cve_id"],
                "cwe_id": _METADATA[i]["cwe_id"],
                "similarity": float(scores[i]),
                "description": _METADATA[i]["description"][:200],
            }
            for i in top_idx
        ]
    }
