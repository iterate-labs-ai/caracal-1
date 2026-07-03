"""CWE tools - lookup, tree path, keyword search, view filter."""

import numpy as np

from eval.s07.cwe_tree_parser import CWEParser

_TREE: CWEParser | None = None


def _get_tree() -> CWEParser:
    """Lazy load CWE XML."""
    global _TREE
    if _TREE is None:
        import os

        path = os.environ.get("CWE_XML_PATH", "data/cwec_latest.xml")
        _TREE = CWEParser(path)
    return _TREE


def cwe_lookup(cwe_id: str) -> dict:
    """Returns description + parents + children + view 1003 status."""
    tree = _get_tree()
    cwe = cwe_id.replace("CWE-", "")
    if cwe not in tree.cwe_map:
        return {"cwe_id": f"CWE-{cwe}", "error": "unknown_cwe"}
    return {
        "cwe_id": f"CWE-{cwe}",
        "parents": tree.cwe_map[cwe]["parents"],
        "depth": tree.get_depth(cwe),
        "in_view_1003": tree.in_view_1003(cwe),
    }


def cwe_tree_path(cwe_id: str) -> dict:
    """Returns ancestors chain (parent of parent of ...)."""
    tree = _get_tree()
    cwe = cwe_id.replace("CWE-", "")
    ancestors = tree.get_ancestors(cwe)
    return {"cwe_id": f"CWE-{cwe}", "ancestors": [f"CWE-{a}" for a in ancestors]}


_KW_INDEX: dict[str, str] | None = None
_KW_EMBEDDINGS = None


def _kw_index() -> tuple[dict[str, str], "np.ndarray"]:
    """Build keyword -> CWE-ID index via micro_rubric + top-25 view."""
    global _KW_INDEX, _KW_EMBEDDINGS
    if _KW_INDEX is not None:
        return _KW_INDEX, _KW_EMBEDDINGS
    from .cwe_micro_rubric import MICRO_RUBRICS

    _KW_INDEX = MICRO_RUBRICS
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = list(MICRO_RUBRICS.values())
    _KW_EMBEDDINGS = model.encode(texts, normalize_embeddings=True)
    return _KW_INDEX, _KW_EMBEDDINGS


def cwe_search_kw(text: str, top_k: int = 5) -> dict:
    """Semantic search: text -> top-K CWE candidates (from top-25 rubrics)."""
    from sentence_transformers import SentenceTransformer

    index, embeddings = _kw_index()
    ids = list(index.keys())
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query = model.encode([text], normalize_embeddings=True)[0]
    scores = embeddings @ query
    top = scores.argsort()[::-1][:top_k]
    return {
        "candidates": [
            {"cwe_id": ids[i], "score": float(scores[i]), "rubric": index[ids[i]][:120]}
            for i in top
        ]
    }


def cwe_view_filter(cwe_id: str, view: str = "1003") -> dict:
    """Check if CWE in named view (1003 = top-25, 699 = software dev)."""
    tree = _get_tree()
    cwe = cwe_id.replace("CWE-", "")
    if view == "1003":
        return {"cwe_id": f"CWE-{cwe}", "in_view": tree.in_view_1003(cwe), "view": view}
    return {"cwe_id": f"CWE-{cwe}", "view_supported": False, "view": view}
