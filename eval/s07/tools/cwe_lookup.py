"""CWE tools - lookup, tree path, keyword search, view filter."""

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


def cwe_search_kw(text: str, top_k: int = 5) -> dict:
    """Keyword search - retorna top-K candidates."""
    # TODO Vitor implement BM25 or sentence-transformers semantic search
    raise NotImplementedError("cwe_search_kw - implement em s07.D")


def cwe_view_filter(cwe_id: str, view: str = "1003") -> dict:
    """Check if CWE in named view (1003 = top-25, 699 = software dev)."""
    tree = _get_tree()
    cwe = cwe_id.replace("CWE-", "")
    if view == "1003":
        return {"cwe_id": f"CWE-{cwe}", "in_view": tree.in_view_1003(cwe), "view": view}
    return {"cwe_id": f"CWE-{cwe}", "view_supported": False, "view": view}
