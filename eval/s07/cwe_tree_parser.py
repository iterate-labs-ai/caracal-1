"""CWE 2025 tree parser (XML) - relacao parent/child/sibling/view 1003.

From R6.2 research. CWE XML download: cwe.mitre.org/data/xml/cwec_latest.xml.zip.
"""

import xml.etree.ElementTree as ET
from pathlib import Path


class CWEParser:
    """Parse CWE XML - DAG aware (multi-parent possible)."""

    NS = {"cwe": "http://cwe.mitre.org/cwe-7"}

    def __init__(self, xml_file: str | Path):
        self.tree = ET.parse(xml_file)
        self.root = self.tree.getroot()
        self.cwe_map: dict[str, dict] = {}
        self.view_1003: set[str] = set()
        self._build_cache()

    def _build_cache(self):
        for w in self.root.findall(".//cwe:Weakness", self.NS):
            cwe_id = w.get("ID")
            self.cwe_map[cwe_id] = {"parents": [], "depth": None}
            for rel in w.findall('.//cwe:Related_Weakness[@Nature="ChildOf"]', self.NS):
                parent_id = rel.get("CWE_ID")
                self.cwe_map[cwe_id]["parents"].append(parent_id)

        for view in self.root.findall('.//cwe:View[@ID="1003"]', self.NS):
            for member in view.findall(".//cwe:View_Member", self.NS):
                self.view_1003.add(member.get("CWE_ID"))

        # Depois do cache so cwe_map/view_1003 sao lidos. Segurar o DOM inteiro
        # custava 78MB residentes de RAM disputando com o load do modelo.
        self.tree = self.root = None

    @staticmethod
    def _key(cwe: str) -> str:
        """Normaliza na BORDA do componente. A arvore keia por ID cru ("1004"),
        entao cada consumidor vinha tirando o prefixo por conta propria - eram 5
        call sites (3 em tools/cwe_lookup.py, 2 em hier_reward.py) e qualquer um
        que esquecesse dava miss silencioso em vez de erro."""
        return str(cwe).strip().upper().removeprefix("CWE-")

    def get_ancestors(self, cwe: str, visited: set | None = None) -> list[str]:
        """Returns all ancestors (transitive parents)."""
        cwe = self._key(cwe)
        if visited is None:
            visited = set()
        if cwe in visited or cwe not in self.cwe_map:
            return []
        visited.add(cwe)
        ancestors = []
        for parent in self.cwe_map[cwe]["parents"]:
            ancestors.append(parent)
            ancestors.extend(self.get_ancestors(parent, visited))
        return ancestors

    def is_ancestor(self, cwe_a: str, cwe_b: str) -> bool:
        """True if cwe_a is ancestor of cwe_b."""
        return self._key(cwe_a) in self.get_ancestors(cwe_b)

    def shared_parent(self, cwe_a: str, cwe_b: str) -> set[str]:
        """Common ancestors between two CWEs (sibling test = non-empty)."""
        return set(self.get_ancestors(cwe_a)) & set(self.get_ancestors(cwe_b))

    def in_view_1003(self, cwe: str) -> bool:
        """Top-25 CWE view."""
        return self._key(cwe) in self.view_1003

    def get_depth(self, cwe: str) -> int:
        """Cached depth from root (cycle-safe via memoization)."""
        cwe = self._key(cwe)
        if cwe not in self.cwe_map:
            return 0
        if self.cwe_map[cwe]["depth"] is not None:
            return self.cwe_map[cwe]["depth"]
        parents = self.cwe_map[cwe].get("parents", [])
        depth = 1 + max((self.get_depth(p) for p in parents), default=-1)
        self.cwe_map[cwe]["depth"] = depth
        return depth


_TREE: "CWEParser | None" = None
_TREE_FAILED = False
_CWE_ZIP_URL = "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip"


def get_cwe_tree() -> "CWEParser | None":
    """Fonte unica da arvore CWE: CWE_XML_PATH -> data/cwec*.xml -> MITRE.
    Devolve None se nao conseguir (o caller decide o fallback).

    Havia dois loaders divergentes (reward.py so glob, tools/cwe_lookup.py so
    env), entao CWE_XML_PATH nao afetava o reward e o XML de 18MB era parseado
    duas vezes quando os dois modulos entravam no mesmo processo.

    A falha e cacheada de proposito: sem isso cada chamada de reward re-tentava
    o download (timeout 60s x ~16 rollouts/step) e o step nunca terminava num
    kernel Kaggle com internet desligada.
    """
    global _TREE, _TREE_FAILED
    if _TREE is not None or _TREE_FAILED:
        return _TREE

    import glob
    import io
    import os
    import urllib.request
    import zipfile
    from pathlib import Path

    path = os.environ.get("CWE_XML_PATH") or next(iter(glob.glob("data/cwec*.xml")), None)
    if path is None or not Path(path).exists():
        try:
            Path("data").mkdir(exist_ok=True)
            with urllib.request.urlopen(_CWE_ZIP_URL, timeout=60) as r:
                zf = zipfile.ZipFile(io.BytesIO(r.read()))
                name = next(n for n in zf.namelist() if n.endswith(".xml"))
                zf.extract(name, "data")
                path = f"data/{name}"
        except (OSError, ValueError, zipfile.BadZipFile, StopIteration):
            _TREE_FAILED = True
            return None

    _TREE = CWEParser(path)
    return _TREE
