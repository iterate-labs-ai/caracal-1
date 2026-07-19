"""Hierarchical CWE reward function - Caracal s07 RLVR.

From R5.2 design + Minerva paper. Partial credit baseado em CWE tree distance.
"""

import re

# Fonte unica do normalize_cwe: a copia local tinha regex antigo (CWE-?\d, sem
# espaco) e rejeitava "CWE 119", divergindo do _common ja corrigido.
from eval.s07.benches._common import normalize_cwe
from eval.s07.cwe_tree_parser import CWEParser

# BOXED so-CWE fica local: o format_bonus premia boxar um CWE especifico, nao
# qualquer coisa (o BOXED_RE do _common casa \boxed{qualquer}).
BOXED_RE = re.compile(r"\\boxed\{(CWE-?\d{1,4})\}", re.IGNORECASE)


def hier_cwe_reward(pred_text: str, gold_cwe: str, tree: CWEParser) -> float:
    """Hierarchical CWE reward.

    exact match     = 1.0
    parent/ancestor = 0.5
    sibling         = 0.3
    same view 1003  = 0.1
    other           = 0.0
    + format bonus  = +0.1 (boxed regex match)
    """
    pred = normalize_cwe(pred_text)
    if not pred:
        return 0.0
    gold = normalize_cwe(gold_cwe) or gold_cwe

    # arvore keia por ID cru ("1004"); normalize_cwe devolve "CWE-1004".
    p_id = pred.removeprefix("CWE-")
    g_id = gold.removeprefix("CWE-")

    base = 0.0
    if pred == gold:
        base = 1.0
    elif tree.is_ancestor(p_id, g_id) or tree.is_ancestor(g_id, p_id):
        base = 0.5
    elif tree.shared_parent(p_id, g_id):
        base = 0.3
    elif tree.in_view_1003(p_id) and tree.in_view_1003(g_id):
        base = 0.1

    format_bonus = 0.1 if BOXED_RE.search(pred_text) else 0.0
    return min(1.0, base + format_bonus)


def length_penalty(text: str, max_tokens: int = 256) -> float:
    """Penalize overly long responses."""
    tokens = len(text.split())
    if tokens <= max_tokens:
        return 0.0
    return -0.05 * ((tokens - max_tokens) / 32)


def composite_reward(
    pred_text: str, gold_cwe: str, tree: CWEParser, tool_args_valid: bool = True
) -> float:
    """Reward composto Minerva-style."""
    r = hier_cwe_reward(pred_text, gold_cwe, tree)
    r += length_penalty(pred_text)
    if tool_args_valid:
        r += 0.05
    return max(-1.0, min(1.0, r))
