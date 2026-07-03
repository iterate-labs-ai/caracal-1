"""Caracal s07 cyber tools registry (10 tools)."""

from .attack_lookup import attack_lookup
from .cve_similar import cve_similar
from .cvss_calc import cvss_calc
from .cwe_lookup import cwe_lookup, cwe_search_kw, cwe_tree_path, cwe_view_filter
from .cwe_micro_rubric import cwe_micro_rubric
from .nvd_fetch import nvd_fetch
from .verify_fit import verify_fit

TOOL_REGISTRY = {
    "cwe_lookup": cwe_lookup,
    "cwe_tree_path": cwe_tree_path,
    "cwe_search_kw": cwe_search_kw,
    "cwe_view_filter": cwe_view_filter,
    "cwe_micro_rubric": cwe_micro_rubric,
    "attack_lookup": attack_lookup,
    "cve_similar": cve_similar,
    "cvss_calc": cvss_calc,
    "nvd_fetch": nvd_fetch,
    "verify_fit": verify_fit,
}
