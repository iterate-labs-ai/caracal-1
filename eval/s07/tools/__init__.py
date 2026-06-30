"""Caracal s07 cyber tools registry.

10 tools que o modelo aprende chamar via special tokens <tool_call>.
"""

from .cwe_lookup import cwe_lookup, cwe_search_kw, cwe_tree_path, cwe_view_filter
from .verify_fit import verify_fit

# TODO Vitor implementar restantes:
# - attack_lookup (MITRE ATT&CK)
# - cve_similar (vector search past CVEs)
# - cwe_micro_rubric (when CWE-X applies)
# - cvss_calc (severity)
# - nvd_fetch (official NVD entry)

TOOL_REGISTRY = {
    "cwe_lookup": cwe_lookup,
    "cwe_tree_path": cwe_tree_path,
    "cwe_search_kw": cwe_search_kw,
    "cwe_view_filter": cwe_view_filter,
    "verify_fit": verify_fit,
}
