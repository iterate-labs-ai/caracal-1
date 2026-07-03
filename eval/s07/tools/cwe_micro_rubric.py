"""Micro-rubric por CWE - "quando aplica X?" rules.

Static rules pra top-25 CWE, extendable via YAML config.
"""

MICRO_RUBRICS = {
    "CWE-79": "Cross-Site Scripting: user input reflected in HTML without escape. Look for: XSS, script injection, DOM manipulation.",
    "CWE-89": "SQL Injection: user input concatenated into SQL query. Look for: sqli, prepared statements, parameterized query.",
    "CWE-121": "Stack Buffer Overflow: writing past stack allocation. Look for: stack, local variable overflow, saved return addr.",
    "CWE-122": "Heap Buffer Overflow: writing past heap allocation. Look for: malloc, heap, new().",
    "CWE-125": "Out-of-bounds Read: read beyond buffer. Look for: OOB read, info disclosure via memory.",
    "CWE-787": "Out-of-bounds Write: write beyond buffer. Look for: memory corruption, OOB write.",
    "CWE-416": "Use After Free: access to freed memory. Look for: UAF, dangling pointer.",
    "CWE-476": "NULL Pointer Dereference: crash from null deref.",
    "CWE-119": "Buffer Overflow (generic): use only when specific type unclear.",
    "CWE-20": "Improper Input Validation: catch-all when nothing more specific fits.",
    "CWE-22": "Path Traversal: ../ in path, escape from restricted dir.",
    "CWE-78": "OS Command Injection: user input to shell/exec.",
    "CWE-94": "Code Injection: eval, deserialization, code execution.",
    "CWE-352": "CSRF: state-changing request from other origin without token.",
    "CWE-434": "Unrestricted Upload: file upload without type/extension check.",
    "CWE-862": "Missing Authorization: functionality accessible without auth check.",
    "CWE-863": "Incorrect Authorization: authorization bypass despite check.",
    "CWE-306": "Missing Authentication: function accessible without any credential.",
    "CWE-798": "Hard-coded Credentials: hardcoded password, token, key in code.",
    "CWE-284": "Improper Access Control: permission model failure (generic).",
    "CWE-200": "Info Exposure: data disclosed to unauthorized party.",
    "CWE-190": "Integer Overflow: arithmetic beyond type range.",
    "CWE-400": "Uncontrolled Resource Consumption: DoS via resource exhaustion.",
    "CWE-502": "Deserialization of Untrusted Data: pickle, ObjectInputStream on untrusted.",
    "CWE-611": "XXE: XML external entity, DTD, SSRF via XML.",
}


def cwe_micro_rubric(cwe_id: str) -> dict:
    """Returns disambiguation rule pra CWE."""
    cwe = cwe_id.upper()
    if not cwe.startswith("CWE-"):
        cwe = f"CWE-{cwe}"
    rubric = MICRO_RUBRICS.get(cwe)
    if not rubric:
        return {"cwe_id": cwe, "rubric": None, "note": "no rubric available (not top-25)"}
    return {"cwe_id": cwe, "rubric": rubric}
