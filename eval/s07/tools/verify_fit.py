"""Verifier tool - separate model judges if CWE fits CVE.

Para 3B avoid self-critique (Meta finding <7B unreliable).
Usa Caracal s05 ou Haiku 4.5 batch como verifier.
"""


def verify_fit(cve_description: str, cwe_id: str) -> dict:
    """Returns {fit: bool, confidence: float, reasoning: str}."""
    # TODO Vitor: Caracal s05 inference OR Haiku 4.5 batch
    raise NotImplementedError("verify_fit - implement em s07.D")
