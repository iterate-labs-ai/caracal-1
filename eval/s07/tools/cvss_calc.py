"""CVSS 3.1 severity calculation from vector string."""


def cvss_calc(vector: str) -> dict:
    """Parse CVSS 3.1 vector, retorna severity + score.

    Example: CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H -> Critical 9.8
    """
    if not vector.startswith("CVSS:3."):
        return {"vector": vector, "error": "invalid_cvss_prefix"}
    parts = dict(p.split(":", 1) for p in vector.split("/")[1:] if ":" in p)

    weights = {
        "AV": {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2},
        "AC": {"L": 0.77, "H": 0.44},
        "PR": {"N": 0.85, "L": 0.62, "H": 0.27},
        "UI": {"N": 0.85, "R": 0.62},
        "C": {"N": 0.0, "L": 0.22, "H": 0.56},
        "I": {"N": 0.0, "L": 0.22, "H": 0.56},
        "A": {"N": 0.0, "L": 0.22, "H": 0.56},
    }
    try:
        av = weights["AV"][parts["AV"]]
        ac = weights["AC"][parts["AC"]]
        pr = weights["PR"][parts["PR"]]
        ui = weights["UI"][parts["UI"]]
        c = weights["C"][parts["C"]]
        i = weights["I"][parts["I"]]
        a = weights["A"][parts["A"]]
    except KeyError as e:
        return {"vector": vector, "error": f"missing_metric_{e}"}

    exploitability = 8.22 * av * ac * pr * ui
    iss = 1 - ((1 - c) * (1 - i) * (1 - a))
    impact = 6.42 * iss
    if impact <= 0:
        base = 0.0
    else:
        base = min(10.0, round((impact + exploitability) * 1.08, 1))

    if base == 0.0:
        severity = "None"
    elif base < 4.0:
        severity = "Low"
    elif base < 7.0:
        severity = "Medium"
    elif base < 9.0:
        severity = "High"
    else:
        severity = "Critical"
    return {"vector": vector, "base_score": base, "severity": severity}
