"""MITRE ATT&CK technique lookup."""

import json
from pathlib import Path

_MITRE_INDEX: dict[str, dict] | None = None


def _load_mitre() -> dict[str, dict]:
    """Load MITRE ATT&CK enterprise techniques from JSON.

    Download: https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json
    """
    global _MITRE_INDEX
    if _MITRE_INDEX is not None:
        return _MITRE_INDEX
    import os

    path = Path(os.environ.get("MITRE_ATTACK_JSON", "data/enterprise-attack.json"))
    if not path.exists():
        _MITRE_INDEX = {}
        return _MITRE_INDEX
    raw = json.loads(path.read_text())
    idx = {}
    for obj in raw.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        ext_refs = obj.get("external_references", [])
        ext = next((r for r in ext_refs if r.get("source_name") == "mitre-attack"), None)
        if not ext:
            continue
        idx[ext["external_id"]] = {
            "name": obj.get("name"),
            "description": obj.get("description", "")[:500],
            "tactics": [p["phase_name"] for p in obj.get("kill_chain_phases", [])],
        }
    _MITRE_INDEX = idx
    return idx


def attack_lookup(technique_id: str) -> dict:
    """Returns MITRE ATT&CK technique detail."""
    idx = _load_mitre()
    tid = technique_id.upper()
    if tid not in idx:
        return {"technique_id": tid, "error": "unknown_technique"}
    return {"technique_id": tid, **idx[tid]}
