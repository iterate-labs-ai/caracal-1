"""Fetch official NVD entry por CVE-ID."""

import os

import requests

NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"


def nvd_fetch(cve_id: str) -> dict:
    """Fetch canonical NVD CVE entry."""
    headers = {"apiKey": os.environ["NVD_API_KEY"]} if os.environ.get("NVD_API_KEY") else {}
    r = requests.get(NVD_API, params={"cveId": cve_id}, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    vulns = data.get("vulnerabilities", [])
    if not vulns:
        return {"cve_id": cve_id, "error": "not_found"}
    cve = vulns[0].get("cve", {})
    descs = cve.get("descriptions", [])
    eng = next((d["value"] for d in descs if d.get("lang") == "en"), "")
    weaknesses = cve.get("weaknesses", [])
    cwes = []
    for w in weaknesses:
        for d in w.get("description", []):
            if d.get("lang") == "en" and d.get("value", "").startswith("CWE-"):
                cwes.append(d["value"])
    metrics = cve.get("metrics", {})
    cvss = None
    if "cvssMetricV31" in metrics and metrics["cvssMetricV31"]:
        m = metrics["cvssMetricV31"][0].get("cvssData", {})
        cvss = m.get("vectorString")
    return {
        "cve_id": cve_id,
        "description": eng,
        "cwes": cwes,
        "cvss_vector": cvss,
        "published": cve.get("published"),
    }
