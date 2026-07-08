"""NVD API v2 - puxa CVEs pra RSD self-improve."""

import json
import time
from pathlib import Path

import requests

NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"
PAGE_SIZE = 2000
RATE_LIMIT_KEY_SEC = 0.6
RATE_LIMIT_NO_KEY_SEC = 6.0


def fetch_recent_cves(
    start_date: str = "2025-01-01T00:00:00.000", limit: int = 10_000, api_key: str | None = None
) -> list[dict]:
    """Fetch CVEs after start_date via NVD API v2."""
    headers = {"apiKey": api_key} if api_key else {}
    sleep = RATE_LIMIT_KEY_SEC if api_key else RATE_LIMIT_NO_KEY_SEC
    out = []
    start_idx = 0
    while len(out) < limit:
        params = {
            "lastModStartDate": start_date,
            "resultsPerPage": PAGE_SIZE,
            "startIndex": start_idx,
        }
        r = requests.get(NVD_API, params=params, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
        vulns = data.get("vulnerabilities", [])
        if not vulns:
            break
        out.extend(vulns)
        start_idx += PAGE_SIZE
        if len(vulns) < PAGE_SIZE:
            break
        time.sleep(sleep)
    return out[:limit]


def extract_cve_cwe(vulns: list[dict]) -> list[dict]:
    """Reduce raw NVD entries to (cve_id, description, cwe_id) tuples."""
    out = []
    for v in vulns:
        cve = v.get("cve", {})
        cve_id = cve.get("id")
        descs = cve.get("descriptions", [])
        eng = next((d["value"] for d in descs if d.get("lang") == "en"), None)
        weaknesses = cve.get("weaknesses", [])
        cwes = []
        for w in weaknesses:
            for d in w.get("description", []):
                if d.get("lang") == "en" and d.get("value", "").startswith("CWE-"):
                    cwes.append(d["value"])
        if not (cve_id and eng and cwes):
            continue
        out.append({"cve_id": cve_id, "description": eng, "cwe_id": cwes[0]})
    return out


def filter_decontam(cves: list[dict], known_ids: set[str]) -> list[dict]:
    return [c for c in cves if c["cve_id"] not in known_ids]


def main():
    import argparse
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-01T00:00:00.000")
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument("--out", type=Path, default=Path("data/nvd_feed.jsonl"))
    args = ap.parse_args()

    api_key = os.environ.get("NVD_API_KEY")
    raw = fetch_recent_cves(args.start, args.limit, api_key)
    reduced = extract_cve_cwe(raw)
    with args.out.open("w") as f:
        for row in reduced:
            f.write(json.dumps(row) + "\n")
    print(f"{len(reduced)} CVEs -> {args.out}")


if __name__ == "__main__":
    main()
