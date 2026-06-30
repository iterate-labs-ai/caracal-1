"""NVD API v2 client - puxa CVEs 2025-2026 pra RSD self-improve.

s07.F outer loop weekly batch.
"""

# TODO Pedro:
# - NVD API client (free tier: 5 req/30s, 50 req/30s with key)
# - filtro lastModStartDate=2025-01-01
# - exclude CVE-IDs ja em training set (decontam)
# - output: ~10K new CVEs/week JSONL

NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"
COMMUNITY_MIRROR = "https://github.com/fkie-cad/nvd-json-data-feeds"
RATE_LIMIT_FREE = (5, 30)  # 5 req per 30s no key
RATE_LIMIT_KEY = (50, 30)  # com NVD API key


def fetch_recent_cves(start_date: str = "2025-01-01", limit: int = 10_000) -> list[dict]:
    """Fetch CVEs after start_date."""
    raise NotImplementedError("NVD fetch - implement em s07.F")


def filter_decontam(cves: list[dict], known_ids: set[str]) -> list[dict]:
    """Skip CVEs ja em training set."""
    return [c for c in cves if c["cve"]["id"] not in known_ids]
