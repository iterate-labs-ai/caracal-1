"""CTI-Bench (NeurIPS 2024, AI4Sec) - 5 subtasks.

- RCM: CVE description -> CWE-NNN (1000 samples)
- MCQ: cyber threat intelligence MCQ (2500 samples)
- VSP: CVE description -> CVSS vector
- TAA: threat actor attribution
- ATE: attack pattern to enterprise techniques
"""

import csv
import os
import tempfile
import urllib.request

from ._common import bootstrap_ci, generate, normalize_cwe

CTI_BENCH_HF_BASE = "https://huggingface.co/datasets/AI4Sec/cti-bench/resolve/main"


def _load_tsv(subset: str) -> list[dict]:
    url = f"{CTI_BENCH_HF_BASE}/{subset}.tsv"
    fd, tmp = tempfile.mkstemp(suffix=".tsv")
    os.close(fd)
    try:
        urllib.request.urlretrieve(url, tmp)
        with open(tmp) as f:
            return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    finally:
        os.unlink(tmp)


def _eval_rcm(model, tok, n: int) -> dict:
    rows = _load_tsv("cti-rcm")[:n]
    per, correct = [], []
    for i, r in enumerate(rows):
        resp = generate(model, tok, r["Prompt"], max_new=256)
        pred = normalize_cwe(resp)
        gold = normalize_cwe(r["GT"])
        ok = int(pred is not None and pred == gold)
        correct.append(ok)
        per.append({"i": i, "pred": pred, "gold": gold, "correct": ok})
        if (i + 1) % 100 == 0:
            print(f"[cti_rcm {i + 1}/{len(rows)}] acc={sum(correct) / (i + 1):.3f}")
    ci_lo, ci_hi = bootstrap_ci(correct)
    return {
        "n": len(correct),
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
        "ci_95_low": ci_lo,
        "ci_95_high": ci_hi,
        "per_sample": per,
    }


def _eval_mcq(model, tok, n: int) -> dict:
    rows = _load_tsv("cti-mcq")[:n]
    per, correct = [], []
    for i, r in enumerate(rows):
        resp = generate(model, tok, r["Prompt"], max_new=64)
        from ._common import normalize_mcq_letter

        pred = normalize_mcq_letter(resp)
        gold = r["GT"].strip().upper()
        ok = int(pred == gold)
        correct.append(ok)
        per.append({"i": i, "pred": pred, "gold": gold, "correct": ok})
    ci_lo, ci_hi = bootstrap_ci(correct)
    return {
        "n": len(correct),
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
        "ci_95_low": ci_lo,
        "ci_95_high": ci_hi,
    }


def _eval_gen_subset(model, tok, subset: str, n: int, gt_normalize=None) -> dict:
    """Generic generation-based subset (VSP, TAA, ATE)."""
    rows = _load_tsv(subset)[:n]
    per, correct = [], []
    for i, r in enumerate(rows):
        resp = generate(model, tok, r["Prompt"], max_new=256)
        pred = gt_normalize(resp) if gt_normalize else resp.strip()
        gold = gt_normalize(r["GT"]) if gt_normalize else r["GT"].strip()
        ok = int(pred == gold)
        correct.append(ok)
        per.append({"i": i, "pred": pred, "gold": gold, "correct": ok})
    ci_lo, ci_hi = bootstrap_ci(correct)
    return {
        "n": len(correct),
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
        "ci_95_low": ci_lo,
        "ci_95_high": ci_hi,
    }


def eval_cti_bench(
    model,
    tok,
    n_rcm: int = 1000,
    n_mcq: int = 500,
    n_vsp: int = 500,
    n_taa: int = 200,
    n_ate: int = 200,
    subsets: list[str] | None = None,
) -> dict:
    """CTI-Bench full: RCM (CVE->CWE), MCQ, VSP (CVSS), TAA (attribution), ATE (technique)."""
    subsets = subsets or ["rcm", "mcq", "vsp", "taa", "ate"]
    out = {}
    if "rcm" in subsets:
        print("[cti_bench] running RCM...")
        out["rcm"] = _eval_rcm(model, tok, n_rcm)
    if "mcq" in subsets:
        print("[cti_bench] running MCQ...")
        out["mcq"] = _eval_mcq(model, tok, n_mcq)
    if "vsp" in subsets:
        print("[cti_bench] running VSP...")
        out["vsp"] = _eval_gen_subset(model, tok, "cti-vsp", n_vsp)
    if "taa" in subsets:
        print("[cti_bench] running TAA...")
        out["taa"] = _eval_gen_subset(model, tok, "cti-taa", n_taa)
    if "ate" in subsets:
        print("[cti_bench] running ATE...")
        out["ate"] = _eval_gen_subset(model, tok, "cti-ate", n_ate)
    return out
