"""CTI-Bench RCM + VSP subtasks - direto pra transferencia treino vuln.

- RCM: CVE description -> CWE-NNN mapping (exact-match, regex)
- VSP: vulnerability -> severity classification (LOW/MED/HIGH/CRIT)

Dataset: AI4Sec/cti-bench (HuggingFace, public, no auth).
Generate-based mas max_new=20 + greedy = rapido em T4 (~30min total).

Usage:
    python eval/run_cti_bench.py --adapter ./ckpt --out cti.json
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval._common import load_model  # noqa: E402
from eval.run_probe import generate  # noqa: E402

logger = logging.getLogger(__name__)

CWE_RE = re.compile(r"CWE-?(\d{1,4})", re.IGNORECASE)
SEVERITY_LABELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

RCM_PROMPT = (
    "Map the following CVE description to the most appropriate CWE identifier. "
    "Output ONLY the CWE in format CWE-NNN.\n\n"
    "CVE description: {description}\n"
    "CWE:"
)
VSP_PROMPT = (
    "Predict the severity of the following vulnerability. "
    "Output ONLY one of: LOW, MEDIUM, HIGH, CRITICAL.\n\n"
    "Vulnerability: {description}\n"
    "Severity:"
)


def load_subset(subset):
    from datasets import load_dataset

    ds = load_dataset("AI4Sec/cti-bench", subset, split="test")
    return list(ds)


def normalize_cwe(s):
    m = CWE_RE.search(s)
    return f"CWE-{int(m.group(1))}" if m else None


def normalize_severity(s):
    up = s.upper()
    for label in SEVERITY_LABELS:
        if label in up:
            return label
    return None


def run_rcm(model, tokenizer, device, max_n=None):
    rows = load_subset("cti-rcm")
    if max_n:
        rows = rows[:max_n]
    n_correct = 0
    results = []
    for i, r in enumerate(rows):
        prompt = RCM_PROMPT.format(description=r["Prompt"])
        resp = generate(model, tokenizer, prompt, device, max_new=20)
        pred = normalize_cwe(resp)
        gold = normalize_cwe(r["GT"])
        correct = pred is not None and pred == gold
        n_correct += int(correct)
        results.append({"i": i, "pred": pred, "gold": gold, "correct": correct})
        if (i + 1) % 100 == 0:
            logger.info(f"[rcm {i + 1}/{len(rows)}] acc = {n_correct / (i + 1):.3f}")
    return {
        "n_total": len(rows),
        "n_correct": n_correct,
        "accuracy": n_correct / len(rows) if rows else 0.0,
        "results": results,
    }


def run_vsp(model, tokenizer, device, max_n=None):
    rows = load_subset("cti-vsp")
    if max_n:
        rows = rows[:max_n]
    n_correct = 0
    results = []
    for i, r in enumerate(rows):
        prompt = VSP_PROMPT.format(description=r["Prompt"])
        resp = generate(model, tokenizer, prompt, device, max_new=10)
        pred = normalize_severity(resp)
        gold = normalize_severity(r["GT"])
        correct = pred is not None and pred == gold
        n_correct += int(correct)
        results.append({"i": i, "pred": pred, "gold": gold, "correct": correct})
        if (i + 1) % 100 == 0:
            logger.info(f"[vsp {i + 1}/{len(rows)}] acc = {n_correct / (i + 1):.3f}")
    return {
        "n_total": len(rows),
        "n_correct": n_correct,
        "accuracy": n_correct / len(rows) if rows else 0.0,
        "results": results,
    }


def run(model, tokenizer, device, max_n=None):
    rcm = run_rcm(model, tokenizer, device, max_n)
    vsp = run_vsp(model, tokenizer, device, max_n)
    return {"rcm": rcm, "vsp": vsp}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir (default base puro)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-n", type=int, default=None, help="Limit n samples per subset")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model, tokenizer, device = load_model(args.adapter)
    summary = run(model, tokenizer, device, args.max_n)
    summary["adapter"] = args.adapter
    Path(args.out).write_text(json.dumps(summary, indent=2))
    logger.info(
        f"rcm acc={summary['rcm']['accuracy']:.3f} vsp acc={summary['vsp']['accuracy']:.3f} -> {args.out}"
    )


if __name__ == "__main__":
    main()
