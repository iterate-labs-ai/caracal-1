"""CTI-Bench RCM subtask - CVE description -> CWE-NNN mapping.

VSP subtask removida: dataset retorna vetor CVSS completo (CVSS:3.1/AV:L/...),
nao label de severidade. Modelo 3B nao gera CVSS vector preciso, 100% fail.

Dataset: AI4Sec/cti-bench (HuggingFace TSV direto, via pandas).
Generate-based mas max_new=20 + greedy = rapido em T4 (~15min RCM 500).

NAO RODAR EM TPU: usa .generate() iterativo, XLA recompila por shape novo.
Em TPU vira >5h e estoura limite. Skip com --skip-cti-bench em notebooks TPU.

Usage:
    python eval/run_cti_bench.py --adapter ./ckpt --out cti.json
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval._common import load_model  # noqa: E402
from eval.run_probe import generate  # noqa: E402

logger = logging.getLogger(__name__)

CWE_RE = re.compile(r"CWE-?(\d{1,4})", re.IGNORECASE)
HF_BASE = "https://huggingface.co/datasets/AI4Sec/cti-bench/resolve/main"

# Dataset coluna "Prompt" ja vem com instrucao oficial CTI-Bench
# ("Analyze... last line of your response contains only the CWE ID").
# NAO envolver em template proprio - duplicaria instrucao e quebraria parse.


def load_subset(subset):
    """csv stdlib - HF datasets library quebra no TSV (quoted fields irregulares)."""
    url = f"{HF_BASE}/{subset}.tsv"
    fd, p = tempfile.mkstemp(suffix=".tsv")
    os.close(fd)
    try:
        urllib.request.urlretrieve(url, p)
        with open(p, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            rows = list(reader)
    finally:
        os.remove(p)
    return rows


def normalize_cwe(s):
    """Extrai CWE-NNN. Prioriza ultima linha (prompt oficial pede ultima linha = CWE ID)."""
    for line in reversed(s.splitlines()):
        m = CWE_RE.search(line)
        if m:
            return f"CWE-{int(m.group(1))}"
    m = CWE_RE.search(s)
    return f"CWE-{int(m.group(1))}" if m else None


def run(model, tokenizer, device, max_n=None):
    """RCM = CVE description -> CWE-NNN. Unica subtask viavel sem CVSS vector gen."""
    rows = load_subset("cti-rcm")
    if max_n:
        rows = rows[:max_n]
    n_correct = 0
    results = []
    for i, r in enumerate(rows):
        # max_new=64 pra permitir "justification" + CWE final como pede prompt oficial
        resp = generate(model, tokenizer, r["Prompt"], device, max_new=64)
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
    logger.info(f"rcm acc={summary['accuracy']:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
