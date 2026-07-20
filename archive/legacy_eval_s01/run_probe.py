"""Probe set rapido (50 prompts) pra ancora de delta entre checkpoints.

Mede perplexidade media + CWE hit rate (substring match) num set fixo.
Roda em CPU se nao houver GPU.

Usage:
    python eval/run_probe.py \\
        --adapter ./ckpt-out \\
        --out eval/reports/probe-step-5000.json
"""

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval._common import BASE_MODEL, REPO_ROOT, load_model  # noqa: E402

logger = logging.getLogger(__name__)

CWE_RE = re.compile(r"CWE-\d{2,4}", re.IGNORECASE)


def load_probes():
    p = REPO_ROOT / "eval" / "probe_set.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def compute_perplexity(model, tokenizer, text, device):
    import torch

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
    return math.exp(out.loss.item())


def generate(model, tokenizer, prompt, device, max_new=128):
    import torch

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)


def score_cwe(expected_cwe, response):
    found = {m.upper() for m in CWE_RE.findall(response)}
    return 1.0 if expected_cwe.upper() in found else 0.0


def run(model, tokenizer, device, max_new=128):
    """Roda probe set. Retorna dict de metrics. Reutilizado por run_bench_all."""
    probes = load_probes()
    results, ppls, cwe_hits, cwe_total = [], [], 0, 0
    for i, probe in enumerate(probes):
        response = generate(model, tokenizer, probe["prompt"], device, max_new)
        ppl = compute_perplexity(model, tokenizer, (probe["prompt"] + response)[:2048], device)
        ppls.append(ppl)
        score = None
        if probe.get("expected_cwe"):
            score = score_cwe(probe["expected_cwe"], response)
            cwe_hits += score
            cwe_total += 1
        results.append(
            {
                "id": probe["id"],
                "expected_cwe": probe.get("expected_cwe"),
                "response": response[:512],
                "ppl": ppl,
                "cwe_hit": score,
            }
        )
        logger.info(f"[{i + 1}/{len(probes)}] {probe['id']} ppl={ppl:.2f} cwe_hit={score}")
    return {
        "n_probes": len(probes),
        "mean_ppl": sum(ppls) / len(ppls) if ppls else 0.0,
        "cwe_hit_rate": cwe_hits / cwe_total if cwe_total else None,
        "cwe_total": cwe_total,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="Path adapter LoRA (default base puro)")
    parser.add_argument("--base", default=BASE_MODEL)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new", type=int, default=128)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model, tokenizer, device = load_model(args.adapter, args.base)
    report = run(model, tokenizer, device, args.max_new)
    report["adapter"] = args.adapter
    report["base"] = args.base

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    logger.info(f"mean_ppl={report['mean_ppl']:.2f} cwe_hit={report['cwe_hit_rate']} -> {out}")


if __name__ == "__main__":
    main()
