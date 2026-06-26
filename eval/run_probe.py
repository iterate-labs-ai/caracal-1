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
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
CWE_RE = re.compile(r"CWE-\d{2,4}", re.IGNORECASE)


def load_probes():
    p = REPO_ROOT / "eval" / "probe_set.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def load_model(adapter_path, base):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        major, _ = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        dtype = torch.float32

    has_adapter = bool(adapter_path) and (Path(adapter_path) / "adapter_config.json").exists()
    tokenizer_path = (
        adapter_path
        if (adapter_path and (Path(adapter_path) / "tokenizer.json").exists())
        else base
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    base_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype, device_map=device)
    if has_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model
    return model, tokenizer, device


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter", default=None, help="Path para adapter LoRA. Vazio = base puro."
    )
    parser.add_argument("--base", default=BASE_MODEL)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new", type=int, default=128)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    probes = load_probes()
    logger.info(f"Loaded {len(probes)} probes")

    model, tokenizer, device = load_model(args.adapter, args.base)
    model.eval()

    results = []
    cwe_hits = 0
    cwe_total = 0
    ppls = []

    for i, probe in enumerate(probes):
        prompt = probe["prompt"]
        expected_cwe = probe.get("expected_cwe")
        response = generate(model, tokenizer, prompt, device, args.max_new)
        full = prompt + response
        ppl = compute_perplexity(model, tokenizer, full[:2048], device)
        ppls.append(ppl)

        score = None
        if expected_cwe:
            score = score_cwe(expected_cwe, response)
            cwe_hits += score
            cwe_total += 1

        results.append(
            {
                "id": probe["id"],
                "expected_cwe": expected_cwe,
                "response": response[:512],
                "ppl": ppl,
                "cwe_hit": score,
            }
        )
        logger.info(f"[{i + 1}/{len(probes)}] {probe['id']} ppl={ppl:.2f} cwe_hit={score}")

    report = {
        "adapter": args.adapter,
        "base": args.base,
        "n_probes": len(probes),
        "mean_ppl": sum(ppls) / len(ppls) if ppls else 0.0,
        "cwe_hit_rate": cwe_hits / cwe_total if cwe_total else None,
        "cwe_total": cwe_total,
        "results": results,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    logger.info(f"Wrote {out}")
    logger.info(f"mean_ppl={report['mean_ppl']:.2f} cwe_hit={report['cwe_hit_rate']}")


if __name__ == "__main__":
    main()
