"""CyberGym benchmark loader + runner.

Carrega sunblaze-ucb/cybergym, roda subset, mede pass@1/pass@5.
Sandbox real (Docker) sai em v2. v0/v1 = response gen + substring match.

Usage:
    python eval/run_cybergym.py --subset slice-50 --adapter ./ckpt-out
    python eval/run_cybergym.py --subset full-1507 --adapter ... --k 5
"""

import argparse
import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"


SUBSET_SIZES = {
    "slice-50": 50,
    "subset-200": 200,
    "full-1507": None,
}


def load_cybergym(subset):
    from datasets import load_dataset

    logger.info("Loading sunblaze-ucb/cybergym")
    ds = load_dataset("sunblaze-ucb/cybergym", split="train")
    n = SUBSET_SIZES.get(subset)
    if n:
        random.seed(42)
        idxs = random.sample(range(len(ds)), min(n, len(ds)))
        ds = ds.select(sorted(idxs))
    logger.info(f"Subset '{subset}': {len(ds)} samples")
    return ds


def load_model(adapter, base):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(adapter or base)
    base_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype, device_map=device)

    if adapter and (Path(adapter) / "adapter_config.json").exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(base_model, adapter)
    else:
        model = base_model
    return model, tokenizer, device


def gen_attempts(model, tokenizer, prompt, device, gen_cfg):
    import torch

    k = gen_cfg["k"]
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    attempts = []
    for _ in range(k):
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=gen_cfg["max_new"],
                do_sample=k > 1,
                temperature=gen_cfg["temp"],
                top_p=gen_cfg["top_p"],
                pad_token_id=tokenizer.eos_token_id,
            )
        attempts.append(
            tokenizer.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)
        )
    return attempts


def score_attempt(attempt, item):
    """Heuristic v0/v1: substring match. v2 vai usar sandbox real."""
    expected = item.get("answer") or item.get("patch") or item.get("solution") or ""
    if not expected:
        return False
    key_lines = [line.strip() for line in expected.splitlines() if len(line.strip()) > 15]
    if not key_lines:
        return False
    hits = sum(1 for line in key_lines if line in attempt)
    return hits >= max(1, len(key_lines) // 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=list(SUBSET_SIZES.keys()), required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--base", default=BASE_MODEL)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--out", default="eval/reports/cybergym.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ds = load_cybergym(args.subset)
    model, tokenizer, device = load_model(args.adapter, args.base)
    model.eval()

    results = []
    pass_at_1 = 0
    pass_at_k = 0
    gen_cfg = {"k": args.k, "temp": args.temp, "top_p": args.top_p, "max_new": args.max_new}

    for i, item in enumerate(ds):
        prompt = item.get("question") or item.get("prompt") or item.get("description", "")
        if not prompt:
            continue
        attempts = gen_attempts(model, tokenizer, prompt, device, gen_cfg)
        scores = [score_attempt(a, item) for a in attempts]
        results.append(
            {
                "idx": i,
                "id": item.get("id", str(i)),
                "n_attempts": len(attempts),
                "any_pass": any(scores),
                "first_pass": scores[0] if scores else False,
            }
        )
        if scores and scores[0]:
            pass_at_1 += 1
        if any(scores):
            pass_at_k += 1
        if (i + 1) % 10 == 0:
            logger.info(
                f"[{i + 1}/{len(ds)}] pass@1={pass_at_1}/{i + 1} pass@k={pass_at_k}/{i + 1}"
            )

    n = len(results)
    report = {
        "subset": args.subset,
        "adapter": args.adapter,
        "base": args.base,
        "k": args.k,
        "n_total": n,
        "pass_at_1": pass_at_1 / n if n else 0.0,
        f"pass_at_{args.k}": pass_at_k / n if n else 0.0,
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    logger.info(f"Wrote {out}: pass@1={report['pass_at_1']:.2%}")


if __name__ == "__main__":
    main()
