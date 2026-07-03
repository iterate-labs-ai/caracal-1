"""s07.F.1 - Recursive Self-Distill outer loop.

Pedro session 6/7 (parte 1). Weekly batch NVD feed + verifier filter + retrain.
"""

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
_nvd_feed = importlib.import_module("data.s07.nvd_feed")
extract_cve_cwe = _nvd_feed.extract_cve_cwe
fetch_recent_cves = _nvd_feed.fetch_recent_cves

CWE_RE = re.compile(r"CWE-?(\d{1,4})", re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{(CWE-?\d{1,4})\}", re.IGNORECASE)


def normalize_cwe(text: str) -> str | None:
    m = BOXED_RE.search(text)
    if m:
        inner = CWE_RE.search(m.group(1))
        if inner:
            return f"CWE-{int(inner.group(1))}"
    for line in reversed(text.splitlines()):
        m = CWE_RE.search(line)
        if m:
            return f"CWE-{int(m.group(1))}"
    return None


def generate_traces(model, tok, cves: list[dict], max_new: int = 256) -> list[dict]:
    import torch

    traces = []
    for c in cves:
        prompt = tok.apply_chat_template(
            [
                {"role": "system", "content": "Analyze CVE. Output \\boxed{CWE-NNN}."},
                {"role": "user", "content": f"CVE Description: {c['description']}"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        text = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)
        pred = normalize_cwe(text)
        traces.append(
            {
                "cve_id": c["cve_id"],
                "description": c["description"],
                "gold": c["cwe_id"],
                "pred": pred,
                "trace": text,
            }
        )
    return traces


def filter_gold(traces: list[dict]) -> list[dict]:
    """Rule-based: keep only where pred == gold."""
    return [t for t in traces if t["pred"] and t["pred"] == t["gold"]]


def build_train_mix(gold: list[dict], original_hf: str, new_ratio: float = 0.30):
    from datasets import Dataset, concatenate_datasets, load_dataset

    original = load_dataset(original_hf, split="train")
    n_new = len(gold)
    n_orig = int(n_new * (1 - new_ratio) / new_ratio)
    orig_sample = original.shuffle(seed=42).select(range(min(n_orig, len(original))))
    new_ds = Dataset.from_list(
        [{"description": g["description"], "cwe_id": g["gold"]} for g in gold]
    )
    return concatenate_datasets([orig_sample, new_ds]).shuffle(seed=42)


def retrain_lora(model, tok, ds, out_dir: Path, epochs: int = 1):
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    def fmt(row):
        text = tok.apply_chat_template(
            [
                {"role": "user", "content": f"CVE Description: {row['description']}"},
                {"role": "assistant", "content": f"\\boxed{{{row['cwe_id']}}}"},
            ],
            tokenize=False,
        )
        return {"text": text}

    ds = ds.map(fmt)

    lora = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(out_dir),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            num_train_epochs=epochs,
            learning_rate=1e-5,
            bf16=True,
            logging_steps=25,
            save_strategy="epoch",
            max_seq_length=1024,
        ),
        train_dataset=ds,
        peft_config=lora,
        tokenizer=tok,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    return trainer.model


def main():
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="pedroafonso2/caracal-s07-rl")
    ap.add_argument("--out-dir", type=Path, default=Path("/kaggle/working/caracal-s07-rsd"))
    ap.add_argument("--start-date", default="2025-01-01T00:00:00.000")
    ap.add_argument("--n-cves", type=int, default=10_000)
    ap.add_argument("--max-rounds", type=int, default=2)
    ap.add_argument("--original-data", default="xamxte/cve-to-cwe")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map={"": "cuda:0"}
    )

    print(f"[s07.F] fetching NVD from {args.start_date}, limit {args.n_cves}")
    api_key = os.environ.get("NVD_API_KEY")
    raw = fetch_recent_cves(args.start_date, args.n_cves, api_key)
    cves = extract_cve_cwe(raw)
    print(f"[s07.F] {len(cves)} CVEs extracted")

    prev_gold_size = 0
    for round_i in range(args.max_rounds):
        print(f"\n=== RSD round {round_i + 1}/{args.max_rounds} ===")
        traces = generate_traces(model, tok, cves)
        gold = filter_gold(traces)
        print(f"[round {round_i + 1}] {len(gold)}/{len(traces)} gold traces (rule-verified)")
        if abs(len(gold) - prev_gold_size) < len(traces) * 0.01:
            print(f"[round {round_i + 1}] converged, stopping")
            break
        prev_gold_size = len(gold)

        ds = build_train_mix(gold, args.original_data)
        round_out = args.out_dir / f"round-{round_i + 1}"
        model = retrain_lora(model, tok, ds, round_out, epochs=1)
        (round_out / "traces.jsonl").write_text("\n".join(json.dumps(t) for t in gold))

    print(f"[s07.F] final model -> {args.out_dir}")


if __name__ == "__main__":
    main()
