"""Continued pretrain Caracal Base 3B usando Unsloth + TRL SFTTrainer.

Stack: Qwen2.5-Coder-3B-Instruct + LoRA r=32 alpha=64 em q/k/v/o + gate/up/down.
Decontamination via CVE-ID blocklist (data/decontamination/cve_blocklist.json).
Pesos vivem como Kaggle Datasets publicos.

Usage:
    python train/continued_pretrain.py \\
        --resume-from ./ckpt-in \\
        --steps-to-run 4000 \\
        --output ./ckpt-out
"""

import argparse
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"

LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

HF_DATASETS = [
    ("PrimeVul", "ussooraj/PrimeVul", "train.jsonl"),
    ("BigVul", "bstee615/bigvul", None),
    ("DiverseVul", "bstee615/diversevul", None),
]

TEXT_FIELDS = ["func", "func_before", "code", "text", "function", "source"]
CVE_FIELDS = ["cve", "cve_id", "CVE", "cveid", "cve_list"]
CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)


def detect_text_field(sample):
    for f in TEXT_FIELDS:
        v = sample.get(f)
        if isinstance(v, str) and len(v) > 20:
            return f
    for k, v in sample.items():
        if isinstance(v, str) and len(v) > 20:
            return k
    raise ValueError(f"No valid text field: {list(sample.keys())}")


def extract_cve_ids(sample):
    found = set()
    for f in CVE_FIELDS:
        v = sample.get(f)
        if isinstance(v, str):
            found.update(m.upper() for m in CVE_RE.findall(v))
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, str):
                    found.update(m.upper() for m in CVE_RE.findall(x))
    for v in sample.values():
        if isinstance(v, str) and len(v) < 10000:
            found.update(m.upper() for m in CVE_RE.findall(v))
    return found


def load_cve_blocklist():
    p = REPO_ROOT / "data" / "decontamination" / "cve_blocklist.json"
    if not p.exists():
        logger.warning(f"Blocklist ausente em {p}. Decontam DISABLED.")
        return set()
    data = json.loads(p.read_text())
    blocklist = {c.upper() for c in data.get("blocked", [])}
    logger.info(f"Loaded {len(blocklist)} blocked CVE IDs")
    return blocklist


def filter_decontam(dataset, blocklist):
    if not blocklist:
        return dataset
    before = len(dataset)
    dataset = dataset.filter(lambda s: not (extract_cve_ids(s) & blocklist))
    logger.info(f"Decontam: {before} -> {len(dataset)}")
    return dataset


def flatten_conversation(sample):
    parts = []
    for turn in sample.get("conversations") or []:
        if not isinstance(turn, dict):
            continue
        val = turn.get("value") or turn.get("content") or ""
        if isinstance(val, str) and val.strip():
            role = turn.get("role") or turn.get("from") or "user"
            parts.append(f"<|{role}|>\n{val}")
    return {"text": "\n\n".join(parts)}


def load_one_dataset(name, hf_id, data_files, decontam, max_n, blocklist):
    from datasets import load_dataset

    logger.info(f"Loading {name} ({hf_id} files={data_files})")
    if data_files:
        ds = load_dataset(hf_id, data_files=data_files, split="train")
    else:
        ds = load_dataset(hf_id, split="train")

    if max_n:
        ds = ds.select(range(min(max_n, len(ds))))
    if decontam:
        ds = filter_decontam(ds, blocklist)

    if "conversations" in ds.column_names:
        logger.info(f"  {name}: schema=conversations, flattening")
        ds = ds.map(flatten_conversation, remove_columns=ds.column_names)
    else:
        field = detect_text_field(ds[0])
        if field != "text":
            ds = ds.rename_column(field, "text")
        ds = ds.remove_columns([c for c in ds.column_names if c != "text"])
    ds = ds.filter(lambda s: isinstance(s["text"], str) and len(s["text"]) >= 50)
    logger.info(f"  {name}: {len(ds)} examples")
    return ds


def load_all_datasets(decontam, max_per_dataset):
    from datasets import concatenate_datasets

    blocklist = load_cve_blocklist() if decontam else set()
    parts = [
        load_one_dataset(name, hf, files, decontam, max_per_dataset, blocklist)
        for name, hf, files in HF_DATASETS
    ]
    combined = concatenate_datasets(parts)
    logger.info(f"Combined: {len(combined)} examples")
    return combined


def build_model(args):
    from unsloth import FastLanguageModel

    if args.resume_from and Path(args.resume_from).exists():
        base_path = args.resume_from
        logger.info(f"Resume from local: {base_path}")
        is_resume = True
    else:
        base_path = BASE_MODEL
        logger.info(f"Cold start: {base_path}")
        is_resume = False

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )

    if not is_resume:
        logger.info(f"LoRA r={args.lora_r} alpha={args.lora_alpha}")
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            target_modules=LORA_TARGETS,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

    return model, tokenizer


def build_trainer(model, tokenizer, dataset, args):
    from trl import SFTConfig, SFTTrainer

    cfg = SFTConfig(
        output_dir=args.output,
        max_steps=args.steps_to_run,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=False,
        fp16=True,
        optim="adamw_8bit",
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=args.save_every,
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
        push_to_hub=False,
        seed=42,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=True,
        dataset_num_proc=2,
    )

    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=cfg,
    )


def save_run_log(trainer, args):
    log_path = Path(args.output) / "train_log.json"
    log_path.write_text(
        json.dumps(
            {
                "global_step": trainer.state.global_step,
                "log_history": trainer.state.log_history[-50:],
                "args": vars(args),
            },
            indent=2,
            default=str,
        )
    )
    logger.info(f"Log: {log_path}")


def train(args):
    model, tokenizer = build_model(args)
    dataset = load_all_datasets(decontam=args.decontam, max_per_dataset=args.max_per_dataset)

    if args.smoke:
        logger.info("SMOKE: 100 samples + 5 steps")
        dataset = dataset.select(range(min(100, len(dataset))))
        args.steps_to_run = 5

    trainer = build_trainer(model, tokenizer, dataset, args)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable: {n:,} | steps: {args.steps_to_run}")
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    save_run_log(trainer, args)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume-from", default=None)
    p.add_argument("--steps-to-run", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--decontam", action="store_true", default=True)
    p.add_argument("--no-decontam", dest="decontam", action="store_false")
    p.add_argument("--max-per-dataset", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
