"""s07.C - Reasoning tokens SFT em teacher CoT traces.

Arthur session 3/7. ~12h Kaggle T4 x2.

Recipe: Fast Quiet-STaR curriculum + Process Reward boxed final.
"""

import argparse
from pathlib import Path


def format_example(row: dict, tokenizer) -> dict:
    """CVE -> <think>reasoning</think> \\boxed{CWE-NNN} traces."""
    system = "You are a defensive cybersecurity assistant. Analyze CVE descriptions step-by-step and output the CWE inside \\boxed{}."
    user = f"CVE Description: {row['description']}\nIdentify the underlying CWE."
    assistant = row["cot_and_answer"]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}


def main():
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="pedroafonso2/caracal-s07-base-tokens")
    ap.add_argument("--traces", default="pedroafonso2/caracal-s05-cot-traces")
    ap.add_argument("--out-dir", type=Path, default=Path("/kaggle/working/caracal-s07-reasoning"))
    ap.add_argument(
        "--epochs-short", type=int, default=4, help="Fast Quiet-STaR phase 1 curriculum"
    )
    ap.add_argument("--epochs-long", type=int, default=4)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map={"": "cuda:0"}
    )

    ds = load_dataset(args.traces, split="train")
    ds = ds.map(lambda r: format_example(r, tokenizer))

    # Curriculum: short traces (< 200 tok) primeiro, depois long (Fast Quiet-STaR)
    def trace_len(r):
        return len(tokenizer.encode(r["cot_and_answer"]))

    ds_short = ds.filter(lambda r: trace_len(r) < 200)
    ds_long = ds.filter(lambda r: trace_len(r) >= 200)

    lora = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    common = dict(
        output_dir=str(args.out_dir),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=25,
        save_strategy="epoch",
        max_seq_length=2048,
    )

    print(f"[s07.C] phase 1: short traces {len(ds_short)} rows x {args.epochs_short} epochs")
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(num_train_epochs=args.epochs_short, **common),
        train_dataset=ds_short,
        peft_config=lora,
        tokenizer=tokenizer,
    )
    trainer.train()

    print(f"[s07.C] phase 2: long traces {len(ds_long)} rows x {args.epochs_long} epochs")
    trainer = SFTTrainer(
        model=trainer.model,
        args=SFTConfig(num_train_epochs=args.epochs_long, **common),
        train_dataset=ds_long,
        peft_config=lora,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.out_dir))
    print(f"[s07.C] saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
