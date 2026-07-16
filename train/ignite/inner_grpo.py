"""Inner-loop GRPO trainer wrapper (Unsloth + vLLM colocate).

Trains one LoRA adapter delta over a task dataset with verifiable reward.
Used by C_rsi_outer.py to train each mutation candidate.

Stack:
- Unsloth GRPOTrainer (4-bit LoRA + PagedAdamW, ~70% VRAM cut)
- vLLM colocate sleep/wake mode (prefix cache + chunked prefill)
- DAPO dynamic sampling (drop all-correct/all-wrong groups)
- Per-group advantage norm
- Rollout truncation on low reward

Refs:
- DeepSeek-R1 (2501.12948) - GRPO rule-based reward
- DAPO (2503.14476 UNVERIFIED) - dynamic sampling
- DeepScaleR blog - 8k->24k curriculum
"""

import json
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_reward_fn(bench: str, gold_key: str = "gold"):
    """Delegate to eval.ignite.reward.build_reward_fn."""
    from eval.ignite.reward import build_reward_fn as _build

    return _build(bench, gold_key)


def train_lora(
    base_model: str,
    adapter_in: str | None,
    dataset_path: Path,
    bench: str,
    out_dir: Path,
    steps: int = 150,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lr: float = 1e-6,
    max_prompt_len: int = 1024,
    max_completion_len: int = 2048,
    num_generations: int = 8,
    per_device_batch: int = 1,
    grad_accum: int = 4,
    max_seq_len: int = 4096,
) -> Path:
    """Train one LoRA delta over dataset_path with RLVR reward for `bench`.

    Args:
        base_model: HF ID or local path (e.g. "unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
        adapter_in: optional prior adapter to merge before training (v_k init)
        dataset_path: JSONL with {prompt, gold, ...}
        bench: "math" | "code" | "lean"
        out_dir: where to save LoRA adapter
        steps: GRPO training steps
        num_generations: k rollouts per prompt (GRPO group size)

    Returns:
        Path to saved LoRA adapter dir.
    """
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_len,
        load_in_4bit=True,
        fast_inference=True,
        gpu_memory_utilization=0.5,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
    )

    if adapter_in:
        model.load_adapter(adapter_in, adapter_name="prev")
        model.set_adapter("prev")

    rows = _load_jsonl(dataset_path)
    ds = Dataset.from_list(
        [
            {
                "prompt": r["prompt"],
                "gold": r.get("gold", ""),
                "tests": r.get("tests", []),
            }
            for r in rows
        ]
    )

    reward_fn = build_reward_fn(bench)

    cfg = GRPOConfig(
        output_dir=str(out_dir),
        learning_rate=lr,
        num_train_epochs=1,
        max_steps=steps,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        num_generations=num_generations,
        max_prompt_length=max_prompt_len,
        max_completion_length=max_completion_len,
        beta=0.001,
        logging_steps=5,
        save_steps=steps // 3,
        report_to="none",
        remove_unused_columns=False,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.5,
    )

    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    return out_dir


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    ap.add_argument("--adapter-in", default=None)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--bench", choices=["math", "code", "lean"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=150)
    args = ap.parse_args()

    out = train_lora(
        base_model=args.base,
        adapter_in=args.adapter_in,
        dataset_path=args.dataset,
        bench=args.bench,
        out_dir=args.out,
        steps=args.steps,
    )
    print(f"saved adapter -> {out}")
