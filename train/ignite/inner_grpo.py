"""Inner-loop GRPO trainer (transformers + trl, no Unsloth).

Trains one LoRA adapter delta over a task dataset with verifiable reward.
Used by C_rsi_outer.py to train each mutation candidate.

Stack (T4-compat):
- transformers + peft LoRA r=32 fp16
- trl GRPOTrainer + rule-based reward
- No Unsloth (kernels require SM 8.0+, T4 is SM 7.5)
- No vLLM colocate (Kaggle T4 x2 tight VRAM); use HF generate() rollout

Refs:
- DeepSeek-R1 (2501.12948) - GRPO rule-based reward
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


# Orcamento de geracao por bench. GRPO gera num_generations x max_completion_len
# tokens POR PROMPT a cada step, entao esse numero domina o custo do step.
# Medido 2026-07-20 no cyber_rcm (tokenizer Qwen2.5-Coder-3B, n=700):
#   prompt  -> mediana 104, p90 169, p99 898  (384 cobre 98%)
#   resposta-> o CWE em si tem 5.6 tokens; sobra folga pra justificativa curta
# Usar 512 de completion pra emitir "CWE-79" era pagar ~4x a mais por step.
# Se a truncagem apertar demais, o bench acusa via `unparsed_frac` (o modelo
# nao chega a emitir o CWE) - e a canaria pra reverter esses numeros.
BENCH_GEN_BUDGET = {
    "cyber_rcm": {"max_prompt_len": 384, "max_completion_len": 160},
    "math": {"max_prompt_len": 640, "max_completion_len": 512},
    "code": {"max_prompt_len": 640, "max_completion_len": 512},
    "lean": {"max_prompt_len": 640, "max_completion_len": 512},
}
_GEN_BUDGET_FALLBACK = {"max_prompt_len": 640, "max_completion_len": 512}


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
    max_prompt_len: int | None = None,
    max_completion_len: int | None = None,
    num_generations: int = 4,
    per_device_batch: int = 4,
    grad_accum: int = 4,
) -> Path:
    """Train one LoRA delta over dataset_path with RLVR reward for `bench`.

    Args:
        base_model: HF ID (e.g. "Qwen/Qwen2.5-3B-Instruct")
        adapter_in: optional prior adapter to merge before training (v_k init)
        dataset_path: JSONL with {prompt, gold, ...}
        bench: "math" | "code" | "lean"
        out_dir: where to save LoRA adapter
        steps: GRPO training steps
        num_generations: k rollouts per prompt (GRPO group size)

    Returns:
        Path to saved LoRA adapter dir.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    hf_base = base_model.replace("unsloth/", "Qwen/").replace("-bnb-4bit", "")

    # TRL GRPO exige global train batch (per_device x n_devices) divisivel por
    # num_generations. Roda single-device, entao per_device precisa ser multiplo.
    if per_device_batch % num_generations != 0:
        per_device_batch = num_generations

    tokenizer = AutoTokenizer.from_pretrained(hf_base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_base,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    )

    if adapter_in:
        model = PeftModel.from_pretrained(model, adapter_in, is_trainable=True)
    else:
        peft_cfg = LoraConfig(
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
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

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

    budget = BENCH_GEN_BUDGET.get(bench, _GEN_BUDGET_FALLBACK)
    if max_prompt_len is None:
        max_prompt_len = budget["max_prompt_len"]
    if max_completion_len is None:
        max_completion_len = budget["max_completion_len"]
    # tokens gerados por step = generations x completion x batch efetivo
    gen_tokens = num_generations * max_completion_len * per_device_batch * grad_accum
    print(
        f"[inner] bench={bench} prompt<={max_prompt_len} completion<={max_completion_len} "
        f"=> ~{gen_tokens} tokens gerados/step",
        flush=True,
    )

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
        # beta>0 instancia um ref model inteiro (~6GB) que nao cabe na T4.
        beta=0.0,
        logging_steps=5,
        save_steps=max(1, steps // 3),
        report_to="none",
        remove_unused_columns=False,
        use_vllm=False,
        fp16=True,
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

    # Devolve a VRAM: o outer loop carrega outro 3B logo em seguida pra avaliar.
    import gc

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    return out_dir


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
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
