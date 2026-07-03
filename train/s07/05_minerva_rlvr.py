"""s07.E - Minerva RLVR (GRPO + hierarchical CWE reward).

Alexandre session 5/7. ~12h Kaggle T4 x2.

Recipe: TRL GRPOTrainer + hier_reward.composite_reward.
"""

import argparse
import re
from pathlib import Path

CWE_RE = re.compile(r"CWE-?(\d{1,4})", re.IGNORECASE)


def build_reward_fn(cwe_xml_path: str):
    """Closure over CWE tree pra ficar picklable."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from eval.s07.cwe_tree_parser import CWEParser
    from eval.s07.hier_reward import composite_reward

    tree = CWEParser(cwe_xml_path)

    def reward_fn(prompts, completions, gold_cwe_ids, **kwargs):
        rewards = []
        for comp, gold in zip(completions, gold_cwe_ids, strict=False):
            text = comp[0]["content"] if isinstance(comp, list) else comp
            rewards.append(composite_reward(text, gold, tree, tool_args_valid=True))
        return rewards

    return reward_fn


def format_grpo_row(row: dict, tokenizer) -> dict:
    """Reformat xamxte row para GRPO training."""
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "Analyze CVE. Reason inside <think></think>. Output \\boxed{CWE-NNN}.",
            },
            {"role": "user", "content": f"CVE Description: {row['description']}"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return {"prompt": prompt, "gold_cwe_ids": row["cwe_id"]}


def main():
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="pedroafonso2/caracal-s07-tools")
    ap.add_argument("--data", default="xamxte/cve-to-cwe")
    ap.add_argument("--cwe-xml", default="data/cwec_latest.xml")
    ap.add_argument("--out-dir", type=Path, default=Path("/kaggle/working/caracal-s07-rl"))
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--n-samples", type=int, default=2000)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map={"": "cuda:0"}
    )

    ds = load_dataset(args.data, split="train").select(range(args.n_samples))
    ds = ds.map(lambda r: format_grpo_row(r, tokenizer))

    lora = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    grpo_cfg = GRPOConfig(
        output_dir=str(args.out_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,  # T4 constraint
        learning_rate=1e-6,
        beta=0.001,
        epsilon=0.2,
        max_prompt_length=512,
        max_completion_length=256,
        total_training_steps=args.steps,
        bf16=True,
        logging_steps=25,
        save_steps=100,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.5,
    )

    reward_fn = build_reward_fn(args.cwe_xml)

    trainer = GRPOTrainer(
        model=model,
        args=grpo_cfg,
        train_dataset=ds,
        reward_funcs=[reward_fn],
        peft_config=lora,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.out_dir))
    print(f"[s07.E] saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
