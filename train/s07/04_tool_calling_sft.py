"""s07.D - Tool calling SFT via xLAM-style synthetic trajectories + Hammer function masking.

Vitor session 4/7. ~12h Kaggle T4 x2.

Recipe:
- Gera 60K trajectories: (CVE -> think -> tool_call -> observation -> answer)
- xLAM JSON schema, Hammer function masking (10% renamed tools)
- LoRA r=8 alpha=16 on attention + MLP
"""

import argparse
import json
import random
from pathlib import Path

TOOL_SCHEMA = [
    {
        "name": "cwe_lookup",
        "description": "Get CWE description and hierarchy",
        "params": ["cwe_id"],
    },
    {"name": "cwe_tree_path", "description": "Get ancestors chain of a CWE", "params": ["cwe_id"]},
    {"name": "cwe_search_kw", "description": "Search CWE by keyword", "params": ["text"]},
    {
        "name": "cwe_micro_rubric",
        "description": "Get disambiguation rule for CWE",
        "params": ["cwe_id"],
    },
    {
        "name": "attack_lookup",
        "description": "MITRE ATT&CK technique lookup",
        "params": ["technique_id"],
    },
    {"name": "cvss_calc", "description": "Parse CVSS 3.1 vector", "params": ["vector"]},
    {"name": "cve_similar", "description": "Find similar past CVEs", "params": ["description"]},
    {
        "name": "verify_fit",
        "description": "Check if CWE fits CVE",
        "params": ["cve_description", "cwe_id"],
    },
    {
        "name": "cwe_view_filter",
        "description": "Check CWE in named view",
        "params": ["cwe_id", "view"],
    },
    {"name": "nvd_fetch", "description": "Fetch official NVD entry", "params": ["cve_id"]},
]


TOOL_ARG_BUILDERS = {
    "cwe_lookup": lambda row: {"cwe_id": row["cwe_id"]},
    "cwe_tree_path": lambda row: {"cwe_id": row["cwe_id"]},
    "cwe_micro_rubric": lambda row: {"cwe_id": row["cwe_id"]},
    "cve_similar": lambda row: {"description": row["description"][:200]},
    "verify_fit": lambda row: {
        "cve_description": row["description"][:200],
        "cwe_id": row["cwe_id"],
    },
}


def build_trajectory(row: dict, use_random_wrong_name: bool = False) -> str:
    """Build synthetic trajectory: think + 1-2 tool calls + answer."""
    cwe = row["cwe_id"]
    tool_name = random.choice(list(TOOL_ARG_BUILDERS.keys()))
    if use_random_wrong_name:
        tool_name = random.choice(["cwe_lookup_v2", "get_cwe", "find_cwe"])

    canonical = tool_name if tool_name in TOOL_ARG_BUILDERS else "cwe_lookup"
    args = TOOL_ARG_BUILDERS[canonical](row)

    obs = {"cwe_id": cwe, "in_view_1003": True, "confidence": 0.85}

    trajectory = (
        f"<think>Analyzing CVE. Need to check {cwe} hierarchy.</think>\n"
        f"<tool_call>{json.dumps({'name': tool_name, 'arguments': args})}</tool_call>\n"
        f"<observation>{json.dumps(obs)}</observation>\n"
        f"<verify>Confirmed via tool.</verify>\n"
        f"\\boxed{{{cwe}}}"
    )
    return trajectory


def format_example(row: dict, tokenizer, hammer_mask_rate: float = 0.1) -> dict:
    use_wrong = random.random() < hammer_mask_rate
    trajectory = build_trajectory(row, use_random_wrong_name=use_wrong)

    tools_desc = "\n".join(
        f"- {t['name']}({', '.join(t['params'])}): {t['description']}" for t in TOOL_SCHEMA
    )
    system = f"You are a cybersecurity CWE classifier. Available tools:\n{tools_desc}"
    user = f"CVE Description: {row['description']}\nIdentify the CWE."

    if use_wrong:
        assistant = (
            trajectory
            + f"\n<verify>Correction: tool name should be cwe_lookup.</verify>\n\\boxed{{{row['cwe_id']}}}"
        )
    else:
        assistant = trajectory

    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        tokenize=False,
    )
    return {"text": text}


def main():
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="pedroafonso2/caracal-s07-reasoning")
    ap.add_argument("--data", default="xamxte/cve-to-cwe")
    ap.add_argument("--out-dir", type=Path, default=Path("/kaggle/working/caracal-s07-tools"))
    ap.add_argument("--n-samples", type=int, default=60_000)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map={"": "cuda:0"}
    )

    ds = load_dataset(args.data, split="train").select(range(args.n_samples))
    ds = ds.map(lambda r: format_example(r, tok))

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
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

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(args.out_dir),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            num_train_epochs=args.epochs,
            learning_rate=5e-5,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            bf16=True,
            logging_steps=25,
            save_strategy="epoch",
            max_seq_length=2048,
        ),
        train_dataset=ds,
        peft_config=lora,
        tokenizer=tok,
    )
    trainer.train()
    trainer.save_model(str(args.out_dir))
    print(f"[s07.D] saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
