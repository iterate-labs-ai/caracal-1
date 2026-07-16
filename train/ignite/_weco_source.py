"""Weco CLI training source (Cond B).

Wrapper: `weco run` invokes this. Weco outer proposes edits to this script.
We train one LoRA delta via inner_grpo and print the eval metric weco parses.

Usage: `weco run --source train/ignite/_weco_source.py --eval eval/ignite/_weco_eval.py --metric math_acc`
"""

from pathlib import Path

from .inner_grpo import train_lora

BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
DATASET_TRAIN = Path("data/ignite/omni_math_train.jsonl")
OUT_DIR = Path("/tmp/weco_adapter")
STEPS = 100
LORA_RANK = 32
LR = 1e-6


def main():
    train_lora(
        base_model=BASE_MODEL,
        adapter_in=None,
        dataset_path=DATASET_TRAIN,
        bench="math",
        out_dir=OUT_DIR,
        steps=STEPS,
        lora_rank=LORA_RANK,
        lr=LR,
    )
    print(f"[weco_source] adapter saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
