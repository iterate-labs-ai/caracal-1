"""Continued pretrain Caracal Base 3B.

Pega Qwen2.5-Coder-3B-Instruct, continua treino com corpus cyber.
Suporta resume from HuggingFace Hub revision.

Usage:
    python train/continued_pretrain.py \
        --config train/configs/caracal_base_3b.yaml \
        --resume-from huggingface://iterate-labs/caracal-base-pretrain@step-XXXX \
        --steps-to-run 5500 \
        --output ./ckpt-out
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from", default=None, help="HF revision or local dir")
    parser.add_argument("--steps-to-run", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wandb-resume", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())

    logger.info(f"Config: {config}")
    logger.info(f"Resume from: {args.resume_from}")
    logger.info(f"Steps to run: {args.steps_to_run}")
    logger.info(f"Output: {args.output}")

    # TODO: implement (C4 task)
    # 1. Load model + tokenizer (or resume from revision)
    # 2. Load dataset corpus_manifest.yaml
    # 3. Setup Unsloth + TRL SFTTrainer
    # 4. Install sigterm_handler
    # 5. Train --steps-to-run steps
    # 6. Save final
    # 7. Push HF Hub if --push set

    logger.warning("STUB · implement C4 task")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
