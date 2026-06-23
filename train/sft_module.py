"""SFT estagio 2 para um modulo especifico.

Cada modulo (Recon, Hypothesizer, Crafter, Validator, Patcher) treina LoRA
sobre Caracal Base usando dataset proprio.

Usage:
    python train/sft_module.py --module recon --config modules/recon/config.yaml
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True, choices=["recon", "hypothesizer", "crafter", "validator", "patcher"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--output", default="./ckpt-out")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    logger.info(f"Training module: {args.module}")
    logger.info(f"Config: {config}")

    # TODO: implement (C20-C24 tasks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
