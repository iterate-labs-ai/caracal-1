"""RL principal · verl + DAPO GRPO.

Sprint 3+. Roda sobre 5 modulos coordenados via protocolo de mensagens.

Usage:
    python train/rl_grpo.py --config train/configs/rl_dapo_v1.yaml
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--steps-to-run", type=int, default=3500)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    logger.info(f"RL config: {config}")

    # TODO: implement (C26-C28 tasks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
