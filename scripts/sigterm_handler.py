"""SIGTERM handler.

Kaggle/Colab matam sessao apos 9h ou 12h. Salva checkpoint final automatico.
Push pra HuggingFace Hub.
"""
from __future__ import annotations
import os
import signal
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_sigterm_handler(
    save_fn,
    output_dir: Path,
    hf_repo: str | None = None,
    final_revision: str = "session-end",
) -> None:
    """Install SIGTERM handler that saves and pushes."""

    def handler(signum, frame):
        logger.warning(f"SIGTERM received ({signum}). Saving final checkpoint...")
        try:
            save_fn(output_dir)
            logger.info(f"Saved to {output_dir}")

            if hf_repo:
                from huggingface_hub import HfApi
                api = HfApi()
                api.upload_folder(
                    folder_path=str(output_dir),
                    repo_id=hf_repo,
                    revision=final_revision,
                    commit_message=f"SIGTERM final save · {final_revision}",
                )
                logger.info(f"Pushed to {hf_repo}@{final_revision}")
        except Exception as e:
            logger.error(f"Failed to save on SIGTERM: {e}")

        sys.exit(0)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
