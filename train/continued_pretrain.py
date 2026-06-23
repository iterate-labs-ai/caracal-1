"""Continued pretrain Caracal Base 3B usando Unsloth + TRL SFTTrainer.

Suporta:
- Resume from HuggingFace Hub revision ou checkpoint local
- SIGTERM handler salva final automatico
- Push HF Hub cada N passos
- W&B tracking compartilhado entre sessoes (resume='allow')
- Compute KERNEL-D integrity check antes de iniciar

Usage:
    python train/continued_pretrain.py \\
        --config train/configs/caracal_base_3b.yaml \\
        --resume-from huggingface://iterate-labs/caracal-base-pretrain@step-5500 \\
        --steps-to-run 5500 \\
        --output ./ckpt-out \\
        --hf-revision-out step-11000
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def parse_resume_from(value: str | None) -> tuple[str | None, str | None]:
    """Parse --resume-from. Returns (repo_id, revision) ou (local_path, None)."""
    if value is None:
        return None, None
    if value.startswith("huggingface://"):
        spec = value[len("huggingface://"):]
        if "@" in spec:
            repo, revision = spec.split("@", 1)
            return repo, revision
        return spec, None
    # Local path
    return value, None


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def install_sigterm_handler(output_dir: Path, hf_repo: str | None, revision: str | None):
    import signal
    from huggingface_hub import HfApi

    def handler(signum, frame):
        logger.warning(f"SIGTERM/SIGINT received ({signum}). Saving final + push HF Hub...")
        try:
            # Trainer should have saved by checkpoint_strategy, mas forca aqui tambem
            if hf_repo and revision:
                api = HfApi()
                api.upload_folder(
                    folder_path=str(output_dir),
                    repo_id=hf_repo,
                    revision=f"{revision}-sigterm",
                    commit_message=f"SIGTERM save · {revision}",
                )
                logger.info(f"Pushed to {hf_repo}@{revision}-sigterm")
        except Exception as e:
            logger.error(f"SIGTERM save failed: {e}")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def check_kernel_integrity():
    """Validar que arquivos KERNEL-D nao foram tamperdos antes de iniciar."""
    root = Path(__file__).parent.parent
    sys.path.insert(0, str(root))
    from harness.stop_pattern_detector import compute_kernel_hashes

    hashes = compute_kernel_hashes(root)
    logger.info(f"KERNEL-D hashes: {json.dumps(hashes, indent=2)}")

    expected_path = root / "harness/kernel/_hashes.json"
    if expected_path.exists():
        expected = json.loads(expected_path.read_text())
        expected_hashes = {k: v for k, v in expected.items() if not k.startswith("_") and v != "TBD"}
        if expected_hashes:
            mismatches = [
                f"{k}: expected {v[:8]}.. got {hashes.get(k, 'MISSING')[:8]}.."
                for k, v in expected_hashes.items()
                if hashes.get(k) != v
            ]
            if mismatches:
                raise RuntimeError(f"KERNEL-D integrity check FAILED:\n" + "\n".join(mismatches))
            logger.info("KERNEL-D integrity OK")


def setup_model_and_tokenizer(config: dict, resume_from: str | None, revision: str | None):
    """Carregar Qwen2.5-Coder-3B-Instruct via Unsloth.

    Se resume_from = HF repo + revision, baixa adapter desse revision.
    Se resume_from = local path, carrega de la.
    Senao, comeca do zero.
    """
    from unsloth import FastLanguageModel  # noqa
    from huggingface_hub import snapshot_download

    base_model = config["model"]["base"]
    max_seq = config["training"]["max_seq_length"]

    # Carrega base sempre (Qwen2.5-Coder-3B)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq,
        dtype=None,  # auto
        load_in_4bit=False,  # bf16
    )

    # Se resume, carrega adapter
    if resume_from:
        if resume_from.startswith("/") or os.path.exists(resume_from):
            adapter_path = resume_from
        else:
            adapter_path = snapshot_download(
                repo_id=resume_from,
                revision=revision,
                allow_patterns=["*.safetensors", "*.json", "*.txt"],
            )
        logger.info(f"Loading adapter from {adapter_path}")
        model.load_adapter(adapter_path)
    else:
        # Criar novo adapter LoRA
        lora_cfg = config["lora"]
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            bias="none",
            use_gradient_checkpointing=True,
        )

    return model, tokenizer


def load_dataset_from_manifest(manifest_path: Path):
    """Stream dataset segundo manifesto."""
    from datasets import load_dataset, concatenate_datasets

    manifest = yaml.safe_load(manifest_path.read_text())
    datasets_list = []

    for source in manifest["sources"]:
        sid = source["id"]
        target_tokens = source["target_tokens"]
        # TODO: cada fonte tem download path/method diferente.
        # Implementacao real precisa custom loader por fonte.
        logger.info(f"Source {sid}: target {target_tokens} tokens (loader TBD)")

    # Por enquanto, usar HF dataset publico como placeholder ate loaders prontos
    logger.warning("Manifest sources not all implemented. Using PrimeVul as placeholder.")
    try:
        ds = load_dataset("secmlr/PrimeVul", split="train", streaming=False)
        return ds
    except Exception as e:
        logger.error(f"Failed to load PrimeVul: {e}. Using dummy dataset.")
        from datasets import Dataset
        return Dataset.from_dict({"text": ["dummy cyber content"] * 100})


def train(args):
    logger.info("=" * 60)
    logger.info("Caracal Base 3B continued pretrain")
    logger.info("=" * 60)

    config = load_config(Path(args.config))
    logger.info(f"Config loaded from {args.config}")

    check_kernel_integrity()

    repo_id, revision = parse_resume_from(args.resume_from)
    if repo_id:
        logger.info(f"Resuming from {repo_id} revision {revision or 'main'}")

    hf_repo_out = config.get("hub", {}).get("repo_id")
    install_sigterm_handler(Path(args.output), hf_repo_out, args.hf_revision_out or "session-end")

    # Load model
    logger.info("Loading model...")
    model, tokenizer = setup_model_and_tokenizer(config, repo_id, revision)

    # Load dataset
    logger.info("Loading dataset...")
    root = Path(__file__).parent.parent
    dataset = load_dataset_from_manifest(root / "data/corpus_manifest.yaml")

    # Setup trainer
    from trl import SFTTrainer
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir=args.output,
        max_steps=args.steps_to_run,
        learning_rate=config["training"]["learning_rate"],
        lr_scheduler_type=config["training"]["lr_scheduler"],
        warmup_ratio=config["training"]["warmup_ratio"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        bf16=config["training"]["bf16"],
        optim=config["training"]["optim"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        save_strategy="steps",
        save_steps=config["hub"]["push_every_n_steps"],
        save_total_limit=3,
        push_to_hub=True,
        hub_model_id=hf_repo_out,
        hub_strategy="every_save",
        report_to="wandb",
        run_name=config["wandb"]["resume_id"],
        logging_steps=10,
    )

    # W&B resume='allow'
    os.environ["WANDB_RESUME"] = "allow"
    os.environ["WANDB_RUN_ID"] = config["wandb"]["resume_id"]
    os.environ["WANDB_PROJECT"] = config["wandb"]["project"]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=config["training"]["max_seq_length"],
        dataset_text_field="text",  # ajustar por dataset
        packing=config["training"]["packing"],
    )

    logger.info("Starting training...")
    trainer.train()

    # Final save + push
    logger.info("Training complete. Saving final...")
    trainer.save_model(args.output)

    if hf_repo_out:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(
            folder_path=args.output,
            repo_id=hf_repo_out,
            revision=args.hf_revision_out or "step-final",
            commit_message=f"final {args.hf_revision_out}",
        )
        logger.info(f"Pushed final to {hf_repo_out}@{args.hf_revision_out}")

    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from", default=None, help="HF revision or local dir")
    parser.add_argument("--steps-to-run", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hf-revision-out", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        train(args)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
