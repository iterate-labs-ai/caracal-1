"""Continued pretrain Caracal Base 3B usando Unsloth + TRL SFTTrainer.

So Kaggle: sem HuggingFace, sem Weights and Biases, sem Secrets.
Pesos vivem como Kaggle Datasets publicos.

Usage (dentro do notebook Kaggle):

    python train/continued_pretrain.py \
        --resume-from ./ckpt-in \
        --steps-to-run 4000 \
        --output ./ckpt-out
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def train(args):
    from unsloth import FastLanguageModel
    from datasets import load_dataset, concatenate_datasets
    from trl import SFTTrainer
    from transformers import TrainingArguments

    MAX_SEQ = args.max_seq_length

    # Base ou checkpoint anterior
    if args.resume_from and Path(args.resume_from).exists():
        base_path = args.resume_from
        logger.info(f"Resume from {base_path}")
    else:
        base_path = "Qwen/Qwen2.5-Coder-3B-Instruct"
        logger.info(f"Comecando do base {base_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_path,
        max_seq_length=MAX_SEQ,
        dtype=None,
        load_in_4bit=False,
    )

    # Se for treino do zero (sem resume), cria adapter LoRA novo
    if not args.resume_from:
        model = FastLanguageModel.get_peft_model(
            model, r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none", use_gradient_checkpointing=True,
        )

    # 3 datasets HF publicos · sem auth
    logger.info("Carregando datasets...")
    parts = []
    for name in ["secmlr/PrimeVul", "bstee615/bigvul", "bstee615/diversevul"]:
        try:
            ds = load_dataset(name, split="train")
            parts.append(ds)
            logger.info(f"  {name}: {len(ds)} exemplos")
        except Exception as e:
            logger.warning(f"  {name} falhou: {e}")
    dataset = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    logger.info(f"Total {len(dataset)} exemplos")

    # Escolher campo de texto (PrimeVul/BigVul/DiverseVul usam diferente)
    sample = dataset[0]
    text_field = next((f for f in ["func", "code", "text", "func_before"] if f in sample), None)
    if not text_field:
        text_field = list(sample.keys())[0]
    logger.info(f"Usando campo: {text_field}")

    training_args = TrainingArguments(
        output_dir=args.output,
        max_steps=args.steps_to_run,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True,
        optim="adamw_8bit",
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
        push_to_hub=False,
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=MAX_SEQ,
        dataset_text_field=text_field,
        packing=True,
    )

    logger.info("Treinando...")
    trainer.train()

    logger.info("Salvando final...")
    trainer.save_model(args.output)
    logger.info(f"Done · {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-from", default=None,
                        help="Local path com checkpoint anterior (./ckpt-in tipicamente)")
    parser.add_argument("--steps-to-run", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        train(args)
    except Exception as e:
        logger.exception(f"Treino falhou: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
