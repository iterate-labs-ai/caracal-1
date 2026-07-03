"""s07.B - LoopUS post-training conversion.

Wrap middle transformer layers em recurrent block (weight-tied),
adiciona LoRA r=32 para adapt (Retrofitted Recurrence 2511.07384),
adaptive halt via linear head.

Kevin session 2/7. ~12h Kaggle T4 x2.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn


class RecurrentBlock(nn.Module):
    """Weight-tied loop over middle transformer layers com adaptive halt."""

    def __init__(self, layers: nn.ModuleList, max_iter: int = 8, hidden_size: int = 2048):
        super().__init__()
        self.layers = layers
        self.max_iter = max_iter
        self.halt_head = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kw):
        halt_logits = []
        for _ in range(self.max_iter):
            for layer in self.layers:
                out = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **kw,
                )
                hidden_states = out[0] if isinstance(out, tuple) else out
            pooled = hidden_states.mean(dim=1)
            halt_prob = torch.sigmoid(self.halt_head(pooled))
            halt_logits.append(halt_prob)
            if (halt_prob > 0.9).all():
                break
        return hidden_states, halt_logits


def wrap_middle_layers(model, encoder_end: int = 10, decoder_start: int = 22, max_iter: int = 8):
    """Replace layers[encoder_end:decoder_start] com RecurrentBlock."""
    layers = model.model.layers
    hidden = model.config.hidden_size
    middle = nn.ModuleList([layers[i] for i in range(encoder_end, decoder_start)])
    recurrent = RecurrentBlock(middle, max_iter=max_iter, hidden_size=hidden)
    # Nova ModuleList: encoder + recurrent (as 1 "layer") + decoder
    encoder = [layers[i] for i in range(encoder_end)]
    decoder = [layers[i] for i in range(decoder_start, len(layers))]
    new_layers = nn.ModuleList(encoder + [recurrent] + decoder)
    model.model.layers = new_layers
    return model, recurrent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="pedroafonso2/caracal-s07-base-tokens")
    ap.add_argument("--data", default="xamxte/cve-to-cwe")
    ap.add_argument("--out-dir", type=Path, default=Path("/kaggle/working/caracal-s07-loopus"))
    ap.add_argument("--n-samples", type=int, default=10_000)
    ap.add_argument("--max-iter", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()

    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    print(f"[s07.B] loading {args.base}")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map={"": "cuda:0"}
    )

    n_layers = len(model.model.layers)
    encoder_end = n_layers // 3
    decoder_start = 2 * n_layers // 3
    print(
        f"[s07.B] wrap layers[{encoder_end}:{decoder_start}] em RecurrentBlock max_iter={args.max_iter}"
    )
    model, _ = wrap_middle_layers(model, encoder_end, decoder_start, args.max_iter)

    lora = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    ds = load_dataset(args.data, split="train").select(range(args.n_samples))

    def fmt(row):
        text = tok.apply_chat_template(
            [
                {"role": "user", "content": f"CVE Description: {row['description']}"},
                {"role": "assistant", "content": f"\\boxed{{{row['cwe_id']}}}"},
            ],
            tokenize=False,
        )
        return tok(text, truncation=True, max_length=1024, padding="max_length")

    ds = ds.map(fmt, remove_columns=ds.column_names)

    training_args = TrainingArguments(
        output_dir=str(args.out_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=args.epochs,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=25,
        save_strategy="epoch",
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=ds, tokenizer=tok)
    trainer.train()
    trainer.save_model(str(args.out_dir))
    print(f"[s07.B] saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
