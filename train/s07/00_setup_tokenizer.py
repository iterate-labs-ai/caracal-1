"""s07.A - Extend Qwen2.5-3B tokenizer com special tokens + resize embeddings.

Pedro session 1/7. ~12h Kaggle T4 x2.

Usage (Kaggle):
    python train/s07/00_setup_tokenizer.py \\
        --base-model Qwen/Qwen2.5-3B-Instruct \\
        --s05-adapter pedroafonso2/caracal-base-3b-s05 \\
        --out-hf pedroafonso2/caracal-s07-base-tokens
"""

import argparse
from pathlib import Path

SPECIAL_TOKENS = [
    "<think>",
    "</think>",
    "<verify>",
    "</verify>",
    "<tool_call>",
    "</tool_call>",
    "<observation>",
    "</observation>",
    "<halt/>",
]


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--s05-adapter", default="pedroafonso2/caracal-base-3b-s05")
    ap.add_argument("--out-dir", type=Path, default=Path("/kaggle/working/caracal-s07-base"))
    args = ap.parse_args()

    print(f"[s07.A] loading base {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map={"": "cuda:0"}
    )

    print(f"[s07.A] attaching s05 adapter {args.s05_adapter}")
    model = PeftModel.from_pretrained(model, args.s05_adapter)
    model = model.merge_and_unload()  # merge s05 knowledge into base

    print(f"[s07.A] adding {len(SPECIAL_TOKENS)} special tokens")
    n_added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    print(f"[s07.A] {n_added} new tokens added, vocab size {len(tokenizer)}")

    # Resize + mean init pros novos tokens
    model.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        embed = model.get_input_embeddings().weight
        mean_vec = embed[:-n_added].mean(dim=0)
        embed[-n_added:] = mean_vec
        out_embed = model.get_output_embeddings().weight
        out_mean = out_embed[:-n_added].mean(dim=0)
        out_embed[-n_added:] = out_mean

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.out_dir)
    model.save_pretrained(args.out_dir)
    print(f"[s07.A] saved to {args.out_dir}")

    print("[s07.A] smoke test generate")
    prompt = "<cve>Buffer overflow in parser X allows RCE.</cve>\n<think>"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    print(tokenizer.decode(out[0], skip_special_tokens=False))


if __name__ == "__main__":
    main()
