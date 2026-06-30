"""Caracal s07.A - Setup tokenizer + base model com special tokens.

Pedro session 1/7. ~12h Kaggle T4 x2.

Stage: L0 (base setup) + L2 prep (tokenizer extension).

Flow:
1. Load Qwen2.5-3B-Instruct base
2. Apply Caracal s05 LoRA (frozen)
3. Extend tokenizer com special tokens:
   <think> </think> <verify> </verify>
   <tool_call> </tool_call> <observation> </observation>
   <halt/> \boxed{}
4. Resize embeddings + smoke test gen
5. Save to HF: pedroafonso2/caracal-s07-base-tokens

Deliverable HF: pedroafonso2/caracal-s07-base-tokens
"""

# TODO Pedro:
# - implement tokenizer extension (transformers.AutoTokenizer.add_special_tokens)
# - resize_token_embeddings com mean init pros novos tokens
# - smoke test generate 10 CVEs com novos tokens, validar parse
# - upload HF dataset

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
    "\\boxed{",
]


def main():
    raise NotImplementedError("s07.A scaffolding - implement after s05 ship")


if __name__ == "__main__":
    main()
