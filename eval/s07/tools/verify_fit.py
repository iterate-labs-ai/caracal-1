"""Separate verifier - Caracal s05 checks if CWE fits CVE."""

_VERIFIER = None
_TOKENIZER = None


def _load_verifier():
    """Lazy load Caracal s05 pra usar como verifier."""
    global _VERIFIER, _TOKENIZER
    if _VERIFIER is not None:
        return _VERIFIER, _TOKENIZER
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = os.environ.get("VERIFIER_BASE", "Qwen/Qwen2.5-3B-Instruct")
    adapter = os.environ.get("VERIFIER_ADAPTER", "pedroafonso2/caracal-base-3b-s05")

    _TOKENIZER = AutoTokenizer.from_pretrained(base)
    _VERIFIER = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map={"": "cuda:0"}
    )
    if adapter:
        from peft import PeftModel

        _VERIFIER = PeftModel.from_pretrained(_VERIFIER, adapter)
    _VERIFIER.eval()
    return _VERIFIER, _TOKENIZER


def verify_fit(cve_description: str, cwe_id: str) -> dict:
    """Returns {fit: bool, confidence: float}."""
    import torch

    model, tok = _load_verifier()
    prompt = tok.apply_chat_template(
        [
            {"role": "system", "content": "Answer only YES or NO."},
            {
                "role": "user",
                "content": f"Does {cwe_id} correctly describe the underlying weakness of this CVE?\nCVE: {cve_description}\n",
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
    text = (
        tok.decode(out.sequences[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        .strip()
        .upper()
    )
    fit = text.startswith("YES")
    # Confidence via logit of first token
    first_logits = out.scores[0][0]
    probs = first_logits.softmax(dim=-1)
    yes_tok = tok.encode("YES", add_special_tokens=False)[0]
    conf = float(probs[yes_tok])
    return {"cwe_id": cwe_id, "fit": fit, "confidence": round(conf, 3)}
