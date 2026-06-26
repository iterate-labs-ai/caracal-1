"""Helpers shared pelos eval scripts (run_probe, run_cybergym, compare_baseline)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"


def pick_dtype():
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def load_model(adapter, base=BASE_MODEL):
    """Carrega base + opcionalmente attach LoRA adapter.

    Tokenizer fallback: pega do adapter se tiver, senao do base.
    Dtype auto-pick: bf16 em Ampere+, fp16 em Turing/T4, fp32 em CPU.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = pick_dtype()

    adapter_dir = Path(adapter) if adapter else None
    has_adapter = adapter_dir is not None and (adapter_dir / "adapter_config.json").exists()
    tokenizer_path = (
        adapter if (adapter_dir and (adapter_dir / "tokenizer.json").exists()) else base
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    base_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype, device_map=device)
    if has_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base_model, adapter)
    else:
        model = base_model
    return model, tokenizer, device
