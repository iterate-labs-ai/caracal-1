"""Helpers shared pelos eval scripts (run_probe, run_cybergym, compare_baseline)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"


def pick_device():
    """Retorna (torch_device, kind). kind in {'xla','cuda','cpu'}.

    torch_xla so existe em ambientes TPU (Kaggle TPU kernel pre-installed).
    Em GPU/CPU o ImportError eh esperado - cai pra CUDA ou CPU.
    """
    import importlib.util

    import torch

    if importlib.util.find_spec("torch_xla") is not None:
        import torch_xla.core.xla_model as xm

        return xm.xla_device(), "xla"
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    return torch.device("cpu"), "cpu"


def pick_dtype(kind):
    import torch

    if kind == "xla":
        return torch.bfloat16
    if kind == "cuda":
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def load_model(adapter, base=BASE_MODEL):
    """Carrega base + opcionalmente attach LoRA adapter.

    Tokenizer fallback: pega do adapter se tiver, senao do base.
    Device auto-pick: TPU (torch_xla) > CUDA > CPU.
    Dtype: bf16 em XLA/Ampere+, fp16 em Turing/T4, fp32 em CPU.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, kind = pick_device()
    dtype = pick_dtype(kind)

    adapter_dir = Path(adapter) if adapter else None
    has_adapter = adapter_dir is not None and (adapter_dir / "adapter_config.json").exists()
    tokenizer = AutoTokenizer.from_pretrained(
        base
    )  # LoRA nao muda vocab; evita quirk extra_special_tokens list/dict

    if kind == "cuda":
        base_model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=dtype, device_map="cuda"
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype).to(device)

    if has_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base_model, adapter)
    else:
        model = base_model
    model.eval()
    return model, tokenizer, device
