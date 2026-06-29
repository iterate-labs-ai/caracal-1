"""Helpers shared pelos eval scripts (run_probe, run_cybergym, compare_baseline)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"


def pick_device():
    """Retorna (torch_device, dtype). TPU > CUDA > CPU; bf16 onde suportado, fp16 em T4, fp32 em CPU."""
    import importlib.util

    import torch

    if importlib.util.find_spec("torch_xla") is not None:
        import torch_xla.core.xla_model as xm

        return xm.xla_device(), torch.bfloat16
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return torch.device("cuda"), (torch.bfloat16 if major >= 8 else torch.float16)
    return torch.device("cpu"), torch.float32


def load_model(adapter, base=BASE_MODEL):
    """Carrega base + opcionalmente attach LoRA adapter. Tokenizer sempre do base.

    low_cpu_mem_usage=True evita pico RAM (T4 Kaggle = 14GB, sem isso 3B fp16
    estoura no transfer host->device).
    """
    import logging
    import os
    import sys

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    log = logging.getLogger(__name__)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = pick_device()
    log.info(f"[load_model] device={device} dtype={dtype} base={base}")
    tokenizer = AutoTokenizer.from_pretrained(base)
    log.info("[load_model] tokenizer loaded, loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    log.info(f"[load_model] base model on {device}, adapter={adapter}")
    if adapter and (Path(adapter) / "adapter_config.json").exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(base_model, adapter)
        log.info("[load_model] adapter attached")
    else:
        model = base_model
    model.eval()
    return model, tokenizer, device
