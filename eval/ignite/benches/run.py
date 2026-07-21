"""Primitivo unico de avaliacao: carrega/mede/libera um adapter num bench.

Antes isto estava copiado em 4 lugares (perf_of, B_grpo_straight._eval, e duas
funcoes no notebook Kaggle) com nomes de chave divergentes (hier vs hier_score)
e loaders diferentes. A dança de free da VRAM - onde mora o vazamento que
estourava a T4 - vivia replicada. Aqui e uma coisa so.

- `score(model, tok, ...)`  -> mede um modelo JA residente (hot path, sem load)
- `bench_adapter(base, adapter, ...)` -> load + score + free (cold path / notebook)

Ambos devolvem o mesmo dict canonico: accuracy / hier_score / unparsed_frac / n.
"""

CANONICAL_KEYS = ("accuracy", "hier_score", "unparsed_frac", "n")


def score(model, tok, bench_name: str, dataset_path, n: int) -> dict:
    """Roda o bench num modelo residente e devolve o dict canonico."""
    from . import BENCH_REGISTRY

    r = BENCH_REGISTRY[bench_name](model, tok, n=n, dataset_path=str(dataset_path))
    return {
        "accuracy": round(r.get("accuracy", 0.0), 4),
        "hier_score": round(r.get("hier_score", 0.0), 4),
        "unparsed_frac": round(r.get("unparsed_frac", 0.0), 4),
        "n": r.get("n", n),
    }


def load_eval_model(base: str, adapter: str | None = None):
    """Loader de avaliacao T4-compat (fp16, sem Unsloth). pad a esquerda pra
    geracao decoder-only nao sair torta."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_base = base.replace("unsloth/", "Qwen/").replace("-bnb-4bit", "")
    tok = AutoTokenizer.from_pretrained(hf_base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        hf_base, torch_dtype=torch.float16, device_map={"": "cuda:0"}, low_cpu_mem_usage=True
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def free_gpu():
    """gc + empty_cache. NAO recebe objetos: `del` num parametro so apaga o
    binding local, o caller continua segurando o modelo - o empty_cache rodava
    com a ref viva e nao liberava (era o vazamento que estourava a T4). O caller
    zera as proprias vars ANTES de chamar."""
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()


def bench_adapter(base: str, adapter, bench_name: str = "cyber_rcm", dataset_path=None, n: int = 150) -> dict:
    """load + score + free. Para quando o modelo nao esta residente."""
    model, tok = load_eval_model(base, str(adapter) if adapter else None)
    try:
        return score(model, tok, bench_name, dataset_path, n)
    finally:
        model = tok = None
        free_gpu()
