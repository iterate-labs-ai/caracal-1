"""Weco CLI eval hook (Cond B).

Weco runs this after each `_weco_source.py` iteration. Must print a line
like `math_acc: 0.42` for weco to parse metric.
"""

from eval.ignite.benches import BENCH_REGISTRY

BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
ADAPTER = "/tmp/weco_adapter"
N_EVAL = 100


def main():
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=4096, load_in_4bit=True, fast_inference=True
    )
    try:
        model.load_adapter(ADAPTER)
    except (FileNotFoundError, ValueError):
        pass

    res = BENCH_REGISTRY["omni_math"](model, tok, n=N_EVAL)
    acc = float(res.get("accuracy", 0.0))
    print(f"math_acc: {acc:.4f}")


if __name__ == "__main__":
    main()
