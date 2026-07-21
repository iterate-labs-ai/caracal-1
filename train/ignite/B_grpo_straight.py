"""GRPO direto (sem outer loop RSI) para mover o cyber de 44.4 rumo a 72-75.

Contraste com C_rsi_outer: sem propor/selecionar mutacao, sem candidatos
descartados. Um so caminho de treino, todo o compute vai pra RL. E o controle
que o plano da fase 2 chama de "f2.4: RL sem o loop recursivo" - responde direto
"GRPO move o CVE->CWE?" antes de gastar em busca de mutacao.

Treina em BLOCOS e avalia entre eles: da a curva de aprendizado (endpoint
sozinho nao distingue "nao aprendeu" de "aprendeu e colapsou") e permite retomar
depois de uma desconexao (Colab free corta a sessao em ~4h). Cada bloco parte do
adapter do bloco anterior.
"""

import argparse
import json
from pathlib import Path

from eval.ignite.benches import BENCH_REGISTRY
from train.ignite.inner_grpo import train_lora


def _eval(base: str, adapter: str | None, dev_path: str, n: int) -> dict:
    import gc

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="cuda")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    r = BENCH_REGISTRY["cyber_rcm"](model, tok, n=n, dataset_path=dev_path)
    model = tok = None
    gc.collect()
    torch.cuda.empty_cache()
    return {"accuracy": r["accuracy"], "hier": r["hier_score"], "unparsed": r["unparsed_frac"]}


def run(cfg):
    """cfg: o Namespace do argparse (base/train/dev/out/total_steps/block/lr/rank/eval_n/resume)."""
    out_root = Path(cfg.out)
    out_root.mkdir(parents=True, exist_ok=True)
    curve_path = out_root / "curve.json"

    curve = json.loads(curve_path.read_text()) if (cfg.resume and curve_path.exists()) else {}
    done = max((int(k) for k in curve), default=0)
    adapter = str(out_root / f"block-{done}") if done else None

    if not curve:  # baseline do v0 (base cru) - a referencia do delta
        curve["0"] = _eval(cfg.base, None, cfg.dev, cfg.eval_n)
        curve_path.write_text(json.dumps(curve, indent=2))
        print(f"[grpo] step 0 (base): {curve['0']}", flush=True)

    step = done
    while step < cfg.total_steps:
        n = min(cfg.block, cfg.total_steps - step)
        nxt = step + n
        # cada bloco continua do adapter anterior: RL cumulativo, nao do zero
        adapter = str(
            train_lora(
                base_model=cfg.base,
                adapter_in=adapter,
                dataset_path=Path(cfg.train),
                bench="cyber_rcm",
                out_dir=out_root / f"block-{nxt}",
                steps=n,
                lora_rank=cfg.rank,
                lr=cfg.lr,
            )
        )
        m = _eval(cfg.base, adapter, cfg.dev, cfg.eval_n)
        curve[str(nxt)] = m
        curve_path.write_text(json.dumps(curve, indent=2))
        print(f"[grpo] step {nxt}: {m}", flush=True)
        step = nxt

    print("\n=== CURVA (base=step0 | piso de colapso CWE-79=0.273) ===")
    for s, v in sorted(curve.items(), key=lambda x: int(x[0])):
        print(f"  step {int(s):3d}: acc={v['accuracy']:.3f} hier={v['hier']:.3f} unparsed={v['unparsed']:.2f}")
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--train", default="data/ignite/cyber_rcm_train.jsonl")
    ap.add_argument("--dev", default="data/ignite/cyber_rcm_dev.jsonl")
    ap.add_argument("--out", default="/kaggle/working/grpo_cyber")
    ap.add_argument("--total-steps", type=int, default=120)
    ap.add_argument("--block", type=int, default=20, help="steps por bloco (eval entre blocos)")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--eval-n", type=int, default=150)
    ap.add_argument("--resume", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
