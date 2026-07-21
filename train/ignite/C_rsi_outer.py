"""Condition C - MAIN outer loop (same-model RSI).

At generation k, v proposes N=8 mutations, trains 8 LoRA candidates, selects
top-2 by dev, verifies on val, retains if delta > eps AND entropy_ok AND passk_ok.

This is the novel contribution: v_k is BOTH the inner (task solver) AND outer
(mutation proposer). Weight-level recursion via LoRA merge.

Ref plan: docs/research/RSI_PROOF_PLAN.md §RSI outer loop
"""

import argparse
import json
from pathlib import Path

from .archive import Archive
from .inner_grpo import train_lora
from .mutations import default_mutation, propose_via_self

EPS_PP = 0.01
ENTROPY_DROP_MAX = 0.30


def free_gpu():
    """Devolve VRAM ao allocator (gc + empty_cache).

    NAO recebe objetos: `del` num parametro so apaga o binding LOCAL, o caller
    continua segurando o modelo, entao o empty_cache rodava com a referencia
    viva e nao liberava nada (era o vazamento que estourava a T4 no gen tardio).
    O caller precisa zerar as proprias vars (`m = None`) ANTES de chamar aqui.
    """
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()


def load_model_and_tok(base: str, adapter: str | None):
    """T4-compat loader: transformers + fp16, no Unsloth (SM 7.5 unsupported)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_base = base.replace("unsloth/", "Qwen/").replace("-bnb-4bit", "")
    tok = AutoTokenizer.from_pretrained(hf_base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_base,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        tok = AutoTokenizer.from_pretrained(adapter, use_fast=True)
    model.eval()
    return model, tok


def eval_on(model, tok, bench_name: str, dataset_path: Path, n: int) -> float:
    from eval.ignite.benches import BENCH_REGISTRY

    fn = BENCH_REGISTRY[bench_name]
    res = fn(model, tok, n=n, dataset_path=str(dataset_path))
    return float(res.get("accuracy", 0.0))


def perf_of(base: str, adapter, bench_name: str, dataset_path: Path, n: int) -> dict:
    """Performance completa de um adapter (nao so accuracy): acerto exato +
    credito parcial hierarquico + unparsed_frac (canaria de colapso de formato).
    Carrega e libera o modelo. Usado pra medir a cada melhoria do RSI."""
    from eval.ignite.benches import BENCH_REGISTRY

    m, t = load_model_and_tok(base, str(adapter) if adapter else None)
    res = BENCH_REGISTRY[bench_name](m, t, n=n, dataset_path=str(dataset_path))
    m = t = None
    free_gpu()
    return {
        "accuracy": round(res.get("accuracy", 0.0), 4),
        "hier_score": round(res.get("hier_score", 0.0), 4),
        "unparsed_frac": round(res.get("unparsed_frac", 0.0), 4),
        "n": res.get("n", n),
    }


def compute_entropy_delta(v_ckpt: str, cand_ckpt: str) -> float:
    """Placeholder. Full impl requires probing logits on held-out prompts.
    Returns 0.0 == no delta. Wire in S07 notebook when compute allows.
    """
    return 0.0


def passk_check(v_ckpt: str, cand_ckpt: str, k: int = 8) -> bool:
    """Beyond-Pass@1 check: pass@k must not tank while pass@1 grows.
    Placeholder - wire full check in S07.
    """
    return True


def outer_loop(
    base: str,
    v0_adapter: str | None,
    dataset_train: Path,
    dataset_dev: Path,
    dataset_val: Path,
    bench: str,
    bench_name: str,
    gens: int,
    cands: int,
    steps: int,
    out_root: Path,
    hf_repo: str | None = None,
    resume: bool = False,
):
    arch = Archive(out_root)
    state = arch.load_state() if resume else {"last_gen": -1, "last_cand": -1}
    start_gen = state["last_gen"] + 1

    base_mut = default_mutation(bench)

    # Resume retoma do adapter retido; do contrario mede delta contra referencia errada.
    resumed_adapter = state.get("v_adapter") or ""
    if resume and resumed_adapter:
        v_adapter = resumed_adapter
        print(f"[outer] resume gen{start_gen} a partir de {v_adapter}")
    else:
        v_adapter = v0_adapter
        if resume:
            print(f"[outer] resume sem v_adapter no state, comecando de {v_adapter}")

    model, tok = load_model_and_tok(base, v_adapter)
    r_prev = eval_on(model, tok, bench_name, dataset_val, n=100)
    print(f"[outer] referencia val_r={r_prev:.4f} (adapter={v_adapter})")

    # Benchmark a cada melhoria: performance completa do modelo retido por geracao
    # (accuracy + hier + unparsed no dev), gravada num arquivo que o notebook le.
    traj_path = Path(out_root) / "trajectory.json"
    if resume and traj_path.exists():
        trajectory = json.loads(traj_path.read_text())
    else:
        base_perf = perf_of(base, v_adapter, bench_name, dataset_dev, n=150)
        trajectory = [{"gen": -1, "label": "base", "retained": None, **base_perf}]
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        traj_path.write_text(json.dumps(trajectory, indent=2))
        print(f"[outer] base perf {base_perf}")

    for k in range(start_gen, gens):
        print(f"\n=== gen {k}/{gens - 1} ===")

        recent = json.dumps({"prev_val_r": r_prev})
        muts = propose_via_self(model, tok, base_mut, recent, n=cands, seed=k, bench=bench)

        # Proposer sai da VRAM: train_lora carrega o proprio 3B e nao cabem dois em T4.
        model = None  # zera ANTES: senao empty_cache roda com a ref viva e nao libera
        free_gpu()

        cands_out = []
        for ci, m in enumerate(muts):
            print(f"[gen{k} cand{ci}] mut={m.hash()} source={m.source}")
            adapter_dir = out_root / f"gen{k}" / f"cand{ci}"
            if steps <= 0:
                # Cond E (inner congelado): so o scaffold muda. HF Trainer trata
                # max_steps=0 como "nao setado" e treinaria uma epoca inteira.
                out = v_adapter
            else:
                try:
                    out = train_lora(
                        base_model=base,
                        adapter_in=v_adapter,
                        dataset_path=dataset_train,
                        bench=bench,
                        out_dir=adapter_dir,
                        steps=steps,
                        lora_rank=m.lora_rank,
                        lora_alpha=m.lora_alpha,
                        lr=m.lr,
                    )
                except (RuntimeError, ValueError) as e:
                    arch.record_candidate(k, ci, m, dev_r=-1.0, reason=f"train_failed: {e}")
                    continue

            # Eval tambem dentro do try: um OOM aqui derrubava a geracao inteira
            # em vez de so descartar o candidato (foi o que matou o v6).
            try:
                cand_model, cand_tok = load_model_and_tok(base, str(out) if out else None)
                dev_r = eval_on(cand_model, cand_tok, bench_name, dataset_dev, n=50)
                cand_model = cand_tok = None  # zera ANTES do empty_cache
                free_gpu()
            except (RuntimeError, ValueError, OSError) as e:
                # load pode ter deixado um modelo residente antes do eval estourar
                cand_model = cand_tok = None
                free_gpu()
                arch.record_candidate(k, ci, m, dev_r=-1.0, reason=f"eval_failed: {e}")
                continue
            cands_out.append((m, out, dev_r))
            arch.record_candidate(k, ci, m, dev_r=dev_r)

        if not cands_out:
            print(f"[gen{k}] no candidates trained, skip")
            continue

        # Top-2 por dev vao pro val (contamination gate). Um modelo por vez na VRAM.
        top2 = sorted(cands_out, key=lambda x: x[2], reverse=True)[:2]
        scored = []
        for m, adapter, dev_r in top2:
            cand_model, cand_tok = load_model_and_tok(base, str(adapter) if adapter else None)
            v_r = eval_on(cand_model, cand_tok, bench_name, dataset_val, n=100)
            cand_model = cand_tok = None  # zera ANTES do empty_cache
            free_gpu()
            scored.append((m, adapter, dev_r, v_r))

        best_m, best_adapter, best_dev, val_r = max(scored, key=lambda x: x[3])

        entropy_delta = compute_entropy_delta(str(v_adapter or "base"), str(best_adapter))
        passk_ok = passk_check(str(v_adapter or "base"), str(best_adapter))
        delta_pp = val_r - r_prev
        retained = delta_pp > EPS_PP and entropy_delta > -ENTROPY_DROP_MAX and passk_ok

        arch.record_candidate(
            k,
            -1,
            best_m,
            dev_r=best_dev,
            val_r=val_r,
            entropy_delta=entropy_delta,
            passk_ok=passk_ok,
            retained=retained,
            reason=None if retained else f"delta_pp={delta_pp:.4f}",
        )

        if retained:
            v_adapter = str(best_adapter)
            r_prev = val_r
            arch.record_generation(k, v_adapter, val_r, delta_pp)
            if hf_repo:
                arch.push_hf(best_adapter, hf_repo, gen=k)
            print(f"[gen{k}] RETAINED val_r={val_r:.4f} delta={delta_pp:.4f}")
        else:
            print(f"[gen{k}] rejected val_r={val_r:.4f} delta={delta_pp:.4f}")

        # Performance completa do modelo VIGENTE apos esta geracao. So re-mede se
        # reteve (o modelo mudou); senao repete o ultimo ponto sem gastar GPU.
        if retained:
            gen_perf = perf_of(base, v_adapter, bench_name, dataset_dev, n=150)
        else:
            gen_perf = {k2: v2 for k2, v2 in trajectory[-1].items() if k2 not in ("gen", "label", "retained")}
        trajectory.append({"gen": k, "label": f"gen{k}", "retained": retained, **gen_perf})
        traj_path.write_text(json.dumps(trajectory, indent=2))
        print(f"[gen{k}] perf {gen_perf}")

        arch.save_state({"last_gen": k, "last_cand": -1, "v_adapter": str(v_adapter or "")})

        # Proposer volta pra VRAM pra propor a proxima geracao.
        if k + 1 < gens:
            model, tok = load_model_and_tok(base, v_adapter)


def main():
    from eval.ignite.reward import PAIRWISE_REWARDS

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    ap.add_argument("--v0-adapter", default=None)
    ap.add_argument("--dataset-train", type=Path, required=True)
    ap.add_argument("--dataset-dev", type=Path, required=True)
    ap.add_argument("--dataset-val", type=Path, required=True)
    ap.add_argument("--bench", choices=sorted(PAIRWISE_REWARDS), required=True)
    ap.add_argument("--bench-name", default="omni_math", help="registry name for eval")
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--cands", type=int, default=8)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hf-repo", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    outer_loop(
        base=args.base,
        v0_adapter=args.v0_adapter,
        dataset_train=args.dataset_train,
        dataset_dev=args.dataset_dev,
        dataset_val=args.dataset_val,
        bench=args.bench,
        bench_name=args.bench_name,
        gens=args.gens,
        cands=args.cands,
        steps=args.steps,
        out_root=args.out,
        hf_repo=args.hf_repo,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
