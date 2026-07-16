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


def load_model_and_tok(base: str, adapter: str | None):
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        model_name=base, max_seq_length=4096, load_in_4bit=True, fast_inference=True
    )
    if adapter:
        model.load_adapter(adapter)
    return model, tok


def eval_on(model, tok, bench_name: str, dataset_path: Path, n: int) -> float:
    from eval.ignite.benches import BENCH_REGISTRY

    fn = BENCH_REGISTRY[bench_name]
    res = fn(model, tok, n=n, dataset_path=str(dataset_path))
    return float(res.get("accuracy", 0.0))


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

    v_adapter = v0_adapter
    base_mut = default_mutation(bench)

    model, tok = load_model_and_tok(base, v_adapter)
    r_prev = eval_on(model, tok, bench_name, dataset_val, n=100)
    print(f"[outer] v_-1 val_r={r_prev:.4f}")

    for k in range(start_gen, gens):
        print(f"\n=== gen {k}/{gens - 1} ===")

        recent = json.dumps({"prev_val_r": r_prev})
        muts = propose_via_self(model, tok, base_mut, recent, n=cands, seed=k)

        cands_out = []
        for ci, m in enumerate(muts):
            print(f"[gen{k} cand{ci}] mut={m.hash()}")
            adapter_dir = out_root / f"gen{k}" / f"cand{ci}"
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

            cand_model, cand_tok = load_model_and_tok(base, str(out))
            dev_r = eval_on(cand_model, cand_tok, bench_name, dataset_dev, n=50)
            del cand_model, cand_tok
            cands_out.append((m, out, dev_r))
            arch.record_candidate(k, ci, m, dev_r=dev_r)

        if not cands_out:
            print(f"[gen{k}] no candidates trained, skip")
            continue

        top2 = sorted(cands_out, key=lambda x: x[2], reverse=True)[:2]
        best_m, best_adapter, best_dev = max(
            top2,
            key=lambda x: eval_on(
                *load_model_and_tok(base, str(x[1])), bench_name, dataset_val, n=100
            ),
        )
        cand_model, cand_tok = load_model_and_tok(base, str(best_adapter))
        val_r = eval_on(cand_model, cand_tok, bench_name, dataset_val, n=100)
        del cand_model, cand_tok

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
            model, tok = load_model_and_tok(base, v_adapter)
            print(f"[gen{k}] RETAINED val_r={val_r:.4f} delta={delta_pp:.4f}")
        else:
            print(f"[gen{k}] rejected val_r={val_r:.4f} delta={delta_pp:.4f}")

        arch.save_state({"last_gen": k, "last_cand": -1, "v_adapter": str(v_adapter or "")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    ap.add_argument("--v0-adapter", default=None)
    ap.add_argument("--dataset-train", type=Path, required=True)
    ap.add_argument("--dataset-dev", type=Path, required=True)
    ap.add_argument("--dataset-val", type=Path, required=True)
    ap.add_argument("--bench", choices=["math", "code", "lean"], required=True)
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
