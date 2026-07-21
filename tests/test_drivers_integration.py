"""Integracao CPU dos drivers de RL (GPU stubada): executa run() e outer_loop()
de verdade e verifica os artefatos (curve.json, trajectory.json, state.json).

E o que garante que o notebook Kaggle nao morre por erro de fiacao depois de
horas de treino: blocos, resume, retencao, schema canonico - tudo exercitado
sem GPU. O treino em si (train_lora) e stubado; a logica dos loops e real.
"""

import json
import types
from pathlib import Path

import pytest


def _fake_adapter_dir(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "adapter_model.safetensors").write_bytes(b"x")
    return out_dir


def test_grpo_driver_blocos_resume(tmp_path, monkeypatch):
    import train.ignite.B_grpo_straight as B

    calls = []

    def fake_train(base_model, adapter_in, dataset_path, bench, out_dir, steps, lora_rank, lr):
        calls.append({"in": adapter_in, "steps": steps})
        return _fake_adapter_dir(out_dir)

    seq = iter([0.44, 0.46, 0.47, 0.49])
    monkeypatch.setattr(B, "train_lora", fake_train)
    monkeypatch.setattr(
        B,
        "bench_adapter",
        lambda base, ad, bn, dp, n: {
            "accuracy": next(seq), "hier_score": 0.5, "unparsed_frac": 0.02, "n": n
        },
    )

    cfg = types.SimpleNamespace(
        base="b", train="t", dev="d", out=str(tmp_path), total_steps=50,
        block=20, lr=1e-6, rank=32, eval_n=150, resume=False,
    )
    curve = B.run(cfg)

    assert set(curve) == {"0", "20", "40", "50"}  # bloco parcial de 10 no fim
    assert calls[0]["in"] is None  # primeiro bloco parte do base
    assert calls[1]["in"].endswith("block-20")  # RL cumulativo
    assert calls[-1]["steps"] == 10

    # resume: treina so o delta, partindo do ultimo adapter
    seq2 = iter([0.50])
    monkeypatch.setattr(
        B,
        "bench_adapter",
        lambda base, ad, bn, dp, n: {
            "accuracy": next(seq2), "hier_score": 0.5, "unparsed_frac": 0.02, "n": n
        },
    )
    calls.clear()
    cfg.total_steps, cfg.resume = 70, True
    curve2 = B.run(cfg)
    assert set(curve2) == {"0", "20", "40", "50", "70"}
    assert len(calls) == 1 and calls[0]["in"].endswith("block-50")


def test_rsi_outer_loop_retencao_trajetoria(tmp_path, monkeypatch):
    import train.ignite.C_rsi_outer as C
    from train.ignite.mutations import default_mutation

    monkeypatch.setattr(C, "load_model_and_tok", lambda base, ad: (object(), object()))
    monkeypatch.setattr(C, "free_gpu", lambda: None)
    monkeypatch.setattr(
        C,
        "train_lora",
        lambda base_model, adapter_in, dataset_path, bench, out_dir, steps,
        lora_rank, lora_alpha, lr: _fake_adapter_dir(out_dir),
    )
    monkeypatch.setattr(
        C,
        "propose_via_self",
        lambda model, tok, base, recent_stats, n, seed=0, temp=0.9, bench="math": [
            default_mutation("cyber_rcm") for _ in range(n)
        ],
    )
    # ref 0.44 | gen0: dev .46/.45 val .47/.46 (retem) | gen1: dev .46/.44 val .47/.465 (rejeita)
    vals = iter([0.44, 0.46, 0.45, 0.47, 0.46, 0.46, 0.44, 0.47, 0.465])
    monkeypatch.setattr(C, "eval_on", lambda m, t, bn, dp, n: next(vals))
    def perf(acc):
        return {"accuracy": acc, "hier_score": acc + 0.1, "unparsed_frac": 0.02, "n": 150}

    monkeypatch.setattr(C, "score", lambda m, t, bn, dp, n: perf(0.44))
    monkeypatch.setattr(C, "perf_of", lambda base, ad, bn, dp, n: perf(0.47))

    C.outer_loop(
        base="b", v0_adapter=None,
        dataset_train=Path("t"), dataset_dev=Path("d"), dataset_val=Path("v"),
        bench="cyber_rcm", bench_name="cyber_rcm",
        gens=2, cands=2, steps=10, out_root=tmp_path,
    )

    traj = json.loads((tmp_path / "trajectory.json").read_text())
    state = json.loads((tmp_path / "state.json").read_text())

    assert [p["gen"] for p in traj] == [-1, 0, 1]
    assert [p["retained"] for p in traj] == [None, True, False]
    assert traj[1]["accuracy"] == 0.47  # retida re-mediu
    assert traj[2]["accuracy"] == 0.47  # rejeitada repetiu last_perf
    canonical = {"accuracy", "hier_score", "unparsed_frac", "n"}
    assert all(canonical <= set(p) for p in traj)
    assert "gen0" in state["v_adapter"] and state["last_gen"] == 1


def test_notebook_final_cell_roda(tmp_path, monkeypatch):
    """Executa a celula do benchmark final do notebook contra artefatos fabricados."""
    pytest.importorskip("numpy")
    import glob as _glob
    import os as _os

    nb = json.loads(Path("notebooks/kaggle/s07/caracal_rl_cyber.ipynb").read_text())
    src = "".join(nb["cells"][7]["source"])
    final_json = tmp_path / "final.json"
    src = src.replace("/kaggle/working/final_benchmark.json", str(final_json))

    # artefatos minimos: curva do GRPO com melhor em block-20 + state do RSI
    grpo, rsi = tmp_path / "g", tmp_path / "r"
    _fake_adapter_dir(grpo / "block-20")
    _fake_adapter_dir(rsi / "gen0" / "cand0")
    (grpo / "curve.json").write_text(json.dumps({"0": {"accuracy": 0.44}, "20": {"accuracy": 0.5}}))
    (rsi / "state.json").write_text(json.dumps({"v_adapter": str(rsi / "gen0" / "cand0")}))

    import eval.s07.benches as sb

    monkeypatch.setattr(
        sb, "BENCH_REGISTRY",
        dict.fromkeys(
            ["cti_bench", "cybermetric", "secqa", "secbench", "mmlu_security", "seceval", "cybersoceval"],
            lambda *a, **k: {"accuracy": 0.5, "n": 10},
        ),
    )
    ns = {
        "os": _os, "json": json, "glob": _glob, "BASE": "b",
        "OUT_GRPO": str(grpo), "OUT_RSI": str(rsi),
        "load_eval_model": lambda base, ad: (object(), object()),
        "free_gpu": lambda: None,
        "has_adapter": lambda d: bool(_glob.glob(d + "/adapter_*.safetensors")),
    }
    exec(src, ns)  # noqa: S102 - o teste existe pra executar a celula real

    res = json.loads(final_json.read_text())
    assert set(res) == {"base", "GRPO", "RSI"}
    assert ns["MODELS"]["GRPO"].endswith("block-20")
    assert not any(str(v).startswith("ERR") for r in res.values() for v in r.values())
