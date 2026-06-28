"""Orchestrator - roda probe + mmlu_security + humaneval reaproveitando run() dos standalones.

Modelo carregado uma vez via load_model() (caro), reutilizado entre evals.
Cada eval gera JSON proprio em out_dir. No fim, agrega summary.json.

Usage:
    python eval/run_bench_all.py --adapter ./ckpt --out-dir ./bench-s01
    python eval/run_bench_all.py --out-dir ./bench-base   # base puro (sem --adapter)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval._common import load_model  # noqa: E402
from eval.run_cti_bench import run as run_cti  # noqa: E402
from eval.run_cybermetric import run as run_cybermetric  # noqa: E402
from eval.run_humaneval import run as run_humaneval  # noqa: E402
from eval.run_mmlu_security import run as run_mmlu  # noqa: E402
from eval.run_probe import run as run_probe  # noqa: E402
from eval.run_secqa import run as run_secqa  # noqa: E402

logger = logging.getLogger(__name__)

EVALS = {
    "probe": lambda m, t, d, o, args: run_probe(m, t, d),
    "mmlu_security": lambda m, t, d, o, args: run_mmlu(m, t, d),
    "secqa": lambda m, t, d, o, args: run_secqa(m, t, d),
    "cybermetric": lambda m, t, d, o, args: run_cybermetric(m, t, d, tier=args.cybermetric_tier),
    "cti_bench": lambda m, t, d, o, args: run_cti(m, t, d, max_n=args.cti_max_n),
    "humaneval": lambda m, t, d, o, args: run_humaneval(m, t, d, limit=args.humaneval_limit),
}


def _step(name, fn, summary, out_dir):
    t0 = time.time()
    logger.info(f"=== {name} START ===")
    try:
        metrics = fn()
        (out_dir / f"{name}.json").write_text(json.dumps(metrics, indent=2))
        summary[name] = {
            "status": "ok",
            "elapsed_s": round(time.time() - t0, 1),
            "metrics": metrics,
        }
        logger.info(f"=== {name} OK ({summary[name]['elapsed_s']:.0f}s) ===")
    except Exception as e:
        summary[name] = {
            "status": "fail",
            "elapsed_s": round(time.time() - t0, 1),
            "error": str(e),
        }
        logger.exception(f"=== {name} FAIL: {e} ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir (default base puro)")
    parser.add_argument(
        "--no-adapter", action="store_true", help="compat noop (default ja eh base)"
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--skip-mmlu", action="store_true")
    parser.add_argument("--skip-secqa", action="store_true")
    parser.add_argument("--skip-cybermetric", action="store_true")
    parser.add_argument("--skip-cti-bench", action="store_true")
    parser.add_argument("--skip-humaneval", action="store_true")
    parser.add_argument("--humaneval-limit", type=int, default=None)
    parser.add_argument(
        "--cybermetric-tier", type=int, default=2000, choices=[80, 500, 2000, 10000]
    )
    parser.add_argument("--cti-max-n", type=int, default=None, help="Limit n per CTI subset")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_load = time.time()
    model, tokenizer, device = load_model(args.adapter)
    logger.info(
        f"loaded model device={device} adapter={args.adapter} in {time.time() - t_load:.0f}s"
    )

    summary = {"adapter": args.adapter, "device": str(device), "started_unix": int(time.time())}
    skips = {
        "probe": args.skip_probe,
        "mmlu_security": args.skip_mmlu,
        "secqa": args.skip_secqa,
        "cybermetric": args.skip_cybermetric,
        "cti_bench": args.skip_cti_bench,
        "humaneval": args.skip_humaneval,
    }
    for name, eval_fn in EVALS.items():
        if skips[name]:
            continue
        _step(name, lambda f=eval_fn: f(model, tokenizer, device, out_dir, args), summary, out_dir)

    summary["finished_unix"] = int(time.time())
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"DONE -> {out_dir}/summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
