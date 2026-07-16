"""Cond B - asymmetric baseline via `weco` CLI (pip install weco).

Uses Weco AIDE² tree-search + Opus-class outer LLM to optimize our train script.
Cond C compares against this to isolate asymmetry contribution.

Requires: pip install weco>=0.3.40 + OPENAI_API_KEY or ANTHROPIC_API_KEY.
"""

import argparse
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("train/ignite/_weco_source.py"))
    ap.add_argument(
        "--eval", dest="eval_script", type=Path, default=Path("eval/ignite/_weco_eval.py")
    )
    ap.add_argument("--metric", default="math_acc")
    ap.add_argument("--goal", default="max")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("results/cond_B_weco.json"))
    args = ap.parse_args()

    if not args.source.exists() or not args.eval_script.exists():
        raise SystemExit(
            f"Missing weco source ({args.source}) or eval ({args.eval_script}). "
            "Create these thin wrappers around inner_grpo.train_lora + eval_on."
        )

    cmd = [
        "weco",
        "run",
        "--source",
        str(args.source),
        "--eval",
        str(args.eval_script),
        "--metric",
        args.metric,
        "--goal",
        args.goal,
        "--steps",
        str(args.steps),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
