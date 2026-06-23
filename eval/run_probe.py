"""Run probe set rapido (100 prompts) para ancora de delta.

Usage:
    python eval/run_probe.py \
        --adapter iterate-labs/caracal-1-crafter-lora \
        --base iterate-labs/caracal-base-3b \
        --out eval/reports/probe-step-XXXX.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # TODO: implement
    result = {
        "adapter": args.adapter,
        "base": args.base,
        "n_prompts": 100,
        "pass_at_1": 0.0,
        "pass_at_5": 0.0,
        "stub": True,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
