"""Run CyberGym benchmark.

Usage:
    python eval/run_cybergym.py --subset slice-50 --adapter ...
    python eval/run_cybergym.py --subset full-1507 --adapter ... --k 5
"""
from __future__ import annotations
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["slice-50", "subset-200", "full-1507"], required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base", default="iterate-labs/caracal-base-3b")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--out", default="eval/reports/cybergym.json")
    args = parser.parse_args()

    # TODO: implement (uses sunblaze-ucb/cybergym)
    print(f"STUB · would run {args.subset} with adapter {args.adapter}")


if __name__ == "__main__":
    main()
