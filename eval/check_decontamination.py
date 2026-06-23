"""Pipeline de decontaminacao 3 camadas.

Filtra corpus de treino contra eval pools:
- CyberGym 1507
- Held-out 30 CVE 2026

Tres camadas:
1. SHA256 exato
2. Jaccard 8-gram > 0.3
3. Cosseno embedding > 0.85

CVE-ID dedup separado.

Usage:
    python eval/check_decontamination.py --strict
    python eval/check_decontamination.py --corpus data/raw/ --report decontam-report.json
"""
from __future__ import annotations
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/raw/")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--report", default="data/decontamination/last-report.json")
    args = parser.parse_args()

    # TODO: implement (task C6)
    print("STUB · implement C6 task")


if __name__ == "__main__":
    main()
