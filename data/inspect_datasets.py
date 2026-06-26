"""Inspeciona os 3 datasets HF antes do treino · T3.

Usage:
    python data/inspect_datasets.py

Salva resultado em data/dataset_inspection.md
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def inspect():
    from datasets import load_dataset

    datasets_specs = [
        ("PrimeVul", "ussooraj/PrimeVul", "train.jsonl"),
        ("BigVul", "bstee615/bigvul", None),
        ("DiverseVul", "bstee615/diversevul", None),
    ]

    report_lines = ["# Inspecao dos 3 datasets HF\n"]

    for name, hf_id, data_files in datasets_specs:
        logger.info(f"Carregando {name}...")
        try:
            if data_files:
                ds = load_dataset(hf_id, data_files=data_files, split="train")
            else:
                ds = load_dataset(hf_id, split="train")
        except Exception as e:
            report_lines.append(f"## {name} · ERRO\n\n{e}\n")
            continue

        sample = ds[0]
        fields = list(sample.keys())

        n_samples = min(100, len(ds))
        avg_chars = sum(len(str(ds[i].get(fields[0], ""))) for i in range(n_samples)) / n_samples
        est_tokens = int(avg_chars * len(ds) / 4)

        report_lines.append(f"## {name} (`{hf_id}`)\n")
        report_lines.append(f"- Total exemplos: {len(ds):,}")
        report_lines.append(
            f"- Campos: `{', '.join(fields[:8])}`{'...' if len(fields) > 8 else ''}"
        )
        report_lines.append(f"- Tamanho medio (primeiro campo): {avg_chars:.0f} chars")
        report_lines.append(f"- Tokens estimados: ~{est_tokens // 1_000_000}M")
        report_lines.append("- Sample primeiro exemplo (truncado):")
        report_lines.append("  ```")
        first_val = str(sample.get(fields[0], ""))[:300]
        report_lines.append(f"  {first_val}")
        report_lines.append("  ```\n")

    report = "\n".join(report_lines)
    Path("data/dataset_inspection.md").write_text(report)
    logger.info("Salvo em data/dataset_inspection.md")
    print(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    inspect()
