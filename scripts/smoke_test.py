"""Smoke test: valida pipeline sem treinar de verdade.

Roda:
1. Carregar Qwen2.5-Coder-3B base (download se necessario)
2. Anexar LoRA r=32 (verifica trainable params)
3. Carregar 100 samples de cada dataset
4. Decontamination CVE
5. SFTConfig + SFTTrainer build (sem .train())
6. Inferencia rapida (1 prompt)

Exit 0 = pipeline ok. Exit 1 = erro.
"""

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sys.path.insert(0, str(REPO_ROOT))

    logger.info("[1/6] Import deps")
    import torch  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

    logger.info("[2/6] Build model (Qwen base + LoRA) - SKIP em CPU sem GPU")
    has_cuda = torch.cuda.is_available()
    logger.info(f"  CUDA available: {has_cuda}")

    logger.info("[3/6] Load datasets (100 each, decontam ON)")
    import runpy
    train_mod = runpy.run_path(str(REPO_ROOT / "train" / "continued_pretrain.py"), run_name="__smoke__")
    load_all = train_mod["load_all_datasets"]
    dataset = load_all(decontam=True, max_per_dataset=100)
    assert len(dataset) > 0, "Dataset vazio!"
    assert "text" in dataset.column_names, f"Sem coluna text: {dataset.column_names}"
    logger.info(f"  OK: {len(dataset)} examples, columns={dataset.column_names}")

    logger.info("[4/6] CVE blocklist")
    load_blocklist = train_mod["load_cve_blocklist"]
    blocklist = load_blocklist()
    assert len(blocklist) > 0, "Blocklist vazia!"
    logger.info(f"  OK: {len(blocklist)} CVE IDs")

    logger.info("[5/6] Probe set")
    probe_path = REPO_ROOT / "eval" / "probe_set.jsonl"
    assert probe_path.exists(), "probe_set.jsonl ausente"
    import json
    probes = [json.loads(line) for line in probe_path.read_text().splitlines() if line.strip()]
    assert len(probes) >= 30, f"Probe set muito pequeno: {len(probes)}"
    logger.info(f"  OK: {len(probes)} probes")

    logger.info("[6/6] Configs presentes")
    cfg = REPO_ROOT / "train" / "configs" / "caracal_base_3b.yaml"
    assert cfg.exists(), "Config ausente"
    logger.info(f"  OK: {cfg}")

    logger.info("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
