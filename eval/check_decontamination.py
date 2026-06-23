"""Pipeline de decontaminacao 3 camadas.

Filtra corpus de treino contra eval pools:
- CyberGym 1507 (carregado de sunblaze-ucb/cybergym)
- Held-out 30 CVE 2026 (eval/held_out_2026.yaml)

Tres camadas:
1. SHA256 hash exato
2. Jaccard 8-gram > 0.3
3. Cosseno embedding > 0.85 (sentence-transformers/all-MiniLM-L6-v2)

CVE-ID dedup separado.

Usage:
    python eval/check_decontamination.py --corpus data/raw/ --report data/decontamination/report.json
    python eval/check_decontamination.py --strict  # exit 1 se qualquer overlap
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Iterator

import yaml

logger = logging.getLogger(__name__)

JACCARD_NGRAM = 8
JACCARD_THRESHOLD = 0.3
COSINE_THRESHOLD = 0.85
MIN_TEXT_LEN = 50  # ignora textos muito curtos


# ============================================================================
# Layer 1: SHA256 hash exato
# ============================================================================

def normalize_text(text: str) -> str:
    """Normaliza pra hash: lowercase, whitespace colapso, remove comentarios."""
    text = re.sub(r"//.*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def sha256_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def check_hash_overlap(corpus_texts: list[str], eval_hashes: set[str]) -> list[int]:
    """Retorna indices dos textos do corpus com hash match exato."""
    overlaps = []
    for i, text in enumerate(corpus_texts):
        if len(text) < MIN_TEXT_LEN:
            continue
        if sha256_hash(text) in eval_hashes:
            overlaps.append(i)
    return overlaps


# ============================================================================
# Layer 2: Jaccard 8-gram
# ============================================================================

def text_to_ngrams(text: str, n: int = JACCARD_NGRAM) -> set[str]:
    """Tokeniza e gera n-grams de palavras."""
    text = normalize_text(text)
    tokens = re.findall(r"\w+", text)
    return set(" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def check_ngram_overlap(
    corpus_texts: list[str], eval_ngrams_list: list[set[str]]
) -> list[tuple[int, int, float]]:
    """Retorna (corpus_idx, eval_idx, jaccard_score) para pares acima do threshold."""
    overlaps = []
    for i, ctext in enumerate(corpus_texts):
        if len(ctext) < MIN_TEXT_LEN:
            continue
        c_ngrams = text_to_ngrams(ctext)
        for j, e_ngrams in enumerate(eval_ngrams_list):
            score = jaccard(c_ngrams, e_ngrams)
            if score >= JACCARD_THRESHOLD:
                overlaps.append((i, j, score))
    return overlaps


# ============================================================================
# Layer 3: Cosseno embedding
# ============================================================================

def compute_embeddings(texts: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Computa embeddings via sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True, batch_size=32, convert_to_numpy=True)


def check_cosine_overlap(corpus_texts: list[str], eval_texts: list[str]) -> list[tuple[int, int, float]]:
    """Embedding cosseno > threshold."""
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    valid_corpus = [(i, t) for i, t in enumerate(corpus_texts) if len(t) >= MIN_TEXT_LEN]
    if not valid_corpus:
        return []

    corpus_indices, corpus_filtered = zip(*valid_corpus)

    logger.info(f"Computing embeddings for {len(corpus_filtered)} corpus + {len(eval_texts)} eval")
    corpus_emb = compute_embeddings(list(corpus_filtered))
    eval_emb = compute_embeddings(eval_texts)

    sim = cosine_similarity(corpus_emb, eval_emb)
    overlaps = []
    for ci, row in enumerate(sim):
        for ei, score in enumerate(row):
            if score >= COSINE_THRESHOLD:
                overlaps.append((corpus_indices[ci], ei, float(score)))
    return overlaps


# ============================================================================
# CVE-ID dedup
# ============================================================================

CVE_PATTERN = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)


def extract_cve_ids(text: str) -> set[str]:
    return set(m.upper() for m in CVE_PATTERN.findall(text))


def check_cve_overlap(corpus_texts: list[str], eval_cve_ids: set[str]) -> list[int]:
    overlaps = []
    for i, text in enumerate(corpus_texts):
        corpus_cves = extract_cve_ids(text)
        if corpus_cves & eval_cve_ids:
            overlaps.append(i)
    return overlaps


# ============================================================================
# Loaders
# ============================================================================

def load_corpus_texts(corpus_path: Path) -> list[str]:
    """Carrega textos do corpus (todos os .txt, .md, .py, .c, .cpp, .json em corpus_path)."""
    if not corpus_path.exists():
        logger.warning(f"Corpus path {corpus_path} nao existe. Usando lista vazia.")
        return []

    texts = []
    extensions = {".txt", ".md", ".py", ".c", ".cpp", ".h", ".json", ".yaml"}
    for f in corpus_path.rglob("*"):
        if f.is_file() and f.suffix in extensions:
            try:
                texts.append(f.read_text(encoding="utf-8", errors="ignore"))
            except Exception as e:
                logger.warning(f"Skip {f}: {e}")
    logger.info(f"Loaded {len(texts)} corpus files from {corpus_path}")
    return texts


def load_eval_texts() -> tuple[list[str], set[str], set[str]]:
    """Carrega CyberGym + held-out 30 CVE 2026.

    Returns (texts, hashes, cve_ids)
    """
    eval_texts = []
    eval_hashes = set()
    eval_cve_ids = set()

    # 1. Held-out 30 CVE 2026
    root = Path(__file__).parent.parent
    heldout_path = root / "eval/held_out_2026.yaml"
    if heldout_path.exists():
        heldout = yaml.safe_load(heldout_path.read_text()) or {}
        for cve in heldout.get("cves", []) or []:
            cve_id = cve.get("cve_id")
            if cve_id:
                eval_cve_ids.add(cve_id.upper())
            # Descricao + PoC + patch texts (quando populado)
            for field in ["description", "poc", "patch_diff"]:
                val = cve.get(field)
                if val:
                    eval_texts.append(val)
                    eval_hashes.add(sha256_hash(val))

    # 2. CyberGym (carrega de HF se possivel)
    try:
        from datasets import load_dataset
        ds = load_dataset("sunblaze-ucb/cybergym", split="train", streaming=False)
        for item in ds:
            for field in ["description", "poc", "patch"]:
                val = item.get(field, "")
                if val and isinstance(val, str):
                    eval_texts.append(val)
                    eval_hashes.add(sha256_hash(val))
            cve_id = item.get("cve_id")
            if cve_id:
                eval_cve_ids.add(cve_id.upper())
    except Exception as e:
        logger.warning(f"Couldnt load CyberGym from HF: {e}. Skipping layer 1 against CyberGym.")

    logger.info(f"Eval set: {len(eval_texts)} texts, {len(eval_hashes)} hashes, {len(eval_cve_ids)} CVE IDs")
    return eval_texts, eval_hashes, eval_cve_ids


# ============================================================================
# Main
# ============================================================================

def run_check(corpus_path: Path, report_path: Path, strict: bool = False) -> int:
    """Run 3-layer check. Return exit code."""
    logger.info("Starting decontamination check")

    corpus_texts = load_corpus_texts(corpus_path)
    if not corpus_texts:
        logger.warning("Empty corpus. Nothing to check.")
        report_path.write_text(json.dumps({"status": "empty_corpus"}, indent=2))
        return 0

    eval_texts, eval_hashes, eval_cve_ids = load_eval_texts()

    # CVE-ID dedup (rapido, faz primeiro)
    cve_overlaps = check_cve_overlap(corpus_texts, eval_cve_ids)
    logger.info(f"CVE-ID dedup: {len(cve_overlaps)} corpus entries with eval CVE IDs")

    # Layer 1: hash exato
    hash_overlaps = check_hash_overlap(corpus_texts, eval_hashes)
    logger.info(f"Layer 1 hash: {len(hash_overlaps)} exact matches")

    # Layer 2: 8-gram Jaccard
    eval_ngrams_list = [text_to_ngrams(t) for t in eval_texts if len(t) >= MIN_TEXT_LEN]
    ngram_overlaps = check_ngram_overlap(corpus_texts, eval_ngrams_list)
    logger.info(f"Layer 2 8-gram: {len(ngram_overlaps)} pairs above {JACCARD_THRESHOLD}")

    # Layer 3: cosseno embedding (so se corpus pequeno o suficiente)
    cosine_overlaps = []
    if len(corpus_texts) <= 10000 and len(eval_texts) <= 5000:
        try:
            cosine_overlaps = check_cosine_overlap(corpus_texts, eval_texts)
            logger.info(f"Layer 3 cosine: {len(cosine_overlaps)} pairs above {COSINE_THRESHOLD}")
        except ImportError:
            logger.warning("sentence-transformers/sklearn nao instalado. Pulando layer 3.")
    else:
        logger.warning("Corpus muito grande pra layer 3 (>10K x >5K). Use sampling.")

    # Report
    report = {
        "corpus_path": str(corpus_path),
        "corpus_size": len(corpus_texts),
        "eval_size": len(eval_texts),
        "cve_id_overlaps": len(cve_overlaps),
        "layer_1_hash_overlaps": len(hash_overlaps),
        "layer_2_ngram_overlaps": len(ngram_overlaps),
        "layer_3_cosine_overlaps": len(cosine_overlaps),
        "thresholds": {
            "jaccard_ngram": JACCARD_NGRAM,
            "jaccard_min": JACCARD_THRESHOLD,
            "cosine_min": COSINE_THRESHOLD,
        },
        "any_overlap": bool(hash_overlaps or ngram_overlaps or cosine_overlaps or cve_overlaps),
        "details": {
            "cve_id_overlap_indices": cve_overlaps[:100],
            "hash_overlap_indices": hash_overlaps[:100],
            "ngram_overlap_examples": ngram_overlaps[:20],
            "cosine_overlap_examples": cosine_overlaps[:20],
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info(f"Report saved to {report_path}")

    if report["any_overlap"]:
        logger.warning(f"OVERLAP DETECTED · check {report_path}")
        if strict:
            return 1
    else:
        logger.info("Decontamination check PASSED · zero overlap")

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/raw/")
    parser.add_argument("--report", default="data/decontamination/last-report.json")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    code = run_check(Path(args.corpus), Path(args.report), strict=args.strict)
    sys.exit(code)


if __name__ == "__main__":
    main()
