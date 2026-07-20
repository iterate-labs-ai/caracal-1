"""Shared inference + parsing utilities for s07 bench suite."""

import os
import re

import numpy as np

# Aceita CWE-119, CWE119, "CWE 119" e "CWE: 119". Sem o separador opcional
# o modelo escrevendo "CWE 119" era contado como erro mesmo estando certo.
CWE_RE = re.compile(r"CWE[\s:\-]*(\d{1,4})", re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{([^\}]+)\}")
LETTER_RE = re.compile(r"\b([ABCD])\b")


def normalize_cwe(text: str) -> str | None:
    m = BOXED_RE.search(text)
    if m:
        inner = CWE_RE.search(m.group(1))
        if inner:
            return f"CWE-{int(inner.group(1))}"
    for line in reversed(text.splitlines()):
        m = CWE_RE.search(line)
        if m:
            return f"CWE-{int(m.group(1))}"
    return None


def normalize_mcq_letter(text: str) -> str | None:
    boxed = BOXED_RE.search(text)
    if boxed:
        m = LETTER_RE.search(boxed.group(1))
        if m:
            return m.group(1).upper()
    for line in reversed(text.splitlines()):
        m = LETTER_RE.search(line)
        if m:
            return m.group(1).upper()
    return None


def normalize_mcq_letters(text: str) -> set[str]:
    """Todas as letras marcadas. Benches multi-resposta (CyberSOCEval) tem gold
    tipo ['A','B'], que um parser de letra unica nao consegue pontuar."""
    boxed = BOXED_RE.search(text)
    scope = boxed.group(1) if boxed else ""
    if not scope:
        for line in reversed(text.splitlines()):
            if LETTER_RE.search(line):
                scope = line
                break
    return {m.group(1).upper() for m in LETTER_RE.finditer(scope)}


# TPU/XLA recompila o grafo a cada shape nova. Com padding pra um bucket fixo
# e max_new fixo, o numero de shapes distintos fica pequeno e a compilacao
# amortiza. Ligado por env var pra nao penalizar a GPU com padding inutil.
XLA_MODE = os.environ.get("IGNITE_XLA", "") == "1"
PROMPT_BUCKETS = (512, 1024, 2048, 4096)


def _bucket_len(n: int) -> int:
    for b in PROMPT_BUCKETS:
        if n <= b:
            return b
    return PROMPT_BUCKETS[-1]


def generate(model, tokenizer, prompt: str, max_new: int = 256) -> str:
    import torch

    if not XLA_MODE:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw = tokenizer(prompt, return_tensors="pt")
    pad_to = _bucket_len(raw["input_ids"].shape[1])
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=pad_to,
        truncation=True,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            min_new_tokens=max_new,  # shape de saida constante: sem early stop
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][pad_to:], skip_special_tokens=False)


def read_jsonl(path) -> list[dict]:
    """Le JSONL ignorando linhas em branco. Estava copiado em 10 arquivos."""
    import json
    from pathlib import Path

    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def bootstrap_ci(
    scores: list[int], ci: float = 0.95, n_resamples: int = 10_000
) -> tuple[float, float]:
    arr = np.array(scores)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    # Vetorizado: uma matriz (n_resamples, n) de indices em vez de 10k iteracoes
    # em Python. Mesmo estimador, ~14x mais rapido.
    boots = arr[np.random.randint(0, n, (n_resamples, n))].mean(axis=1)
    alpha = 1 - ci
    return (
        float(np.percentile(boots, 100 * alpha / 2)),
        float(np.percentile(boots, 100 * (1 - alpha / 2))),
    )


def mcq_chat_prompt(
    tokenizer, system: str, question: str, choices: dict[str, str], multi: bool = False
) -> str:
    formatted = "\n".join(f"{k}. {v}" for k, v in choices.items())
    instr = (
        "Multiple answers may be correct. List every correct letter inside \\boxed{}, e.g. \\boxed{A, C}."
        if multi
        else "Answer with A, B, C, or D inside \\boxed{}."
    )
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{question}\n\n{formatted}\n\n{instr}"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
