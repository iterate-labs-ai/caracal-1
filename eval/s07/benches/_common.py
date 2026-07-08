"""Shared inference + parsing utilities for s07 bench suite."""

import re

import numpy as np

CWE_RE = re.compile(r"CWE-?(\d{1,4})", re.IGNORECASE)
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


def generate(model, tokenizer, prompt: str, max_new: int = 256) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)


def bootstrap_ci(
    scores: list[int], ci: float = 0.95, n_resamples: int = 10_000
) -> tuple[float, float]:
    arr = np.array(scores)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    boots = [arr[np.random.randint(0, n, n)].mean() for _ in range(n_resamples)]
    alpha = 1 - ci
    return (
        float(np.percentile(boots, 100 * alpha / 2)),
        float(np.percentile(boots, 100 * (1 - alpha / 2))),
    )


def mcq_chat_prompt(tokenizer, system: str, question: str, choices: dict[str, str]) -> str:
    formatted = "\n".join(f"{k}. {v}" for k, v in choices.items())
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"{question}\n\n{formatted}\n\nAnswer with A, B, C, or D inside \\boxed{{}}.",
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
