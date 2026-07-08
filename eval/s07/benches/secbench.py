"""SecBench (arxiv 2412.20787) - 44K MCQ + 3K SAQ multi-domain cyber."""

from ._common import bootstrap_ci, generate, mcq_chat_prompt, normalize_mcq_letter


SUBSET_FILE = {"mcq": "data/MCQs_2730.jsonl", "saq": "data/SAQs_270.jsonl"}


def _load(subset: str, n: int) -> list[dict]:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="secbench-hf/SecBench",
        filename=SUBSET_FILE[subset],
        repo_type="dataset",
    )
    ds = load_dataset("json", data_files=path, split="train")
    return list(ds.select(range(min(n, len(ds)))))


def eval_secbench(model, tok, n_mcq: int = 1000, subset: str = "mcq") -> dict:
    rows = _load(subset, n_mcq)
    per, correct = [], []
    system = "You are a cybersecurity expert. Answer the multiple choice question."

    for i, r in enumerate(rows):
        q = r.get("question") or r.get("Question", "")
        choices = {}
        for letter in "ABCD":
            key_upper = letter
            key_lower = letter.lower()
            val = r.get(f"option_{key_lower}") or r.get(f"Option_{key_upper}") or r.get(key_upper)
            if val:
                choices[key_upper] = val
        if not choices:
            continue
        prompt = mcq_chat_prompt(tok, system, q, choices)
        resp = generate(model, tok, prompt, max_new=64)
        pred = normalize_mcq_letter(resp)
        gold = (r.get("answer") or r.get("Answer") or "").strip().upper()
        ok = int(pred == gold)
        correct.append(ok)
        per.append({"i": i, "pred": pred, "gold": gold, "correct": ok})
        if (i + 1) % 100 == 0:
            print(f"[secbench {i + 1}/{len(rows)}] acc={sum(correct) / (i + 1):.3f}")
    ci_lo, ci_hi = bootstrap_ci(correct)
    return {
        "subset": subset,
        "n": len(correct),
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
        "ci_95_low": ci_lo,
        "ci_95_high": ci_hi,
    }
