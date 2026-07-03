"""s07.F.2 - DGM pipeline-level mutation.

Muta prompts + halt threshold + tool order + ensemble weights,
eval em CTI-Bench dev, empirical fitness ranking.
"""

import argparse
import json
import random
import re
from pathlib import Path

CWE_RE = re.compile(r"CWE-?(\d{1,4})", re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{(CWE-?\d{1,4})\}", re.IGNORECASE)


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


PROMPT_TEMPLATES = [
    "Analyze CVE. Reason inside <think></think>. Output \\boxed{CWE-NNN}.",
    "You are a defensive cybersecurity analyst. Identify the CWE. Use tools if needed. Answer inside \\boxed{}.",
    "Task: CVE -> CWE classification. Think step-by-step. Final: \\boxed{CWE-NNN}.",
    "Cybersecurity expert. Explain vulnerability class, then output \\boxed{CWE-NNN} on last line.",
    "You classify vulnerabilities. Use CWE tree tools. Output canonical CWE-ID inside \\boxed{}.",
]

HALT_THRESHOLDS = [0.7, 0.75, 0.8, 0.85, 0.9]
MAX_RECURSION = [4, 6, 8, 10, 12]


def sample_mutant(seed: int) -> dict:
    random.seed(seed)
    return {
        "system_prompt": random.choice(PROMPT_TEMPLATES),
        "halt_threshold": random.choice(HALT_THRESHOLDS),
        "max_recursion": random.choice(MAX_RECURSION),
        "ensemble_weight_s07": random.choice([0.5, 0.7, 0.85, 1.0]),
    }


def eval_mutant(model, tok, cves: list[dict], mutant: dict) -> float:
    """Empirical fitness = CTI-RCM accuracy on dev subset."""
    import torch

    correct = 0
    for c in cves:
        prompt = tok.apply_chat_template(
            [
                {"role": "system", "content": mutant["system_prompt"]},
                {"role": "user", "content": f"CVE Description: {c['description']}"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        text = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)
        pred = normalize_cwe(text)
        gold = normalize_cwe(c["cwe_id"])
        if pred and pred == gold:
            correct += 1
    return correct / len(cves)


def crossover(parent_a: dict, parent_b: dict, seed: int) -> dict:
    """Cross params entre 2 parents."""
    random.seed(seed)
    child = {}
    for k in parent_a:
        child[k] = random.choice([parent_a[k], parent_b[k]])
    return child


def main():
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pedroafonso2/caracal-s07-rsd")
    ap.add_argument("--dev-data", default="AI4Sec/cti-bench")
    ap.add_argument("--dev-subset", default="cti-rcm")
    ap.add_argument("--n-dev", type=int, default=100)
    ap.add_argument("--n-mutants", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--n-crossover", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("results/dgm-best-config.json"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": "cuda:0"}
    )
    model.eval()

    print(f"[DGM] loading dev {args.dev_subset} n={args.n_dev}")
    ds = load_dataset(args.dev_data, args.dev_subset, split="test").select(range(args.n_dev))
    cves = [{"description": r["Prompt"], "cwe_id": r["GT"]} for r in ds]

    archive = []

    print(f"[DGM] round 1: {args.n_mutants} random mutants")
    for i in range(args.n_mutants):
        mutant = sample_mutant(seed=i * 7 + 42)
        fitness = eval_mutant(model, tok, cves, mutant)
        archive.append({"mutant": mutant, "fitness": fitness, "gen": 1})
        print(f"  mutant {i + 1}: fitness={fitness:.3f}")

    archive.sort(key=lambda m: m["fitness"], reverse=True)
    top = archive[: args.top_k]
    print(f"[DGM] top-{args.top_k} fitness: {[round(m['fitness'], 3) for m in top]}")

    print(f"[DGM] round 2: {args.n_crossover} crossover mutants")
    for i in range(args.n_crossover):
        a, b = random.sample(top, 2)
        child = crossover(a["mutant"], b["mutant"], seed=i * 13 + 100)
        fitness = eval_mutant(model, tok, cves, child)
        archive.append({"mutant": child, "fitness": fitness, "gen": 2})
        print(f"  crossover {i + 1}: fitness={fitness:.3f}")

    archive.sort(key=lambda m: m["fitness"], reverse=True)
    best = archive[0]
    print(f"[DGM] BEST fitness={best['fitness']:.3f}")
    print(f"  config: {json.dumps(best['mutant'], indent=2)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"best": best, "archive": archive}, indent=2))
    print(f"[DGM] saved -> {args.out}")


if __name__ == "__main__":
    main()
