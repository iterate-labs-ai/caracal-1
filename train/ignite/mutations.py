"""Mutation proposer: v_k self-proposes N candidate modifications.

Each mutation is a dict of (system_prompt, cot_scaffold, curriculum_bin,
lora_rank, lora_alpha, lr). Sampled by the model itself (same-model RSI)
or via random search fallback if model API not available.

Ref: DGM (2505.22954) sample_mutant + crossover pattern.
"""

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field


@dataclass
class Mutation:
    system_prompt: str
    cot_scaffold: str
    curriculum_bin: str
    lora_rank: int
    lora_alpha: int
    lr: float
    seed: int = 0
    meta: dict = field(default_factory=dict)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


DEFAULT_SYSTEM_MATH = (
    "You are a math olympiad expert. Solve the problem step by step. "
    "Output the final answer inside \\boxed{}."
)
DEFAULT_SYSTEM_CODE = (
    "You are an expert programmer. Solve the coding problem. "
    "Output the code inside a python code fence."
)
DEFAULT_SYSTEM_LEAN = (
    "You are a Lean 4 theorem prover. Prove the theorem using Mathlib tactics. "
    "Output the proof inside a ```lean code fence."
)

COT_SCAFFOLDS = [
    "Think step by step.",
    "Break the problem into subproblems, then combine.",
    "First identify what is known and what is asked. Then plan. Then execute.",
    "Verify each step against constraints before proceeding.",
    "Try multiple approaches; discard failed ones and pick the best.",
]

CURRICULUM_BINS = ["easy", "medium", "hard", "mixed"]

LORA_RANK_GRID = [16, 32, 64]
LORA_ALPHA_GRID = [32, 64, 128]
LR_GRID = [5e-7, 1e-6, 3e-6]


def default_mutation(bench: str) -> Mutation:
    sys_map = {
        "math": DEFAULT_SYSTEM_MATH,
        "code": DEFAULT_SYSTEM_CODE,
        "lean": DEFAULT_SYSTEM_LEAN,
    }
    return Mutation(
        system_prompt=sys_map.get(bench, DEFAULT_SYSTEM_MATH),
        cot_scaffold=COT_SCAFFOLDS[0],
        curriculum_bin="mixed",
        lora_rank=32,
        lora_alpha=64,
        lr=1e-6,
    )


def sample_random(bench: str, seed: int) -> Mutation:
    rng = random.Random(seed)
    base = default_mutation(bench)
    return Mutation(
        system_prompt=base.system_prompt,
        cot_scaffold=rng.choice(COT_SCAFFOLDS),
        curriculum_bin=rng.choice(CURRICULUM_BINS),
        lora_rank=rng.choice(LORA_RANK_GRID),
        lora_alpha=rng.choice(LORA_ALPHA_GRID),
        lr=rng.choice(LR_GRID),
        seed=seed,
    )


PROPOSE_PROMPT = """You are the outer-loop optimizer of your own training pipeline. Propose ONE mutation to improve your next iteration.

Current config:
- system_prompt: {system_prompt!r}
- cot_scaffold: {cot_scaffold!r}
- curriculum_bin: {curriculum_bin}
- lora_rank: {lora_rank}, lora_alpha: {lora_alpha}, lr: {lr}

Recent trace (last generation gain per task):
{recent_stats}

Output ONE JSON object with the mutated fields. Keys: system_prompt, cot_scaffold, curriculum_bin, lora_rank, lora_alpha, lr. Wrap in ```json code fence."""


def parse_proposal(text: str, base: Mutation, seed: int) -> Mutation:
    import re

    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        return sample_random(base.system_prompt[:20], seed)
    try:
        d = json.loads(m.group(1))
    except json.JSONDecodeError:
        return sample_random(base.system_prompt[:20], seed)
    return Mutation(
        system_prompt=str(d.get("system_prompt", base.system_prompt)),
        cot_scaffold=str(d.get("cot_scaffold", base.cot_scaffold)),
        curriculum_bin=str(d.get("curriculum_bin", base.curriculum_bin)),
        lora_rank=int(d.get("lora_rank", base.lora_rank)),
        lora_alpha=int(d.get("lora_alpha", base.lora_alpha)),
        lr=float(d.get("lr", base.lr)),
        seed=seed,
    )


def propose_via_self(
    model, tok, base: Mutation, recent_stats: str, n: int, temp: float = 0.9, seed: int = 0
) -> list[Mutation]:
    """v proposes N mutations to itself. Falls back to random if model errors."""
    from eval.s07.benches._common import generate

    muts: list[Mutation] = []
    for i in range(n):
        prompt = PROPOSE_PROMPT.format(
            system_prompt=base.system_prompt,
            cot_scaffold=base.cot_scaffold,
            curriculum_bin=base.curriculum_bin,
            lora_rank=base.lora_rank,
            lora_alpha=base.lora_alpha,
            lr=base.lr,
            recent_stats=recent_stats,
        )
        chat = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        try:
            resp = generate(model, tok, chat, max_new=512)
            m = parse_proposal(resp, base, seed=seed * 100 + i)
        except (RuntimeError, ValueError, KeyError):
            m = sample_random(base.system_prompt[:20], seed=seed * 100 + i)
        muts.append(m)
    return muts
