"""RLVR reward functions for Ignite-3B.

All rewards deterministic (no LLM-judge). Return float in [-1, 1].
Shafayat-proof (2505.21444): pure exogenous verifiable reward.

Closures shape: reward_fn(pred_text: str, gold: dict) -> float
where gold has bench-specific structure.
"""

from eval.ignite.tools.code_exec import code_reward
from eval.ignite.tools.math_verify import verify as math_verify

FORMAT_BONUS = 0.1  # small bonus for well-formed \\boxed{} answer


def math_reward(pred_text: str, gold: str, format_bonus: bool = True) -> float:
    """Math answer reward: 1.0 exact / SymPy-eq, 0.0 wrong, -1.0 malformed.
    Adds +0.1 format bonus if \\boxed{} present (capped at 1.0)."""
    r = math_verify(pred_text, gold)
    if r == -1.0:
        return -1.0
    if format_bonus and "\\boxed{" in pred_text:
        r = min(1.0, r + FORMAT_BONUS)
    return r


def code_task_reward(pred_text: str, tests: list[str], timeout_s: float = 10.0) -> float:
    """Code exec reward: fraction of tests passed.
    Extracts fenced ```python ... ``` block if present, else uses pred_text as-is."""
    code = _extract_python_block(pred_text)
    return code_reward(code, tests, timeout_s)


def lean_reward(pred_text: str, thm_state: dict) -> float:
    """Lean proof reward: 1.0 if kernel accepts, 0.0 otherwise.
    Placeholder — implementation in tools/lean_tool.py (Pantograph)."""
    from eval.ignite.tools.lean_tool import verify_proof

    return verify_proof(pred_text, thm_state)


def _extract_python_block(text: str) -> str:
    """Extract python code from ```python...``` fence, else return text."""
    import re

    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    return text


def cyber_rcm_reward(pred_text: str, gold_cwe: str, hierarchical: bool = True) -> float:
    """RL reward pra CVE->CWE (fase 2 cyber). RLVR: gold verificavel, sem LLM-judge.

    hierarchical=True: acerto exato=1.0, ancestral=0.5, irmao=0.3 (arvore CWE).
    Reward denso faz o GRPO aprender mais rapido que binario 0/1. Se a arvore
    nao carregar (sem internet no kernel), cai pro binario.
    """
    from eval.s07.benches._common import normalize_cwe
    from eval.s07.cwe_tree_parser import get_cwe_tree
    from eval.s07.hier_reward import hier_cwe_reward

    pred = normalize_cwe(pred_text)
    gold = normalize_cwe(gold_cwe) or gold_cwe
    if pred is None:
        return -1.0
    tree = get_cwe_tree() if hierarchical else None
    if tree is None:
        return 1.0 if pred == gold else 0.0
    return hier_cwe_reward(pred_text, gold, tree)


def majority_vote_reward(completions: list[str]) -> list[float]:
    """Self-reward por voto majoritario, sem gold. Usado na Cond D.

    Reproduz o modo de falha de Shafayat (2505.21444): sem verificador externo,
    o sinal so mede concordancia interna, entao o modelo pode convergir pra uma
    resposta errada e ser recompensado por isso. E o contraste da Cond C (RLVR).
    """
    from collections import Counter

    from eval.ignite.tools.math_verify import extract_boxed

    answers = [(extract_boxed(c) or "").strip() for c in completions]
    valid = [a for a in answers if a]
    if not valid:
        return [0.0] * len(completions)
    majority, _ = Counter(valid).most_common(1)[0]
    return [1.0 if a == majority else 0.0 for a in answers]


# bench -> (chave do kwarg com a referencia, funcao de reward por item)
PAIRWISE_REWARDS = {
    "math": ("gold", math_reward),
    "code": ("tests", code_task_reward),
    "lean": ("thm_state", lean_reward),
    "cyber_rcm": ("gold", cyber_rcm_reward),  # fase 2: RL cyber CVE->CWE
}


def build_reward_fn(bench: str, gold_key: str = "gold"):
    """Factory returning closure for GRPO reward_funcs.

    Args:
        bench: "math" | "code" | "lean" | "cyber_rcm" (RLVR, referencia externa)
               ou "llmjudge" (Cond D: voto majoritario, sem referencia)
        gold_key: key in row dict holding gold answer (default "gold")

    Returns: (completions, **kwargs) -> list[float] compatible with TRL GRPOTrainer.
    """
    if bench == "llmjudge":

        def _judge(completions, **kwargs):
            return majority_vote_reward(completions)

        return _judge

    if bench not in PAIRWISE_REWARDS:
        raise ValueError(f"unknown bench: {bench}")

    ref_key, score = PAIRWISE_REWARDS[bench]

    def _paired(completions, **kwargs):
        refs = kwargs.get(ref_key) or kwargs.get(gold_key) or kwargs.get("gold")
        return [score(c, r) for c, r in zip(completions, refs, strict=False)]

    return _paired


if __name__ == "__main__":
    assert math_reward("\\boxed{42}", "42") > 0.9
    assert math_reward("42", "42") == 1.0
    assert math_reward("\\boxed{99}", "42") == 0.0 + FORMAT_BONUS
    assert code_task_reward("```python\ndef f(): return 1\n```", ["assert f() == 1"]) == 1.0
    print("reward smoke OK")
