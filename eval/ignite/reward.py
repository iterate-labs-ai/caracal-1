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


def build_reward_fn(bench: str, gold_key: str = "gold"):
    """Factory returning closure for GRPO reward_funcs.

    Args:
        bench: one of "math", "code", "lean"
        gold_key: key in row dict holding gold answer (default "gold")

    Returns: (completions, **kwargs) -> list[float] compatible with TRL GRPOTrainer.
    """
    if bench == "math":

        def _r(completions, **kwargs):
            golds = kwargs.get(gold_key) or kwargs.get("gold")
            return [math_reward(c, g) for c, g in zip(completions, golds, strict=False)]

        return _r
    if bench == "code":

        def _r(completions, **kwargs):
            tests_list = kwargs.get("tests")
            return [code_task_reward(c, t) for c, t in zip(completions, tests_list, strict=False)]

        return _r
    if bench == "lean":

        def _r(completions, **kwargs):
            thms = kwargs.get("thm_state") or kwargs.get("gold")
            return [lean_reward(c, t) for c, t in zip(completions, thms, strict=False)]

        return _r
    raise ValueError(f"unknown bench: {bench}")


if __name__ == "__main__":
    assert math_reward("\\boxed{42}", "42") > 0.9
    assert math_reward("42", "42") == 1.0
    assert math_reward("\\boxed{99}", "42") == 0.0 + FORMAT_BONUS
    assert code_task_reward("```python\ndef f(): return 1\n```", ["assert f() == 1"]) == 1.0
    print("reward smoke OK")
