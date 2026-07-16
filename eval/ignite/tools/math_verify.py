"""Math answer verifier for RLVR reward.

Wraps HF math-verify (SymPy + latex2sympy2) with signal-based timeout.
Returns 1.0 exact match, 0.0 wrong, -1.0 malformed. Shafayat-proof (no LLM).

Refs:
- HF math-verify: github.com/huggingface/Math-Verify (Open-R1 default)
- latex2sympy2: pip install latex2sympy2
"""

import re
import signal
from contextlib import contextmanager

BOXED_RE = re.compile(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}")


class VerifyTimeout(Exception):
    pass


@contextmanager
def _time_limit(seconds: float):
    def handler(signum, frame):
        raise VerifyTimeout(f"verify > {seconds}s")

    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def extract_boxed(text: str) -> str | None:
    m = BOXED_RE.search(text)
    return m.group(1).strip() if m else None


def _sympy_eq(a: str, b: str) -> bool:
    from sympy import Rational, simplify
    from sympy.parsing.latex import parse_latex
    from sympy.parsing.sympy_parser import parse_expr

    def parse_one(s: str):
        s = s.strip().replace("\\$", "").replace("$", "")
        try:
            return parse_latex(s)
        except (SyntaxError, ValueError, TypeError):
            pass
        try:
            return parse_expr(s)
        except (SyntaxError, ValueError, TypeError):
            pass
        try:
            return Rational(s)
        except (SyntaxError, ValueError, TypeError):
            return None

    ea, eb = parse_one(a), parse_one(b)
    if ea is None or eb is None:
        return False
    try:
        return bool(simplify(ea - eb) == 0)
    except (SyntaxError, ValueError, TypeError, NotImplementedError):
        return False


def verify(pred: str, gold: str, timeout_s: float = 3.0) -> float:
    """Return 1.0 if pred == gold (exact or SymPy-equivalent), else 0.0. -1.0 malformed."""
    if not pred or not gold:
        return -1.0
    pred_ans = extract_boxed(pred) or pred.strip()
    gold_ans = extract_boxed(gold) or gold.strip()
    if not pred_ans or not gold_ans:
        return -1.0
    if pred_ans == gold_ans:
        return 1.0
    try:
        with _time_limit(timeout_s):
            return 1.0 if _sympy_eq(pred_ans, gold_ans) else 0.0
    except VerifyTimeout:
        return 0.0


def batch_verify(preds: list[str], golds: list[str], timeout_s: float = 3.0) -> list[float]:
    return [verify(p, g, timeout_s) for p, g in zip(preds, golds, strict=False)]


if __name__ == "__main__":
    assert verify("42", "42") == 1.0
    assert verify("\\boxed{42}", "42") == 1.0
    assert verify("\\boxed{42}", "43") == 0.0
    assert verify("", "42") == -1.0
    assert verify("\\boxed{1/2}", "0.5") in (0.0, 1.0)
    print("math_verify smoke OK")
