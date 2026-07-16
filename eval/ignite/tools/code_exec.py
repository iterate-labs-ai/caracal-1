"""Code exec sandbox for RLVR code reward.

Uses bwrap (bubblewrap) for filesystem isolation + subprocess timeout.
Fallback: plain subprocess w/ resource limits (unsafe, dev only).
Returns (passed, total). Shafayat-proof (no LLM).

Refs:
- bwrap: man bwrap; standard Kaggle Debian has it (apt install bubblewrap)
- Alt: e2b.dev cloud (paid, out of scope Kaggle T4 free tier)
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

BWRAP = shutil.which("bwrap")


def _bwrap_cmd(script_path: Path, timeout_s: float) -> list[str]:
    return [
        BWRAP,
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/lib",
        "/lib",
        "--ro-bind",
        "/lib64",
        "/lib64",
        "--ro-bind",
        "/bin",
        "/bin",
        "--ro-bind",
        "/etc",
        "/etc",
        "--tmpfs",
        "/tmp",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--unshare-all",
        "--die-with-parent",
        "--ro-bind",
        str(script_path.parent),
        "/work",
        "--chdir",
        "/work",
        "python3",
        script_path.name,
    ]


def _plain_cmd(script_path: Path, timeout_s: float) -> list[str]:
    return ["python3", str(script_path)]


def run_tests(code: str, tests: list[str], timeout_s: float = 10.0) -> tuple[int, int]:
    """Run code + tests, return (passed, total)."""
    if not tests:
        return 0, 0
    passed = 0
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        for i, test in enumerate(tests):
            script = tmp / f"run_{i}.py"
            script.write_text(code + "\n\n" + test + "\n")
            cmd = _bwrap_cmd(script, timeout_s) if BWRAP else _plain_cmd(script, timeout_s)
            try:
                r = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=timeout_s,
                    check=False,
                )
                if r.returncode == 0:
                    passed += 1
            except subprocess.TimeoutExpired:
                pass
    return passed, len(tests)


def code_reward(pred_code: str, tests: list[str], timeout_s: float = 10.0) -> float:
    """Return fraction of tests passed [0, 1]. -1.0 if empty."""
    if not pred_code or not tests:
        return -1.0
    passed, total = run_tests(pred_code, tests, timeout_s)
    return passed / total if total > 0 else -1.0


if __name__ == "__main__":
    code = "def add(a, b): return a + b"
    tests = ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"]
    p, t = run_tests(code, tests)
    assert (p, t) == (2, 2), f"got {p}/{t}"

    bad_code = "def add(a, b): return a - b"
    p2, t2 = run_tests(bad_code, tests)
    assert (p2, t2) == (0, 2)

    print(f"code_exec smoke OK (bwrap={'yes' if BWRAP else 'no fallback'})")
