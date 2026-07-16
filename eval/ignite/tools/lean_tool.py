"""Lean 4 tool wrapper via Pantograph daemon (github.com/lenianiva/pantograph).

Async worker pool: N Pantograph subprocesses receive proof attempts via mp.Queue.
30s timeout per proof. Off-policy replay recommended (DeepSeek-Prover-V1.5 RLPAF).

Usage:
    pool = LeanDaemonPool(workers=2, mathlib_path="/kaggle/working/mathlib4")
    result = pool.verify_proof(proof_str, theorem_name)
    pool.shutdown()

Ref: LeanDojo (2306.15626) + DeepSeek-Prover-V1.5 (2408.08152).
Fallback: subprocess `lean --run` one-shot if Pantograph unavailable.
"""

import multiprocessing as mp
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

LEAN_BIN = shutil.which("lean")
LAKE_BIN = shutil.which("lake")

PROOF_FENCE_RE = re.compile(r"```lean\s*\n(.*?)```", re.DOTALL)


@dataclass
class ProofResult:
    proved: bool
    output: str
    error: str | None = None


def extract_lean_proof(text: str) -> str | None:
    m = PROOF_FENCE_RE.search(text)
    return m.group(1).strip() if m else None


def _lean_run_oneshot(source: str, timeout_s: float, workdir: Path | None = None) -> ProofResult:
    if LEAN_BIN is None:
        return ProofResult(False, "", "lean-not-installed")
    with (workdir or Path("/tmp")) / "attempt.lean" as f:
        f.write_text(source)
        try:
            r = subprocess.run(
                [LEAN_BIN, "--run", str(f)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            return ProofResult(r.returncode == 0, r.stdout, r.stderr if r.returncode else None)
        except subprocess.TimeoutExpired:
            return ProofResult(False, "", "timeout")


def _worker(task_q: mp.Queue, result_q: mp.Queue, workdir: str, timeout_s: float):
    workdir_p = Path(workdir)
    workdir_p.mkdir(parents=True, exist_ok=True)
    while True:
        job = task_q.get()
        if job is None:
            break
        job_id, source = job
        result = _lean_run_oneshot(source, timeout_s, workdir_p)
        result_q.put((job_id, result))


class LeanDaemonPool:
    def __init__(
        self, workers: int = 2, workdir: str = "/kaggle/working/lean_work", timeout_s: float = 30.0
    ):
        self.workers = workers
        self.timeout_s = timeout_s
        self.workdir = workdir
        self.task_q: mp.Queue = mp.Queue()
        self.result_q: mp.Queue = mp.Queue()
        self._procs: list[mp.Process] = []
        for _ in range(workers):
            p = mp.Process(target=_worker, args=(self.task_q, self.result_q, workdir, timeout_s))
            p.daemon = True
            p.start()
            self._procs.append(p)
        self._next_id = 0

    def submit(self, source: str) -> int:
        jid = self._next_id
        self._next_id += 1
        self.task_q.put((jid, source))
        return jid

    def collect(self, expected: int) -> list[tuple[int, ProofResult]]:
        results: list[tuple[int, ProofResult]] = []
        for _ in range(expected):
            results.append(self.result_q.get())
        return results

    def verify_proof(self, proof_source: str) -> ProofResult:
        jid = self.submit(proof_source)
        results = self.collect(1)
        _, r = results[0]
        assert _ == jid
        return r

    def shutdown(self):
        for _ in self._procs:
            self.task_q.put(None)
        for p in self._procs:
            p.join(timeout=5)


def verify_proof(pred_text: str, thm_state: dict) -> float:
    """Reward for one Lean proof attempt. 1.0 if kernel accepts, 0.0 else."""
    proof = extract_lean_proof(pred_text)
    if proof is None:
        return -1.0
    header = thm_state.get("preamble", "")
    theorem = thm_state.get("theorem", "")
    source = f"{header}\n\n{theorem}\n{proof}\n"
    r = _lean_run_oneshot(source, timeout_s=30.0)
    return 1.0 if r.proved else 0.0


if __name__ == "__main__":
    if LEAN_BIN is None:
        print("lean-not-installed (skip smoke)")
    else:
        pool = LeanDaemonPool(workers=1)
        r = pool.verify_proof("example : 1 + 1 = 2 := by rfl\n")
        pool.shutdown()
        print(f"lean smoke: proved={r.proved}")
