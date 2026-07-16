"""Ignite-3B bench registry.

L1 gate: math + code + Lean, verifiable-reward only.
"""

from .omni_math import eval_omni_math

BENCH_REGISTRY = {
    "omni_math": eval_omni_math,
}
