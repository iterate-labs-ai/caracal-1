"""Ignite-3B bench registry.

L1 gate: math + code + Lean, verifiable-reward only.
"""

from .aime import eval_aime
from .bigcodebench import eval_bigcodebench
from .cyber_rcm import eval_cyber_rcm
from .livecodebench import eval_livecodebench
from .matharena import eval_matharena
from .omni_math import eval_omni_math
from .putnam_lean import eval_putnam_lean

BENCH_REGISTRY = {
    "omni_math": eval_omni_math,
    "livecodebench": eval_livecodebench,
    "bigcodebench": eval_bigcodebench,
    "matharena": eval_matharena,
    "aime": eval_aime,
    "putnam_lean": eval_putnam_lean,
    "cyber_rcm": eval_cyber_rcm,  # fase 2: RSI+RL cyber no Caracal
}
