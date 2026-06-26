"""DGM parent sampling formula: weight = score / (1 + children_with_edit_cap).

Nao-greedy. Preserve diversidade. Originario Sakana DGM 2025.
"""

from __future__ import annotations

import math
import random

from .schema import Checkpoint


def sample_parents(
    candidates: list[Checkpoint],
    n: int = 1,
    temperature: float = 1.0,
    pareto_only: bool = True,
) -> list[Checkpoint]:
    """DGM formula: p_i = score_i / (1 + children_with_edit_cap_i).

    Softmax com temperature configuravel.
    """
    if pareto_only:
        candidates = [c for c in candidates if c.pareto_flag]

    if not candidates:
        return []

    weights = [c.scores.cybergym_pass_at_5 / (1 + c.children_with_edit_cap) for c in candidates]
    # Softmax com temperature
    max_w = max(weights)
    exp_w = [math.exp((w - max_w) / temperature) for w in weights]
    total = sum(exp_w)
    probs = [e / total for e in exp_w]

    # Anti-collapse: se top-K >0.95 prob total, forca 5% nos outros
    if max(probs) > 0.95:
        force_other = 0.05 / max(1, len(probs) - 1)
        adjusted = [p * 0.95 if p == max(probs) else p + force_other for p in probs]
        probs = [p / sum(adjusted) for p in adjusted]

    sampled = random.choices(candidates, weights=probs, k=n)
    return sampled


def update_children_count(parent: Checkpoint, child: Checkpoint) -> None:
    """Apos novo descendente nascer, incrementa parent counter."""
    parent.children_with_edit_cap += 1
