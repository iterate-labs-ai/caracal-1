"""MAP-Elites 450 celulas para diversidade de comportamento.

30 CWE x 5 primitives x 3 langs = 450 cells.
"""

from __future__ import annotations

from collections import defaultdict

from .schema import CWE_CLASSES, EXPLOIT_PRIMITIVES, LANGUAGES, BehaviorCell, Checkpoint


class MAPElites:
    def __init__(self):
        self.cells: dict[str, list[Checkpoint]] = defaultdict(list)
        self.all_cells = self._enumerate_cells()

    def _enumerate_cells(self) -> list[BehaviorCell]:
        return [
            BehaviorCell(cwe_class=cwe, exploit_primitive=prim, language=lang)
            for cwe in CWE_CLASSES
            for prim in EXPLOIT_PRIMITIVES
            for lang in LANGUAGES
        ]

    def add(self, checkpoint: Checkpoint, cell: BehaviorCell) -> bool:
        """Add checkpoint to cell if it improves the best in cell."""
        existing = self.cells[cell.id]
        if not existing:
            self.cells[cell.id].append(checkpoint)
            return True

        current_best = max(existing, key=lambda c: c.scores.cybergym_pass_at_5)
        if checkpoint.scores.cybergym_pass_at_5 > current_best.scores.cybergym_pass_at_5:
            self.cells[cell.id].append(checkpoint)
            # Keep top 32 per cell
            self.cells[cell.id] = sorted(
                self.cells[cell.id],
                key=lambda c: c.scores.cybergym_pass_at_5,
                reverse=True,
            )[:32]
            return True
        return False

    def query_cell(self, cell: BehaviorCell) -> list[Checkpoint]:
        return self.cells.get(cell.id, [])

    def coverage(self) -> float:
        """Fraction of cells with at least one checkpoint."""
        return len([c for c in self.cells.values() if c]) / len(self.all_cells)

    def best_per_cell(self) -> dict[str, Checkpoint]:
        return {
            cell_id: max(ckpts, key=lambda c: c.scores.cybergym_pass_at_5)
            for cell_id, ckpts in self.cells.items()
            if ckpts
        }
