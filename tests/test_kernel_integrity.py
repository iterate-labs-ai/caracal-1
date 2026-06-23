"""Test kernel-D integrity check."""
from pathlib import Path

from harness.stop_pattern_detector import compute_kernel_hashes


def test_compute_kernel_hashes():
    root = Path(__file__).parent.parent
    hashes = compute_kernel_hashes(root)
    assert isinstance(hashes, dict)
    # At least classes.yaml deve existir
    if (root / "harness/kernel/classes.yaml").exists():
        assert "harness/kernel/classes.yaml" in hashes
