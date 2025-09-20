"""
Unit tests for model pruning utilities.
"""

import numpy as np

try:
    from src.vega.federated import model_pruning
except ImportError:
    import src.vega.federated.model_pruning as model_pruning


def test_magnitude_prune():
    arr = np.array([1.0, -2.0, 0.5, 0.1, -0.2, 3.0])
    pruned = model_pruning.magnitude_prune(arr, sparsity=0.5)
    # Should prune 3 smallest magnitude elements
    assert np.count_nonzero(pruned) == 3
    assert np.allclose(sorted(np.abs(pruned[pruned != 0])), [1.0, 2.0, 3.0])


def test_random_prune():
    arr = np.ones(10)
    pruned = model_pruning.random_prune(arr, sparsity=0.3, seed=42)
    assert np.count_nonzero(pruned) == 7
    # Pruned indices should be reproducible
    pruned2 = model_pruning.random_prune(arr, sparsity=0.3, seed=42)
    assert np.allclose(pruned, pruned2)


def test_mask_prune():
    arr = np.arange(5)
    mask = np.array([1, 0, 1, 0, 1])
    pruned = model_pruning.mask_prune(arr, mask)
    assert np.allclose(pruned, [0, 0, 2, 0, 4])


def run_all():
    test_magnitude_prune()
    test_random_prune()
    test_mask_prune()
    print("All model pruning tests passed.")


if __name__ == "__main__":
    run_all()
