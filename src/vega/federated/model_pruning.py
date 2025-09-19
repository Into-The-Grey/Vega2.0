"""
Model Pruning Utilities for Federated Learning

Supports magnitude-based, random, and custom mask pruning for model weights.
Framework-agnostic (numpy/tensor), with optional PyTorch support.
"""

import numpy as np
from typing import Optional, Callable

try:
    import torch
except ImportError:
    torch = None


def magnitude_prune(weights, sparsity: float) -> np.ndarray:
    """
    Prune weights by zeroing out the smallest magnitude elements.
    Args:
        weights: numpy array or torch tensor
        sparsity: float in (0, 1), fraction of weights to prune
    Returns:
        Pruned weights (same type as input)
    """
    if torch is not None and isinstance(weights, torch.Tensor):
        arr = weights.detach().cpu().numpy()
    else:
        arr = np.asarray(weights)
    flat = arr.flatten()
    k = int(sparsity * flat.size)
    if k == 0:
        return weights
    idx = np.argpartition(np.abs(flat), k)[:k]
    pruned = flat.copy()
    pruned[idx] = 0
    pruned = pruned.reshape(arr.shape)
    if torch is not None and isinstance(weights, torch.Tensor):
        return torch.from_numpy(pruned).to(weights.device)
    return pruned


def random_prune(weights, sparsity: float, seed: Optional[int] = None) -> np.ndarray:
    """
    Prune weights by randomly zeroing out elements.
    Args:
        weights: numpy array or torch tensor
        sparsity: float in (0, 1), fraction of weights to prune
        seed: random seed
    Returns:
        Pruned weights (same type as input)
    """
    if torch is not None and isinstance(weights, torch.Tensor):
        arr = weights.detach().cpu().numpy()
    else:
        arr = np.asarray(weights)
    flat = arr.flatten()
    k = int(sparsity * flat.size)
    if k == 0:
        return weights
    rng = np.random.default_rng(seed)
    idx = rng.choice(flat.size, k, replace=False)
    pruned = flat.copy()
    pruned[idx] = 0
    pruned = pruned.reshape(arr.shape)
    if torch is not None and isinstance(weights, torch.Tensor):
        return torch.from_numpy(pruned).to(weights.device)
    return pruned


def mask_prune(weights, mask: np.ndarray) -> np.ndarray:
    """
    Prune weights using a custom binary mask (0=prune, 1=keep).
    Args:
        weights: numpy array or torch tensor
        mask: binary mask, same shape as weights
    Returns:
        Pruned weights (same type as input)
    """
    if torch is not None and isinstance(weights, torch.Tensor):
        arr = weights.detach().cpu().numpy()
    else:
        arr = np.asarray(weights)
    pruned = arr * mask
    if torch is not None and isinstance(weights, torch.Tensor):
        return torch.from_numpy(pruned).to(weights.device)
    return pruned
