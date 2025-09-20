"""
Unit tests for GradientCompression module.
"""

import numpy as np

try:
    from src.vega.federated.gradient_compression import (
        GradientCompression,
        GradientCompressionConfig,
    )
except ImportError:
    # Allow running as a script from this directory
    from src.vega.federated.gradient_compression import GradientCompression, GradientCompressionConfig


def test_topk():
    arr = np.random.randn(1000)
    config = GradientCompressionConfig(method="topk", k=20)
    gc = GradientCompression(config)
    comp = gc.compress(arr)
    rec = gc.decompress(comp)
    assert np.count_nonzero(rec) == 20
    assert rec.shape == arr.shape


def test_random():
    arr = np.random.randn(1000)
    config = GradientCompressionConfig(method="random", k=20)
    gc = GradientCompression(config)
    comp = gc.compress(arr)
    rec = gc.decompress(comp)
    assert np.count_nonzero(rec) == 20
    assert rec.shape == arr.shape


def test_quant8():
    arr = np.random.randn(1000)
    config = GradientCompressionConfig(method="quant8")
    gc = GradientCompression(config)
    comp = gc.compress(arr)
    rec = gc.decompress(comp)
    mse = np.mean((arr - rec) ** 2)
    assert mse < 1e-2
    assert rec.shape == arr.shape


def test_quant16():
    arr = np.random.randn(1000)
    config = GradientCompressionConfig(method="quant16")
    gc = GradientCompression(config)
    comp = gc.compress(arr)
    rec = gc.decompress(comp)
    mse = np.mean((arr - rec) ** 2)
    assert mse < 1e-4
    assert rec.shape == arr.shape


def test_sparse():
    # All zeros
    arr = np.zeros(1000)
    config = GradientCompressionConfig(method="sparse")
    gc = GradientCompression(config)
    comp = gc.compress(arr)
    rec = gc.decompress(comp)
    assert np.count_nonzero(rec) == 0
    assert rec.shape == arr.shape
    # All nonzeros
    arr = np.ones(1000)
    comp = gc.compress(arr)
    rec = gc.decompress(comp)
    assert np.allclose(arr, rec)
    # Random sparsity
    arr = np.random.randn(1000)
    arr[arr < 1.0] = 0  # Make sparse
    comp = gc.compress(arr)
    rec = gc.decompress(comp)
    assert np.allclose(arr, rec)
    # Thresholded sparsity
    arr = np.random.randn(1000)
    config = GradientCompressionConfig(method="sparse")
    gc = GradientCompression(config)
    comp = gc.compress(arr, threshold=1.5)
    rec = gc.decompress(comp)
    mask = np.abs(arr) > 1.5
    assert np.allclose(arr[mask], rec[mask])
    assert np.count_nonzero(rec) == np.count_nonzero(mask)


def run_all():
    test_topk()
    test_random()
    test_quant8()
    test_quant16()
    test_sparse()
    print("All GradientCompression tests passed.")


if __name__ == "__main__":
    run_all()
