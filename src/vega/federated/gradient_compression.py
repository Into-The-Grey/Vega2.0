"""
Gradient Compression Techniques for Federated Learning

Implements top-k sparsification, random sparsification, and quantization for model updates.
Modular and pluggable into any federated learning workflow.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch
import logging

logger = logging.getLogger(__name__)


@dataclass
class GradientCompressionConfig:
    method: str = "topk"  # "topk", "random", "quant8", "quant16"
    k: int = 100  # For top-k and random sparsification
    quant_bits: int = 8  # For quantization


class GradientCompression:
    def __init__(self, config: GradientCompressionConfig):
        self.config = config

    def compress(self, update: np.ndarray, threshold: float = 0.0) -> Dict[str, Any]:
        method = self.config.method
        if method == "topk":
            flat = update.flatten()
            idx = np.argpartition(np.abs(flat), -self.config.k)[-self.config.k :]
            values = flat[idx]
            return {
                "method": "topk",
                "idx": idx,
                "values": values,
                "shape": update.shape,
            }
        elif method == "random":
            flat = update.flatten()
            idx = np.random.choice(len(flat), self.config.k, replace=False)
            values = flat[idx]
            return {
                "method": "random",
                "idx": idx,
                "values": values,
                "shape": update.shape,
            }
        elif method == "quant8":
            min_val, max_val = update.min(), update.max()
            scale = (max_val - min_val) / 255 if max_val > min_val else 1.0
            quantized = np.round((update - min_val) / scale).astype(np.uint8)
            return {
                "method": "quant8",
                "quantized": quantized,
                "min": min_val,
                "scale": scale,
            }
        elif method == "quant16":
            min_val, max_val = update.min(), update.max()
            scale = (max_val - min_val) / 65535 if max_val > min_val else 1.0
            quantized = np.round((update - min_val) / scale).astype(np.uint16)
            return {
                "method": "quant16",
                "quantized": quantized,
                "min": min_val,
                "scale": scale,
            }
        elif method == "sparse":
            flat = update.flatten()
            if threshold > 0.0:
                mask = np.abs(flat) > threshold
            else:
                mask = flat != 0
            idx = np.where(mask)[0]
            values = flat[idx]
            return {
                "method": "sparse",
                "idx": idx,
                "values": values,
                "shape": update.shape,
            }
        else:
            raise ValueError(f"Unknown compression method: {self.config.method}")

    def decompress(self, compressed: Dict[str, Any]) -> np.ndarray:
        method = compressed["method"]
        if method in ("topk", "random", "sparse"):
            arr = np.zeros(
                np.prod(compressed["shape"]), dtype=compressed["values"].dtype
            )
            arr[compressed["idx"]] = compressed["values"]
            return arr.reshape(compressed["shape"])
        elif method == "quant8":
            return (
                compressed["quantized"].astype(np.float32) * compressed["scale"]
                + compressed["min"]
            )
        elif method == "quant16":
            return (
                compressed["quantized"].astype(np.float32) * compressed["scale"]
                + compressed["min"]
            )
        else:
            raise ValueError(f"Unknown decompression method: {method}")


def test_gradient_compression():
    print("Testing Gradient Compression...")
    np.random.seed(42)
    arr = np.random.randn(1000)
    # Top-k
    config = GradientCompressionConfig(method="topk", k=10)
    gc = GradientCompression(config)
    comp = gc.compress(arr)
    rec = gc.decompress(comp)
    print(f"Top-k: Nonzero in rec: {np.count_nonzero(rec)}")
    assert np.count_nonzero(rec) == 10
    # Random
    config = GradientCompressionConfig(method="random", k=10)
    gc = GradientCompression(config)
    comp = gc.compress(arr)
    rec = gc.decompress(comp)
    print(f"Random: Nonzero in rec: {np.count_nonzero(rec)}")
    assert np.count_nonzero(rec) == 10
    # Quant8
    config = GradientCompressionConfig(method="quant8")
    gc = GradientCompression(config)
    comp = gc.compress(arr)
    rec = gc.decompress(comp)
    print(f"Quant8: MSE: {np.mean((arr - rec) ** 2):.6f}")
    assert rec.shape == arr.shape
    # Quant16
    config = GradientCompressionConfig(method="quant16")
    gc = GradientCompression(config)
    comp = gc.compress(arr)
    rec = gc.decompress(comp)
    print(f"Quant16: MSE: {np.mean((arr - rec) ** 2):.6f}")
    assert rec.shape == arr.shape
    print("🎉 Gradient Compression test completed successfully!")
    return True


if __name__ == "__main__":
    test_gradient_compression()
