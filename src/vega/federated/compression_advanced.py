"""
Communication-Efficient Protocols for Federated Learning

Advanced compression techniques for reducing communication overhead in federated learning,
including gradient sparsification, quantization methods, sketching techniques, and error
feedback mechanisms. This module provides comprehensive compression strategies to optimize
bandwidth usage while maintaining model accuracy.

Features:
- Gradient sparsification (Top-K, Random-K, Threshold-based)
- Quantization methods (Uniform, Non-uniform, Adaptive)
- Sketching techniques (Count Sketch, Johnson-Lindenstrauss)
- Error feedback and compression error accumulation
- Bandwidth-aware compression selection
- Communication round optimization
- Compression performance monitoring
"""

import asyncio
import logging
import numpy as np
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import hashlib
import json


logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Types of compression algorithms."""

    SPARSIFICATION = "sparsification"
    QUANTIZATION = "quantization"
    SKETCHING = "sketching"
    HYBRID = "hybrid"


class SparsificationMethod(Enum):
    """Gradient sparsification methods."""

    TOP_K = "top_k"
    RANDOM_K = "random_k"
    THRESHOLD = "threshold"
    MAGNITUDE_BASED = "magnitude_based"
    LAYERWISE_TOP_K = "layerwise_top_k"


class QuantizationMethod(Enum):
    """Quantization methods."""

    UNIFORM = "uniform"
    NON_UNIFORM = "non_uniform"
    ADAPTIVE = "adaptive"
    STOCHASTIC = "stochastic"
    SIGNSGD = "signsgd"


class SketchingMethod(Enum):
    """Sketching methods."""

    COUNT_SKETCH = "count_sketch"
    JOHNSON_LINDENSTRAUSS = "johnson_lindenstrauss"
    RANDOM_PROJECTION = "random_projection"
    FEATURE_HASHING = "feature_hashing"


@dataclass
class CompressionConfig:
    """Configuration for compression algorithms."""

    compression_type: CompressionType = CompressionType.SPARSIFICATION
    compression_ratio: float = 0.1  # For sparsification: fraction to keep
    quantization_bits: int = 8  # For quantization
    sketch_size: int = 1024  # For sketching
    enable_error_feedback: bool = True
    adaptive_threshold: bool = True
    bandwidth_budget: Optional[float] = None  # MB per round
    convergence_tolerance: float = 1e-4


@dataclass
class CompressionResult:
    """Result of compression operation."""

    compressed_data: Any
    compression_ratio: float
    original_size: int
    compressed_size: int
    compression_time: float
    decompression_time: float = 0.0
    compression_error: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def compression_efficiency(self) -> float:
        """Compression efficiency as size reduction ratio."""
        return 1.0 - (self.compressed_size / max(self.original_size, 1))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "compression_ratio": self.compression_ratio,
            "original_size": self.original_size,
            "compressed_size": self.compressed_size,
            "compression_time": self.compression_time,
            "decompression_time": self.decompression_time,
            "compression_error": self.compression_error,
            "compression_efficiency": self.compression_efficiency,
            "metadata": self.metadata,
        }


class CompressionAlgorithm(ABC):
    """Abstract base class for compression algorithms."""

    def __init__(self, config: CompressionConfig):
        self.config = config
        self.error_accumulator = {}  # For error feedback
        self.compression_history = []

    @abstractmethod
    def compress(self, data: Dict[str, np.ndarray]) -> CompressionResult:
        """Compress the given data."""
        pass

    @abstractmethod
    def decompress(self, compressed_result: CompressionResult) -> Dict[str, np.ndarray]:
        """Decompress the compressed data."""
        pass

    def _calculate_data_size(self, data: Dict[str, np.ndarray]) -> int:
        """Calculate total size of data in bytes."""
        total_size = 0
        for key, tensor in data.items():
            total_size += tensor.nbytes
        return total_size

    def _apply_error_feedback(
        self, data: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Apply error feedback mechanism."""
        if not self.config.enable_error_feedback:
            return data

        corrected_data = {}
        for key, tensor in data.items():
            if key in self.error_accumulator:
                # Add accumulated error to current gradient
                corrected_data[key] = tensor + self.error_accumulator[key]
            else:
                corrected_data[key] = tensor.copy()

        return corrected_data

    def _update_error_feedback(
        self, original: Dict[str, np.ndarray], compressed: Dict[str, np.ndarray]
    ) -> None:
        """Update error accumulator."""
        if not self.config.enable_error_feedback:
            return

        for key in original.keys():
            error = original[key] - compressed.get(key, np.zeros_like(original[key]))

            if key in self.error_accumulator:
                self.error_accumulator[key] += error
            else:
                self.error_accumulator[key] = error

    def _calculate_compression_error(
        self, original: Dict[str, np.ndarray], compressed: Dict[str, np.ndarray]
    ) -> float:
        """Calculate compression error as relative L2 norm."""
        total_error = 0.0
        total_norm = 0.0

        for key in original.keys():
            orig_tensor = original[key]
            comp_tensor = compressed.get(key, np.zeros_like(orig_tensor))

            error = np.linalg.norm(orig_tensor - comp_tensor)
            norm = np.linalg.norm(orig_tensor)

            total_error += error**2
            total_norm += norm**2

        return np.sqrt(total_error) / max(np.sqrt(total_norm), 1e-8)


class GradientSparsification(CompressionAlgorithm):
    """Gradient sparsification algorithms."""

    def __init__(
        self,
        config: CompressionConfig,
        method: SparsificationMethod = SparsificationMethod.TOP_K,
    ):
        super().__init__(config)
        self.method = method
        self.adaptive_ratio = config.compression_ratio

    def compress(self, data: Dict[str, np.ndarray]) -> CompressionResult:
        """Compress gradients using sparsification."""
        start_time = time.time()

        # Apply error feedback
        corrected_data = self._apply_error_feedback(data)

        compressed_data = {}
        original_size = self._calculate_data_size(corrected_data)
        total_elements = sum(tensor.size for tensor in corrected_data.values())

        if self.method == SparsificationMethod.TOP_K:
            compressed_data = self._top_k_sparsification(corrected_data)
        elif self.method == SparsificationMethod.RANDOM_K:
            compressed_data = self._random_k_sparsification(corrected_data)
        elif self.method == SparsificationMethod.THRESHOLD:
            compressed_data = self._threshold_sparsification(corrected_data)
        elif self.method == SparsificationMethod.MAGNITUDE_BASED:
            compressed_data = self._magnitude_based_sparsification(corrected_data)
        elif self.method == SparsificationMethod.LAYERWISE_TOP_K:
            compressed_data = self._layerwise_top_k_sparsification(corrected_data)

        compressed_size = self._calculate_compressed_size(compressed_data)
        compression_time = time.time() - start_time

        # Update error feedback
        decompressed_data = self._decompress_sparse(compressed_data, corrected_data)
        self._update_error_feedback(corrected_data, decompressed_data)

        # Calculate compression metrics
        actual_sparsity = self._calculate_sparsity(compressed_data)
        compression_error = self._calculate_compression_error(
            corrected_data, decompressed_data
        )

        result = CompressionResult(
            compressed_data=compressed_data,
            compression_ratio=actual_sparsity,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_time=compression_time,
            compression_error=compression_error,
            metadata={
                "method": self.method.value,
                "total_elements": total_elements,
                "kept_elements": sum(
                    len(indices) for _, indices, _ in compressed_data.values()
                ),
                "sparsity_level": actual_sparsity,
            },
        )

        self.compression_history.append(result.to_dict())
        return result

    def decompress(self, compressed_result: CompressionResult) -> Dict[str, np.ndarray]:
        """Decompress sparsified gradients."""
        start_time = time.time()

        compressed_data = compressed_result.compressed_data
        decompressed = {}

        for key, (shape, indices, values) in compressed_data.items():
            tensor = np.zeros(shape, dtype=np.float32)
            if len(indices) > 0 and len(values) > 0:
                # Handle both 1D and multi-dimensional indices
                if isinstance(indices[0], (list, tuple, np.ndarray)):
                    # Multi-dimensional indices
                    indices_tuple = tuple(zip(*indices))
                    tensor[indices_tuple] = values
                else:
                    # 1D indices
                    tensor.flat[indices] = values
            decompressed[key] = tensor

        compressed_result.decompression_time = time.time() - start_time
        return decompressed

    def _top_k_sparsification(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Top-K sparsification algorithm."""
        compressed = {}

        for key, tensor in data.items():
            flat_tensor = tensor.flatten()
            k = max(1, int(len(flat_tensor) * self.config.compression_ratio))

            # Get top-k indices by absolute magnitude
            top_k_indices = np.argpartition(np.abs(flat_tensor), -k)[-k:]
            top_k_values = flat_tensor[top_k_indices]

            compressed[key] = (
                tensor.shape,
                top_k_indices.tolist(),
                top_k_values.tolist(),
            )

        return compressed

    def _random_k_sparsification(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Random-K sparsification algorithm."""
        compressed = {}

        for key, tensor in data.items():
            flat_tensor = tensor.flatten()
            k = max(1, int(len(flat_tensor) * self.config.compression_ratio))

            # Random sample k indices
            random_indices = np.random.choice(len(flat_tensor), size=k, replace=False)
            random_values = flat_tensor[random_indices]

            compressed[key] = (
                tensor.shape,
                random_indices.tolist(),
                random_values.tolist(),
            )

        return compressed

    def _threshold_sparsification(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Threshold-based sparsification algorithm."""
        compressed = {}

        for key, tensor in data.items():
            flat_tensor = tensor.flatten()

            if self.config.adaptive_threshold:
                # Adaptive threshold based on percentile
                threshold = np.percentile(
                    np.abs(flat_tensor), 100 * (1 - self.config.compression_ratio)
                )
            else:
                # Fixed threshold
                threshold = np.std(flat_tensor) * 0.5

            # Keep elements above threshold
            mask = np.abs(flat_tensor) >= threshold
            indices = np.where(mask)[0]
            values = flat_tensor[indices]

            compressed[key] = (tensor.shape, indices.tolist(), values.tolist())

        return compressed

    def _magnitude_based_sparsification(
        self, data: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Magnitude-based sparsification with layerwise adaptation."""
        compressed = {}

        for key, tensor in data.items():
            flat_tensor = tensor.flatten()

            # Calculate layer-specific sparsification ratio
            layer_magnitude = np.mean(np.abs(flat_tensor))
            adaptive_ratio = min(
                self.config.compression_ratio * 2,
                self.config.compression_ratio * (1 + layer_magnitude),
            )

            k = max(1, int(len(flat_tensor) * adaptive_ratio))

            # Get top elements by magnitude
            top_indices = np.argpartition(np.abs(flat_tensor), -k)[-k:]
            top_values = flat_tensor[top_indices]

            compressed[key] = (tensor.shape, top_indices.tolist(), top_values.tolist())

        return compressed

    def _layerwise_top_k_sparsification(
        self, data: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Layer-wise top-K sparsification with different ratios per layer."""
        compressed = {}

        # Calculate layer importance scores
        layer_scores = {}
        for key, tensor in data.items():
            layer_scores[key] = np.mean(np.abs(tensor))

        # Normalize scores
        total_score = sum(layer_scores.values())
        for key in layer_scores:
            layer_scores[key] /= max(total_score, 1e-8)

        for key, tensor in data.items():
            flat_tensor = tensor.flatten()

            # Adaptive compression ratio based on layer importance
            layer_ratio = self.config.compression_ratio * (1 + layer_scores[key])
            k = max(1, int(len(flat_tensor) * layer_ratio))

            # Get top-k elements
            top_k_indices = np.argpartition(np.abs(flat_tensor), -k)[-k:]
            top_k_values = flat_tensor[top_k_indices]

            compressed[key] = (
                tensor.shape,
                top_k_indices.tolist(),
                top_k_values.tolist(),
            )

        return compressed

    def _decompress_sparse(
        self, compressed_data: Dict[str, Any], original_data: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Helper to decompress sparse data for error feedback."""
        decompressed = {}

        for key, (shape, indices, values) in compressed_data.items():
            tensor = np.zeros(shape, dtype=np.float32)
            if len(indices) > 0 and len(values) > 0:
                tensor.flat[indices] = values
            decompressed[key] = tensor

        return decompressed

    def _calculate_compressed_size(self, compressed_data: Dict[str, Any]) -> int:
        """Calculate size of compressed data."""
        total_size = 0
        for key, (shape, indices, values) in compressed_data.items():
            # Size = shape info + indices + values
            total_size += len(shape) * 4  # int32 for shape
            total_size += len(indices) * 4  # int32 for indices
            total_size += len(values) * 4  # float32 for values
        return total_size

    def _calculate_sparsity(self, compressed_data: Dict[str, Any]) -> float:
        """Calculate actual sparsity level."""
        total_elements = 0
        kept_elements = 0

        for key, (shape, indices, values) in compressed_data.items():
            total_elements += np.prod(shape)
            kept_elements += len(indices)

        return kept_elements / max(total_elements, 1)

    def _calculate_compression_error(
        self, original: Dict[str, np.ndarray], compressed: Dict[str, np.ndarray]
    ) -> float:
        """Calculate compression error as L2 norm."""
        total_error = 0.0
        total_norm = 0.0

        for key in original.keys():
            orig_tensor = original[key]
            comp_tensor = compressed.get(key, np.zeros_like(orig_tensor))

            error = np.linalg.norm(orig_tensor - comp_tensor)
            norm = np.linalg.norm(orig_tensor)

            total_error += error**2
            total_norm += norm**2

        return np.sqrt(total_error) / max(np.sqrt(total_norm), 1e-8)


class Quantization(CompressionAlgorithm):
    """Quantization algorithms for gradient compression."""

    def __init__(
        self,
        config: CompressionConfig,
        method: QuantizationMethod = QuantizationMethod.UNIFORM,
    ):
        super().__init__(config)
        self.method = method
        self.quantization_levels = 2**config.quantization_bits

    def compress(self, data: Dict[str, np.ndarray]) -> CompressionResult:
        """Compress gradients using quantization."""
        start_time = time.time()

        # Apply error feedback
        corrected_data = self._apply_error_feedback(data)

        compressed_data = {}
        original_size = self._calculate_data_size(corrected_data)

        if self.method == QuantizationMethod.UNIFORM:
            compressed_data = self._uniform_quantization(corrected_data)
        elif self.method == QuantizationMethod.NON_UNIFORM:
            compressed_data = self._non_uniform_quantization(corrected_data)
        elif self.method == QuantizationMethod.ADAPTIVE:
            compressed_data = self._adaptive_quantization(corrected_data)
        elif self.method == QuantizationMethod.STOCHASTIC:
            compressed_data = self._stochastic_quantization(corrected_data)
        elif self.method == QuantizationMethod.SIGNSGD:
            compressed_data = self._sign_sgd_quantization(corrected_data)

        compressed_size = self._calculate_quantized_size(compressed_data)
        compression_time = time.time() - start_time

        # Update error feedback
        decompressed_data = self._dequantize(compressed_data)
        self._update_error_feedback(corrected_data, decompressed_data)

        # Calculate compression metrics
        compression_error = self._calculate_compression_error(
            corrected_data, decompressed_data
        )

        result = CompressionResult(
            compressed_data=compressed_data,
            compression_ratio=compressed_size / max(original_size, 1),
            original_size=original_size,
            compressed_size=compressed_size,
            compression_time=compression_time,
            compression_error=compression_error,
            metadata={
                "method": self.method.value,
                "quantization_bits": self.config.quantization_bits,
                "quantization_levels": self.quantization_levels,
            },
        )

        self.compression_history.append(result.to_dict())
        return result

    def decompress(self, compressed_result: CompressionResult) -> Dict[str, np.ndarray]:
        """Decompress quantized gradients."""
        start_time = time.time()

        decompressed = self._dequantize(compressed_result.compressed_data)
        compressed_result.decompression_time = time.time() - start_time

        return decompressed

    def _uniform_quantization(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Uniform quantization algorithm."""
        quantized = {}

        for key, tensor in data.items():
            # Find min and max values
            min_val = np.min(tensor)
            max_val = np.max(tensor)

            if min_val == max_val:
                # Handle constant tensors
                quantized[key] = {
                    "shape": tensor.shape,
                    "method": "uniform",
                    "min_val": min_val,
                    "max_val": max_val,
                    "quantized": np.zeros(tensor.shape, dtype=np.uint8),
                }
            else:
                # Quantize to discrete levels
                scale = (max_val - min_val) / (self.quantization_levels - 1)
                quantized_tensor = np.round((tensor - min_val) / scale).astype(np.uint8)

                quantized[key] = {
                    "shape": tensor.shape,
                    "method": "uniform",
                    "min_val": min_val,
                    "max_val": max_val,
                    "scale": scale,
                    "quantized": quantized_tensor,
                }

        return quantized

    def _non_uniform_quantization(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Non-uniform quantization based on gradient distribution."""
        quantized = {}

        for key, tensor in data.items():
            flat_tensor = tensor.flatten()

            # Use percentiles for non-uniform quantization levels
            percentiles = np.linspace(0, 100, self.quantization_levels + 1)
            quantization_boundaries = np.percentile(flat_tensor, percentiles)

            # Quantize using boundaries
            quantized_tensor = np.digitize(flat_tensor, quantization_boundaries[1:-1])
            quantized_tensor = np.clip(
                quantized_tensor, 0, self.quantization_levels - 1
            )

            quantized[key] = {
                "shape": tensor.shape,
                "method": "non_uniform",
                "boundaries": quantization_boundaries.tolist(),
                "quantized": quantized_tensor.reshape(tensor.shape).astype(np.uint8),
            }

        return quantized

    def _adaptive_quantization(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Adaptive quantization with layer-specific bit allocation."""
        quantized = {}

        # Calculate layer importance for bit allocation
        layer_importance = {}
        total_variance = 0

        for key, tensor in data.items():
            variance = np.var(tensor)
            layer_importance[key] = variance
            total_variance += variance

        for key, tensor in data.items():
            # Adaptive bit allocation based on layer importance
            importance_ratio = layer_importance[key] / max(total_variance, 1e-8)
            adaptive_bits = max(
                2, int(self.config.quantization_bits * (0.5 + importance_ratio))
            )
            adaptive_levels = 2**adaptive_bits

            # Uniform quantization with adaptive levels
            min_val = np.min(tensor)
            max_val = np.max(tensor)

            if min_val == max_val:
                quantized_tensor = np.zeros(tensor.shape, dtype=np.uint8)
                scale = 1.0
            else:
                scale = (max_val - min_val) / (adaptive_levels - 1)
                quantized_tensor = np.round((tensor - min_val) / scale).astype(np.uint8)

            quantized[key] = {
                "shape": tensor.shape,
                "method": "adaptive",
                "min_val": min_val,
                "max_val": max_val,
                "scale": scale,
                "bits": adaptive_bits,
                "quantized": quantized_tensor,
            }

        return quantized

    def _stochastic_quantization(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Stochastic quantization with probabilistic rounding."""
        quantized = {}

        for key, tensor in data.items():
            min_val = np.min(tensor)
            max_val = np.max(tensor)

            if min_val == max_val:
                quantized_tensor = np.zeros(tensor.shape, dtype=np.uint8)
            else:
                # Stochastic rounding
                scale = (max_val - min_val) / (self.quantization_levels - 1)
                normalized = (tensor - min_val) / scale

                # Probabilistic rounding
                floor_vals = np.floor(normalized).astype(np.uint8)
                ceil_probs = normalized - floor_vals
                random_vals = np.random.random(tensor.shape)

                quantized_tensor = floor_vals + (random_vals < ceil_probs).astype(
                    np.uint8
                )
                quantized_tensor = np.clip(
                    quantized_tensor, 0, self.quantization_levels - 1
                )

            quantized[key] = {
                "shape": tensor.shape,
                "method": "stochastic",
                "min_val": min_val,
                "max_val": max_val,
                "scale": scale if min_val != max_val else 1.0,
                "quantized": quantized_tensor,
            }

        return quantized

    def _sign_sgd_quantization(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """SignSGD quantization (1-bit quantization)."""
        quantized = {}

        for key, tensor in data.items():
            # Extract sign and magnitude
            signs = np.sign(tensor).astype(np.int8)
            magnitude = np.mean(np.abs(tensor))

            quantized[key] = {
                "shape": tensor.shape,
                "method": "signsgd",
                "signs": signs,
                "magnitude": magnitude,
            }

        return quantized

    def _dequantize(self, quantized_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Dequantize compressed data."""
        dequantized = {}

        for key, data in quantized_data.items():
            method = data["method"]

            if method == "uniform":
                if data["min_val"] == data["max_val"]:
                    tensor = np.full(data["shape"], data["min_val"], dtype=np.float32)
                else:
                    tensor = (
                        data["quantized"].astype(np.float32) * data["scale"]
                        + data["min_val"]
                    )

            elif method == "non_uniform":
                boundaries = np.array(data["boundaries"])
                quantized_vals = data["quantized"]

                # Map quantized indices to boundary midpoints
                midpoints = (boundaries[:-1] + boundaries[1:]) / 2
                tensor = midpoints[np.clip(quantized_vals, 0, len(midpoints) - 1)]

            elif method == "adaptive":
                if data["min_val"] == data["max_val"]:
                    tensor = np.full(data["shape"], data["min_val"], dtype=np.float32)
                else:
                    tensor = (
                        data["quantized"].astype(np.float32) * data["scale"]
                        + data["min_val"]
                    )

            elif method == "stochastic":
                tensor = (
                    data["quantized"].astype(np.float32) * data["scale"]
                    + data["min_val"]
                )

            elif method == "signsgd":
                tensor = data["signs"].astype(np.float32) * data["magnitude"]

            else:
                raise ValueError(f"Unknown quantization method: {method}")

            dequantized[key] = tensor.astype(np.float32)

        return dequantized

    def _calculate_quantized_size(self, quantized_data: Dict[str, Any]) -> int:
        """Calculate size of quantized data."""
        total_size = 0

        for key, data in quantized_data.items():
            method = data["method"]

            if method in ["uniform", "adaptive", "stochastic"]:
                total_size += data["quantized"].nbytes
                total_size += 8  # min_val, max_val
                total_size += 4  # scale

            elif method == "non_uniform":
                total_size += data["quantized"].nbytes
                total_size += len(data["boundaries"]) * 4  # boundaries

            elif method == "signsgd":
                total_size += data["signs"].nbytes
                total_size += 4  # magnitude

        return total_size


class Sketching(CompressionAlgorithm):
    """Sketching algorithms for gradient compression."""

    def __init__(
        self,
        config: CompressionConfig,
        method: SketchingMethod = SketchingMethod.COUNT_SKETCH,
    ):
        super().__init__(config)
        self.method = method
        self.sketch_size = config.sketch_size

        # Initialize sketching matrices/hash functions
        self._initialize_sketching_parameters()

    def _initialize_sketching_parameters(self):
        """Initialize parameters for sketching methods."""
        if self.method == SketchingMethod.COUNT_SKETCH:
            # Count sketch uses hash functions
            self.hash_functions = []
            self.sign_functions = []

        elif self.method == SketchingMethod.JOHNSON_LINDENSTRAUSS:
            # JL uses random projection matrix
            self.projection_matrices = {}

        elif self.method == SketchingMethod.RANDOM_PROJECTION:
            # Random projection matrices
            self.projection_matrices = {}

    def compress(self, data: Dict[str, np.ndarray]) -> CompressionResult:
        """Compress gradients using sketching."""
        start_time = time.time()

        # Apply error feedback
        corrected_data = self._apply_error_feedback(data)

        compressed_data = {}
        original_size = self._calculate_data_size(corrected_data)

        if self.method == SketchingMethod.COUNT_SKETCH:
            compressed_data = self._count_sketch(corrected_data)
        elif self.method == SketchingMethod.JOHNSON_LINDENSTRAUSS:
            compressed_data = self._johnson_lindenstrauss(corrected_data)
        elif self.method == SketchingMethod.RANDOM_PROJECTION:
            compressed_data = self._random_projection(corrected_data)
        elif self.method == SketchingMethod.FEATURE_HASHING:
            compressed_data = self._feature_hashing(corrected_data)

        compressed_size = self._calculate_sketch_size(compressed_data)
        compression_time = time.time() - start_time

        # Update error feedback (sketching has inherent information loss)
        decompressed_data = self._desketch(compressed_data, corrected_data)
        self._update_error_feedback(corrected_data, decompressed_data)

        # Calculate compression metrics
        compression_error = self._calculate_compression_error(
            corrected_data, decompressed_data
        )

        result = CompressionResult(
            compressed_data=compressed_data,
            compression_ratio=compressed_size / max(original_size, 1),
            original_size=original_size,
            compressed_size=compressed_size,
            compression_time=compression_time,
            compression_error=compression_error,
            metadata={
                "method": self.method.value,
                "sketch_size": self.sketch_size,
                "compression_factor": original_size / max(compressed_size, 1),
            },
        )

        self.compression_history.append(result.to_dict())
        return result

    def decompress(self, compressed_result: CompressionResult) -> Dict[str, np.ndarray]:
        """Decompress sketched gradients."""
        start_time = time.time()

        # Note: Sketching generally doesn't allow perfect reconstruction
        # This returns an approximation based on the original shapes
        decompressed = {}

        for key, sketch_data in compressed_result.compressed_data.items():
            original_shape = sketch_data["original_shape"]
            sketch = sketch_data["sketch"]

            if self.method == SketchingMethod.COUNT_SKETCH:
                # Simple reconstruction: reshape sketch to original dimensions
                total_elements = np.prod(original_shape)
                if sketch.size >= total_elements:
                    # Truncate sketch
                    decompressed[key] = sketch[:total_elements].reshape(original_shape)
                else:
                    # Pad sketch with zeros
                    padded = np.zeros(total_elements)
                    padded[: sketch.size] = sketch
                    decompressed[key] = padded.reshape(original_shape)
            else:
                # For other methods, use similar approach
                total_elements = np.prod(original_shape)
                if sketch.size >= total_elements:
                    decompressed[key] = sketch[:total_elements].reshape(original_shape)
                else:
                    padded = np.zeros(total_elements)
                    padded[: sketch.size] = sketch
                    decompressed[key] = padded.reshape(original_shape)

        compressed_result.decompression_time = time.time() - start_time
        return decompressed

    def _count_sketch(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Count sketch compression."""
        sketched = {}

        for key, tensor in data.items():
            flat_tensor = tensor.flatten()
            sketch = np.zeros(self.sketch_size, dtype=np.float32)

            # Simple hash-based count sketch
            for i, value in enumerate(flat_tensor):
                hash_val = hash(f"{key}_{i}") % self.sketch_size
                sign_val = 1 if hash(f"sign_{key}_{i}") % 2 == 0 else -1
                sketch[hash_val] += sign_val * value

            sketched[key] = {
                "method": "count_sketch",
                "original_shape": tensor.shape,
                "sketch": sketch,
                "sketch_size": self.sketch_size,
            }

        return sketched

    def _johnson_lindenstrauss(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Johnson-Lindenstrauss sketching."""
        sketched = {}

        for key, tensor in data.items():
            flat_tensor = tensor.flatten()

            # Create or reuse projection matrix
            if key not in self.projection_matrices:
                # Random Gaussian projection matrix
                self.projection_matrices[key] = np.random.randn(
                    self.sketch_size, len(flat_tensor)
                ) / np.sqrt(self.sketch_size)

            projection_matrix = self.projection_matrices[key]

            # Project to lower dimension
            sketch = projection_matrix @ flat_tensor

            sketched[key] = {
                "method": "johnson_lindenstrauss",
                "original_shape": tensor.shape,
                "sketch": sketch,
                "sketch_size": self.sketch_size,
            }

        return sketched

    def _random_projection(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Random projection sketching."""
        sketched = {}

        for key, tensor in data.items():
            flat_tensor = tensor.flatten()

            # Sparse random projection for efficiency
            if key not in self.projection_matrices:
                # Sparse random matrix (more efficient)
                projection_matrix = np.random.choice(
                    [-1, 0, 1],
                    size=(self.sketch_size, len(flat_tensor)),
                    p=[0.1, 0.8, 0.1],  # Mostly zeros
                ) / np.sqrt(self.sketch_size)
                self.projection_matrices[key] = projection_matrix

            projection_matrix = self.projection_matrices[key]
            sketch = projection_matrix @ flat_tensor

            sketched[key] = {
                "method": "random_projection",
                "original_shape": tensor.shape,
                "sketch": sketch,
                "sketch_size": self.sketch_size,
            }

        return sketched

    def _feature_hashing(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Feature hashing sketching."""
        sketched = {}

        for key, tensor in data.items():
            flat_tensor = tensor.flatten()
            sketch = np.zeros(self.sketch_size, dtype=np.float32)

            # Feature hashing with collision handling
            for i, value in enumerate(flat_tensor):
                hash_val = hash(f"{key}_{i}") % self.sketch_size
                sketch[hash_val] += value  # Simple accumulation

            sketched[key] = {
                "method": "feature_hashing",
                "original_shape": tensor.shape,
                "sketch": sketch,
                "sketch_size": self.sketch_size,
            }

        return sketched

    def _desketch(
        self, sketched_data: Dict[str, Any], original_shapes: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Approximate reconstruction from sketches (for error feedback)."""
        reconstructed = {}

        for key, sketch_info in sketched_data.items():
            original_shape = original_shapes[key].shape

            # Simple reconstruction: reshape sketch to original dimensions
            # This is a crude approximation for error feedback purposes
            sketch = sketch_info["sketch"]

            if sketch.size >= np.prod(original_shape):
                # Truncate sketch
                reconstructed[key] = sketch[: np.prod(original_shape)].reshape(
                    original_shape
                )
            else:
                # Pad sketch with zeros
                padded = np.zeros(np.prod(original_shape))
                padded[: sketch.size] = sketch
                reconstructed[key] = padded.reshape(original_shape)

        return reconstructed

    def _calculate_sketch_size(self, sketched_data: Dict[str, Any]) -> int:
        """Calculate size of sketched data."""
        total_size = 0

        for key, sketch_info in sketched_data.items():
            total_size += sketch_info["sketch"].nbytes
            total_size += len(sketch_info["original_shape"]) * 4  # shape info

        return total_size


@dataclass
class CommunicationMetrics:
    """Metrics for communication efficiency."""

    round_number: int
    participant_id: str
    bytes_sent: int
    bytes_received: int
    compression_time: float
    decompression_time: float
    compression_ratio: float
    compression_error: float
    bandwidth_utilization: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "round_number": self.round_number,
            "participant_id": self.participant_id,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "compression_time": self.compression_time,
            "decompression_time": self.decompression_time,
            "compression_ratio": self.compression_ratio,
            "compression_error": self.compression_error,
            "bandwidth_utilization": self.bandwidth_utilization,
            "timestamp": self.timestamp,
        }


class CommunicationOptimizer:
    """Optimizer for selecting best compression strategy based on conditions."""

    def __init__(self, bandwidth_budget: float = 10.0):  # MB
        self.bandwidth_budget = bandwidth_budget
        self.performance_history = {}
        self.network_conditions = {"latency": 50, "bandwidth": 100}  # ms, Mbps

    def select_compression_strategy(
        self, data_size: int, accuracy_target: float
    ) -> CompressionConfig:
        """Select optimal compression strategy based on conditions."""

        # Simple heuristics for strategy selection
        if data_size > self.bandwidth_budget * 1024 * 1024:  # Convert MB to bytes
            # Large data - use aggressive compression
            if accuracy_target > 0.95:
                # High accuracy needed - use quantization
                return CompressionConfig(
                    compression_type=CompressionType.QUANTIZATION,
                    quantization_bits=8,
                    enable_error_feedback=True,
                )
            else:
                # Can tolerate accuracy loss - use sparsification
                return CompressionConfig(
                    compression_type=CompressionType.SPARSIFICATION,
                    compression_ratio=0.05,  # Keep 5%
                    enable_error_feedback=True,
                )
        else:
            # Small data - use light compression
            return CompressionConfig(
                compression_type=CompressionType.QUANTIZATION,
                quantization_bits=16,
                enable_error_feedback=True,
            )

    def update_network_conditions(self, latency: float, bandwidth: float):
        """Update network condition estimates."""
        self.network_conditions["latency"] = latency
        self.network_conditions["bandwidth"] = bandwidth

    def record_performance(self, strategy: str, metrics: CommunicationMetrics):
        """Record performance of compression strategy."""
        if strategy not in self.performance_history:
            self.performance_history[strategy] = []

        self.performance_history[strategy].append(metrics.to_dict())

        # Keep only recent history
        if len(self.performance_history[strategy]) > 100:
            self.performance_history[strategy] = self.performance_history[strategy][
                -100:
            ]

    def get_strategy_performance(self, strategy: str) -> Dict[str, float]:
        """Get average performance metrics for a strategy."""
        if (
            strategy not in self.performance_history
            or not self.performance_history[strategy]
        ):
            return {}

        metrics = self.performance_history[strategy]

        return {
            "avg_compression_ratio": np.mean([m["compression_ratio"] for m in metrics]),
            "avg_compression_error": np.mean([m["compression_error"] for m in metrics]),
            "avg_compression_time": np.mean([m["compression_time"] for m in metrics]),
            "avg_bandwidth_utilization": np.mean(
                [m["bandwidth_utilization"] for m in metrics]
            ),
        }


# Utility functions for creating compression algorithms


def create_sparsification_compressor(
    method: SparsificationMethod = SparsificationMethod.TOP_K,
    compression_ratio: float = 0.1,
) -> GradientSparsification:
    """Create a gradient sparsification compressor."""
    config = CompressionConfig(
        compression_type=CompressionType.SPARSIFICATION,
        compression_ratio=compression_ratio,
        enable_error_feedback=True,
    )
    return GradientSparsification(config, method)


def create_quantization_compressor(
    method: QuantizationMethod = QuantizationMethod.UNIFORM, bits: int = 8
) -> Quantization:
    """Create a quantization compressor."""
    config = CompressionConfig(
        compression_type=CompressionType.QUANTIZATION,
        quantization_bits=bits,
        enable_error_feedback=True,
    )
    return Quantization(config, method)


def create_sketching_compressor(
    method: SketchingMethod = SketchingMethod.COUNT_SKETCH, sketch_size: int = 1024
) -> Sketching:
    """Create a sketching compressor."""
    config = CompressionConfig(
        compression_type=CompressionType.SKETCHING,
        sketch_size=sketch_size,
        enable_error_feedback=False,  # Sketching doesn't typically use error feedback
    )
    return Sketching(config, method)


def generate_mock_gradients(
    layers: List[Tuple[str, Tuple[int, ...]]], distribution: str = "normal"
) -> Dict[str, np.ndarray]:
    """Generate mock gradient data for testing."""
    gradients = {}

    for layer_name, shape in layers:
        if distribution == "normal":
            gradients[layer_name] = np.random.normal(0, 0.1, shape).astype(np.float32)
        elif distribution == "uniform":
            gradients[layer_name] = np.random.uniform(-0.5, 0.5, shape).astype(
                np.float32
            )
        elif distribution == "sparse":
            dense_grad = np.random.normal(0, 0.1, shape)
            # Make 90% of values zero
            mask = np.random.random(shape) < 0.1
            gradients[layer_name] = (dense_grad * mask).astype(np.float32)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    return gradients


if __name__ == "__main__":
    # Example usage and testing
    print("Communication-Efficient Protocols Demo")
    print("=" * 50)

    # Generate test data
    layers = [
        ("conv1.weight", (64, 3, 7, 7)),
        ("conv1.bias", (64,)),
        ("fc.weight", (1000, 2048)),
        ("fc.bias", (1000,)),
    ]

    gradients = generate_mock_gradients(layers, "normal")
    print(f"Generated gradients for {len(gradients)} layers")

    original_size = sum(tensor.nbytes for tensor in gradients.values())
    print(f"Original data size: {original_size / 1024:.2f} KB")

    # Test different compression methods
    compressors = [
        (
            "Top-K Sparsification",
            create_sparsification_compressor(SparsificationMethod.TOP_K, 0.1),
        ),
        (
            "Uniform Quantization",
            create_quantization_compressor(QuantizationMethod.UNIFORM, 8),
        ),
        (
            "Count Sketch",
            create_sketching_compressor(SketchingMethod.COUNT_SKETCH, 512),
        ),
    ]

    for name, compressor in compressors:
        print(f"\nTesting {name}:")

        # Compress
        result = compressor.compress(gradients)
        print(f"  Compressed size: {result.compressed_size / 1024:.2f} KB")
        print(f"  Compression ratio: {result.compression_efficiency:.3f}")
        print(f"  Compression error: {result.compression_error:.6f}")
        print(f"  Compression time: {result.compression_time:.3f}s")

        # Decompress
        reconstructed = compressor.decompress(result)
        print(f"  Decompression time: {result.decompression_time:.3f}s")

        # Verify shapes match
        for key in gradients.keys():
            if key in reconstructed:
                assert (
                    gradients[key].shape == reconstructed[key].shape
                ), f"Shape mismatch for {key}"

        print(f"  ✓ All shapes verified")
