"""
Model Serialization Framework for Federated Learning

Handles PyTorch and TensorFlow model weight extraction, serialization,
compression, and integrity verification for the federated learning system.

Design Principles:
- Support both PyTorch and TensorFlow models
- Unified ModelWeights interface
- Compression for efficient transfer
- Checksum verification for integrity
- Compatible with personal/family scale (2-3 participants)
"""

import hashlib
import pickle
import gzip
import json
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

try:
    import torch
    import torch.nn as nn

    HAS_PYTORCH = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_PYTORCH = False

try:
    import tensorflow as tf

    HAS_TENSORFLOW = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_TENSORFLOW = False

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelWeights:
    """
    Unified representation of model weights for federated learning.

    Supports both PyTorch and TensorFlow models with compression and integrity verification.
    """

    weights: Dict[str, np.ndarray]
    model_type: str  # 'pytorch' or 'tensorflow'
    architecture_info: Dict[str, Any]
    metadata: Dict[str, Any]
    checksum: str
    compressed: bool = False

    def __post_init__(self):
        """Validate the model weights after initialization."""
        if not self.weights:
            raise ValueError("Model weights cannot be empty")

        if self.model_type not in ["pytorch", "tensorflow"]:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Verify checksum if provided
        if self.checksum:
            calculated_checksum = self._calculate_checksum()
            if calculated_checksum != self.checksum:
                logger.warning(
                    f"Checksum mismatch: expected {self.checksum}, got {calculated_checksum}"
                )

    def _calculate_checksum(self) -> str:
        """Calculate SHA256 checksum of the weights."""
        # Create a deterministic representation of weights
        weights_bytes = pickle.dumps(self.weights, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(weights_bytes).hexdigest()

    def update_checksum(self):
        """Update the checksum after weights modification."""
        self.checksum = self._calculate_checksum()

    def compress(self) -> "ModelWeights":
        """Compress the model weights for efficient transfer."""
        if self.compressed:
            return self

        compressed_weights = {}
        for name, weight in self.weights.items():
            # Compress numpy array
            weight_bytes = weight.tobytes()
            compressed_bytes = gzip.compress(weight_bytes)
            compressed_weights[name] = {
                "data": compressed_bytes,
                "shape": weight.shape,
                "dtype": str(weight.dtype),
            }

        return ModelWeights(
            weights=compressed_weights,
            model_type=self.model_type,
            architecture_info=self.architecture_info,
            metadata=self.metadata,
            checksum=self.checksum,
            compressed=True,
        )

    def decompress(self) -> "ModelWeights":
        """Decompress the model weights."""
        if not self.compressed:
            return self

        decompressed_weights = {}
        for name, compressed_data in self.weights.items():
            # Decompress numpy array
            decompressed_bytes = gzip.decompress(compressed_data["data"])
            weight = np.frombuffer(decompressed_bytes, dtype=compressed_data["dtype"])
            weight = weight.reshape(compressed_data["shape"])
            decompressed_weights[name] = weight

        return ModelWeights(
            weights=decompressed_weights,
            model_type=self.model_type,
            architecture_info=self.architecture_info,
            metadata=self.metadata,
            checksum=self.checksum,
            compressed=False,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)

        # Convert numpy arrays to lists for JSON serialization
        if not self.compressed:
            weights_serializable = {}
            for name, weight in self.weights.items():
                weights_serializable[name] = {
                    "data": weight.tolist(),
                    "shape": weight.shape,
                    "dtype": str(weight.dtype),
                }
            data["weights"] = weights_serializable

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelWeights":
        """Create ModelWeights from dictionary."""
        # Convert serialized weights back to numpy arrays
        if not data.get("compressed", False):
            weights = {}
            for name, weight_data in data["weights"].items():
                weight_array = np.array(weight_data["data"], dtype=weight_data["dtype"])
                weight_array = weight_array.reshape(weight_data["shape"])
                weights[name] = weight_array
            data["weights"] = weights

        return cls(**data)

    def save(self, filepath: Union[str, Path]):
        """Save model weights to file."""
        filepath = Path(filepath)

        if filepath.suffix.lower() == ".json":
            # Save as JSON
            with open(filepath, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            # Save as pickle (more efficient for numpy arrays)
            with open(filepath, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ModelWeights":
        """Load model weights from file."""
        filepath = Path(filepath)

        if filepath.suffix.lower() == ".json":
            # Load from JSON
            with open(filepath, "r") as f:
                data = json.load(f)
            return cls.from_dict(data)
        else:
            # Load from pickle
            with open(filepath, "rb") as f:
                return pickle.load(f)

    @classmethod
    def from_pytorch_model(cls, model: "torch.nn.Module") -> "ModelWeights":
        """Create ModelWeights from PyTorch model."""
        if not HAS_PYTORCH:
            raise ImportError("PyTorch not available")

        weights = {}
        for name, param in model.named_parameters():
            weights[name] = param.detach().cpu().numpy()

        # Get model architecture info
        architecture_info = {
            "model_class": model.__class__.__name__,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "layer_names": list(dict(model.named_modules()).keys()),
        }

        model_weights = cls(
            weights=weights,
            model_type="pytorch",
            architecture_info=architecture_info,
            metadata={"created_from": "pytorch_model"},
            checksum="",
            compressed=False,
        )
        model_weights.update_checksum()
        return model_weights

    def to_pytorch_state_dict(self) -> Dict[str, "torch.Tensor"]:
        """Convert ModelWeights to PyTorch state dict."""
        if not HAS_PYTORCH:
            raise ImportError("PyTorch not available")

        if self.model_type != "pytorch":
            raise ValueError(f"Cannot convert {self.model_type} weights to PyTorch")

        # Decompress if needed
        weights = self.decompress() if self.compressed else self

        state_dict = {}
        for name, weight in weights.weights.items():
            state_dict[name] = torch.tensor(weight)

        return state_dict

    @classmethod
    def from_pytorch_state_dict(
        cls, state_dict: Dict[str, "torch.Tensor"]
    ) -> "ModelWeights":
        """Create ModelWeights from PyTorch state dict."""
        if not HAS_PYTORCH:
            raise ImportError("PyTorch not available")

        weights = {}
        for name, tensor in state_dict.items():
            weights[name] = tensor.detach().cpu().numpy()

        architecture_info = {
            "num_parameters": sum(tensor.numel() for tensor in state_dict.values()),
            "parameter_names": list(state_dict.keys()),
        }

        model_weights = cls(
            weights=weights,
            model_type="pytorch",
            architecture_info=architecture_info,
            metadata={"created_from": "pytorch_state_dict"},
            checksum="",
            compressed=False,
        )
        model_weights.update_checksum()
        return model_weights


class ModelSerializer:
    """
    Model serialization utilities for PyTorch and TensorFlow models.

    Extracts weights from trained models and creates ModelWeights objects
    suitable for federated learning operations.
    """

    @staticmethod
    def extract_pytorch_weights(
        model: "torch.nn.Module", include_metadata: bool = True
    ) -> ModelWeights:
        """
        Extract weights from PyTorch model.

        Args:
            model: PyTorch model
            include_metadata: Whether to include additional metadata

        Returns:
            ModelWeights object containing the model's state
        """
        if not HAS_PYTORCH:
            raise ImportError("PyTorch not available. Install with: pip install torch")

        # Extract state dict
        state_dict = model.state_dict()

        # Convert tensors to numpy arrays
        weights = {}
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                weights[name] = param.detach().cpu().numpy()
            else:
                weights[name] = np.array(param)

        # Extract architecture information
        architecture_info = {
            "model_class": model.__class__.__name__,
            "model_module": model.__class__.__module__,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        }

        # Additional metadata
        metadata = {}
        if include_metadata:
            metadata.update(
                {
                    "parameter_names": list(weights.keys()),
                    "weight_shapes": {
                        name: list(weight.shape) for name, weight in weights.items()
                    },
                    "total_size_bytes": sum(
                        weight.nbytes for weight in weights.values()
                    ),
                }
            )

        model_weights = ModelWeights(
            weights=weights,
            model_type="pytorch",
            architecture_info=architecture_info,
            metadata=metadata,
            checksum="",  # Will be calculated in __post_init__
        )

        model_weights.update_checksum()
        return model_weights

    @staticmethod
    def extract_tensorflow_weights(
        model: "tf.keras.Model", include_metadata: bool = True
    ) -> ModelWeights:
        """
        Extract weights from TensorFlow/Keras model.

        Args:
            model: TensorFlow/Keras model
            include_metadata: Whether to include additional metadata

        Returns:
            ModelWeights object containing the model's state
        """
        if not HAS_TENSORFLOW:
            raise ImportError(
                "TensorFlow not available. Install with: pip install tensorflow"
            )

        # Extract weights
        weights = {}
        for i, layer in enumerate(model.layers):
            layer_weights = layer.get_weights()
            for j, weight in enumerate(layer_weights):
                weight_name = (
                    f"{layer.name}_weight_{j}"
                    if len(layer_weights) > 1
                    else f"{layer.name}_weight"
                )
                weights[weight_name] = weight

        # Extract architecture information
        architecture_info = {
            "model_class": model.__class__.__name__,
            "model_module": model.__class__.__module__,
            "num_layers": len(model.layers),
            "num_parameters": model.count_params(),
            "input_shape": model.input_shape if hasattr(model, "input_shape") else None,
            "output_shape": (
                model.output_shape if hasattr(model, "output_shape") else None
            ),
        }

        # Additional metadata
        metadata = {}
        if include_metadata:
            metadata.update(
                {
                    "layer_names": [layer.name for layer in model.layers],
                    "weight_shapes": {
                        name: list(weight.shape) for name, weight in weights.items()
                    },
                    "total_size_bytes": sum(
                        weight.nbytes for weight in weights.values()
                    ),
                }
            )

        model_weights = ModelWeights(
            weights=weights,
            model_type="tensorflow",
            architecture_info=architecture_info,
            metadata=metadata,
            checksum="",  # Will be calculated in __post_init__
        )

        model_weights.update_checksum()
        return model_weights

    @staticmethod
    def load_pytorch_weights(
        model_weights: ModelWeights, model: "torch.nn.Module"
    ) -> "torch.nn.Module":
        """
        Load weights into PyTorch model.

        Args:
            model_weights: ModelWeights object
            model: PyTorch model to load weights into

        Returns:
            Model with loaded weights
        """
        if not HAS_PYTORCH:
            raise ImportError("PyTorch not available. Install with: pip install torch")

        if model_weights.model_type != "pytorch":
            raise ValueError(
                f"Expected PyTorch weights, got {model_weights.model_type}"
            )

        # Ensure weights are decompressed
        if model_weights.compressed:
            model_weights = model_weights.decompress()

        # Convert numpy arrays back to tensors
        state_dict = {}
        for name, weight in model_weights.weights.items():
            state_dict[name] = torch.from_numpy(weight)

        # Load state dict
        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def load_tensorflow_weights(
        model_weights: ModelWeights, model: "tf.keras.Model"
    ) -> "tf.keras.Model":
        """
        Load weights into TensorFlow/Keras model.

        Args:
            model_weights: ModelWeights object
            model: TensorFlow/Keras model to load weights into

        Returns:
            Model with loaded weights
        """
        if not HAS_TENSORFLOW:
            raise ImportError(
                "TensorFlow not available. Install with: pip install tensorflow"
            )

        if model_weights.model_type != "tensorflow":
            raise ValueError(
                f"Expected TensorFlow weights, got {model_weights.model_type}"
            )

        # Ensure weights are decompressed
        if model_weights.compressed:
            model_weights = model_weights.decompress()

        # Group weights by layer
        layer_weights = {}
        for weight_name, weight in model_weights.weights.items():
            # Extract layer name (everything before the last "_weight")
            parts = weight_name.split("_weight")
            if len(parts) > 1:
                layer_name = "_weight".join(parts[:-1])
                weight_index = int(parts[-1]) if parts[-1] else 0
            else:
                layer_name = weight_name
                weight_index = 0

            if layer_name not in layer_weights:
                layer_weights[layer_name] = {}
            layer_weights[layer_name][weight_index] = weight

        # Load weights into layers
        for layer in model.layers:
            if layer.name in layer_weights:
                weights_list = []
                layer_weight_dict = layer_weights[layer.name]
                for i in sorted(layer_weight_dict.keys()):
                    weights_list.append(layer_weight_dict[i])
                layer.set_weights(weights_list)

        return model

    # ------------------------------------------------------------------
    # Model inspection & compatibility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def inspect_model_architecture(model: Any) -> Dict[str, Any]:
        """Generate a lightweight description of a model architecture.

        The resulting dictionary is fully JSON-serialisable, making it safe
        to share across participants without exposing raw weights.
        """

        if HAS_PYTORCH and isinstance(model, nn.Module):
            return ModelSerializer._inspect_pytorch_model(model)
        if HAS_TENSORFLOW and isinstance(model, tf.keras.Model):
            return ModelSerializer._inspect_tensorflow_model(model)

        raise TypeError(
            "Unsupported model type for inspection. Provide a PyTorch module "
            "or a TensorFlow/Keras model."
        )

    @staticmethod
    def compare_architecture_info(
        local_info: Dict[str, Any], reference_info: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Compare two architecture descriptors.

        Returns a tuple ``(is_compatible, details)`` where ``details`` contains
        any mismatches that were detected. The comparison is intentionally
        strict—if hashes differ we immediately surface a mismatch—yet the
        method remains descriptive so callers can present helpful diagnostics
        to users.
        """

        details: Dict[str, Any] = {"mismatches": []}

        if not local_info or not reference_info:
            details["mismatches"].append("missing_architecture_info")
            return False, details

        if local_info.get("model_type") != reference_info.get("model_type"):
            details["mismatches"].append(
                {
                    "field": "model_type",
                    "local": local_info.get("model_type"),
                    "reference": reference_info.get("model_type"),
                }
            )

        if local_info.get("architecture_hash") == reference_info.get(
            "architecture_hash"
        ):
            is_compatible = len(details["mismatches"]) == 0
            return is_compatible, details

        comparable_fields = [
            "model_class",
            "module",
            "layer_count",
            "parameter_count",
            "trainable_parameter_count",
        ]
        for field in comparable_fields:
            if local_info.get(field) != reference_info.get(field):
                details["mismatches"].append(
                    {
                        "field": field,
                        "local": local_info.get(field),
                        "reference": reference_info.get(field),
                    }
                )

        # Compare parameter shapes
        local_shapes = local_info.get("parameter_shapes", {})
        reference_shapes = reference_info.get("parameter_shapes", {})
        if set(local_shapes.keys()) != set(reference_shapes.keys()):
            details["mismatches"].append(
                {
                    "field": "parameter_names",
                    "local": sorted(local_shapes.keys()),
                    "reference": sorted(reference_shapes.keys()),
                }
            )
        else:
            for name in local_shapes.keys():
                if local_shapes[name] != reference_shapes[name]:
                    details["mismatches"].append(
                        {
                            "field": f"parameter_shape::{name}",
                            "local": local_shapes[name],
                            "reference": reference_shapes[name],
                        }
                    )

        is_compatible = len(details["mismatches"]) == 0
        return is_compatible, details

    # ----------------------------
    # Internal helpers
    # ----------------------------

    @staticmethod
    def _inspect_pytorch_model(model: "nn.Module") -> Dict[str, Any]:
        import inspect

        parameter_shapes = {
            name: list(param.shape)
            for name, param in model.named_parameters(recurse=True)
        }

        layer_types = []
        for name, module in model.named_modules():
            if name == "":
                continue  # Skip the top-level container entry
            layer_types.append({"name": name, "type": module.__class__.__name__})

        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in model.parameters())

        info = {
            "model_type": "pytorch",
            "model_class": model.__class__.__name__,
            "module": model.__class__.__module__,
            "layer_count": len(layer_types),
            "layer_types": layer_types,
            "parameter_count": int(total_params),
            "trainable_parameter_count": int(trainable_params),
            "parameter_shapes": parameter_shapes,
        }

        info["architecture_hash"] = _hash_architecture(info)
        info["forward_signature"] = str(
            inspect.signature(model.forward)  # type: ignore[attr-defined]
        )
        return info

    @staticmethod
    def _inspect_tensorflow_model(model: "tf.keras.Model") -> Dict[str, Any]:
        config = model.get_config()
        layer_types = [
            {"name": layer.get("name"), "type": layer.get("class_name")}
            for layer in config.get("layers", [])
        ]

        parameter_shapes = {}
        for weight in model.weights:
            parameter_shapes[weight.name] = list(weight.shape.as_list())

        def _count_parameters(weights: List["tf.Variable"]) -> int:
            total = 0
            for tensor in weights:
                shape = tensor.shape.as_list()
                if None in shape:
                    continue
                total += int(np.prod(shape))
            return total

        trainable_params = _count_parameters(model.trainable_weights)
        total_params = _count_parameters(model.weights)

        info = {
            "model_type": "tensorflow",
            "model_class": model.__class__.__name__,
            "module": model.__class__.__module__,
            "layer_count": len(layer_types),
            "layer_types": layer_types,
            "parameter_count": total_params,
            "trainable_parameter_count": trainable_params,
            "parameter_shapes": parameter_shapes,
        }

        info["architecture_hash"] = _hash_architecture(info)
        return info


def create_test_pytorch_model() -> Optional["torch.nn.Module"]:
    """Create a simple PyTorch model for testing."""
    if not HAS_PYTORCH:
        return None

    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(10, 5)
            self.fc2 = nn.Linear(5, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    return SimpleNet()


def create_test_tensorflow_model() -> Optional["tf.keras.Model"]:
    """Create a simple TensorFlow model for testing."""
    if not HAS_TENSORFLOW:
        return None

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(5, activation="relu", input_shape=(10,)),
            tf.keras.layers.Dense(1),
        ]
    )

    return model


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _hash_architecture(architecture_info: Dict[str, Any]) -> str:
    """Create a reproducible hash for the architecture description."""

    try:
        serialisable = json.dumps(architecture_info, sort_keys=True, default=list)
    except TypeError:
        # Fallback: convert non-serialisable objects to string representations.
        normalised = {}
        for key, value in architecture_info.items():
            try:
                normalised[key] = json.loads(json.dumps(value, default=list))
            except TypeError:
                normalised[key] = str(value)
        serialisable = json.dumps(normalised, sort_keys=True)

    return hashlib.sha256(serialisable.encode("utf-8")).hexdigest()


# Example usage and testing
if __name__ == "__main__":
    # Test PyTorch model serialization
    if HAS_PYTORCH:
        print("Testing PyTorch model serialization...")
        model = create_test_pytorch_model()
        weights = ModelSerializer.extract_pytorch_weights(model)
        print(f"Extracted weights: {list(weights.weights.keys())}")
        print(f"Checksum: {weights.checksum}")
        print(f"Compressed size: {len(weights.compress().weights)} items")

    # Test TensorFlow model serialization
    if HAS_TENSORFLOW:
        print("\nTesting TensorFlow model serialization...")
        model = create_test_tensorflow_model()
        weights = ModelSerializer.extract_tensorflow_weights(model)
        print(f"Extracted weights: {list(weights.weights.keys())}")
        print(f"Checksum: {weights.checksum}")
        print(f"Compressed size: {len(weights.compress().weights)} items")
