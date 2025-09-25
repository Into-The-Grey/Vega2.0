"""
Federated Model Pruning Implementation
=====================================

This module provides comprehensive model pruning capabilities for federated learning,
including structured pruning, unstructured pruning, federated knowledge distillation,
and sparsity-aware aggregation algorithms.

Features:
- Structured pruning (channel, filter, layer removal)
- Unstructured pruning (weight magnitude, gradient-based)
- Federated knowledge distillation for compressed models
- Sparsity-aware aggregation algorithms
- Performance monitoring and recovery mechanisms
- Adaptive pruning based on participant constraints

Author: Vega2.0 Federated Learning Team
Date: September 2025
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import json
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PruningType(Enum):
    """Types of pruning strategies available."""

    STRUCTURED_CHANNEL = "structured_channel"
    STRUCTURED_FILTER = "structured_filter"
    STRUCTURED_LAYER = "structured_layer"
    UNSTRUCTURED_MAGNITUDE = "unstructured_magnitude"
    UNSTRUCTURED_GRADIENT = "unstructured_gradient"
    UNSTRUCTURED_RANDOM = "unstructured_random"
    HYBRID_STRUCTURED_UNSTRUCTURED = "hybrid_structured_unstructured"


class SparsitySchedule(Enum):
    """Sparsity scheduling strategies."""

    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    POLYNOMIAL = "polynomial"
    COSINE = "cosine"
    ADAPTIVE = "adaptive"


@dataclass
class PruningConfig:
    """Configuration for pruning operations."""

    pruning_type: PruningType
    target_sparsity: float
    schedule: SparsitySchedule = SparsitySchedule.LINEAR
    pruning_frequency: int = 5  # Every N federated rounds
    recovery_threshold: float = 0.05  # Accuracy drop threshold for recovery
    distillation_enabled: bool = True
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.7
    min_sparsity: float = 0.0
    max_sparsity: float = 0.95

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.0 <= self.target_sparsity <= 1.0:
            raise ValueError("target_sparsity must be between 0.0 and 1.0")
        if not 0.0 <= self.recovery_threshold <= 1.0:
            raise ValueError("recovery_threshold must be between 0.0 and 1.0")
        if self.pruning_frequency <= 0:
            raise ValueError("pruning_frequency must be positive")


@dataclass
class PruningMetrics:
    """Metrics tracking for pruning operations."""

    sparsity_ratio: float
    model_size_reduction: float
    accuracy_before: float
    accuracy_after: float
    accuracy_drop: float
    compression_time: float
    inference_speedup: float
    memory_reduction: float
    flops_reduction: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "sparsity_ratio": self.sparsity_ratio,
            "model_size_reduction": self.model_size_reduction,
            "accuracy_before": self.accuracy_before,
            "accuracy_after": self.accuracy_after,
            "accuracy_drop": self.accuracy_drop,
            "compression_time": self.compression_time,
            "inference_speedup": self.inference_speedup,
            "memory_reduction": self.memory_reduction,
            "flops_reduction": self.flops_reduction,
            "timestamp": self.timestamp,
        }


class BasePruning(ABC):
    """Abstract base class for pruning algorithms."""

    def __init__(self, config: PruningConfig):
        self.config = config
        self.metrics_history: List[PruningMetrics] = []

    @abstractmethod
    async def prune_model(
        self, model: nn.Module, round_num: int
    ) -> Tuple[nn.Module, PruningMetrics]:
        """Prune the model based on the pruning strategy."""
        pass

    @abstractmethod
    def calculate_sparsity(self, model: nn.Module) -> float:
        """Calculate current sparsity ratio of the model."""
        pass

    def get_current_sparsity_target(self, round_num: int, total_rounds: int) -> float:
        """Calculate current sparsity target based on schedule."""
        if self.config.schedule == SparsitySchedule.CONSTANT:
            return self.config.target_sparsity

        progress = min(round_num / total_rounds, 1.0)

        if self.config.schedule == SparsitySchedule.LINEAR:
            return self.config.min_sparsity + progress * (
                self.config.target_sparsity - self.config.min_sparsity
            )

        elif self.config.schedule == SparsitySchedule.EXPONENTIAL:
            return self.config.target_sparsity * (1 - np.exp(-5 * progress))

        elif self.config.schedule == SparsitySchedule.POLYNOMIAL:
            return self.config.target_sparsity * (progress**3)

        elif self.config.schedule == SparsitySchedule.COSINE:
            return self.config.target_sparsity * (1 - np.cos(progress * np.pi / 2))

        else:  # ADAPTIVE - implemented by specific pruning classes
            return self.config.target_sparsity


class StructuredPruning(BasePruning):
    """Structured pruning implementation for channels, filters, and layers."""

    def __init__(self, config: PruningConfig):
        super().__init__(config)
        self.importance_scores: Dict[str, torch.Tensor] = {}

    async def prune_model(
        self, model: nn.Module, round_num: int, total_rounds: int = 100
    ) -> Tuple[nn.Module, PruningMetrics]:
        """Perform structured pruning on the model."""
        start_time = time.time()

        # Calculate current sparsity target
        current_target = self.get_current_sparsity_target(round_num, total_rounds)

        # Get model metrics before pruning
        accuracy_before = await self._evaluate_model_accuracy(model)
        size_before = self._calculate_model_size(model)
        flops_before = self._calculate_flops(model)

        # Perform structured pruning based on type
        if self.config.pruning_type == PruningType.STRUCTURED_CHANNEL:
            pruned_model = await self._prune_channels(model, current_target)
        elif self.config.pruning_type == PruningType.STRUCTURED_FILTER:
            pruned_model = await self._prune_filters(model, current_target)
        elif self.config.pruning_type == PruningType.STRUCTURED_LAYER:
            pruned_model = await self._prune_layers(model, current_target)
        else:
            raise ValueError(
                f"Unsupported structured pruning type: {self.config.pruning_type}"
            )

        # Calculate metrics after pruning
        accuracy_after = await self._evaluate_model_accuracy(pruned_model)
        size_after = self._calculate_model_size(pruned_model)
        flops_after = self._calculate_flops(pruned_model)
        compression_time = time.time() - start_time

        # Create metrics
        metrics = PruningMetrics(
            sparsity_ratio=self.calculate_sparsity(pruned_model),
            model_size_reduction=(size_before - size_after) / size_before,
            accuracy_before=accuracy_before,
            accuracy_after=accuracy_after,
            accuracy_drop=accuracy_before - accuracy_after,
            compression_time=compression_time,
            inference_speedup=flops_before / flops_after if flops_after > 0 else 1.0,
            memory_reduction=(size_before - size_after) / size_before,
            flops_reduction=(
                (flops_before - flops_after) / flops_before if flops_before > 0 else 0.0
            ),
        )

        self.metrics_history.append(metrics)
        logger.info(
            f"Structured pruning completed: {metrics.sparsity_ratio:.3f} sparsity, "
            f"{metrics.accuracy_drop:.3f} accuracy drop"
        )

        return pruned_model, metrics

    async def _prune_channels(
        self, model: nn.Module, target_sparsity: float
    ) -> nn.Module:
        """Prune channels based on importance scores."""
        # Calculate channel importance scores
        channel_scores = await self._calculate_channel_importance(model)

        # Create a copy of the model for pruning
        pruned_model = self._deep_copy_model(model)

        for name, module in pruned_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if name in channel_scores:
                    scores = channel_scores[name]
                    num_channels = scores.size(0)
                    num_to_prune = int(num_channels * target_sparsity)

                    if num_to_prune > 0:
                        # Get indices of channels to remove
                        _, indices_to_remove = torch.topk(
                            scores, num_to_prune, largest=False
                        )

                        # Create mask for remaining channels
                        mask = torch.ones(num_channels, dtype=torch.bool)
                        mask[indices_to_remove] = False

                        # Prune the layer
                        if isinstance(module, nn.Conv2d):
                            module.weight.data = module.weight.data[mask]
                            if module.bias is not None:
                                module.bias.data = module.bias.data[mask]
                        elif isinstance(module, nn.Linear):
                            module.weight.data = module.weight.data[mask]
                            if module.bias is not None:
                                module.bias.data = module.bias.data[mask]

        return pruned_model

    async def _prune_filters(
        self, model: nn.Module, target_sparsity: float
    ) -> nn.Module:
        """Prune filters based on importance scores."""
        # Calculate filter importance scores
        filter_scores = await self._calculate_filter_importance(model)

        # Create a copy of the model for pruning
        pruned_model = self._deep_copy_model(model)

        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Conv2d):
                if name in filter_scores:
                    scores = filter_scores[name]
                    num_filters = scores.size(0)
                    num_to_prune = int(num_filters * target_sparsity)

                    if num_to_prune > 0:
                        # Get indices of filters to remove
                        _, indices_to_remove = torch.topk(
                            scores, num_to_prune, largest=False
                        )

                        # Create mask for remaining filters
                        mask = torch.ones(num_filters, dtype=torch.bool)
                        mask[indices_to_remove] = False

                        # Prune the filters
                        module.weight.data = module.weight.data[mask]
                        if module.bias is not None:
                            module.bias.data = module.bias.data[mask]

        return pruned_model

    async def _prune_layers(
        self, model: nn.Module, target_sparsity: float
    ) -> nn.Module:
        """Prune entire layers based on importance scores."""
        # Calculate layer importance scores
        layer_scores = await self._calculate_layer_importance(model)

        # Determine which layers to remove
        layer_names = list(layer_scores.keys())
        scores = torch.tensor([layer_scores[name] for name in layer_names])
        num_layers = len(layer_names)
        num_to_prune = int(num_layers * target_sparsity)

        if num_to_prune > 0:
            _, indices_to_remove = torch.topk(scores, num_to_prune, largest=False)
            layers_to_remove = [layer_names[idx] for idx in indices_to_remove]

            # Create new model without pruned layers
            pruned_model = self._remove_layers(model, layers_to_remove)
        else:
            pruned_model = self._deep_copy_model(model)

        return pruned_model

    async def _calculate_channel_importance(
        self, model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """Calculate importance scores for channels."""
        importance_scores = {}

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weights = module.weight.data

                if isinstance(module, nn.Conv2d):
                    # For conv layers, calculate L1 norm across spatial dimensions
                    scores = torch.norm(weights.view(weights.size(0), -1), p=1, dim=1)
                else:  # Linear layer
                    # For linear layers, calculate L1 norm of weights
                    scores = torch.norm(weights, p=1, dim=1)

                importance_scores[name] = scores

        return importance_scores

    async def _calculate_filter_importance(
        self, model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """Calculate importance scores for filters."""
        importance_scores = {}

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                weights = module.weight.data
                # Calculate L2 norm for each filter
                scores = torch.norm(weights.view(weights.size(0), -1), p=2, dim=1)
                importance_scores[name] = scores

        return importance_scores

    async def _calculate_layer_importance(self, model: nn.Module) -> Dict[str, float]:
        """Calculate importance scores for entire layers."""
        importance_scores = {}

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                # Calculate average magnitude of weights
                if hasattr(module, "weight") and module.weight is not None:
                    avg_magnitude = torch.mean(torch.abs(module.weight.data)).item()
                    importance_scores[name] = avg_magnitude
                else:
                    importance_scores[name] = 0.0

        return importance_scores

    def calculate_sparsity(self, model: nn.Module) -> float:
        """Calculate overall sparsity ratio of the model."""
        total_params = 0
        zero_params = 0

        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param.data == 0).sum().item()

        return zero_params / total_params if total_params > 0 else 0.0

    def _deep_copy_model(self, model: nn.Module) -> nn.Module:
        """Create a deep copy of the model."""
        # Simple implementation - in practice, would use proper model cloning
        import copy

        return copy.deepcopy(model)

    def _remove_layers(self, model: nn.Module, layer_names: List[str]) -> nn.Module:
        """Remove specified layers from the model."""
        # Simplified implementation - would need model-specific logic
        pruned_model = self._deep_copy_model(model)

        for layer_name in layer_names:
            # Remove the layer (simplified - would need proper graph modification)
            if hasattr(pruned_model, layer_name):
                delattr(pruned_model, layer_name)

        return pruned_model

    def _calculate_model_size(self, model: nn.Module) -> int:
        """Calculate model size in bytes."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size

    def _calculate_flops(self, model: nn.Module) -> int:
        """Estimate FLOPs for the model (simplified implementation)."""
        flops = 0
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                # Simplified FLOP calculation for conv layers
                kernel_ops = module.kernel_size[0] * module.kernel_size[1]
                output_elements = (
                    224 * 224 // (module.stride[0] * module.stride[1])
                )  # Assume 224x224 input
                flops += (
                    kernel_ops
                    * module.in_channels
                    * module.out_channels
                    * output_elements
                )
            elif isinstance(module, nn.Linear):
                flops += module.in_features * module.out_features
        return flops

    async def _evaluate_model_accuracy(self, model: nn.Module) -> float:
        """Evaluate model accuracy (simplified implementation)."""
        # In practice, would use proper validation dataset
        # For now, return a simulated accuracy
        sparsity = self.calculate_sparsity(model)
        base_accuracy = 0.85
        accuracy_drop = sparsity * 0.1  # Simplified relationship
        return max(base_accuracy - accuracy_drop, 0.0)


class UnstructuredPruning(BasePruning):
    """Unstructured pruning implementation for weight-level sparsity."""

    def __init__(self, config: PruningConfig):
        super().__init__(config)
        self.weight_masks: Dict[str, torch.Tensor] = {}

    async def prune_model(
        self, model: nn.Module, round_num: int, total_rounds: int = 100
    ) -> Tuple[nn.Module, PruningMetrics]:
        """Perform unstructured pruning on the model."""
        start_time = time.time()

        # Calculate current sparsity target
        current_target = self.get_current_sparsity_target(round_num, total_rounds)

        # Get model metrics before pruning
        accuracy_before = await self._evaluate_model_accuracy(model)
        size_before = self._calculate_model_size(model)

        # Perform unstructured pruning based on type
        if self.config.pruning_type == PruningType.UNSTRUCTURED_MAGNITUDE:
            pruned_model = await self._magnitude_pruning(model, current_target)
        elif self.config.pruning_type == PruningType.UNSTRUCTURED_GRADIENT:
            pruned_model = await self._gradient_pruning(model, current_target)
        elif self.config.pruning_type == PruningType.UNSTRUCTURED_RANDOM:
            pruned_model = await self._random_pruning(model, current_target)
        else:
            raise ValueError(
                f"Unsupported unstructured pruning type: {self.config.pruning_type}"
            )

        # Calculate metrics after pruning
        accuracy_after = await self._evaluate_model_accuracy(pruned_model)
        size_after = self._calculate_model_size(pruned_model)
        compression_time = time.time() - start_time
        sparsity_ratio = self.calculate_sparsity(pruned_model)

        # Create metrics
        metrics = PruningMetrics(
            sparsity_ratio=sparsity_ratio,
            model_size_reduction=(size_before - size_after) / size_before,
            accuracy_before=accuracy_before,
            accuracy_after=accuracy_after,
            accuracy_drop=accuracy_before - accuracy_after,
            compression_time=compression_time,
            inference_speedup=1.0
            + sparsity_ratio * 0.5,  # Simplified speedup calculation
            memory_reduction=(size_before - size_after) / size_before,
            flops_reduction=sparsity_ratio,
        )

        self.metrics_history.append(metrics)
        logger.info(
            f"Unstructured pruning completed: {metrics.sparsity_ratio:.3f} sparsity, "
            f"{metrics.accuracy_drop:.3f} accuracy drop"
        )

        return pruned_model, metrics

    async def _magnitude_pruning(
        self, model: nn.Module, target_sparsity: float
    ) -> nn.Module:
        """Perform magnitude-based pruning."""
        pruned_model = self._deep_copy_model(model)

        # Collect all weights and their magnitudes
        all_weights = []
        weight_locations = []

        for name, module in pruned_model.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                weights = module.weight.data.view(-1)
                magnitudes = torch.abs(weights)

                all_weights.append(magnitudes)
                weight_locations.extend([(name, i) for i in range(len(weights))])

        # Concatenate all weights
        all_weights = torch.cat(all_weights)

        # Calculate global threshold
        num_weights = len(all_weights)
        num_to_prune = int(num_weights * target_sparsity)

        if num_to_prune > 0:
            # Use correct index for kthvalue (1-indexed)
            threshold = torch.kthvalue(all_weights, num_to_prune + 1).values

            # Apply pruning masks
            for name, module in pruned_model.named_modules():
                if hasattr(module, "weight") and module.weight is not None:
                    mask = torch.abs(module.weight.data) > threshold
                    module.weight.data *= mask.float()
                    self.weight_masks[name] = mask

        return pruned_model

    async def _gradient_pruning(
        self, model: nn.Module, target_sparsity: float
    ) -> nn.Module:
        """Perform gradient-based pruning."""
        # Note: This is a simplified implementation
        # In practice, would need access to gradient information
        pruned_model = self._deep_copy_model(model)

        # Simulate gradient-based importance (using weight magnitude as proxy)
        for name, module in pruned_model.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                weights = module.weight.data
                # Simulate gradient importance using weight magnitude
                importance = torch.abs(weights)

                # Calculate threshold for this layer
                num_weights = weights.numel()
                num_to_prune = int(num_weights * target_sparsity)

                if num_to_prune > 0:
                    threshold = torch.kthvalue(importance.view(-1), num_to_prune).values
                    mask = importance >= threshold
                    module.weight.data *= mask.float()
                    self.weight_masks[name] = mask

        return pruned_model

    async def _random_pruning(
        self, model: nn.Module, target_sparsity: float
    ) -> nn.Module:
        """Perform random pruning."""
        pruned_model = self._deep_copy_model(model)

        for name, module in pruned_model.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                weights = module.weight.data

                # Create random mask
                mask = torch.rand_like(weights) >= target_sparsity
                module.weight.data *= mask.float()
                self.weight_masks[name] = mask

        return pruned_model

    def calculate_sparsity(self, model: nn.Module) -> float:
        """Calculate overall sparsity ratio of the model."""
        total_params = 0
        zero_params = 0

        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param.data == 0).sum().item()

        return zero_params / total_params if total_params > 0 else 0.0

    def _deep_copy_model(self, model: nn.Module) -> nn.Module:
        """Create a deep copy of the model."""
        import copy

        return copy.deepcopy(model)

    def _calculate_model_size(self, model: nn.Module) -> int:
        """Calculate model size in bytes."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size

    async def _evaluate_model_accuracy(self, model: nn.Module) -> float:
        """Evaluate model accuracy (simplified implementation)."""
        sparsity = self.calculate_sparsity(model)
        base_accuracy = 0.85
        accuracy_drop = (
            sparsity * 0.15
        )  # Unstructured pruning typically causes more accuracy drop
        return max(base_accuracy - accuracy_drop, 0.0)


class FederatedDistillation:
    """Federated knowledge distillation for pruned models."""

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        self.temperature = temperature
        self.alpha = alpha
        self.distillation_history: List[Dict[str, Any]] = []

    async def distill_knowledge(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        data_loader: Optional[Any] = None,
        num_epochs: int = 5,
    ) -> Tuple[nn.Module, Dict[str, float]]:
        """Perform knowledge distillation from teacher to student model."""
        start_time = time.time()

        # Set models to appropriate modes
        teacher_model.eval()
        student_model.train()

        # Initialize optimizer for student
        optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)

        total_loss = 0.0
        num_batches = 0

        # Simulate training loop (in practice, would use real data loader)
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            # Simulate batch processing
            for batch_idx in range(10):  # Simulate 10 batches per epoch
                # Generate synthetic data for demonstration
                batch_size = 32
                # Use proper input size for the model
                if hasattr(student_model, "fc1"):
                    # For models with fc1, determine input size from first layer
                    input_features = student_model.fc1.in_features
                    inputs = torch.randn(batch_size, input_features)
                else:
                    # Default to image input
                    inputs = torch.randn(batch_size, 3, 224, 224)
                targets = torch.randint(0, 10, (batch_size,))

                optimizer.zero_grad()

                # Get teacher and student outputs
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)

                student_outputs = student_model(inputs)

                # Calculate distillation loss
                distillation_loss = self._calculate_distillation_loss(
                    student_outputs, teacher_outputs, targets
                )

                distillation_loss.backward()
                optimizer.step()

                epoch_loss += distillation_loss.item()
                num_batches += 1

            total_loss += epoch_loss
            logger.info(
                f"Distillation Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}"
            )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        distillation_time = time.time() - start_time

        # Calculate final accuracy (simplified)
        student_accuracy = await self._evaluate_student_accuracy(student_model)
        teacher_accuracy = await self._evaluate_teacher_accuracy(teacher_model)

        metrics = {
            "distillation_loss": avg_loss,
            "distillation_time": distillation_time,
            "student_accuracy": student_accuracy,
            "teacher_accuracy": teacher_accuracy,
            "knowledge_retention": (
                student_accuracy / teacher_accuracy if teacher_accuracy > 0 else 0.0
            ),
            "num_epochs": num_epochs,
            "num_batches": num_batches,
        }

        self.distillation_history.append(metrics)

        logger.info(
            f"Knowledge distillation completed: {student_accuracy:.3f} student accuracy, "
            f"{metrics['knowledge_retention']:.3f} knowledge retention"
        )

        return student_model, metrics

    def _calculate_distillation_loss(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate knowledge distillation loss."""
        # Soft targets from teacher
        soft_targets = F.softmax(teacher_outputs / self.temperature, dim=1)
        soft_student = F.log_softmax(student_outputs / self.temperature, dim=1)

        # KL divergence loss for knowledge distillation
        distillation_loss = F.kl_div(soft_student, soft_targets, reduction="batchmean")
        distillation_loss *= self.temperature**2

        # Hard target loss
        hard_loss = F.cross_entropy(student_outputs, targets)

        # Combine losses
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss

        return total_loss

    async def _evaluate_student_accuracy(self, student_model: nn.Module) -> float:
        """Evaluate student model accuracy."""
        # Simplified accuracy calculation
        return 0.75  # Placeholder

    async def _evaluate_teacher_accuracy(self, teacher_model: nn.Module) -> float:
        """Evaluate teacher model accuracy."""
        # Simplified accuracy calculation
        return 0.85  # Placeholder


class SparsityAggregator:
    """Sparsity-aware aggregation for federated learning with pruned models."""

    def __init__(self):
        self.aggregation_history: List[Dict[str, Any]] = []

    async def aggregate_sparse_models(
        self,
        participant_models: List[
            Tuple[nn.Module, float, Dict[str, torch.Tensor]]
        ],  # (model, weight, mask)
        global_model: nn.Module,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Aggregate sparse models from participants."""
        start_time = time.time()

        if not participant_models:
            return global_model, {}

        # Initialize aggregated parameters
        aggregated_params = {}
        total_weight = 0.0

        # Collect sparsity statistics
        sparsity_stats = []

        for model, weight, mask in participant_models:
            sparsity = self._calculate_model_sparsity(model)
            sparsity_stats.append(sparsity)
            total_weight += weight

        # Normalize weights
        if total_weight > 0:
            normalized_weights = [w / total_weight for _, w, _ in participant_models]
        else:
            normalized_weights = [1.0 / len(participant_models)] * len(
                participant_models
            )

        # Aggregate parameters with sparsity awareness
        for name, param in global_model.named_parameters():
            aggregated_param = torch.zeros_like(param.data)

            for i, (model, _, mask) in enumerate(participant_models):
                participant_param = dict(model.named_parameters())[name]

                # Apply sparsity-aware aggregation
                if name in mask:
                    # Use mask to aggregate only non-zero parameters
                    participant_param = participant_param * mask[name].float()

                aggregated_param += normalized_weights[i] * participant_param

            aggregated_params[name] = aggregated_param

        # Update global model
        for name, param in global_model.named_parameters():
            if name in aggregated_params:
                param.data = aggregated_params[name]

        aggregation_time = time.time() - start_time

        # Calculate aggregation metrics
        avg_sparsity = np.mean(sparsity_stats)
        sparsity_variance = np.var(sparsity_stats)

        metrics = {
            "aggregation_time": aggregation_time,
            "num_participants": len(participant_models),
            "average_sparsity": avg_sparsity,
            "sparsity_variance": sparsity_variance,
            "min_sparsity": min(sparsity_stats) if sparsity_stats else 0.0,
            "max_sparsity": max(sparsity_stats) if sparsity_stats else 0.0,
            "total_weight": total_weight,
            "effective_participants": sum(1 for w in normalized_weights if w > 0.01),
        }

        self.aggregation_history.append(metrics)

        logger.info(
            f"Sparse model aggregation completed: {avg_sparsity:.3f} avg sparsity, "
            f"{len(participant_models)} participants"
        )

        return global_model, metrics

    def _calculate_model_sparsity(self, model: nn.Module) -> float:
        """Calculate sparsity ratio of a model."""
        total_params = 0
        zero_params = 0

        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param.data == 0).sum().item()

        return zero_params / total_params if total_params > 0 else 0.0


class PruningCoordinator:
    """Main coordinator for federated model pruning operations."""

    def __init__(self, config: PruningConfig):
        self.config = config
        self.structured_pruner = (
            StructuredPruning(config)
            if config.pruning_type.value.startswith("structured")
            else None
        )
        self.unstructured_pruner = (
            UnstructuredPruning(config)
            if config.pruning_type.value.startswith("unstructured")
            else None
        )
        self.distillation = FederatedDistillation(
            config.distillation_temperature, config.distillation_alpha
        )
        self.aggregator = SparsityAggregator()
        self.coordination_history: List[Dict[str, Any]] = []

    async def coordinate_pruning_round(
        self,
        participant_models: List[nn.Module],
        participant_weights: List[float],
        global_model: nn.Module,
        round_num: int,
        total_rounds: int = 100,
    ) -> Tuple[nn.Module, List[nn.Module], Dict[str, Any]]:
        """Coordinate a full pruning round across participants."""
        start_time = time.time()

        logger.info(f"Starting pruning coordination round {round_num}")

        # Check if pruning should be applied this round
        if round_num % self.config.pruning_frequency != 0:
            logger.info(
                f"Skipping pruning round {round_num} (frequency: {self.config.pruning_frequency})"
            )
            return global_model, participant_models, {"skipped": True}

        pruned_participants = []
        pruning_metrics = []
        masks = []

        # Apply pruning to each participant model
        for i, model in enumerate(participant_models):
            logger.info(f"Pruning participant model {i + 1}/{len(participant_models)}")

            # Choose appropriate pruner
            if self.structured_pruner and self.config.pruning_type.value.startswith(
                "structured"
            ):
                pruned_model, metrics = await self.structured_pruner.prune_model(
                    model, round_num, total_rounds
                )
                mask = {}  # Structured pruning doesn't use weight masks
            elif self.unstructured_pruner and self.config.pruning_type.value.startswith(
                "unstructured"
            ):
                pruned_model, metrics = await self.unstructured_pruner.prune_model(
                    model, round_num, total_rounds
                )
                mask = self.unstructured_pruner.weight_masks.copy()
            else:
                raise ValueError(
                    f"No suitable pruner for type: {self.config.pruning_type}"
                )

            # Check if accuracy drop is within acceptable range
            if metrics.accuracy_drop > self.config.recovery_threshold:
                logger.warning(
                    f"Participant {i + 1} accuracy drop ({metrics.accuracy_drop:.3f}) "
                    f"exceeds threshold ({self.config.recovery_threshold:.3f})"
                )

                # Apply knowledge distillation for recovery
                if self.config.distillation_enabled:
                    logger.info(
                        f"Applying knowledge distillation for participant {i + 1}"
                    )
                    pruned_model, distillation_metrics = (
                        await self.distillation.distill_knowledge(
                            teacher_model=model, student_model=pruned_model
                        )
                    )
                    metrics.accuracy_after = distillation_metrics["student_accuracy"]
                    metrics.accuracy_drop = (
                        metrics.accuracy_before - metrics.accuracy_after
                    )

            pruned_participants.append(pruned_model)
            pruning_metrics.append(metrics)
            masks.append(mask)

        # Aggregate pruned models
        participant_data = [
            (model, weight, mask)
            for model, weight, mask in zip(
                pruned_participants, participant_weights, masks
            )
        ]
        global_model, aggregation_metrics = (
            await self.aggregator.aggregate_sparse_models(
                participant_data, global_model
            )
        )

        coordination_time = time.time() - start_time

        # Compile overall metrics
        overall_metrics = {
            "coordination_time": coordination_time,
            "round_num": round_num,
            "num_participants": len(participant_models),
            "pruning_metrics": [m.to_dict() for m in pruning_metrics],
            "aggregation_metrics": aggregation_metrics,
            "average_sparsity": np.mean([m.sparsity_ratio for m in pruning_metrics]),
            "average_accuracy_drop": np.mean(
                [m.accuracy_drop for m in pruning_metrics]
            ),
            "max_accuracy_drop": max([m.accuracy_drop for m in pruning_metrics]),
            "pruning_type": self.config.pruning_type.value,
            "target_sparsity": self.config.target_sparsity,
            "distillation_applied": any(
                m.accuracy_drop > self.config.recovery_threshold
                for m in pruning_metrics
            ),
        }

        self.coordination_history.append(overall_metrics)

        logger.info(
            f"Pruning coordination round {round_num} completed: "
            f"{overall_metrics['average_sparsity']:.3f} avg sparsity, "
            f"{overall_metrics['average_accuracy_drop']:.3f} avg accuracy drop"
        )

        return global_model, pruned_participants, overall_metrics

    def get_pruning_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all pruning operations."""
        if not self.coordination_history:
            return {"message": "No pruning operations completed yet"}

        total_rounds = len(self.coordination_history)

        sparsity_progression = [
            round_data["average_sparsity"] for round_data in self.coordination_history
        ]
        accuracy_drops = [
            round_data["average_accuracy_drop"]
            for round_data in self.coordination_history
        ]

        summary = {
            "total_pruning_rounds": total_rounds,
            "pruning_type": self.config.pruning_type.value,
            "target_sparsity": self.config.target_sparsity,
            "final_average_sparsity": (
                sparsity_progression[-1] if sparsity_progression else 0.0
            ),
            "sparsity_progression": sparsity_progression,
            "average_accuracy_drop": np.mean(accuracy_drops) if accuracy_drops else 0.0,
            "max_accuracy_drop": max(accuracy_drops) if accuracy_drops else 0.0,
            "min_accuracy_drop": min(accuracy_drops) if accuracy_drops else 0.0,
            "distillation_enabled": self.config.distillation_enabled,
            "recovery_threshold": self.config.recovery_threshold,
            "pruning_frequency": self.config.pruning_frequency,
            "total_coordination_time": sum(
                round_data["coordination_time"]
                for round_data in self.coordination_history
            ),
            "average_coordination_time": np.mean(
                [
                    round_data["coordination_time"]
                    for round_data in self.coordination_history
                ]
            ),
            "configuration": {
                "pruning_type": self.config.pruning_type.value,
                "target_sparsity": self.config.target_sparsity,
                "schedule": self.config.schedule.value,
                "pruning_frequency": self.config.pruning_frequency,
                "recovery_threshold": self.config.recovery_threshold,
                "distillation_enabled": self.config.distillation_enabled,
                "distillation_temperature": self.config.distillation_temperature,
                "distillation_alpha": self.config.distillation_alpha,
            },
        }

        return summary

    async def save_pruning_history(self, filepath: str):
        """Save pruning history to file."""
        history_data = {
            "config": {
                "pruning_type": self.config.pruning_type.value,
                "target_sparsity": self.config.target_sparsity,
                "schedule": self.config.schedule.value,
                "pruning_frequency": self.config.pruning_frequency,
                "recovery_threshold": self.config.recovery_threshold,
                "distillation_enabled": self.config.distillation_enabled,
                "distillation_temperature": self.config.distillation_temperature,
                "distillation_alpha": self.config.distillation_alpha,
            },
            "coordination_history": self.coordination_history,
            "summary": self.get_pruning_summary(),
        }

        with open(filepath, "w") as f:
            json.dump(history_data, f, indent=2, default=str)

        logger.info(f"Pruning history saved to {filepath}")


# Example usage and testing functions
async def create_test_models():
    """Create test models for pruning validation."""

    # Simple CNN model
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(256, num_classes)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    return SimpleCNN()


async def demonstrate_pruning():
    """Demonstrate the federated model pruning system."""
    print("=== Federated Model Pruning Demonstration ===\n")

    # Create test configuration
    config = PruningConfig(
        pruning_type=PruningType.UNSTRUCTURED_MAGNITUDE,
        target_sparsity=0.7,
        schedule=SparsitySchedule.LINEAR,
        pruning_frequency=2,
        recovery_threshold=0.05,
        distillation_enabled=True,
        distillation_temperature=4.0,
        distillation_alpha=0.7,
    )

    # Create test models
    global_model = await create_test_models()
    participant_models = [await create_test_models() for _ in range(3)]
    participant_weights = [1.0, 1.0, 1.0]  # Equal weights

    # Initialize pruning coordinator
    coordinator = PruningCoordinator(config)

    print(f"Configuration:")
    print(f"- Pruning Type: {config.pruning_type.value}")
    print(f"- Target Sparsity: {config.target_sparsity}")
    print(f"- Schedule: {config.schedule.value}")
    print(f"- Pruning Frequency: Every {config.pruning_frequency} rounds")
    print(f"- Recovery Threshold: {config.recovery_threshold}")
    print(
        f"- Knowledge Distillation: {'Enabled' if config.distillation_enabled else 'Disabled'}"
    )
    print()

    # Simulate federated learning rounds with pruning
    total_rounds = 10
    for round_num in range(1, total_rounds + 1):
        print(f"Round {round_num}/{total_rounds}")

        global_model, participant_models, metrics = (
            await coordinator.coordinate_pruning_round(
                participant_models=participant_models,
                participant_weights=participant_weights,
                global_model=global_model,
                round_num=round_num,
                total_rounds=total_rounds,
            )
        )

        if not metrics.get("skipped", False):
            print(f"  Average Sparsity: {metrics['average_sparsity']:.3f}")
            print(f"  Average Accuracy Drop: {metrics['average_accuracy_drop']:.3f}")
            print(f"  Max Accuracy Drop: {metrics['max_accuracy_drop']:.3f}")
            print(f"  Coordination Time: {metrics['coordination_time']:.2f}s")
            if metrics["distillation_applied"]:
                print(f"  Knowledge distillation applied for recovery")
        else:
            print(f"  Pruning skipped (frequency: {config.pruning_frequency})")

        print()

    # Get final summary
    summary = coordinator.get_pruning_summary()
    print("=== Final Pruning Summary ===")
    print(f"Total Pruning Rounds: {summary['total_pruning_rounds']}")
    print(f"Final Average Sparsity: {summary['final_average_sparsity']:.3f}")
    print(f"Average Accuracy Drop: {summary['average_accuracy_drop']:.3f}")
    print(f"Max Accuracy Drop: {summary['max_accuracy_drop']:.3f}")
    print(f"Total Coordination Time: {summary['total_coordination_time']:.2f}s")
    print(f"Average Coordination Time: {summary['average_coordination_time']:.2f}s")
    print()

    # Save history
    await coordinator.save_pruning_history("pruning_history.json")
    print("Pruning history saved to pruning_history.json")

    return coordinator, summary


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstrate_pruning())
