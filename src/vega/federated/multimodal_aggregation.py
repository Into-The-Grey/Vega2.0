"""
Specialized Aggregation Strategies for Multi-Modal Federated Learning

This module implements modality-specific aggregation strategies, weighted fusion
based on data quality and quantity, and adaptive aggregation that considers
different modalities' characteristics in federated learning environments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging
import warnings

from .multimodal import DataModality, MultiModalBatch
from .cross_modal import CrossModalLearningFramework

# Configure logging
logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Multi-modal aggregation strategies."""

    FEDAVG_MULTIMODAL = "fedavg_multimodal"
    MODALITY_WEIGHTED = "modality_weighted"
    ADAPTIVE_MODALITY = "adaptive_modality"
    HIERARCHICAL_MULTIMODAL = "hierarchical_multimodal"
    QUALITY_AWARE = "quality_aware"
    CONTRASTIVE_AGGREGATION = "contrastive_aggregation"


@dataclass
class ParticipantModalityInfo:
    """Information about a participant's modality data."""

    participant_id: str
    modality: DataModality
    data_size: int
    data_quality_score: float = 1.0
    modality_coverage: float = 1.0  # How much of the modality space is covered
    local_performance: Dict[str, float] = field(default_factory=dict)
    gradient_norm: Optional[float] = None
    privacy_budget: Optional[float] = None


@dataclass
class MultiModalAggregationConfig:
    """Configuration for multi-modal aggregation."""

    strategy: AggregationStrategy = AggregationStrategy.ADAPTIVE_MODALITY

    # Weighting parameters
    use_data_size_weighting: bool = True
    use_quality_weighting: bool = True
    use_performance_weighting: bool = True

    # Quality assessment
    quality_threshold: float = 0.5
    quality_weight_alpha: float = 0.3

    # Adaptive parameters
    adaptation_rate: float = 0.1
    momentum: float = 0.9

    # Hierarchical parameters
    modality_level_weight: float = 0.7
    cross_modal_weight: float = 0.3

    # Contrastive parameters
    contrastive_temperature: float = 0.1
    contrastive_weight: float = 0.1

    # Privacy parameters
    enable_differential_privacy: bool = False
    noise_multiplier: float = 1.0
    clipping_threshold: float = 1.0

    # Secure aggregation parameters
    enable_secure_aggregation: bool = False
    secure_aggregation_method: str = (
        "additive_masking"  # "additive_masking", "threshold_secret_sharing"
    )


class ModalityAggregator(ABC):
    """Abstract base class for modality-specific aggregators."""

    def __init__(self, config: MultiModalAggregationConfig):
        self.config = config

    @abstractmethod
    def aggregate(
        self,
        participant_updates: Dict[str, Dict[str, torch.Tensor]],
        participant_info: Dict[str, ParticipantModalityInfo],
    ) -> Dict[str, torch.Tensor]:
        """Aggregate updates from participants."""
        pass

    @abstractmethod
    def compute_weights(
        self, participant_info: Dict[str, ParticipantModalityInfo]
    ) -> Dict[str, float]:
        """Compute aggregation weights for participants."""
        pass


class FedAvgMultiModalAggregator(ModalityAggregator):
    """FedAvg-style aggregation adapted for multi-modal data."""

    def aggregate(
        self,
        participant_updates: Dict[str, Dict[str, torch.Tensor]],
        participant_info: Dict[str, ParticipantModalityInfo],
    ) -> Dict[str, torch.Tensor]:
        """Aggregate using weighted average based on data sizes."""
        if not participant_updates:
            raise ValueError("No participant updates provided")

        # Compute weights
        weights = self.compute_weights(participant_info)

        # Get parameter names from first participant
        first_participant = next(iter(participant_updates.values()))
        param_names = list(first_participant.keys())

        # Initialize aggregated parameters
        aggregated_params = {}

        for param_name in param_names:
            weighted_sum = None
            total_weight = 0.0

            for participant_id, updates in participant_updates.items():
                if participant_id in weights and param_name in updates:
                    weight = weights[participant_id]
                    param_update = updates[param_name]

                    if weighted_sum is None:
                        weighted_sum = weight * param_update
                    else:
                        weighted_sum += weight * param_update

                    total_weight += weight

            if weighted_sum is not None and total_weight > 0:
                aggregated_params[param_name] = weighted_sum / total_weight
            else:
                logger.warning(f"No valid updates for parameter: {param_name}")

        return aggregated_params

    def compute_weights(
        self, participant_info: Dict[str, ParticipantModalityInfo]
    ) -> Dict[str, float]:
        """Compute weights based on data size."""
        total_data_size = sum(info.data_size for info in participant_info.values())

        if total_data_size == 0:
            # Equal weights if no data size information
            num_participants = len(participant_info)
            return {pid: 1.0 / num_participants for pid in participant_info.keys()}

        weights = {}
        for participant_id, info in participant_info.items():
            weights[participant_id] = info.data_size / total_data_size

        return weights


class ModalityWeightedAggregator(ModalityAggregator):
    """Aggregator that considers modality-specific characteristics."""

    def __init__(self, config: MultiModalAggregationConfig):
        super().__init__(config)
        self.modality_weights = {}  # Learned modality importance weights

    def aggregate(
        self,
        participant_updates: Dict[str, Dict[str, torch.Tensor]],
        participant_info: Dict[str, ParticipantModalityInfo],
    ) -> Dict[str, torch.Tensor]:
        """Aggregate with modality-aware weighting."""
        # Group participants by modality
        modality_groups = self._group_by_modality(participant_info)

        # Aggregate within each modality first
        modality_aggregates = {}
        for modality, participant_ids in modality_groups.items():
            modality_updates = {
                pid: participant_updates[pid]
                for pid in participant_ids
                if pid in participant_updates
            }
            modality_info = {pid: participant_info[pid] for pid in participant_ids}

            if modality_updates:
                # Use FedAvg within modality
                fedavg_aggregator = FedAvgMultiModalAggregator(self.config)
                modality_aggregate = fedavg_aggregator.aggregate(
                    modality_updates, modality_info
                )
                modality_aggregates[modality] = modality_aggregate

        # Combine modality aggregates
        if len(modality_aggregates) == 1:
            return next(iter(modality_aggregates.values()))

        return self._combine_modality_aggregates(
            modality_aggregates, modality_groups, participant_info
        )

    def compute_weights(
        self, participant_info: Dict[str, ParticipantModalityInfo]
    ) -> Dict[str, float]:
        """Compute weights considering modality characteristics."""
        weights = {}

        for participant_id, info in participant_info.items():
            weight = 1.0

            # Data size weight
            if self.config.use_data_size_weighting:
                total_data = sum(pi.data_size for pi in participant_info.values())
                if total_data > 0:
                    weight *= info.data_size / total_data

            # Quality weight
            if self.config.use_quality_weighting:
                quality_weight = max(
                    info.data_quality_score, self.config.quality_threshold
                )
                weight *= quality_weight**self.config.quality_weight_alpha

            # Performance weight
            if self.config.use_performance_weighting and info.local_performance:
                avg_performance = np.mean(list(info.local_performance.values()))
                weight *= avg_performance

            # Modality coverage weight
            weight *= info.modality_coverage

            weights[participant_id] = weight

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {pid: w / total_weight for pid, w in weights.items()}

        return weights

    def _group_by_modality(
        self, participant_info: Dict[str, ParticipantModalityInfo]
    ) -> Dict[DataModality, List[str]]:
        """Group participants by their primary modality."""
        modality_groups = {}

        for participant_id, info in participant_info.items():
            modality = info.modality
            if modality not in modality_groups:
                modality_groups[modality] = []
            modality_groups[modality].append(participant_id)

        return modality_groups

    def _combine_modality_aggregates(
        self,
        modality_aggregates: Dict[DataModality, Dict[str, torch.Tensor]],
        modality_groups: Dict[DataModality, List[str]],
        participant_info: Dict[str, ParticipantModalityInfo],
    ) -> Dict[str, torch.Tensor]:
        """Combine aggregates from different modalities."""
        # Compute modality weights based on group sizes and quality
        modality_weights = {}

        for modality, participant_ids in modality_groups.items():
            total_data = sum(participant_info[pid].data_size for pid in participant_ids)
            avg_quality = np.mean(
                [participant_info[pid].data_quality_score for pid in participant_ids]
            )

            modality_weight = total_data * avg_quality
            modality_weights[modality] = modality_weight

        # Normalize modality weights
        total_modality_weight = sum(modality_weights.values())
        if total_modality_weight > 0:
            modality_weights = {
                m: w / total_modality_weight for m, w in modality_weights.items()
            }

        # Combine modality aggregates
        param_names = set()
        for aggregate in modality_aggregates.values():
            param_names.update(aggregate.keys())

        combined_params = {}
        for param_name in param_names:
            weighted_sum = None
            total_weight = 0.0

            for modality, aggregate in modality_aggregates.items():
                if param_name in aggregate:
                    weight = modality_weights.get(modality, 0.0)
                    param_tensor = aggregate[param_name]

                    if weighted_sum is None:
                        weighted_sum = weight * param_tensor
                    else:
                        weighted_sum += weight * param_tensor

                    total_weight += weight

            if weighted_sum is not None and total_weight > 0:
                combined_params[param_name] = weighted_sum / total_weight

        return combined_params


class AdaptiveModalityAggregator(ModalityAggregator):
    """Adaptive aggregator that learns optimal modality weighting."""

    def __init__(self, config: MultiModalAggregationConfig):
        super().__init__(config)
        self.modality_performance_history = {}
        self.adaptive_weights = {}
        self.round_count = 0

    def aggregate(
        self,
        participant_updates: Dict[str, Dict[str, torch.Tensor]],
        participant_info: Dict[str, ParticipantModalityInfo],
    ) -> Dict[str, torch.Tensor]:
        """Adaptive aggregation with learned modality weights."""
        self.round_count += 1

        # Update performance history
        self._update_performance_history(participant_info)

        # Compute adaptive weights
        adaptive_weights = self._compute_adaptive_weights(participant_info)

        # Weighted aggregation
        return self._weighted_aggregate(participant_updates, adaptive_weights)

    def compute_weights(
        self, participant_info: Dict[str, ParticipantModalityInfo]
    ) -> Dict[str, float]:
        """Compute adaptive weights based on historical performance."""
        return self._compute_adaptive_weights(participant_info)

    def _update_performance_history(
        self, participant_info: Dict[str, ParticipantModalityInfo]
    ) -> None:
        """Update performance history for adaptive weighting."""
        for participant_id, info in participant_info.items():
            modality = info.modality

            if modality not in self.modality_performance_history:
                self.modality_performance_history[modality] = []

            if info.local_performance:
                avg_performance = np.mean(list(info.local_performance.values()))
                self.modality_performance_history[modality].append(avg_performance)

                # Keep only recent history
                max_history = 10
                if len(self.modality_performance_history[modality]) > max_history:
                    self.modality_performance_history[modality] = (
                        self.modality_performance_history[modality][-max_history:]
                    )

    def _compute_adaptive_weights(
        self, participant_info: Dict[str, ParticipantModalityInfo]
    ) -> Dict[str, float]:
        """Compute adaptive weights based on modality performance trends."""
        weights = {}

        # Compute modality performance scores
        modality_scores = {}
        for modality, history in self.modality_performance_history.items():
            if history:
                # Recent performance trend
                recent_performance = (
                    np.mean(history[-3:]) if len(history) >= 3 else np.mean(history)
                )
                # Performance improvement trend
                if len(history) >= 2:
                    trend = history[-1] - history[-2]
                else:
                    trend = 0.0

                modality_scores[modality] = recent_performance + 0.1 * trend
            else:
                modality_scores[modality] = 0.5  # Default score

        # Update adaptive weights with momentum
        for modality, score in modality_scores.items():
            if modality in self.adaptive_weights:
                # Momentum update
                self.adaptive_weights[modality] = (
                    self.config.momentum * self.adaptive_weights[modality]
                    + self.config.adaptation_rate * score
                )
            else:
                self.adaptive_weights[modality] = score

        # Compute participant weights
        for participant_id, info in participant_info.items():
            modality = info.modality
            base_weight = self.adaptive_weights.get(modality, 0.5)

            # Adjust based on participant-specific factors
            participant_weight = base_weight

            if self.config.use_data_size_weighting:
                total_data = sum(pi.data_size for pi in participant_info.values())
                if total_data > 0:
                    participant_weight *= info.data_size / total_data

            if self.config.use_quality_weighting:
                participant_weight *= info.data_quality_score

            weights[participant_id] = participant_weight

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {pid: w / total_weight for pid, w in weights.items()}

        return weights

    def _weighted_aggregate(
        self,
        participant_updates: Dict[str, Dict[str, torch.Tensor]],
        weights: Dict[str, float],
    ) -> Dict[str, torch.Tensor]:
        """Perform weighted aggregation of participant updates."""
        if not participant_updates:
            return {}

        # Get parameter names
        first_participant = next(iter(participant_updates.values()))
        param_names = list(first_participant.keys())

        aggregated_params = {}

        for param_name in param_names:
            weighted_sum = None
            total_weight = 0.0

            for participant_id, updates in participant_updates.items():
                if participant_id in weights and param_name in updates:
                    weight = weights[participant_id]
                    param_update = updates[param_name]

                    if weighted_sum is None:
                        weighted_sum = weight * param_update
                    else:
                        weighted_sum += weight * param_update

                    total_weight += weight

            if weighted_sum is not None and total_weight > 0:
                aggregated_params[param_name] = weighted_sum / total_weight

        return aggregated_params


class QualityAwareAggregator(ModalityAggregator):
    """Aggregator that emphasizes data quality over quantity."""

    def aggregate(
        self,
        participant_updates: Dict[str, Dict[str, torch.Tensor]],
        participant_info: Dict[str, ParticipantModalityInfo],
    ) -> Dict[str, torch.Tensor]:
        """Aggregate with emphasis on data quality."""
        # Filter participants by quality threshold
        high_quality_participants = {
            pid: info
            for pid, info in participant_info.items()
            if info.data_quality_score >= self.config.quality_threshold
        }

        if not high_quality_participants:
            logger.warning(
                "No participants meet quality threshold, using all participants"
            )
            high_quality_participants = participant_info

        # Filter updates to include only high-quality participants
        quality_updates = {
            pid: participant_updates[pid]
            for pid in high_quality_participants.keys()
            if pid in participant_updates
        }

        # Compute quality-based weights
        weights = self._compute_quality_weights(high_quality_participants)

        # Weighted aggregation
        return self._weighted_aggregate(quality_updates, weights)

    def compute_weights(
        self, participant_info: Dict[str, ParticipantModalityInfo]
    ) -> Dict[str, float]:
        """Compute quality-based weights."""
        return self._compute_quality_weights(participant_info)

    def _compute_quality_weights(
        self, participant_info: Dict[str, ParticipantModalityInfo]
    ) -> Dict[str, float]:
        """Compute weights based on data quality scores."""
        weights = {}

        for participant_id, info in participant_info.items():
            # Quality score with exponential emphasis
            quality_weight = info.data_quality_score ** (
                1 / self.config.quality_weight_alpha
            )

            # Modality coverage factor
            coverage_weight = info.modality_coverage

            # Combined weight
            weights[participant_id] = quality_weight * coverage_weight

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {pid: w / total_weight for pid, w in weights.items()}

        return weights

    def _weighted_aggregate(
        self,
        participant_updates: Dict[str, Dict[str, torch.Tensor]],
        weights: Dict[str, float],
    ) -> Dict[str, torch.Tensor]:
        """Perform weighted aggregation."""
        if not participant_updates:
            return {}

        first_participant = next(iter(participant_updates.values()))
        param_names = list(first_participant.keys())

        aggregated_params = {}

        for param_name in param_names:
            weighted_sum = None
            total_weight = 0.0

            for participant_id, updates in participant_updates.items():
                if participant_id in weights and param_name in updates:
                    weight = weights[participant_id]
                    param_update = updates[param_name]

                    if weighted_sum is None:
                        weighted_sum = weight * param_update
                    else:
                        weighted_sum += weight * param_update

                    total_weight += weight

            if weighted_sum is not None and total_weight > 0:
                aggregated_params[param_name] = weighted_sum / total_weight

        return aggregated_params


class HierarchicalMultiModalAggregator(ModalityAggregator):
    """Hierarchical aggregation: first within modalities, then across modalities."""

    def aggregate(
        self,
        participant_updates: Dict[str, Dict[str, torch.Tensor]],
        participant_info: Dict[str, ParticipantModalityInfo],
    ) -> Dict[str, torch.Tensor]:
        """Hierarchical aggregation with modality-level and cross-modal levels."""
        # Group by modality
        modality_groups = {}
        for participant_id, info in participant_info.items():
            modality = info.modality
            if modality not in modality_groups:
                modality_groups[modality] = {}
            modality_groups[modality][participant_id] = info

        # Stage 1: Aggregate within each modality
        modality_aggregates = {}
        for modality, modality_participants in modality_groups.items():
            modality_updates = {
                pid: participant_updates[pid]
                for pid in modality_participants.keys()
                if pid in participant_updates
            }

            if modality_updates:
                # Use quality-aware aggregation within modality
                intra_aggregator = QualityAwareAggregator(self.config)
                modality_aggregate = intra_aggregator.aggregate(
                    modality_updates, modality_participants
                )
                modality_aggregates[modality] = modality_aggregate

        # Stage 2: Aggregate across modalities
        if len(modality_aggregates) == 1:
            return next(iter(modality_aggregates.values()))

        return self._cross_modal_aggregate(
            modality_aggregates, modality_groups, participant_info
        )

    def compute_weights(
        self, participant_info: Dict[str, ParticipantModalityInfo]
    ) -> Dict[str, float]:
        """Compute hierarchical weights."""
        # This is handled within the aggregation process
        return {}

    def _cross_modal_aggregate(
        self,
        modality_aggregates: Dict[DataModality, Dict[str, torch.Tensor]],
        modality_groups: Dict[DataModality, Dict[str, ParticipantModalityInfo]],
        all_participant_info: Dict[str, ParticipantModalityInfo],
    ) -> Dict[str, torch.Tensor]:
        """Aggregate across modalities with learned weights."""
        # Compute modality importance weights
        modality_weights = {}
        for modality, participants in modality_groups.items():
            # Consider total data size and average quality
            total_data = sum(info.data_size for info in participants.values())
            avg_quality = np.mean(
                [info.data_quality_score for info in participants.values()]
            )
            perf_list = []
            for info in participants.values():
                if info.local_performance and len(info.local_performance) > 0:
                    perf = float(
                        np.mean([float(v) for v in info.local_performance.values()])
                    )
                else:
                    perf = 0.5
                perf_list.append(perf)
            avg_performance = float(np.mean(perf_list)) if perf_list else 0.5

            # Combined modality weight
            modality_weight = (
                self.config.modality_level_weight * (total_data * avg_quality)
                + self.config.cross_modal_weight * avg_performance
            )
            modality_weights[modality] = modality_weight

        # Normalize modality weights
        total_weight = sum(modality_weights.values())
        if total_weight > 0:
            modality_weights = {
                m: w / total_weight for m, w in modality_weights.items()
            }

        # Combine modality aggregates
        param_names = set()
        for aggregate in modality_aggregates.values():
            param_names.update(aggregate.keys())

        combined_params = {}
        for param_name in param_names:
            weighted_sum = None
            total_weight = 0.0

            for modality, aggregate in modality_aggregates.items():
                if param_name in aggregate:
                    weight = modality_weights.get(modality, 0.0)
                    param_tensor = aggregate[param_name]

                    if weighted_sum is None:
                        weighted_sum = weight * param_tensor
                    else:
                        weighted_sum += weight * param_tensor

                    total_weight += weight

            if weighted_sum is not None and total_weight > 0:
                combined_params[param_name] = weighted_sum / total_weight

        return combined_params


class MultiModalAggregationCoordinator:
    """Coordinator for multi-modal federated learning aggregation."""

    def __init__(self, config: MultiModalAggregationConfig):
        self.config = config
        self.aggregator = self._create_aggregator()
        self.aggregation_history = []

    def _create_aggregator(self) -> ModalityAggregator:
        """Create the appropriate aggregator based on strategy."""
        if self.config.strategy == AggregationStrategy.FEDAVG_MULTIMODAL:
            return FedAvgMultiModalAggregator(self.config)
        elif self.config.strategy == AggregationStrategy.MODALITY_WEIGHTED:
            return ModalityWeightedAggregator(self.config)
        elif self.config.strategy == AggregationStrategy.ADAPTIVE_MODALITY:
            return AdaptiveModalityAggregator(self.config)
        elif self.config.strategy == AggregationStrategy.QUALITY_AWARE:
            return QualityAwareAggregator(self.config)
        elif self.config.strategy == AggregationStrategy.HIERARCHICAL_MULTIMODAL:
            return HierarchicalMultiModalAggregator(self.config)
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.config.strategy}")

    def aggregate_round(
        self,
        participant_updates: Dict[str, Dict[str, torch.Tensor]],
        participant_info: Dict[str, ParticipantModalityInfo],
        round_number: int,
    ) -> Dict[str, torch.Tensor]:
        """Perform aggregation for a federated learning round, with optional secure aggregation and DP."""
        logger.info(
            f"Starting aggregation round {round_number} with {len(participant_updates)} participants"
        )

        # Secure aggregation (additive masking)
        if getattr(self.config, "enable_secure_aggregation", False):
            method = getattr(
                self.config, "secure_aggregation_method", "additive_masking"
            )
            if method == "additive_masking":
                participant_updates, unmasking_secrets = self._apply_additive_masking(
                    participant_updates
                )
            elif method == "threshold_secret_sharing":
                # Placeholder for future extension
                logger.warning(
                    "Threshold secret sharing not yet implemented. Proceeding without masking."
                )
                unmasking_secrets = None
            else:
                logger.warning(
                    f"Unknown secure aggregation method: {method}. Proceeding without masking."
                )
                unmasking_secrets = None
        else:
            unmasking_secrets = None

        # Apply differential privacy if enabled
        if self.config.enable_differential_privacy:
            participant_updates = self._apply_differential_privacy(participant_updates)

        # Perform aggregation
        aggregated_params = self.aggregator.aggregate(
            participant_updates, participant_info
        )

        # Unmask aggregated parameters if secure aggregation was used
        if (
            getattr(self.config, "enable_secure_aggregation", False)
            and unmasking_secrets is not None
        ):
            aggregated_params = self._remove_additive_masking(
                aggregated_params, unmasking_secrets
            )

        # Store aggregation metadata
        self.aggregation_history.append(
            {
                "round": round_number,
                "num_participants": len(participant_updates),
                "modalities": list(
                    set(info.modality for info in participant_info.values())
                ),
                "total_data_size": sum(
                    info.data_size for info in participant_info.values()
                ),
                "avg_quality": np.mean(
                    [info.data_quality_score for info in participant_info.values()]
                ),
                "secure_aggregation": getattr(
                    self.config, "enable_secure_aggregation", False
                ),
                "secure_aggregation_method": getattr(
                    self.config, "secure_aggregation_method", None
                ),
            }
        )

        logger.info(f"Aggregation round {round_number} completed")
        return aggregated_params

    def _apply_additive_masking(
        self, participant_updates: Dict[str, Dict[str, torch.Tensor]]
    ) -> tuple:
        """Apply additive masking to each participant's updates for secure aggregation.
        Returns (masked_updates, unmasking_secrets) where unmasking_secrets is a dict of param_name: total_mask_sum.
        """
        import copy

        masked_updates = copy.deepcopy(participant_updates)
        unmasking_secrets = {}
        # For each parameter, sum all masks to remove after aggregation
        param_names = set()
        for updates in participant_updates.values():
            param_names.update(updates.keys())
        # Generate random masks for each participant and param
        masks = {pid: {} for pid in participant_updates.keys()}
        for param_name in param_names:
            total_mask = None
            for pid in participant_updates.keys():
                shape = participant_updates[pid][param_name].shape
                mask = torch.randn(
                    shape, device=participant_updates[pid][param_name].device
                )
                masks[pid][param_name] = mask
                masked_updates[pid][param_name] = (
                    participant_updates[pid][param_name] + mask
                )
                if total_mask is None:
                    total_mask = mask.clone()
                else:
                    total_mask += mask
            unmasking_secrets[param_name] = total_mask
        return masked_updates, unmasking_secrets

    def _remove_additive_masking(
        self, aggregated_params: Dict[str, torch.Tensor], unmasking_secrets: dict
    ) -> Dict[str, torch.Tensor]:
        """Remove additive masks from aggregated parameters after aggregation."""
        for param_name, mask_sum in unmasking_secrets.items():
            if param_name in aggregated_params:
                aggregated_params[param_name] = aggregated_params[
                    param_name
                ] - mask_sum / max(1, len(unmasking_secrets))
        return aggregated_params

    def _apply_differential_privacy(
        self, participant_updates: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Apply differential privacy (DP) to participant updates: gradient clipping and noise addition."""
        noisy_updates = {}
        clip = getattr(self.config, "clipping_threshold", 1.0)
        noise_mult = getattr(self.config, "noise_multiplier", 1.0)

        for participant_id, updates in participant_updates.items():
            noisy_participant_updates = {}
            for param_name, param_tensor in updates.items():
                # Clip parameter update
                clipped_tensor = torch.clamp(param_tensor, -clip, clip)
                # Add Gaussian noise
                noise = torch.normal(
                    mean=0.0,
                    std=noise_mult * clip,
                    size=clipped_tensor.shape,
                    device=clipped_tensor.device,
                )
                noisy_tensor = clipped_tensor + noise
                noisy_participant_updates[param_name] = noisy_tensor
            noisy_updates[participant_id] = noisy_participant_updates
        return noisy_updates

    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get statistics about the aggregation process."""
        if not self.aggregation_history:
            return {}

        stats = {
            "total_rounds": len(self.aggregation_history),
            "avg_participants_per_round": np.mean(
                [h["num_participants"] for h in self.aggregation_history]
            ),
            "total_modalities": len(
                set().union(*[h["modalities"] for h in self.aggregation_history])
            ),
            "avg_data_size_per_round": np.mean(
                [h["total_data_size"] for h in self.aggregation_history]
            ),
            "avg_quality_per_round": np.mean(
                [h["avg_quality"] for h in self.aggregation_history]
            ),
        }

        return stats


def create_multimodal_aggregator(
    strategy: AggregationStrategy = AggregationStrategy.ADAPTIVE_MODALITY,
    config: Optional[MultiModalAggregationConfig] = None,
) -> MultiModalAggregationCoordinator:
    """Factory function to create multi-modal aggregation coordinator."""

    if config is None:
        config = MultiModalAggregationConfig(strategy=strategy)
    else:
        config.strategy = strategy

    coordinator = MultiModalAggregationCoordinator(config)

    logger.info(f"Created multi-modal aggregator with strategy: {strategy.value}")

    return coordinator


if __name__ == "__main__":
    # Example usage
    print("Multi-Modal Aggregation Strategies")
    print("=" * 40)

    # Create different aggregators
    strategies = [
        AggregationStrategy.FEDAVG_MULTIMODAL,
        AggregationStrategy.ADAPTIVE_MODALITY,
        AggregationStrategy.QUALITY_AWARE,
        AggregationStrategy.HIERARCHICAL_MULTIMODAL,
    ]

    for strategy in strategies:
        coordinator = create_multimodal_aggregator(strategy)
        print(f"✓ Created {strategy.value} aggregator")

    print(f"\nAvailable aggregation strategies: {len(strategies)}")
