"""
Hierarchical Aggregation System for Cross-Silo Federated Learning

Implements multi-level aggregation where organizations aggregate locally,
then contribute to global aggregation. Supports different strategies at each level.

Design Principles:
- Three-tier aggregation: Silo → Organization → Global
- Configurable strategies per level
- Asynchronous aggregation with timeouts
- Performance tracking and convergence detection
- Byzantine fault tolerance at each level
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .model_serialization import ModelWeights, ModelSerializer
from .fedavg import FedAvg, FedAvgConfig, AsyncAggregator
from .cross_silo import (
    Organization,
    Silo,
    HierarchicalParticipant,
    CrossSiloSession,
    FederationLevel,
    OrganizationManager,
)
from .security import (
    audit_log,
    is_anomalous_update,
    check_model_consistency,
    validate_model_update_pipeline,
)

logger = logging.getLogger(__name__)


@dataclass
class AggregationResult:
    """Result of aggregation at any level."""

    level: FederationLevel
    aggregated_weights: ModelWeights
    contributing_entities: List[str]  # participant/silo/org IDs
    aggregation_metadata: Dict[str, Any]
    performance_metrics: Dict[str, float]
    anomalies_detected: List[str]
    validation_passed: bool
    aggregation_time: float
    round_number: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "aggregated_weights": self.aggregated_weights.to_dict(),
            "contributing_entities": self.contributing_entities,
            "aggregation_metadata": self.aggregation_metadata,
            "performance_metrics": self.performance_metrics,
            "anomalies_detected": self.anomalies_detected,
            "validation_passed": self.validation_passed,
            "aggregation_time": self.aggregation_time,
            "round_number": self.round_number,
        }


@dataclass
class LevelAggregationConfig:
    """Configuration for aggregation at a specific level."""

    level: FederationLevel
    strategy: str = "federated_averaging"
    min_contributors: int = 2
    max_wait_time: float = 300.0  # seconds
    convergence_threshold: float = 0.001
    byzantine_tolerance: float = 0.3  # fraction of byzantine participants to tolerate

    # Weighting strategy
    weighting_method: str = (
        "data_size"  # "equal", "data_size", "performance", "adaptive"
    )
    performance_weight: float = 0.3  # when using performance weighting

    # Security settings
    anomaly_detection: bool = True
    model_validation: bool = True
    require_signatures: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "strategy": self.strategy,
            "min_contributors": self.min_contributors,
            "max_wait_time": self.max_wait_time,
            "convergence_threshold": self.convergence_threshold,
            "byzantine_tolerance": self.byzantine_tolerance,
            "weighting_method": self.weighting_method,
            "performance_weight": self.performance_weight,
            "anomaly_detection": self.anomaly_detection,
            "model_validation": self.model_validation,
            "require_signatures": self.require_signatures,
        }


class HierarchicalAggregator:
    """Manages multi-level aggregation in hierarchical federation."""

    def __init__(self, org_manager: OrganizationManager):
        """Initialize hierarchical aggregator."""
        self.org_manager = org_manager
        self.aggregation_configs: Dict[FederationLevel, LevelAggregationConfig] = {}
        self.level_aggregators: Dict[FederationLevel, AsyncAggregator] = {}

        # Track aggregation state
        self.pending_updates: Dict[str, Dict[str, ModelWeights]] = (
            {}
        )  # level -> entity_id -> weights
        self.aggregation_results: Dict[str, AggregationResult] = (
            {}
        )  # session_id -> result
        self.convergence_history: Dict[str, List[float]] = (
            {}
        )  # session_id -> convergence values

        self.logger = logging.getLogger(__name__ + ".HierarchicalAggregator")

        # Initialize default configurations
        self._setup_default_configs()

    def _setup_default_configs(self):
        """Setup default aggregation configurations for each level."""
        # Silo-level aggregation (fastest, most permissive)
        self.aggregation_configs[FederationLevel.SILO] = LevelAggregationConfig(
            level=FederationLevel.SILO,
            strategy="federated_averaging",
            min_contributors=2,
            max_wait_time=120.0,
            convergence_threshold=0.01,
            byzantine_tolerance=0.2,
            weighting_method="data_size",
        )

        # Organization-level aggregation (medium strictness)
        self.aggregation_configs[FederationLevel.ORGANIZATION] = LevelAggregationConfig(
            level=FederationLevel.ORGANIZATION,
            strategy="federated_averaging",
            min_contributors=2,
            max_wait_time=300.0,
            convergence_threshold=0.005,
            byzantine_tolerance=0.25,
            weighting_method="adaptive",
        )

        # Global-level aggregation (most strict)
        self.aggregation_configs[FederationLevel.GLOBAL] = LevelAggregationConfig(
            level=FederationLevel.GLOBAL,
            strategy="federated_averaging",
            min_contributors=2,
            max_wait_time=600.0,
            convergence_threshold=0.001,
            byzantine_tolerance=0.3,
            weighting_method="performance",
        )

        # Initialize aggregators
        for level, config in self.aggregation_configs.items():
            fedavg_config = FedAvgConfig(
                min_participants=config.min_contributors,
                convergence_threshold=config.convergence_threshold,
                max_rounds=1,  # Single round per aggregation
                patience=1,
                adaptive_learning_rate=True,
                byzantine_tolerance=config.byzantine_tolerance,
            )
            self.level_aggregators[level] = AsyncAggregator(fedavg_config)

    async def aggregate_silo_level(
        self,
        session_id: str,
        silo_id: str,
        participant_updates: Dict[str, ModelWeights],
        round_number: int,
    ) -> AggregationResult:
        """Aggregate participant updates within a silo."""
        start_time = time.time()
        config = self.aggregation_configs[FederationLevel.SILO]

        await audit_log(
            "silo_aggregation_started",
            {
                "session_id": session_id,
                "silo_id": silo_id,
                "participant_count": len(participant_updates),
                "round": round_number,
            },
            "system",
        )

        # Validate minimum contributors
        if len(participant_updates) < config.min_contributors:
            raise ValueError(
                f"Insufficient participants for silo aggregation: {len(participant_updates)} < {config.min_contributors}"
            )

        # Security validation
        anomalies = []
        validated_updates = {}

        for participant_id, weights in participant_updates.items():
            # Check for anomalies
            if config.anomaly_detection and await is_anomalous_update(weights):
                anomalies.append(f"Anomalous update from participant {participant_id}")
                continue

            # Model validation
            if config.model_validation:
                try:
                    result = await validate_model_update_pipeline(weights, {})
                    if not result["valid"]:
                        anomalies.append(
                            f"Invalid model from participant {participant_id}: {result['reason']}"
                        )
                        continue
                except Exception as e:
                    anomalies.append(
                        f"Validation failed for participant {participant_id}: {str(e)}"
                    )
                    continue

            validated_updates[participant_id] = weights

        if len(validated_updates) < config.min_contributors:
            raise ValueError(
                f"Insufficient valid participants after validation: {len(validated_updates)} < {config.min_contributors}"
            )

        # Calculate weights for aggregation
        participant_weights = await self._calculate_participant_weights(
            silo_id, list(validated_updates.keys()), config.weighting_method
        )

        # Perform aggregation
        aggregator = self.level_aggregators[FederationLevel.SILO]
        aggregated_weights = await aggregator.aggregate_weights(
            list(validated_updates.values()), participant_weights
        )

        # Calculate performance metrics
        performance_metrics = {
            "convergence_rate": await self._calculate_convergence_rate(
                session_id, round_number
            ),
            "participation_rate": len(validated_updates) / len(participant_updates),
            "anomaly_rate": len(anomalies) / len(participant_updates),
            "aggregation_efficiency": 1.0
            - (time.time() - start_time) / config.max_wait_time,
        }

        result = AggregationResult(
            level=FederationLevel.SILO,
            aggregated_weights=aggregated_weights,
            contributing_entities=list(validated_updates.keys()),
            aggregation_metadata={
                "silo_id": silo_id,
                "strategy": config.strategy,
                "weights_used": participant_weights,
                "config": config.to_dict(),
            },
            performance_metrics=performance_metrics,
            anomalies_detected=anomalies,
            validation_passed=len(anomalies) == 0,
            aggregation_time=time.time() - start_time,
            round_number=round_number,
        )

        await audit_log(
            "silo_aggregation_completed",
            {"session_id": session_id, "silo_id": silo_id, "result": result.to_dict()},
            "system",
        )

        self.logger.info(
            f"Completed silo aggregation for {silo_id} in {result.aggregation_time:.2f}s"
        )
        return result

    async def aggregate_organization_level(
        self,
        session_id: str,
        org_id: str,
        silo_updates: Dict[str, ModelWeights],
        round_number: int,
    ) -> AggregationResult:
        """Aggregate silo updates within an organization."""
        start_time = time.time()
        config = self.aggregation_configs[FederationLevel.ORGANIZATION]

        await audit_log(
            "organization_aggregation_started",
            {
                "session_id": session_id,
                "org_id": org_id,
                "silo_count": len(silo_updates),
                "round": round_number,
            },
            "system",
        )

        # Validate minimum contributors
        if len(silo_updates) < config.min_contributors:
            raise ValueError(
                f"Insufficient silos for organization aggregation: {len(silo_updates)} < {config.min_contributors}"
            )

        # Security validation for silo updates
        anomalies = []
        validated_updates = {}

        for silo_id, weights in silo_updates.items():
            # Check for anomalies
            if config.anomaly_detection and await is_anomalous_update(weights):
                anomalies.append(f"Anomalous update from silo {silo_id}")
                continue

            # Model consistency check
            if config.model_validation:
                try:
                    # Cross-validate with other silo updates
                    other_weights = [
                        w for sid, w in silo_updates.items() if sid != silo_id
                    ]
                    if other_weights and not await check_model_consistency(
                        weights, other_weights[0]
                    ):
                        anomalies.append(f"Inconsistent model from silo {silo_id}")
                        continue
                except Exception as e:
                    anomalies.append(
                        f"Consistency check failed for silo {silo_id}: {str(e)}"
                    )
                    continue

            validated_updates[silo_id] = weights

        if len(validated_updates) < config.min_contributors:
            raise ValueError(
                f"Insufficient valid silos after validation: {len(validated_updates)} < {config.min_contributors}"
            )

        # Calculate weights for aggregation (silo-level)
        silo_weights = await self._calculate_silo_weights(
            org_id, list(validated_updates.keys()), config.weighting_method
        )

        # Perform aggregation
        aggregator = self.level_aggregators[FederationLevel.ORGANIZATION]
        aggregated_weights = await aggregator.aggregate_weights(
            list(validated_updates.values()), silo_weights
        )

        # Calculate performance metrics
        performance_metrics = {
            "convergence_rate": await self._calculate_convergence_rate(
                session_id, round_number
            ),
            "participation_rate": len(validated_updates) / len(silo_updates),
            "anomaly_rate": len(anomalies) / len(silo_updates),
            "aggregation_efficiency": 1.0
            - (time.time() - start_time) / config.max_wait_time,
        }

        result = AggregationResult(
            level=FederationLevel.ORGANIZATION,
            aggregated_weights=aggregated_weights,
            contributing_entities=list(validated_updates.keys()),
            aggregation_metadata={
                "org_id": org_id,
                "strategy": config.strategy,
                "weights_used": silo_weights,
                "config": config.to_dict(),
            },
            performance_metrics=performance_metrics,
            anomalies_detected=anomalies,
            validation_passed=len(anomalies) == 0,
            aggregation_time=time.time() - start_time,
            round_number=round_number,
        )

        await audit_log(
            "organization_aggregation_completed",
            {"session_id": session_id, "org_id": org_id, "result": result.to_dict()},
            "system",
        )

        self.logger.info(
            f"Completed organization aggregation for {org_id} in {result.aggregation_time:.2f}s"
        )
        return result

    async def aggregate_global_level(
        self,
        session_id: str,
        organization_updates: Dict[str, ModelWeights],
        round_number: int,
    ) -> AggregationResult:
        """Aggregate organization updates at the global level."""
        start_time = time.time()
        config = self.aggregation_configs[FederationLevel.GLOBAL]

        await audit_log(
            "global_aggregation_started",
            {
                "session_id": session_id,
                "organization_count": len(organization_updates),
                "round": round_number,
            },
            "system",
        )

        # Validate minimum contributors
        if len(organization_updates) < config.min_contributors:
            raise ValueError(
                f"Insufficient organizations for global aggregation: {len(organization_updates)} < {config.min_contributors}"
            )

        # Security validation for organization updates
        anomalies = []
        validated_updates = {}

        for org_id, weights in organization_updates.items():
            # Check for anomalies (most strict at global level)
            if config.anomaly_detection and await is_anomalous_update(weights):
                anomalies.append(f"Anomalous update from organization {org_id}")
                continue

            # Cross-organization model consistency
            if config.model_validation:
                try:
                    other_weights = [
                        w for oid, w in organization_updates.items() if oid != org_id
                    ]
                    if other_weights:
                        # Use stricter consistency threshold for global level
                        consistent = await check_model_consistency(
                            weights, other_weights[0]
                        )
                        if not consistent:
                            anomalies.append(
                                f"Inconsistent model from organization {org_id}"
                            )
                            continue
                except Exception as e:
                    anomalies.append(
                        f"Consistency check failed for organization {org_id}: {str(e)}"
                    )
                    continue

            validated_updates[org_id] = weights

        if len(validated_updates) < config.min_contributors:
            raise ValueError(
                f"Insufficient valid organizations after validation: {len(validated_updates)} < {config.min_contributors}"
            )

        # Calculate weights for aggregation (organization-level)
        org_weights = await self._calculate_organization_weights(
            list(validated_updates.keys()), config.weighting_method
        )

        # Perform aggregation
        aggregator = self.level_aggregators[FederationLevel.GLOBAL]
        aggregated_weights = await aggregator.aggregate_weights(
            list(validated_updates.values()), org_weights
        )

        # Calculate performance metrics
        performance_metrics = {
            "convergence_rate": await self._calculate_convergence_rate(
                session_id, round_number
            ),
            "participation_rate": len(validated_updates) / len(organization_updates),
            "anomaly_rate": len(anomalies) / len(organization_updates),
            "aggregation_efficiency": 1.0
            - (time.time() - start_time) / config.max_wait_time,
            "global_consensus": await self._calculate_global_consensus(
                list(validated_updates.values())
            ),
        }

        result = AggregationResult(
            level=FederationLevel.GLOBAL,
            aggregated_weights=aggregated_weights,
            contributing_entities=list(validated_updates.keys()),
            aggregation_metadata={
                "strategy": config.strategy,
                "weights_used": org_weights,
                "config": config.to_dict(),
                "cross_domain_adaptation": True,
            },
            performance_metrics=performance_metrics,
            anomalies_detected=anomalies,
            validation_passed=len(anomalies) == 0,
            aggregation_time=time.time() - start_time,
            round_number=round_number,
        )

        await audit_log(
            "global_aggregation_completed",
            {"session_id": session_id, "result": result.to_dict()},
            "system",
        )

        self.logger.info(
            f"Completed global aggregation in {result.aggregation_time:.2f}s"
        )
        return result

    async def _calculate_participant_weights(
        self, silo_id: str, participant_ids: List[str], weighting_method: str
    ) -> List[float]:
        """Calculate weights for participant aggregation."""
        if weighting_method == "equal":
            return [1.0 / len(participant_ids)] * len(participant_ids)

        weights = []
        for participant_id in participant_ids:
            participant = self.org_manager.participants.get(participant_id)
            if not participant:
                weights.append(1.0 / len(participant_ids))
                continue

            if weighting_method == "data_size":
                data_size = participant.data_characteristics.get("sample_count", 1000)
                weights.append(data_size)

            elif weighting_method == "performance":
                accuracy = participant.performance_metrics.get("accuracy", 0.5)
                weights.append(accuracy)

            elif weighting_method == "adaptive":
                data_size = participant.data_characteristics.get("sample_count", 1000)
                accuracy = participant.performance_metrics.get("accuracy", 0.5)
                compute_power = participant.compute_resources.get("relative_power", 1.0)

                # Adaptive weighting: 40% data, 30% performance, 30% compute
                weight = 0.4 * data_size + 0.3 * accuracy + 0.3 * compute_power
                weights.append(weight)

            else:
                weights.append(1.0)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(participant_ids)] * len(participant_ids)

        return weights

    async def _calculate_silo_weights(
        self, org_id: str, silo_ids: List[str], weighting_method: str
    ) -> List[float]:
        """Calculate weights for silo aggregation."""
        if weighting_method == "equal":
            return [1.0 / len(silo_ids)] * len(silo_ids)

        weights = []
        for silo_id in silo_ids:
            silo = self.org_manager.silos.get(silo_id)
            if not silo:
                weights.append(1.0 / len(silo_ids))
                continue

            if weighting_method == "data_size":
                weights.append(silo.data_size)

            elif weighting_method == "performance":
                # Calculate average performance of participants in this silo
                participant_ids = self.org_manager.silo_participants.get(silo_id, set())
                avg_accuracy = 0.5
                if participant_ids:
                    accuracies = [
                        self.org_manager.participants[pid].performance_metrics.get(
                            "accuracy", 0.5
                        )
                        for pid in participant_ids
                        if pid in self.org_manager.participants
                    ]
                    if accuracies:
                        avg_accuracy = sum(accuracies) / len(accuracies)
                weights.append(avg_accuracy)

            elif weighting_method == "adaptive":
                data_size = silo.data_size
                participant_count = len(
                    self.org_manager.silo_participants.get(silo_id, set())
                )
                compute_rating = {"low": 0.5, "medium": 1.0, "high": 1.5}.get(
                    silo.compute_capacity, 1.0
                )

                # Adaptive weighting for silos
                weight = (
                    0.5 * data_size + 0.3 * participant_count + 0.2 * compute_rating
                )
                weights.append(weight)

            else:
                weights.append(1.0)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(silo_ids)] * len(silo_ids)

        return weights

    async def _calculate_organization_weights(
        self, org_ids: List[str], weighting_method: str
    ) -> List[float]:
        """Calculate weights for organization aggregation."""
        if weighting_method == "equal":
            return [1.0 / len(org_ids)] * len(org_ids)

        weights = []
        for org_id in org_ids:
            org = self.org_manager.organizations.get(org_id)
            if not org:
                weights.append(1.0 / len(org_ids))
                continue

            if weighting_method == "data_size":
                # Sum data sizes across all silos in the organization
                total_data = sum(
                    self.org_manager.silos[silo_id].data_size
                    for silo_id in self.org_manager.org_silos.get(org_id, set())
                    if silo_id in self.org_manager.silos
                )
                weights.append(total_data)

            elif weighting_method == "performance":
                # Calculate organization-wide performance
                weights.append(org.total_participants)  # Use participant count as proxy

            elif weighting_method == "adaptive":
                total_data = sum(
                    self.org_manager.silos[silo_id].data_size
                    for silo_id in self.org_manager.org_silos.get(org_id, set())
                    if silo_id in self.org_manager.silos
                )
                silo_count = len(self.org_manager.org_silos.get(org_id, set()))
                participant_count = org.total_participants

                # Adaptive weighting for organizations
                weight = 0.4 * total_data + 0.3 * participant_count + 0.3 * silo_count
                weights.append(weight)

            else:
                weights.append(1.0)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(org_ids)] * len(org_ids)

        return weights

    async def _calculate_convergence_rate(
        self, session_id: str, round_number: int
    ) -> float:
        """Calculate convergence rate for the session."""
        if session_id not in self.convergence_history:
            self.convergence_history[session_id] = []

        # Placeholder convergence calculation
        # In practice, this would compare current model performance with previous rounds
        if round_number <= 1:
            return 0.0

        # Simulate convergence improvement
        base_rate = max(0.1, 1.0 - round_number * 0.1)
        return base_rate

    async def _calculate_global_consensus(
        self, organization_weights: List[ModelWeights]
    ) -> float:
        """Calculate global consensus score among organization models."""
        if len(organization_weights) < 2:
            return 1.0

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(organization_weights)):
            for j in range(i + 1, len(organization_weights)):
                try:
                    # Simple cosine similarity between weight vectors
                    weights1 = np.concatenate(
                        [
                            layer.flatten()
                            for layer in organization_weights[i].layer_weights
                        ]
                    )
                    weights2 = np.concatenate(
                        [
                            layer.flatten()
                            for layer in organization_weights[j].layer_weights
                        ]
                    )

                    # Cosine similarity
                    dot_product = np.dot(weights1, weights2)
                    norms = np.linalg.norm(weights1) * np.linalg.norm(weights2)

                    if norms > 0:
                        similarity = dot_product / norms
                        similarities.append(similarity)
                except Exception as e:
                    self.logger.warning(f"Error calculating similarity: {e}")
                    similarities.append(0.5)  # Default moderate similarity

        if similarities:
            return float(np.mean(similarities))
        return 1.0

    def configure_level(self, level: FederationLevel, config: LevelAggregationConfig):
        """Configure aggregation for a specific level."""
        self.aggregation_configs[level] = config

        # Update aggregator
        fedavg_config = FedAvgConfig(
            min_participants=config.min_contributors,
            convergence_threshold=config.convergence_threshold,
            max_rounds=1,
            patience=1,
            adaptive_learning_rate=True,
            byzantine_tolerance=config.byzantine_tolerance,
        )
        self.level_aggregators[level] = AsyncAggregator(fedavg_config)

        self.logger.info(f"Updated configuration for {level.value} level aggregation")

    def get_aggregation_statistics(self) -> Dict[str, Any]:
        """Get statistics about aggregation performance."""
        stats = {
            "total_aggregations": len(self.aggregation_results),
            "aggregations_by_level": {},
            "average_aggregation_times": {},
            "validation_success_rates": {},
            "anomaly_detection_rates": {},
        }

        # Group results by level
        results_by_level = {}
        for result in self.aggregation_results.values():
            level = result.level.value
            if level not in results_by_level:
                results_by_level[level] = []
            results_by_level[level].append(result)

        # Calculate statistics per level
        for level, results in results_by_level.items():
            stats["aggregations_by_level"][level] = len(results)

            if results:
                # Average aggregation times
                avg_time = sum(r.aggregation_time for r in results) / len(results)
                stats["average_aggregation_times"][level] = avg_time

                # Validation success rates
                success_rate = sum(1 for r in results if r.validation_passed) / len(
                    results
                )
                stats["validation_success_rates"][level] = success_rate

                # Anomaly detection rates
                anomaly_rate = sum(len(r.anomalies_detected) for r in results) / len(
                    results
                )
                stats["anomaly_detection_rates"][level] = anomaly_rate

        return stats
