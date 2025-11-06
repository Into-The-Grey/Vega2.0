"""
Communication Optimization Coordinator for Federated Learning

This module provides intelligent coordination of compression strategies based on
network conditions, participant capabilities, and training requirements.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import asyncio
from abc import ABC, abstractmethod

from .compression_advanced import (
    CompressionAlgorithm,
    GradientSparsification,
    Quantization,
    Sketching,
    CompressionResult,
    SparsificationMethod,
    QuantizationMethod,
    SketchingMethod,
)

logger = logging.getLogger(__name__)


class NetworkCondition(Enum):
    """Network condition categories for compression strategy selection."""

    EXCELLENT = "excellent"  # High bandwidth, low latency
    GOOD = "good"  # Medium bandwidth, medium latency
    POOR = "poor"  # Low bandwidth, high latency
    CRITICAL = "critical"  # Very low bandwidth, very high latency


class CompressionStrategy(Enum):
    """Compression strategy types for different scenarios."""

    AGGRESSIVE = "aggressive"  # Maximum compression for poor networks
    BALANCED = "balanced"  # Balance compression vs accuracy
    CONSERVATIVE = "conservative"  # Minimal compression for good networks
    ADAPTIVE = "adaptive"  # Dynamic based on conditions


@dataclass
class NetworkMetrics:
    """Network performance metrics for compression decisions."""

    bandwidth_mbps: float = 0.0
    latency_ms: float = 0.0
    packet_loss: float = 0.0
    jitter_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def is_stale(self, max_age: float = 60.0) -> bool:
        """Check if metrics are too old to be reliable."""
        return (time.time() - self.timestamp) > max_age


@dataclass
class ParticipantProfile:
    """Profile of a federated learning participant's capabilities."""

    participant_id: str
    compute_power: float = 1.0  # Relative compute capability
    network_metrics: NetworkMetrics = field(default_factory=NetworkMetrics)
    preferred_compression: Optional[CompressionStrategy] = None
    compression_history: List[Dict[str, Any]] = field(default_factory=list)

    def get_network_condition(self) -> NetworkCondition:
        """Classify current network condition."""
        if self.network_metrics.is_stale():
            return NetworkCondition.POOR  # Conservative default

        bandwidth = self.network_metrics.bandwidth_mbps
        latency = self.network_metrics.latency_ms

        if bandwidth >= 100 and latency <= 50:
            return NetworkCondition.EXCELLENT
        elif bandwidth >= 50 and latency <= 100:
            return NetworkCondition.GOOD
        elif bandwidth >= 10 and latency <= 200:
            return NetworkCondition.POOR
        else:
            return NetworkCondition.CRITICAL


class CompressionOptimizer:
    """Optimizes compression parameters based on conditions and history."""

    def __init__(self):
        self.performance_history: List[Dict[str, Any]] = []
        self.strategy_cache: Dict[str, Dict[str, Any]] = {}

    def optimize_sparsification(
        self,
        condition: NetworkCondition,
        model_size: int,
        accuracy_threshold: float = 0.95,
    ) -> Dict[str, Any]:
        """Optimize sparsification parameters for given conditions."""

        base_configs = {
            NetworkCondition.EXCELLENT: {
                "method": SparsificationMethod.TOP_K,
                "sparsity_ratio": 0.1,  # Light compression
                "adaptive": True,
            },
            NetworkCondition.GOOD: {
                "method": SparsificationMethod.TOP_K,
                "sparsity_ratio": 0.3,  # Moderate compression
                "adaptive": True,
            },
            NetworkCondition.POOR: {
                "method": SparsificationMethod.MAGNITUDE_BASED,
                "sparsity_ratio": 0.5,  # Heavy compression
                "adaptive": True,
            },
            NetworkCondition.CRITICAL: {
                "method": SparsificationMethod.LAYERWISE_TOP_K,
                "sparsity_ratio": 0.8,  # Maximum compression
                "adaptive": True,
            },
        }

        config = base_configs[condition].copy()

        # Adjust for model size
        if model_size > 100_000_000:  # Very large model
            config["sparsity_ratio"] = min(config["sparsity_ratio"] + 0.2, 0.9)
        elif model_size < 1_000_000:  # Small model
            config["sparsity_ratio"] = max(config["sparsity_ratio"] - 0.1, 0.05)

        return config

    def optimize_quantization(
        self, condition: NetworkCondition, precision_requirements: str = "medium"
    ) -> Dict[str, Any]:
        """Optimize quantization parameters for given conditions."""

        precision_bits = {"low": 4, "medium": 8, "high": 16}

        base_configs = {
            NetworkCondition.EXCELLENT: {
                "method": QuantizationMethod.UNIFORM,
                "bits": precision_bits.get(precision_requirements, 8),
                "adaptive": False,
            },
            NetworkCondition.GOOD: {
                "method": QuantizationMethod.ADAPTIVE,
                "bits": precision_bits.get(precision_requirements, 8),
                "adaptive": True,
            },
            NetworkCondition.POOR: {
                "method": QuantizationMethod.STOCHASTIC,
                "bits": max(precision_bits.get(precision_requirements, 8) - 2, 4),
                "adaptive": True,
            },
            NetworkCondition.CRITICAL: {
                "method": QuantizationMethod.SIGNSGD,
                "bits": 1,  # Binary quantization
                "adaptive": True,
            },
        }

        return base_configs[condition]

    def optimize_sketching(
        self, condition: NetworkCondition, model_size: int
    ) -> Dict[str, Any]:
        """Optimize sketching parameters for given conditions."""

        # Base sketch sizes based on network condition
        base_sketch_sizes = {
            NetworkCondition.EXCELLENT: min(model_size // 100, 10000),
            NetworkCondition.GOOD: min(model_size // 500, 5000),
            NetworkCondition.POOR: min(model_size // 1000, 2000),
            NetworkCondition.CRITICAL: min(model_size // 2000, 1000),
        }

        sketch_methods = {
            NetworkCondition.EXCELLENT: SketchingMethod.RANDOM_PROJECTION,
            NetworkCondition.GOOD: SketchingMethod.JOHNSON_LINDENSTRAUSS,
            NetworkCondition.POOR: SketchingMethod.COUNT_SKETCH,
            NetworkCondition.CRITICAL: SketchingMethod.FEATURE_HASHING,
        }

        return {
            "method": sketch_methods[condition],
            "sketch_size": max(
                base_sketch_sizes[condition], 100
            ),  # Minimum sketch size
            "adaptive": True,
        }


class CommunicationCoordinator:
    """
    Main coordinator for federated learning communication optimization.

    Manages compression strategies, participant profiles, and adaptive optimization
    based on network conditions and training requirements.
    """

    def __init__(
        self,
        strategy: CompressionStrategy = CompressionStrategy.ADAPTIVE,
        monitoring_interval: float = 30.0,
    ):
        self.strategy = strategy
        self.monitoring_interval = monitoring_interval
        self.participants: Dict[str, ParticipantProfile] = {}
        self.optimizer = CompressionOptimizer()
        self.global_metrics: NetworkMetrics = NetworkMetrics()
        self.compression_algorithms: Dict[str, CompressionAlgorithm] = {}
        self.performance_stats: Dict[str, List[float]] = {
            "compression_ratios": [],
            "compression_errors": [],
            "compression_times": [],
            "transmission_times": [],
        }

        # Initialize default compression algorithms
        self._initialize_algorithms()

    def _initialize_algorithms(self):
        """Initialize default compression algorithms."""
        # Create default compression config
        from .compression_advanced import CompressionConfig, CompressionType

        default_config = CompressionConfig(
            compression_type=CompressionType.SPARSIFICATION,
            compression_ratio=0.1,
            enable_error_feedback=True,
        )

        self.compression_algorithms = {
            "sparsification": GradientSparsification(default_config),
            "quantization": Quantization(default_config),
            "sketching": Sketching(default_config),
        }

    def _create_compression_config(self, **kwargs) -> "CompressionConfig":
        """Helper to create compression config from keyword arguments."""
        from .compression_advanced import CompressionConfig, CompressionType

        # Map old API parameters to new config
        config_params = {}

        if "bits" in kwargs:
            config_params["quantization_bits"] = kwargs["bits"]
        if "sketch_size" in kwargs:
            config_params["sketch_size"] = kwargs["sketch_size"]
        if "sparsity_ratio" in kwargs:
            config_params["compression_ratio"] = kwargs["sparsity_ratio"]

        # Determine compression type based on what's being created
        if "method" in kwargs:
            if hasattr(kwargs["method"], "name"):
                method_name = kwargs["method"].name
                if "QUANT" in method_name or "SIGNSGD" in method_name:
                    config_params["compression_type"] = CompressionType.QUANTIZATION
                elif "SKETCH" in method_name:
                    config_params["compression_type"] = CompressionType.SKETCHING
                else:
                    config_params["compression_type"] = CompressionType.SPARSIFICATION

        return CompressionConfig(**config_params)

    async def register_participant(
        self, participant_id: str, capabilities: Dict[str, Any]
    ) -> bool:
        """Register a new participant with their capabilities."""
        try:
            profile = ParticipantProfile(
                participant_id=participant_id,
                compute_power=capabilities.get("compute_power", 1.0),
                preferred_compression=capabilities.get("preferred_compression"),
            )

            self.participants[participant_id] = profile
            logger.info(f"Registered participant {participant_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to register participant {participant_id}: {e}")
            return False

    async def update_network_metrics(
        self, participant_id: str, metrics: NetworkMetrics
    ) -> None:
        """Update network metrics for a participant."""
        if participant_id in self.participants:
            self.participants[participant_id].network_metrics = metrics

            # Update global metrics (simple average)
            all_metrics = [
                p.network_metrics
                for p in self.participants.values()
                if not p.network_metrics.is_stale()
            ]

            if all_metrics:
                self.global_metrics.bandwidth_mbps = np.mean(
                    [m.bandwidth_mbps for m in all_metrics]
                )
                self.global_metrics.latency_ms = np.mean(
                    [m.latency_ms for m in all_metrics]
                )
                self.global_metrics.packet_loss = np.mean(
                    [m.packet_loss for m in all_metrics]
                )
                self.global_metrics.jitter_ms = np.mean(
                    [m.jitter_ms for m in all_metrics]
                )
                self.global_metrics.timestamp = time.time()

    def select_compression_strategy(
        self,
        participant_id: str,
        model_gradients: Dict[str, np.ndarray],
        training_round: int,
    ) -> List[CompressionAlgorithm]:
        """Select optimal compression strategy for a participant."""

        if participant_id not in self.participants:
            logger.warning(
                f"Unknown participant {participant_id}, using default strategy"
            )
            return [self.compression_algorithms["quantization"]]

        participant = self.participants[participant_id]
        condition = participant.get_network_condition()
        model_size = sum(grad.size for grad in model_gradients.values())

        strategies = []

        if self.strategy == CompressionStrategy.ADAPTIVE:
            strategies = self._adaptive_strategy_selection(
                condition, model_size, training_round
            )
        elif self.strategy == CompressionStrategy.AGGRESSIVE:
            strategies = self._aggressive_strategy_selection(condition, model_size)
        elif self.strategy == CompressionStrategy.CONSERVATIVE:
            strategies = self._conservative_strategy_selection(condition, model_size)
        else:  # BALANCED
            strategies = self._balanced_strategy_selection(condition, model_size)

        # Log strategy selection
        strategy_names = [type(s).__name__ for s in strategies]
        logger.info(
            f"Selected compression strategies for {participant_id}: {strategy_names}"
        )

        return strategies

    def _adaptive_strategy_selection(
        self, condition: NetworkCondition, model_size: int, training_round: int
    ) -> List[CompressionAlgorithm]:
        """Adaptive strategy selection based on conditions and history."""

        strategies = []

        if condition == NetworkCondition.EXCELLENT:
            # Light compression, focus on accuracy
            config = self.optimizer.optimize_quantization(condition, "high")
            comp_config = self._create_compression_config(
                method=config["method"], bits=config["bits"]
            )
            strategies.append(Quantization(comp_config, method=config["method"]))

        elif condition == NetworkCondition.GOOD:
            # Balanced compression
            sparsity_config = self.optimizer.optimize_sparsification(
                condition, model_size
            )
            quant_config = self.optimizer.optimize_quantization(condition, "medium")

            sparse_comp_config = self._create_compression_config(
                method=sparsity_config["method"],
                sparsity_ratio=sparsity_config["sparsity_ratio"],
            )
            quant_comp_config = self._create_compression_config(
                method=quant_config["method"], bits=quant_config["bits"]
            )

            strategies.extend(
                [
                    GradientSparsification(
                        sparse_comp_config, method=sparsity_config["method"]
                    ),
                    Quantization(quant_comp_config, method=quant_config["method"]),
                ]
            )

        elif condition == NetworkCondition.POOR:
            # Heavy compression
            sparsity_config = self.optimizer.optimize_sparsification(
                condition, model_size
            )
            quant_config = self.optimizer.optimize_quantization(condition, "low")

            sparse_comp_config = self._create_compression_config(
                method=sparsity_config["method"],
                sparsity_ratio=sparsity_config["sparsity_ratio"],
            )
            quant_comp_config = self._create_compression_config(
                method=quant_config["method"], bits=quant_config["bits"]
            )

            strategies.extend(
                [
                    GradientSparsification(
                        sparse_comp_config, method=sparsity_config["method"]
                    ),
                    Quantization(quant_comp_config, method=quant_config["method"]),
                ]
            )

        else:  # CRITICAL
            # Maximum compression with sketching
            sketch_config = self.optimizer.optimize_sketching(condition, model_size)
            quant_config = self.optimizer.optimize_quantization(condition, "low")

            sketch_comp_config = self._create_compression_config(
                method=sketch_config["method"], sketch_size=sketch_config["sketch_size"]
            )
            quant_comp_config = self._create_compression_config(
                method=quant_config["method"], bits=quant_config["bits"]
            )

            strategies.extend(
                [
                    Sketching(sketch_comp_config, method=sketch_config["method"]),
                    Quantization(quant_comp_config, method=quant_config["method"]),
                ]
            )

        return strategies

    def _aggressive_strategy_selection(
        self, condition: NetworkCondition, model_size: int
    ) -> List[CompressionAlgorithm]:
        """Aggressive compression for minimal bandwidth usage."""

        sparsity_config = self.optimizer.optimize_sparsification(condition, model_size)
        sketch_config = self.optimizer.optimize_sketching(condition, model_size)

        sparse_comp_config = self._create_compression_config(
            method=sparsity_config["method"],
            sparsity_ratio=min(sparsity_config["sparsity_ratio"] + 0.2, 0.9),
        )
        sketch_comp_config = self._create_compression_config(
            method=sketch_config["method"],
            sketch_size=max(sketch_config["sketch_size"] // 2, 50),
        )
        quant_comp_config = self._create_compression_config(
            method=QuantizationMethod.SIGNSGD, bits=1
        )

        return [
            GradientSparsification(
                sparse_comp_config, method=sparsity_config["method"]
            ),
            Sketching(sketch_comp_config, method=sketch_config["method"]),
            Quantization(quant_comp_config, method=QuantizationMethod.SIGNSGD),
        ]

    def _conservative_strategy_selection(
        self, condition: NetworkCondition, model_size: int
    ) -> List[CompressionAlgorithm]:
        """Conservative compression preserving accuracy."""

        comp_config = self._create_compression_config(
            method=QuantizationMethod.UNIFORM, bits=16
        )
        return [
            Quantization(
                comp_config, method=QuantizationMethod.UNIFORM
            )  # High precision
        ]

    def _balanced_strategy_selection(
        self, condition: NetworkCondition, model_size: int
    ) -> List[CompressionAlgorithm]:
        """Balanced compression strategy."""

        sparsity_config = self.optimizer.optimize_sparsification(condition, model_size)
        quant_config = self.optimizer.optimize_quantization(condition, "medium")

        sparse_comp_config = self._create_compression_config(
            method=sparsity_config["method"],
            sparsity_ratio=sparsity_config["sparsity_ratio"],
        )
        quant_comp_config = self._create_compression_config(
            method=quant_config["method"], bits=quant_config["bits"]
        )

        return [
            GradientSparsification(
                sparse_comp_config, method=sparsity_config["method"]
            ),
            Quantization(quant_comp_config, method=quant_config["method"]),
        ]

    async def compress_and_transmit(
        self,
        participant_id: str,
        model_gradients: Dict[str, np.ndarray],
        training_round: int,
    ) -> Tuple[CompressionResult, float]:
        """Compress gradients and simulate transmission."""

        # Select compression strategy
        strategies = self.select_compression_strategy(
            participant_id, model_gradients, training_round
        )

        # Apply compression strategies sequentially
        compressed_data = model_gradients
        total_compression_time = 0.0
        final_result = None

        for strategy in strategies:
            result = strategy.compress(compressed_data)
            compressed_data = strategy.decompress(result)
            total_compression_time += result.compression_time
            final_result = result

        # Simulate transmission time based on network conditions
        if participant_id in self.participants:
            participant = self.participants[participant_id]
            bandwidth_mbps = participant.network_metrics.bandwidth_mbps

            if bandwidth_mbps > 0:
                data_size_mb = final_result.compressed_size / (
                    1024 * 1024
                )  # Convert bytes to MB
                transmission_time = data_size_mb / bandwidth_mbps
            else:
                transmission_time = 10.0  # Default assumption
        else:
            transmission_time = 5.0  # Default assumption

        # Update performance statistics
        if final_result:
            self.performance_stats["compression_ratios"].append(
                final_result.compression_ratio
            )
            self.performance_stats["compression_errors"].append(
                final_result.compression_error
            )
            self.performance_stats["compression_times"].append(total_compression_time)
            self.performance_stats["transmission_times"].append(transmission_time)

        return final_result, transmission_time

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of communication performance."""

        if not self.performance_stats["compression_ratios"]:
            return {"status": "No data available"}

        return {
            "avg_compression_ratio": float(
                np.mean(self.performance_stats["compression_ratios"])
            ),
            "avg_compression_error": float(
                np.mean(self.performance_stats["compression_errors"])
            ),
            "avg_compression_time": float(
                np.mean(self.performance_stats["compression_times"])
            ),
            "avg_transmission_time": float(
                np.mean(self.performance_stats["transmission_times"])
            ),
            "total_rounds": len(self.performance_stats["compression_ratios"]),
            "participant_count": len(self.participants),
            "global_bandwidth": self.global_metrics.bandwidth_mbps,
            "global_latency": self.global_metrics.latency_ms,
        }

    async def optimize_communication(self) -> Dict[str, Any]:
        """Continuously optimize communication strategies."""

        optimization_results = {
            "participants_optimized": 0,
            "strategies_updated": 0,
            "performance_improvement": 0.0,
        }

        # Analyze performance history and adjust strategies
        for participant_id, participant in self.participants.items():
            # Update network condition classification
            condition = participant.get_network_condition()

            # Analyze compression history for this participant
            if len(participant.compression_history) > 5:
                recent_performance = participant.compression_history[-5:]
                avg_error = np.mean(
                    [h.get("compression_error", 0) for h in recent_performance]
                )
                avg_time = np.mean(
                    [h.get("compression_time", 0) for h in recent_performance]
                )

                # Adjust strategy based on performance
                if avg_error > 0.5 and condition != NetworkCondition.CRITICAL:
                    # High error, reduce compression
                    participant.preferred_compression = CompressionStrategy.CONSERVATIVE
                    optimization_results["strategies_updated"] += 1
                elif avg_time > 1.0:
                    # Slow compression, switch to faster methods
                    participant.preferred_compression = CompressionStrategy.BALANCED
                    optimization_results["strategies_updated"] += 1

            optimization_results["participants_optimized"] += 1

        return optimization_results

    async def start_monitoring(self):
        """Start continuous monitoring and optimization."""
        logger.info("Starting communication coordinator monitoring")

        while True:
            try:
                # Perform optimization
                results = await self.optimize_communication()
                logger.info(f"Optimization results: {results}")

                # Wait for next monitoring interval
                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in communication monitoring: {e}")
                await asyncio.sleep(self.monitoring_interval)


# Convenience functions for easy integration


async def create_coordinator(
    strategy: CompressionStrategy = CompressionStrategy.ADAPTIVE,
) -> CommunicationCoordinator:
    """Create and initialize a communication coordinator."""
    coordinator = CommunicationCoordinator(strategy=strategy)
    return coordinator


def estimate_communication_savings(
    original_size_mb: float,
    compression_ratio: float,
    bandwidth_mbps: float,
    participant_count: int,
) -> Dict[str, float]:
    """Estimate communication savings from compression."""

    compressed_size_mb = original_size_mb * (1 - compression_ratio)

    # Calculate transmission times
    original_time = (original_size_mb * participant_count) / bandwidth_mbps
    compressed_time = (compressed_size_mb * participant_count) / bandwidth_mbps

    time_savings = original_time - compressed_time
    bandwidth_savings = (original_size_mb - compressed_size_mb) * participant_count

    return {
        "original_transmission_time": original_time,
        "compressed_transmission_time": compressed_time,
        "time_savings_seconds": time_savings,
        "bandwidth_savings_mb": bandwidth_savings,
        "efficiency_improvement": (
            (time_savings / original_time) * 100 if original_time > 0 else 0
        ),
    }
