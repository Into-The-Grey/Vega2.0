"""
Distributed Compression Validation Suite for Federated Learning

This module provides comprehensive testing and validation of compression algorithms
in distributed federated learning environments with multi-participant scenarios,
performance benchmarking, and automated effectiveness analysis.
"""

import asyncio
import time
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
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

from .communication_coordinator import (
    CommunicationCoordinator,
    NetworkMetrics,
    NetworkCondition,
    CompressionStrategy,
    ParticipantProfile,
)

logger = logging.getLogger(__name__)


class TestScenario(Enum):
    """Different test scenarios for validation."""

    HOMOGENEOUS_NETWORK = "homogeneous_network"  # All participants have similar network
    HETEROGENEOUS_NETWORK = "heterogeneous_network"  # Mixed network conditions
    DYNAMIC_NETWORK = "dynamic_network"  # Network conditions change over time
    LARGE_SCALE = "large_scale"  # Many participants
    RESOURCE_CONSTRAINED = "resource_constrained"  # Limited bandwidth/compute
    HIGH_ACCURACY = "high_accuracy"  # Accuracy-critical scenarios


class ModelType(Enum):
    """Different neural network model types for testing."""

    SMALL_CNN = "small_cnn"  # Small convolutional network
    LARGE_CNN = "large_cnn"  # Large convolutional network
    TRANSFORMER = "transformer"  # Transformer model
    MLP = "mlp"  # Multi-layer perceptron
    CUSTOM = "custom"  # Custom model definition


@dataclass
class ValidationMetrics:
    """Comprehensive metrics for validation analysis."""

    test_id: str
    scenario: TestScenario
    model_type: ModelType
    participant_count: int
    network_conditions: List[NetworkCondition]

    # Compression Performance
    avg_compression_ratio: float = 0.0
    min_compression_ratio: float = 0.0
    max_compression_ratio: float = 0.0
    std_compression_ratio: float = 0.0

    # Accuracy Metrics
    avg_compression_error: float = 0.0
    max_compression_error: float = 0.0
    convergence_rounds: int = 0
    final_accuracy: float = 0.0

    # Performance Metrics
    avg_compression_time: float = 0.0
    avg_transmission_time: float = 0.0
    total_bandwidth_saved: float = 0.0
    total_time_saved: float = 0.0

    # Resource Utilization
    memory_overhead: float = 0.0
    cpu_utilization: float = 0.0

    # Reliability Metrics
    success_rate: float = 0.0
    error_count: int = 0
    timeout_count: int = 0

    timestamp: float = field(default_factory=time.time)


@dataclass
class BenchmarkResult:
    """Results from comparative benchmarking."""

    algorithm_name: str
    scenario: TestScenario
    metrics: ValidationMetrics
    relative_performance: float = 0.0  # Compared to baseline
    ranking: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm_name": self.algorithm_name,
            "scenario": self.scenario.value,
            "metrics": asdict(self.metrics),
            "relative_performance": self.relative_performance,
            "ranking": self.ranking,
        }


class ModelGenerator:
    """Generates different types of neural network models for testing."""

    @staticmethod
    def generate_model(
        model_type: ModelType, size_factor: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """Generate model gradients for testing."""

        if model_type == ModelType.SMALL_CNN:
            return ModelGenerator._generate_small_cnn(size_factor)
        elif model_type == ModelType.LARGE_CNN:
            return ModelGenerator._generate_large_cnn(size_factor)
        elif model_type == ModelType.TRANSFORMER:
            return ModelGenerator._generate_transformer(size_factor)
        elif model_type == ModelType.MLP:
            return ModelGenerator._generate_mlp(size_factor)
        else:
            return ModelGenerator._generate_small_cnn(size_factor)  # Default

    @staticmethod
    def _generate_small_cnn(size_factor: float) -> Dict[str, np.ndarray]:
        """Generate small CNN model."""
        base_size = int(32 * size_factor)
        return {
            "conv1.weight": np.random.normal(0, 0.1, (base_size, 3, 5, 5)),
            "conv1.bias": np.random.normal(0, 0.1, (base_size,)),
            "conv2.weight": np.random.normal(0, 0.1, (base_size * 2, base_size, 3, 3)),
            "conv2.bias": np.random.normal(0, 0.1, (base_size * 2,)),
            "fc1.weight": np.random.normal(0, 0.1, (128, base_size * 2 * 7 * 7)),
            "fc1.bias": np.random.normal(0, 0.1, (128,)),
            "fc2.weight": np.random.normal(0, 0.1, (10, 128)),
            "fc2.bias": np.random.normal(0, 0.1, (10,)),
        }

    @staticmethod
    def _generate_large_cnn(size_factor: float) -> Dict[str, np.ndarray]:
        """Generate large CNN model."""
        base_size = int(128 * size_factor)
        return {
            "conv1.weight": np.random.normal(0, 0.1, (base_size, 3, 7, 7)),
            "conv1.bias": np.random.normal(0, 0.1, (base_size,)),
            "conv2.weight": np.random.normal(0, 0.1, (base_size * 2, base_size, 5, 5)),
            "conv2.bias": np.random.normal(0, 0.1, (base_size * 2,)),
            "conv3.weight": np.random.normal(
                0, 0.1, (base_size * 4, base_size * 2, 3, 3)
            ),
            "conv3.bias": np.random.normal(0, 0.1, (base_size * 4,)),
            "fc1.weight": np.random.normal(0, 0.1, (1000, base_size * 4 * 4 * 4)),
            "fc1.bias": np.random.normal(0, 0.1, (1000,)),
            "fc2.weight": np.random.normal(0, 0.1, (500, 1000)),
            "fc2.bias": np.random.normal(0, 0.1, (500,)),
            "fc3.weight": np.random.normal(0, 0.1, (100, 500)),
            "fc3.bias": np.random.normal(0, 0.1, (100,)),
        }

    @staticmethod
    def _generate_transformer(size_factor: float) -> Dict[str, np.ndarray]:
        """Generate transformer model."""
        d_model = int(512 * size_factor)
        d_ff = d_model * 4
        return {
            "embedding.weight": np.random.normal(0, 0.1, (10000, d_model)),
            "pos_encoding.weight": np.random.normal(0, 0.1, (1000, d_model)),
            "attention.query.weight": np.random.normal(0, 0.1, (d_model, d_model)),
            "attention.key.weight": np.random.normal(0, 0.1, (d_model, d_model)),
            "attention.value.weight": np.random.normal(0, 0.1, (d_model, d_model)),
            "attention.out.weight": np.random.normal(0, 0.1, (d_model, d_model)),
            "ffn.fc1.weight": np.random.normal(0, 0.1, (d_ff, d_model)),
            "ffn.fc1.bias": np.random.normal(0, 0.1, (d_ff,)),
            "ffn.fc2.weight": np.random.normal(0, 0.1, (d_model, d_ff)),
            "ffn.fc2.bias": np.random.normal(0, 0.1, (d_model,)),
            "layer_norm1.weight": np.random.normal(1, 0.1, (d_model,)),
            "layer_norm1.bias": np.random.normal(0, 0.1, (d_model,)),
            "layer_norm2.weight": np.random.normal(1, 0.1, (d_model,)),
            "layer_norm2.bias": np.random.normal(0, 0.1, (d_model,)),
        }

    @staticmethod
    def _generate_mlp(size_factor: float) -> Dict[str, np.ndarray]:
        """Generate MLP model."""
        base_size = int(256 * size_factor)
        return {
            "fc1.weight": np.random.normal(0, 0.1, (base_size, 784)),
            "fc1.bias": np.random.normal(0, 0.1, (base_size,)),
            "fc2.weight": np.random.normal(0, 0.1, (base_size * 2, base_size)),
            "fc2.bias": np.random.normal(0, 0.1, (base_size * 2,)),
            "fc3.weight": np.random.normal(0, 0.1, (base_size, base_size * 2)),
            "fc3.bias": np.random.normal(0, 0.1, (base_size,)),
            "fc4.weight": np.random.normal(0, 0.1, (10, base_size)),
            "fc4.bias": np.random.normal(0, 0.1, (10,)),
        }


class NetworkSimulator:
    """Simulates different network conditions for testing."""

    @staticmethod
    def generate_network_conditions(
        scenario: TestScenario, participant_count: int
    ) -> List[NetworkMetrics]:
        """Generate network conditions for different scenarios."""

        if scenario == TestScenario.HOMOGENEOUS_NETWORK:
            return NetworkSimulator._homogeneous_network(participant_count)
        elif scenario == TestScenario.HETEROGENEOUS_NETWORK:
            return NetworkSimulator._heterogeneous_network(participant_count)
        elif scenario == TestScenario.DYNAMIC_NETWORK:
            return NetworkSimulator._dynamic_network(participant_count)
        elif scenario == TestScenario.LARGE_SCALE:
            return NetworkSimulator._large_scale_network(participant_count)
        elif scenario == TestScenario.RESOURCE_CONSTRAINED:
            return NetworkSimulator._resource_constrained_network(participant_count)
        else:  # HIGH_ACCURACY
            return NetworkSimulator._high_accuracy_network(participant_count)

    @staticmethod
    def _homogeneous_network(count: int) -> List[NetworkMetrics]:
        """All participants have similar good network conditions."""
        return [
            NetworkMetrics(bandwidth_mbps=100, latency_ms=50, packet_loss=0.001)
            for _ in range(count)
        ]

    @staticmethod
    def _heterogeneous_network(count: int) -> List[NetworkMetrics]:
        """Mixed network conditions across participants."""
        conditions = []
        for i in range(count):
            if i < count // 4:  # 25% excellent
                conditions.append(NetworkMetrics(bandwidth_mbps=150, latency_ms=20))
            elif i < count // 2:  # 25% good
                conditions.append(NetworkMetrics(bandwidth_mbps=75, latency_ms=80))
            elif i < 3 * count // 4:  # 25% poor
                conditions.append(NetworkMetrics(bandwidth_mbps=20, latency_ms=180))
            else:  # 25% critical
                conditions.append(NetworkMetrics(bandwidth_mbps=5, latency_ms=300))
        return conditions

    @staticmethod
    def _dynamic_network(count: int) -> List[NetworkMetrics]:
        """Dynamic conditions that change over time."""
        # Start with varying conditions
        return NetworkSimulator._heterogeneous_network(count)

    @staticmethod
    def _large_scale_network(count: int) -> List[NetworkMetrics]:
        """Many participants with realistic distribution."""
        conditions = []
        for i in range(count):
            # Realistic distribution: mostly good/poor, few excellent/critical
            rand = np.random.random()
            if rand < 0.1:  # 10% excellent
                conditions.append(NetworkMetrics(bandwidth_mbps=200, latency_ms=15))
            elif rand < 0.4:  # 30% good
                conditions.append(NetworkMetrics(bandwidth_mbps=80, latency_ms=60))
            elif rand < 0.8:  # 40% poor
                conditions.append(NetworkMetrics(bandwidth_mbps=25, latency_ms=150))
            else:  # 20% critical
                conditions.append(NetworkMetrics(bandwidth_mbps=8, latency_ms=250))
        return conditions

    @staticmethod
    def _resource_constrained_network(count: int) -> List[NetworkMetrics]:
        """Limited bandwidth for all participants."""
        return [
            NetworkMetrics(
                bandwidth_mbps=np.random.uniform(5, 15),
                latency_ms=np.random.uniform(150, 300),
                packet_loss=np.random.uniform(0.01, 0.05),
            )
            for _ in range(count)
        ]

    @staticmethod
    def _high_accuracy_network(count: int) -> List[NetworkMetrics]:
        """Excellent network for accuracy-critical scenarios."""
        return [
            NetworkMetrics(bandwidth_mbps=200, latency_ms=10, packet_loss=0.0001)
            for _ in range(count)
        ]


class CompressionValidator:
    """Validates compression algorithms in distributed scenarios."""

    def __init__(self):
        self.test_results: List[ValidationMetrics] = []
        self.benchmark_results: List[BenchmarkResult] = []

    async def run_comprehensive_validation(
        self,
        scenarios: List[TestScenario],
        model_types: List[ModelType],
        participant_counts: List[int],
        algorithms: List[CompressionAlgorithm],
    ) -> List[ValidationMetrics]:
        """Run comprehensive validation across all scenarios."""

        all_results = []
        total_tests = (
            len(scenarios)
            * len(model_types)
            * len(participant_counts)
            * len(algorithms)
        )
        test_count = 0

        logger.info(
            f"Starting comprehensive validation: {total_tests} test configurations"
        )

        for scenario in scenarios:
            for model_type in model_types:
                for participant_count in participant_counts:
                    for algorithm in algorithms:
                        test_count += 1
                        logger.info(f"Running test {test_count}/{total_tests}")

                        try:
                            result = await self._run_single_test(
                                scenario, model_type, participant_count, algorithm
                            )
                            all_results.append(result)
                        except Exception as e:
                            logger.error(f"Test {test_count} failed: {e}")
                            # Continue with other tests

        self.test_results.extend(all_results)
        return all_results

    async def _run_single_test(
        self,
        scenario: TestScenario,
        model_type: ModelType,
        participant_count: int,
        algorithm: CompressionAlgorithm,
    ) -> ValidationMetrics:
        """Run a single validation test."""

        test_id = f"{scenario.value}_{model_type.value}_{participant_count}_{type(algorithm).__name__}"

        # Generate test model
        model_gradients = ModelGenerator.generate_model(model_type)

        # Generate network conditions
        network_conditions = NetworkSimulator.generate_network_conditions(
            scenario, participant_count
        )

        # Set up coordinator
        coordinator = CommunicationCoordinator(strategy=CompressionStrategy.ADAPTIVE)

        # Register participants
        for i, network_metrics in enumerate(network_conditions):
            participant_id = f"participant_{i}"
            await coordinator.register_participant(
                participant_id, {"compute_power": np.random.uniform(0.5, 2.0)}
            )
            await coordinator.update_network_metrics(participant_id, network_metrics)

        # Run compression tests
        compression_results = []
        transmission_times = []
        errors = []
        timeouts = 0

        for round_num in range(5):  # Multiple rounds for stability
            round_results = []
            round_times = []

            for i in range(participant_count):
                participant_id = f"participant_{i}"

                try:
                    start_time = time.time()

                    # Test compression
                    compressed_result = algorithm.compress(model_gradients)
                    decompressed_gradients = algorithm.decompress(compressed_result)

                    # Calculate compression error
                    compression_error = self._calculate_compression_error(
                        model_gradients, decompressed_gradients
                    )

                    # Simulate transmission
                    network_metrics = network_conditions[i]
                    transmission_time = self._simulate_transmission(
                        compressed_result.compressed_size_kb, network_metrics
                    )

                    total_time = time.time() - start_time

                    if total_time > 60:  # Timeout threshold
                        timeouts += 1
                        continue

                    round_results.append(
                        {
                            "compression_ratio": compressed_result.compression_ratio,
                            "compression_error": compression_error,
                            "compression_time": compressed_result.compression_time,
                            "transmission_time": transmission_time,
                            "total_time": total_time,
                        }
                    )

                except Exception as e:
                    errors.append(str(e))
                    logger.warning(f"Error in test for {participant_id}: {e}")

            compression_results.extend(round_results)

        # Calculate metrics
        if compression_results:
            metrics = self._calculate_validation_metrics(
                test_id,
                scenario,
                model_type,
                participant_count,
                [
                    (
                        nc.get_network_condition()
                        if hasattr(nc, "get_network_condition")
                        else self._classify_network_condition(nc)
                    )
                    for nc in network_conditions
                ],
                compression_results,
                len(errors),
                timeouts,
            )
        else:
            # Create empty metrics if no successful results
            metrics = ValidationMetrics(
                test_id=test_id,
                scenario=scenario,
                model_type=model_type,
                participant_count=participant_count,
                network_conditions=[],
                success_rate=0.0,
                error_count=len(errors),
                timeout_count=timeouts,
            )

        return metrics

    def _classify_network_condition(self, metrics: NetworkMetrics) -> NetworkCondition:
        """Classify network condition from metrics."""
        bandwidth = metrics.bandwidth_mbps
        latency = metrics.latency_ms

        if bandwidth >= 100 and latency <= 50:
            return NetworkCondition.EXCELLENT
        elif bandwidth >= 50 and latency <= 100:
            return NetworkCondition.GOOD
        elif bandwidth >= 10 and latency <= 200:
            return NetworkCondition.POOR
        else:
            return NetworkCondition.CRITICAL

    def _calculate_compression_error(
        self, original: Dict[str, np.ndarray], reconstructed: Dict[str, np.ndarray]
    ) -> float:
        """Calculate compression error between original and reconstructed gradients."""
        total_error = 0.0
        total_norm = 0.0

        for key in original.keys():
            if key in reconstructed:
                orig_norm = np.linalg.norm(original[key])
                error_norm = np.linalg.norm(original[key] - reconstructed[key])

                if orig_norm > 0:
                    total_error += error_norm
                    total_norm += orig_norm

        return total_error / total_norm if total_norm > 0 else 0.0

    def _simulate_transmission(
        self, size_kb: float, network_metrics: NetworkMetrics
    ) -> float:
        """Simulate network transmission time."""
        if network_metrics.bandwidth_mbps <= 0:
            return 10.0  # Default assumption

        size_mb = size_kb / 1024
        base_time = size_mb / network_metrics.bandwidth_mbps

        # Add latency impact
        latency_factor = 1 + (network_metrics.latency_ms / 1000.0)

        # Add packet loss impact
        loss_factor = (
            1 / (1 - network_metrics.packet_loss)
            if network_metrics.packet_loss < 0.99
            else 2.0
        )

        return base_time * latency_factor * loss_factor

    def _calculate_validation_metrics(
        self,
        test_id: str,
        scenario: TestScenario,
        model_type: ModelType,
        participant_count: int,
        network_conditions: List[NetworkCondition],
        results: List[Dict[str, Any]],
        error_count: int,
        timeout_count: int,
    ) -> ValidationMetrics:
        """Calculate comprehensive validation metrics."""

        if not results:
            return ValidationMetrics(
                test_id=test_id,
                scenario=scenario,
                model_type=model_type,
                participant_count=participant_count,
                network_conditions=network_conditions,
                success_rate=0.0,
                error_count=error_count,
                timeout_count=timeout_count,
            )

        # Extract values for analysis
        compression_ratios = [r["compression_ratio"] for r in results]
        compression_errors = [r["compression_error"] for r in results]
        compression_times = [r["compression_time"] for r in results]
        transmission_times = [r["transmission_time"] for r in results]

        # Calculate statistics
        total_attempts = len(results) + error_count + timeout_count
        success_rate = len(results) / total_attempts if total_attempts > 0 else 0.0

        return ValidationMetrics(
            test_id=test_id,
            scenario=scenario,
            model_type=model_type,
            participant_count=participant_count,
            network_conditions=network_conditions,
            # Compression Performance
            avg_compression_ratio=float(statistics.mean(compression_ratios)),
            min_compression_ratio=float(min(compression_ratios)),
            max_compression_ratio=float(max(compression_ratios)),
            std_compression_ratio=float(
                statistics.stdev(compression_ratios)
                if len(compression_ratios) > 1
                else 0
            ),
            # Accuracy Metrics
            avg_compression_error=float(statistics.mean(compression_errors)),
            max_compression_error=float(max(compression_errors)),
            # Performance Metrics
            avg_compression_time=float(statistics.mean(compression_times)),
            avg_transmission_time=float(statistics.mean(transmission_times)),
            # Reliability Metrics
            success_rate=success_rate,
            error_count=error_count,
            timeout_count=timeout_count,
        )

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.test_results:
            return {"status": "No test results available"}

        report = {
            "summary": {
                "total_tests": len(self.test_results),
                "successful_tests": len(
                    [r for r in self.test_results if r.success_rate > 0.5]
                ),
                "avg_success_rate": statistics.mean(
                    [r.success_rate for r in self.test_results]
                ),
                "total_errors": sum([r.error_count for r in self.test_results]),
                "total_timeouts": sum([r.timeout_count for r in self.test_results]),
            },
            "compression_performance": {
                "avg_compression_ratio": statistics.mean(
                    [
                        r.avg_compression_ratio
                        for r in self.test_results
                        if r.avg_compression_ratio > 0
                    ]
                ),
                "avg_compression_error": statistics.mean(
                    [
                        r.avg_compression_error
                        for r in self.test_results
                        if r.avg_compression_error >= 0
                    ]
                ),
                "avg_compression_time": statistics.mean(
                    [
                        r.avg_compression_time
                        for r in self.test_results
                        if r.avg_compression_time > 0
                    ]
                ),
            },
            "scenario_analysis": self._analyze_by_scenario(),
            "model_analysis": self._analyze_by_model(),
            "scale_analysis": self._analyze_by_scale(),
            "algorithm_comparison": self._compare_algorithms(),
        }

        return report

    def _analyze_by_scenario(self) -> Dict[str, Dict[str, float]]:
        """Analyze results by test scenario."""
        scenario_analysis = {}

        for scenario in TestScenario:
            scenario_results = [r for r in self.test_results if r.scenario == scenario]
            if scenario_results:
                scenario_analysis[scenario.value] = {
                    "test_count": len(scenario_results),
                    "avg_success_rate": statistics.mean(
                        [r.success_rate for r in scenario_results]
                    ),
                    "avg_compression_ratio": statistics.mean(
                        [
                            r.avg_compression_ratio
                            for r in scenario_results
                            if r.avg_compression_ratio > 0
                        ]
                    ),
                    "avg_compression_error": statistics.mean(
                        [
                            r.avg_compression_error
                            for r in scenario_results
                            if r.avg_compression_error >= 0
                        ]
                    ),
                }

        return scenario_analysis

    def _analyze_by_model(self) -> Dict[str, Dict[str, float]]:
        """Analyze results by model type."""
        model_analysis = {}

        for model_type in ModelType:
            model_results = [r for r in self.test_results if r.model_type == model_type]
            if model_results:
                model_analysis[model_type.value] = {
                    "test_count": len(model_results),
                    "avg_compression_ratio": statistics.mean(
                        [
                            r.avg_compression_ratio
                            for r in model_results
                            if r.avg_compression_ratio > 0
                        ]
                    ),
                    "avg_compression_error": statistics.mean(
                        [
                            r.avg_compression_error
                            for r in model_results
                            if r.avg_compression_error >= 0
                        ]
                    ),
                    "avg_compression_time": statistics.mean(
                        [
                            r.avg_compression_time
                            for r in model_results
                            if r.avg_compression_time > 0
                        ]
                    ),
                }

        return model_analysis

    def _analyze_by_scale(self) -> Dict[str, Dict[str, float]]:
        """Analyze results by participant scale."""
        scale_analysis = {}

        # Group by participant count ranges
        scales = {
            "small": (1, 10),
            "medium": (11, 50),
            "large": (51, 200),
            "very_large": (201, float("inf")),
        }

        for scale_name, (min_count, max_count) in scales.items():
            scale_results = [
                r
                for r in self.test_results
                if min_count <= r.participant_count <= max_count
            ]
            if scale_results:
                scale_analysis[scale_name] = {
                    "test_count": len(scale_results),
                    "avg_participant_count": statistics.mean(
                        [r.participant_count for r in scale_results]
                    ),
                    "avg_success_rate": statistics.mean(
                        [r.success_rate for r in scale_results]
                    ),
                    "avg_transmission_time": statistics.mean(
                        [
                            r.avg_transmission_time
                            for r in scale_results
                            if r.avg_transmission_time > 0
                        ]
                    ),
                }

        return scale_analysis

    def _compare_algorithms(self) -> Dict[str, Dict[str, Any]]:
        """Compare algorithm performance."""
        algorithm_comparison = {}

        # Group results by algorithm (inferred from test_id)
        algorithm_groups = {}
        for result in self.test_results:
            # Extract algorithm name from test_id
            parts = result.test_id.split("_")
            if len(parts) >= 4:
                algorithm_name = parts[-1]
                if algorithm_name not in algorithm_groups:
                    algorithm_groups[algorithm_name] = []
                algorithm_groups[algorithm_name].append(result)

        # Calculate comparison metrics
        for algorithm_name, results in algorithm_groups.items():
            if results:
                algorithm_comparison[algorithm_name] = {
                    "test_count": len(results),
                    "avg_compression_ratio": statistics.mean(
                        [
                            r.avg_compression_ratio
                            for r in results
                            if r.avg_compression_ratio > 0
                        ]
                    ),
                    "avg_compression_error": statistics.mean(
                        [
                            r.avg_compression_error
                            for r in results
                            if r.avg_compression_error >= 0
                        ]
                    ),
                    "avg_success_rate": statistics.mean(
                        [r.success_rate for r in results]
                    ),
                    "reliability_score": statistics.mean(
                        [
                            r.success_rate * (1 - r.avg_compression_error)
                            for r in results
                            if r.avg_compression_error >= 0
                        ]
                    ),
                }

        return algorithm_comparison

    def export_results(self, filepath: str) -> bool:
        """Export test results to JSON file."""
        try:
            export_data = {
                "validation_results": [asdict(result) for result in self.test_results],
                "benchmark_results": [
                    result.to_dict() for result in self.benchmark_results
                ],
                "performance_report": self.generate_performance_report(),
                "export_timestamp": time.time(),
            }

            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Results exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            return False


# Convenience functions for easy testing


async def run_quick_validation() -> ValidationMetrics:
    """Run a quick validation test with basic scenarios."""
    validator = CompressionValidator()

    # Basic test configuration
    scenarios = [TestScenario.HOMOGENEOUS_NETWORK, TestScenario.HETEROGENEOUS_NETWORK]
    model_types = [ModelType.SMALL_CNN, ModelType.MLP]
    participant_counts = [5, 10]
    algorithms = [
        GradientSparsification(method=SparsificationMethod.TOP_K, sparsity_ratio=0.3),
        Quantization(method=QuantizationMethod.UNIFORM, bits=8),
    ]

    results = await validator.run_comprehensive_validation(
        scenarios, model_types, participant_counts, algorithms
    )

    return results


async def run_benchmark_comparison(
    algorithms: List[CompressionAlgorithm],
) -> List[BenchmarkResult]:
    """Run comparative benchmarking of compression algorithms."""
    validator = CompressionValidator()

    # Comprehensive benchmark
    scenarios = [TestScenario.HETEROGENEOUS_NETWORK, TestScenario.RESOURCE_CONSTRAINED]
    model_types = [ModelType.SMALL_CNN]
    participant_counts = [20]

    all_results = []

    for algorithm in algorithms:
        results = await validator.run_comprehensive_validation(
            scenarios, model_types, participant_counts, [algorithm]
        )

        for result in results:
            benchmark_result = BenchmarkResult(
                algorithm_name=type(algorithm).__name__,
                scenario=result.scenario,
                metrics=result,
            )
            all_results.append(benchmark_result)

    # Calculate relative performance rankings
    for scenario in scenarios:
        scenario_results = [r for r in all_results if r.scenario == scenario]
        scenario_results.sort(
            key=lambda x: x.metrics.success_rate
            * (1 - x.metrics.avg_compression_error),
            reverse=True,
        )

        for i, result in enumerate(scenario_results):
            result.ranking = i + 1
            if i > 0:
                baseline_score = scenario_results[0].metrics.success_rate * (
                    1 - scenario_results[0].metrics.avg_compression_error
                )
                current_score = result.metrics.success_rate * (
                    1 - result.metrics.avg_compression_error
                )
                result.relative_performance = (
                    current_score / baseline_score if baseline_score > 0 else 0
                )
            else:
                result.relative_performance = 1.0  # Best performer

    return all_results


def estimate_validation_time(
    scenarios: List[TestScenario],
    model_types: List[ModelType],
    participant_counts: List[int],
    algorithms: List[CompressionAlgorithm],
) -> float:
    """Estimate total validation time in seconds."""

    # Base time estimates per test (in seconds)
    base_times = {
        TestScenario.HOMOGENEOUS_NETWORK: 10,
        TestScenario.HETEROGENEOUS_NETWORK: 15,
        TestScenario.DYNAMIC_NETWORK: 20,
        TestScenario.LARGE_SCALE: 30,
        TestScenario.RESOURCE_CONSTRAINED: 25,
        TestScenario.HIGH_ACCURACY: 12,
    }

    model_multipliers = {
        ModelType.SMALL_CNN: 1.0,
        ModelType.LARGE_CNN: 3.0,
        ModelType.TRANSFORMER: 4.0,
        ModelType.MLP: 0.8,
        ModelType.CUSTOM: 1.5,
    }

    total_time = 0
    for scenario in scenarios:
        for model_type in model_types:
            for participant_count in participant_counts:
                for algorithm in algorithms:
                    base_time = base_times.get(scenario, 15)
                    model_multiplier = model_multipliers.get(model_type, 1.0)
                    scale_multiplier = 1 + (
                        participant_count / 50
                    )  # Linear scale impact

                    test_time = base_time * model_multiplier * scale_multiplier
                    total_time += test_time

    return total_time
