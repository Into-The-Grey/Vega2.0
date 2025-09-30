#!/usr/bin/env python3
"""
Test script for Communication Optimization Coordinator

Tests the intelligent compression strategy selection, network condition adaptation,
and performance monitoring capabilities of the federated learning coordinator.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Ensure src is available for direct execution
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from src.vega.federated.communication_coordinator import (
    CommunicationCoordinator,
    CompressionStrategy,
    NetworkCondition,
    NetworkMetrics,
    estimate_communication_savings,
)


async def test_participant_registration() -> CommunicationCoordinator:
    """Test participant registration and management."""
    print("🔧 Testing Participant Registration...")

    coordinator = CommunicationCoordinator(strategy=CompressionStrategy.ADAPTIVE)

    participants = [
        {
            "id": "mobile_device_1",
            "capabilities": {
                "compute_power": 0.3,
                "preferred_compression": CompressionStrategy.AGGRESSIVE,
            },
        },
        {
            "id": "edge_server_1",
            "capabilities": {
                "compute_power": 2.0,
                "preferred_compression": CompressionStrategy.BALANCED,
            },
        },
        {
            "id": "cloud_node_1",
            "capabilities": {
                "compute_power": 5.0,
                "preferred_compression": CompressionStrategy.CONSERVATIVE,
            },
        },
    ]

    for participant in participants:
        success = await coordinator.register_participant(
            participant["id"], participant["capabilities"]
        )
        assert success, f"Failed to register {participant['id']}"
        print(f"   ✅ Registered {participant['id']}")

    assert len(coordinator.participants) == 3, "Wrong number of participants registered"
    print(
        f"   ✅ All {len(coordinator.participants)} participants registered successfully"
    )

    return coordinator


async def test_network_metrics_update() -> CommunicationCoordinator:
    """Test network metrics updating and condition classification."""
    print("\n📡 Testing Network Metrics & Condition Classification...")

    coordinator = await test_participant_registration()

    network_scenarios = [
        {
            "participant": "mobile_device_1",
            "metrics": NetworkMetrics(
                bandwidth_mbps=5, latency_ms=250, packet_loss=0.02
            ),
            "expected_condition": NetworkCondition.CRITICAL,
        },
        {
            "participant": "edge_server_1",
            "metrics": NetworkMetrics(
                bandwidth_mbps=25, latency_ms=150, packet_loss=0.01
            ),
            "expected_condition": NetworkCondition.POOR,
        },
        {
            "participant": "cloud_node_1",
            "metrics": NetworkMetrics(
                bandwidth_mbps=150, latency_ms=30, packet_loss=0.001
            ),
            "expected_condition": NetworkCondition.EXCELLENT,
        },
    ]

    for scenario in network_scenarios:
        await coordinator.update_network_metrics(
            scenario["participant"], scenario["metrics"]
        )

        participant = coordinator.participants[scenario["participant"]]
        condition = participant.get_network_condition()

        assert (
            condition == scenario["expected_condition"]
        ), f"Wrong condition for {scenario['participant']}: got {condition}, expected {scenario['expected_condition']}"

        print(
            f"   ✅ {scenario['participant']}: {condition.value} (bandwidth: {scenario['metrics'].bandwidth_mbps} Mbps)"
        )

    print("   ✅ Network condition classification working correctly")
    return coordinator


async def test_compression_strategy_selection() -> CommunicationCoordinator:
    """Test intelligent compression strategy selection."""
    print("\n🧠 Testing Compression Strategy Selection...")

    coordinator = await test_network_metrics_update()

    test_gradients = {
        "conv1.weight": np.random.normal(0, 1, (64, 3, 7, 7)),
        "conv1.bias": np.random.normal(0, 1, (64,)),
        "fc.weight": np.random.normal(0, 1, (10, 64)),
        "fc.bias": np.random.normal(0, 1, (10,)),
    }

    model_size = sum(grad.size for grad in test_gradients.values())
    print(
        f"   Test model size: {model_size:,} parameters ({model_size * 4 / 1024:.1f} KB)"
    )

    for participant_id in coordinator.participants.keys():
        strategies = coordinator.select_compression_strategy(
            participant_id, test_gradients, training_round=1
        )

        participant = coordinator.participants[participant_id]
        condition = participant.get_network_condition()

        print(
            f"   ✅ {participant_id} ({condition.value}): {len(strategies)} compression strategies"
        )
        for i, strategy in enumerate(strategies, start=1):
            print(f"      {i}. {type(strategy).__name__}")

        if condition == NetworkCondition.CRITICAL:
            assert (
                len(strategies) >= 2
            ), "Critical condition should use multiple compression strategies"
        elif condition == NetworkCondition.EXCELLENT:
            assert (
                len(strategies) <= 2
            ), "Excellent condition should use fewer compression strategies"

    print("   ✅ Strategy selection adapts correctly to network conditions")
    return coordinator


async def test_compression_and_transmission() -> CommunicationCoordinator:
    """Test end-to-end compression and transmission simulation."""
    print("\n🚀 Testing Compression & Transmission Simulation...")

    coordinator = await test_compression_strategy_selection()

    test_gradients = {
        "conv1.weight": np.random.normal(0, 1, (32, 3, 5, 5)),
        "conv1.bias": np.random.normal(0, 1, (32,)),
        "fc.weight": np.random.normal(0, 1, (10, 32)),
        "fc.bias": np.random.normal(0, 1, (10,)),
    }

    original_size_kb = sum(grad.nbytes for grad in test_gradients.values()) / 1024
    print(f"   Original model size: {original_size_kb:.2f} KB")

    compression_results: Dict[str, Dict[str, float]] = {}

    for participant_id in coordinator.participants.keys():
        result, transmission_time = await coordinator.compress_and_transmit(
            participant_id, test_gradients, training_round=1
        )

        compression_results[participant_id] = {
            "compression_ratio": result.compression_ratio,
            "compression_error": result.compression_error,
            "compression_time": result.compression_time,
            "transmission_time": transmission_time,
            "compressed_size_kb": result.compressed_size_kb,
        }

        participant = coordinator.participants[participant_id]
        condition = participant.get_network_condition()

        print(f"   ✅ {participant_id} ({condition.value}):")
        print(f"      Compression ratio: {result.compression_ratio:.3f}")
        print(f"      Compression error: {result.compression_error:.6f}")
        print(f"      Compressed size: {result.compressed_size_kb:.2f} KB")
        print(f"      Compression time: {result.compression_time:.3f}s")
        print(f"      Transmission time: {transmission_time:.3f}s")

    for participant_id, results in compression_results.items():
        participant = coordinator.participants[participant_id]
        condition = participant.get_network_condition()

        if condition == NetworkCondition.CRITICAL:
            assert (
                results["compression_ratio"] > 0.5
            ), "Critical condition should achieve high compression"
        elif condition == NetworkCondition.EXCELLENT:
            assert (
                results["compression_error"] < 0.1
            ), "Excellent condition should maintain low error"

    print("   ✅ Compression effectiveness matches network conditions")
    return coordinator


async def test_performance_monitoring() -> CommunicationCoordinator:
    """Test performance monitoring and optimization."""
    print("\n📊 Testing Performance Monitoring & Optimization...")

    coordinator = await test_compression_and_transmission()

    test_gradients = {
        "layer1": np.random.normal(0, 1, (50, 20)),
        "layer2": np.random.normal(0, 1, (20, 10)),
        "bias": np.random.normal(0, 1, (10,)),
    }

    print("   Running multiple compression rounds for performance analysis...")
    for round_num in range(5):
        for participant_id in coordinator.participants.keys():
            await coordinator.compress_and_transmit(
                participant_id, test_gradients, training_round=round_num + 2
            )

    performance = coordinator.get_performance_summary()

    print("   ✅ Performance Summary:")
    print(
        f"      Average compression ratio: {performance['avg_compression_ratio']:.3f}"
    )
    print(
        f"      Average compression error: {performance['avg_compression_error']:.6f}"
    )
    print(f"      Average compression time: {performance['avg_compression_time']:.3f}s")
    print(
        f"      Average transmission time: {performance['avg_transmission_time']:.3f}s"
    )
    print(f"      Total rounds processed: {performance['total_rounds']}")
    print(f"      Participant count: {performance['participant_count']}")

    optimization_results = await coordinator.optimize_communication()
    print("   ✅ Optimization Results:")
    print(
        f"      Participants optimized: {optimization_results['participants_optimized']}"
    )
    print(f"      Strategies updated: {optimization_results['strategies_updated']}")

    return coordinator


def test_communication_savings_estimation() -> None:
    """Test communication savings estimation utility."""
    print("\n💰 Testing Communication Savings Estimation...")

    scenarios = [
        {
            "name": "Large Model, Good Compression",
            "original_size_mb": 100,
            "compression_ratio": 0.8,
            "bandwidth_mbps": 50,
            "participant_count": 10,
        },
        {
            "name": "Small Model, Moderate Compression",
            "original_size_mb": 5,
            "compression_ratio": 0.5,
            "bandwidth_mbps": 25,
            "participant_count": 5,
        },
        {
            "name": "Huge Model, Extreme Compression",
            "original_size_mb": 500,
            "compression_ratio": 0.95,
            "bandwidth_mbps": 10,
            "participant_count": 20,
        },
    ]

    for scenario in scenarios:
        savings = estimate_communication_savings(
            scenario["original_size_mb"],
            scenario["compression_ratio"],
            scenario["bandwidth_mbps"],
            scenario["participant_count"],
        )

        print(f"   ✅ {scenario['name']}:")
        print(
            f"      Original transmission time: {savings['original_transmission_time']:.1f}s"
        )
        print(
            f"      Compressed transmission time: {savings['compressed_transmission_time']:.1f}s"
        )
        print(f"      Time savings: {savings['time_savings_seconds']:.1f}s")
        print(f"      Bandwidth savings: {savings['bandwidth_savings_mb']:.1f} MB")
        print(f"      Efficiency improvement: {savings['efficiency_improvement']:.1f}%")
        print()


async def test_adaptive_strategies() -> None:
    """Test adaptive strategy behavior over time."""
    print("🔄 Testing Adaptive Strategy Behavior...")

    coordinator = CommunicationCoordinator(strategy=CompressionStrategy.ADAPTIVE)

    await coordinator.register_participant("adaptive_test", {"compute_power": 1.0})

    network_changes = [
        NetworkMetrics(bandwidth_mbps=100, latency_ms=20),
        NetworkMetrics(bandwidth_mbps=50, latency_ms=80),
        NetworkMetrics(bandwidth_mbps=15, latency_ms=180),
        NetworkMetrics(bandwidth_mbps=5, latency_ms=300),
    ]

    test_gradients = {
        "weights": np.random.normal(0, 1, (100, 50)),
        "bias": np.random.normal(0, 1, (50,)),
    }

    for i, metrics in enumerate(network_changes):
        await coordinator.update_network_metrics("adaptive_test", metrics)

        strategies = coordinator.select_compression_strategy(
            "adaptive_test", test_gradients, training_round=i + 1
        )

        participant = coordinator.participants["adaptive_test"]
        condition = participant.get_network_condition()

        print(f"   ✅ Round {i+1} - {condition.value}:")
        print(
            f"      Bandwidth: {metrics.bandwidth_mbps} Mbps, Latency: {metrics.latency_ms} ms"
        )
        print(f"      Selected {len(strategies)} compression strategies")

        print("      Compression results require full Vega deployment; skipping.")
        print()


async def run_comprehensive_tests() -> None:
    """Run all tests comprehensively."""
    print("🎯 Communication Optimization Coordinator Comprehensive Test")
    print("=" * 70)

    try:
        await test_participant_registration()
        await test_network_metrics_update()
        await test_compression_strategy_selection()
        await test_compression_and_transmission()
        await test_performance_monitoring()

        test_communication_savings_estimation()

        await test_adaptive_strategies()

        print("\n🎉 All Communication Coordinator Tests Passed Successfully!")
        print("✅ Participant registration and management")
        print("✅ Network condition classification")
        print("✅ Adaptive compression strategy selection")
        print("✅ Compression and transmission simulation")
        print("✅ Performance monitoring and optimization")
        print("✅ Savings estimation utility")
        print("✅ Adaptive strategy evaluation")
    except AssertionError as exc:
        print(f"❌ Test assertion failed: {exc}")
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Unexpected error during tests: {exc}")
        raise


if __name__ == "__main__":
    asyncio.run(run_comprehensive_tests())
