"""
Comprehensive Test Suite for Adaptive Pruning Orchestrator
=========================================================

This test suite validates all components of the adaptive pruning orchestration
system, including participant profiling, sparsity scheduling, performance
monitoring, and distillation coordination.

Author: Vega2.0 Federated Learning Team
Date: September 2025
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

try:
    from src.vega.federated.pruning_orchestrator import (
        AdaptivePruningOrchestrator,
        DistillationCoordinator,
        ParticipantCapability,
        ParticipantProfile,
        PerformanceMetrics,
        PerformanceMonitor,
        PruningStrategy,
        SparsityScheduleConfig,
        SparsityScheduler,
    )

    print("✓ Successfully imported adaptive orchestrator components")
except ImportError as e:  # pragma: no cover - fallback execution
    print(f"✗ Import error: {e}")
    raise


class TestModel(nn.Module):
    """Test model for orchestration testing."""

    def __init__(
        self, input_size: int = 100, hidden_size: int = 50, output_size: int = 10
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


async def test_participant_profile() -> ParticipantProfile:
    """Test participant profile functionality."""
    print("\n=== Testing Participant Profile ===")

    profile = ParticipantProfile(
        participant_id="test_participant",
        capability=ParticipantCapability.MEDIUM,
        preferred_strategy=PruningStrategy.BALANCED,
        max_sparsity=0.8,
        min_sparsity=0.1,
        computational_budget=1.0,
        accuracy_tolerance=0.05,
    )

    print(f"Created profile for {profile.participant_id}")
    print(f"Capability: {profile.capability.value}")
    print(f"Strategy: {profile.preferred_strategy.value}")
    print(f"Sparsity range: {profile.min_sparsity} - {profile.max_sparsity}")

    test_accuracies = [0.85, 0.83, 0.86, 0.84, 0.87]
    test_times = [60, 58, 62, 59, 61]
    test_comm_times = [10, 12, 9, 11, 10]

    for acc, train_time, comm in zip(test_accuracies, test_times, test_comm_times):
        profile.update_metrics(acc, train_time, comm)

    print(f"Updated metrics - Stability score: {profile.stability_score:.3f}")
    print(f"Recent accuracies: {[f'{a:.3f}' for a in profile.recent_accuracy]}")

    base_sparsity = 0.5
    recommended = profile.get_recommended_sparsity(base_sparsity)
    print(f"Recommended sparsity for base {base_sparsity}: {recommended:.3f}")

    assert 0.1 <= recommended <= 0.8, "Recommended sparsity outside bounds"
    assert len(profile.recent_accuracy) == 5, "Expected 5 accuracy values"

    print("✓ Participant profile tests passed")
    return profile


async def test_sparsity_scheduler() -> SparsityScheduler:
    """Test sparsity scheduling functionality."""
    print("\n=== Testing Sparsity Scheduler ===")

    config = SparsityScheduleConfig(
        initial_sparsity=0.1,
        final_sparsity=0.8,
        warmup_rounds=3,
        cooldown_rounds=2,
        adaptation_rate=0.1,
    )

    scheduler = SparsityScheduler(config)

    profiles = {
        "high_cap": ParticipantProfile(
            "high_cap",
            ParticipantCapability.HIGH,
            PruningStrategy.AGGRESSIVE,
            max_sparsity=0.9,
        ),
        "med_cap": ParticipantProfile(
            "med_cap",
            ParticipantCapability.MEDIUM,
            PruningStrategy.BALANCED,
            max_sparsity=0.7,
        ),
        "low_cap": ParticipantProfile(
            "low_cap",
            ParticipantCapability.LOW,
            PruningStrategy.CONSERVATIVE,
            max_sparsity=0.5,
        ),
    }

    total_rounds = 10
    print(f"Testing sparsity scheduling over {total_rounds} rounds")

    for round_num in range(1, total_rounds + 1):
        targets = scheduler.calculate_target_sparsity(round_num, total_rounds, profiles)

        print(f"Round {round_num}: {targets}")

        for participant_id, target in targets.items():
            profile = profiles[participant_id]
            assert (
                profile.min_sparsity <= target <= profile.max_sparsity
            ), f"Target outside bounds for {participant_id}"

        mock_metrics = [
            PerformanceMetrics(
                round_num=round_num,
                participant_id=pid,
                accuracy_before=0.85,
                accuracy_after=0.85 - target * 0.1,
                training_time=60,
                communication_time=10,
                memory_usage=0.5,
                model_size=1000,
                sparsity_ratio=target,
                convergence_rate=0.02,
            )
            for pid, target in targets.items()
        ]

        scheduler.update_performance(mock_metrics)

    assert (
        len(scheduler.adaptation_history) == total_rounds
    ), "Unexpected adaptation history length"
    assert scheduler.global_performance, "No global performance recorded"

    print("✓ Sparsity scheduler tests passed")
    print(f"  Final global performance: {scheduler.global_performance[-1]:.3f}")
    print(f"  Adaptations recorded: {len(scheduler.adaptation_history)}")

    return scheduler


async def test_performance_monitor() -> (
    Tuple[PerformanceMonitor, Dict[str, Dict[str, float]]]
):
    """Test performance monitoring functionality."""
    print("\n=== Testing Performance Monitor ===")

    monitor = PerformanceMonitor()

    profiles = {
        "participant_1": ParticipantProfile(
            "participant_1", ParticipantCapability.HIGH, PruningStrategy.AGGRESSIVE
        ),
        "participant_2": ParticipantProfile(
            "participant_2", ParticipantCapability.MEDIUM, PruningStrategy.BALANCED
        ),
        "participant_3": ParticipantProfile(
            "participant_3", ParticipantCapability.LOW, PruningStrategy.CONSERVATIVE
        ),
    }

    test_metrics = [
        PerformanceMetrics(
            round_num=1,
            participant_id="participant_1",
            accuracy_before=0.85,
            accuracy_after=0.83,
            training_time=45,
            communication_time=8,
            memory_usage=0.4,
            model_size=1000,
            sparsity_ratio=0.3,
            convergence_rate=0.02,
        ),
        PerformanceMetrics(
            round_num=1,
            participant_id="participant_2",
            accuracy_before=0.82,
            accuracy_after=0.70,
            training_time=150,
            communication_time=15,
            memory_usage=0.85,
            model_size=1200,
            sparsity_ratio=0.5,
            convergence_rate=0.01,
        ),
        PerformanceMetrics(
            round_num=1,
            participant_id="participant_3",
            accuracy_before=0.80,
            accuracy_after=0.78,
            training_time=70,
            communication_time=12,
            memory_usage=0.6,
            model_size=800,
            sparsity_ratio=0.2,
            convergence_rate=0.015,
        ),
    ]

    monitoring_results = await monitor.monitor_round(test_metrics, profiles)

    print("Monitoring results:")
    print(f"  Alerts generated: {len(monitoring_results['alerts'])}")
    print(f"  Recommendations: {len(monitoring_results['recommendations'])}")
    print(f"  Participant health statuses:")

    for pid, health in monitoring_results["participant_health"].items():
        print(f"    {pid}: {health['status']} (score: {health['score']:.3f})")
        if health["issues"]:
            print(f"      Issues: {health['issues']}")

    assert monitoring_results["alerts"], "Expected alerts for struggling participant"
    assert monitoring_results["recommendations"], "Expected recommendations"

    summary = monitoring_results["summary"]
    assert summary["participants"] == 3, "Unexpected participant count"
    assert "struggling_participants" in summary

    print("✓ Performance monitor tests passed")
    print(
        f"  Summary: {summary['struggling_participants']} struggling, {summary['healthy_participants']} healthy"
    )

    return monitor, monitoring_results


async def test_distillation_coordinator() -> (
    Tuple[DistillationCoordinator, Dict[str, Dict[str, float]]]
):
    """Test distillation coordination functionality."""
    print("\n=== Testing Distillation Coordinator ===")

    coordinator = DistillationCoordinator()

    teacher_model = TestModel()
    student_models: Dict[str, nn.Module] = {
        "struggling_1": TestModel(),
        "struggling_2": TestModel(),
    }
    struggling_participants = list(student_models.keys())

    print(
        f"Testing distillation for {len(struggling_participants)} struggling participants"
    )

    mock_monitor = PerformanceMonitor()

    distillation_results = await coordinator.coordinate_recovery_distillation(
        struggling_participants=struggling_participants,
        participant_models=student_models,
        global_model=teacher_model,
        performance_monitor=mock_monitor,
    )

    print("Distillation results:")
    for pid, result in distillation_results["results"].items():
        print(f"  {pid}: knowledge retention = {result['knowledge_retention']:.3f}")

    assert len(distillation_results["results"]) == len(struggling_participants)
    assert "success_rate" in distillation_results
    assert len(coordinator.distillation_history) == 1

    for pid, result in distillation_results["results"].items():
        retention = result["knowledge_retention"]
        assert 0.0 <= retention <= 1.0, "Invalid retention value"
        if retention > 0.7:
            print(f"    ✓ Good knowledge retention for {pid}")

    success_rate = distillation_results["success_rate"]
    print("✓ Distillation coordinator tests passed")
    print(f"  Overall success rate: {success_rate:.3f}")

    return coordinator, distillation_results


async def test_adaptive_orchestrator() -> (
    Tuple[AdaptivePruningOrchestrator, Dict[str, Any]]
):
    """Test the main adaptive orchestrator functionality."""
    print("\n=== Testing Adaptive Orchestrator ===")

    schedule_config = SparsityScheduleConfig(
        initial_sparsity=0.1,
        final_sparsity=0.7,
        warmup_rounds=2,
        cooldown_rounds=1,
        adaptation_rate=0.15,
    )

    orchestrator = AdaptivePruningOrchestrator(schedule_config)

    test_participants = [
        ("high_performer", ParticipantCapability.HIGH, PruningStrategy.AGGRESSIVE),
        ("average_performer", ParticipantCapability.MEDIUM, PruningStrategy.BALANCED),
        ("low_performer", ParticipantCapability.LOW, PruningStrategy.CONSERVATIVE),
        (
            "variable_performer",
            ParticipantCapability.VARIABLE,
            PruningStrategy.ADAPTIVE,
        ),
    ]

    for pid, capability, strategy in test_participants:
        orchestrator.register_participant(pid, capability, strategy)

    print(f"Registered {len(test_participants)} test participants")

    participant_models: Dict[str, nn.Module] = {
        pid: TestModel() for pid, _, _ in test_participants
    }
    global_model = TestModel()

    total_rounds = 6
    orchestration_results: List[Dict[str, Any]] = []

    print(f"Running {total_rounds} orchestration rounds...")

    for round_num in range(1, total_rounds + 1):
        results = await orchestrator.orchestrate_pruning_round(
            round_num=round_num,
            total_rounds=total_rounds,
            participant_models=participant_models,
            global_model=global_model,
        )

        print(f"  Round {round_num} results:")
        for metrics in results["participant_metrics"]:
            print(
                f"    {metrics['participant_id']}: sparsity={metrics['sparsity_ratio']:.3f}, accuracy_after={metrics['accuracy_after']:.3f}"
            )

        orchestration_results.append(results)

    final_summary = orchestrator.get_orchestration_summary()

    print("Final orchestration summary:")
    print(f"  Total rounds: {final_summary['total_rounds']}")
    print(f"  Final accuracy: {final_summary['final_average_accuracy']:.3f}")
    print(f"  Final sparsity: {final_summary['final_average_sparsity']:.3f}")
    print(f"  Accuracy improvement: {final_summary['accuracy_improvement']:.3f}")
    print(
        f"  Distillation interventions: {final_summary['distillation_interventions']}"
    )
    print(f"  Total adaptations: {final_summary['adaptations_made']}")

    assert len(orchestration_results) == total_rounds, "Round count mismatch"
    assert final_summary["total_rounds"] == total_rounds, "Summary round count mismatch"
    assert final_summary["total_participants"] == len(
        test_participants
    ), "Participant count mismatch"

    first_sparsity = orchestration_results[0]["performance_summary"]["average_sparsity"]
    last_sparsity = orchestration_results[-1]["performance_summary"]["average_sparsity"]
    assert last_sparsity > first_sparsity, "Sparsity should increase over rounds"

    print("✓ Adaptive orchestrator tests passed")

    history_path = Path("/tmp/test_orchestration_history.json")
    await orchestrator.save_orchestration_history(str(history_path))

    import json

    with history_path.open("r") as f:
        saved_history = json.load(f)

    assert "participant_profiles" in saved_history, "Missing participant profiles"
    assert "orchestration_history" in saved_history, "Missing orchestration history"
    assert "summary" in saved_history, "Missing summary"

    print(f"✓ History successfully saved to {history_path}")

    return orchestrator, final_summary


async def test_integration_scenarios() -> bool:
    """Test complex integration scenarios."""
    print("\n=== Testing Integration Scenarios ===")

    print("\nScenario 1: Heterogeneous Participants")

    config = SparsityScheduleConfig(initial_sparsity=0.05, final_sparsity=0.85)
    orchestrator = AdaptivePruningOrchestrator(config)

    diverse_participants = [
        (
            "mobile_device",
            ParticipantCapability.LOW,
            PruningStrategy.CONSERVATIVE,
            {"max_sparsity": 0.4, "bandwidth": 20.0},
        ),
        (
            "edge_server",
            ParticipantCapability.MEDIUM,
            PruningStrategy.BALANCED,
            {"max_sparsity": 0.7, "bandwidth": 100.0},
        ),
        (
            "cloud_instance",
            ParticipantCapability.HIGH,
            PruningStrategy.AGGRESSIVE,
            {"max_sparsity": 0.95, "bandwidth": 1000.0},
        ),
        (
            "variable_device",
            ParticipantCapability.VARIABLE,
            PruningStrategy.ADAPTIVE,
            {"max_sparsity": 0.8, "bandwidth": 50.0},
        ),
    ]

    for pid, cap, strategy, kwargs in diverse_participants:
        orchestrator.register_participant(pid, cap, strategy, **kwargs)

    participant_models: Dict[str, nn.Module] = {
        pid: TestModel() for pid, _, _, _ in diverse_participants
    }
    global_model = TestModel()

    for round_num in range(1, 4):
        results = await orchestrator.orchestrate_pruning_round(
            round_num, 10, participant_models, global_model
        )

        sparsity_targets = results["sparsity_targets"]
        print(f"  Round {round_num} sparsity targets: {sparsity_targets}")

        min_target = min(sparsity_targets.values())
        max_target = max(sparsity_targets.values())
        print(f"    Sparsity range: {min_target:.3f} - {max_target:.3f}")

        assert max_target - min_target > 0.001, "Expected some sparsity variation"

    print("✓ Heterogeneous participant scenario passed")

    print("\nScenario 2: Performance Degradation Recovery")

    responsive_config = SparsityScheduleConfig(
        adaptation_rate=0.25, performance_threshold=0.03
    )

    recovery_orchestrator = AdaptivePruningOrchestrator(responsive_config)

    for i in range(3):
        recovery_orchestrator.register_participant(
            f"participant_{i}", ParticipantCapability.MEDIUM, PruningStrategy.ADAPTIVE
        )

    recovery_orchestrator.scheduler.global_performance = [
        0.80,
        0.75,
        0.68,
        0.62,
        0.58,
    ]

    models: Dict[str, nn.Module] = {f"participant_{i}": TestModel() for i in range(3)}

    results = await recovery_orchestrator.orchestrate_pruning_round(
        6, 10, models, TestModel()
    )

    adaptations = recovery_orchestrator.scheduler.adaptation_history
    if adaptations:
        last_adaptation = adaptations[-1]
        print(
            f"  Adaptation triggered: base sparsity = {last_adaptation['base_sparsity']:.3f}"
        )

    recommendations = results["monitoring_results"]["recommendations"]
    print(f"  Recommendations generated: {len(recommendations)}")

    print("✓ Performance degradation recovery scenario passed")

    return True


async def run_all_tests() -> bool:
    """Run complete test suite for adaptive orchestrator."""
    print("Starting Adaptive Pruning Orchestrator Test Suite")
    print("=" * 60)

    try:
        profile = await test_participant_profile()
        scheduler = await test_sparsity_scheduler()
        monitor, monitor_results = await test_performance_monitor()
        coordinator, distill_results = await test_distillation_coordinator()

        orchestrator, summary = await test_adaptive_orchestrator()
        integration_success = await test_integration_scenarios()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 60)

        print("\nTest Results Summary:")
        print("✓ Participant Profile: Working correctly")
        print(
            f"✓ Sparsity Scheduler: {len(scheduler.adaptation_history)} adaptations recorded"
        )
        print(
            f"✓ Performance Monitor: {len(monitor_results['alerts'])} alerts, {len(monitor_results['recommendations'])} recommendations"
        )
        print(
            f"✓ Distillation Coordinator: {distill_results['success_rate']:.3f} success rate"
        )
        print(
            f"✓ Adaptive Orchestrator: {summary['total_rounds']} rounds, {summary['final_average_accuracy']:.3f} final accuracy"
        )
        print(f"✓ Integration Scenarios: {integration_success}")

        return True

    except Exception as exc:  # noqa: BLE001
        print(f"\n❌ Test failed with error: {exc}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
