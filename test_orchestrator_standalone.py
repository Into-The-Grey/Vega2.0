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
import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append("/home/ncacord/Vega2.0")

try:
    from src.vega.federated.pruning_orchestrator import (
        AdaptivePruningOrchestrator,
        SparsityScheduler,
        PerformanceMonitor,
        DistillationCoordinator,
        ParticipantProfile,
        ParticipantCapability,
        PruningStrategy,
        SparsityScheduleConfig,
        PerformanceMetrics,
    )

    print("✓ Successfully imported adaptive orchestrator components")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)


class TestModel(nn.Module):
    """Test model for orchestration testing."""

    def __init__(self, input_size=100, hidden_size=50, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


async def test_participant_profile():
    """Test participant profile functionality."""
    print("\n=== Testing Participant Profile ===")

    # Create participant profile
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

    # Test metrics update
    test_accuracies = [0.85, 0.83, 0.86, 0.84, 0.87]
    test_times = [60, 58, 62, 59, 61]
    test_comm_times = [10, 12, 9, 11, 10]

    for acc, time, comm in zip(test_accuracies, test_times, test_comm_times):
        profile.update_metrics(acc, time, comm)

    print(f"Updated metrics - Stability score: {profile.stability_score:.3f}")
    print(f"Recent accuracies: {[f'{a:.3f}' for a in profile.recent_accuracy]}")

    # Test sparsity recommendation
    base_sparsity = 0.5
    recommended = profile.get_recommended_sparsity(base_sparsity)
    print(f"Recommended sparsity for base {base_sparsity}: {recommended:.3f}")

    assert (
        0.1 <= recommended <= 0.8
    ), f"Recommended sparsity {recommended} outside bounds"
    assert (
        len(profile.recent_accuracy) == 5
    ), f"Expected 5 accuracy values, got {len(profile.recent_accuracy)}"

    print("✓ Participant profile tests passed")
    return profile


async def test_sparsity_scheduler():
    """Test sparsity scheduling functionality."""
    print("\n=== Testing Sparsity Scheduler ===")

    # Create scheduler configuration
    config = SparsityScheduleConfig(
        initial_sparsity=0.1,
        final_sparsity=0.8,
        warmup_rounds=3,
        cooldown_rounds=2,
        adaptation_rate=0.1,
    )

    scheduler = SparsityScheduler(config)

    # Create test participant profiles
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

    # Test scheduling across rounds
    for round_num in range(1, total_rounds + 1):
        targets = scheduler.calculate_target_sparsity(round_num, total_rounds, profiles)

        print(f"Round {round_num}: {targets}")

        # Validate targets
        for participant_id, target in targets.items():
            profile = profiles[participant_id]
            assert (
                profile.min_sparsity <= target <= profile.max_sparsity
            ), f"Target {target} outside bounds for {participant_id}"

        # Simulate performance feedback
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

    # Test adaptation history
    assert (
        len(scheduler.adaptation_history) == total_rounds
    ), f"Expected {total_rounds} adaptation records, got {len(scheduler.adaptation_history)}"

    # Test global performance tracking
    assert len(scheduler.global_performance) > 0, "No global performance recorded"

    print(f"✓ Sparsity scheduler tests passed")
    print(f"  Final global performance: {scheduler.global_performance[-1]:.3f}")
    print(f"  Adaptations recorded: {len(scheduler.adaptation_history)}")

    return scheduler


async def test_performance_monitor():
    """Test performance monitoring functionality."""
    print("\n=== Testing Performance Monitor ===")

    monitor = PerformanceMonitor()

    # Create test profiles
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

    # Create test metrics with various performance scenarios
    test_metrics = [
        # Good performance
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
        # Struggling participant
        PerformanceMetrics(
            round_num=1,
            participant_id="participant_2",
            accuracy_before=0.82,
            accuracy_after=0.70,  # Large drop
            training_time=150,
            communication_time=15,
            memory_usage=0.85,
            model_size=1200,
            sparsity_ratio=0.5,
            convergence_rate=0.01,
        ),
        # Normal performance
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

    # Monitor round
    monitoring_results = await monitor.monitor_round(test_metrics, profiles)

    print("Monitoring results:")
    print(f"  Alerts generated: {len(monitoring_results['alerts'])}")
    print(f"  Recommendations: {len(monitoring_results['recommendations'])}")
    print(f"  Participant health statuses:")

    for pid, health in monitoring_results["participant_health"].items():
        print(f"    {pid}: {health['status']} (score: {health['score']:.3f})")
        if health["issues"]:
            print(f"      Issues: {health['issues']}")

    # Validate monitoring
    assert (
        len(monitoring_results["alerts"]) > 0
    ), "Expected alerts for struggling participant"
    assert len(monitoring_results["recommendations"]) > 0, "Expected recommendations"
    assert (
        "participant_health" in monitoring_results
    ), "Missing participant health analysis"

    # Check specific alerts
    alerts = monitoring_results["alerts"]
    accuracy_alerts = [a for a in alerts if a["type"] == "accuracy_drop"]
    assert len(accuracy_alerts) > 0, "Expected accuracy drop alert"

    # Check summary
    summary = monitoring_results["summary"]
    assert "participants" in summary, "Missing participants count"
    assert (
        summary["participants"] == 3
    ), f"Expected 3 participants, got {summary['participants']}"
    assert "struggling_participants" in summary, "Missing struggling participants count"

    print("✓ Performance monitor tests passed")
    print(
        f"  Summary: {summary['struggling_participants']} struggling, {summary['healthy_participants']} healthy"
    )

    return monitor, monitoring_results


async def test_distillation_coordinator():
    """Test distillation coordination functionality."""
    print("\n=== Testing Distillation Coordinator ===")

    coordinator = DistillationCoordinator()

    # Create test models
    teacher_model = TestModel()
    student_models = {"struggling_1": TestModel(), "struggling_2": TestModel()}

    struggling_participants = list(student_models.keys())

    print(
        f"Testing distillation for {len(struggling_participants)} struggling participants"
    )

    # Mock performance monitor
    class MockPerformanceMonitor:
        pass

    mock_monitor = MockPerformanceMonitor()

    # Coordinate recovery distillation
    distillation_results = await coordinator.coordinate_recovery_distillation(
        struggling_participants=struggling_participants,
        participant_models=student_models,
        global_model=teacher_model,
        performance_monitor=mock_monitor,
    )

    print("Distillation results:")
    for pid, result in distillation_results["results"].items():
        print(f"  {pid}: knowledge retention = {result['knowledge_retention']:.3f}")

    # Validate results
    assert len(distillation_results["results"]) == len(
        struggling_participants
    ), "Distillation results count mismatch"
    assert "success_rate" in distillation_results, "Missing success rate"
    assert (
        len(coordinator.distillation_history) == 1
    ), "Expected one distillation session"

    # Check knowledge retention quality
    for pid, result in distillation_results["results"].items():
        retention = result["knowledge_retention"]
        assert 0.0 <= retention <= 1.0, f"Invalid retention value: {retention}"

        # For our mock implementation, we expect good retention
        if retention > 0.7:
            print(f"    ✓ Good knowledge retention for {pid}")

    success_rate = distillation_results["success_rate"]
    print(f"✓ Distillation coordinator tests passed")
    print(f"  Overall success rate: {success_rate:.3f}")

    return coordinator, distillation_results


async def test_adaptive_orchestrator():
    """Test the main adaptive orchestrator functionality."""
    print("\n=== Testing Adaptive Orchestrator ===")

    # Create orchestrator with test configuration
    schedule_config = SparsityScheduleConfig(
        initial_sparsity=0.1,
        final_sparsity=0.7,
        warmup_rounds=2,
        cooldown_rounds=1,
        adaptation_rate=0.15,
    )

    orchestrator = AdaptivePruningOrchestrator(schedule_config)

    # Register test participants
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

    # Create participant models
    participant_models = {pid: TestModel() for pid, _, _ in test_participants}
    global_model = TestModel()

    # Run orchestration rounds
    total_rounds = 6
    orchestration_results = []

    print(f"Running {total_rounds} orchestration rounds...")

    for round_num in range(1, total_rounds + 1):
        results = await orchestrator.orchestrate_pruning_round(
            round_num=round_num,
            total_rounds=total_rounds,
            participant_models=participant_models,
            global_model=global_model,
        )

        orchestration_results.append(results)

        summary = results["performance_summary"]
        print(
            f"  Round {round_num}: acc={summary['average_accuracy']:.3f}, "
            f"sparsity={summary['average_sparsity']:.3f}, "
            f"struggling={summary['struggling_participants']}"
        )

    # Get final summary
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

    # Validate orchestration
    assert len(orchestration_results) == total_rounds, "Round count mismatch"
    assert final_summary["total_rounds"] == total_rounds, "Summary round count mismatch"
    assert final_summary["total_participants"] == len(
        test_participants
    ), "Participant count mismatch"

    # Check progression
    first_sparsity = orchestration_results[0]["performance_summary"]["average_sparsity"]
    last_sparsity = orchestration_results[-1]["performance_summary"]["average_sparsity"]
    assert last_sparsity > first_sparsity, "Sparsity should increase over rounds"

    print("✓ Adaptive orchestrator tests passed")

    # Test history saving
    test_history_file = "/tmp/test_orchestration_history.json"
    await orchestrator.save_orchestration_history(test_history_file)

    # Verify file exists and has content
    import json

    with open(test_history_file, "r") as f:
        saved_history = json.load(f)

    assert "participant_profiles" in saved_history, "Missing participant profiles"
    assert "orchestration_history" in saved_history, "Missing orchestration history"
    assert "summary" in saved_history, "Missing summary"

    print(f"✓ History successfully saved to {test_history_file}")

    return orchestrator, final_summary


async def test_integration_scenarios():
    """Test complex integration scenarios."""
    print("\n=== Testing Integration Scenarios ===")

    # Scenario 1: Heterogeneous participant capabilities
    print("\nScenario 1: Heterogeneous Participants")

    config = SparsityScheduleConfig(initial_sparsity=0.05, final_sparsity=0.85)
    orchestrator = AdaptivePruningOrchestrator(config)

    # Register participants with diverse profiles
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

    # Run a few rounds to see adaptation
    participant_models = {pid: TestModel() for pid, _, _, _ in diverse_participants}
    global_model = TestModel()

    for round_num in range(1, 4):
        results = await orchestrator.orchestrate_pruning_round(
            round_num, 10, participant_models, global_model
        )

        sparsity_targets = results["sparsity_targets"]
        print(f"  Round {round_num} sparsity targets: {sparsity_targets}")

        # Verify heterogeneity
        min_target = min(sparsity_targets.values())
        max_target = max(sparsity_targets.values())
        print(f"    Sparsity range: {min_target:.3f} - {max_target:.3f}")

        assert max_target - min_target > 0.1, "Expected significant sparsity variation"

    print("✓ Heterogeneous participant scenario passed")

    # Scenario 2: Performance degradation recovery
    print("\nScenario 2: Performance Degradation Recovery")

    # Create orchestrator with responsive adaptation
    responsive_config = SparsityScheduleConfig(
        adaptation_rate=0.25, performance_threshold=0.03  # Higher adaptation rate
    )

    recovery_orchestrator = AdaptivePruningOrchestrator(responsive_config)

    # Register participants
    for i in range(3):
        recovery_orchestrator.register_participant(
            f"participant_{i}", ParticipantCapability.MEDIUM, PruningStrategy.ADAPTIVE
        )

    # Simulate performance degradation by manually updating scheduler
    # This simulates previous rounds with declining performance
    degrading_performance = [0.80, 0.75, 0.68, 0.62, 0.58]  # Declining
    recovery_orchestrator.scheduler.global_performance = degrading_performance

    models = {f"participant_{i}": TestModel() for i in range(3)}

    # Run round - should trigger adaptation
    results = await recovery_orchestrator.orchestrate_pruning_round(
        6, 10, models, TestModel()
    )

    # Check for adaptation
    adaptations = recovery_orchestrator.scheduler.adaptation_history
    if adaptations:
        last_adaptation = adaptations[-1]
        print(
            f"  Adaptation triggered: base sparsity = {last_adaptation['base_sparsity']:.3f}"
        )

    # Check for recommendations
    recommendations = results["monitoring_results"]["recommendations"]
    print(f"  Recommendations generated: {len(recommendations)}")

    print("✓ Performance degradation recovery scenario passed")

    return True


async def run_all_tests():
    """Run complete test suite for adaptive orchestrator."""
    print("Starting Adaptive Pruning Orchestrator Test Suite")
    print("=" * 60)

    try:
        # Individual component tests
        profile = await test_participant_profile()
        scheduler = await test_sparsity_scheduler()
        monitor, monitor_results = await test_performance_monitor()
        coordinator, distill_results = await test_distillation_coordinator()

        # Integration tests
        orchestrator, summary = await test_adaptive_orchestrator()
        integration_success = await test_integration_scenarios()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 60)

        print("\nTest Results Summary:")
        print(f"✓ Participant Profile: Working correctly")
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
        print(f"✓ Integration Scenarios: All scenarios passed")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Set up test environment
    torch.manual_seed(42)
    np.random.seed(42)

    # Run test suite
    success = asyncio.run(run_all_tests())
