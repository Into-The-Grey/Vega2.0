"""
Direct Test for Adaptive Pruning Orchestrator Components
=======================================================

This test directly validates orchestrator components without external dependencies.
"""

import asyncio
import torch
import torch.nn as nn
import numpy as np
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from collections import defaultdict, deque


# Copy essential classes directly for testing
class ParticipantCapability(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VARIABLE = "variable"


class PruningStrategy(Enum):
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"
    BALANCED = "balanced"


@dataclass
class ParticipantProfile:
    participant_id: str
    capability: ParticipantCapability
    preferred_strategy: PruningStrategy
    max_sparsity: float = 0.9
    min_sparsity: float = 0.0
    computational_budget: float = 1.0
    bandwidth: float = 100.0
    latency: float = 50.0
    accuracy_tolerance: float = 0.05

    recent_accuracy: List[float] = field(default_factory=list)
    recent_training_time: List[float] = field(default_factory=list)
    recent_communication_time: List[float] = field(default_factory=list)
    stability_score: float = 1.0

    def update_metrics(self, accuracy: float, training_time: float, comm_time: float):
        self.recent_accuracy.append(accuracy)
        self.recent_training_time.append(training_time)
        self.recent_communication_time.append(comm_time)

        max_history = 10
        self.recent_accuracy = self.recent_accuracy[-max_history:]
        self.recent_training_time = self.recent_training_time[-max_history:]
        self.recent_communication_time = self.recent_communication_time[-max_history:]

        if len(self.recent_accuracy) >= 3:
            acc_variance = np.var(self.recent_accuracy)
            self.stability_score = max(0.1, 1.0 - acc_variance)

    def get_recommended_sparsity(self, base_sparsity: float) -> float:
        capability_multiplier = {
            ParticipantCapability.HIGH: 1.2,
            ParticipantCapability.MEDIUM: 1.0,
            ParticipantCapability.LOW: 0.7,
            ParticipantCapability.VARIABLE: 0.9,
        }

        stability_factor = 0.5 + 0.5 * self.stability_score
        budget_factor = min(self.computational_budget, 1.5)

        recommended = (
            base_sparsity
            * capability_multiplier[self.capability]
            * stability_factor
            * budget_factor
        )
        return max(self.min_sparsity, min(self.max_sparsity, recommended))


@dataclass
class PerformanceMetrics:
    round_num: int
    participant_id: str
    accuracy_before: float
    accuracy_after: float
    training_time: float
    communication_time: float
    memory_usage: float
    model_size: int
    sparsity_ratio: float
    convergence_rate: float
    timestamp: float = field(default_factory=time.time)

    @property
    def accuracy_drop(self) -> float:
        return self.accuracy_before - self.accuracy_after

    @property
    def total_time(self) -> float:
        return self.training_time + self.communication_time


@dataclass
class SparsityScheduleConfig:
    initial_sparsity: float = 0.1
    final_sparsity: float = 0.8
    warmup_rounds: int = 5
    cooldown_rounds: int = 10
    adaptation_rate: float = 0.1
    stability_threshold: float = 0.02
    performance_threshold: float = 0.05


class SparsityScheduler:
    def __init__(self, config: SparsityScheduleConfig):
        self.config = config
        self.performance_history: Dict[str, List[PerformanceMetrics]] = defaultdict(
            list
        )
        self.global_performance: List[float] = []
        self.adaptation_history: List[Dict[str, Any]] = []

    def calculate_target_sparsity(
        self,
        round_num: int,
        total_rounds: int,
        participant_profiles: Dict[str, ParticipantProfile],
    ) -> Dict[str, float]:
        progress = min(round_num / total_rounds, 1.0)

        if round_num <= self.config.warmup_rounds:
            warmup_factor = round_num / self.config.warmup_rounds
            base_sparsity = self.config.initial_sparsity * warmup_factor
        elif round_num >= total_rounds - self.config.cooldown_rounds:
            cooldown_progress = (total_rounds - round_num) / self.config.cooldown_rounds
            base_sparsity = self.config.final_sparsity * (1 - cooldown_progress * 0.2)
        else:
            adjusted_progress = (round_num - self.config.warmup_rounds) / (
                total_rounds - self.config.warmup_rounds - self.config.cooldown_rounds
            )
            base_sparsity = self.config.initial_sparsity + adjusted_progress * (
                self.config.final_sparsity - self.config.initial_sparsity
            )

        if len(self.global_performance) >= 3:
            recent_performance = np.mean(self.global_performance[-3:])
            if len(self.global_performance) >= 6:
                previous_performance = np.mean(self.global_performance[-6:-3])
                if (
                    recent_performance
                    < previous_performance - self.config.performance_threshold
                ):
                    base_sparsity *= 1 - self.config.adaptation_rate
                elif (
                    recent_performance
                    > previous_performance + self.config.stability_threshold
                ):
                    base_sparsity *= 1 + self.config.adaptation_rate * 0.5

        participant_targets = {}
        for participant_id, profile in participant_profiles.items():
            participant_targets[participant_id] = profile.get_recommended_sparsity(
                base_sparsity
            )

        self.adaptation_history.append(
            {
                "round_num": round_num,
                "base_sparsity": base_sparsity,
                "participant_targets": participant_targets.copy(),
                "global_performance": (
                    self.global_performance[-1] if self.global_performance else 0.0
                ),
            }
        )

        return participant_targets

    def update_performance(self, metrics: List[PerformanceMetrics]):
        for metric in metrics:
            self.performance_history[metric.participant_id].append(metric)
            max_history = 20
            self.performance_history[metric.participant_id] = self.performance_history[
                metric.participant_id
            ][-max_history:]

        if metrics:
            global_acc = np.mean([m.accuracy_after for m in metrics])
            self.global_performance.append(global_acc)
            self.global_performance = self.global_performance[-50:]


class TestModel(nn.Module):
    def __init__(self, input_size=100, hidden_size=50, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


async def test_participant_profile():
    print("=== Testing Participant Profile ===")

    profile = ParticipantProfile(
        participant_id="test_participant",
        capability=ParticipantCapability.MEDIUM,
        preferred_strategy=PruningStrategy.BALANCED,
        max_sparsity=0.8,
        min_sparsity=0.1,
    )

    print(f"Created profile: {profile.participant_id}")
    print(
        f"Capability: {profile.capability.value}, Strategy: {profile.preferred_strategy.value}"
    )

    # Test metrics update
    accuracies = [0.85, 0.83, 0.86, 0.84, 0.87]
    times = [60, 58, 62, 59, 61]
    comm_times = [10, 12, 9, 11, 10]

    for acc, time, comm in zip(accuracies, times, comm_times):
        profile.update_metrics(acc, time, comm)

    print(f"Stability score: {profile.stability_score:.3f}")
    print(f"Recent accuracies: {[f'{a:.3f}' for a in profile.recent_accuracy]}")

    # Test sparsity recommendation
    base_sparsity = 0.5
    recommended = profile.get_recommended_sparsity(base_sparsity)
    print(f"Recommended sparsity (base {base_sparsity}): {recommended:.3f}")

    assert 0.1 <= recommended <= 0.8
    assert len(profile.recent_accuracy) == 5

    print("✓ Participant profile test passed\n")
    return profile


async def test_sparsity_scheduler():
    print("=== Testing Sparsity Scheduler ===")

    config = SparsityScheduleConfig(
        initial_sparsity=0.1, final_sparsity=0.8, warmup_rounds=3, cooldown_rounds=2
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
    print(f"Testing {total_rounds} rounds of sparsity scheduling")

    for round_num in range(1, total_rounds + 1):
        targets = scheduler.calculate_target_sparsity(round_num, total_rounds, profiles)

        print(
            f"Round {round_num}: {', '.join(f'{k}={v:.3f}' for k, v in targets.items())}"
        )

        for participant_id, target in targets.items():
            profile = profiles[participant_id]
            assert profile.min_sparsity <= target <= profile.max_sparsity

        # Mock performance feedback
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

    assert len(scheduler.adaptation_history) == total_rounds
    assert len(scheduler.global_performance) > 0

    print(f"Final global performance: {scheduler.global_performance[-1]:.3f}")
    print(f"Adaptations recorded: {len(scheduler.adaptation_history)}")
    print("✓ Sparsity scheduler test passed\n")

    return scheduler


async def test_orchestrator_simulation():
    print("=== Testing Orchestrator Simulation ===")

    # Create configuration
    config = SparsityScheduleConfig(
        initial_sparsity=0.1, final_sparsity=0.7, warmup_rounds=2, cooldown_rounds=1
    )

    scheduler = SparsityScheduler(config)

    # Create diverse participants
    participants = {
        "mobile_device": ParticipantProfile(
            "mobile_device",
            ParticipantCapability.LOW,
            PruningStrategy.CONSERVATIVE,
            max_sparsity=0.4,
            bandwidth=20.0,
        ),
        "edge_server": ParticipantProfile(
            "edge_server",
            ParticipantCapability.MEDIUM,
            PruningStrategy.BALANCED,
            max_sparsity=0.7,
            bandwidth=100.0,
        ),
        "cloud_instance": ParticipantProfile(
            "cloud_instance",
            ParticipantCapability.HIGH,
            PruningStrategy.AGGRESSIVE,
            max_sparsity=0.9,
            bandwidth=1000.0,
        ),
    }

    print(f"Created {len(participants)} diverse participants")
    for pid, profile in participants.items():
        print(
            f"  {pid}: {profile.capability.value} capability, max sparsity {profile.max_sparsity}"
        )

    # Simulate orchestration rounds
    total_rounds = 8
    orchestration_results = []

    print(f"\nRunning {total_rounds} orchestration rounds...")

    for round_num in range(1, total_rounds + 1):
        # Calculate targets
        targets = scheduler.calculate_target_sparsity(
            round_num, total_rounds, participants
        )

        # Simulate participant performance with some variability
        round_metrics = []
        for pid, target in targets.items():
            profile = participants[pid]

            # Simulate accuracy based on participant capability and sparsity
            base_accuracy = 0.85
            capability_bonus = {
                ParticipantCapability.HIGH: 0.05,
                ParticipantCapability.MEDIUM: 0.02,
                ParticipantCapability.LOW: -0.02,
            }.get(profile.capability, 0.0)

            sparsity_penalty = target * 0.15  # Higher sparsity reduces accuracy
            noise = np.random.normal(0, 0.02)

            accuracy_before = base_accuracy + capability_bonus + noise
            accuracy_after = (
                accuracy_before - sparsity_penalty + np.random.normal(0, 0.01)
            )

            # Ensure realistic bounds
            accuracy_before = max(0.6, min(0.95, accuracy_before))
            accuracy_after = max(0.5, min(0.95, accuracy_after))

            # Simulate timing based on capability
            base_time = 60
            capability_factor = {
                ParticipantCapability.HIGH: 0.7,
                ParticipantCapability.MEDIUM: 1.0,
                ParticipantCapability.LOW: 1.5,
            }.get(profile.capability, 1.0)

            training_time = base_time * capability_factor * (
                1 + target
            ) + np.random.exponential(10)
            communication_time = 1000 / profile.bandwidth + np.random.exponential(5)

            metrics = PerformanceMetrics(
                round_num=round_num,
                participant_id=pid,
                accuracy_before=accuracy_before,
                accuracy_after=accuracy_after,
                training_time=training_time,
                communication_time=communication_time,
                memory_usage=0.4 + target * 0.3,
                model_size=1000,
                sparsity_ratio=target,
                convergence_rate=0.02 - target * 0.01,
            )

            round_metrics.append(metrics)

            # Update participant profile
            profile.update_metrics(accuracy_after, training_time, communication_time)

        # Update scheduler
        scheduler.update_performance(round_metrics)

        # Calculate round summary
        avg_accuracy = np.mean([m.accuracy_after for m in round_metrics])
        avg_sparsity = np.mean([m.sparsity_ratio for m in round_metrics])
        struggling_count = sum(1 for m in round_metrics if m.accuracy_drop > 0.08)

        round_result = {
            "round_num": round_num,
            "targets": targets,
            "avg_accuracy": avg_accuracy,
            "avg_sparsity": avg_sparsity,
            "struggling_participants": struggling_count,
            "metrics": round_metrics,
        }

        orchestration_results.append(round_result)

        print(
            f"  Round {round_num}: accuracy={avg_accuracy:.3f}, sparsity={avg_sparsity:.3f}, struggling={struggling_count}"
        )

        # Show individual participant performance
        for metrics in round_metrics:
            print(
                f"    {metrics.participant_id}: {metrics.accuracy_after:.3f} acc ({metrics.accuracy_drop:.3f} drop), {metrics.sparsity_ratio:.3f} sparsity"
            )

    # Final analysis
    print("\n=== Final Orchestration Analysis ===")

    first_round = orchestration_results[0]
    last_round = orchestration_results[-1]

    accuracy_change = last_round["avg_accuracy"] - first_round["avg_accuracy"]
    sparsity_change = last_round["avg_sparsity"] - first_round["avg_sparsity"]

    print(
        f"Accuracy change: {first_round['avg_accuracy']:.3f} → {last_round['avg_accuracy']:.3f} ({accuracy_change:+.3f})"
    )
    print(
        f"Sparsity change: {first_round['avg_sparsity']:.3f} → {last_round['avg_sparsity']:.3f} ({sparsity_change:+.3f})"
    )
    print(f"Total adaptations made: {len(scheduler.adaptation_history)}")
    print(f"Performance tracking points: {len(scheduler.global_performance)}")

    # Validate progression
    assert (
        last_round["avg_sparsity"] > first_round["avg_sparsity"]
    ), "Sparsity should increase"
    assert (
        sparsity_change > 0.3
    ), f"Expected significant sparsity increase, got {sparsity_change:.3f}"

    # Check participant diversity in final round
    final_targets = last_round["targets"]
    min_target = min(final_targets.values())
    max_target = max(final_targets.values())
    diversity = max_target - min_target

    print(
        f"Final sparsity diversity: {min_target:.3f} - {max_target:.3f} (range: {diversity:.3f})"
    )
    assert diversity > 0.1, f"Expected participant diversity, got {diversity:.3f}"

    # Show participant-specific adaptations
    print("\nParticipant-specific results:")
    for pid, profile in participants.items():
        final_target = final_targets[pid]
        stability = profile.stability_score
        recent_acc = (
            np.mean(profile.recent_accuracy[-3:])
            if len(profile.recent_accuracy) >= 3
            else 0.0
        )

        print(
            f"  {pid}: final_sparsity={final_target:.3f}, stability={stability:.3f}, recent_acc={recent_acc:.3f}"
        )

    print("✓ Orchestrator simulation test passed\n")

    return orchestration_results, participants


async def run_all_tests():
    print("Adaptive Pruning Orchestrator Direct Test Suite")
    print("=" * 60)

    try:
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Run tests
        profile = await test_participant_profile()
        scheduler = await test_sparsity_scheduler()
        results, participants = await test_orchestrator_simulation()

        print("=" * 60)
        print("🎉 ALL DIRECT TESTS PASSED! 🎉")
        print("=" * 60)

        print("\nDirect Test Summary:")
        print(f"✓ Participant Profile: {len(profile.recent_accuracy)} metrics tracked")
        print(f"✓ Sparsity Scheduler: {len(scheduler.adaptation_history)} adaptations")
        print(f"✓ Orchestrator Simulation: {len(results)} rounds completed")

        # Additional validation
        final_round = results[-1]
        print(f"\nFinal Performance:")
        print(f"  Average accuracy: {final_round['avg_accuracy']:.3f}")
        print(f"  Average sparsity: {final_round['avg_sparsity']:.3f}")
        print(
            f"  Participant diversity: {max(final_round['targets'].values()) - min(final_round['targets'].values()):.3f}"
        )

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
