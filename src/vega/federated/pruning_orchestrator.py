"""
Adaptive Pruning Orchestrator for Federated Learning
====================================================

This module provides intelligent orchestration of pruning operations across
federated learning rounds, including dynamic sparsity scheduling, participant-specific
strategies, performance monitoring, and recovery mechanisms.

Features:
- Dynamic sparsity scheduling based on training progress
- Participant-specific pruning strategies based on computational constraints
- Model performance monitoring and automatic adjustment
- Federated distillation coordination with teacher-student networks
- Recovery mechanisms for over-pruned models
- Adaptive learning rate adjustment for pruned models

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
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParticipantCapability(Enum):
    """Participant computational capability levels."""

    HIGH = "high"  # Strong devices, can handle heavy pruning
    MEDIUM = "medium"  # Average devices, moderate pruning
    LOW = "low"  # Weak devices, light pruning only
    VARIABLE = "variable"  # Capability changes over time


class PruningStrategy(Enum):
    """Different pruning strategies for participants."""

    AGGRESSIVE = "aggressive"  # High sparsity, fast convergence
    CONSERVATIVE = "conservative"  # Low sparsity, stable training
    ADAPTIVE = "adaptive"  # Dynamic adjustment based on performance
    BALANCED = "balanced"  # Moderate sparsity with good stability


@dataclass
class ParticipantProfile:
    """Profile information for a federated learning participant."""

    participant_id: str
    capability: ParticipantCapability
    preferred_strategy: PruningStrategy
    max_sparsity: float = 0.9
    min_sparsity: float = 0.0
    computational_budget: float = 1.0  # Relative computational capacity
    bandwidth: float = 100.0  # Mbps
    latency: float = 50.0  # milliseconds
    accuracy_tolerance: float = 0.05  # Maximum acceptable accuracy drop

    # Dynamic metrics
    recent_accuracy: List[float] = field(default_factory=list)
    recent_training_time: List[float] = field(default_factory=list)
    recent_communication_time: List[float] = field(default_factory=list)
    stability_score: float = 1.0

    def update_metrics(self, accuracy: float, training_time: float, comm_time: float):
        """Update participant performance metrics."""
        self.recent_accuracy.append(accuracy)
        self.recent_training_time.append(training_time)
        self.recent_communication_time.append(comm_time)

        # Keep only recent history
        max_history = 10
        self.recent_accuracy = self.recent_accuracy[-max_history:]
        self.recent_training_time = self.recent_training_time[-max_history:]
        self.recent_communication_time = self.recent_communication_time[-max_history:]

        # Calculate stability score
        if len(self.recent_accuracy) >= 3:
            acc_variance = np.var(self.recent_accuracy)
            self.stability_score = max(0.1, 1.0 - acc_variance)

    def get_recommended_sparsity(self, base_sparsity: float) -> float:
        """Get recommended sparsity based on participant profile."""
        # Adjust based on capability
        capability_multiplier = {
            ParticipantCapability.HIGH: 1.2,
            ParticipantCapability.MEDIUM: 1.0,
            ParticipantCapability.LOW: 0.7,
            ParticipantCapability.VARIABLE: 0.9,
        }

        # Adjust based on stability
        stability_factor = 0.5 + 0.5 * self.stability_score

        # Adjust based on computational budget
        budget_factor = min(self.computational_budget, 1.5)

        recommended = (
            base_sparsity
            * capability_multiplier[self.capability]
            * stability_factor
            * budget_factor
        )

        return max(self.min_sparsity, min(self.max_sparsity, recommended))


@dataclass
class SparsityScheduleConfig:
    """Configuration for sparsity scheduling."""

    initial_sparsity: float = 0.1
    final_sparsity: float = 0.8
    warmup_rounds: int = 5
    cooldown_rounds: int = 10
    adaptation_rate: float = 0.1
    stability_threshold: float = 0.02
    performance_threshold: float = 0.05


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""

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


class SparsityScheduler:
    """Dynamic sparsity scheduler for federated pruning."""

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
        """Calculate target sparsity for each participant."""
        # Base sparsity progression
        progress = min(round_num / total_rounds, 1.0)

        # Apply warmup and cooldown
        if round_num <= self.config.warmup_rounds:
            warmup_factor = round_num / self.config.warmup_rounds
            base_sparsity = self.config.initial_sparsity * warmup_factor
        elif round_num >= total_rounds - self.config.cooldown_rounds:
            cooldown_progress = (total_rounds - round_num) / self.config.cooldown_rounds
            base_sparsity = self.config.final_sparsity * (1 - cooldown_progress * 0.2)
        else:
            # Normal progression
            adjusted_progress = (round_num - self.config.warmup_rounds) / (
                total_rounds - self.config.warmup_rounds - self.config.cooldown_rounds
            )
            base_sparsity = self.config.initial_sparsity + adjusted_progress * (
                self.config.final_sparsity - self.config.initial_sparsity
            )

        # Adaptive adjustment based on global performance
        if len(self.global_performance) >= 3:
            recent_performance = np.mean(self.global_performance[-3:])
            if len(self.global_performance) >= 6:
                previous_performance = np.mean(self.global_performance[-6:-3])
                if (
                    recent_performance
                    < previous_performance - self.config.performance_threshold
                ):
                    # Performance degrading - reduce sparsity
                    base_sparsity *= 1 - self.config.adaptation_rate
                    logger.info(
                        f"Reducing sparsity due to performance degradation: {base_sparsity:.3f}"
                    )
                elif (
                    recent_performance
                    > previous_performance + self.config.stability_threshold
                ):
                    # Performance improving - can increase sparsity
                    base_sparsity *= 1 + self.config.adaptation_rate * 0.5
                    logger.info(
                        f"Increasing sparsity due to good performance: {base_sparsity:.3f}"
                    )

        # Calculate participant-specific targets
        participant_targets = {}
        for participant_id, profile in participant_profiles.items():
            participant_targets[participant_id] = profile.get_recommended_sparsity(
                base_sparsity
            )

        # Log adaptation decision
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
        """Update performance history with new metrics."""
        for metric in metrics:
            self.performance_history[metric.participant_id].append(metric)

            # Keep only recent history
            max_history = 20
            self.performance_history[metric.participant_id] = self.performance_history[
                metric.participant_id
            ][-max_history:]

        # Calculate global performance
        if metrics:
            global_acc = np.mean([m.accuracy_after for m in metrics])
            self.global_performance.append(global_acc)
            self.global_performance = self.global_performance[
                -50:
            ]  # Keep recent history


class PerformanceMonitor:
    """Monitor and analyze performance across federated pruning rounds."""

    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.alerts: List[Dict[str, Any]] = []
        self.thresholds = {
            "accuracy_drop": 0.1,
            "training_time": 300.0,  # 5 minutes
            "memory_usage": 0.8,  # 80% of capacity
            "convergence_rate": 0.01,
        }

    async def monitor_round(
        self,
        round_metrics: List[PerformanceMetrics],
        participant_profiles: Dict[str, ParticipantProfile],
    ) -> Dict[str, Any]:
        """Monitor performance for a training round."""
        monitoring_results = {
            "alerts": [],
            "recommendations": [],
            "summary": {},
            "participant_health": {},
        }

        # Analyze each participant
        for metrics in round_metrics:
            participant_health = await self._analyze_participant_health(
                metrics, participant_profiles.get(metrics.participant_id)
            )
            monitoring_results["participant_health"][
                metrics.participant_id
            ] = participant_health

            # Check for alerts
            alerts = self._check_performance_alerts(metrics)
            monitoring_results["alerts"].extend(alerts)

            # Update participant profile
            if metrics.participant_id in participant_profiles:
                participant_profiles[metrics.participant_id].update_metrics(
                    metrics.accuracy_after,
                    metrics.training_time,
                    metrics.communication_time,
                )

        # Generate recommendations
        recommendations = await self._generate_recommendations(
            round_metrics, participant_profiles
        )
        monitoring_results["recommendations"] = recommendations

        # Create summary
        monitoring_results["summary"] = self._create_round_summary(round_metrics)

        # Store metrics
        self.metrics_history.extend(round_metrics)

        return monitoring_results

    async def _analyze_participant_health(
        self, metrics: PerformanceMetrics, profile: Optional[ParticipantProfile]
    ) -> Dict[str, Any]:
        """Analyze health status of a participant."""
        health_status = {"status": "healthy", "issues": [], "score": 1.0, "trends": {}}

        # Check accuracy drop
        if metrics.accuracy_drop > self.thresholds["accuracy_drop"]:
            health_status["issues"].append(
                f"High accuracy drop: {metrics.accuracy_drop:.3f}"
            )
            health_status["score"] *= 0.7

        # Check training time
        if metrics.training_time > self.thresholds["training_time"]:
            health_status["issues"].append(
                f"Long training time: {metrics.training_time:.1f}s"
            )
            health_status["score"] *= 0.8

        # Check memory usage
        if metrics.memory_usage > self.thresholds["memory_usage"]:
            health_status["issues"].append(
                f"High memory usage: {metrics.memory_usage:.2f}"
            )
            health_status["score"] *= 0.9

        # Analyze trends if profile available
        if profile and len(profile.recent_accuracy) >= 3:
            acc_trend = np.polyfit(
                range(len(profile.recent_accuracy)), profile.recent_accuracy, 1
            )[0]
            health_status["trends"]["accuracy"] = (
                "improving" if acc_trend > 0 else "degrading"
            )

            if acc_trend < -0.02:  # Significant degradation
                health_status["issues"].append("Accuracy trend degrading")
                health_status["score"] *= 0.8

        # Determine overall status
        if health_status["score"] < 0.5:
            health_status["status"] = "critical"
        elif health_status["score"] < 0.7:
            health_status["status"] = "warning"
        elif len(health_status["issues"]) > 0:
            health_status["status"] = "attention"

        return health_status

    def _check_performance_alerts(
        self, metrics: PerformanceMetrics
    ) -> List[Dict[str, Any]]:
        """Check for performance alerts."""
        alerts = []

        # Critical accuracy drop
        if metrics.accuracy_drop > self.thresholds["accuracy_drop"]:
            alerts.append(
                {
                    "type": "accuracy_drop",
                    "severity": "high",
                    "participant": metrics.participant_id,
                    "round": metrics.round_num,
                    "value": metrics.accuracy_drop,
                    "threshold": self.thresholds["accuracy_drop"],
                    "message": f"Participant {metrics.participant_id} experienced {metrics.accuracy_drop:.3f} accuracy drop",
                }
            )

        # Long training time
        if metrics.training_time > self.thresholds["training_time"]:
            alerts.append(
                {
                    "type": "training_time",
                    "severity": "medium",
                    "participant": metrics.participant_id,
                    "round": metrics.round_num,
                    "value": metrics.training_time,
                    "threshold": self.thresholds["training_time"],
                    "message": f"Participant {metrics.participant_id} training took {metrics.training_time:.1f}s",
                }
            )

        return alerts

    async def _generate_recommendations(
        self,
        round_metrics: List[PerformanceMetrics],
        participant_profiles: Dict[str, ParticipantProfile],
    ) -> List[Dict[str, Any]]:
        """Generate optimization recommendations."""
        recommendations = []

        # Analyze overall round performance
        avg_accuracy_drop = np.mean([m.accuracy_drop for m in round_metrics])

        if avg_accuracy_drop > 0.05:
            recommendations.append(
                {
                    "type": "reduce_sparsity",
                    "priority": "high",
                    "message": f"Consider reducing sparsity targets (avg drop: {avg_accuracy_drop:.3f})",
                    "action": "Reduce target sparsity by 10-20%",
                }
            )

        # Check for struggling participants
        for metrics in round_metrics:
            if metrics.accuracy_drop > 0.08:
                recommendations.append(
                    {
                        "type": "participant_assistance",
                        "priority": "high",
                        "participant": metrics.participant_id,
                        "message": f"Participant {metrics.participant_id} needs assistance",
                        "action": "Apply knowledge distillation or reduce participant sparsity",
                    }
                )

        # Check for training efficiency
        long_training = [m for m in round_metrics if m.training_time > 200]
        if len(long_training) > len(round_metrics) * 0.3:
            recommendations.append(
                {
                    "type": "optimize_training",
                    "priority": "medium",
                    "message": "Multiple participants have long training times",
                    "action": "Consider increasing pruning frequency or optimizing model architecture",
                }
            )

        return recommendations

    def _create_round_summary(
        self, round_metrics: List[PerformanceMetrics]
    ) -> Dict[str, Any]:
        """Create summary statistics for the round."""
        if not round_metrics:
            return {}

        accuracies = [m.accuracy_after for m in round_metrics]
        accuracy_drops = [m.accuracy_drop for m in round_metrics]
        training_times = [m.training_time for m in round_metrics]
        sparsity_ratios = [m.sparsity_ratio for m in round_metrics]

        return {
            "participants": len(round_metrics),
            "average_accuracy": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "average_accuracy_drop": np.mean(accuracy_drops),
            "max_accuracy_drop": np.max(accuracy_drops),
            "average_training_time": np.mean(training_times),
            "total_training_time": np.sum(training_times),
            "average_sparsity": np.mean(sparsity_ratios),
            "sparsity_range": (np.min(sparsity_ratios), np.max(sparsity_ratios)),
            "healthy_participants": sum(
                1 for m in round_metrics if m.accuracy_drop < 0.05
            ),
            "struggling_participants": sum(
                1 for m in round_metrics if m.accuracy_drop > 0.08
            ),
        }


class DistillationCoordinator:
    """Coordinate federated knowledge distillation for recovery."""

    def __init__(self):
        self.distillation_history: List[Dict[str, Any]] = []
        self.teacher_models: Dict[str, nn.Module] = {}  # Store teacher models

    async def coordinate_recovery_distillation(
        self,
        struggling_participants: List[str],
        participant_models: Dict[str, nn.Module],
        global_model: nn.Module,
        performance_monitor: PerformanceMonitor,
    ) -> Dict[str, Any]:
        """Coordinate knowledge distillation for struggling participants."""
        logger.info(
            f"Coordinating recovery distillation for {len(struggling_participants)} participants"
        )

        distillation_results = {}

        for participant_id in struggling_participants:
            if participant_id in participant_models:
                logger.info(
                    f"Applying knowledge distillation for participant {participant_id}"
                )

                # Use global model as teacher
                teacher_model = global_model
                student_model = participant_models[participant_id]

                # Perform distillation
                improved_model, metrics = await self._perform_distillation(
                    teacher_model, student_model, participant_id
                )

                # Update participant model
                participant_models[participant_id] = improved_model
                distillation_results[participant_id] = metrics

                # Log results
                logger.info(
                    f"Distillation completed for {participant_id}: "
                    f"retention = {metrics['knowledge_retention']:.3f}"
                )

        # Record distillation session
        session_record = {
            "timestamp": time.time(),
            "participants": struggling_participants,
            "results": distillation_results,
            "success_rate": sum(
                1
                for r in distillation_results.values()
                if r["knowledge_retention"] > 0.7
            )
            / len(distillation_results),
        }

        self.distillation_history.append(session_record)

        return session_record

    async def _perform_distillation(
        self, teacher_model: nn.Module, student_model: nn.Module, participant_id: str
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Perform knowledge distillation for a specific participant."""
        from .pruning import FederatedDistillation

        distillation = FederatedDistillation(temperature=4.0, alpha=0.8)

        # Shorter distillation for recovery
        improved_model, metrics = await distillation.distill_knowledge(
            teacher_model=teacher_model, student_model=student_model, num_epochs=3
        )

        return improved_model, metrics


class AdaptivePruningOrchestrator:
    """Main orchestrator for adaptive federated pruning."""

    def __init__(self, schedule_config: SparsityScheduleConfig):
        self.scheduler = SparsityScheduler(schedule_config)
        self.monitor = PerformanceMonitor()
        self.distillation_coordinator = DistillationCoordinator()
        self.participant_profiles: Dict[str, ParticipantProfile] = {}
        self.orchestration_history: List[Dict[str, Any]] = []

    def register_participant(
        self,
        participant_id: str,
        capability: ParticipantCapability,
        strategy: PruningStrategy,
        **kwargs,
    ):
        """Register a new participant with their profile."""
        profile = ParticipantProfile(
            participant_id=participant_id,
            capability=capability,
            preferred_strategy=strategy,
            **kwargs,
        )
        self.participant_profiles[participant_id] = profile
        logger.info(
            f"Registered participant {participant_id} with {capability.value} capability"
        )

    async def orchestrate_pruning_round(
        self,
        round_num: int,
        total_rounds: int,
        participant_models: Dict[str, nn.Module],
        global_model: nn.Module,
        from_pruning_coordinator: Any = None,  # PruningCoordinator instance
    ) -> Dict[str, Any]:
        """Orchestrate an adaptive pruning round."""
        logger.info(f"Orchestrating adaptive pruning round {round_num}/{total_rounds}")

        round_start_time = time.time()

        # Calculate participant-specific sparsity targets
        sparsity_targets = self.scheduler.calculate_target_sparsity(
            round_num, total_rounds, self.participant_profiles
        )

        logger.info(f"Sparsity targets: {sparsity_targets}")

        # Perform pruning for each participant (simplified simulation)
        round_metrics = []
        updated_models = {}

        for participant_id, target_sparsity in sparsity_targets.items():
            if participant_id in participant_models:
                # Simulate pruning and collect metrics
                model = participant_models[participant_id]

                # Simulate performance metrics
                accuracy_before = 0.85 + np.random.normal(0, 0.05)
                accuracy_after = (
                    accuracy_before - target_sparsity * 0.1 + np.random.normal(0, 0.02)
                )
                training_time = 60 + target_sparsity * 30 + np.random.exponential(20)
                communication_time = 10 + np.random.exponential(5)
                memory_usage = (
                    0.4 + target_sparsity * 0.3 + np.random.uniform(-0.1, 0.1)
                )

                metrics = PerformanceMetrics(
                    round_num=round_num,
                    participant_id=participant_id,
                    accuracy_before=max(0, accuracy_before),
                    accuracy_after=max(0, accuracy_after),
                    training_time=training_time,
                    communication_time=communication_time,
                    memory_usage=max(0, min(1, memory_usage)),
                    model_size=sum(p.numel() for p in model.parameters()),
                    sparsity_ratio=target_sparsity,
                    convergence_rate=0.02 - target_sparsity * 0.01,
                )

                round_metrics.append(metrics)
                updated_models[participant_id] = model

        # Monitor performance and get insights
        monitoring_results = await self.monitor.monitor_round(
            round_metrics, self.participant_profiles
        )

        # Update scheduler with performance data
        self.scheduler.update_performance(round_metrics)

        # Handle struggling participants
        struggling_participants = [
            m.participant_id for m in round_metrics if m.accuracy_drop > 0.08
        ]

        distillation_results = {}
        if struggling_participants:
            distillation_results = (
                await self.distillation_coordinator.coordinate_recovery_distillation(
                    struggling_participants, updated_models, global_model, self.monitor
                )
            )

        round_time = time.time() - round_start_time

        # Create comprehensive round results
        round_results = {
            "round_num": round_num,
            "orchestration_time": round_time,
            "sparsity_targets": sparsity_targets,
            "participant_metrics": [m.__dict__ for m in round_metrics],
            "monitoring_results": monitoring_results,
            "distillation_results": distillation_results,
            "performance_summary": {
                "average_accuracy": np.mean([m.accuracy_after for m in round_metrics]),
                "average_sparsity": np.mean(list(sparsity_targets.values())),
                "struggling_participants": len(struggling_participants),
                "healthy_participants": monitoring_results["summary"].get(
                    "healthy_participants", 0
                ),
            },
            "adaptations_made": len(self.scheduler.adaptation_history),
        }

        self.orchestration_history.append(round_results)

        logger.info(
            f"Round {round_num} orchestration completed: "
            f"{round_results['performance_summary']['average_accuracy']:.3f} avg accuracy, "
            f"{round_results['performance_summary']['average_sparsity']:.3f} avg sparsity"
        )

        return round_results

    def get_orchestration_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of orchestration activities."""
        if not self.orchestration_history:
            return {"message": "No orchestration rounds completed"}

        total_rounds = len(self.orchestration_history)

        # Aggregate metrics
        avg_accuracies = [
            r["performance_summary"]["average_accuracy"]
            for r in self.orchestration_history
        ]
        avg_sparsities = [
            r["performance_summary"]["average_sparsity"]
            for r in self.orchestration_history
        ]
        struggling_counts = [
            r["performance_summary"]["struggling_participants"]
            for r in self.orchestration_history
        ]

        summary = {
            "total_rounds": total_rounds,
            "total_orchestration_time": sum(
                r["orchestration_time"] for r in self.orchestration_history
            ),
            "average_orchestration_time": np.mean(
                [r["orchestration_time"] for r in self.orchestration_history]
            ),
            "accuracy_progression": avg_accuracies,
            "sparsity_progression": avg_sparsities,
            "final_average_accuracy": avg_accuracies[-1] if avg_accuracies else 0.0,
            "final_average_sparsity": avg_sparsities[-1] if avg_sparsities else 0.0,
            "accuracy_improvement": (
                avg_accuracies[-1] - avg_accuracies[0]
                if len(avg_accuracies) > 1
                else 0.0
            ),
            "total_struggling_participant_incidents": sum(struggling_counts),
            "distillation_interventions": sum(
                1 for r in self.orchestration_history if r["distillation_results"]
            ),
            "total_participants": len(self.participant_profiles),
            "participant_capabilities": {
                cap.value: sum(
                    1 for p in self.participant_profiles.values() if p.capability == cap
                )
                for cap in ParticipantCapability
            },
            "adaptations_made": len(self.scheduler.adaptation_history),
            "alerts_generated": sum(
                len(r["monitoring_results"]["alerts"])
                for r in self.orchestration_history
            ),
            "recommendations_made": sum(
                len(r["monitoring_results"]["recommendations"])
                for r in self.orchestration_history
            ),
        }

        return summary

    async def save_orchestration_history(self, filepath: str):
        """Save complete orchestration history to file."""
        history_data = {
            "participant_profiles": {
                pid: {
                    "participant_id": profile.participant_id,
                    "capability": profile.capability.value,
                    "preferred_strategy": profile.preferred_strategy.value,
                    "max_sparsity": profile.max_sparsity,
                    "min_sparsity": profile.min_sparsity,
                    "computational_budget": profile.computational_budget,
                    "bandwidth": profile.bandwidth,
                    "latency": profile.latency,
                    "accuracy_tolerance": profile.accuracy_tolerance,
                    "stability_score": profile.stability_score,
                }
                for pid, profile in self.participant_profiles.items()
            },
            "orchestration_history": self.orchestration_history,
            "scheduler_adaptations": self.scheduler.adaptation_history,
            "distillation_history": self.distillation_coordinator.distillation_history,
            "summary": self.get_orchestration_summary(),
        }

        with open(filepath, "w") as f:
            json.dump(history_data, f, indent=2, default=str)

        logger.info(f"Orchestration history saved to {filepath}")


# Example usage and demonstration
async def demonstrate_adaptive_orchestration():
    """Demonstrate the adaptive pruning orchestration system."""
    print("=== Adaptive Federated Pruning Orchestration Demonstration ===\n")

    # Create orchestrator
    schedule_config = SparsityScheduleConfig(
        initial_sparsity=0.1,
        final_sparsity=0.8,
        warmup_rounds=3,
        cooldown_rounds=2,
        adaptation_rate=0.15,
    )

    orchestrator = AdaptivePruningOrchestrator(schedule_config)

    # Register diverse participants
    participants = [
        (
            "participant_1",
            ParticipantCapability.HIGH,
            PruningStrategy.AGGRESSIVE,
            {"max_sparsity": 0.9, "bandwidth": 150.0},
        ),
        (
            "participant_2",
            ParticipantCapability.MEDIUM,
            PruningStrategy.BALANCED,
            {"max_sparsity": 0.7, "bandwidth": 100.0},
        ),
        (
            "participant_3",
            ParticipantCapability.LOW,
            PruningStrategy.CONSERVATIVE,
            {"max_sparsity": 0.5, "bandwidth": 50.0},
        ),
        (
            "participant_4",
            ParticipantCapability.VARIABLE,
            PruningStrategy.ADAPTIVE,
            {"max_sparsity": 0.8, "bandwidth": 80.0},
        ),
    ]

    for pid, capability, strategy, kwargs in participants:
        orchestrator.register_participant(pid, capability, strategy, **kwargs)

    print(f"Registered {len(participants)} participants with diverse capabilities")

    # Create mock models
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(100, 10)

        def forward(self, x):
            return self.fc(x)

    participant_models = {pid: MockModel() for pid, _, _, _ in participants}
    global_model = MockModel()

    # Simulate federated learning rounds
    total_rounds = 10

    print(f"\nStarting {total_rounds} rounds of adaptive orchestration...\n")

    for round_num in range(1, total_rounds + 1):
        print(f"Round {round_num}/{total_rounds}")

        results = await orchestrator.orchestrate_pruning_round(
            round_num=round_num,
            total_rounds=total_rounds,
            participant_models=participant_models,
            global_model=global_model,
        )

        summary = results["performance_summary"]
        print(f"  Average Accuracy: {summary['average_accuracy']:.3f}")
        print(f"  Average Sparsity: {summary['average_sparsity']:.3f}")
        print(f"  Struggling Participants: {summary['struggling_participants']}")
        print(f"  Healthy Participants: {summary['healthy_participants']}")

        if results["distillation_results"]:
            print(
                f"  Knowledge Distillation Applied: {len(results['distillation_results'])} participants"
            )

        alerts = results["monitoring_results"]["alerts"]
        if alerts:
            print(f"  Alerts Generated: {len(alerts)}")

        recommendations = results["monitoring_results"]["recommendations"]
        if recommendations:
            print(f"  Recommendations: {len(recommendations)}")

        print()

    # Get final summary
    summary = orchestrator.get_orchestration_summary()

    print("=== Orchestration Summary ===")
    print(f"Total Rounds: {summary['total_rounds']}")
    print(f"Final Average Accuracy: {summary['final_average_accuracy']:.3f}")
    print(f"Final Average Sparsity: {summary['final_average_sparsity']:.3f}")
    print(f"Accuracy Improvement: {summary['accuracy_improvement']:.3f}")
    print(f"Total Orchestration Time: {summary['total_orchestration_time']:.2f}s")
    print(f"Distillation Interventions: {summary['distillation_interventions']}")
    print(f"Total Adaptations Made: {summary['adaptations_made']}")
    print(f"Total Alerts: {summary['alerts_generated']}")
    print(f"Total Recommendations: {summary['recommendations_made']}")

    # Save history
    await orchestrator.save_orchestration_history("adaptive_orchestration_history.json")
    print("\nOrchestration history saved to adaptive_orchestration_history.json")

    return orchestrator, summary


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstrate_adaptive_orchestration())
