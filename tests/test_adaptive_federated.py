"""
Test suite for Adaptive Federated Learning system

This module provides comprehensive tests for the adaptive federated learning
components including performance monitoring, hyperparameter optimization,
adaptive communication, and participant selection.
"""

import pytest
import asyncio
import time
import torch
import torch.nn as nn
from unittest.mock import Mock, AsyncMock, patch
from collections import deque

from src.vega.federated.adaptive import (
    AdaptiveFederatedLearning,
    PerformanceMonitor,
    HyperparameterOptimizer,
    AdaptiveCommunicationManager,
    ParticipantSelector,
    LearningAlgorithm,
    AdaptationTrigger,
    NetworkCondition,
    ParticipantPerformance,
    AdaptationEvent,
)
from src.vega.federated.participant import Participant


class SimpleModel(nn.Module):
    """Simple model for testing"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


class TestPerformanceMonitor:
    """Test performance monitoring functionality"""

    def setup_method(self):
        self.monitor = PerformanceMonitor(history_size=10)

    def test_record_participant_performance(self):
        """Test recording participant performance"""
        performance = ParticipantPerformance(
            participant_id="participant_1",
            accuracy=0.85,
            loss=0.3,
            training_time=100.0,
            communication_time=20.0,
            reliability_score=0.9,
            contribution_score=0.8,
            resource_utilization=0.7,
        )

        self.monitor.record_participant_performance(performance)

        assert "participant_1" in self.monitor.performance_history
        assert len(self.monitor.performance_history["participant_1"]) == 1
        assert self.monitor.performance_history["participant_1"][0].accuracy == 0.85

    def test_record_global_metrics(self):
        """Test recording global model metrics"""
        self.monitor.record_global_metrics(accuracy=0.9, loss=0.2, round_num=1)

        assert len(self.monitor.global_metrics) == 1
        metrics = self.monitor.global_metrics[0]
        assert metrics["accuracy"] == 0.9
        assert metrics["loss"] == 0.2
        assert metrics["round"] == 1
        assert "timestamp" in metrics

    def test_performance_trend_analysis(self):
        """Test performance trend detection"""
        participant_id = "participant_1"

        # Add improving trend data
        for i in range(10):
            performance = ParticipantPerformance(
                participant_id=participant_id,
                accuracy=0.7 + i * 0.02,  # Improving accuracy
                loss=0.5 - i * 0.02,  # Decreasing loss
                training_time=100.0,
                communication_time=20.0,
                reliability_score=0.9,
                contribution_score=0.8,
                resource_utilization=0.7,
            )
            self.monitor.record_participant_performance(performance)

        # Test trend detection
        accuracy_trend = self.monitor.get_performance_trend(
            participant_id, "accuracy", window=10
        )
        assert accuracy_trend == "improving"

    def test_anomaly_detection_performance_degradation(self):
        """Test detection of performance degradation"""
        # Add metrics showing degradation
        accuracies = [0.9, 0.85, 0.8, 0.75, 0.7]  # Degrading
        for i, acc in enumerate(accuracies):
            self.monitor.record_global_metrics(accuracy=acc, loss=0.3, round_num=i)

        triggers = self.monitor.detect_anomalies()
        assert AdaptationTrigger.PERFORMANCE_DEGRADATION in triggers

    def test_anomaly_detection_convergence_stagnation(self):
        """Test detection of convergence stagnation"""
        # Add metrics showing stagnation (very low variance)
        for i in range(15):
            self.monitor.record_global_metrics(accuracy=0.85, loss=0.25, round_num=i)

        triggers = self.monitor.detect_anomalies()
        assert AdaptationTrigger.CONVERGENCE_STAGNATION in triggers

    def test_network_condition_recording(self):
        """Test network condition recording"""
        condition = NetworkCondition(
            bandwidth_mbps=50.0, latency_ms=100.0, packet_loss_rate=0.05, jitter_ms=10.0
        )

        self.monitor.record_network_condition(condition)

        assert len(self.monitor.network_conditions) == 1
        recorded = self.monitor.network_conditions[0]
        assert recorded.bandwidth_mbps == 50.0
        assert recorded.latency_ms == 100.0


class TestHyperparameterOptimizer:
    """Test hyperparameter optimization functionality"""

    def setup_method(self):
        self.optimizer = HyperparameterOptimizer()

    def test_suggest_fedavg_parameters(self):
        """Test parameter suggestion for FedAvg"""
        params = self.optimizer.suggest_parameters(LearningAlgorithm.FEDAVG)

        assert "learning_rate" in params
        assert "local_epochs" in params
        assert 0.001 <= params["learning_rate"] <= 0.1
        assert params["local_epochs"] in [1.0, 3.0, 5.0, 10.0]

    def test_suggest_fedprox_parameters(self):
        """Test parameter suggestion for FedProx"""
        params = self.optimizer.suggest_parameters(LearningAlgorithm.FEDPROX)

        assert "learning_rate" in params
        assert "mu" in params
        assert "local_epochs" in params
        assert 0.001 <= params["learning_rate"] <= 0.1
        assert 0.001 <= params["mu"] <= 1.0

    def test_parameter_optimization_with_feedback(self):
        """Test parameter optimization with performance feedback"""
        # Simulate parameter history with feedback
        self.optimizer.update_performance({"learning_rate": 0.01}, 0.8)
        self.optimizer.update_performance({"learning_rate": 0.05}, 0.85)
        self.optimizer.update_performance({"learning_rate": 0.02}, 0.9)  # Best
        self.optimizer.update_performance({"learning_rate": 0.1}, 0.7)

        # Get new suggestions
        params = self.optimizer.suggest_parameters(LearningAlgorithm.FEDAVG)

        # Should suggest something close to the best performing value (0.02)
        assert abs(params["learning_rate"] - 0.02) < 0.05

    def test_discrete_parameter_optimization(self):
        """Test optimization of discrete parameters"""
        # Add feedback for local epochs
        for epochs in [1, 3, 5, 10]:
            performance = 0.8 + (epochs * 0.02)  # Higher epochs = better performance
            self.optimizer.update_performance(
                {"local_epochs": float(epochs)}, performance
            )

        params = self.optimizer.suggest_parameters(LearningAlgorithm.FEDAVG)

        # Should prefer higher epoch values
        assert params["local_epochs"] >= 5.0


class TestAdaptiveCommunicationManager:
    """Test adaptive communication management"""

    def setup_method(self):
        self.comm_manager = AdaptiveCommunicationManager()

    def test_adapt_to_high_latency(self):
        """Test adaptation to high latency conditions"""
        high_latency_condition = NetworkCondition(
            bandwidth_mbps=50.0,
            latency_ms=800.0,  # High latency
            packet_loss_rate=0.02,
            jitter_ms=20.0,
        )

        config = self.comm_manager.adapt_communication_protocol(
            high_latency_condition, ["p1", "p2", "p3"]
        )

        assert config["compression_enabled"] is True
        assert config["quantization_bits"] == 8
        assert config["sparsification_ratio"] == 0.9

    def test_adapt_to_low_bandwidth(self):
        """Test adaptation to low bandwidth conditions"""
        low_bandwidth_condition = NetworkCondition(
            bandwidth_mbps=5.0,  # Low bandwidth
            latency_ms=200.0,
            packet_loss_rate=0.02,
            jitter_ms=15.0,
        )

        config = self.comm_manager.adapt_communication_protocol(
            low_bandwidth_condition, ["p1", "p2"]
        )

        assert config["compression_enabled"] is True
        assert config["quantization_bits"] == 8
        assert config["sparsification_ratio"] == 0.9

    def test_adapt_to_high_packet_loss(self):
        """Test adaptation to high packet loss"""
        high_loss_condition = NetworkCondition(
            bandwidth_mbps=20.0,
            latency_ms=100.0,
            packet_loss_rate=0.08,  # High packet loss
            jitter_ms=10.0,
        )

        config = self.comm_manager.adapt_communication_protocol(
            high_loss_condition, ["p1", "p2", "p3"]
        )

        assert config["communication_rounds"] == 2  # Increased redundancy

    def test_adapt_to_very_poor_conditions(self):
        """Test adaptation to very poor network conditions"""
        poor_condition = NetworkCondition(
            bandwidth_mbps=2.0,  # Very low bandwidth
            latency_ms=1200.0,  # Very high latency
            packet_loss_rate=0.15,  # Very high packet loss
            jitter_ms=50.0,
        )

        config = self.comm_manager.adapt_communication_protocol(poor_condition, ["p1"])

        assert config["compression_enabled"] is True
        assert config["quantization_bits"] == 4  # Aggressive quantization
        assert config["sparsification_ratio"] == 0.95
        assert config["batch_size_multiplier"] == 0.5
        assert config["communication_rounds"] == 2


class TestParticipantSelector:
    """Test participant selection functionality"""

    def setup_method(self):
        self.selector = ParticipantSelector(min_participants=2, max_participants=5)

    def test_select_participants_basic(self):
        """Test basic participant selection"""
        participants = [
            ParticipantPerformance(
                participant_id=f"p{i}",
                accuracy=0.8 + i * 0.05,
                loss=0.3 - i * 0.02,
                training_time=100.0,
                communication_time=20.0,
                reliability_score=0.9,
                contribution_score=0.8,
                resource_utilization=0.7,
            )
            for i in range(10)
        ]

        selected = self.selector.select_participants(participants, target_count=3)

        assert len(selected) == 3
        # Should select highest performing participants
        assert "p9" in selected  # Highest accuracy
        assert "p8" in selected
        assert "p7" in selected

    def test_select_all_when_few_available(self):
        """Test selection when fewer participants than minimum"""
        participants = [
            ParticipantPerformance(
                participant_id="p1",
                accuracy=0.8,
                loss=0.3,
                training_time=100.0,
                communication_time=20.0,
                reliability_score=0.9,
                contribution_score=0.8,
                resource_utilization=0.7,
            )
        ]

        selected = self.selector.select_participants(participants)

        assert len(selected) == 1
        assert "p1" in selected

    def test_participant_scoring(self):
        """Test participant scoring mechanism"""
        high_performer = ParticipantPerformance(
            participant_id="high",
            accuracy=0.95,
            loss=0.1,
            training_time=50.0,  # Fast training
            communication_time=10.0,
            reliability_score=0.95,
            contribution_score=0.9,
            resource_utilization=0.8,
        )

        low_performer = ParticipantPerformance(
            participant_id="low",
            accuracy=0.7,
            loss=0.5,
            training_time=200.0,  # Slow training
            communication_time=50.0,
            reliability_score=0.6,
            contribution_score=0.5,
            resource_utilization=0.9,
        )

        high_score = self.selector._calculate_participant_score(high_performer)
        low_score = self.selector._calculate_participant_score(low_performer)

        assert high_score > low_score


class TestAdaptiveFederatedLearning:
    """Test the main adaptive federated learning system"""

    def setup_method(self):
        self.adaptive_fl = AdaptiveFederatedLearning()

        # Create mock participants
        self.participants = []
        for i in range(5):
            participant = Mock(spec=Participant)
            participant.id = f"participant_{i}"
            participant.train = AsyncMock(return_value={"accuracy": 0.8, "loss": 0.3})
            self.participants.append(participant)

        self.global_model = SimpleModel()

    @patch("src.vega.federated.adaptive.FedAvgAlgorithm")
    @patch("src.vega.federated.adaptive.FedProxAlgorithm")
    @patch("src.vega.federated.adaptive.SCAFFOLDAlgorithm")
    async def test_basic_adaptive_training(
        self, mock_scaffold, mock_fedprox, mock_fedavg
    ):
        """Test basic adaptive training functionality"""
        # Setup mocks
        mock_algorithm = AsyncMock()
        mock_algorithm.run_round = AsyncMock(
            return_value={
                "accuracy": 0.85,
                "loss": 0.25,
                "participant_0_accuracy": 0.8,
                "participant_1_accuracy": 0.9,
            }
        )

        mock_fedavg.return_value = mock_algorithm
        mock_fedprox.return_value = mock_algorithm
        mock_scaffold.return_value = mock_algorithm

        # Run training
        results = await self.adaptive_fl.run_adaptive_training(
            participants=self.participants[:2],
            global_model=self.global_model,
            num_rounds=3,
        )

        assert "accuracy_history" in results
        assert "loss_history" in results
        assert len(results["accuracy_history"]) == 3
        assert len(results["loss_history"]) == 3
        assert results["final_accuracy"] > 0

    def test_algorithm_switching_on_performance_degradation(self):
        """Test algorithm switching when performance degrades"""
        # Simulate performance degradation
        for i in range(6):
            accuracy = 0.9 - i * 0.05  # Degrading performance
            self.adaptive_fl.performance_monitor.record_global_metrics(accuracy, 0.3, i)

        triggers = self.adaptive_fl.performance_monitor.detect_anomalies()
        assert AdaptationTrigger.PERFORMANCE_DEGRADATION in triggers

    def test_hyperparameter_optimization_integration(self):
        """Test integration with hyperparameter optimization"""
        params = self.adaptive_fl.hyperparameter_optimizer.suggest_parameters(
            LearningAlgorithm.FEDAVG
        )

        assert "learning_rate" in params
        assert "local_epochs" in params

        # Update with performance feedback
        self.adaptive_fl.hyperparameter_optimizer.update_performance(params, 0.85)

        # Get new parameters (should be influenced by feedback)
        new_params = self.adaptive_fl.hyperparameter_optimizer.suggest_parameters(
            LearningAlgorithm.FEDAVG
        )

        assert "learning_rate" in new_params
        assert "local_epochs" in new_params

    async def test_network_condition_adaptation(self):
        """Test adaptation to network conditions"""
        # Test network measurement
        condition = await self.adaptive_fl._measure_network_conditions()

        assert isinstance(condition, NetworkCondition)
        assert condition.bandwidth_mbps > 0
        assert condition.latency_ms > 0
        assert 0 <= condition.packet_loss_rate <= 1

    def test_adaptation_event_recording(self):
        """Test that adaptation events are properly recorded"""
        initial_events = len(self.adaptive_fl.adaptation_events)

        # Simulate adaptation trigger
        triggers = [AdaptationTrigger.PERFORMANCE_DEGRADATION]

        # This would be called during training
        asyncio.run(
            self.adaptive_fl._handle_adaptation_triggers(triggers, self.participants)
        )

        assert len(self.adaptive_fl.adaptation_events) > initial_events

        latest_event = self.adaptive_fl.adaptation_events[-1]
        assert latest_event.trigger == AdaptationTrigger.PERFORMANCE_DEGRADATION
        assert latest_event.action_taken != ""

    def test_robust_algorithm_selection(self):
        """Test selection of robust algorithms"""
        # Test switching from FedAvg
        self.adaptive_fl.current_algorithm = LearningAlgorithm.FEDAVG
        robust = self.adaptive_fl._select_robust_algorithm()
        assert robust == LearningAlgorithm.FEDPROX

        # Test switching from FedProx
        self.adaptive_fl.current_algorithm = LearningAlgorithm.FEDPROX
        robust = self.adaptive_fl._select_robust_algorithm()
        assert robust == LearningAlgorithm.SCAFFOLD

        # Test switching from SCAFFOLD
        self.adaptive_fl.current_algorithm = LearningAlgorithm.SCAFFOLD
        robust = self.adaptive_fl._select_robust_algorithm()
        assert robust == LearningAlgorithm.FEDPROX


@pytest.fixture
def sample_participants():
    """Fixture for sample participant performances"""
    return [
        ParticipantPerformance(
            participant_id=f"p{i}",
            accuracy=0.7 + i * 0.05,
            loss=0.5 - i * 0.03,
            training_time=100 + i * 10,
            communication_time=20 + i * 2,
            reliability_score=0.8 + i * 0.02,
            contribution_score=0.75 + i * 0.03,
            resource_utilization=0.6 + i * 0.05,
        )
        for i in range(8)
    ]


class TestAdaptationIntegration:
    """Test integration of all adaptive components"""

    def test_end_to_end_adaptation_flow(self, sample_participants):
        """Test complete adaptation workflow"""
        adaptive_fl = AdaptiveFederatedLearning()

        # 1. Record participant performances
        for perf in sample_participants:
            adaptive_fl.performance_monitor.record_participant_performance(perf)

        # 2. Select participants
        selector = ParticipantSelector()
        selected_ids = selector.select_participants(sample_participants, target_count=3)
        assert len(selected_ids) == 3

        # 3. Optimize hyperparameters
        optimizer = HyperparameterOptimizer()
        params = optimizer.suggest_parameters(LearningAlgorithm.FEDAVG)
        assert "learning_rate" in params

        # 4. Adapt communication
        comm_manager = AdaptiveCommunicationManager()
        network_condition = NetworkCondition(
            bandwidth_mbps=10.0, latency_ms=200.0, packet_loss_rate=0.05, jitter_ms=15.0
        )
        comm_config = comm_manager.adapt_communication_protocol(
            network_condition, selected_ids
        )
        assert "compression_enabled" in comm_config

        # All components work together
        assert len(selected_ids) > 0
        assert len(params) > 0
        assert len(comm_config) > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
