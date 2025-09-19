"""
Comprehensive unit tests for Vega 2.0 federated learning participant module.
Tests participant registration, training, communication, and security integration.
"""

import pytest
import asyncio
import json
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from typing import Dict, Any, List

# Import the participant module
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.vega.federated.participant import (
    FederatedParticipant,
    LocalTrainingConfig,
    ParticipantMetrics,
    TrainingProgressCallback,
    TrainingState,
)
from src.vega.federated.model_serialization import ModelWeights


class TestLocalTrainingConfig:
    """Test LocalTrainingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LocalTrainingConfig()

        assert config.epochs == 5
        assert config.batch_size == 32
        assert config.learning_rate == 0.001
        assert config.optimizer == "adam"
        assert config.loss_function == "mse"
        assert config.validation_split == 0.2
        assert config.early_stopping_patience == 3

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LocalTrainingConfig(
            epochs=10,
            batch_size=64,
            learning_rate=0.01,
            optimizer="sgd",
            loss_function="crossentropy",
        )

        assert config.epochs == 10
        assert config.batch_size == 64
        assert config.learning_rate == 0.01
        assert config.optimizer == "sgd"
        assert config.loss_function == "crossentropy"


class TestParticipantMetrics:
    """Test ParticipantMetrics dataclass."""

    def test_metrics_initialization(self):
        """Test metrics initialization with default values."""
        metrics = ParticipantMetrics()

        assert metrics.total_training_time == 0.0
        assert metrics.total_rounds_participated == 0
        assert metrics.average_loss == 0.0
        assert metrics.best_accuracy == 0.0
        assert metrics.data_points_count == 0
        assert metrics.training_errors == 0
        assert metrics.communication_failures == 0

    def test_metrics_update(self):
        """Test metrics updates."""
        metrics = ParticipantMetrics()

        # Update some metrics
        metrics.total_training_time = 120.5
        metrics.total_rounds_participated = 5
        metrics.average_loss = 0.25
        metrics.best_accuracy = 0.95

        assert metrics.total_training_time == 120.5
        assert metrics.total_rounds_participated == 5
        assert metrics.average_loss == 0.25
        assert metrics.best_accuracy == 0.95


class TestTrainingState:
    """Test TrainingState enum."""

    def test_training_states(self):
        """Test all training state values."""
        assert TrainingState.IDLE.value == "idle"
        assert TrainingState.TRAINING.value == "training"
        assert TrainingState.UPLOADING.value == "uploading"
        assert TrainingState.WAITING.value == "waiting"
        assert TrainingState.ERROR.value == "error"


class TestFederatedParticipant:
    """Test FederatedParticipant class functionality."""

    @pytest.fixture
    def mock_communication_manager(self):
        """Create mock communication manager."""
        comm_manager = AsyncMock()
        comm_manager.send_message = AsyncMock()
        comm_manager.receive_message = AsyncMock()
        comm_manager.register_participant = AsyncMock()
        return comm_manager

    @pytest.fixture
    def mock_model_serializer(self):
        """Create mock model serializer."""
        serializer = MagicMock()
        serializer.serialize_weights = MagicMock()
        serializer.deserialize_weights = MagicMock()
        return serializer

    @pytest.fixture
    def sample_config(self):
        """Create sample training configuration."""
        return LocalTrainingConfig(epochs=3, batch_size=16, learning_rate=0.01)

    @pytest.fixture
    def participant(
        self, mock_communication_manager, mock_model_serializer, sample_config
    ):
        """Create FederatedParticipant instance with mocks."""
        with patch(
            "vega.federated.participant.CommunicationManager",
            return_value=mock_communication_manager,
        ), patch(
            "vega.federated.participant.ModelSerializer",
            return_value=mock_model_serializer,
        ):
            participant = FederatedParticipant(
                participant_id="test_participant",
                coordinator_endpoint="http://localhost:8000",
                api_key="test_api_key",
                config=sample_config,
            )
            participant.communication_manager = mock_communication_manager
            participant.model_serializer = mock_model_serializer
            return participant

    def test_participant_initialization(self, sample_config):
        """Test participant initialization."""
        with patch("vega.federated.participant.CommunicationManager"), patch(
            "vega.federated.participant.ModelSerializer"
        ):
            participant = FederatedParticipant(
                participant_id="test_participant",
                coordinator_endpoint="http://localhost:8000",
                api_key="test_api_key",
                config=sample_config,
            )

        assert participant.participant_id == "test_participant"
        assert participant.coordinator_endpoint == "http://localhost:8000"
        assert participant.api_key == "test_api_key"
        assert participant.config == sample_config
        assert participant.state == TrainingState.IDLE
        assert participant.current_session_id is None
        assert participant.current_round == 0
        assert isinstance(participant.metrics, ParticipantMetrics)

    @pytest.mark.asyncio
    async def test_register_with_coordinator(
        self, participant, mock_communication_manager
    ):
        """Test participant registration with coordinator."""
        mock_communication_manager.register_participant.return_value = True

        result = await participant.register_with_coordinator()

        assert result is True
        mock_communication_manager.register_participant.assert_called_once_with(
            participant.participant_id, participant.api_key
        )

    @pytest.mark.asyncio
    async def test_register_with_coordinator_failure(
        self, participant, mock_communication_manager
    ):
        """Test participant registration failure."""
        mock_communication_manager.register_participant.return_value = False

        result = await participant.register_with_coordinator()

        assert result is False
        assert participant.metrics.communication_failures == 1

    @pytest.mark.asyncio
    async def test_join_session(self, participant, mock_communication_manager):
        """Test joining a federated learning session."""
        session_info = {
            "session_id": "session_123",
            "model_config": {"layers": [10, 5, 1]},
            "round_number": 1,
        }

        mock_communication_manager.send_message.return_value = True

        result = await participant.join_session("session_123")

        assert result is True
        assert participant.current_session_id == "session_123"
        mock_communication_manager.send_message.assert_called()

    @pytest.mark.asyncio
    async def test_receive_global_model(
        self, participant, mock_communication_manager, mock_model_serializer
    ):
        """Test receiving global model from coordinator."""
        global_weights = ModelWeights({"layer1": [1.0, 2.0, 3.0]})

        mock_communication_manager.receive_message.return_value = {
            "type": "global_model",
            "weights": global_weights.to_dict(),
            "round_number": 2,
        }

        result = await participant.receive_global_model()

        assert result is not None
        assert participant.current_round == 2
        mock_model_serializer.deserialize_weights.assert_called()

    @pytest.mark.asyncio
    async def test_local_training(self, participant):
        """Test local training functionality."""
        # Mock training data
        training_data = [([1.0, 2.0], [0.5]), ([2.0, 3.0], [1.0]), ([3.0, 4.0], [1.5])]

        participant.set_training_data(training_data)

        # Mock the actual training logic
        with patch.object(participant, "_run_local_training") as mock_training:
            mock_training.return_value = {
                "loss": 0.25,
                "accuracy": 0.85,
                "training_time": 10.5,
            }

            result = await participant.train_local_model()

            assert result["loss"] == 0.25
            assert result["accuracy"] == 0.85
            assert participant.state == TrainingState.IDLE
            assert participant.metrics.total_training_time > 0

    @pytest.mark.asyncio
    async def test_send_model_updates(
        self, participant, mock_communication_manager, mock_model_serializer
    ):
        """Test sending model updates to coordinator."""
        local_weights = ModelWeights({"layer1": [1.1, 2.1, 3.1]})

        mock_model_serializer.serialize_weights.return_value = local_weights
        mock_communication_manager.send_message.return_value = True

        with patch(
            "vega.federated.participant.create_model_signature"
        ) as mock_signature:
            mock_signature.return_value = "test_signature"

            result = await participant.send_model_updates(local_weights)

            assert result is True
            assert participant.state == TrainingState.IDLE
            mock_communication_manager.send_message.assert_called()

    @pytest.mark.asyncio
    async def test_security_validation(self, participant):
        """Test security validation during model updates."""
        # Test with anomalous data
        anomalous_weights = ModelWeights(
            {
                "layer1": [float("inf"), 2.0, 3.0],  # Infinite value
                "layer2": [1.0, float("nan"), 3.0],  # NaN value
            }
        )

        with patch(
            "vega.federated.participant.validate_model_update_pipeline"
        ) as mock_validate:
            mock_validate.return_value = {
                "is_valid": False,
                "validation_errors": ["nan_inf_values", "large_values"],
            }

            result = await participant.send_model_updates(anomalous_weights)

            assert result is False
            assert participant.metrics.training_errors > 0

    def test_set_training_data(self, participant):
        """Test setting training data."""
        training_data = [([1.0, 2.0], [0.5]), ([2.0, 3.0], [1.0])]

        participant.set_training_data(training_data)

        assert participant.training_data == training_data
        assert participant.metrics.data_points_count == 2

    def test_set_progress_callback(self, participant):
        """Test setting progress callback."""
        callback = MagicMock()

        participant.set_progress_callback(callback)

        assert participant.progress_callback == callback

    @pytest.mark.asyncio
    async def test_training_with_callback(self, participant):
        """Test training with progress callback."""
        callback = MagicMock()
        participant.set_progress_callback(callback)

        training_data = [([1.0, 2.0], [0.5])]
        participant.set_training_data(training_data)

        with patch.object(participant, "_run_local_training") as mock_training:
            mock_training.return_value = {
                "loss": 0.25,
                "accuracy": 0.85,
                "training_time": 10.5,
            }

            await participant.train_local_model()

            # Verify callback was called
            callback.assert_called()

    @pytest.mark.asyncio
    async def test_error_handling(self, participant, mock_communication_manager):
        """Test error handling during communication failures."""
        mock_communication_manager.send_message.side_effect = Exception("Network error")

        local_weights = ModelWeights({"layer1": [1.0, 2.0, 3.0]})

        result = await participant.send_model_updates(local_weights)

        assert result is False
        assert participant.state == TrainingState.ERROR
        assert participant.metrics.communication_failures > 0

    def test_metrics_tracking(self, participant):
        """Test metrics tracking functionality."""
        initial_metrics = participant.get_metrics()

        # Simulate training completion
        participant.metrics.total_rounds_participated += 1
        participant.metrics.total_training_time += 15.5
        participant.metrics.average_loss = 0.3
        participant.metrics.best_accuracy = 0.9

        updated_metrics = participant.get_metrics()

        assert updated_metrics.total_rounds_participated == 1
        assert updated_metrics.total_training_time == 15.5
        assert updated_metrics.average_loss == 0.3
        assert updated_metrics.best_accuracy == 0.9

    @pytest.mark.asyncio
    async def test_full_training_cycle(
        self, participant, mock_communication_manager, mock_model_serializer
    ):
        """Test complete training cycle from registration to model update."""
        # Setup mocks
        mock_communication_manager.register_participant.return_value = True
        mock_communication_manager.send_message.return_value = True
        mock_communication_manager.receive_message.return_value = {
            "type": "global_model",
            "weights": {"layer1": [1.0, 2.0, 3.0]},
            "round_number": 1,
        }

        training_data = [([1.0, 2.0], [0.5])]
        participant.set_training_data(training_data)

        # Execute full cycle
        registered = await participant.register_with_coordinator()
        joined = await participant.join_session("session_123")
        global_model = await participant.receive_global_model()

        with patch.object(participant, "_run_local_training") as mock_training:
            mock_training.return_value = {
                "loss": 0.25,
                "accuracy": 0.85,
                "training_time": 10.5,
            }

            training_result = await participant.train_local_model()

            local_weights = ModelWeights({"layer1": [1.1, 2.1, 3.1]})
            with patch("vega.federated.participant.create_model_signature"):
                with patch(
                    "vega.federated.participant.validate_model_update_pipeline"
                ) as mock_validate:
                    mock_validate.return_value = {
                        "is_valid": True,
                        "validation_errors": [],
                    }
                    sent = await participant.send_model_updates(local_weights)

        # Verify full cycle completion
        assert registered is True
        assert joined is True
        assert global_model is not None
        assert training_result["loss"] == 0.25
        assert sent is True
        assert participant.metrics.total_rounds_participated > 0


if __name__ == "__main__":
    pytest.main([__file__])
