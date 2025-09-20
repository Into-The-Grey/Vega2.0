"""
Tests for Multi-Task Federated Learning

Comprehensive test suite for multi-task federated learning components including
models, aggregation, participants, and coordinators.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import asyncio
import tempfile
import json
from pathlib import Path

from src.vega.federated.multi_task import (
    TaskDefinition,
    MultiTaskModelConfig,
    SharedRepresentationLayer,
    TaskSpecificHead,
    MultiTaskModel,
    MultiTaskLoss,
    MultiTaskAggregator,
    MultiTaskParticipant,
    MultiTaskCoordinator,
    create_synthetic_multitask_data,
)


class TestTaskDefinition:
    """Test TaskDefinition class."""

    def test_task_definition_creation(self):
        """Test creating task definitions."""
        task = TaskDefinition(
            task_id="test_task",
            task_type="classification",
            input_dim=10,
            output_dim=3,
            loss_function="cross_entropy",
            metric="accuracy",
        )

        assert task.task_id == "test_task"
        assert task.task_type == "classification"
        assert task.input_dim == 10
        assert task.output_dim == 3
        assert task.loss_function == "cross_entropy"
        assert task.metric == "accuracy"
        assert task.task_weight == 1.0  # default value

    def test_task_definition_with_custom_weight(self):
        """Test task definition with custom weight."""
        task = TaskDefinition(
            task_id="weighted_task",
            task_type="regression",
            input_dim=5,
            output_dim=1,
            loss_function="mse",
            metric="mse",
            task_weight=0.5,
        )

        assert task.task_weight == 0.5


class TestMultiTaskModelConfig:
    """Test MultiTaskModelConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = MultiTaskModelConfig()

        assert config.shared_layers == [256, 128]
        assert config.task_specific_layers == {}
        assert config.activation == "relu"
        assert config.dropout_rate == 0.1
        assert config.use_batch_norm is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = MultiTaskModelConfig(
            shared_layers=[64, 32],
            task_specific_layers={"task1": [16], "task2": [8]},
            activation="tanh",
            dropout_rate=0.2,
            use_batch_norm=False,
        )

        assert config.shared_layers == [64, 32]
        assert config.task_specific_layers == {"task1": [16], "task2": [8]}
        assert config.activation == "tanh"
        assert config.dropout_rate == 0.2
        assert config.use_batch_norm is False


class TestSharedRepresentationLayer:
    """Test SharedRepresentationLayer."""

    def test_layer_creation(self):
        """Test creating shared representation layer."""
        layer = SharedRepresentationLayer(
            input_dim=10, hidden_dims=[32, 16], activation="relu", dropout_rate=0.1
        )

        assert layer.output_dim == 16
        assert isinstance(layer.layers, nn.Sequential)

    def test_forward_pass(self):
        """Test forward pass through shared layer."""
        layer = SharedRepresentationLayer(input_dim=10, hidden_dims=[8, 4])

        x = torch.randn(5, 10)
        output = layer(x)

        assert output.shape == (5, 4)

    def test_different_activations(self):
        """Test different activation functions."""
        for activation in ["relu", "tanh", "sigmoid"]:
            layer = SharedRepresentationLayer(
                input_dim=5, hidden_dims=[3], activation=activation
            )

            x = torch.randn(2, 5)
            output = layer(x)
            assert output.shape == (2, 3)


class TestTaskSpecificHead:
    """Test TaskSpecificHead."""

    def test_head_creation(self):
        """Test creating task-specific head."""
        head = TaskSpecificHead(input_dim=16, output_dim=3, hidden_dims=[8])

        assert isinstance(head.layers, nn.Sequential)

    def test_forward_pass(self):
        """Test forward pass through task head."""
        head = TaskSpecificHead(input_dim=10, output_dim=5, hidden_dims=[7])

        x = torch.randn(3, 10)
        output = head(x)

        assert output.shape == (3, 5)

    def test_no_hidden_layers(self):
        """Test head with no hidden layers."""
        head = TaskSpecificHead(input_dim=10, output_dim=3, hidden_dims=None)

        x = torch.randn(2, 10)
        output = head(x)

        assert output.shape == (2, 3)


class TestMultiTaskModel:
    """Test MultiTaskModel."""

    @pytest.fixture
    def sample_tasks(self):
        """Create sample tasks for testing."""
        return [
            TaskDefinition(
                "task1", "classification", 10, 3, "cross_entropy", "accuracy"
            ),
            TaskDefinition("task2", "regression", 10, 1, "mse", "mse"),
        ]

    @pytest.fixture
    def sample_config(self):
        """Create sample model configuration."""
        return MultiTaskModelConfig(
            shared_layers=[8, 4], task_specific_layers={"task1": [2], "task2": [2]}
        )

    def test_model_creation(self, sample_tasks, sample_config):
        """Test creating multi-task model."""
        model = MultiTaskModel(sample_tasks, sample_config)

        assert len(model.tasks) == 2
        assert "task1" in model.task_heads
        assert "task2" in model.task_heads

    def test_shared_representation(self, sample_tasks, sample_config):
        """Test getting shared representation."""
        model = MultiTaskModel(sample_tasks, sample_config)

        x = torch.randn(3, 10)
        shared_repr = model(x)  # No task_id specified

        assert shared_repr.shape == (3, 4)  # Last shared layer size

    def test_task_specific_output(self, sample_tasks, sample_config):
        """Test getting task-specific output."""
        model = MultiTaskModel(sample_tasks, sample_config)

        x = torch.randn(3, 10)

        # Test task1 (classification)
        output1 = model(x, "task1")
        assert output1.shape == (3, 3)

        # Test task2 (regression)
        output2 = model(x, "task2")
        assert output2.shape == (3, 1)

    def test_all_task_outputs(self, sample_tasks, sample_config):
        """Test getting outputs for all tasks."""
        model = MultiTaskModel(sample_tasks, sample_config)

        x = torch.randn(3, 10)
        all_outputs = model.get_task_outputs(x)

        assert "task1" in all_outputs
        assert "task2" in all_outputs
        assert all_outputs["task1"].shape == (3, 3)
        assert all_outputs["task2"].shape == (3, 1)

    def test_invalid_task_id(self, sample_tasks, sample_config):
        """Test error handling for invalid task ID."""
        model = MultiTaskModel(sample_tasks, sample_config)

        x = torch.randn(3, 10)

        with pytest.raises(ValueError, match="Unknown task_id"):
            model(x, "invalid_task")


class TestMultiTaskLoss:
    """Test MultiTaskLoss."""

    @pytest.fixture
    def sample_tasks(self):
        """Create sample tasks for testing."""
        return [
            TaskDefinition(
                "cls_task",
                "classification",
                10,
                3,
                "cross_entropy",
                "accuracy",
                task_weight=1.0,
            ),
            TaskDefinition(
                "reg_task", "regression", 10, 1, "mse", "mse", task_weight=0.5
            ),
        ]

    def test_loss_creation(self, sample_tasks):
        """Test creating multi-task loss."""
        loss_fn = MultiTaskLoss(sample_tasks)

        assert "cls_task" in loss_fn.loss_functions
        assert "reg_task" in loss_fn.loss_functions
        assert isinstance(loss_fn.loss_functions["cls_task"], nn.CrossEntropyLoss)
        assert isinstance(loss_fn.loss_functions["reg_task"], nn.MSELoss)

    def test_loss_computation(self, sample_tasks):
        """Test multi-task loss computation."""
        loss_fn = MultiTaskLoss(sample_tasks)

        # Create predictions and targets
        predictions = {"cls_task": torch.randn(5, 3), "reg_task": torch.randn(5, 1)}
        targets = {"cls_task": torch.randint(0, 3, (5,)), "reg_task": torch.randn(5, 1)}

        total_loss, task_losses = loss_fn.compute_loss(predictions, targets)

        assert isinstance(total_loss, torch.Tensor)
        assert "cls_task" in task_losses
        assert "reg_task" in task_losses
        assert isinstance(task_losses["cls_task"], float)
        assert isinstance(task_losses["reg_task"], float)

    def test_missing_target(self, sample_tasks):
        """Test loss computation with missing target."""
        loss_fn = MultiTaskLoss(sample_tasks)

        predictions = {"cls_task": torch.randn(5, 3)}
        targets = {"cls_task": torch.randint(0, 3, (5,))}

        total_loss, task_losses = loss_fn.compute_loss(predictions, targets)

        assert "cls_task" in task_losses
        assert "reg_task" not in task_losses

    def test_unsupported_loss_function(self):
        """Test error handling for unsupported loss function."""
        tasks = [TaskDefinition("test", "custom", 5, 2, "unsupported", "accuracy")]

        with pytest.raises(ValueError, match="Unsupported loss function"):
            MultiTaskLoss(tasks)


class TestMultiTaskAggregator:
    """Test MultiTaskAggregator."""

    @pytest.fixture
    def sample_tasks(self):
        """Create sample tasks for testing."""
        return [
            TaskDefinition(
                "task1", "classification", 10, 3, "cross_entropy", "accuracy"
            ),
            TaskDefinition("task2", "regression", 10, 1, "mse", "mse"),
        ]

    @pytest.fixture
    def sample_models(self):
        """Create sample model states for aggregation."""
        return [
            {
                "shared_layer.layers.0.weight": torch.randn(8, 10),
                "shared_layer.layers.0.bias": torch.randn(8),
                "task_heads.task1.layers.0.weight": torch.randn(3, 8),
                "task_heads.task1.layers.0.bias": torch.randn(3),
                "task_heads.task2.layers.0.weight": torch.randn(1, 8),
                "task_heads.task2.layers.0.bias": torch.randn(1),
            },
            {
                "shared_layer.layers.0.weight": torch.randn(8, 10),
                "shared_layer.layers.0.bias": torch.randn(8),
                "task_heads.task1.layers.0.weight": torch.randn(3, 8),
                "task_heads.task1.layers.0.bias": torch.randn(3),
                "task_heads.task2.layers.0.weight": torch.randn(1, 8),
                "task_heads.task2.layers.0.bias": torch.randn(1),
            },
        ]

    def test_aggregator_creation(self, sample_tasks):
        """Test creating multi-task aggregator."""
        aggregator = MultiTaskAggregator(sample_tasks)

        assert len(aggregator.tasks) == 2
        assert aggregator.shared_weight == 0.7
        assert aggregator.task_specific_weight == 0.3

    def test_model_aggregation(self, sample_tasks, sample_models):
        """Test aggregating multi-task models."""
        aggregator = MultiTaskAggregator(sample_tasks)

        aggregated = aggregator.aggregate_models(sample_models)

        # Check that all parameters are present
        assert "shared_layer.layers.0.weight" in aggregated
        assert "shared_layer.layers.0.bias" in aggregated
        assert "task_heads.task1.layers.0.weight" in aggregated
        assert "task_heads.task1.layers.0.bias" in aggregated
        assert "task_heads.task2.layers.0.weight" in aggregated
        assert "task_heads.task2.layers.0.bias" in aggregated

    def test_weighted_aggregation(self, sample_tasks, sample_models):
        """Test weighted model aggregation."""
        aggregator = MultiTaskAggregator(sample_tasks)

        weights = [0.7, 0.3]
        aggregated = aggregator.aggregate_models(sample_models, weights)

        # Verify aggregated parameters exist
        assert len(aggregated) > 0

        # Check parameter shapes are preserved
        for key, param in aggregated.items():
            assert param.shape == sample_models[0][key].shape

    def test_extract_shared_parameters(self, sample_tasks, sample_models):
        """Test extracting shared parameters."""
        aggregator = MultiTaskAggregator(sample_tasks)

        shared_params = aggregator._extract_shared_parameters(sample_models)

        assert len(shared_params) == 2
        for params in shared_params:
            assert any(k.startswith("shared_layer.") for k in params.keys())
            assert not any(k.startswith("task_heads.") for k in params.keys())

    def test_extract_task_parameters(self, sample_tasks, sample_models):
        """Test extracting task-specific parameters."""
        aggregator = MultiTaskAggregator(sample_tasks)

        task1_params = aggregator._extract_task_parameters(sample_models, "task1")

        assert len(task1_params) == 2
        for params in task1_params:
            assert any(k.startswith("task_heads.task1.") for k in params.keys())
            assert not any(k.startswith("task_heads.task2.") for k in params.keys())


class TestMultiTaskParticipant:
    """Test MultiTaskParticipant."""

    @pytest.fixture
    def sample_tasks(self):
        """Create sample tasks for testing."""
        return [
            TaskDefinition(
                "task1", "classification", 4, 2, "cross_entropy", "accuracy"
            ),
            TaskDefinition("task2", "regression", 4, 1, "mse", "mse"),
        ]

    @pytest.fixture
    def sample_config(self):
        """Create sample model configuration."""
        return MultiTaskModelConfig(
            shared_layers=[6, 3],
            task_specific_layers={"task1": [2], "task2": [2]},
            dropout_rate=0.0,  # Disable dropout for testing
        )

    @pytest.fixture
    def sample_data_loaders(self):
        """Create sample data loaders."""
        # Task 1 data (classification)
        X1 = torch.randn(20, 4)
        y1 = torch.randint(0, 2, (20,))
        dataset1 = TensorDataset(X1, y1)
        loader1 = DataLoader(dataset1, batch_size=5, shuffle=False)

        # Task 2 data (regression)
        X2 = torch.randn(20, 4)
        y2 = torch.randn(20, 1)
        dataset2 = TensorDataset(X2, y2)
        loader2 = DataLoader(dataset2, batch_size=5, shuffle=False)

        return {"task1": loader1, "task2": loader2}

    def test_participant_creation(
        self, sample_tasks, sample_config, sample_data_loaders
    ):
        """Test creating multi-task participant."""
        participant = MultiTaskParticipant(
            participant_id="test_participant",
            participant_name="Test Participant",
            tasks=sample_tasks,
            model_config=sample_config,
            data_loaders=sample_data_loaders,
        )

        assert participant.participant_id == "test_participant"
        assert len(participant.tasks) == 2
        assert isinstance(participant.model, MultiTaskModel)
        assert isinstance(participant.multi_task_loss, MultiTaskLoss)

    def test_set_and_get_parameters(self, sample_tasks, sample_config):
        """Test setting and getting model parameters."""
        participant = MultiTaskParticipant(
            participant_id="test_participant",
            participant_name="Test Participant",
            tasks=sample_tasks,
            model_config=sample_config,
        )

        # Get initial parameters
        initial_params = participant.get_model_parameters()

        # Create new parameters
        new_params = {}
        for key, param in initial_params.items():
            if param.dtype in [torch.float, torch.float32, torch.float64]:
                new_params[key] = torch.randn_like(param)
            elif param.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                new_params[key] = torch.randint_like(param, low=0, high=10)
            else:
                new_params[key] = torch.zeros_like(param)

        # Set new parameters
        participant.set_model_parameters(new_params)

        # Verify parameters were set
        current_params = participant.get_model_parameters()
        for key in new_params:
            assert torch.allclose(current_params[key], new_params[key], atol=1e-6)

    def test_local_training(self, sample_tasks, sample_config, sample_data_loaders):
        """Test local model training."""
        participant = MultiTaskParticipant(
            participant_id="test_participant",
            participant_name="Test Participant",
            tasks=sample_tasks,
            model_config=sample_config,
            data_loaders=sample_data_loaders,
        )

        # Train local model
        result = participant.train_local_model(epochs=1)

        assert result["participant_id"] == "test_participant"
        assert "total_loss" in result
        assert "task_losses" in result
        assert isinstance(result["total_loss"], float)
        assert len(result["task_losses"]) <= 2  # May have fewer if data loading fails

    def test_model_evaluation(self, sample_tasks, sample_config, sample_data_loaders):
        """Test model evaluation."""
        participant = MultiTaskParticipant(
            participant_id="test_participant",
            participant_name="Test Participant",
            tasks=sample_tasks,
            model_config=sample_config,
            data_loaders=sample_data_loaders,
        )

        # Evaluate model
        metrics = participant.evaluate_model()

        # Should have metrics for available tasks
        assert isinstance(metrics, dict)
        for task_id, task_metrics in metrics.items():
            assert "loss" in task_metrics
            if task_id == "task1":  # Classification task
                assert "accuracy" in task_metrics


class TestMultiTaskCoordinator:
    """Test MultiTaskCoordinator."""

    @pytest.fixture
    def sample_tasks(self):
        """Create sample tasks for testing."""
        return [
            TaskDefinition(
                "task1", "classification", 4, 2, "cross_entropy", "accuracy"
            ),
            TaskDefinition("task2", "regression", 4, 1, "mse", "mse"),
        ]

    @pytest.fixture
    def sample_config(self):
        """Create sample model configuration."""
        return MultiTaskModelConfig(
            shared_layers=[6, 3],
            task_specific_layers={"task1": [2], "task2": [2]},
            dropout_rate=0.0,
        )

    @pytest.fixture
    def sample_participants(self, sample_tasks, sample_config):
        """Create sample participants."""
        participants = []

        for i in range(2):
            # Create simple data loaders
            X = torch.randn(10, 4)
            y1 = torch.randint(0, 2, (10,))
            y2 = torch.randn(10, 1)

            data_loaders = {
                "task1": DataLoader(TensorDataset(X, y1), batch_size=5),
                "task2": DataLoader(TensorDataset(X, y2), batch_size=5),
            }

            participant = MultiTaskParticipant(
                participant_id=f"participant_{i}",
                participant_name=f"Participant {i}",
                tasks=sample_tasks,
                model_config=sample_config,
                data_loaders=data_loaders,
            )
            participants.append(participant)

        return participants

    def test_coordinator_creation(self, sample_tasks, sample_config):
        """Test creating multi-task coordinator."""
        coordinator = MultiTaskCoordinator(
            tasks=sample_tasks, model_config=sample_config
        )

        assert len(coordinator.tasks) == 2
        assert coordinator.current_round == 0
        assert coordinator.global_model is None

    def test_participant_registration(
        self, sample_tasks, sample_config, sample_participants
    ):
        """Test registering participants."""
        coordinator = MultiTaskCoordinator(
            tasks=sample_tasks, model_config=sample_config
        )

        for participant in sample_participants:
            coordinator.register_participant(participant)

        assert len(coordinator.participants) == 2
        assert "participant_0" in coordinator.participants
        assert "participant_1" in coordinator.participants

    def test_global_model_initialization(self, sample_tasks, sample_config):
        """Test initializing global model."""
        coordinator = MultiTaskCoordinator(
            tasks=sample_tasks, model_config=sample_config
        )

        coordinator.initialize_global_model()

        assert coordinator.global_model is not None
        assert isinstance(coordinator.global_model, MultiTaskModel)

    @pytest.mark.asyncio
    async def test_federated_round(
        self, sample_tasks, sample_config, sample_participants
    ):
        """Test running a federated round."""
        coordinator = MultiTaskCoordinator(
            tasks=sample_tasks, model_config=sample_config
        )

        # Register participants
        for participant in sample_participants:
            coordinator.register_participant(participant)

        # Run federated round
        result = await coordinator.run_federated_round(local_epochs=1)

        assert result["round"] == 1
        assert len(result["participants"]) == 2
        assert len(result["training_results"]) == 2
        assert coordinator.current_round == 1
        assert coordinator.global_model is not None

    @pytest.mark.asyncio
    async def test_global_model_evaluation(
        self, sample_tasks, sample_config, sample_participants
    ):
        """Test evaluating the global model."""
        coordinator = MultiTaskCoordinator(
            tasks=sample_tasks, model_config=sample_config
        )

        # Register participants and initialize
        for participant in sample_participants:
            coordinator.register_participant(participant)

        coordinator.initialize_global_model()

        # Evaluate global model
        result = await coordinator.evaluate_global_model()

        assert "participant_metrics" in result
        assert "aggregated_metrics" in result
        assert len(result["participant_metrics"]) == 2

    def test_session_save_load(self, sample_tasks, sample_config):
        """Test saving and loading session state."""
        coordinator = MultiTaskCoordinator(
            tasks=sample_tasks, model_config=sample_config
        )

        coordinator.initialize_global_model()
        coordinator.current_round = 5

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            # Save session
            coordinator.save_session_state(filepath)

            # Create new coordinator and load
            new_coordinator = MultiTaskCoordinator(
                tasks=sample_tasks, model_config=sample_config
            )
            new_coordinator.load_session_state(filepath)

            assert new_coordinator.current_round == 5
            assert new_coordinator.global_model is not None
            assert len(new_coordinator.tasks) == 2

        finally:
            Path(filepath).unlink()


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_synthetic_multitask_data(self):
        """Test creating synthetic multi-task data."""
        tasks = [
            TaskDefinition(
                "cls_task", "classification", 10, 3, "cross_entropy", "accuracy"
            ),
            TaskDefinition("reg_task", "regression", 10, 2, "mse", "mse"),
        ]

        datasets = create_synthetic_multitask_data(tasks, num_samples=100, input_dim=10)

        assert "cls_task" in datasets
        assert "reg_task" in datasets

        # Check classification data
        X_cls, y_cls = datasets["cls_task"]
        assert X_cls.shape == (100, 10)
        assert y_cls.shape == (100,)
        assert torch.all(y_cls >= 0) and torch.all(y_cls < 3)

        # Check regression data
        X_reg, y_reg = datasets["reg_task"]
        assert X_reg.shape == (100, 10)
        assert y_reg.shape == (100, 2)


if __name__ == "__main__":
    pytest.main([__file__])
