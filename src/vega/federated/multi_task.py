"""
Multi-Task Federated Learning Implementation

This module implements multi-task federated learning where participants can train
on multiple different tasks simultaneously while sharing knowledge across tasks.
Features shared representations, task-specific heads, and multi-task aggregation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import json
import logging
from pathlib import Path
from copy import deepcopy
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..core.participant import FederatedParticipant
from ..core.fedavg import FedAvg
from ..core.communication import SecureCommunication
from ..core.security import SecurityManager


logger = logging.getLogger(__name__)


@dataclass
class TaskDefinition:
    """Definition of a specific learning task."""

    task_id: str
    task_type: str  # 'classification', 'regression', 'generation', etc.
    input_dim: int
    output_dim: int
    loss_function: str  # 'cross_entropy', 'mse', 'custom'
    metric: str  # 'accuracy', 'f1', 'mse', etc.
    data_schema: Dict[str, Any] = field(default_factory=dict)
    preprocessing: Optional[Dict[str, Any]] = None
    task_weight: float = 1.0  # Weight for this task in multi-task learning


@dataclass
class MultiTaskModelConfig:
    """Configuration for multi-task model architecture."""

    shared_layers: List[int] = field(default_factory=lambda: [256, 128])
    task_specific_layers: Dict[str, List[int]] = field(default_factory=dict)
    activation: str = "relu"
    dropout_rate: float = 0.1
    use_batch_norm: bool = True
    shared_embedding_dim: int = 128


class SharedRepresentationLayer(nn.Module):
    """Shared representation layer for multi-task learning."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        self.layers = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x):
        return self.layers(x)


class TaskSpecificHead(nn.Module):
    """Task-specific head for individual tasks."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64]

        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            # No activation on final layer
            if i < len(dims) - 2:
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())

                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MultiTaskModel(nn.Module):
    """Multi-task neural network with shared representations and task-specific heads."""

    def __init__(self, tasks: List[TaskDefinition], config: MultiTaskModelConfig):
        super().__init__()

        self.tasks = {task.task_id: task for task in tasks}
        self.config = config

        # Determine input dimension (assuming all tasks have same input)
        input_dim = tasks[0].input_dim

        # Shared representation layer
        self.shared_layer = SharedRepresentationLayer(
            input_dim=input_dim,
            hidden_dims=config.shared_layers,
            activation=config.activation,
            dropout_rate=config.dropout_rate,
            use_batch_norm=config.use_batch_norm,
        )

        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        shared_output_dim = self.shared_layer.output_dim

        for task in tasks:
            task_layers = config.task_specific_layers.get(task.task_id, [32])
            self.task_heads[task.task_id] = TaskSpecificHead(
                input_dim=shared_output_dim,
                output_dim=task.output_dim,
                hidden_dims=task_layers,
                activation=config.activation,
                dropout_rate=config.dropout_rate,
            )

    def forward(self, x, task_id: str = None):
        """Forward pass through shared layer and optionally task-specific head."""
        shared_repr = self.shared_layer(x)

        if task_id is None:
            # Return shared representation
            return shared_repr

        if task_id not in self.task_heads:
            raise ValueError(f"Unknown task_id: {task_id}")

        return self.task_heads[task_id](shared_repr)

    def get_task_outputs(self, x):
        """Get outputs for all tasks."""
        shared_repr = self.shared_layer(x)
        return {task_id: head(shared_repr) for task_id, head in self.task_heads.items()}


class MultiTaskLoss:
    """Multi-task loss function with task weighting."""

    def __init__(self, tasks: List[TaskDefinition]):
        self.tasks = {task.task_id: task for task in tasks}
        self.loss_functions = {}

        for task in tasks:
            if task.loss_function == "cross_entropy":
                self.loss_functions[task.task_id] = nn.CrossEntropyLoss()
            elif task.loss_function == "mse":
                self.loss_functions[task.task_id] = nn.MSELoss()
            elif task.loss_function == "bce":
                self.loss_functions[task.task_id] = nn.BCEWithLogitsLoss()
            else:
                raise ValueError(f"Unsupported loss function: {task.loss_function}")

    def compute_loss(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute weighted multi-task loss."""
        task_losses = {}
        total_weight = sum(task.task_weight for task in self.tasks.values())

        total_loss = 0.0
        for task_id, pred in predictions.items():
            if task_id in targets:
                task_loss = self.loss_functions[task_id](pred, targets[task_id])
                task_weight = self.tasks[task_id].task_weight / total_weight
                weighted_loss = task_weight * task_loss

                task_losses[task_id] = task_loss.item()
                total_loss += weighted_loss

        return total_loss, task_losses


class MultiTaskAggregator:
    """Aggregator for multi-task federated learning."""

    def __init__(
        self,
        tasks: List[TaskDefinition],
        shared_weight: float = 0.7,
        task_specific_weight: float = 0.3,
    ):
        self.tasks = {task.task_id: task for task in tasks}
        self.shared_weight = shared_weight
        self.task_specific_weight = task_specific_weight

    def aggregate_models(
        self,
        participant_models: List[Dict[str, torch.Tensor]],
        participant_weights: List[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate multi-task models from participants."""
        if participant_weights is None:
            participant_weights = [1.0] * len(participant_models)

        # Normalize weights
        total_weight = sum(participant_weights)
        participant_weights = [w / total_weight for w in participant_weights]

        aggregated_state = {}

        # Aggregate shared layers
        shared_params = self._extract_shared_parameters(participant_models)
        aggregated_shared = self._weighted_average(shared_params, participant_weights)

        # Aggregate task-specific layers
        for task_id in self.tasks.keys():
            task_params = self._extract_task_parameters(participant_models, task_id)
            if task_params:  # Only aggregate if participants have this task
                aggregated_task = self._weighted_average(
                    task_params, participant_weights
                )
                aggregated_state.update(aggregated_task)

        aggregated_state.update(aggregated_shared)
        return aggregated_state

    def _extract_shared_parameters(
        self, models: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """Extract shared layer parameters from models."""
        shared_params = []
        for model_state in models:
            shared_state = {
                k: v for k, v in model_state.items() if k.startswith("shared_layer.")
            }
            shared_params.append(shared_state)
        return shared_params

    def _extract_task_parameters(
        self, models: List[Dict[str, torch.Tensor]], task_id: str
    ) -> List[Dict[str, torch.Tensor]]:
        """Extract task-specific parameters from models."""
        task_params = []
        task_prefix = f"task_heads.{task_id}."

        for model_state in models:
            task_state = {
                k: v for k, v in model_state.items() if k.startswith(task_prefix)
            }
            if task_state:  # Only include if participant has this task
                task_params.append(task_state)

        return task_params

    def _weighted_average(
        self, param_lists: List[Dict[str, torch.Tensor]], weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """Compute weighted average of parameters."""
        if not param_lists:
            return {}

        averaged_params = {}

        # Get all parameter names
        all_keys = set()
        for params in param_lists:
            all_keys.update(params.keys())

        for key in all_keys:
            # Only average parameters that exist in all models
            tensors = []
            valid_weights = []

            for i, params in enumerate(param_lists):
                if key in params:
                    tensors.append(params[key])
                    valid_weights.append(weights[i])

            if tensors:
                # Normalize weights for this parameter
                total_weight = sum(valid_weights)
                normalized_weights = [w / total_weight for w in valid_weights]

                # Compute weighted average
                weighted_sum = sum(
                    w * tensor for w, tensor in zip(normalized_weights, tensors)
                )
                averaged_params[key] = weighted_sum

        return averaged_params


class MultiTaskParticipant(FederatedParticipant):
    """Federated participant supporting multi-task learning."""

    def __init__(
        self,
        participant_id: str,
        tasks: List[TaskDefinition],
        model_config: MultiTaskModelConfig,
        data_loaders: Dict[str, Any] = None,
    ):
        super().__init__(participant_id)

        self.tasks = {task.task_id: task for task in tasks}
        self.model_config = model_config
        self.data_loaders = data_loaders or {}

        # Initialize multi-task model
        self.model = MultiTaskModel(tasks, model_config)
        self.multi_task_loss = MultiTaskLoss(tasks)

        # Optimizers
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Training state
        self.current_round = 0
        self.training_history = []

    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set model parameters from aggregated weights."""
        self.model.load_state_dict(parameters)

    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current model parameters."""
        return self.model.state_dict()

    def train_local_model(
        self, epochs: int = 1, task_subset: List[str] = None
    ) -> Dict[str, Any]:
        """Train the multi-task model locally."""
        self.model.train()

        if task_subset is None:
            task_subset = list(self.tasks.keys())

        total_loss = 0.0
        task_losses = {task_id: 0.0 for task_id in task_subset}
        num_batches = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_task_losses = {task_id: 0.0 for task_id in task_subset}
            epoch_batches = 0

            # Train on each task
            for task_id in task_subset:
                if task_id not in self.data_loaders:
                    logger.warning(f"No data loader for task {task_id}")
                    continue

                data_loader = self.data_loaders[task_id]

                for batch_data, batch_targets in data_loader:
                    self.optimizer.zero_grad()

                    # Get predictions for all tasks
                    predictions = self.model.get_task_outputs(batch_data)

                    # Create target dictionary for this batch
                    targets = {task_id: batch_targets}

                    # Compute multi-task loss
                    loss, batch_task_losses = self.multi_task_loss.compute_loss(
                        predictions, targets
                    )

                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    for tid, tloss in batch_task_losses.items():
                        epoch_task_losses[tid] += tloss
                    epoch_batches += 1

            if epoch_batches > 0:
                epoch_loss /= epoch_batches
                for task_id in epoch_task_losses:
                    epoch_task_losses[task_id] /= epoch_batches

                total_loss += epoch_loss
                for task_id in task_losses:
                    task_losses[task_id] += epoch_task_losses[task_id]
                num_batches += 1

        # Average over epochs
        if num_batches > 0:
            total_loss /= num_batches
            for task_id in task_losses:
                task_losses[task_id] /= num_batches

        training_result = {
            "participant_id": self.participant_id,
            "round": self.current_round,
            "total_loss": total_loss,
            "task_losses": task_losses,
            "tasks_trained": task_subset,
            "epochs": epochs,
        }

        self.training_history.append(training_result)
        return training_result

    def evaluate_model(self, task_subset: List[str] = None) -> Dict[str, float]:
        """Evaluate the multi-task model."""
        self.model.eval()

        if task_subset is None:
            task_subset = list(self.tasks.keys())

        metrics = {}

        with torch.no_grad():
            for task_id in task_subset:
                if task_id not in self.data_loaders:
                    continue

                task_metrics = self._evaluate_task(task_id)
                metrics[task_id] = task_metrics

        return metrics

    def _evaluate_task(self, task_id: str) -> Dict[str, float]:
        """Evaluate performance on a specific task."""
        data_loader = self.data_loaders[task_id]
        task = self.tasks[task_id]

        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0

        for batch_data, batch_targets in data_loader:
            predictions = self.model(batch_data, task_id)

            # Compute loss
            if task.loss_function == "cross_entropy":
                loss = nn.CrossEntropyLoss()(predictions, batch_targets)
                _, predicted = torch.max(predictions.data, 1)
                correct_predictions += (predicted == batch_targets).sum().item()
            elif task.loss_function == "mse":
                loss = nn.MSELoss()(predictions, batch_targets)
            else:
                loss = torch.tensor(0.0)

            total_loss += loss.item() * batch_data.size(0)
            total_samples += batch_data.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        metrics = {"loss": avg_loss}

        if task.task_type == "classification" and total_samples > 0:
            accuracy = correct_predictions / total_samples
            metrics["accuracy"] = accuracy

        return metrics


class MultiTaskCoordinator:
    """Coordinator for multi-task federated learning sessions."""

    def __init__(
        self,
        tasks: List[TaskDefinition],
        model_config: MultiTaskModelConfig,
        aggregation_strategy: str = "fedavg",
    ):
        self.tasks = {task.task_id: task for task in tasks}
        self.model_config = model_config
        self.aggregation_strategy = aggregation_strategy

        # Initialize aggregator
        self.aggregator = MultiTaskAggregator(tasks)

        # Session state
        self.participants = {}
        self.current_round = 0
        self.global_model = None

        # Training history
        self.training_history = []

        # Security
        self.security_manager = SecurityManager()

    def register_participant(self, participant: MultiTaskParticipant):
        """Register a participant for federated learning."""
        self.participants[participant.participant_id] = participant
        participant.current_round = self.current_round

        logger.info(
            f"Registered participant {participant.participant_id} with tasks: "
            f"{list(participant.tasks.keys())}"
        )

    def initialize_global_model(self):
        """Initialize the global multi-task model."""
        task_list = list(self.tasks.values())
        self.global_model = MultiTaskModel(task_list, self.model_config)

        logger.info(f"Initialized global multi-task model with {len(task_list)} tasks")

    async def run_federated_round(
        self,
        selected_participants: List[str] = None,
        local_epochs: int = 1,
        task_subset: List[str] = None,
    ) -> Dict[str, Any]:
        """Run a single round of multi-task federated learning."""
        if self.global_model is None:
            self.initialize_global_model()

        if selected_participants is None:
            selected_participants = list(self.participants.keys())

        self.current_round += 1

        logger.info(
            f"Starting federated round {self.current_round} with "
            f"{len(selected_participants)} participants"
        )

        # Distribute global model to participants
        global_params = self.global_model.state_dict()
        for participant_id in selected_participants:
            participant = self.participants[participant_id]
            participant.set_model_parameters(global_params)
            participant.current_round = self.current_round

        # Local training
        training_results = []
        participant_models = []
        participant_weights = []

        for participant_id in selected_participants:
            participant = self.participants[participant_id]

            # Train local model
            training_result = participant.train_local_model(
                epochs=local_epochs, task_subset=task_subset
            )
            training_results.append(training_result)

            # Collect model parameters
            local_params = participant.get_model_parameters()
            participant_models.append(local_params)
            participant_weights.append(1.0)  # Equal weighting for now

        # Aggregate models
        aggregated_params = self.aggregator.aggregate_models(
            participant_models, participant_weights
        )

        # Update global model
        self.global_model.load_state_dict(aggregated_params)

        # Record round results
        round_result = {
            "round": self.current_round,
            "participants": selected_participants,
            "tasks_trained": task_subset or list(self.tasks.keys()),
            "training_results": training_results,
            "aggregation_strategy": self.aggregation_strategy,
        }

        self.training_history.append(round_result)

        logger.info(f"Completed federated round {self.current_round}")
        return round_result

    async def evaluate_global_model(
        self, task_subset: List[str] = None
    ) -> Dict[str, Any]:
        """Evaluate the global model across all participants."""
        if self.global_model is None:
            raise ValueError("Global model not initialized")

        if task_subset is None:
            task_subset = list(self.tasks.keys())

        evaluation_results = {}

        # Distribute global model to participants for evaluation
        global_params = self.global_model.state_dict()

        for participant_id, participant in self.participants.items():
            participant.set_model_parameters(global_params)
            participant_metrics = participant.evaluate_model(task_subset)
            evaluation_results[participant_id] = participant_metrics

        # Aggregate evaluation metrics
        aggregated_metrics = self._aggregate_evaluation_metrics(evaluation_results)

        return {
            "round": self.current_round,
            "participant_metrics": evaluation_results,
            "aggregated_metrics": aggregated_metrics,
            "tasks_evaluated": task_subset,
        }

    def _aggregate_evaluation_metrics(
        self, participant_metrics: Dict[str, Dict[str, Dict[str, float]]]
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate evaluation metrics across participants."""
        aggregated = {}

        # Get all tasks and metrics
        all_tasks = set()
        all_metrics = set()

        for participant_results in participant_metrics.values():
            for task_id, task_metrics in participant_results.items():
                all_tasks.add(task_id)
                all_metrics.update(task_metrics.keys())

        # Aggregate by task
        for task_id in all_tasks:
            aggregated[task_id] = {}

            for metric_name in all_metrics:
                values = []
                for participant_results in participant_metrics.values():
                    if (
                        task_id in participant_results
                        and metric_name in participant_results[task_id]
                    ):
                        values.append(participant_results[task_id][metric_name])

                if values:
                    aggregated[task_id][metric_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                    }

        return aggregated

    def save_session_state(self, filepath: str):
        """Save the current session state."""
        state = {
            "tasks": [task.__dict__ for task in self.tasks.values()],
            "model_config": self.model_config.__dict__,
            "current_round": self.current_round,
            "training_history": self.training_history,
            "aggregation_strategy": self.aggregation_strategy,
        }

        if self.global_model is not None:
            state["global_model_state"] = {
                k: v.cpu().numpy().tolist()
                for k, v in self.global_model.state_dict().items()
            }

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved session state to {filepath}")

    def load_session_state(self, filepath: str):
        """Load session state from file."""
        with open(filepath, "r") as f:
            state = json.load(f)

        # Restore tasks
        self.tasks = {}
        for task_data in state["tasks"]:
            task = TaskDefinition(**task_data)
            self.tasks[task.task_id] = task

        # Restore model config
        self.model_config = MultiTaskModelConfig(**state["model_config"])

        # Restore session state
        self.current_round = state["current_round"]
        self.training_history = state["training_history"]
        self.aggregation_strategy = state["aggregation_strategy"]

        # Restore global model if available
        if "global_model_state" in state:
            self.initialize_global_model()
            model_state = {
                k: torch.tensor(v) for k, v in state["global_model_state"].items()
            }
            self.global_model.load_state_dict(model_state)

        logger.info(f"Loaded session state from {filepath}")


def create_synthetic_multitask_data(
    tasks: List[TaskDefinition], num_samples: int = 1000, input_dim: int = 20
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Create synthetic data for multi-task learning testing."""
    datasets = {}

    # Generate shared input features
    X = torch.randn(num_samples, input_dim)

    for task in tasks:
        if task.task_type == "classification":
            # Generate classification targets
            y = torch.randint(0, task.output_dim, (num_samples,))
        elif task.task_type == "regression":
            # Generate regression targets
            y = torch.randn(num_samples, task.output_dim)
        else:
            # Default to random targets
            y = torch.randn(num_samples, task.output_dim)

        datasets[task.task_id] = (X, y)

    return datasets
