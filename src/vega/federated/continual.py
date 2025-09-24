"""
Continual Federated Learning (CFL) with catastrophic forgetting prevention.

Implements Elastic Weight Consolidation (EWC) for federated learning scenarios
where participants learn multiple tasks sequentially without forgetting previous tasks.

Design goals:
- EWC regularization to prevent catastrophic forgetting
- Task-aware federated learning with task identification
- Fisher Information Matrix computation for importance weighting
- Federated aggregation preserving continual learning properties
- Simple synthetic tasks for demonstration and testing

Key concepts:
- Each participant learns tasks T1, T2, T3, ... sequentially
- EWC penalty preserves important weights from previous tasks
- Fisher Information Matrix estimates parameter importance
- Federated aggregation combines EWC-regularized models
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict


@dataclass
class Task:
    """Represents a learning task with input-output mapping."""

    task_id: str
    input_dim: int
    output_dim: int
    data_generator: callable  # function that generates (X, y) pairs

    def generate_batch(
        self, batch_size: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """Generate a batch of training data for this task."""
        return self.data_generator(batch_size)


class LinearModel:
    """Simple linear model for continual learning demonstration."""

    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Initialize with small random weights
        self.W = [
            [random.gauss(0, 0.1) for _ in range(output_dim)] for _ in range(input_dim)
        ]
        self.b = [0.0] * output_dim

    def forward(self, x: List[float]) -> List[float]:
        """Forward pass: y = Wx + b."""
        y = self.b.copy()
        for i in range(self.output_dim):
            for j in range(self.input_dim):
                y[i] += self.W[j][i] * x[j]
        return y

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters as a dictionary."""
        return {"W": [row[:] for row in self.W], "b": self.b[:]}

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters from a dictionary."""
        self.W = [row[:] for row in params["W"]]
        self.b = params["b"][:]

    def mse_loss(self, x: List[float], y_true: List[float]) -> float:
        """Compute mean squared error loss."""
        y_pred = self.forward(x)
        loss = 0.0
        for pred, true in zip(y_pred, y_true):
            diff = pred - true
            # Clamp to prevent overflow
            if abs(diff) > 100.0:
                diff = 100.0 if diff > 0 else -100.0
            loss += diff * diff
        return loss / len(y_true)

    def gradient(self, x: List[float], y_true: List[float]) -> Dict[str, Any]:
        """Compute gradients of MSE loss w.r.t. parameters."""
        y_pred = self.forward(x)
        errors = [pred - true for pred, true in zip(y_pred, y_true)]

        # Gradients w.r.t. weights: dL/dW[j,i] = 2 * error[i] * x[j] / output_dim
        grad_W = [
            [2 * errors[i] * x[j] / self.output_dim for i in range(self.output_dim)]
            for j in range(self.input_dim)
        ]

        # Gradients w.r.t. bias: dL/db[i] = 2 * error[i] / output_dim
        grad_b = [2 * errors[i] / self.output_dim for i in range(self.output_dim)]

        return {"W": grad_W, "b": grad_b}


@dataclass
class EWCConfig:
    """Configuration for Elastic Weight Consolidation."""

    lambda_ewc: float = 1000.0  # EWC regularization strength
    fisher_samples: int = 100  # Number of samples for Fisher computation
    fisher_alpha: float = 0.9  # EMA coefficient for Fisher updates


class EWCParticipant:
    """Federated participant with Elastic Weight Consolidation for continual learning."""

    def __init__(
        self, participant_id: str, input_dim: int, output_dim: int, config: EWCConfig
    ):
        self.participant_id = participant_id
        self.config = config
        self.model = LinearModel(input_dim, output_dim)

        # EWC-specific state
        self.fisher_info: Dict[str, Any] = {
            "W": [[0.0] * output_dim for _ in range(input_dim)],
            "b": [0.0] * output_dim,
        }
        self.optimal_params: Dict[str, Any] = self.model.get_params()
        self.task_history: List[str] = []

    def compute_fisher_information(self, task: Task) -> None:
        """Compute Fisher Information Matrix using current task data."""
        # Generate samples for Fisher computation
        X_batch, y_batch = task.generate_batch(self.config.fisher_samples)

        # Initialize Fisher accumulators
        fisher_W = [[0.0] * self.model.output_dim for _ in range(self.model.input_dim)]
        fisher_b = [0.0] * self.model.output_dim

        # Accumulate gradients squared
        for x, y in zip(X_batch, y_batch):
            grads = self.model.gradient(x, y)

            # Fisher = E[grad^2], so accumulate squared gradients
            for j in range(self.model.input_dim):
                for i in range(self.model.output_dim):
                    fisher_W[j][i] += grads["W"][j][i] ** 2

            for i in range(self.model.output_dim):
                fisher_b[i] += grads["b"][i] ** 2

        # Average and update Fisher with EMA
        alpha = self.config.fisher_alpha
        for j in range(self.model.input_dim):
            for i in range(self.model.output_dim):
                fisher_W[j][i] /= self.config.fisher_samples
                self.fisher_info["W"][j][i] = (
                    alpha * self.fisher_info["W"][j][i] + (1 - alpha) * fisher_W[j][i]
                )

        for i in range(self.model.output_dim):
            fisher_b[i] /= self.config.fisher_samples
            self.fisher_info["b"][i] = (
                alpha * self.fisher_info["b"][i] + (1 - alpha) * fisher_b[i]
            )

    def ewc_penalty(self) -> float:
        """Compute EWC regularization penalty."""
        penalty = 0.0
        current_params = self.model.get_params()

        # Penalty = 0.5 * lambda * sum(Fisher * (param - optimal_param)^2)
        for j in range(self.model.input_dim):
            for i in range(self.model.output_dim):
                diff = current_params["W"][j][i] - self.optimal_params["W"][j][i]
                penalty += self.fisher_info["W"][j][i] * (diff**2)

        for i in range(self.model.output_dim):
            diff = current_params["b"][i] - self.optimal_params["b"][i]
            penalty += self.fisher_info["b"][i] * (diff**2)

        return 0.5 * self.config.lambda_ewc * penalty

    def train_on_task(
        self, task: Task, steps: int = 100, lr: float = 0.01
    ) -> Dict[str, float]:
        """Train on a specific task with EWC regularization."""
        total_loss = 0.0
        ewc_loss = 0.0

        for step in range(steps):
            # Generate training batch
            X_batch, y_batch = task.generate_batch(1)  # Single sample per step
            x, y = X_batch[0], y_batch[0]

            # Compute gradients
            grads = self.model.gradient(x, y)
            data_loss = self.model.mse_loss(x, y)

            # EWC penalty gradients
            ewc_penalty_val = self.ewc_penalty()
            current_params = self.model.get_params()

            # Add EWC gradients to data gradients
            for j in range(self.model.input_dim):
                for i in range(self.model.output_dim):
                    ewc_grad = (
                        self.config.lambda_ewc
                        * self.fisher_info["W"][j][i]
                        * (current_params["W"][j][i] - self.optimal_params["W"][j][i])
                    )
                    grads["W"][j][i] += ewc_grad

            for i in range(self.model.output_dim):
                ewc_grad = (
                    self.config.lambda_ewc
                    * self.fisher_info["b"][i]
                    * (current_params["b"][i] - self.optimal_params["b"][i])
                )
                grads["b"][i] += ewc_grad

            # Gradient descent update with clipping
            for j in range(self.model.input_dim):
                for i in range(self.model.output_dim):
                    grad = grads["W"][j][i]
                    # Clip gradients to prevent instability
                    grad = max(-10.0, min(10.0, grad))
                    self.model.W[j][i] -= lr * grad

            for i in range(self.model.output_dim):
                grad = grads["b"][i]
                grad = max(-10.0, min(10.0, grad))
                self.model.b[i] -= lr * grad

            total_loss += data_loss
            ewc_loss += ewc_penalty_val

        avg_loss = total_loss / steps
        avg_ewc = ewc_loss / steps

        return {
            "data_loss": avg_loss,
            "ewc_penalty": avg_ewc,
            "total_loss": avg_loss + avg_ewc,
        }

    def finish_task(self, task: Task) -> None:
        """Complete learning on a task and update EWC state."""
        # Compute Fisher information for the completed task
        self.compute_fisher_information(task)

        # Update optimal parameters
        self.optimal_params = self.model.get_params()

        # Record task in history
        self.task_history.append(task.task_id)

    def evaluate_on_task(self, task: Task, num_samples: int = 50) -> float:
        """Evaluate model performance on a task."""
        X_batch, y_batch = task.generate_batch(num_samples)
        total_loss = 0.0

        for x, y in zip(X_batch, y_batch):
            total_loss += self.model.mse_loss(x, y)

        return total_loss / num_samples


def federated_continual_aggregate(participants: List[EWCParticipant]) -> Dict[str, Any]:
    """Aggregate EWC participant models using FedAvg on model parameters."""
    if not participants:
        return {}

    # Initialize aggregated parameters
    input_dim = participants[0].model.input_dim
    output_dim = participants[0].model.output_dim

    agg_params = {
        "W": [[0.0] * output_dim for _ in range(input_dim)],
        "b": [0.0] * output_dim,
    }

    # Average parameters across participants
    n = len(participants)
    for participant in participants:
        params = participant.model.get_params()

        for j in range(input_dim):
            for i in range(output_dim):
                agg_params["W"][j][i] += params["W"][j][i] / n

        for i in range(output_dim):
            agg_params["b"][i] += params["b"][i] / n

    return agg_params


def run_continual_federated_learning(
    num_participants: int = 3,
    tasks: Optional[List[Task]] = None,
    steps_per_task: int = 200,
    fed_rounds_per_task: int = 5,
    lr: float = 0.01,
    ewc_lambda: float = 1000.0,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Run continual federated learning across a sequence of tasks."""
    if seed is not None:
        random.seed(seed)

    # Create default tasks if none provided
    if tasks is None:
        tasks = create_synthetic_task_sequence()

    # Initialize participants
    input_dim = tasks[0].input_dim
    output_dim = tasks[0].output_dim
    config = EWCConfig(lambda_ewc=ewc_lambda)

    participants = [
        EWCParticipant(f"participant_{i}", input_dim, output_dim, config)
        for i in range(num_participants)
    ]

    # Track results across tasks
    results = {
        "tasks": [task.task_id for task in tasks],
        "performance_matrix": [],  # [task_idx][participant_idx] = performance
        "federated_history": [],
    }

    # Learn tasks sequentially
    for task_idx, task in enumerate(tasks):
        print(f"Learning task {task.task_id}...")

        # Federated learning rounds for this task
        for fed_round in range(fed_rounds_per_task):
            # Local training with EWC
            round_metrics = []
            for participant in participants:
                metrics = participant.train_on_task(
                    task, steps=steps_per_task // fed_rounds_per_task, lr=lr
                )
                round_metrics.append(metrics)

            # Federated aggregation
            agg_params = federated_continual_aggregate(participants)

            # Update all participants with aggregated parameters
            for participant in participants:
                participant.model.set_params(agg_params)

            # Log round metrics
            avg_data_loss = sum(m["data_loss"] for m in round_metrics) / len(
                round_metrics
            )
            avg_ewc_penalty = sum(m["ewc_penalty"] for m in round_metrics) / len(
                round_metrics
            )

            results["federated_history"].append(
                {
                    "task": task.task_id,
                    "round": fed_round + 1,
                    "avg_data_loss": avg_data_loss,
                    "avg_ewc_penalty": avg_ewc_penalty,
                }
            )

        # Finish task for all participants (compute Fisher, update optimal params)
        for participant in participants:
            participant.finish_task(task)

        # Evaluate on all tasks learned so far (test for catastrophic forgetting)
        task_performance = []
        for eval_task in tasks[: task_idx + 1]:
            participant_scores = []
            for participant in participants:
                score = participant.evaluate_on_task(eval_task)
                participant_scores.append(score)
            task_performance.append(participant_scores)

        results["performance_matrix"].append(task_performance)

    return results


def create_synthetic_task_sequence() -> List[Task]:
    """Create a sequence of synthetic linear regression tasks."""

    def task1_generator(batch_size: int):
        """Task 1: y = 2*x1 + 3*x2 + noise"""
        X, y = [], []
        for _ in range(batch_size):
            x1, x2 = random.gauss(0, 1), random.gauss(0, 1)
            target = 2.0 * x1 + 3.0 * x2 + random.gauss(0, 0.1)
            X.append([x1, x2])
            y.append([target])
        return X, y

    def task2_generator(batch_size: int):
        """Task 2: y = -1*x1 + 4*x2 + noise"""
        X, y = [], []
        for _ in range(batch_size):
            x1, x2 = random.gauss(0, 1), random.gauss(0, 1)
            target = -1.0 * x1 + 4.0 * x2 + random.gauss(0, 0.1)
            X.append([x1, x2])
            y.append([target])
        return X, y

    def task3_generator(batch_size: int):
        """Task 3: y = 5*x1 - 2*x2 + noise"""
        X, y = [], []
        for _ in range(batch_size):
            x1, x2 = random.gauss(0, 1), random.gauss(0, 1)
            target = 5.0 * x1 - 2.0 * x2 + random.gauss(0, 0.1)
            X.append([x1, x2])
            y.append([target])
        return X, y

    return [
        Task("task1", 2, 1, task1_generator),
        Task("task2", 2, 1, task2_generator),
        Task("task3", 2, 1, task3_generator),
    ]


def demo() -> None:
    """Run a continual federated learning demo."""
    print("Continual Federated Learning Demo")
    print("=" * 50)

    results = run_continual_federated_learning(
        num_participants=3,
        steps_per_task=150,
        fed_rounds_per_task=3,
        lr=0.02,
        ewc_lambda=500.0,
        seed=123,
    )

    print(f"Tasks learned: {results['tasks']}")
    print("\nFinal performance matrix (lower = better):")
    print("Rows = learning progression, Cols = task evaluation")

    # Display final performance matrix
    final_matrix = results["performance_matrix"][-1]  # After learning all tasks
    for task_idx, task_name in enumerate(results["tasks"]):
        avg_score = sum(final_matrix[task_idx]) / len(final_matrix[task_idx])
        print(f"{task_name}: avg loss = {avg_score:.3f}")

    print(f"\nTotal federated rounds: {len(results['federated_history'])}")


if __name__ == "__main__":
    demo()
