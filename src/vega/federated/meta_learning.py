"""
Federated Meta-Learning implementation using Model-Agnostic Meta-Learning (MAML).

Implements federated meta-learning where participants collaboratively learn a meta-model
that can quickly adapt to new tasks. Uses MAML algorithm with gradient-based optimization
for fast adaptation across heterogeneous federated tasks.

Design goals:
- Model-agnostic meta-learning across federated participants
- Fast adaptation to new tasks with few gradient steps
- Task distribution and sampling across participants
- Inner-loop task-specific adaptation
- Outer-loop meta-parameter updates via federated aggregation

Key concepts:
- Meta-learner maintains global meta-parameters θ
- Each participant samples tasks and performs inner-loop adaptation
- Inner loop: task-specific adaptation with gradient descent
- Outer loop: meta-parameter updates using adapted gradients
- Federated aggregation of meta-gradients across participants
- Quick adaptation to unseen tasks with minimal data
"""

from __future__ import annotations

import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable
from collections import defaultdict
import copy


@dataclass
class Task:
    """Represents a learning task for meta-learning."""

    task_id: str
    input_dim: int
    output_dim: int
    data: List[Tuple[List[float], List[float]]]  # (input, target) pairs
    task_type: str = "regression"  # "regression" or "classification"

    def sample_support_query(
        self, k_shot: int = 5, q_query: int = 10
    ) -> Tuple[
        List[Tuple[List[float], List[float]]], List[Tuple[List[float], List[float]]]
    ]:
        """Sample support and query sets for few-shot learning."""
        if len(self.data) < k_shot + q_query:
            # Use all data if not enough samples
            mid = len(self.data) // 2
            support = self.data[:mid] if mid > 0 else self.data[:1]
            query = self.data[mid:] if mid < len(self.data) else self.data[-1:]
        else:
            shuffled = self.data.copy()
            random.shuffle(shuffled)
            support = shuffled[:k_shot]
            query = shuffled[k_shot : k_shot + q_query]

        return support, query


@dataclass
class MAMLConfig:
    """Configuration for Model-Agnostic Meta-Learning."""

    inner_lr: float = 0.01  # Learning rate for inner loop adaptation
    outer_lr: float = 0.001  # Learning rate for outer loop meta-updates
    inner_steps: int = 5  # Number of inner loop gradient steps
    k_shot: int = 5  # Number of support examples per task
    q_query: int = 10  # Number of query examples per task
    task_batch_size: int = 4  # Number of tasks per meta-update
    second_order: bool = True  # Use second-order gradients (more accurate but slower)
    clip_grad_norm: float = 1.0  # Gradient clipping threshold


class SimpleMetaModel:
    """Simple neural network for meta-learning demonstrations."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize parameters with small random values
        self.params = self._init_params()

    def _init_params(self) -> Dict[str, Any]:
        """Initialize model parameters."""
        # Xavier/Glorot initialization
        w1_bound = math.sqrt(6.0 / (self.input_dim + self.hidden_dim))
        w2_bound = math.sqrt(6.0 / (self.hidden_dim + self.output_dim))

        return {
            "w1": [
                [random.uniform(-w1_bound, w1_bound) for _ in range(self.hidden_dim)]
                for _ in range(self.input_dim)
            ],
            "b1": [0.0] * self.hidden_dim,
            "w2": [
                [random.uniform(-w2_bound, w2_bound) for _ in range(self.output_dim)]
                for _ in range(self.hidden_dim)
            ],
            "b2": [0.0] * self.output_dim,
        }

    def forward(
        self, x: List[float], params: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """Forward pass through the network."""
        if params is None:
            params = self.params

        # Layer 1: Linear + ReLU
        h = [0.0] * self.hidden_dim
        for i in range(self.hidden_dim):
            for j in range(self.input_dim):
                h[i] += params["w1"][j][i] * x[j]
            h[i] += params["b1"][i]
            h[i] = max(0, h[i])  # ReLU activation

        # Layer 2: Linear output
        y = [0.0] * self.output_dim
        for i in range(self.output_dim):
            for j in range(self.hidden_dim):
                y[i] += params["w2"][j][i] * h[j]
            y[i] += params["b2"][i]

        return y

    def loss(
        self,
        x: List[float],
        target: List[float],
        params: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute mean squared error loss."""
        prediction = self.forward(x, params)
        mse = sum((pred - tgt) ** 2 for pred, tgt in zip(prediction, target)) / len(
            target
        )
        return mse

    def compute_gradients(
        self,
        x: List[float],
        target: List[float],
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute gradients via numerical differentiation."""
        if params is None:
            params = self.params

        gradients = {}
        eps = 1e-5

        for param_name, param_values in params.items():
            if param_name in ["w1", "w2"]:
                # 2D weight matrices
                gradients[param_name] = []
                for i in range(len(param_values)):
                    grad_row = []
                    for j in range(len(param_values[i])):
                        # Finite difference
                        param_values[i][j] += eps
                        loss_plus = self.loss(x, target, params)
                        param_values[i][j] -= 2 * eps
                        loss_minus = self.loss(x, target, params)
                        param_values[i][j] += eps  # Restore original value

                        grad = (loss_plus - loss_minus) / (2 * eps)
                        grad_row.append(grad)
                    gradients[param_name].append(grad_row)
            else:
                # 1D bias vectors
                gradients[param_name] = []
                for i in range(len(param_values)):
                    param_values[i] += eps
                    loss_plus = self.loss(x, target, params)
                    param_values[i] -= 2 * eps
                    loss_minus = self.loss(x, target, params)
                    param_values[i] += eps  # Restore original value

                    grad = (loss_plus - loss_minus) / (2 * eps)
                    gradients[param_name].append(grad)

        return gradients

    def apply_gradients(
        self,
        gradients: Dict[str, Any],
        learning_rate: float,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Apply gradients to parameters and return updated parameters."""
        if params is None:
            params = copy.deepcopy(self.params)
        else:
            params = copy.deepcopy(params)

        for param_name, grad_values in gradients.items():
            if param_name in ["w1", "w2"]:
                # 2D weight matrices
                for i in range(len(grad_values)):
                    for j in range(len(grad_values[i])):
                        params[param_name][i][j] -= learning_rate * grad_values[i][j]
            else:
                # 1D bias vectors
                for i in range(len(grad_values)):
                    params[param_name][i] -= learning_rate * grad_values[i]

        return params

    def get_params(self) -> Dict[str, Any]:
        """Get copy of current parameters."""
        return copy.deepcopy(self.params)

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters."""
        self.params = copy.deepcopy(params)


class FederatedMAML:
    """Federated Model-Agnostic Meta-Learning implementation."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, config: MAMLConfig
    ):
        self.config = config
        self.meta_model = SimpleMetaModel(input_dim, hidden_dim, output_dim)
        self.participants: List[str] = []
        self.task_distributions: Dict[str, List[Task]] = {}

        # Training metrics
        self.meta_train_losses: List[float] = []
        self.adaptation_histories: List[Dict[str, Any]] = []

    def register_participant(self, participant_id: str, tasks: List[Task]) -> None:
        """Register a participant with their task distribution."""
        self.participants.append(participant_id)
        self.task_distributions[participant_id] = tasks

    def inner_loop_adaptation(
        self, task: Task, meta_params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[float]]:
        """Perform inner loop task-specific adaptation."""
        support_set, _ = task.sample_support_query(
            self.config.k_shot, self.config.q_query
        )

        adapted_params = copy.deepcopy(meta_params)
        inner_losses = []

        # Inner loop gradient descent
        for step in range(self.config.inner_steps):
            # Compute gradients on support set
            total_gradients = None
            total_loss = 0.0

            for x, target in support_set:
                loss = self.meta_model.loss(x, target, adapted_params)
                total_loss += loss

                gradients = self.meta_model.compute_gradients(x, target, adapted_params)

                if total_gradients is None:
                    total_gradients = gradients
                else:
                    # Accumulate gradients
                    for param_name in gradients:
                        if param_name in ["w1", "w2"]:
                            for i in range(len(gradients[param_name])):
                                for j in range(len(gradients[param_name][i])):
                                    total_gradients[param_name][i][j] += gradients[
                                        param_name
                                    ][i][j]
                        else:
                            for i in range(len(gradients[param_name])):
                                total_gradients[param_name][i] += gradients[param_name][
                                    i
                                ]

            # Average gradients
            num_samples = len(support_set)
            if num_samples > 0:
                for param_name in total_gradients:
                    if param_name in ["w1", "w2"]:
                        for i in range(len(total_gradients[param_name])):
                            for j in range(len(total_gradients[param_name][i])):
                                total_gradients[param_name][i][j] /= num_samples
                    else:
                        for i in range(len(total_gradients[param_name])):
                            total_gradients[param_name][i] /= num_samples

            inner_losses.append(total_loss / num_samples)

            # Update adapted parameters
            adapted_params = self.meta_model.apply_gradients(
                total_gradients, self.config.inner_lr, adapted_params
            )

        return adapted_params, inner_losses

    def compute_meta_gradients(
        self, participant_id: str, meta_params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float]:
        """Compute meta-gradients for a participant."""
        participant_tasks = self.task_distributions[participant_id]

        # Sample tasks for this meta-update
        sampled_tasks = random.sample(
            participant_tasks, min(self.config.task_batch_size, len(participant_tasks))
        )

        meta_gradients = None
        total_meta_loss = 0.0
        num_tasks = 0

        for task in sampled_tasks:
            # Inner loop adaptation
            adapted_params, _ = self.inner_loop_adaptation(task, meta_params)

            # Compute meta-gradients on query set
            _, query_set = task.sample_support_query(
                self.config.k_shot, self.config.q_query
            )

            for x, target in query_set:
                meta_loss = self.meta_model.loss(x, target, adapted_params)
                total_meta_loss += meta_loss

                # Compute gradients w.r.t. meta-parameters
                task_meta_gradients = self.meta_model.compute_gradients(
                    x, target, adapted_params
                )

                if meta_gradients is None:
                    meta_gradients = task_meta_gradients
                else:
                    # Accumulate meta-gradients
                    for param_name in task_meta_gradients:
                        if param_name in ["w1", "w2"]:
                            for i in range(len(task_meta_gradients[param_name])):
                                for j in range(len(task_meta_gradients[param_name][i])):
                                    meta_gradients[param_name][i][
                                        j
                                    ] += task_meta_gradients[param_name][i][j]
                        else:
                            for i in range(len(task_meta_gradients[param_name])):
                                meta_gradients[param_name][i] += task_meta_gradients[
                                    param_name
                                ][i]
                num_tasks += 1

        # Average meta-gradients
        if num_tasks > 0 and meta_gradients is not None:
            for param_name in meta_gradients:
                if param_name in ["w1", "w2"]:
                    for i in range(len(meta_gradients[param_name])):
                        for j in range(len(meta_gradients[param_name][i])):
                            meta_gradients[param_name][i][j] /= num_tasks
                else:
                    for i in range(len(meta_gradients[param_name])):
                        meta_gradients[param_name][i] /= num_tasks

            avg_meta_loss = total_meta_loss / num_tasks
        else:
            # Return zero gradients if no valid tasks
            meta_gradients = {
                param_name: (
                    [[0.0 for _ in row] for row in param_values]
                    if isinstance(param_values[0], list)
                    else [0.0 for _ in param_values]
                )
                for param_name, param_values in meta_params.items()
            }
            avg_meta_loss = 0.0

        return meta_gradients, avg_meta_loss

    def federated_meta_update(self, num_rounds: int = 10) -> Dict[str, Any]:
        """Run federated meta-learning for specified rounds."""
        meta_params = self.meta_model.get_params()

        for round_num in range(num_rounds):
            # Collect meta-gradients from all participants
            participant_gradients = []
            participant_losses = []

            for participant_id in self.participants:
                meta_grads, meta_loss = self.compute_meta_gradients(
                    participant_id, meta_params
                )
                participant_gradients.append(meta_grads)
                participant_losses.append(meta_loss)

            # Federated averaging of meta-gradients
            if participant_gradients:
                aggregated_gradients = self._aggregate_gradients(participant_gradients)

                # Apply clipping if configured
                if self.config.clip_grad_norm > 0:
                    aggregated_gradients = self._clip_gradients(
                        aggregated_gradients, self.config.clip_grad_norm
                    )

                # Update meta-parameters
                meta_params = self.meta_model.apply_gradients(
                    aggregated_gradients, self.config.outer_lr, meta_params
                )

                # Record training metrics
                avg_meta_loss = (
                    sum(participant_losses) / len(participant_losses)
                    if participant_losses
                    else 0.0
                )
                self.meta_train_losses.append(avg_meta_loss)

        # Update meta-model with final parameters
        self.meta_model.set_params(meta_params)

        return {
            "final_meta_params": meta_params,
            "meta_train_losses": self.meta_train_losses,
            "num_rounds": num_rounds,
            "num_participants": len(self.participants),
        }

    def _aggregate_gradients(
        self, gradient_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate gradients from multiple participants using FedAvg."""
        if not gradient_list:
            return {}

        aggregated = None
        num_participants = len(gradient_list)

        for gradients in gradient_list:
            if aggregated is None:
                aggregated = copy.deepcopy(gradients)
            else:
                for param_name in gradients:
                    if param_name in ["w1", "w2"]:
                        for i in range(len(gradients[param_name])):
                            for j in range(len(gradients[param_name][i])):
                                aggregated[param_name][i][j] += gradients[param_name][
                                    i
                                ][j]
                    else:
                        for i in range(len(gradients[param_name])):
                            aggregated[param_name][i] += gradients[param_name][i]

        # Average the gradients
        if aggregated and num_participants > 0:
            for param_name in aggregated:
                if param_name in ["w1", "w2"]:
                    for i in range(len(aggregated[param_name])):
                        for j in range(len(aggregated[param_name][i])):
                            aggregated[param_name][i][j] /= num_participants
                else:
                    for i in range(len(aggregated[param_name])):
                        aggregated[param_name][i] /= num_participants

        return aggregated

    def _clip_gradients(
        self, gradients: Dict[str, Any], max_norm: float
    ) -> Dict[str, Any]:
        """Apply gradient clipping to prevent exploding gradients."""
        # Compute gradient norm
        total_norm = 0.0
        for param_name, grad_values in gradients.items():
            if param_name in ["w1", "w2"]:
                for row in grad_values:
                    for val in row:
                        total_norm += val**2
            else:
                for val in grad_values:
                    total_norm += val**2

        total_norm = math.sqrt(total_norm)

        # Apply clipping if needed
        if total_norm > max_norm:
            scale_factor = max_norm / total_norm
            clipped_gradients = copy.deepcopy(gradients)

            for param_name in clipped_gradients:
                if param_name in ["w1", "w2"]:
                    for i in range(len(clipped_gradients[param_name])):
                        for j in range(len(clipped_gradients[param_name][i])):
                            clipped_gradients[param_name][i][j] *= scale_factor
                else:
                    for i in range(len(clipped_gradients[param_name])):
                        clipped_gradients[param_name][i] *= scale_factor

            return clipped_gradients

        return gradients

    def evaluate_adaptation(
        self, test_tasks: List[Task], adaptation_steps: int = 5
    ) -> Dict[str, Any]:
        """Evaluate how quickly the meta-model adapts to new tasks."""
        meta_params = self.meta_model.get_params()
        adaptation_results = []

        for task in test_tasks:
            # Test adaptation on this task
            support_set, query_set = task.sample_support_query(
                self.config.k_shot, self.config.q_query
            )

            # Track performance before and after adaptation
            pre_adaptation_loss = 0.0
            for x, target in query_set:
                pre_adaptation_loss += self.meta_model.loss(x, target, meta_params)
            pre_adaptation_loss /= len(query_set)

            # Perform adaptation
            adapted_params, inner_losses = self.inner_loop_adaptation(task, meta_params)

            # Test post-adaptation performance
            post_adaptation_loss = 0.0
            for x, target in query_set:
                post_adaptation_loss += self.meta_model.loss(x, target, adapted_params)
            post_adaptation_loss /= len(query_set)

            adaptation_results.append(
                {
                    "task_id": task.task_id,
                    "pre_adaptation_loss": pre_adaptation_loss,
                    "post_adaptation_loss": post_adaptation_loss,
                    "improvement": pre_adaptation_loss - post_adaptation_loss,
                    "inner_losses": inner_losses,
                }
            )

        return {
            "adaptation_results": adaptation_results,
            "avg_improvement": sum(r["improvement"] for r in adaptation_results)
            / len(adaptation_results),
            "avg_pre_loss": sum(r["pre_adaptation_loss"] for r in adaptation_results)
            / len(adaptation_results),
            "avg_post_loss": sum(r["post_adaptation_loss"] for r in adaptation_results)
            / len(adaptation_results),
        }


def generate_sine_wave_tasks(
    num_tasks: int = 10, num_samples_per_task: int = 20
) -> List[Task]:
    """Generate sine wave regression tasks with different amplitudes and phases."""
    tasks = []

    for i in range(num_tasks):
        # Random amplitude and phase for each task
        amplitude = random.uniform(0.5, 2.0)
        phase = random.uniform(0, 2 * math.pi)
        frequency = random.uniform(0.5, 2.0)

        # Generate data points
        data = []
        for _ in range(num_samples_per_task):
            x_val = random.uniform(-2.0, 2.0)
            y_val = amplitude * math.sin(frequency * x_val + phase) + random.gauss(
                0, 0.1
            )
            data.append(([x_val], [y_val]))

        task = Task(
            task_id=f"sine_task_{i}",
            input_dim=1,
            output_dim=1,
            data=data,
            task_type="regression",
        )
        tasks.append(task)

    return tasks


def run_federated_maml(
    num_participants: int = 3,
    tasks_per_participant: int = 8,
    meta_rounds: int = 15,
    config: Optional[MAMLConfig] = None,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Run federated MAML experiment."""
    if seed is not None:
        random.seed(seed)

    if config is None:
        config = MAMLConfig()

    # Generate diverse tasks for each participant
    all_tasks = generate_sine_wave_tasks(num_participants * tasks_per_participant)

    # Initialize federated MAML
    fed_maml = FederatedMAML(input_dim=1, hidden_dim=10, output_dim=1, config=config)

    # Distribute tasks to participants
    for p in range(num_participants):
        participant_id = f"participant_{p}"
        start_idx = p * tasks_per_participant
        end_idx = start_idx + tasks_per_participant
        participant_tasks = all_tasks[start_idx:end_idx]
        fed_maml.register_participant(participant_id, participant_tasks)

    # Run federated meta-learning
    training_results = fed_maml.federated_meta_update(meta_rounds)

    # Generate test tasks for evaluation
    test_tasks = generate_sine_wave_tasks(5, 15)
    evaluation_results = fed_maml.evaluate_adaptation(test_tasks)

    return {
        "training_results": training_results,
        "evaluation_results": evaluation_results,
        "config": {
            "num_participants": num_participants,
            "tasks_per_participant": tasks_per_participant,
            "meta_rounds": meta_rounds,
            "inner_lr": config.inner_lr,
            "outer_lr": config.outer_lr,
            "inner_steps": config.inner_steps,
            "k_shot": config.k_shot,
        },
    }


def demo() -> None:
    """Run a federated meta-learning demo."""
    print("Federated Meta-Learning (MAML) Demo")
    print("=" * 50)

    config = MAMLConfig(
        inner_lr=0.01, outer_lr=0.001, inner_steps=3, k_shot=5, q_query=10
    )

    results = run_federated_maml(
        num_participants=4,
        tasks_per_participant=6,
        meta_rounds=12,
        config=config,
        seed=123,
    )

    print("Training Results:")
    training = results["training_results"]
    print(f"  Meta-learning rounds: {training['num_rounds']}")
    print(f"  Participants: {training['num_participants']}")

    if training["meta_train_losses"]:
        initial_loss = training["meta_train_losses"][0]
        final_loss = training["meta_train_losses"][-1]
        print(f"  Initial meta-loss: {initial_loss:.4f}")
        print(f"  Final meta-loss: {final_loss:.4f}")
        print(
            f"  Loss reduction: {((initial_loss - final_loss) / initial_loss * 100):.1f}%"
        )

    print("\nAdaptation Evaluation:")
    evaluation = results["evaluation_results"]
    print(f"  Test tasks: {len(evaluation['adaptation_results'])}")
    print(f"  Avg pre-adaptation loss: {evaluation['avg_pre_loss']:.4f}")
    print(f"  Avg post-adaptation loss: {evaluation['avg_post_loss']:.4f}")
    print(f"  Avg improvement: {evaluation['avg_improvement']:.4f}")
    print(
        f"  Adaptation efficiency: {(evaluation['avg_improvement'] / evaluation['avg_pre_loss'] * 100):.1f}%"
    )

    print("\nper-Task Adaptation:")
    for i, result in enumerate(evaluation["adaptation_results"][:3]):
        print(
            f"  Task {i+1}: {result['pre_adaptation_loss']:.4f} → {result['post_adaptation_loss']:.4f} "
            f"(improvement: {result['improvement']:.4f})"
        )


if __name__ == "__main__":
    demo()
