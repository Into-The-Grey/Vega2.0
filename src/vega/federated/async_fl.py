"""
Asynchronous Federated Learning (AFL) with staleness tolerance and dynamic scheduling.

Implements asynchronous federated learning where participants can contribute updates
at different times without waiting for global synchronization rounds. Features
staleness-aware aggregation and dynamic participant selection.

Design goals:
- Non-blocking aggregation with immediate model updates
- Staleness tolerance using age-weighted averaging
- Dynamic participant scheduling based on availability
- Gradient staleness compensation mechanisms
- Real-time federated learning without synchronization barriers

Key concepts:
- Participants send updates asynchronously when ready
- Coordinator maintains global model with continuous updates
- Stale gradients are weighted based on their age
- Fast participants don't wait for slow ones
- Bounded staleness prevents outdated contributions
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set
from collections import defaultdict, deque
import threading
import queue


@dataclass
class AsyncUpdate:
    """Represents an asynchronous model update from a participant."""

    participant_id: str
    model_params: Dict[str, Any]  # Model parameters (weights, biases)
    gradient: Optional[Dict[str, Any]]  # Optional gradient information
    timestamp: float  # When the update was created
    local_epoch: int  # Local training epoch when update was generated
    staleness: int = 0  # How many global updates have occurred since this was created


@dataclass
class ParticipantState:
    """Tracks the state of an asynchronous participant."""

    participant_id: str
    last_update_time: float = 0.0
    total_updates: int = 0
    current_staleness: int = 0
    is_active: bool = True
    avg_update_interval: float = 1.0  # Average time between updates
    contribution_weight: float = 1.0  # Dynamic weight based on performance


@dataclass
class AsyncFLConfig:
    """Configuration for Asynchronous Federated Learning."""

    max_staleness: int = 5  # Maximum allowed staleness for updates
    staleness_decay: float = 0.8  # Decay factor for stale updates
    min_participants: int = 2  # Minimum participants before aggregation
    max_buffer_size: int = 100  # Maximum number of buffered updates
    update_threshold: int = 3  # Number of updates to trigger aggregation
    participant_timeout: float = 10.0  # Timeout for inactive participants
    dynamic_weighting: bool = True  # Enable dynamic participant weighting


class SimpleAsyncModel:
    """Simple linear model for async federated learning demonstration."""

    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Initialize small random weights
        self.weights = [
            [random.gauss(0, 0.1) for _ in range(output_dim)] for _ in range(input_dim)
        ]
        self.bias = [0.0] * output_dim
        self.version = 0  # Model version counter

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "weights": [row[:] for row in self.weights],
            "bias": self.bias[:],
            "version": self.version,
        }

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters."""
        self.weights = [row[:] for row in params["weights"]]
        self.bias = params["bias"][:]
        if "version" in params:
            self.version = params["version"]

    def forward(self, x: List[float]) -> List[float]:
        """Forward pass."""
        y = self.bias[:]
        for i in range(self.output_dim):
            for j in range(self.input_dim):
                y[i] += self.weights[j][i] * x[j]
        return y

    def compute_gradient(self, x: List[float], y_true: List[float]) -> Dict[str, Any]:
        """Compute gradients for a single sample."""
        y_pred = self.forward(x)
        errors = [pred - true for pred, true in zip(y_pred, y_true)]

        # Gradients
        grad_weights = [
            [2 * errors[i] * x[j] / self.output_dim for i in range(self.output_dim)]
            for j in range(self.input_dim)
        ]
        grad_bias = [2 * errors[i] / self.output_dim for i in range(self.output_dim)]

        return {"weights": grad_weights, "bias": grad_bias}

    def apply_gradient(self, gradient: Dict[str, Any], lr: float = 0.01) -> None:
        """Apply gradient update to model parameters."""
        # Update weights
        for j in range(self.input_dim):
            for i in range(self.output_dim):
                self.weights[j][i] -= lr * gradient["weights"][j][i]

        # Update bias
        for i in range(self.output_dim):
            self.bias[i] -= lr * gradient["bias"][i]

        self.version += 1


class AsyncParticipant:
    """Asynchronous federated learning participant."""

    def __init__(
        self,
        participant_id: str,
        input_dim: int,
        output_dim: int,
        update_interval_range: Tuple[float, float] = (0.5, 2.0),
    ):
        self.participant_id = participant_id
        self.model = SimpleAsyncModel(input_dim, output_dim)
        self.update_interval_range = update_interval_range
        self.is_running = False
        self.update_queue = queue.Queue()

        # Local training state
        self.local_epoch = 0
        self.local_data = []  # Will be set by training data

    def set_training_data(self, X: List[List[float]], y: List[List[float]]) -> None:
        """Set local training data."""
        self.local_data = list(zip(X, y))

    def local_training_step(self, lr: float = 0.01) -> Dict[str, Any]:
        """Perform one local training step and return gradient."""
        if not self.local_data:
            return {"weights": [[0.0]], "bias": [0.0]}

        # Sample a batch (single example for simplicity)
        x, y_true = random.choice(self.local_data)
        gradient = self.model.compute_gradient(x, y_true)
        self.model.apply_gradient(gradient, lr)
        self.local_epoch += 1

        return gradient

    async def run_async_training(
        self, coordinator_queue: asyncio.Queue, num_updates: int = 10, lr: float = 0.01
    ) -> None:
        """Run asynchronous training loop."""
        self.is_running = True

        for _ in range(num_updates):
            if not self.is_running:
                break

            # Perform local training
            gradient = self.local_training_step(lr)

            # Create async update
            update = AsyncUpdate(
                participant_id=self.participant_id,
                model_params=self.model.get_params(),
                gradient=gradient,
                timestamp=time.time(),
                local_epoch=self.local_epoch,
            )

            # Send update to coordinator
            await coordinator_queue.put(update)

            # Random delay to simulate realistic async behavior
            delay = random.uniform(*self.update_interval_range)
            await asyncio.sleep(delay)

        self.is_running = False

    def update_model(self, global_params: Dict[str, Any]) -> None:
        """Update local model with global parameters."""
        self.model.set_params(global_params)


class AsyncFLCoordinator:
    """Asynchronous Federated Learning Coordinator."""

    def __init__(self, input_dim: int, output_dim: int, config: AsyncFLConfig):
        self.config = config
        self.global_model = SimpleAsyncModel(input_dim, output_dim)
        self.participants: Dict[str, ParticipantState] = {}
        self.update_buffer: deque = deque(maxlen=config.max_buffer_size)
        self.global_version = 0
        self.total_updates_processed = 0

        # Threading for async processing
        self.update_queue: asyncio.Queue = None
        self.is_running = False

        # Metrics
        self.aggregation_history: List[Dict[str, Any]] = []

    def register_participant(self, participant_id: str) -> None:
        """Register a new participant."""
        self.participants[participant_id] = ParticipantState(participant_id)

    def compute_staleness(self, update: AsyncUpdate) -> int:
        """Compute staleness of an update."""
        return max(0, self.global_version - update.model_params.get("version", 0))

    def compute_update_weight(self, update: AsyncUpdate, staleness: int) -> float:
        """Compute weight for an update based on staleness and participant performance."""
        # Base staleness weight
        staleness_weight = self.config.staleness_decay**staleness

        # Participant contribution weight
        participant_weight = 1.0
        if update.participant_id in self.participants:
            participant_weight = self.participants[
                update.participant_id
            ].contribution_weight

        return staleness_weight * participant_weight

    def aggregate_updates(self, updates: List[AsyncUpdate]) -> Dict[str, Any]:
        """Aggregate multiple updates into global model parameters."""
        if not updates:
            return self.global_model.get_params()

        # Compute weights for each update
        weighted_updates = []
        total_weight = 0.0

        for update in updates:
            staleness = self.compute_staleness(update)
            if staleness <= self.config.max_staleness:  # Only use non-stale updates
                weight = self.compute_update_weight(update, staleness)
                weighted_updates.append((update, weight))
                total_weight += weight

        if total_weight == 0:
            return self.global_model.get_params()

        # Initialize aggregated parameters
        input_dim = len(self.global_model.weights)
        output_dim = len(self.global_model.bias)

        agg_weights = [[0.0] * output_dim for _ in range(input_dim)]
        agg_bias = [0.0] * output_dim

        # Weighted aggregation
        for update, weight in weighted_updates:
            params = update.model_params
            norm_weight = weight / total_weight

            # Aggregate weights
            for j in range(input_dim):
                for i in range(output_dim):
                    agg_weights[j][i] += norm_weight * params["weights"][j][i]

            # Aggregate bias
            for i in range(output_dim):
                agg_bias[i] += norm_weight * params["bias"][i]

        return {
            "weights": agg_weights,
            "bias": agg_bias,
            "version": self.global_version + 1,
        }

    async def process_updates(self) -> None:
        """Main async loop for processing participant updates."""
        pending_updates = []

        while self.is_running:
            try:
                # Get update with timeout
                update = await asyncio.wait_for(self.update_queue.get(), timeout=1.0)

                # Update participant state
                if update.participant_id in self.participants:
                    participant = self.participants[update.participant_id]
                    participant.last_update_time = time.time()
                    participant.total_updates += 1
                    participant.current_staleness = self.compute_staleness(update)

                # Add to pending updates
                pending_updates.append(update)
                self.total_updates_processed += 1

                # Trigger aggregation if threshold reached
                if len(pending_updates) >= self.config.update_threshold:
                    # Perform aggregation
                    aggregated_params = self.aggregate_updates(pending_updates)
                    self.global_model.set_params(aggregated_params)
                    self.global_version += 1

                    # Record aggregation metrics
                    self.aggregation_history.append(
                        {
                            "global_version": self.global_version,
                            "updates_aggregated": len(pending_updates),
                            "avg_staleness": sum(
                                self.compute_staleness(u) for u in pending_updates
                            )
                            / len(pending_updates),
                            "timestamp": time.time(),
                        }
                    )

                    # Clear pending updates
                    pending_updates.clear()

            except asyncio.TimeoutError:
                # Process any pending updates on timeout
                if (
                    pending_updates
                    and len(pending_updates) >= self.config.min_participants
                ):
                    aggregated_params = self.aggregate_updates(pending_updates)
                    self.global_model.set_params(aggregated_params)
                    self.global_version += 1
                    pending_updates.clear()
                continue

    async def start_coordination(self) -> None:
        """Start the async coordination process."""
        self.update_queue = asyncio.Queue()
        self.is_running = True
        # Start processing updates in background
        asyncio.create_task(self.process_updates())

    def stop_coordination(self) -> None:
        """Stop the coordination process."""
        self.is_running = False

    def get_global_params(self) -> Dict[str, Any]:
        """Get current global model parameters."""
        return self.global_model.get_params()


async def run_async_federated_learning(
    num_participants: int = 3,
    input_dim: int = 2,
    output_dim: int = 1,
    num_updates_per_participant: int = 10,
    config: Optional[AsyncFLConfig] = None,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Run asynchronous federated learning simulation."""
    if seed is not None:
        random.seed(seed)

    if config is None:
        config = AsyncFLConfig()

    # Initialize coordinator
    coordinator = AsyncFLCoordinator(input_dim, output_dim, config)
    await coordinator.start_coordination()

    # Create participants with varying update speeds
    participants = []
    for i in range(num_participants):
        # Different update intervals to simulate heterogeneous participants
        interval_range = (0.1 + i * 0.3, 0.5 + i * 0.5)
        participant = AsyncParticipant(
            f"participant_{i}", input_dim, output_dim, interval_range
        )
        participants.append(participant)
        coordinator.register_participant(participant.participant_id)

        # Generate heterogeneous local data for each participant
        X_local = [[random.gauss(i, 1), random.gauss(i, 1)] for _ in range(20)]
        y_local = [[sum(x) + random.gauss(0, 0.1)] for x in X_local]
        participant.set_training_data(X_local, y_local)

    # Start async training for all participants
    training_tasks = []
    for participant in participants:
        task = asyncio.create_task(
            participant.run_async_training(
                coordinator.update_queue, num_updates_per_participant
            )
        )
        training_tasks.append(task)

    # Wait for all participants to complete
    await asyncio.gather(*training_tasks)

    # Allow final aggregation
    await asyncio.sleep(1.0)

    # Stop coordination
    coordinator.stop_coordination()

    # Collect results
    results = {
        "global_model_params": coordinator.get_global_params(),
        "global_version": coordinator.global_version,
        "total_updates_processed": coordinator.total_updates_processed,
        "aggregation_history": coordinator.aggregation_history,
        "participant_states": {
            pid: {
                "total_updates": state.total_updates,
                "current_staleness": state.current_staleness,
                "last_update_time": state.last_update_time,
            }
            for pid, state in coordinator.participants.items()
        },
    }

    return results


def generate_heterogeneous_async_data(
    num_participants: int, samples_per_participant: int = 50
) -> List[Tuple[List[List[float]], List[List[float]]]]:
    """Generate heterogeneous data for async participants."""
    datasets = []

    for i in range(num_participants):
        # Each participant has slightly different data distribution
        X = [
            [random.gauss(i * 0.5, 1), random.gauss(i * 0.5, 1)]
            for _ in range(samples_per_participant)
        ]
        # Different target functions per participant
        if i % 3 == 0:
            y = [[2 * x[0] + x[1] + random.gauss(0, 0.1)] for x in X]
        elif i % 3 == 1:
            y = [[x[0] + 3 * x[1] + random.gauss(0, 0.1)] for x in X]
        else:
            y = [[x[0] - x[1] + random.gauss(0, 0.1)] for x in X]

        datasets.append((X, y))

    return datasets


async def demo() -> None:
    """Run an async federated learning demo."""
    print("Asynchronous Federated Learning Demo")
    print("=" * 50)

    config = AsyncFLConfig(max_staleness=3, update_threshold=2, min_participants=2)

    results = await run_async_federated_learning(
        num_participants=4,
        input_dim=2,
        output_dim=1,
        num_updates_per_participant=8,
        config=config,
        seed=123,
    )

    print(f"Global model version: {results['global_version']}")
    print(f"Total updates processed: {results['total_updates_processed']}")
    print(f"Number of aggregations: {len(results['aggregation_history'])}")

    print("\nParticipant Statistics:")
    for pid, stats in results["participant_states"].items():
        print(
            f"  {pid}: {stats['total_updates']} updates, staleness: {stats['current_staleness']}"
        )

    if results["aggregation_history"]:
        print("\nAggregation History (last 3):")
        for agg in results["aggregation_history"][-3:]:
            print(
                f"  Version {agg['global_version']}: {agg['updates_aggregated']} updates, "
                f"avg staleness: {agg['avg_staleness']:.2f}"
            )


if __name__ == "__main__":
    asyncio.run(demo())
