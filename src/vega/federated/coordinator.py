"""
Central Coordinator Service for Federated Learning

Hybrid-centralized coordinator that manages federated learning sessions,
participant coordination, and model aggregation for personal/family use.

Design Principles:
- Central hub for coordination (hybrid-centralized architecture)
- Manages 2-3 participants maximum
- Integration with existing Vega config and database
- Session-based federated learning
- Manual participant registration
- Dynamic encryption integration
- Trusted environment model
"""

import asyncio
import uuid
import time
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

from .model_serialization import ModelWeights, ModelSerializer
from .communication import CommunicationManager, FederatedMessage
from .encryption import DynamicEncryption
from .fedavg import FedAvg, FedAvgConfig, AsyncAggregator

# Integration with existing Vega components
try:
    from ..core.config import get_config
    from ..core.db import get_database_session, Base
    from sqlalchemy import Column, String, Integer, Float, Text, Boolean, DateTime
    from sqlalchemy.sql import func

    HAS_VEGA_INTEGRATION = True
except ImportError:
    HAS_VEGA_INTEGRATION = False

logger = logging.getLogger(__name__)


class SessionStatus(Enum):
    """Federated learning session status."""

    CREATED = "created"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AggregationStrategy(Enum):
    """Model aggregation strategies."""

    FEDERATED_AVERAGING = "federated_averaging"
    WEIGHTED_AVERAGING = "weighted_averaging"
    MEDIAN_AGGREGATION = "median_aggregation"


@dataclass
class FederatedSession:
    """Represents a federated learning session."""

    session_id: str
    name: str
    description: str
    coordinator_id: str
    participants: List[str]
    model_type: str  # 'pytorch' or 'tensorflow'
    aggregation_strategy: AggregationStrategy
    status: SessionStatus
    created_at: float
    updated_at: float
    current_round: int = 0
    max_rounds: int = 10
    min_participants: int = 2
    round_timeout_seconds: int = 300
    convergence_threshold: float = 0.001
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        data["aggregation_strategy"] = self.aggregation_strategy.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FederatedSession":
        """Create from dictionary."""
        data["status"] = SessionStatus(data["status"])
        data["aggregation_strategy"] = AggregationStrategy(data["aggregation_strategy"])
        return cls(**data)


@dataclass
class RoundResult:
    """Results from a federated learning round."""

    session_id: str
    round_number: int
    participant_weights: Dict[str, ModelWeights]
    aggregated_weights: ModelWeights
    aggregation_metrics: Dict[str, float]
    round_duration: float
    timestamp: float


# Database model for session persistence (if Vega integration available)
if HAS_VEGA_INTEGRATION:

    class FederatedSessionDB(Base):
        __tablename__ = "federated_sessions"

        session_id = Column(String, primary_key=True)
        name = Column(String, nullable=False)
        description = Column(Text)
        coordinator_id = Column(String, nullable=False)
        participants = Column(Text)  # JSON string
        model_type = Column(String, nullable=False)
        aggregation_strategy = Column(String, nullable=False)
        status = Column(String, nullable=False)
        created_at = Column(DateTime, default=func.now())
        updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
        current_round = Column(Integer, default=0)
        max_rounds = Column(Integer, default=10)
        min_participants = Column(Integer, default=2)
        round_timeout_seconds = Column(Integer, default=300)
        convergence_threshold = Column(Float, default=0.001)
        metadata = Column(Text)  # JSON string


class ModelAggregator:
    """
    Model aggregation utilities for federated learning.

    Implements various aggregation strategies for combining participant models.
    """

    @staticmethod
    def federated_averaging(
        participant_weights: Dict[str, ModelWeights],
        participant_data_sizes: Optional[Dict[str, int]] = None,
    ) -> ModelWeights:
        """
        Perform federated averaging aggregation.

        Args:
            participant_weights: Dictionary of participant_id -> ModelWeights
            participant_data_sizes: Optional data sizes for weighted averaging

        Returns:
            Aggregated ModelWeights
        """
        if not participant_weights:
            raise ValueError("No participant weights provided")

        # Get reference weights for structure
        reference_weights = next(iter(participant_weights.values()))

        # Ensure all weights are decompressed
        decompressed_weights = {}
        for pid, weights in participant_weights.items():
            if weights.compressed:
                decompressed_weights[pid] = weights.decompress()
            else:
                decompressed_weights[pid] = weights

        # Calculate weights (uniform if no data sizes provided)
        total_participants = len(participant_weights)
        if participant_data_sizes:
            total_data = sum(participant_data_sizes.values())
            aggregation_weights = {
                pid: participant_data_sizes.get(pid, 1) / total_data
                for pid in participant_weights.keys()
            }
        else:
            aggregation_weights = {
                pid: 1.0 / total_participants for pid in participant_weights.keys()
            }

        # Aggregate weights
        aggregated = {}
        for weight_name in reference_weights.weights.keys():
            # Initialize with zeros
            aggregated_weight = None

            for pid, weights in decompressed_weights.items():
                if weight_name in weights.weights:
                    participant_weight = weights.weights[weight_name]
                    weighted_contribution = (
                        participant_weight * aggregation_weights[pid]
                    )

                    if aggregated_weight is None:
                        aggregated_weight = weighted_contribution.copy()
                    else:
                        aggregated_weight += weighted_contribution

            if aggregated_weight is not None:
                aggregated[weight_name] = aggregated_weight

        # Create aggregated ModelWeights
        aggregated_model_weights = ModelWeights(
            weights=aggregated,
            model_type=reference_weights.model_type,
            architecture_info=reference_weights.architecture_info.copy(),
            metadata={
                "aggregation_strategy": "federated_averaging",
                "num_participants": len(participant_weights),
                "aggregation_weights": aggregation_weights,
                "aggregated_at": time.time(),
            },
            checksum="",  # Will be calculated
        )

        aggregated_model_weights.update_checksum()
        return aggregated_model_weights

    @staticmethod
    def weighted_averaging(
        participant_weights: Dict[str, ModelWeights], weights: Dict[str, float]
    ) -> ModelWeights:
        """
        Perform weighted averaging with custom weights.

        Args:
            participant_weights: Dictionary of participant_id -> ModelWeights
            weights: Dictionary of participant_id -> weight

        Returns:
            Aggregated ModelWeights
        """
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {pid: w / total_weight for pid, w in weights.items()}

        return ModelAggregator.federated_averaging(
            participant_weights, normalized_weights
        )

    @staticmethod
    def median_aggregation(
        participant_weights: Dict[str, ModelWeights],
    ) -> ModelWeights:
        """
        Perform median aggregation (robust to outliers).

        Args:
            participant_weights: Dictionary of participant_id -> ModelWeights

        Returns:
            Aggregated ModelWeights
        """
        import numpy as np

        if not participant_weights:
            raise ValueError("No participant weights provided")

        # Get reference weights for structure
        reference_weights = next(iter(participant_weights.values()))

        # Ensure all weights are decompressed
        decompressed_weights = {}
        for pid, weights in participant_weights.items():
            if weights.compressed:
                decompressed_weights[pid] = weights.decompress()
            else:
                decompressed_weights[pid] = weights

        # Aggregate using median
        aggregated = {}
        for weight_name in reference_weights.weights.keys():
            # Collect all participant weights for this parameter
            param_weights = []
            for pid, weights in decompressed_weights.items():
                if weight_name in weights.weights:
                    param_weights.append(weights.weights[weight_name])

            if param_weights:
                # Stack along new axis and compute median
                stacked_weights = np.stack(param_weights, axis=0)
                median_weight = np.median(stacked_weights, axis=0)
                aggregated[weight_name] = median_weight

        # Create aggregated ModelWeights
        aggregated_model_weights = ModelWeights(
            weights=aggregated,
            model_type=reference_weights.model_type,
            architecture_info=reference_weights.architecture_info.copy(),
            metadata={
                "aggregation_strategy": "median_aggregation",
                "num_participants": len(participant_weights),
                "aggregated_at": time.time(),
            },
            checksum="",  # Will be calculated
        )

        aggregated_model_weights.update_checksum()
        return aggregated_model_weights


class FederatedCoordinator:
    """
    Central coordinator for federated learning sessions.

    Manages participant coordination, session lifecycle, and model aggregation
    for the hybrid-centralized architecture.
    """

    def __init__(
        self,
        coordinator_id: str,
        coordinator_name: str,
        host: str = "127.0.0.1",
        port: int = 8000,
    ):
        """
        Initialize federated coordinator.

        Args:
            coordinator_id: Unique coordinator identifier
            coordinator_name: Human-readable name
            host: Coordinator host address
            port: Coordinator port
        """
        self.coordinator_id = coordinator_id
        self.coordinator_name = coordinator_name
        self.host = host
        self.port = port

        # Initialize communication
        self.comm_manager = CommunicationManager(
            participant_id=coordinator_id,
            participant_name=coordinator_name,
            host=host,
            port=port,
        )

        # Session management
        self.active_sessions: Dict[str, FederatedSession] = {}
        self.session_results: Dict[str, List[RoundResult]] = {}

        # Aggregation strategies
        self.aggregation_strategies = {
            AggregationStrategy.FEDERATED_AVERAGING: ModelAggregator.federated_averaging,
            AggregationStrategy.WEIGHTED_AVERAGING: ModelAggregator.weighted_averaging,
            AggregationStrategy.MEDIAN_AGGREGATION: ModelAggregator.median_aggregation,
        }
        # FedAvg helper for selection & async policy
        self.fedavg = FedAvg(FedAvgConfig())

        logger.info(f"Federated coordinator initialized: {coordinator_name}")

    def create_session(
        self,
        name: str,
        description: str,
        model_type: str,
        participants: List[str],
        aggregation_strategy: AggregationStrategy = AggregationStrategy.FEDERATED_AVERAGING,
        max_rounds: int = 10,
        min_participants: int = 2,
        round_timeout_seconds: int = 300,
        convergence_threshold: float = 0.001,
    ) -> FederatedSession:
        """
        Create a new federated learning session.

        Args:
            name: Session name
            description: Session description
            model_type: Type of models ('pytorch' or 'tensorflow')
            participants: List of participant IDs
            aggregation_strategy: Strategy for model aggregation
            max_rounds: Maximum number of rounds
            min_participants: Minimum participants required
            round_timeout_seconds: Timeout for each round
            convergence_threshold: Convergence threshold for early stopping

        Returns:
            Created FederatedSession
        """
        session_id = str(uuid.uuid4())

        session = FederatedSession(
            session_id=session_id,
            name=name,
            description=description,
            coordinator_id=self.coordinator_id,
            participants=participants,
            model_type=model_type,
            aggregation_strategy=aggregation_strategy,
            status=SessionStatus.CREATED,
            created_at=time.time(),
            updated_at=time.time(),
            max_rounds=max_rounds,
            min_participants=min_participants,
            round_timeout_seconds=round_timeout_seconds,
            convergence_threshold=convergence_threshold,
        )

        self.active_sessions[session_id] = session
        self.session_results[session_id] = []

        # Save to database if available
        if HAS_VEGA_INTEGRATION:
            self._save_session_to_db(session)

        logger.info(f"Created federated learning session: {name} ({session_id})")
        return session

    def get_session(self, session_id: str) -> Optional[FederatedSession]:
        """Get session by ID."""
        return self.active_sessions.get(session_id)

    def list_sessions(
        self, status_filter: Optional[SessionStatus] = None
    ) -> List[FederatedSession]:
        """
        List active sessions.

        Args:
            status_filter: Filter by session status

        Returns:
            List of sessions
        """
        sessions = list(self.active_sessions.values())

        if status_filter:
            sessions = [s for s in sessions if s.status == status_filter]

        return sessions

    async def start_session(self, session_id: str) -> bool:
        """
        Start a federated learning session.

        Args:
            session_id: Session to start

        Returns:
            True if started successfully
        """
        session = self.active_sessions.get(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return False

        if session.status != SessionStatus.CREATED:
            logger.error(
                f"Session {session_id} cannot be started (status: {session.status})"
            )
            return False

        try:
            # Update session status
            session.status = SessionStatus.INITIALIZING
            session.updated_at = time.time()

            # Check participant availability
            health_check = await self.comm_manager.health_check()
            active_participants = [
                pid
                for pid in session.participants
                if pid
                in [
                    p.participant_id
                    for p in self.comm_manager.registry.get_active_participants()
                ]
            ]

            if len(active_participants) < session.min_participants:
                logger.error(
                    f"Insufficient participants: {len(active_participants)} < {session.min_participants}"
                )
                session.status = SessionStatus.FAILED
                return False

            # Initialize session with participants
            init_message = {
                "session_id": session_id,
                "session_config": session.to_dict(),
                "coordinator_id": self.coordinator_id,
            }

            responses = await self.comm_manager.broadcast_to_all(
                message_type="session_init", data=init_message, session_id=session_id
            )

            # Check initialization responses
            successful_inits = [
                r
                for r in responses.values()
                if r.get("success") and r.get("data", {}).get("status") == "ready"
            ]

            if len(successful_inits) >= session.min_participants:
                session.status = SessionStatus.RUNNING
                logger.info(f"Session {session_id} started successfully")
                return True
            else:
                session.status = SessionStatus.FAILED
                logger.error(
                    f"Session initialization failed: {len(successful_inits)} participants ready"
                )
                return False

        except Exception as e:
            logger.error(f"Error starting session {session_id}: {e}")
            session.status = SessionStatus.FAILED
            return False
        finally:
            session.updated_at = time.time()
            if HAS_VEGA_INTEGRATION:
                self._save_session_to_db(session)

    async def run_training_round(self, session_id: str) -> Optional[RoundResult]:
        """
        Execute a single training round.

        Args:
            session_id: Session to run round for

        Returns:
            Round result or None if failed
        """
        session = self.active_sessions.get(session_id)
        if not session or session.status != SessionStatus.RUNNING:
            return None

        try:
            round_start_time = time.time()

            # Selection: choose a subset of participants based on strategy
            active_pids = [
                p.participant_id
                for p in self.comm_manager.registry.get_active_participants()
                if p.participant_id in session.participants
            ]
            # Use prior metrics if available for performance-based selection
            prev_results = self.session_results.get(session_id, [])
            metrics_by_pid = {}
            if prev_results:
                # Last round's metrics
                last = prev_results[-1]
                # No per-participant loss in current structure; placeholder using training_samples descending
                for pid in session.participants:
                    metrics_by_pid[pid] = {"loss": 0.0, "training_samples": 1}

            # Number to select (default: all)
            num_select = len(active_pids)
            sel_indices = self.fedavg.select_participants(
                active_pids,
                [metrics_by_pid.get(pid, {}) for pid in active_pids],
                num_select=num_select,
            )
            selected_pids = [active_pids[i] for i in sel_indices]

            training_request = {
                "session_id": session_id,
                "round_number": session.current_round + 1,
                "timeout_seconds": session.round_timeout_seconds,
            }

            # Async mode: accept updates until min threshold or timeout
            async_mode = session.metadata.get("async_mode", False)
            min_updates = int(
                session.metadata.get("min_updates", max(1, session.min_participants))
            )
            aggregator = None
            if async_mode:
                aggregator = AsyncAggregator(
                    min_updates=min_updates,
                    timeout_seconds=session.round_timeout_seconds,
                )
                aggregator.start_round(f"{session_id}:{session.current_round+1}")

            # Send to selected participants
            responses = await self.comm_manager.send_to_participants(
                selected_pids,
                message_type="training_round",
                data=training_request,
                session_id=session_id,
            )

            # Collect model weights from participants
            participant_weights: Dict[str, ModelWeights] = {}
            participant_metrics: Dict[str, Dict[str, Any]] = {}

            for participant_id in selected_pids:
                r = responses.get(participant_id)
                if r and r.get("success"):
                    participant_response = r.get("data", {})
                    if "model_weights" in participant_response:
                        weights_data = participant_response["model_weights"]
                        model_weights = ModelWeights.from_dict(weights_data)
                        participant_weights[participant_id] = model_weights
                        participant_metrics[participant_id] = participant_response.get(
                            "metrics", {}
                        )

                        if async_mode and aggregator is not None:
                            # Convert to numpy dict for FedAvg.aggregate helper
                            np_weights = {
                                k: v for k, v in model_weights.weights.items()
                            }
                            data_size = int(
                                participant_metrics[participant_id].get(
                                    "training_samples", 1
                                )
                            )
                            aggregator.submit_update(np_weights, data_size)

            # Async aggregation readiness
            if async_mode and aggregator is not None:
                if not aggregator.is_ready():
                    # Wait remaining time (simple sleep loop); in real system this would be event-driven
                    end_by = round_start_time + session.round_timeout_seconds
                    while time.time() < end_by and not aggregator.is_ready():
                        await asyncio.sleep(0.1)

                # Aggregate asynchronously collected updates
                if aggregator.is_ready():
                    np_agg = aggregator.aggregate(
                        self.fedavg,
                        robust=(
                            session.aggregation_strategy
                            == AggregationStrategy.MEDIAN_AGGREGATION
                        ),
                    )
                    # Map back to ModelWeights shape using first structure
                    if participant_weights:
                        ref = next(iter(participant_weights.values()))
                        agg_weights = {k: np_agg[k] for k in ref.weights.keys()}
                        aggregated_weights = ModelWeights(
                            weights=agg_weights,
                            model_type=ref.model_type,
                            architecture_info=ref.architecture_info.copy(),
                            metadata={"aggregation_strategy": "async_fedavg"},
                            checksum="",
                        )
                        aggregated_weights.update_checksum()
                    else:
                        logger.warning("No updates aggregated in async mode")
                        return None
                else:
                    logger.warning("Async aggregation not ready before timeout")
                    return None
            else:
                # Synchronous aggregation path (existing behavior)
                if len(participant_weights) < session.min_participants:
                    logger.warning(
                        f"Insufficient weight submissions: {len(participant_weights)} < {session.min_participants}"
                    )
                    return None

                aggregation_func = self.aggregation_strategies[
                    session.aggregation_strategy
                ]

                if (
                    session.aggregation_strategy
                    == AggregationStrategy.WEIGHTED_AVERAGING
                ):
                    data_sizes = {
                        pid: metrics.get("training_samples", 1)
                        for pid, metrics in participant_metrics.items()
                    }
                    aggregated_weights = aggregation_func(
                        participant_weights, data_sizes
                    )
                else:
                    aggregated_weights = aggregation_func(participant_weights)

            # Calculate aggregation metrics
            aggregation_metrics = {
                "num_participants": len(participant_weights),
                "aggregation_strategy": session.aggregation_strategy.value,
                "round_duration": time.time() - round_start_time,
            }

            # Convergence check (existing)
            if session.current_round > 0:
                previous_results = self.session_results[session_id]
                if previous_results:
                    prev_weights = previous_results[-1].aggregated_weights
                    weight_diff = self._calculate_weight_difference(
                        prev_weights, aggregated_weights
                    )
                    aggregation_metrics["weight_difference"] = weight_diff
                    aggregation_metrics["converged"] = (
                        weight_diff < session.convergence_threshold
                    )

            # Create round result
            round_result = RoundResult(
                session_id=session_id,
                round_number=session.current_round + 1,
                participant_weights=participant_weights,
                aggregated_weights=aggregated_weights,
                aggregation_metrics=aggregation_metrics,
                round_duration=time.time() - round_start_time,
                timestamp=time.time(),
            )

            # Update session
            session.current_round += 1
            session.updated_at = time.time()

            # Store round result
            self.session_results[session_id].append(round_result)

            # Send aggregated weights back to participants
            update_message = {
                "session_id": session_id,
                "round_number": round_result.round_number,
                "aggregated_weights": aggregated_weights.to_dict(),
                "aggregation_metrics": aggregation_metrics,
            }

            await self.comm_manager.broadcast_to_all(
                message_type="weight_update", data=update_message, session_id=session_id
            )

            # Check for session completion
            if session.current_round >= session.max_rounds or aggregation_metrics.get(
                "converged", False
            ):
                session.status = SessionStatus.COMPLETED
                logger.info(
                    f"Session {session_id} completed after {session.current_round} rounds"
                )

            # Save session state
            if HAS_VEGA_INTEGRATION:
                self._save_session_to_db(session)

            logger.info(
                f"Completed round {round_result.round_number} for session {session_id}"
            )
            return round_result

        except Exception as e:
            logger.error(f"Error in training round for session {session_id}: {e}")
            session.status = SessionStatus.FAILED
            session.updated_at = time.time()
            return None

    async def run_full_session(self, session_id: str) -> bool:
        """
        Run a complete federated learning session.

        Args:
            session_id: Session to run

        Returns:
            True if completed successfully
        """
        # Start the session
        if not await self.start_session(session_id):
            return False

        session = self.active_sessions[session_id]

        # Run training rounds
        while (
            session.status == SessionStatus.RUNNING
            and session.current_round < session.max_rounds
        ):

            round_result = await self.run_training_round(session_id)

            if round_result is None:
                logger.error(f"Round failed, stopping session {session_id}")
                session.status = SessionStatus.FAILED
                break

            # Check for convergence
            if round_result.aggregation_metrics.get("converged", False):
                logger.info(
                    f"Session {session_id} converged after {session.current_round} rounds"
                )
                break

        # Finalize session
        session.updated_at = time.time()
        if HAS_VEGA_INTEGRATION:
            self._save_session_to_db(session)

        return session.status == SessionStatus.COMPLETED

    def get_session_results(self, session_id: str) -> List[RoundResult]:
        """Get results for a session."""
        return self.session_results.get(session_id, [])

    def _calculate_weight_difference(
        self, weights1: ModelWeights, weights2: ModelWeights
    ) -> float:
        """Calculate difference between two sets of weights."""
        import numpy as np

        if weights1.compressed:
            weights1 = weights1.decompress()
        if weights2.compressed:
            weights2 = weights2.decompress()

        total_diff = 0.0
        total_params = 0

        for name in weights1.weights.keys():
            if name in weights2.weights:
                w1 = weights1.weights[name]
                w2 = weights2.weights[name]
                diff = np.mean(np.abs(w1 - w2))
                total_diff += diff * w1.size
                total_params += w1.size

        return total_diff / total_params if total_params > 0 else 0.0

    def _save_session_to_db(self, session: FederatedSession):
        """Save session to database (if available)."""
        if not HAS_VEGA_INTEGRATION:
            return

        try:
            with get_database_session() as db_session:
                # Check if session exists
                existing = (
                    db_session.query(FederatedSessionDB)
                    .filter_by(session_id=session.session_id)
                    .first()
                )

                if existing:
                    # Update existing
                    existing.status = session.status.value
                    existing.current_round = session.current_round
                    existing.metadata = json.dumps(session.metadata)
                else:
                    # Create new
                    db_session = FederatedSessionDB(
                        session_id=session.session_id,
                        name=session.name,
                        description=session.description,
                        coordinator_id=session.coordinator_id,
                        participants=json.dumps(session.participants),
                        model_type=session.model_type,
                        aggregation_strategy=session.aggregation_strategy.value,
                        status=session.status.value,
                        current_round=session.current_round,
                        max_rounds=session.max_rounds,
                        min_participants=session.min_participants,
                        round_timeout_seconds=session.round_timeout_seconds,
                        convergence_threshold=session.convergence_threshold,
                        metadata=json.dumps(session.metadata),
                    )
                    db_session.add(db_session)

                db_session.commit()

        except Exception as e:
            logger.error(f"Error saving session to database: {e}")

    async def cleanup(self):
        """Cleanup coordinator resources."""
        await self.comm_manager.cleanup()


# Example usage and testing
if __name__ == "__main__":

    async def test_coordinator():
        # Test coordinator setup
        coordinator = FederatedCoordinator(
            coordinator_id="coordinator_1", coordinator_name="Main Coordinator"
        )

        try:
            # Create a test session
            session = coordinator.create_session(
                name="Test Federated Learning",
                description="Testing federated learning setup",
                model_type="pytorch",
                participants=["participant_1", "participant_2"],
                max_rounds=5,
            )

            print(f"Created session: {session.session_id}")
            print(f"Session status: {session.status}")

            # Test aggregation
            from .model_serialization import create_test_pytorch_model

            if create_test_pytorch_model():
                model1 = create_test_pytorch_model()
                model2 = create_test_pytorch_model()

                weights1 = ModelSerializer.extract_pytorch_weights(model1)
                weights2 = ModelSerializer.extract_pytorch_weights(model2)

                participant_weights = {
                    "participant_1": weights1,
                    "participant_2": weights2,
                }

                aggregated = ModelAggregator.federated_averaging(participant_weights)
                print(f"Aggregated weights checksum: {aggregated.checksum}")

        finally:
            await coordinator.cleanup()

    # Run test
    asyncio.run(test_coordinator())
