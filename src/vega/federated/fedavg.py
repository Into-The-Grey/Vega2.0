"""
Federated Averaging (FedAvg) Algorithm for Federated Learning

Implements weighted averaging, convergence detection, participant selection, async aggregation, and basic Byzantine fault tolerance.
Modular and pluggable into any federated learning workflow.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
import random
import logging
import time

from .security import (
    audit_log,
    is_anomalous_update,
    check_model_consistency,
    validate_model_update_pipeline,
)

logger = logging.getLogger(__name__)


@dataclass
class FedAvgConfig:
    convergence_threshold: float = 1e-4
    patience: int = 5
    max_rounds: int = 100
    selection_strategy: str = "random"  # "random", "performance"
    byzantine_tolerance: bool = False
    byzantine_method: str = "median"  # "median", "krum", "none"
    async_aggregation: bool = False
    seed: int = 42


class FedAvg:
    def __init__(self, config: FedAvgConfig):
        self.config = config
        self.convergence_history: List[float] = []
        self.current_round = 0
        random.seed(config.seed)

    def select_participants(
        self,
        participants: List[Any],
        metrics: List[Dict[str, Any]],
        num_select: Optional[int] = None,
    ) -> List[int]:
        if self.config.selection_strategy == "random":
            num = num_select or len(participants)
            return random.sample(range(len(participants)), num)
        elif self.config.selection_strategy == "performance":
            sorted_idx = sorted(
                range(len(metrics)), key=lambda i: metrics[i].get("loss", float("inf"))
            )
            num = num_select or len(participants)
            return sorted_idx[:num]
        else:
            return list(range(len(participants)))

    def aggregate(
        self,
        client_weights: List[Dict[str, np.ndarray]],
        client_sizes: List[int],
        participant_ids: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        enable_security: bool = True,
        anomaly_threshold: float = 10.0,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Perform federated averaging with security validation.

        Args:
            client_weights: List of client model weights
            client_sizes: List of client dataset sizes
            participant_ids: Optional list of participant IDs for audit logging
            session_id: Optional session ID for audit logging
            enable_security: Whether to perform security validation
            anomaly_threshold: Threshold for anomaly detection

        Returns:
            Tuple of (aggregated_weights, security_report)
        """
        total_size = sum(client_sizes)
        if total_size <= 0:
            raise ValueError("Total client size must be positive")

        security_report = {
            "participants_validated": len(client_weights),
            "anomalous_participants": [],
            "security_validations": [],
            "aggregation_method": (
                "standard"
                if not self.config.byzantine_tolerance
                else self.config.byzantine_method
            ),
            "total_participants": len(client_weights),
            "timestamp": time.time(),
        }

        # Security validation if enabled
        filtered_weights = []
        filtered_sizes = []
        filtered_participant_ids = []

        if enable_security:
            audit_log(
                "aggregation_security_validation_started",
                {
                    "total_participants": len(client_weights),
                    "byzantine_tolerance": self.config.byzantine_tolerance,
                    "anomaly_threshold": anomaly_threshold,
                },
                session_id=session_id,
            )

            for i, weights in enumerate(client_weights):
                participant_id = (
                    participant_ids[i] if participant_ids else f"participant_{i}"
                )

                # Convert numpy arrays to serializable format for validation
                weights_dict = {}
                for key, array in weights.items():
                    if isinstance(array, np.ndarray):
                        weights_dict[key] = array.tolist()
                    else:
                        weights_dict[key] = array

                # Run security validation
                validation_result = validate_model_update_pipeline(
                    model_update={"weights": weights_dict},
                    participant_id=participant_id,
                    session_id=session_id,
                    anomaly_threshold=anomaly_threshold,
                )

                security_report["security_validations"].append(
                    {
                        "participant_id": participant_id,
                        "validation_result": validation_result,
                        "dataset_size": client_sizes[i] if i < len(client_sizes) else 0,
                    }
                )

                if validation_result["passed_validation"]:
                    filtered_weights.append(weights)
                    filtered_sizes.append(
                        client_sizes[i] if i < len(client_sizes) else 0
                    )
                    filtered_participant_ids.append(participant_id)
                else:
                    security_report["anomalous_participants"].append(
                        {
                            "participant_id": participant_id,
                            "reason": "Failed security validation",
                            "validation_details": validation_result,
                        }
                    )

                    audit_log(
                        "aggregation_participant_rejected",
                        {
                            "participant_id": participant_id,
                            "reason": "Failed security validation",
                            "validation_result": validation_result,
                        },
                        participant_id=participant_id,
                        session_id=session_id,
                    )

            if len(filtered_weights) == 0:
                audit_log(
                    "aggregation_no_valid_participants",
                    {"original_count": len(client_weights)},
                    session_id=session_id,
                )
                raise ValueError("No participants passed security validation")

            # Use filtered weights for aggregation
            client_weights = filtered_weights
            client_sizes = filtered_sizes
            security_report["participants_after_filtering"] = len(client_weights)

        else:
            # No security filtering
            security_report["participants_after_filtering"] = len(client_weights)

        # Perform aggregation (with Byzantine tolerance if configured)
        if self.config.byzantine_tolerance:
            agg = self.byzantine_robust_aggregate(
                client_weights, self.config.byzantine_method
            )
            security_report["aggregation_method"] = (
                f"byzantine_robust_{self.config.byzantine_method}"
            )
        else:
            # Standard weighted averaging
            total_size = sum(client_sizes)
            agg: Dict[str, np.ndarray] = {}
            for key in client_weights[0]:
                agg[key] = sum(
                    w[key] * (sz / total_size)
                    for w, sz in zip(client_weights, client_sizes)
                )

        # Final security validation of aggregated result
        if enable_security:
            agg_dict = {}
            for key, array in agg.items():
                if isinstance(array, np.ndarray):
                    agg_dict[key] = array.tolist()
                else:
                    agg_dict[key] = array

            final_validation = validate_model_update_pipeline(
                model_update={"weights": agg_dict},
                participant_id="aggregated_result",
                session_id=session_id,
                anomaly_threshold=anomaly_threshold,
            )

            security_report["final_validation"] = final_validation

            if not final_validation["passed_validation"]:
                audit_log(
                    "aggregation_result_invalid",
                    {"validation_result": final_validation},
                    session_id=session_id,
                )
                raise ValueError("Aggregated result failed security validation")

        # Audit log successful aggregation
        audit_log(
            "aggregation_completed",
            {
                "total_participants": len(client_weights),
                "aggregation_method": security_report["aggregation_method"],
                "security_enabled": enable_security,
                "anomalous_participants_count": len(
                    security_report["anomalous_participants"]
                ),
            },
            session_id=session_id,
        )

        return agg, security_report

    def byzantine_robust_aggregate(
        self, client_weights: List[Dict[str, np.ndarray]], method: str = "median"
    ) -> Dict[str, np.ndarray]:
        agg: Dict[str, np.ndarray] = {}
        for key in client_weights[0]:
            stacked = np.stack([w[key] for w in client_weights])
            if method == "median":
                agg[key] = np.median(stacked, axis=0)
            elif method == "krum":
                # Simple Krum: pick the update closest to others (not full Krum)
                dists = np.sum(
                    (stacked[:, None] - stacked[None, :]) ** 2,
                    axis=(
                        tuple(range(2, stacked.ndim + 1)) if stacked.ndim >= 2 else (1,)
                    ),
                )
                scores = np.sum(np.sort(dists, axis=1)[:, 1:], axis=1)
                agg[key] = stacked[np.argmin(scores)]
            else:
                agg[key] = np.mean(stacked, axis=0)
        return agg

    def check_convergence(self, loss: float) -> bool:
        self.convergence_history.append(loss)
        if len(self.convergence_history) < self.config.patience:
            return False
        recent = self.convergence_history[-self.config.patience :]
        if max(recent) - min(recent) < self.config.convergence_threshold:
            logger.info(
                f"FedAvg converged: loss change {max(recent) - min(recent):.6f} < threshold {self.config.convergence_threshold}"
            )
            return True
        if self.current_round >= self.config.max_rounds:
            logger.info(f"FedAvg reached max rounds: {self.config.max_rounds}")
            return True
        return False


class AsyncAggregator:
    """
    Collects client updates asynchronously and aggregates when a minimum
    number is reached or a timeout elapses. Intended for small, local
    deployments where a simple threshold/timeout policy is sufficient.
    """

    def __init__(
        self,
        min_updates: int,
        timeout_seconds: float,
        time_provider: Callable[[], float] = time.time,
    ):
        if min_updates <= 0:
            raise ValueError("min_updates must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.min_updates = min_updates
        self.timeout_seconds = timeout_seconds
        self._time = time_provider
        self._round_id: Optional[str] = None
        self._start_time: float = 0.0
        self._updates: List[Tuple[Dict[str, np.ndarray], int]] = []

    def start_round(self, round_id: str) -> None:
        self._round_id = round_id
        self._start_time = self._time()
        self._updates = []

    def submit_update(self, weights: Dict[str, np.ndarray], data_size: int) -> None:
        if self._round_id is None:
            raise RuntimeError("Round not started")
        if data_size <= 0:
            raise ValueError("data_size must be positive")
        self._updates.append((weights, data_size))

    def is_ready(self) -> bool:
        if self._round_id is None:
            return False
        if len(self._updates) >= self.min_updates:
            return True
        return (self._time() - self._start_time) >= self.timeout_seconds

    def aggregate(
        self, fedavg: FedAvg, robust: bool = False
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if not self.is_ready():
            raise RuntimeError("Aggregation not ready: threshold and timeout not met")
        if not self._updates:
            raise RuntimeError("No updates to aggregate")
        client_weights = [u[0] for u in self._updates]
        client_sizes = [u[1] for u in self._updates]
        if robust and fedavg.config.byzantine_tolerance:
            # For backwards compatibility, wrap single result in tuple
            robust_result = fedavg.byzantine_robust_aggregate(
                client_weights, fedavg.config.byzantine_method
            )
            return robust_result, {
                "aggregation_method": "byzantine_robust",
                "security_enabled": False,
            }
        return fedavg.aggregate(client_weights, client_sizes)


def test_fedavg():
    print("Testing FedAvg Algorithm...")
    config = FedAvgConfig(
        convergence_threshold=1e-3,
        patience=3,
        byzantine_tolerance=True,
        byzantine_method="median",
    )
    fedavg = FedAvg(config)
    np.random.seed(42)
    # Simulate 3 clients with different data sizes and updates
    client_sizes = [50, 30, 20]
    updates = [
        {"w": np.ones((2, 2)) * 1.0},
        {"w": np.ones((2, 2)) * 2.0},
        {"w": np.ones((2, 2)) * 100.0},  # Byzantine
    ]
    # Test byzantine robust aggregation
    agg = fedavg.byzantine_robust_aggregate(updates, method="median")
    print(f"Median aggregation result: {agg['w']}")
    assert np.allclose(agg["w"], np.ones((2, 2)) * 2.0)
    # Test weighted averaging (now returns tuple)
    agg2_result, security_report = fedavg.aggregate(
        updates, client_sizes, enable_security=False
    )
    print(f"Weighted average result: {agg2_result['w']}")
    print(f"Security report: {security_report}")
    # Simulate convergence
    for loss in [1.0, 0.8, 0.7, 0.69, 0.68, 0.68]:
        if fedavg.check_convergence(loss):
            print(f"Converged at loss {loss}")
            break
    # Test AsyncAggregator
    t = [0.0]
    now = lambda: t[0]
    async_agg = AsyncAggregator(min_updates=2, timeout_seconds=5.0, time_provider=now)
    async_agg.start_round("r1")
    async_agg.submit_update({"w": np.array([1.0, 1.0])}, data_size=10)
    assert not async_agg.is_ready()
    async_agg.submit_update({"w": np.array([3.0, 3.0])}, data_size=30)
    assert async_agg.is_ready()
    agg_async_result, _ = async_agg.aggregate(fedavg)
    assert np.allclose(agg_async_result["w"], np.array([2.5, 2.5]))
    print("🎉 FedAvg algorithm test completed successfully!")
    return True


if __name__ == "__main__":
    test_fedavg()
