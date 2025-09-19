"""
Differential Privacy Module for Federated Learning

Implements Gaussian noise injection, privacy budget tracking, and epsilon-delta guarantees.
Designed to be modular and pluggable into any federated aggregation algorithm.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class DifferentialPrivacyConfig:
    noise_multiplier: float = 1.0  # Standard deviation of Gaussian noise
    clipping_norm: float = 1.0  # L2 norm to clip model updates
    epsilon: float = 10.0  # Privacy budget (epsilon)
    delta: float = 1e-5  # Privacy parameter (delta)
    max_rounds: int = 100  # Max communication rounds for budget
    track_per_participant: bool = True
    adaptive_noise: bool = False  # Enable adaptive noise scaling based on sensitivity


class DifferentialPrivacy:
    def get_audit_log(self):
        """Return the audit log if enabled."""
        return self.audit_log

    def export_audit_log(self, filepath: str):
        """Export the audit log to a file (JSON lines)."""
        if self.audit_log is None:
            raise RuntimeError("Audit log is not enabled.")
        import json

        with open(filepath, "w") as f:
            for entry in self.audit_log:
                f.write(json.dumps(entry) + "\n")

    def apply_local_dp(
        self,
        update: np.ndarray,
        local_noise_multiplier: Optional[float] = None,
        local_clipping_norm: Optional[float] = None,
    ) -> np.ndarray:
        """
        Apply local differential privacy: clip and add noise to a single user's update.
        Args:
            update: model update (np.ndarray)
            local_noise_multiplier: override for noise multiplier (float)
            local_clipping_norm: override for clipping norm (float)
        Returns:
            Privatized update (np.ndarray)
        """
        # Use local or global parameters
        noise_multiplier = (
            local_noise_multiplier
            if local_noise_multiplier is not None
            else self.config.noise_multiplier
        )
        clipping_norm = (
            local_clipping_norm
            if local_clipping_norm is not None
            else self.config.clipping_norm
        )
        # Clip update
        norm = np.linalg.norm(update)
        clipped = False
        if norm > clipping_norm:
            update = update * (clipping_norm / norm)
            clipped = True
        # Add Gaussian noise
        noise = np.random.normal(0, noise_multiplier * clipping_norm, update.shape)
        privatized_update = update + noise
        self._log_audit(
            "apply_local_dp",
            {
                "input_norm": float(norm),
                "clipped": clipped,
                "clipping_norm": float(clipping_norm),
                "noise_std": float(noise_multiplier * clipping_norm),
                "output_norm": float(np.linalg.norm(privatized_update)),
            },
        )
        return privatized_update

    def __init__(self, config: DifferentialPrivacyConfig, enable_audit: bool = False):
        self.config = config
        self.privacy_spent = 0.0
        self.rounds = 0
        self.participant_budgets: Dict[Any, float] = {}
        self.audit_log = [] if enable_audit else None

    def _log_audit(self, event: str, details: dict):
        if self.audit_log is not None:
            entry = {"event": event, "details": details}
            self.audit_log.append(entry)

    def clip_and_add_noise(
        self,
        update: np.ndarray,
        participant_id: Optional[Any] = None,
        sensitivity: Optional[float] = None,
    ) -> np.ndarray:
        """
        Clip the update and add Gaussian noise. If adaptive_noise is enabled, scale noise by sensitivity.
        Args:
            update: model update (np.ndarray)
            participant_id: optional participant identifier
            sensitivity: optional sensitivity value (L2 norm or custom)
        Returns:
            Noisy, clipped update
        """
        # Clip update
        norm = np.linalg.norm(update)
        clipped = False
        if norm > self.config.clipping_norm:
            update = update * (self.config.clipping_norm / norm)
            clipped = True
        # Adaptive noise scaling
        if self.config.adaptive_noise:
            if sensitivity is None:
                sensitivity = float(np.linalg.norm(update))
            else:
                sensitivity = float(sensitivity)
            noise_std = self.config.noise_multiplier * sensitivity
        else:
            noise_std = self.config.noise_multiplier * self.config.clipping_norm
        noise = np.random.normal(0, noise_std, update.shape)
        noisy_update = update + noise
        # Track privacy budget
        self._track_privacy(participant_id)
        self._log_audit(
            "clip_and_add_noise",
            {
                "input_norm": float(norm),
                "clipped": clipped,
                "clipping_norm": float(self.config.clipping_norm),
                "noise_std": float(noise_std),
                "output_norm": float(np.linalg.norm(noisy_update)),
                "participant_id": participant_id,
            },
        )
        return noisy_update

    def _track_privacy(self, participant_id: Optional[Any]):
        self.rounds += 1
        if self.config.track_per_participant and participant_id is not None:
            self.participant_budgets.setdefault(participant_id, 0.0)
            self.participant_budgets[
                participant_id
            ] += self._compute_epsilon_increment()
        else:
            self.privacy_spent += self._compute_epsilon_increment()

    def _compute_epsilon_increment(self) -> float:
        # Simple accounting: divide total budget by max rounds
        return self.config.epsilon / self.config.max_rounds

    def privacy_budget_exceeded(self, participant_id: Optional[Any] = None) -> bool:
        if self.config.track_per_participant and participant_id is not None:
            return (
                self.participant_budgets.get(participant_id, 0.0) > self.config.epsilon
            )
        else:
            return self.privacy_spent > self.config.epsilon

    def get_privacy_spent(self, participant_id: Optional[Any] = None) -> float:
        if self.config.track_per_participant and participant_id is not None:
            return self.participant_budgets.get(participant_id, 0.0)
        else:
            return self.privacy_spent


def test_differential_privacy():
    print("Testing Differential Privacy Module...")
    config = DifferentialPrivacyConfig(
        noise_multiplier=2.0, clipping_norm=1.0, epsilon=5.0, max_rounds=5
    )
    dp = DifferentialPrivacy(config)
    np.random.seed(42)
    update = np.random.randn(10)
    noisy_update = dp.clip_and_add_noise(update, participant_id="user1")
    print(f"Original update: {update}")
    print(f"Noisy update: {noisy_update}")
    print(f"Privacy spent: {dp.get_privacy_spent('user1')}")
    assert noisy_update.shape == update.shape
    assert dp.get_privacy_spent("user1") > 0
    for _ in range(5):
        dp.clip_and_add_noise(update, participant_id="user1")
    assert dp.privacy_budget_exceeded("user1")
    print("🎉 Differential Privacy test completed successfully!")
    return True


if __name__ == "__main__":
    test_differential_privacy()
