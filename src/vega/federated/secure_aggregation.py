"""
Secure Aggregation Protocol for Federated Learning

Implements additive secret sharing for secure multi-party aggregation of model weights.
Supports 2-3 participants (personal/family scale) and integrates with ModelWeights.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class SecureAggregationConfig:
    num_participants: int = 2
    random_seed: int = 42


class SecureAggregation:
    def __init__(self, config: SecureAggregationConfig):
        self.config = config
        np.random.seed(config.random_seed)

    def generate_shares(self, update: np.ndarray) -> List[np.ndarray]:
        """
        Split the update into num_participants additive shares.
        Returns a list of shares (one per participant).
        """
        shares = []
        for _ in range(self.config.num_participants - 1):
            share = np.random.randn(*update.shape)
            shares.append(share)
        # Last share ensures sum of shares equals update
        last_share = update - sum(shares)
        shares.append(last_share)
        return shares

    def aggregate_shares(self, shares_list: List[List[np.ndarray]]) -> np.ndarray:
        """
        Aggregate shares from all participants to reconstruct the sum of updates.
        shares_list: List of lists, each inner list is shares for one participant.
        """
        # Transpose: group shares by share index
        grouped = list(zip(*shares_list))
        # Sum each group to reconstruct each participant's update
        updates = [sum(group) for group in grouped]
        # Aggregate all updates
        return sum(updates)


def test_secure_aggregation():
    print("Testing Secure Aggregation Protocol...")
    config = SecureAggregationConfig(num_participants=3, random_seed=123)
    sa = SecureAggregation(config)
    np.random.seed(123)
    # Simulate 3 clients with random updates
    updates = [np.random.randn(5) for _ in range(3)]
    # Each client generates shares
    all_shares = [sa.generate_shares(update) for update in updates]
    # Each participant receives one share from each client
    shares_for_participants = list(zip(*all_shares))  # 3 lists of 3 shares
    # Each participant sums their received shares
    participant_sums = [sum(shares) for shares in shares_for_participants]
    # Server aggregates participant sums to reconstruct total update
    reconstructed = sum(participant_sums)
    expected = sum(updates)
    print(f"Expected sum: {expected}")
    print(f"Reconstructed sum: {reconstructed}")
    assert np.allclose(reconstructed, expected, atol=1e-6)
    print("🎉 Secure Aggregation test completed successfully!")
    return True


if __name__ == "__main__":
    test_secure_aggregation()
