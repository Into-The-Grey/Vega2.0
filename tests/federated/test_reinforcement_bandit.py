import math
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.vega.federated.reinforcement import BanditEnv, run_federated_bandit


def test_federated_bandit_reward_improves_and_is_deterministic():
    # Three heterogeneous clients with the same number of arms
    client_envs = [
        BanditEnv([0.1, 0.3, 0.7]),
        BanditEnv([0.2, 0.5, 0.6]),
        BanditEnv([0.05, 0.4, 0.8]),
    ]

    result = run_federated_bandit(
        client_envs, rounds=10, local_steps_per_round=150, lr=0.1, seed=123
    )

    history = result["history"]
    assert len(history) == 10

    # Check that the average reward improves over the first few rounds
    # Allow for small stochastic fluctuations but expect upward trend overall
    start = history[0]["avg_reward"]
    mid = history[4]["avg_reward"]
    end = history[-1]["avg_reward"]

    assert mid > start - 0.05  # not worse than a small tolerance
    assert end > start  # improved by the end

    # Determinism: running again with the same seed yields the same history
    result2 = run_federated_bandit(
        client_envs, rounds=10, local_steps_per_round=150, lr=0.1, seed=123
    )
    history2 = result2["history"]

    assert [round(h["avg_reward"], 6) for h in history] == [
        round(h["avg_reward"], 6) for h in history2
    ]

    # Validate theta length matches arms
    assert len(result["final_theta"]) == 3


if __name__ == "__main__":
    test_federated_bandit_reward_improves_and_is_deterministic()
    print("✓ All tests passed")
