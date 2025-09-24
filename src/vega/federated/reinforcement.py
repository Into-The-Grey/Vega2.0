"""
Federated Reinforcement Learning (FRL) for simple multi-armed bandits.

Design goals:
- No heavy dependencies (standard library only)
- Deterministic when seed is set
- Federated rounds with local policy-gradient (REINFORCE) and FedAvg aggregation
- Usable directly via run_federated_bandit()

This is intentionally simple and fast to run in tests.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict


@dataclass
class BanditEnv:
    """Bernoulli multi-armed bandit environment.

    Each arm i yields reward 1 with probability probs[i], else 0.
    """

    probs: List[float]

    def step(self, action: int) -> int:
        p = self.probs[action]
        return 1 if random.random() < p else 0

    @property
    def n_arms(self) -> int:
        return len(self.probs)


class SoftmaxPolicy:
    """Softmax policy over arms with REINFORCE update.

    theta: list of real-valued preferences, one per arm.
    pi(a) = exp(theta[a]) / sum(exp(theta))
    """

    def __init__(self, n_arms: int, theta: Optional[List[float]] = None):
        self.n_arms = n_arms
        self.theta = [0.0] * n_arms if theta is None else list(theta)

    def probs(self) -> List[float]:
        mx = max(self.theta)
        exps = [math.exp(t - mx) for t in self.theta]
        s = sum(exps)
        return [e / s for e in exps]

    def act(self) -> int:
        ps = self.probs()
        r = random.random()
        cum = 0.0
        for i, p in enumerate(ps):
            cum += p
            if r <= cum:
                return i
        return len(ps) - 1  # numerical fallback

    def update_reinforce(self, action: int, reward: float, lr: float, baseline: float):
        """One-step REINFORCE update with baseline.

        grad log pi(a) for softmax: 1[a] - pi, applied to theta vector.
        """
        ps = self.probs()
        advantage = reward - baseline
        for i in range(self.n_arms):
            indicator = 1.0 if i == action else 0.0
            self.theta[i] += lr * advantage * (indicator - ps[i])


@dataclass
class LocalFRLConfig:
    steps_per_round: int = 200
    lr: float = 0.1
    baseline_momentum: float = 0.9  # running average baseline


def local_train_bandit(
    env: BanditEnv, policy: SoftmaxPolicy, cfg: LocalFRLConfig
) -> Dict[str, float]:
    """Run local bandit training for a number of steps, in-place update of policy.

    Returns metrics: avg_reward, baseline
    """
    avg_reward = 0.0
    baseline = 0.0
    b_m = cfg.baseline_momentum
    for t in range(cfg.steps_per_round):
        a = policy.act()
        r = env.step(a)
        # Update baseline (EMA)
        baseline = b_m * baseline + (1 - b_m) * r
        policy.update_reinforce(a, r, cfg.lr, baseline)
        # Track average reward
        avg_reward += (r - avg_reward) / (t + 1)
    return {"avg_reward": avg_reward, "baseline": baseline}


def fedavg_thetas(
    thetas: List[List[float]], weights: Optional[List[float]] = None
) -> List[float]:
    """Federated averaging over theta vectors.

    thetas: list of client parameter lists (same length)
    weights: optional list of client weights (sum>0). If None, equal weighting.
    """
    if not thetas:
        raise ValueError("No thetas provided")
    n_clients = len(thetas)
    n_arms = len(thetas[0])
    if weights is None:
        weights = [1.0] * n_clients
    wsum = sum(weights)
    if wsum <= 0:
        raise ValueError("Sum of weights must be positive")
    out = [0.0] * n_arms
    for c in range(n_clients):
        w = weights[c] / wsum
        th = thetas[c]
        if len(th) != n_arms:
            raise ValueError("Theta length mismatch among clients")
        for i in range(n_arms):
            out[i] += w * th[i]
    return out


def run_federated_bandit(
    client_envs: List[BanditEnv],
    rounds: int = 10,
    local_steps_per_round: int = 200,
    lr: float = 0.1,
    seed: Optional[int] = 42,
) -> Dict[str, object]:
    """Run federated bandit training across clients with FedAvg.

    Returns:
        dict with keys:
          - history: list of dicts with round metrics
          - final_theta: global theta list
          - n_arms: number of arms
    """
    if seed is not None:
        random.seed(seed)

    if not client_envs:
        raise ValueError("At least one client environment required")
    n_clients = len(client_envs)
    n_arms = client_envs[0].n_arms
    # Validate same arm count
    for env in client_envs:
        if env.n_arms != n_arms:
            raise ValueError("All client envs must have the same number of arms")

    # Initialize global policy
    global_theta = [0.0] * n_arms

    history: List[Dict[str, float]] = []
    for rnd in range(rounds):
        # Distribute global policy
        local_thetas: List[List[float]] = []
        rewards_this_round: List[float] = []
        for c in range(n_clients):
            policy = SoftmaxPolicy(n_arms, theta=global_theta)
            cfg = LocalFRLConfig(steps_per_round=local_steps_per_round, lr=lr)
            metrics = local_train_bandit(client_envs[c], policy, cfg)
            local_thetas.append(policy.theta)
            rewards_this_round.append(metrics["avg_reward"])

        # Aggregate
        global_theta = fedavg_thetas(local_thetas)
        avg_reward = sum(rewards_this_round) / len(rewards_this_round)
        history.append({"round": rnd + 1, "avg_reward": avg_reward})

    return {"history": history, "final_theta": global_theta, "n_arms": n_arms}


def demo() -> None:
    """Small demo with 3 heterogeneous clients."""
    client_envs = [
        BanditEnv([0.1, 0.3, 0.7]),
        BanditEnv([0.2, 0.5, 0.6]),
        BanditEnv([0.05, 0.4, 0.8]),
    ]
    result = run_federated_bandit(
        client_envs, rounds=8, local_steps_per_round=100, lr=0.1, seed=123
    )
    print("Federated RL (bandit) training history:")
    for r in result["history"]:
        print(f"Round {r['round']:2d} - avg reward: {r['avg_reward']:.3f}")


if __name__ == "__main__":
    demo()
