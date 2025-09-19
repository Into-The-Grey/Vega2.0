"""
Adaptive Learning Rate System for Federated Learning

Supports performance-based, data-based, and convergence-based learning rate adaptation.
Modular and pluggable into any federated learning workflow.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveLRConfig:
    base_lr: float = 0.01
    min_lr: float = 1e-5
    max_lr: float = 1.0
    performance_patience: int = 3
    improvement_threshold: float = 1e-3
    decay_factor: float = 0.5
    increase_factor: float = 1.1
    mode: str = "performance"  # "performance", "convergence", "data"


class AdaptiveLearningRate:
    def __init__(self, config: AdaptiveLRConfig):
        self.config = config
        self.lr = config.base_lr
        self.history: List[float] = []
        self.patience_counter = 0

    def update(self, metrics: Dict[str, Any]) -> float:
        """
        Update learning rate based on provided metrics.
        metrics: should include 'loss' and optionally 'data_size', 'accuracy', etc.
        """
        if self.config.mode == "performance":
            return self._performance_based(metrics)
        elif self.config.mode == "convergence":
            return self._convergence_based(metrics)
        elif self.config.mode == "data":
            return self._data_based(metrics)
        else:
            logger.warning(
                f"Unknown adaptive LR mode: {self.config.mode}, using base_lr."
            )
            return self.lr

    def _performance_based(self, metrics: Dict[str, Any]) -> float:
        loss = metrics.get("loss")
        if loss is None:
            return self.lr
        self.history.append(loss)
        if len(self.history) < 2:
            return self.lr
        improvement = self.history[-2] - self.history[-1]
        if improvement < self.config.improvement_threshold:
            self.patience_counter += 1
            if self.patience_counter >= self.config.performance_patience:
                self.lr = max(self.lr * self.config.decay_factor, self.config.min_lr)
                self.patience_counter = 0
        else:
            self.lr = min(self.lr * self.config.increase_factor, self.config.max_lr)
            self.patience_counter = 0
        return self.lr

    def _convergence_based(self, metrics: Dict[str, Any]) -> float:
        # Example: decay LR if loss plateaus, increase if loss drops sharply
        loss = metrics.get("loss")
        if loss is None:
            return self.lr
        self.history.append(loss)
        if len(self.history) < 2:
            return self.lr
        improvement = self.history[-2] - self.history[-1]
        if improvement < self.config.improvement_threshold:
            self.lr = max(self.lr * self.config.decay_factor, self.config.min_lr)
        else:
            self.lr = min(self.lr * self.config.increase_factor, self.config.max_lr)
        return self.lr

    def _data_based(self, metrics: Dict[str, Any]) -> float:
        # Example: scale LR by data size (more data, higher LR)
        data_size = metrics.get("data_size", 1)
        self.lr = min(self.config.base_lr * np.log1p(data_size), self.config.max_lr)
        return self.lr


def test_adaptive_lr():
    print("Testing Adaptive Learning Rate System...")
    config = AdaptiveLRConfig(
        base_lr=0.1,
        min_lr=0.01,
        max_lr=1.0,
        performance_patience=2,
        improvement_threshold=0.05,
    )
    alr = AdaptiveLearningRate(config)
    # Simulate loss decreasing, then plateauing
    losses = [1.0, 0.8, 0.7, 0.68, 0.67, 0.66, 0.66, 0.66]
    lrs = []
    for loss in losses:
        lr = alr.update({"loss": loss})
        lrs.append(lr)
        print(f"Loss: {loss:.4f}, LR: {lr:.4f}")
    assert min(lrs) >= config.min_lr and max(lrs) <= config.max_lr
    print("🎉 Adaptive Learning Rate test completed successfully!")
    return True


if __name__ == "__main__":
    test_adaptive_lr()
