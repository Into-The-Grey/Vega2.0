"""Autonomous parameter optimization for Vega."""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Awaitable, Callable, Dict, Mapping, MutableMapping

from .monitoring import LOG_DIR, PerformanceMonitor
from .parameters import ParameterStateManager, TUNABLE_KEYS


class AutonomousSelfOptimizer:
    """Runs closed-loop experiments to tune generation parameters."""

    def __init__(
        self,
        evaluator: Callable[[Mapping[str, str]], Awaitable[float]],
        monitor: PerformanceMonitor | None = None,
        param_manager: ParameterStateManager | None = None,
        exploration_sigma: float = 0.05,
    ) -> None:
        self.monitor = monitor or PerformanceMonitor()
        self.param_manager = param_manager or ParameterStateManager()
        self.evaluator = evaluator
        self.sigma = exploration_sigma
        self.history_file = LOG_DIR / "experiments.jsonl"
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def run_cycle(
        self, iterations: int = 5, improvement_threshold: float = 0.01
    ) -> Dict[str, str]:
        """Execute a full optimization loop."""

        async with self._lock:
            baseline = self.param_manager.current_parameters()
            if baseline:
                self.param_manager.backup_current()
            baseline_score = await self.evaluator(baseline)
            await self.monitor.record("baseline", "score", baseline_score)

            best_score = baseline_score
            best_params = baseline.copy()

            for iteration in range(iterations):
                candidate = self._perturb(best_params or baseline)
                self.param_manager.apply_parameters(candidate)
                score = await self.evaluator(candidate)
                await self._log_experiment(iteration, candidate, score)

                if score > best_score * (1 + improvement_threshold):
                    best_score = score
                    best_params = candidate.copy()
                    await self.monitor.record(
                        "optimizer", "accepted_score", score, iteration=iteration
                    )
                else:
                    # revert to best known parameters
                    self.param_manager.apply_parameters(best_params)
                    await self.monitor.record(
                        "optimizer", "rejected_score", score, iteration=iteration
                    )

            self.param_manager.apply_parameters(best_params)
            await self.monitor.record("optimizer", "final_score", best_score)
            return best_params

    def _perturb(self, base: Mapping[str, str]) -> Dict[str, str]:
        candidate: Dict[str, str] = dict(base)
        rng = random.Random(time.time())
        for key in TUNABLE_KEYS:
            current = float(candidate.get(key, self._default_for(key)))
            noise = rng.gauss(0, self.sigma * max(abs(current), 1e-3))
            updated = max(0.0, current + noise)
            if key in {"GEN_TOP_K"}:
                updated = max(1.0, updated)
            candidate[key] = str(round(updated, 4))
        return candidate

    async def _log_experiment(
        self, iteration: int, params: Mapping[str, str], score: float
    ) -> None:
        payload = {
            "iteration": iteration,
            "score": score,
            "params": dict(params),
            "timestamp": time.time(),
        }
        with self.history_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + os.linesep)

    @staticmethod
    def _default_for(key: str) -> float:
        defaults = {
            "GEN_TEMPERATURE": 0.7,
            "GEN_TOP_P": 0.9,
            "GEN_TOP_K": 40,
            "GEN_REPEAT_PENALTY": 1.1,
            "GEN_PRESENCE_PENALTY": 0.0,
            "GEN_FREQUENCY_PENALTY": 0.0,
        }
        return float(defaults.get(key, 0.0))
