#!/usr/bin/env python3
"""Direct integration tests for the adaptive pruning orchestrator."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.vega.federated.pruning_orchestrator import (  # noqa: E402
    AdaptivePruningOrchestrator,
    ParticipantCapability,
    PruningStrategy,
    SparsityScheduleConfig,
)


class TinyClassifier(nn.Module):
    """Small fully connected network used for simulated pruning rounds."""

    def __init__(
        self, input_size: int = 128, hidden_size: int = 64, output_size: int = 10
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(tensor)))


def _register_participants(
    orchestrator: AdaptivePruningOrchestrator,
) -> List[Tuple[str, ParticipantCapability]]:
    """Register a heterogeneous participant set for the scenario."""

    participants = [
        (
            "edge_gpu",
            ParticipantCapability.HIGH,
            PruningStrategy.AGGRESSIVE,
            {"max_sparsity": 0.9},
        ),
        (
            "industrial_gateway",
            ParticipantCapability.MEDIUM,
            PruningStrategy.BALANCED,
            {"max_sparsity": 0.75},
        ),
        (
            "iot_sensor",
            ParticipantCapability.LOW,
            PruningStrategy.CONSERVATIVE,
            {"max_sparsity": 0.45},
        ),
        (
            "mobile_tablet",
            ParticipantCapability.VARIABLE,
            PruningStrategy.ADAPTIVE,
            {"max_sparsity": 0.65},
        ),
    ]

    for participant_id, capability, strategy, kwargs in participants:
        orchestrator.register_participant(
            participant_id, capability, strategy, **kwargs
        )

    return [(pid, capability) for pid, capability, *_ in participants]


async def _execute_rounds(
    orchestrator: AdaptivePruningOrchestrator, rounds: int = 4
) -> List[Dict[str, float]]:
    """Run a handful of orchestration rounds and capture summaries."""

    participant_models: Dict[str, nn.Module] = {
        participant_id: TinyClassifier()
        for participant_id in orchestrator.participant_profiles
    }
    global_model = TinyClassifier()

    summaries: List[Dict[str, float]] = []
    for round_num in range(1, rounds + 1):
        result = await orchestrator.orchestrate_pruning_round(
            round_num=round_num,
            total_rounds=rounds,
            participant_models=participant_models,
            global_model=global_model,
        )

        perf_summary = result["performance_summary"]
        participant_metrics = result["participant_metrics"]

        # Structural assertions for the orchestration result payload.
        assert isinstance(perf_summary, dict)
        assert isinstance(participant_metrics, list) and participant_metrics
        assert set(perf_summary.keys()) >= {
            "average_accuracy",
            "average_sparsity",
            "struggling_participants",
        }
        assert len(participant_metrics) == len(orchestrator.participant_profiles)

        summaries.append(perf_summary)

    return summaries


async def run_direct_integration_test(rounds: int = 4) -> Dict[str, float]:
    """Coordinate the full direct integration scenario."""

    schedule = SparsityScheduleConfig(
        initial_sparsity=0.1,
        final_sparsity=0.8,
        warmup_rounds=1,
        cooldown_rounds=1,
        adaptation_rate=0.12,
        performance_threshold=0.04,
        stability_threshold=0.01,
    )

    orchestrator = AdaptivePruningOrchestrator(schedule)
    registered = _register_participants(orchestrator)
    assert registered  # Ensure participant pool is non-empty.

    round_summaries = await _execute_rounds(orchestrator, rounds=rounds)
    orchestration_summary = orchestrator.get_orchestration_summary()

    # Validate longitudinal metrics from the orchestrator summary.
    assert orchestration_summary["total_rounds"] == rounds
    assert orchestration_summary["total_participants"] == len(registered)
    assert len(orchestration_summary["sparsity_progression"]) == rounds
    assert len(orchestration_summary["accuracy_progression"]) == rounds

    # Final sparsity should trend upward relative to the opening round.
    start_sparsity = round_summaries[0]["average_sparsity"]
    final_sparsity = round_summaries[-1]["average_sparsity"]
    assert final_sparsity > start_sparsity * 0.85

    # Ensure the orchestrator produced actionable monitoring output.
    assert orchestration_summary["adaptations_made"] >= rounds

    return {
        "start_sparsity": float(start_sparsity),
        "final_sparsity": float(final_sparsity),
        "participants": len(registered),
        "distillation_interventions": orchestration_summary[
            "distillation_interventions"
        ],
    }


async def run_all_tests() -> bool:
    """Entry point used by both pytest and manual execution."""

    torch.manual_seed(42)
    np.random.seed(42)

    summary = await run_direct_integration_test(rounds=4)

    print("\nDirect orchestrator integration summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    return True


if __name__ == "__main__":
    try:
        exit_code = 0 if asyncio.run(run_all_tests()) else 1
    except KeyboardInterrupt:  # pragma: no cover - manual interruption convenience
        exit_code = 130

    raise SystemExit(exit_code)
