#!/usr/bin/env python3
"""
Standalone Test Script for Federated Model Pruning Implementation
================================================================

This script validates the federated model pruning system without importing
the full Vega system, testing just the pruning modules directly.

Author: Vega2.0 Federated Learning Team
Date: September 2025
"""

import asyncio
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Import pruning modules directly
sys.path.insert(0, str(Path(__file__).parent / "src" / "vega" / "federated"))

from pruning import (
    PruningConfig,
    PruningType,
    SparsitySchedule,
    PruningCoordinator,
    StructuredPruning,
    UnstructuredPruning,
    FederatedDistillation,
    SparsityAggregator,
)

print("✅ Successfully imported pruning modules")


class TestModel(nn.Module):
    """Simple test model for pruning validation."""

    def __init__(self, input_size=784, hidden_size=512, num_classes=10):
        super(TestModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class TestCNN(nn.Module):
    """Simple CNN for structured pruning tests."""

    def __init__(self, num_classes=10):
        super(TestCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


async def test_basic_pruning():
    """Test basic pruning functionality."""
    print("\n=== Testing Basic Pruning Functionality ===")

    # Test unstructured magnitude pruning
    config = PruningConfig(
        pruning_type=PruningType.UNSTRUCTURED_MAGNITUDE, target_sparsity=0.7
    )

    model = TestModel()
    original_sparsity = sum((p == 0).sum().item() for p in model.parameters()) / sum(
        p.numel() for p in model.parameters()
    )

    pruner = UnstructuredPruning(config)
    pruned_model, metrics = await pruner.prune_model(
        model, round_num=1, total_rounds=10
    )

    print(f"✅ Unstructured pruning completed")
    print(f"   - Original sparsity: {original_sparsity:.3f}")
    print(f"   - Final sparsity: {metrics.sparsity_ratio:.3f}")
    print(f"   - Target sparsity: {config.target_sparsity}")
    print(f"   - Accuracy drop: {metrics.accuracy_drop:.3f}")
    print(f"   - Compression time: {metrics.compression_time:.3f}s")

    if abs(metrics.sparsity_ratio - config.target_sparsity) < 0.1:
        print("✅ Sparsity target achieved successfully")
        return True
    else:
        print(
            f"❌ Sparsity mismatch: got {metrics.sparsity_ratio:.3f}, expected {config.target_sparsity}"
        )
        return False


async def test_structured_pruning():
    """Test structured pruning functionality."""
    print("\n=== Testing Structured Pruning Functionality ===")

    # Test structured channel pruning
    config = PruningConfig(
        pruning_type=PruningType.STRUCTURED_CHANNEL, target_sparsity=0.5
    )

    model = TestCNN()
    original_params = sum(p.numel() for p in model.parameters())

    pruner = StructuredPruning(config)
    pruned_model, metrics = await pruner.prune_model(
        model, round_num=1, total_rounds=10
    )

    final_params = sum(p.numel() for p in pruned_model.parameters())

    print(f"✅ Structured pruning completed")
    print(f"   - Original parameters: {original_params}")
    print(f"   - Final parameters: {final_params}")
    print(
        f"   - Parameter reduction: {(original_params - final_params) / original_params:.3f}"
    )
    print(f"   - Model size reduction: {metrics.model_size_reduction:.3f}")
    print(f"   - Accuracy drop: {metrics.accuracy_drop:.3f}")
    print(f"   - Inference speedup: {metrics.inference_speedup:.2f}x")

    if metrics.model_size_reduction > 0.1:  # Some reduction achieved
        print("✅ Structured pruning achieved model compression")
        return True
    else:
        print(f"❌ Insufficient compression: {metrics.model_size_reduction:.3f}")
        return False


async def test_knowledge_distillation():
    """Test knowledge distillation functionality."""
    print("\n=== Testing Knowledge Distillation ===")

    teacher_model = TestModel(hidden_size=512)
    student_model = TestModel(hidden_size=256)

    distillation = FederatedDistillation(temperature=4.0, alpha=0.7)

    distilled_student, metrics = await distillation.distill_knowledge(
        teacher_model=teacher_model, student_model=student_model, num_epochs=2
    )

    print(f"✅ Knowledge distillation completed")
    print(f"   - Teacher accuracy: {metrics['teacher_accuracy']:.3f}")
    print(f"   - Student accuracy: {metrics['student_accuracy']:.3f}")
    print(f"   - Knowledge retention: {metrics['knowledge_retention']:.3f}")
    print(f"   - Distillation loss: {metrics['distillation_loss']:.4f}")
    print(f"   - Distillation time: {metrics['distillation_time']:.2f}s")

    if metrics["knowledge_retention"] > 0.5:  # Reasonable retention
        print("✅ Knowledge distillation successful")
        return True
    else:
        print(f"❌ Poor knowledge retention: {metrics['knowledge_retention']:.3f}")
        return False


async def test_sparsity_aggregation():
    """Test sparsity-aware model aggregation."""
    print("\n=== Testing Sparsity-Aware Aggregation ===")

    # Create models with different sparsities
    models = []
    weights = [1.0, 0.8, 1.2]
    masks = []

    for i in range(3):
        model = TestModel()

        # Apply different sparsity levels
        sparsity = 0.2 + i * 0.2  # 0.2, 0.4, 0.6

        mask = {}
        for name, param in model.named_parameters():
            if "weight" in name:
                param_mask = torch.rand_like(param) >= sparsity
                param.data *= param_mask.float()
                mask[name] = param_mask
            else:
                mask[name] = torch.ones_like(param, dtype=torch.bool)

        models.append(model)
        masks.append(mask)

    global_model = TestModel()
    aggregator = SparsityAggregator()

    participant_data = list(zip(models, weights, masks))
    aggregated_model, metrics = await aggregator.aggregate_sparse_models(
        participant_data, global_model
    )

    print(f"✅ Sparsity-aware aggregation completed")
    print(f"   - Participants: {metrics['num_participants']}")
    print(f"   - Average sparsity: {metrics['average_sparsity']:.3f}")
    print(f"   - Min sparsity: {metrics['min_sparsity']:.3f}")
    print(f"   - Max sparsity: {metrics['max_sparsity']:.3f}")
    print(f"   - Sparsity variance: {metrics['sparsity_variance']:.6f}")
    print(f"   - Aggregation time: {metrics['aggregation_time']:.3f}s")

    if metrics["num_participants"] == 3:
        print("✅ Aggregation successful")
        return True
    else:
        print(f"❌ Aggregation failed")
        return False


async def test_pruning_coordinator():
    """Test the main pruning coordinator."""
    print("\n=== Testing Pruning Coordinator ===")

    config = PruningConfig(
        pruning_type=PruningType.UNSTRUCTURED_MAGNITUDE,
        target_sparsity=0.6,
        pruning_frequency=2,
        recovery_threshold=0.1,
        distillation_enabled=True,
    )

    coordinator = PruningCoordinator(config)

    # Create test scenario
    global_model = TestModel()
    participant_models = [TestModel() for _ in range(3)]
    participant_weights = [1.0, 1.0, 1.0]

    # Test multiple rounds
    successful_rounds = 0
    for round_num in range(1, 7):  # 6 rounds
        global_model, updated_participants, metrics = (
            await coordinator.coordinate_pruning_round(
                participant_models=participant_models,
                participant_weights=participant_weights,
                global_model=global_model,
                round_num=round_num,
                total_rounds=6,
            )
        )

        if not metrics.get("skipped", False):
            successful_rounds += 1
            print(f"   Round {round_num}: {metrics['average_sparsity']:.3f} sparsity")
            participant_models = updated_participants
        else:
            print(f"   Round {round_num}: Skipped")

    # Get summary
    summary = coordinator.get_pruning_summary()

    print(f"✅ Coordinator test completed")
    print(f"   - Successful rounds: {successful_rounds}")
    print(f"   - Final sparsity: {summary['final_average_sparsity']:.3f}")
    print(f"   - Average accuracy drop: {summary['average_accuracy_drop']:.3f}")
    print(f"   - Total coordination time: {summary['total_coordination_time']:.2f}s")

    if successful_rounds > 0:
        print("✅ Coordinator functioning correctly")
        return True
    else:
        print("❌ No successful rounds")
        return False


async def test_sparsity_schedules():
    """Test different sparsity scheduling strategies."""
    print("\n=== Testing Sparsity Schedules ===")

    schedules_to_test = [
        SparsitySchedule.CONSTANT,
        SparsitySchedule.LINEAR,
        SparsitySchedule.EXPONENTIAL,
        SparsitySchedule.POLYNOMIAL,
        SparsitySchedule.COSINE,
    ]

    target_sparsity = 0.8
    total_rounds = 5

    for schedule in schedules_to_test:
        config = PruningConfig(
            pruning_type=PruningType.UNSTRUCTURED_MAGNITUDE,
            target_sparsity=target_sparsity,
            schedule=schedule,
        )

        pruner = UnstructuredPruning(config)

        # Test progression
        start_sparsity = pruner.get_current_sparsity_target(1, total_rounds)
        end_sparsity = pruner.get_current_sparsity_target(total_rounds, total_rounds)

        print(f"   {schedule.value}: {start_sparsity:.3f} → {end_sparsity:.3f}")

        # Validate that we reach target
        if abs(end_sparsity - target_sparsity) > 0.01:
            print(f"❌ Schedule {schedule.value} doesn't reach target")
            return False

    print("✅ All sparsity schedules working correctly")
    return True


async def run_comprehensive_test():
    """Run comprehensive test suite."""
    print("🚀 Starting Federated Model Pruning Tests")
    print("=" * 60)

    test_functions = [
        ("Basic Pruning", test_basic_pruning),
        ("Structured Pruning", test_structured_pruning),
        ("Knowledge Distillation", test_knowledge_distillation),
        ("Sparsity Aggregation", test_sparsity_aggregation),
        ("Sparsity Schedules", test_sparsity_schedules),
        ("Pruning Coordinator", test_pruning_coordinator),
    ]

    results = {}
    start_time = time.time()

    for test_name, test_func in test_functions:
        try:
            print(f"\n{'=' * 15} {test_name} {'=' * 15}")
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            results[test_name] = False
            print(f"❌ {test_name} failed with error: {e}")

    total_time = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")

    print("-" * 60)
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"EXECUTION TIME: {total_time:.2f} seconds")

    if passed == total:
        print(
            "\n🎉 ALL TESTS PASSED! Federated Model Pruning implementation working correctly!"
        )
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Implementation needs review.")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_comprehensive_test())
    exit_code = 0 if success else 1
    exit(exit_code)
