#!/usr/bin/env python3
"""
Test script for Federated Model Pruning Implementation
=====================================================

This script validates the federated model pruning system including:
- Structured and unstructured pruning algorithms
- Knowledge distillation for pruned models
- Sparsity-aware aggregation
- Comprehensive metrics tracking

Author: Vega2.0 Federated Learning Team
Date: September 2025
"""

import asyncio
import sys
import time
import torch
import torch.nn as nn
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.vega.federated.pruning import (
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
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Attempting direct import...")
    from vega.federated.pruning import (
        PruningConfig,
        PruningType,
        SparsitySchedule,
        PruningCoordinator,
        StructuredPruning,
        UnstructuredPruning,
        FederatedDistillation,
        SparsityAggregator,
    )

    print("✅ Successfully imported pruning modules (direct import)")


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


async def test_pruning_config():
    """Test pruning configuration validation."""
    print("\n=== Testing Pruning Configuration ===")

    # Valid configuration
    try:
        config = PruningConfig(
            pruning_type=PruningType.UNSTRUCTURED_MAGNITUDE,
            target_sparsity=0.8,
            schedule=SparsitySchedule.LINEAR,
            pruning_frequency=3,
            recovery_threshold=0.05,
        )
        print("✅ Valid configuration created successfully")
        print(f"   - Pruning type: {config.pruning_type.value}")
        print(f"   - Target sparsity: {config.target_sparsity}")
        print(f"   - Schedule: {config.schedule.value}")
        print(f"   - Pruning frequency: {config.pruning_frequency}")
        print(f"   - Recovery threshold: {config.recovery_threshold}")
    except Exception as e:
        print(f"❌ Failed to create valid configuration: {e}")
        return False

    # Invalid configurations
    test_cases = [
        ({"target_sparsity": 1.5}, "target_sparsity > 1.0"),
        ({"target_sparsity": -0.1}, "target_sparsity < 0.0"),
        ({"recovery_threshold": 1.5}, "recovery_threshold > 1.0"),
        ({"pruning_frequency": 0}, "pruning_frequency <= 0"),
    ]

    for invalid_params, description in test_cases:
        try:
            base_config = {
                "pruning_type": PruningType.UNSTRUCTURED_MAGNITUDE,
                "target_sparsity": 0.5,
            }
            base_config.update(invalid_params)
            PruningConfig(**base_config)
            print(f"❌ Should have failed for {description}")
            return False
        except ValueError:
            print(f"✅ Correctly rejected invalid config: {description}")
        except Exception as e:
            print(f"❌ Unexpected error for {description}: {e}")
            return False

    return True


async def test_unstructured_pruning():
    """Test unstructured pruning algorithms."""
    print("\n=== Testing Unstructured Pruning ===")

    # Create test configuration
    config = PruningConfig(
        pruning_type=PruningType.UNSTRUCTURED_MAGNITUDE,
        target_sparsity=0.7,
        schedule=SparsitySchedule.LINEAR,
    )

    # Create test model
    model = TestModel()
    original_params = sum(p.numel() for p in model.parameters())

    # Initialize pruner
    pruner = UnstructuredPruning(config)

    try:
        # Test magnitude pruning
        pruned_model, metrics = await pruner.prune_model(
            model, round_num=5, total_rounds=10
        )

        print("✅ Magnitude pruning completed successfully")
        print(f"   - Original parameters: {original_params}")
        print(f"   - Sparsity ratio: {metrics.sparsity_ratio:.3f}")
        print(f"   - Model size reduction: {metrics.model_size_reduction:.3f}")
        print(f"   - Accuracy before: {metrics.accuracy_before:.3f}")
        print(f"   - Accuracy after: {metrics.accuracy_after:.3f}")
        print(f"   - Accuracy drop: {metrics.accuracy_drop:.3f}")
        print(f"   - Compression time: {metrics.compression_time:.3f}s")

        # Validate sparsity
        if abs(metrics.sparsity_ratio - 0.7) > 0.1:  # Allow some tolerance
            print(
                f"⚠️  Sparsity ratio {metrics.sparsity_ratio:.3f} deviates from target 0.7"
            )
        else:
            print("✅ Sparsity ratio matches target")

        # Test different pruning types
        test_types = [
            PruningType.UNSTRUCTURED_GRADIENT,
            PruningType.UNSTRUCTURED_RANDOM,
        ]

        for pruning_type in test_types:
            config.pruning_type = pruning_type
            type_pruner = UnstructuredPruning(config)

            test_model = TestModel()  # Fresh model
            pruned_model, metrics = await type_pruner.prune_model(
                test_model, round_num=1
            )

            print(
                f"✅ {pruning_type.value} pruning: {metrics.sparsity_ratio:.3f} sparsity"
            )

        return True

    except Exception as e:
        print(f"❌ Unstructured pruning test failed: {e}")
        return False


async def test_structured_pruning():
    """Test structured pruning algorithms."""
    print("\n=== Testing Structured Pruning ===")

    # Create test configuration
    config = PruningConfig(
        pruning_type=PruningType.STRUCTURED_CHANNEL,
        target_sparsity=0.5,
        schedule=SparsitySchedule.LINEAR,
    )

    # Create test model (CNN for better structured pruning)
    model = TestCNN()
    original_params = sum(p.numel() for p in model.parameters())

    # Initialize pruner
    pruner = StructuredPruning(config)

    try:
        # Test channel pruning
        pruned_model, metrics = await pruner.prune_model(
            model, round_num=3, total_rounds=10
        )

        print("✅ Channel pruning completed successfully")
        print(f"   - Original parameters: {original_params}")
        print(f"   - Sparsity ratio: {metrics.sparsity_ratio:.3f}")
        print(f"   - Model size reduction: {metrics.model_size_reduction:.3f}")
        print(f"   - Accuracy before: {metrics.accuracy_before:.3f}")
        print(f"   - Accuracy after: {metrics.accuracy_after:.3f}")
        print(f"   - Accuracy drop: {metrics.accuracy_drop:.3f}")
        print(f"   - Inference speedup: {metrics.inference_speedup:.2f}x")
        print(f"   - FLOPs reduction: {metrics.flops_reduction:.3f}")

        # Test filter pruning
        config.pruning_type = PruningType.STRUCTURED_FILTER
        filter_pruner = StructuredPruning(config)

        test_model = TestCNN()  # Fresh model
        pruned_model, filter_metrics = await filter_pruner.prune_model(
            test_model, round_num=1
        )

        print(f"✅ Filter pruning: {filter_metrics.sparsity_ratio:.3f} sparsity")

        return True

    except Exception as e:
        print(f"❌ Structured pruning test failed: {e}")
        return False


async def test_knowledge_distillation():
    """Test federated knowledge distillation."""
    print("\n=== Testing Knowledge Distillation ===")

    try:
        # Create teacher and student models
        teacher_model = TestModel(hidden_size=512)  # Larger teacher
        student_model = TestModel(hidden_size=256)  # Smaller student

        # Initialize distillation
        distillation = FederatedDistillation(temperature=4.0, alpha=0.7)

        # Perform distillation
        distilled_student, metrics = await distillation.distill_knowledge(
            teacher_model=teacher_model, student_model=student_model, num_epochs=3
        )

        print("✅ Knowledge distillation completed successfully")
        print(f"   - Distillation loss: {metrics['distillation_loss']:.4f}")
        print(f"   - Distillation time: {metrics['distillation_time']:.2f}s")
        print(f"   - Student accuracy: {metrics['student_accuracy']:.3f}")
        print(f"   - Teacher accuracy: {metrics['teacher_accuracy']:.3f}")
        print(f"   - Knowledge retention: {metrics['knowledge_retention']:.3f}")
        print(f"   - Number of epochs: {metrics['num_epochs']}")

        # Validate knowledge retention
        if metrics["knowledge_retention"] < 0.5:
            print("⚠️  Low knowledge retention - may need adjustment")
        else:
            print("✅ Good knowledge retention achieved")

        return True

    except Exception as e:
        print(f"❌ Knowledge distillation test failed: {e}")
        return False


async def test_sparsity_aggregation():
    """Test sparsity-aware model aggregation."""
    print("\n=== Testing Sparsity-Aware Aggregation ===")

    try:
        # Create test models with different sparsities
        models = []
        weights = []
        masks = []

        for i in range(3):
            model = TestModel()

            # Apply different levels of sparsity
            sparsity_level = 0.3 + i * 0.2  # 0.3, 0.5, 0.7

            # Create sparsity mask
            mask = {}
            for name, param in model.named_parameters():
                if "weight" in name:
                    param_mask = torch.rand_like(param) >= sparsity_level
                    param.data *= param_mask.float()
                    mask[name] = param_mask
                else:
                    mask[name] = torch.ones_like(param, dtype=torch.bool)

            models.append(model)
            weights.append(1.0)  # Equal weights
            masks.append(mask)

        # Create global model
        global_model = TestModel()

        # Initialize aggregator
        aggregator = SparsityAggregator()

        # Perform aggregation
        participant_data = list(zip(models, weights, masks))
        aggregated_model, metrics = await aggregator.aggregate_sparse_models(
            participant_data, global_model
        )

        print("✅ Sparsity-aware aggregation completed successfully")
        print(f"   - Aggregation time: {metrics['aggregation_time']:.3f}s")
        print(f"   - Number of participants: {metrics['num_participants']}")
        print(f"   - Average sparsity: {metrics['average_sparsity']:.3f}")
        print(f"   - Sparsity variance: {metrics['sparsity_variance']:.6f}")
        print(f"   - Min sparsity: {metrics['min_sparsity']:.3f}")
        print(f"   - Max sparsity: {metrics['max_sparsity']:.3f}")
        print(f"   - Effective participants: {metrics['effective_participants']}")

        # Validate aggregated model
        aggregated_sparsity = aggregator._calculate_model_sparsity(aggregated_model)
        print(f"   - Final aggregated sparsity: {aggregated_sparsity:.3f}")

        return True

    except Exception as e:
        print(f"❌ Sparsity aggregation test failed: {e}")
        return False


async def test_pruning_coordinator():
    """Test the main pruning coordinator."""
    print("\n=== Testing Pruning Coordinator ===")

    try:
        # Create test configuration
        config = PruningConfig(
            pruning_type=PruningType.UNSTRUCTURED_MAGNITUDE,
            target_sparsity=0.6,
            schedule=SparsitySchedule.LINEAR,
            pruning_frequency=2,
            recovery_threshold=0.1,
            distillation_enabled=True,
        )

        # Initialize coordinator
        coordinator = PruningCoordinator(config)

        # Create test models
        global_model = TestModel()
        participant_models = [TestModel() for _ in range(3)]
        participant_weights = [1.0, 0.8, 1.2]  # Different weights

        print(f"✅ Coordinator initialized with {len(participant_models)} participants")

        # Test coordination rounds
        total_rounds = 6
        successful_rounds = 0

        for round_num in range(1, total_rounds + 1):
            global_model, updated_participants, metrics = (
                await coordinator.coordinate_pruning_round(
                    participant_models=participant_models,
                    participant_weights=participant_weights,
                    global_model=global_model,
                    round_num=round_num,
                    total_rounds=total_rounds,
                )
            )

            if not metrics.get("skipped", False):
                successful_rounds += 1
                print(
                    f"   Round {round_num}: {metrics['average_sparsity']:.3f} avg sparsity, "
                    f"{metrics['average_accuracy_drop']:.3f} avg accuracy drop"
                )

                # Update participant models for next round
                participant_models = updated_participants

                # Check if distillation was applied
                if metrics["distillation_applied"]:
                    print(f"     → Knowledge distillation applied for recovery")
            else:
                print(
                    f"   Round {round_num}: Skipped (frequency: {config.pruning_frequency})"
                )

        print(f"✅ Completed {successful_rounds} successful pruning rounds")

        # Test summary generation
        summary = coordinator.get_pruning_summary()

        print("✅ Coordinator summary generated:")
        print(f"   - Total pruning rounds: {summary['total_pruning_rounds']}")
        print(f"   - Final average sparsity: {summary['final_average_sparsity']:.3f}")
        print(f"   - Average accuracy drop: {summary['average_accuracy_drop']:.3f}")
        print(
            f"   - Total coordination time: {summary['total_coordination_time']:.2f}s"
        )

        # Test history saving
        history_file = "test_pruning_history.json"
        await coordinator.save_pruning_history(history_file)

        # Verify file was created
        if Path(history_file).exists():
            print(f"✅ History saved to {history_file}")
            # Clean up
            Path(history_file).unlink()
        else:
            print(f"⚠️  History file {history_file} not found")

        return True

    except Exception as e:
        print(f"❌ Pruning coordinator test failed: {e}")
        return False


async def test_sparsity_schedules():
    """Test different sparsity scheduling strategies."""
    print("\n=== Testing Sparsity Schedules ===")

    try:
        schedules = [
            SparsitySchedule.CONSTANT,
            SparsitySchedule.LINEAR,
            SparsitySchedule.EXPONENTIAL,
            SparsitySchedule.POLYNOMIAL,
            SparsitySchedule.COSINE,
        ]

        target_sparsity = 0.8
        total_rounds = 10

        for schedule in schedules:
            config = PruningConfig(
                pruning_type=PruningType.UNSTRUCTURED_MAGNITUDE,
                target_sparsity=target_sparsity,
                schedule=schedule,
            )

            pruner = UnstructuredPruning(config)

            # Test schedule progression
            sparsity_values = []
            for round_num in range(1, total_rounds + 1):
                current_sparsity = pruner.get_current_sparsity_target(
                    round_num, total_rounds
                )
                sparsity_values.append(current_sparsity)

            print(
                f"✅ {schedule.value}: {sparsity_values[0]:.3f} → {sparsity_values[-1]:.3f}"
            )

            # Validate final sparsity
            if abs(sparsity_values[-1] - target_sparsity) > 0.01:
                print(
                    f"⚠️  Final sparsity {sparsity_values[-1]:.3f} deviates from target {target_sparsity}"
                )

        return True

    except Exception as e:
        print(f"❌ Sparsity schedule test failed: {e}")
        return False


async def run_comprehensive_test():
    """Run comprehensive test suite for federated model pruning."""
    print("🚀 Starting Comprehensive Federated Model Pruning Tests")
    print("=" * 60)

    test_functions = [
        ("Configuration Validation", test_pruning_config),
        ("Unstructured Pruning", test_unstructured_pruning),
        ("Structured Pruning", test_structured_pruning),
        ("Knowledge Distillation", test_knowledge_distillation),
        ("Sparsity Aggregation", test_sparsity_aggregation),
        ("Sparsity Schedules", test_sparsity_schedules),
        ("Pruning Coordinator", test_pruning_coordinator),
    ]

    results = {}
    start_time = time.time()

    for test_name, test_func in test_functions:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            result = await test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"\n{test_name}: {status}")
        except Exception as e:
            results[test_name] = False
            print(f"\n{test_name}: ❌ FAILED with exception: {e}")

    total_time = time.time() - start_time

    # Print final summary
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")

    print("-" * 60)
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"EXECUTION TIME: {total_time:.2f} seconds")

    if passed == total:
        print(
            "\n🎉 ALL TESTS PASSED! Federated Model Pruning implementation is working correctly."
        )
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    # Run the comprehensive test suite
    success = asyncio.run(run_comprehensive_test())
    exit_code = 0 if success else 1
    exit(exit_code)
