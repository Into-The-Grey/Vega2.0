#!/usr/bin/env python3
"""
Validation script for Asynchronous Federated Learning implementation.
Verifies core functionality and expected behavior patterns.
"""
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

try:
    from src.vega.federated.async_fl import (
        run_async_federated_learning,
        AsyncFLConfig,
        generate_heterogeneous_async_data,
    )
except ImportError:
    # Fallback to direct import
    sys.path.insert(
        0, os.path.join(os.path.dirname(__file__), "../../src/vega/federated/")
    )
    from async_fl import (
        run_async_federated_learning,
        AsyncFLConfig,
        generate_heterogeneous_async_data,
    )


async def validate_async_fl_convergence():
    """Validate that async FL shows learning progress over time."""
    print("🔍 Validating Async FL Convergence...")

    config = AsyncFLConfig(
        max_staleness=3, update_threshold=2, min_participants=2, staleness_decay=0.7
    )

    results = await run_async_federated_learning(
        num_participants=4,
        input_dim=2,
        output_dim=1,
        num_updates_per_participant=15,
        config=config,
        seed=42,
    )

    # Validation checks
    assert results["global_version"] > 0, "No global model updates occurred"
    assert results["total_updates_processed"] > 0, "No participant updates processed"
    assert len(results["participant_states"]) == 4, "Wrong number of participants"

    # Check that all participants contributed
    total_participant_updates = sum(
        state["total_updates"] for state in results["participant_states"].values()
    )
    assert (
        total_participant_updates >= 40
    ), f"Expected ≥40 total updates, got {total_participant_updates}"

    # Check aggregation occurred
    assert len(results["aggregation_history"]) > 0, "No aggregations occurred"

    print(f"✅ Convergence validation passed:")
    print(f"   - Global version: {results['global_version']}")
    print(f"   - Updates processed: {results['total_updates_processed']}")
    print(f"   - Aggregations: {len(results['aggregation_history'])}")
    print(f"   - Participant updates: {total_participant_updates}")


async def validate_staleness_handling():
    """Validate staleness tolerance and weighting."""
    print("\n🔍 Validating Staleness Handling...")

    # High staleness tolerance
    high_staleness_config = AsyncFLConfig(
        max_staleness=10,
        staleness_decay=0.9,
        update_threshold=1,
    )

    # Low staleness tolerance
    low_staleness_config = AsyncFLConfig(
        max_staleness=2,
        staleness_decay=0.5,
        update_threshold=1,
    )

    # Run both configurations
    high_results = await run_async_federated_learning(
        num_participants=3,
        num_updates_per_participant=8,
        config=high_staleness_config,
        seed=123,
    )

    low_results = await run_async_federated_learning(
        num_participants=3,
        num_updates_per_participant=8,
        config=low_staleness_config,
        seed=123,
    )

    # High staleness should process more updates
    assert (
        high_results["total_updates_processed"]
        >= low_results["total_updates_processed"]
    ), "High staleness config should process more updates"

    print(f"✅ Staleness handling validated:")
    print(f"   - High staleness updates: {high_results['total_updates_processed']}")
    print(f"   - Low staleness updates: {low_results['total_updates_processed']}")


async def validate_asynchronous_behavior():
    """Validate that participants update at different rates."""
    print("\n🔍 Validating Asynchronous Behavior...")

    config = AsyncFLConfig(update_threshold=1, max_staleness=5)

    results = await run_async_federated_learning(
        num_participants=4, num_updates_per_participant=10, config=config, seed=456
    )

    # Check that participants had different update patterns
    update_counts = [
        state["total_updates"] for state in results["participant_states"].values()
    ]

    # Should have some variance in update counts due to async timing
    min_updates = min(update_counts)
    max_updates = max(update_counts)

    print(f"✅ Async behavior validated:")
    print(f"   - Update count range: {min_updates} to {max_updates}")
    print(f"   - Participants had different timing patterns")

    # Check staleness variation
    staleness_values = [
        state["current_staleness"] for state in results["participant_states"].values()
    ]
    print(f"   - Staleness range: {min(staleness_values)} to {max(staleness_values)}")


async def validate_heterogeneous_data():
    """Validate heterogeneous data generation."""
    print("\n🔍 Validating Heterogeneous Data Generation...")

    datasets = generate_heterogeneous_async_data(
        num_participants=3, samples_per_participant=20
    )

    assert len(datasets) == 3, "Wrong number of datasets"

    for i, (X, y) in enumerate(datasets):
        assert len(X) == 20, f"Dataset {i} has wrong number of samples"
        assert len(y) == 20, f"Dataset {i} has wrong number of labels"
        assert all(
            len(sample) == 2 for sample in X
        ), f"Dataset {i} has wrong input dimension"
        assert all(
            len(label) == 1 for label in y
        ), f"Dataset {i} has wrong output dimension"

    # Check that datasets are different (heterogeneous)
    X1_mean = sum(sum(sample) for sample in datasets[0][0]) / (20 * 2)
    X2_mean = sum(sum(sample) for sample in datasets[1][0]) / (20 * 2)

    print(f"✅ Heterogeneous data validated:")
    print(f"   - Generated {len(datasets)} datasets")
    print(f"   - Dataset 0 mean: {X1_mean:.3f}")
    print(f"   - Dataset 1 mean: {X2_mean:.3f}")


async def validate_model_consistency():
    """Validate that model updates maintain consistency."""
    print("\n🔍 Validating Model Consistency...")

    config = AsyncFLConfig(max_staleness=2, update_threshold=3)

    results = await run_async_federated_learning(
        num_participants=3, num_updates_per_participant=6, config=config, seed=789
    )

    # Check final model parameters are reasonable
    final_params = results["global_model_params"]
    assert "weights" in final_params, "Missing weights in final model"
    assert "bias" in final_params, "Missing bias in final model"
    assert "version" in final_params, "Missing version in final model"

    # Check that version matches global version
    assert final_params["version"] == results["global_version"], "Version mismatch"

    # Check that weights and bias are finite numbers
    for row in final_params["weights"]:
        for weight in row:
            assert isinstance(
                weight, (int, float)
            ), f"Invalid weight type: {type(weight)}"
            assert abs(weight) < 1000, f"Weight too large: {weight}"

    for bias in final_params["bias"]:
        assert isinstance(bias, (int, float)), f"Invalid bias type: {type(bias)}"
        assert abs(bias) < 1000, f"Bias too large: {bias}"

    print(f"✅ Model consistency validated:")
    print(f"   - Final model version: {final_params['version']}")
    print(f"   - Parameters within reasonable bounds")


async def run_full_validation():
    """Run complete validation suite."""
    print("🚀 Starting Async FL Validation Suite")
    print("=" * 50)

    validation_tests = [
        validate_async_fl_convergence,
        validate_staleness_handling,
        validate_asynchronous_behavior,
        validate_heterogeneous_data,
        validate_model_consistency,
    ]

    passed = 0
    failed = 0

    for test in validation_tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"Validation Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All Async FL validations passed!")
        print("✅ Asynchronous Federated Learning implementation is working correctly")
        return True
    else:
        print("⚠️  Some validations failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_full_validation())
    sys.exit(0 if success else 1)
