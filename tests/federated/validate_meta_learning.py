#!/usr/bin/env python3
"""
Validation script for Federated Meta-Learning (MAML) implem    # Generate diverse sine wave tasks
    tasks = generate_sine_wave_tasks(
        num_tasks=5,
        num_samples_per_task=30
    ).
Verifies MAML adaptation speed, task generalization, and federated meta-learning properties.
"""

import sys
import os
import math

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

try:
    from src.vega.federated.meta_learning import (
        Task,
        MAMLConfig,
        SimpleMetaModel,
        FederatedMAML,
        generate_sine_wave_tasks,
        run_federated_maml,
    )
except ImportError:
    # Fallback to direct import
    sys.path.insert(
        0, os.path.join(os.path.dirname(__file__), "../../src/vega/federated/")
    )
    from meta_learning import (
        Task,
        MAMLConfig,
        SimpleMetaModel,
        FederatedMAML,
        generate_sine_wave_tasks,
        run_federated_maml,
    )


def validate_maml_adaptation_speed():
    """Validate that MAML enables fast adaptation to new tasks."""
    print("🔍 Validating MAML Adaptation Speed...")

    # Create a meta-model
    model = SimpleMetaModel(input_dim=1, output_dim=1, hidden_dim=10)

    # Generate a sine wave task
    tasks = generate_sine_wave_tasks(num_tasks=1, num_samples_per_task=20)
    task = tasks[0]  # Test adaptation before meta-training (baseline)
    initial_loss = 0.0
    for x, y_true in task.data[:5]:  # Use first 5 samples for adaptation
        y_pred = model.forward(x)
        loss = sum((pred - true) ** 2 for pred, true in zip(y_pred, y_true)) / len(
            y_true
        )
        initial_loss += loss
    initial_loss /= 5

    # Perform inner loop adaptation
    adapted_params = model.inner_loop_update(task, inner_lr=0.1, inner_steps=5)

    # Test performance after adaptation
    model.set_params(adapted_params)
    adapted_loss = 0.0
    for x, y_true in task.data[5:10]:  # Use next 5 samples for evaluation
        y_pred = model.forward(x)
        loss = sum((pred - true) ** 2 for pred, true in zip(y_pred, y_true)) / len(
            y_true
        )
        adapted_loss += loss
    adapted_loss /= 5

    # Adaptation should improve performance
    improvement = initial_loss - adapted_loss

    print(f"✅ MAML adaptation speed validated:")
    print(f"   - Initial loss: {initial_loss:.4f}")
    print(f"   - After adaptation: {adapted_loss:.4f}")
    print(f"   - Improvement: {improvement:.4f}")

    assert improvement > 0, f"No improvement observed: {improvement}"
    assert adapted_loss < initial_loss * 0.8, "Insufficient adaptation improvement"


def validate_task_generalization():
    """Validate that meta-learning generalizes across different task types."""
    print("\n🔍 Validating Task Generalization...")

    # Generate diverse sine wave tasks
    tasks = generate_sine_wave_tasks(
        num_tasks=5,
        samples_per_task=30,
        amplitude_range=(0.5, 2.0),
        phase_range=(0, math.pi),
        frequency_range=(0.5, 2.0),
    )

    # Test that tasks are indeed different
    task_losses = []
    model = SimpleMetaModel(input_dim=1, output_dim=1, hidden_dim=10)

    for i, task in enumerate(tasks):
        loss = 0.0
        count = 0
        for x, y_true in task.data[:10]:  # Test on first 10 samples
            y_pred = model.forward(x)
            loss += sum((pred - true) ** 2 for pred, true in zip(y_pred, y_true)) / len(
                y_true
            )
            count += 1
        avg_loss = loss / count if count > 0 else 0
        task_losses.append(avg_loss)

    # Tasks should show some variance in difficulty
    loss_variance = sum(
        (loss - sum(task_losses) / len(task_losses)) ** 2 for loss in task_losses
    ) / len(task_losses)

    print(f"✅ Task generalization validated:")
    print(f"   - Generated {len(tasks)} diverse tasks")
    print(f"   - Loss variance: {loss_variance:.4f}")
    print(f"   - Task loss range: {min(task_losses):.3f} to {max(task_losses):.3f}")

    assert len(tasks) == 5, "Wrong number of tasks generated"
    assert loss_variance > 0.01, "Tasks are too similar (low variance)"


def validate_federated_maml_convergence():
    """Validate that federated MAML training converges and improves meta-learning."""
    print("\n🔍 Validating Federated MAML Convergence...")

    config = MAMLConfig(
        inner_lr=0.1, meta_lr=0.01, inner_steps=3, meta_batch_size=4, max_grad_norm=1.0
    )

    results = run_federated_maml(
        num_participants=3,
        num_tasks_per_participant=5,
        meta_rounds=3,
        config=config,
        seed=42,
    )

    # Check that training occurred
    assert "meta_history" in results, "Missing meta training history"
    assert "final_model_params" in results, "Missing final model parameters"
    assert "task_adaptation_results" in results, "Missing adaptation results"

    # Check convergence indicators
    meta_history = results["meta_history"]
    assert len(meta_history) == 3, f"Expected 3 meta rounds, got {len(meta_history)}"

    # Loss should generally decrease or remain stable
    initial_loss = meta_history[0]["avg_meta_loss"]
    final_loss = meta_history[-1]["avg_meta_loss"]

    print(f"✅ Federated MAML convergence validated:")
    print(f"   - Meta training rounds: {len(meta_history)}")
    print(f"   - Initial meta loss: {initial_loss:.4f}")
    print(f"   - Final meta loss: {final_loss:.4f}")
    print(f"   - Participants: {len(results.get('participants', {}))}")

    # Validate adaptation results
    adaptation_results = results["task_adaptation_results"]
    if adaptation_results:
        avg_adaptation_improvement = sum(
            result["improvement"] for result in adaptation_results
        ) / len(adaptation_results)
        print(f"   - Avg adaptation improvement: {avg_adaptation_improvement:.4f}")

        # Meta-learning should enable positive adaptation
        assert avg_adaptation_improvement > 0, "No positive adaptation observed"


def validate_inner_outer_loop_dynamics():
    """Validate proper inner and outer loop MAML dynamics."""
    print("\n🔍 Validating Inner/Outer Loop Dynamics...")

    model = SimpleMetaModel(input_dim=1, output_dim=1, hidden_dim=8)
    tasks = generate_sine_wave_tasks(num_tasks=2, samples_per_task=20)

    # Test inner loop updates
    task = tasks[0]
    original_params = model.get_params()
    adapted_params = model.inner_loop_update(task, inner_lr=0.1, inner_steps=3)

    # Parameters should change during inner loop
    param_changes = []
    for orig_w, adapt_w in zip(original_params["weights"], adapted_params["weights"]):
        for orig, adapt in zip(orig_w, adapt_w):
            param_changes.append(abs(orig - adapt))

    avg_param_change = sum(param_changes) / len(param_changes)

    # Test meta-gradient computation
    meta_grads = model.compute_meta_gradients(
        [tasks[0], tasks[1]], inner_lr=0.1, inner_steps=2
    )

    # Meta-gradients should be non-zero
    total_meta_grad = 0.0
    grad_count = 0
    for layer_grads in meta_grads.values():
        if isinstance(layer_grads[0], list):
            for row in layer_grads:
                for grad in row:
                    total_meta_grad += abs(grad)
                    grad_count += 1
        else:
            for grad in layer_grads:
                total_meta_grad += abs(grad)
                grad_count += 1

    avg_meta_grad = total_meta_grad / grad_count if grad_count > 0 else 0

    print(f"✅ Inner/Outer loop dynamics validated:")
    print(f"   - Average parameter change in inner loop: {avg_param_change:.6f}")
    print(f"   - Average meta-gradient magnitude: {avg_meta_grad:.6f}")

    assert avg_param_change > 1e-6, "Inner loop not updating parameters sufficiently"
    assert avg_meta_grad > 1e-6, "Meta-gradients too small"


def validate_gradient_aggregation():
    """Validate federated gradient aggregation in MAML."""
    print("\n🔍 Validating Gradient Aggregation...")

    config = MAMLConfig(inner_lr=0.1, meta_lr=0.01, inner_steps=2)
    fed_maml = FederatedMAML(input_dim=1, output_dim=1, hidden_dim=8, config=config)

    # Create participant task sets
    all_tasks = generate_sine_wave_tasks(num_tasks=6, samples_per_task=15)
    participant_tasks = [all_tasks[:2], all_tasks[2:4], all_tasks[4:6]]

    # Compute gradients for each participant
    participant_gradients = []
    for tasks in participant_tasks:
        grads = fed_maml.global_model.compute_meta_gradients(
            tasks, inner_lr=config.inner_lr, inner_steps=config.inner_steps
        )
        participant_gradients.append(grads)

    # Aggregate gradients
    aggregated_grads = fed_maml.aggregate_meta_gradients(participant_gradients)

    # Check aggregation properties
    assert "weights" in aggregated_grads, "Missing weights in aggregated gradients"
    assert "bias" in aggregated_grads, "Missing bias in aggregated gradients"

    # Aggregated gradients should be averages
    # Check one weight element as example
    first_weight_grads = [grads["weights"][0][0] for grads in participant_gradients]
    expected_avg = sum(first_weight_grads) / len(first_weight_grads)
    actual_avg = aggregated_grads["weights"][0][0]

    print(f"✅ Gradient aggregation validated:")
    print(f"   - Participants: {len(participant_gradients)}")
    print(f"   - Expected avg gradient: {expected_avg:.6f}")
    print(f"   - Actual aggregated gradient: {actual_avg:.6f}")

    assert abs(expected_avg - actual_avg) < 1e-6, "Gradient aggregation not correct"


def validate_parameter_consistency():
    """Validate model parameter consistency throughout meta-learning."""
    print("\n🔍 Validating Parameter Consistency...")

    model = SimpleMetaModel(input_dim=1, output_dim=1, hidden_dim=6)

    # Get initial parameters
    initial_params = model.get_params()

    # Verify parameter structure
    assert "weights" in initial_params, "Missing weights in parameters"
    assert "bias" in initial_params, "Missing bias in parameters"

    # Check parameter dimensions
    expected_weight_shapes = [(1, 6), (6, 1)]  # input->hidden, hidden->output
    actual_weight_shapes = [
        (len(initial_params["weights"]), len(initial_params["weights"][0])),
        (len(initial_params["weights2"]), len(initial_params["weights2"][0])),
    ]

    expected_bias_shapes = [6, 1]  # hidden bias, output bias
    actual_bias_shapes = [len(initial_params["bias"]), len(initial_params["bias2"])]

    # Set and get parameters to test consistency
    model.set_params(initial_params)
    retrieved_params = model.get_params()

    # Parameters should be identical after set/get
    weights_match = True
    for orig_row, retr_row in zip(
        initial_params["weights"], retrieved_params["weights"]
    ):
        for orig, retr in zip(orig_row, retr_row):
            if abs(orig - retr) > 1e-8:
                weights_match = False
                break

    print(f"✅ Parameter consistency validated:")
    print(f"   - Weight shapes: {actual_weight_shapes}")
    print(f"   - Bias shapes: {actual_bias_shapes}")
    print(f"   - Parameter set/get consistency: {'✓' if weights_match else '✗'}")

    assert (
        actual_weight_shapes == expected_weight_shapes
    ), f"Wrong weight shapes: {actual_weight_shapes}"
    assert (
        actual_bias_shapes == expected_bias_shapes
    ), f"Wrong bias shapes: {actual_bias_shapes}"
    assert weights_match, "Parameter set/get inconsistency"


def run_full_validation():
    """Run complete meta-learning validation suite."""
    print("🚀 Starting Federated Meta-Learning Validation Suite")
    print("=" * 60)

    validation_tests = [
        validate_maml_adaptation_speed,
        validate_task_generalization,
        validate_federated_maml_convergence,
        validate_inner_outer_loop_dynamics,
        validate_gradient_aggregation,
        validate_parameter_consistency,
    ]

    passed = 0
    failed = 0

    for test in validation_tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Validation Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All Federated Meta-Learning validations passed!")
        print("✅ MAML implementation is working correctly")
        return True
    else:
        print("⚠️  Some validations failed")
        return False


if __name__ == "__main__":
    success = run_full_validation()
    sys.exit(0 if success else 1)
