"""
Unit tests for Federated Meta-Learning (MAML) implementation.
"""

import sys
import os
import math

# Add the project root to Python path for imports
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
except ImportError as e:
    print(f"Import error: {e}")
    print("Attempting direct import from local module...")
    try:
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
    except ImportError as e2:
        print(f"Direct import also failed: {e2}")
        sys.exit(1)


def test_task_creation():
    """Test Task dataclass functionality."""
    data = [([1.0], [2.0]), ([2.0], [4.0]), ([3.0], [6.0])]
    task = Task(
        task_id="test_task",
        input_dim=1,
        output_dim=1,
        data=data,
        task_type="regression",
    )

    assert task.task_id == "test_task"
    assert task.input_dim == 1
    assert task.output_dim == 1
    assert len(task.data) == 3
    assert task.task_type == "regression"

    # Test support/query sampling
    support, query = task.sample_support_query(k_shot=2, q_query=1)
    assert len(support) <= 2
    assert len(query) <= 1
    assert len(support) + len(query) <= len(task.data)


def test_maml_config():
    """Test MAMLConfig dataclass."""
    config = MAMLConfig()

    # Test default values
    assert config.inner_lr == 0.01
    assert config.outer_lr == 0.001
    assert config.inner_steps == 5
    assert config.k_shot == 5
    assert config.q_query == 10
    assert config.task_batch_size == 4
    assert config.second_order == True

    # Test custom config
    custom_config = MAMLConfig(inner_lr=0.05, outer_lr=0.002, inner_steps=3)
    assert custom_config.inner_lr == 0.05
    assert custom_config.outer_lr == 0.002
    assert custom_config.inner_steps == 3


def test_simple_meta_model():
    """Test SimpleMetaModel functionality."""
    model = SimpleMetaModel(input_dim=2, hidden_dim=5, output_dim=1)

    assert model.input_dim == 2
    assert model.hidden_dim == 5
    assert model.output_dim == 1

    # Test parameter structure
    params = model.get_params()
    assert "w1" in params
    assert "b1" in params
    assert "w2" in params
    assert "b2" in params

    assert len(params["w1"]) == 2  # input_dim
    assert len(params["w1"][0]) == 5  # hidden_dim
    assert len(params["b1"]) == 5  # hidden_dim
    assert len(params["w2"]) == 5  # hidden_dim
    assert len(params["w2"][0]) == 1  # output_dim
    assert len(params["b2"]) == 1  # output_dim

    # Test forward pass
    x = [1.0, 0.5]
    y = model.forward(x)
    assert len(y) == 1
    assert isinstance(y[0], float)

    # Test loss computation
    target = [2.0]
    loss = model.loss(x, target)
    assert isinstance(loss, float)
    assert loss >= 0

    # Test gradient computation
    gradients = model.compute_gradients(x, target)
    assert "w1" in gradients
    assert "b1" in gradients
    assert "w2" in gradients
    assert "b2" in gradients

    # Test gradient application with larger learning rate
    updated_params = model.apply_gradients(gradients, learning_rate=0.1)
    assert "w1" in updated_params

    # Check that parameters changed (with tolerance for small changes)
    original_params = model.get_params()
    params_changed = False
    for param_name in original_params:
        if param_name in ["w1", "w2"]:
            for i in range(len(original_params[param_name])):
                for j in range(len(original_params[param_name][i])):
                    if (
                        abs(
                            updated_params[param_name][i][j]
                            - original_params[param_name][i][j]
                        )
                        > 1e-6
                    ):
                        params_changed = True
                        break
        else:
            for i in range(len(original_params[param_name])):
                if (
                    abs(updated_params[param_name][i] - original_params[param_name][i])
                    > 1e-6
                ):
                    params_changed = True
                    break
        if params_changed:
            break

    assert params_changed, "No significant parameter changes detected"


def test_federated_maml():
    """Test FederatedMAML functionality."""
    config = MAMLConfig(inner_steps=2, k_shot=3, q_query=5, task_batch_size=2)
    fed_maml = FederatedMAML(input_dim=1, hidden_dim=5, output_dim=1, config=config)

    assert len(fed_maml.participants) == 0
    assert len(fed_maml.task_distributions) == 0

    # Create test tasks
    tasks = generate_sine_wave_tasks(num_tasks=4, num_samples_per_task=10)

    # Register participants
    fed_maml.register_participant("participant_0", tasks[:2])
    fed_maml.register_participant("participant_1", tasks[2:])

    assert len(fed_maml.participants) == 2
    assert "participant_0" in fed_maml.task_distributions
    assert "participant_1" in fed_maml.task_distributions
    assert len(fed_maml.task_distributions["participant_0"]) == 2
    assert len(fed_maml.task_distributions["participant_1"]) == 2


def test_inner_loop_adaptation():
    """Test inner loop task adaptation."""
    config = MAMLConfig(inner_steps=3, k_shot=3, q_query=5)
    fed_maml = FederatedMAML(input_dim=1, hidden_dim=4, output_dim=1, config=config)

    # Create a simple task
    task_data = [([x], [2 * x + 1]) for x in [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]
    task = Task("test_task", 1, 1, task_data, "regression")

    meta_params = fed_maml.meta_model.get_params()
    adapted_params, inner_losses = fed_maml.inner_loop_adaptation(task, meta_params)

    # Check that adaptation occurred
    assert "w1" in adapted_params
    assert "b1" in adapted_params
    assert "w2" in adapted_params
    assert "b2" in adapted_params

    # Check that inner losses were recorded
    assert len(inner_losses) == config.inner_steps
    assert all(isinstance(loss, float) for loss in inner_losses)
    assert all(loss >= 0 for loss in inner_losses)

    # Check that parameters changed during adaptation
    params_changed = False
    for param_name in meta_params:
        if param_name in ["w1", "w2"]:
            for i in range(len(meta_params[param_name])):
                for j in range(len(meta_params[param_name][i])):
                    if (
                        adapted_params[param_name][i][j]
                        != meta_params[param_name][i][j]
                    ):
                        params_changed = True
                        break
        else:
            for i in range(len(meta_params[param_name])):
                if adapted_params[param_name][i] != meta_params[param_name][i]:
                    params_changed = True
                    break

    assert params_changed, "Parameters did not change during adaptation"


def test_meta_gradient_computation():
    """Test meta-gradient computation."""
    config = MAMLConfig(inner_steps=2, task_batch_size=2, k_shot=3, q_query=4)
    fed_maml = FederatedMAML(input_dim=1, hidden_dim=3, output_dim=1, config=config)

    # Create and register tasks
    tasks = generate_sine_wave_tasks(num_tasks=3, num_samples_per_task=8)
    fed_maml.register_participant("participant_0", tasks)

    meta_params = fed_maml.meta_model.get_params()
    meta_gradients, meta_loss = fed_maml.compute_meta_gradients(
        "participant_0", meta_params
    )

    # Check meta-gradients structure
    assert "w1" in meta_gradients
    assert "b1" in meta_gradients
    assert "w2" in meta_gradients
    assert "b2" in meta_gradients

    # Check meta-loss
    assert isinstance(meta_loss, float)
    assert meta_loss >= 0

    # Check gradient values are finite
    for param_name, grad_values in meta_gradients.items():
        if param_name in ["w1", "w2"]:
            for row in grad_values:
                for val in row:
                    assert math.isfinite(
                        val
                    ), f"Non-finite gradient in {param_name}: {val}"
        else:
            for val in grad_values:
                assert math.isfinite(val), f"Non-finite gradient in {param_name}: {val}"


def test_gradient_aggregation():
    """Test gradient aggregation across participants."""
    config = MAMLConfig(inner_steps=2)
    fed_maml = FederatedMAML(input_dim=1, hidden_dim=3, output_dim=1, config=config)

    # Create mock gradients for two participants
    grad1 = {
        "w1": [[1.0, 2.0, 3.0]],
        "b1": [0.5, 0.6, 0.7],
        "w2": [[1.5], [2.5], [3.5]],
        "b2": [1.0],
    }

    grad2 = {
        "w1": [[2.0, 4.0, 6.0]],
        "b1": [1.0, 1.2, 1.4],
        "w2": [[3.0], [5.0], [7.0]],
        "b2": [2.0],
    }

    aggregated = fed_maml._aggregate_gradients([grad1, grad2])

    # Check aggregated results (should be averages)
    assert aggregated["w1"][0][0] == 1.5  # (1.0 + 2.0) / 2
    assert aggregated["w1"][0][1] == 3.0  # (2.0 + 4.0) / 2
    assert aggregated["b1"][0] == 0.75  # (0.5 + 1.0) / 2
    assert aggregated["w2"][0][0] == 2.25  # (1.5 + 3.0) / 2
    assert aggregated["b2"][0] == 1.5  # (1.0 + 2.0) / 2


def test_gradient_clipping():
    """Test gradient clipping functionality."""
    config = MAMLConfig()
    fed_maml = FederatedMAML(input_dim=1, hidden_dim=2, output_dim=1, config=config)

    # Create large gradients that should be clipped
    large_gradients = {
        "w1": [[10.0, 20.0]],
        "b1": [30.0, 40.0],
        "w2": [[50.0], [60.0]],
        "b2": [70.0],
    }

    clipped = fed_maml._clip_gradients(large_gradients, max_norm=1.0)

    # Check that gradients were scaled down
    total_norm_squared = 0.0
    for param_name, grad_values in clipped.items():
        if param_name in ["w1", "w2"]:
            for row in grad_values:
                for val in row:
                    total_norm_squared += val**2
        else:
            for val in grad_values:
                total_norm_squared += val**2

    total_norm = math.sqrt(total_norm_squared)
    assert (
        total_norm <= 1.01
    ), f"Gradient norm after clipping: {total_norm}"  # Small tolerance for floating point


def test_generate_sine_wave_tasks():
    """Test sine wave task generation."""
    tasks = generate_sine_wave_tasks(num_tasks=5, num_samples_per_task=10)

    assert len(tasks) == 5

    for i, task in enumerate(tasks):
        assert task.task_id == f"sine_task_{i}"
        assert task.input_dim == 1
        assert task.output_dim == 1
        assert len(task.data) == 10
        assert task.task_type == "regression"

        # Check data format
        for x, y in task.data:
            assert len(x) == 1
            assert len(y) == 1
            assert isinstance(x[0], float)
            assert isinstance(y[0], float)


def test_federated_maml_training():
    """Test full federated MAML training loop."""
    config = MAMLConfig(inner_steps=2, k_shot=3, q_query=4, task_batch_size=2)
    fed_maml = FederatedMAML(input_dim=1, hidden_dim=4, output_dim=1, config=config)

    # Create tasks for participants
    tasks = generate_sine_wave_tasks(num_tasks=6, num_samples_per_task=10)

    fed_maml.register_participant("p1", tasks[:3])
    fed_maml.register_participant("p2", tasks[3:])

    # Run training
    results = fed_maml.federated_meta_update(num_rounds=3)

    # Check results structure
    assert "final_meta_params" in results
    assert "meta_train_losses" in results
    assert "num_rounds" in results
    assert "num_participants" in results

    assert results["num_rounds"] == 3
    assert results["num_participants"] == 2
    assert len(results["meta_train_losses"]) == 3

    # Check that losses are finite
    for loss in results["meta_train_losses"]:
        assert isinstance(loss, float)
        assert math.isfinite(loss)
        assert loss >= 0


def test_adaptation_evaluation():
    """Test task adaptation evaluation."""
    config = MAMLConfig(inner_steps=3, k_shot=4, q_query=6)
    fed_maml = FederatedMAML(input_dim=1, hidden_dim=5, output_dim=1, config=config)

    # Train with some tasks
    train_tasks = generate_sine_wave_tasks(num_tasks=4, num_samples_per_task=12)
    fed_maml.register_participant("p1", train_tasks)
    fed_maml.federated_meta_update(num_rounds=2)

    # Evaluate on test tasks
    test_tasks = generate_sine_wave_tasks(num_tasks=3, num_samples_per_task=10)
    evaluation = fed_maml.evaluate_adaptation(test_tasks, adaptation_steps=3)

    # Check evaluation structure
    assert "adaptation_results" in evaluation
    assert "avg_improvement" in evaluation
    assert "avg_pre_loss" in evaluation
    assert "avg_post_loss" in evaluation

    assert len(evaluation["adaptation_results"]) == 3

    for result in evaluation["adaptation_results"]:
        assert "task_id" in result
        assert "pre_adaptation_loss" in result
        assert "post_adaptation_loss" in result
        assert "improvement" in result
        assert "inner_losses" in result

        # Check that values are reasonable
        assert isinstance(result["pre_adaptation_loss"], float)
        assert isinstance(result["post_adaptation_loss"], float)
        assert result["pre_adaptation_loss"] >= 0
        assert result["post_adaptation_loss"] >= 0


def run_all_tests():
    """Run all tests directly without pytest."""
    test_functions = [
        test_task_creation,
        test_maml_config,
        test_simple_meta_model,
        test_federated_maml,
        test_inner_loop_adaptation,
        test_meta_gradient_computation,
        test_gradient_aggregation,
        test_gradient_clipping,
        test_generate_sine_wave_tasks,
        test_federated_maml_training,
        test_adaptation_evaluation,
    ]

    print("Running Federated Meta-Learning unit tests...")
    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            print(f"✅ {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__}: {e}")
            failed += 1

    print(f"\nTest Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All Meta-Learning tests passed!")
        return True
    else:
        print("⚠️  Some tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
