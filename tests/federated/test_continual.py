import math
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.vega.federated.continual import (
    Task,
    LinearModel,
    EWCParticipant,
    EWCConfig,
    run_continual_federated_learning,
    create_synthetic_task_sequence,
)


def test_ewc_participant_learns_and_retains():
    """Test that EWC participant can learn tasks and retain knowledge."""

    # Create a simple linear task: y = 2*x + 1
    def simple_task_generator(batch_size: int):
        import random

        random.seed(42)  # Deterministic for testing
        X, y = [], []
        for _ in range(batch_size):
            x = random.gauss(0, 1)
            target = 2.0 * x + 1.0 + random.gauss(0, 0.05)  # Low noise
            X.append([x])
            y.append([target])
        return X, y

    task = Task("simple_linear", 1, 1, simple_task_generator)
    config = EWCConfig(lambda_ewc=100.0, fisher_samples=50)
    participant = EWCParticipant("test_p1", 1, 1, config)

    # Train on task
    metrics = participant.train_on_task(task, steps=200, lr=0.05)

    # Check that training reduced loss
    assert metrics["data_loss"] < 1.0  # Should learn the linear relationship

    # Finish task (compute Fisher info)
    participant.finish_task(task)

    # Verify task is in history
    assert "simple_linear" in participant.task_history

    # Test retention: evaluate on the same task should still be good
    eval_loss = participant.evaluate_on_task(task, num_samples=100)
    assert eval_loss < 1.0  # Model should still perform well


def test_continual_learning_prevents_catastrophic_forgetting():
    """Test that continual learning with EWC prevents catastrophic forgetting."""
    task_sequence = create_synthetic_task_sequence()[:2]  # Use first 2 tasks

    results = run_continual_federated_learning(
        num_participants=2,
        tasks=task_sequence,
        steps_per_task=100,
        fed_rounds_per_task=2,
        lr=0.03,
        ewc_lambda=200.0,  # Moderate EWC strength
        seed=42,
    )

    # Check basic structure
    assert len(results["tasks"]) == 2
    assert len(results["performance_matrix"]) == 2  # 2 learning stages

    # After learning both tasks, check performance on first task
    final_performance = results["performance_matrix"][-1]
    task1_final_loss = sum(final_performance[0]) / len(final_performance[0])

    # Performance on task 1 should not be catastrophically bad
    # (Note: some degradation is expected, but not complete forgetting)
    assert task1_final_loss < 5.0  # Reasonable threshold for retained knowledge

    # Check that federated history is recorded
    assert len(results["federated_history"]) == 4  # 2 tasks × 2 rounds each


def test_linear_model_basic_functionality():
    """Test basic forward pass and gradient computation of LinearModel."""
    model = LinearModel(2, 1)

    # Set known weights for testing
    model.W = [[1.0], [2.0]]  # W[0][0] = 1.0, W[1][0] = 2.0
    model.b = [0.5]

    # Test forward pass: y = 1.0*x1 + 2.0*x2 + 0.5
    x = [3.0, 4.0]
    y_pred = model.forward(x)
    expected = 1.0 * 3.0 + 2.0 * 4.0 + 0.5  # = 3 + 8 + 0.5 = 11.5
    assert abs(y_pred[0] - expected) < 1e-6

    # Test MSE loss
    y_true = [10.0]
    loss = model.mse_loss(x, y_true)
    expected_loss = (11.5 - 10.0) ** 2  # = 2.25
    assert abs(loss - expected_loss) < 1e-6

    # Test gradient computation (non-zero gradients expected)
    grads = model.gradient(x, y_true)
    assert abs(grads["W"][0][0]) > 0  # Should have non-zero gradient
    assert abs(grads["W"][1][0]) > 0
    assert abs(grads["b"][0]) > 0


def test_task_generation_determinism():
    """Test that task generators produce deterministic data with seeding."""
    tasks = create_synthetic_task_sequence()

    import random

    random.seed(123)
    X1, y1 = tasks[0].generate_batch(10)

    random.seed(123)
    X2, y2 = tasks[0].generate_batch(10)

    # Should be identical
    for i in range(10):
        assert X1[i] == X2[i]
        assert y1[i] == y2[i]


if __name__ == "__main__":
    test_linear_model_basic_functionality()
    print("✓ Linear model functionality test passed")

    test_task_generation_determinism()
    print("✓ Task generation determinism test passed")

    test_ewc_participant_learns_and_retains()
    print("✓ EWC participant learning and retention test passed")

    test_continual_learning_prevents_catastrophic_forgetting()
    print("✓ Continual learning catastrophic forgetting prevention test passed")

    print("✓ All Continual FL tests passed")
