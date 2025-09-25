"""
Unit tests for Asynchronous Federated Learning implementation.
"""

import asyncio
import sys
import os

# Add the project root to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

try:
    from src.vega.federated.async_fl import (
        AsyncUpdate,
        ParticipantState,
        AsyncFLConfig,
        SimpleAsyncModel,
        AsyncParticipant,
        AsyncFLCoordinator,
        run_async_federated_learning,
        generate_heterogeneous_async_data,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Attempting direct import from local module...")
    try:
        sys.path.insert(
            0, os.path.join(os.path.dirname(__file__), "../../src/vega/federated/")
        )
        from async_fl import (
            AsyncUpdate,
            ParticipantState,
            AsyncFLConfig,
            SimpleAsyncModel,
            AsyncParticipant,
            AsyncFLCoordinator,
            run_async_federated_learning,
            generate_heterogeneous_async_data,
        )
    except ImportError as e2:
        print(f"Direct import also failed: {e2}")
        sys.exit(1)


def test_simple_async_model():
    """Test basic SimpleAsyncModel functionality."""
    model = SimpleAsyncModel(input_dim=2, output_dim=1)

    # Test parameter getting/setting
    params = model.get_params()
    assert "weights" in params
    assert "bias" in params
    assert "version" in params
    assert len(params["weights"]) == 2
    assert len(params["bias"]) == 1

    # Test forward pass
    x = [1.0, 2.0]
    y = model.forward(x)
    assert len(y) == 1
    assert isinstance(y[0], float)

    # Test gradient computation
    y_true = [3.0]
    grad = model.compute_gradient(x, y_true)
    assert "weights" in grad
    assert "bias" in grad

    # Test gradient application
    initial_version = model.version
    model.apply_gradient(grad, lr=0.01)
    assert model.version == initial_version + 1


def test_async_update_creation():
    """Test AsyncUpdate dataclass creation."""
    import time

    update = AsyncUpdate(
        participant_id="test_participant",
        model_params={"weights": [[1.0]], "bias": [0.5]},
        gradient={"weights": [[0.1]], "bias": [0.05]},
        timestamp=time.time(),
        local_epoch=1,
    )

    assert update.participant_id == "test_participant"
    assert update.staleness == 0  # Default value
    assert "weights" in update.model_params
    assert "gradient" in update.__dict__


def test_participant_state():
    """Test ParticipantState tracking."""
    state = ParticipantState("participant_1")

    assert state.participant_id == "participant_1"
    assert state.total_updates == 0
    assert state.is_active == True
    assert state.contribution_weight == 1.0


def test_async_fl_config():
    """Test AsyncFLConfig dataclass."""
    config = AsyncFLConfig()

    # Test default values
    assert config.max_staleness == 5
    assert config.staleness_decay == 0.8
    assert config.min_participants == 2
    assert config.update_threshold == 3

    # Test custom config
    custom_config = AsyncFLConfig(max_staleness=10, update_threshold=5)
    assert custom_config.max_staleness == 10
    assert custom_config.update_threshold == 5


def test_async_participant():
    """Test AsyncParticipant functionality."""
    participant = AsyncParticipant("test_p", input_dim=2, output_dim=1)

    assert participant.participant_id == "test_p"
    assert participant.model.input_dim == 2
    assert participant.model.output_dim == 1
    assert participant.local_epoch == 0

    # Set training data
    X = [[1.0, 2.0], [2.0, 3.0]]
    y = [[3.0], [5.0]]
    participant.set_training_data(X, y)
    assert len(participant.local_data) == 2

    # Test local training step
    initial_epoch = participant.local_epoch
    grad = participant.local_training_step()
    assert participant.local_epoch == initial_epoch + 1
    assert "weights" in grad
    assert "bias" in grad


def test_async_fl_coordinator():
    """Test AsyncFLCoordinator functionality."""
    config = AsyncFLConfig(max_staleness=3, update_threshold=2)
    coordinator = AsyncFLCoordinator(input_dim=2, output_dim=1, config=config)

    assert coordinator.global_version == 0
    assert coordinator.total_updates_processed == 0

    # Register participants
    coordinator.register_participant("p1")
    coordinator.register_participant("p2")
    assert "p1" in coordinator.participants
    assert "p2" in coordinator.participants

    # Test staleness computation
    import time

    update = AsyncUpdate(
        participant_id="p1",
        model_params={"weights": [[1.0]], "bias": [0.5], "version": 0},
        gradient={"weights": [[0.1]], "bias": [0.01]},
        timestamp=time.time(),
        local_epoch=1,
    )
    staleness = coordinator.compute_staleness(update)
    assert staleness >= 0

    # Test weight computation
    weight = coordinator.compute_update_weight(update, staleness=1)
    assert isinstance(weight, float)
    assert weight > 0


def test_generate_heterogeneous_data():
    """Test heterogeneous data generation."""
    datasets = generate_heterogeneous_async_data(
        num_participants=3, samples_per_participant=10
    )

    assert len(datasets) == 3
    for X, y in datasets:
        assert len(X) == 10
        assert len(y) == 10
        assert len(X[0]) == 2  # 2D input
        assert len(y[0]) == 1  # 1D output


async def test_async_federated_learning_integration():
    """Integration test for complete async federated learning."""
    config = AsyncFLConfig(max_staleness=2, update_threshold=2, min_participants=2)

    results = await run_async_federated_learning(
        num_participants=3,
        input_dim=2,
        output_dim=1,
        num_updates_per_participant=5,
        config=config,
        seed=42,
    )

    # Verify results structure
    assert "global_model_params" in results
    assert "global_version" in results
    assert "total_updates_processed" in results
    assert "aggregation_history" in results
    assert "participant_states" in results

    # Verify some training occurred
    assert results["global_version"] > 0
    assert results["total_updates_processed"] > 0
    assert len(results["participant_states"]) == 3

    # Verify each participant contributed
    for participant_id, state in results["participant_states"].items():
        assert state["total_updates"] > 0


def test_aggregation_with_staleness():
    """Test aggregation behavior with stale updates."""
    config = AsyncFLConfig(max_staleness=2, staleness_decay=0.5)
    coordinator = AsyncFLCoordinator(input_dim=1, output_dim=1, config=config)

    import time

    # Create updates with different staleness levels
    updates = [
        AsyncUpdate(
            participant_id="p1",
            model_params={"weights": [[1.0]], "bias": [0.0], "version": 2},
            gradient={"weights": [[0.1]], "bias": [0.01]},
            timestamp=time.time(),
            local_epoch=1,
        ),
        AsyncUpdate(
            participant_id="p2",
            model_params={"weights": [[2.0]], "bias": [1.0], "version": 1},
            gradient={"weights": [[0.2]], "bias": [0.02]},
            timestamp=time.time(),
            local_epoch=1,
        ),
    ]

    # Set coordinator to version 3 to make updates stale
    coordinator.global_version = 3

    # Aggregate updates
    aggregated = coordinator.aggregate_updates(updates)

    assert "weights" in aggregated
    assert "bias" in aggregated
    assert aggregated["version"] == 4  # global_version + 1


def run_all_tests():
    """Run all tests directly without pytest."""
    test_functions = [
        test_simple_async_model,
        test_async_update_creation,
        test_participant_state,
        test_async_fl_config,
        test_async_participant,
        test_async_fl_coordinator,
        test_generate_heterogeneous_data,
        test_aggregation_with_staleness,
    ]

    print("Running Async FL unit tests...")
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

    # Run async test separately
    print("\nRunning async integration test...")
    try:
        asyncio.run(test_async_federated_learning_integration())
        print("✅ test_async_federated_learning_integration")
        passed += 1
    except Exception as e:
        print(f"❌ test_async_federated_learning_integration: {e}")
        failed += 1

    print(f"\nTest Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All Async FL tests passed!")
        return True
    else:
        print("⚠️  Some tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
