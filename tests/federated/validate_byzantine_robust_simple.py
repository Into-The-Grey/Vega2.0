"""
Simple validation script for Byzantine-robust federated learning.

Tests core functionality with controlled parameters to ensure stability.
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

try:
    from vega.federated.byzantine_robust import (
        ByzantineConfig,
        ParticipantUpdate,
        SimpleByzantineModel,
        ByzantineRobustAggregator,
        ByzantineAttackSimulator,
        run_byzantine_robust_fl,
    )

    print("✓ Byzantine-robust imports successful")

    # Test 1: Basic configuration
    config = ByzantineConfig(
        aggregation_method="trimmed_mean",
        byzantine_ratio=0.25,
        attack_intensity=0.5,  # Reduced intensity for stability
        simulate_attacks=True,
    )
    print("✓ Byzantine configuration created")

    # Test 2: Simple model
    model = SimpleByzantineModel(input_dim=2, output_dim=1, hidden_dim=5)
    x_test = [1.0, 2.0]
    y_test = [0.5]

    output = model.forward(x_test)
    loss = model.compute_loss(x_test, y_test)
    gradients = model.compute_gradient(x_test, y_test)

    print(f"✓ Model forward pass: output={output[0]:.4f}")
    print(f"✓ Model loss computation: loss={loss:.4f}")
    print("✓ Model gradient computation successful")

    # Test 3: Attack simulation
    test_params = {
        "weights": [[0.1, 0.2], [0.3, 0.4]],
        "bias": [0.5, 0.6],
    }

    attacked_params = ByzantineAttackSimulator.apply_gaussian_noise_attack(
        test_params, 0.1
    )
    print("✓ Gaussian noise attack simulation")

    attacked_params = ByzantineAttackSimulator.apply_sign_flip_attack(test_params, 1.0)
    print("✓ Sign flip attack simulation")

    attacked_params = ByzantineAttackSimulator.apply_zero_update_attack(test_params)
    print("✓ Zero update attack simulation")

    # Test 4: Aggregation methods
    aggregator = ByzantineRobustAggregator(config)

    # Create simple test updates
    updates = [
        ParticipantUpdate(
            participant_id=f"p{i}",
            model_params={
                "weights": [[i * 0.01, i * 0.02], [i * 0.03, i * 0.04]],
                "bias": [i * 0.05, i * 0.06],
            },
        )
        for i in range(1, 5)  # Start from 1 to avoid zero values
    ]

    # Test each aggregation method
    methods = ["krum", "multi_krum", "trimmed_mean", "median"]
    for method in methods:
        config.aggregation_method = method
        try:
            result = aggregator.aggregate(updates)
            print(f"✓ {method.upper()} aggregation successful")
        except Exception as e:
            print(f"✗ {method.upper()} aggregation failed: {e}")

    # Test 5: Full Byzantine-robust FL (with conservative parameters)
    conservative_config = ByzantineConfig(
        aggregation_method="trimmed_mean",
        byzantine_ratio=0.2,  # Lower ratio
        attack_intensity=0.1,  # Much lower intensity
        trimmed_mean_beta=0.2,  # More aggressive trimming
    )

    try:
        results = run_byzantine_robust_fl(
            num_participants=5,
            byzantine_ratio=0.2,
            num_rounds=3,  # Fewer rounds for testing
            local_steps=2,
            config=conservative_config,
            seed=42,
        )

        print("✓ Byzantine-robust FL simulation successful")

        history = results["training_history"]
        if history:
            initial_loss = history[0]["global_loss"]
            final_loss = history[-1]["global_loss"]
            print(f"  Initial loss: {initial_loss:.4f}")
            print(f"  Final loss: {final_loss:.4f}")

            # Check for reasonable loss values (not infinite or extremely large)
            if final_loss < 1000 and not (final_loss != final_loss):  # Check for NaN
                print("✓ Loss values are reasonable")
            else:
                print("⚠ Loss values may be unstable")

        byzantine_count = len(results["byzantine_participants"])
        total_participants = 5
        print(f"  Byzantine participants: {byzantine_count}/{total_participants}")
        print(f"  Aggregation method: {results['aggregation_method']}")

    except Exception as e:
        print(f"✗ Byzantine-robust FL simulation failed: {e}")

    print("\n" + "=" * 50)
    print("Byzantine-robust federated learning validation complete")

except ImportError as e:
    print(f"✗ Import failed: {e}")
except Exception as e:
    print(f"✗ Validation failed: {e}")
    import traceback

    traceback.print_exc()
