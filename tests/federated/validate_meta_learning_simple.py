#!/usr/bin/env python3
"""
Simple validation for Federated Meta-Learning (MAML) implementation.
Tests basic functionality by running the demo and checking outputs.
"""

import sys
import os
import subprocess


def validate_meta_learning_demo():
    """Validate meta-learning by running the demo and checking output."""
    print("🔍 Validating Federated Meta-Learning Implementation...")

    # Run the meta-learning demo
    result = subprocess.run(
        [sys.executable, "src/vega/federated/meta_learning.py"],
        capture_output=True,
        text=True,
        cwd="/home/ncacord/Vega2.0",
    )

    if result.returncode != 0:
        print(f"❌ Demo failed with return code {result.returncode}")
        print(f"Error output: {result.stderr}")
        return False

    output = result.stdout

    # Check for expected output patterns
    expected_patterns = [
        "Federated Meta-Learning (MAML) Demo",
        "Meta-learning rounds:",
        "Participants:",
        "Loss reduction:",
        "Adaptation efficiency:",
        "per-Task Adaptation:",
    ]

    missing_patterns = []
    for pattern in expected_patterns:
        if pattern not in output:
            missing_patterns.append(pattern)

    if missing_patterns:
        print(f"❌ Missing expected patterns: {missing_patterns}")
        print(f"Actual output: {output}")
        return False

    # Extract key metrics from output
    lines = output.split("\n")
    loss_reduction = None
    adaptation_efficiency = None

    for line in lines:
        if "Loss reduction:" in line:
            try:
                loss_reduction = float(line.split("Loss reduction: ")[1].split("%")[0])
            except:
                pass
        elif "Adaptation efficiency:" in line:
            try:
                adaptation_efficiency = float(
                    line.split("Adaptation efficiency: ")[1].split("%")[0]
                )
            except:
                pass

    # Validate that learning occurred
    if loss_reduction is None or loss_reduction <= 0:
        print(f"❌ No meta-learning improvement detected: {loss_reduction}")
        return False

    if adaptation_efficiency is None or adaptation_efficiency <= 0:
        print(f"❌ No adaptation improvement detected: {adaptation_efficiency}")
        return False

    print(f"✅ Meta-Learning validation passed:")
    print(f"   - Loss reduction: {loss_reduction}%")
    print(f"   - Adaptation efficiency: {adaptation_efficiency}%")
    print(f"   - All expected output patterns found")

    return True


def validate_import_functionality():
    """Validate that all classes can be imported correctly."""
    print("\n🔍 Validating Meta-Learning Imports...")

    try:
        # Change to the correct directory
        os.chdir("/home/ncacord/Vega2.0")
        sys.path.insert(0, "src/vega/federated/")

        from meta_learning import (
            Task,
            MAMLConfig,
            SimpleMetaModel,
            FederatedMAML,
            generate_sine_wave_tasks,
            run_federated_maml,
        )

        # Test basic instantiation
        config = MAMLConfig(inner_lr=0.05, outer_lr=0.001)
        model = SimpleMetaModel(input_dim=1, hidden_dim=5, output_dim=1)
        fed_maml = FederatedMAML(input_dim=1, hidden_dim=5, output_dim=1, config=config)

        # Test task generation
        tasks = generate_sine_wave_tasks(num_tasks=2, num_samples_per_task=10)

        print(f"✅ Import validation passed:")
        print(f"   - All classes imported successfully")
        print(f"   - Config created: inner_lr={config.inner_lr}")
        print(
            f"   - Model created: {model.input_dim}→{model.hidden_dim}→{model.output_dim}"
        )
        print(f"   - Generated {len(tasks)} tasks")

        return True

    except Exception as e:
        print(f"❌ Import validation failed: {e}")
        return False


def run_validation():
    """Run complete validation suite."""
    print("🚀 Starting Meta-Learning Validation")
    print("=" * 50)

    validations = [
        validate_import_functionality,
        validate_meta_learning_demo,
    ]

    passed = 0
    failed = 0

    for validation in validations:
        try:
            if validation():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {validation.__name__} failed with exception: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"Validation Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All Meta-Learning validations passed!")
        print("✅ Federated Meta-Learning (MAML) implementation is working correctly")
        return True
    else:
        print("⚠️  Some validations failed")
        return False


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
