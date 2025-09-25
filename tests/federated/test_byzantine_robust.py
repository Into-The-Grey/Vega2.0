"""
Unit tests for Byzantine-robust federated learning implementation.

Tests the core functionality of Byzantine-fault tolerant federated learning
including robust aggregation algorithms, attack simulation, and defense
mechanisms against malicious participants.
"""

import unittest
import random
import statistics
from unittest.mock import patch, MagicMock

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.vega.federated.byzantine_robust import (
    ByzantineConfig,
    ParticipantUpdate,
    SimpleByzantineModel,
    ByzantineRobustAggregator,
    ByzantineAttackSimulator,
    run_byzantine_robust_fl,
)


class TestByzantineConfig(unittest.TestCase):
    """Test Byzantine configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ByzantineConfig()

        self.assertEqual(config.aggregation_method, "krum")
        self.assertEqual(config.byzantine_ratio, 0.3)
        self.assertIsNone(config.krum_f)
        self.assertEqual(config.trimmed_mean_beta, 0.1)
        self.assertEqual(config.selection_size, 1)
        self.assertEqual(config.distance_metric, "euclidean")
        self.assertTrue(config.simulate_attacks)
        self.assertIn("gaussian_noise", config.attack_types)

    def test_custom_config(self):
        """Test custom configuration."""
        config = ByzantineConfig(
            aggregation_method="multi_krum",
            byzantine_ratio=0.5,
            krum_f=3,
            attack_intensity=5.0,
        )

        self.assertEqual(config.aggregation_method, "multi_krum")
        self.assertEqual(config.byzantine_ratio, 0.5)
        self.assertEqual(config.krum_f, 3)
        self.assertEqual(config.attack_intensity, 5.0)


class TestParticipantUpdate(unittest.TestCase):
    """Test participant update handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_params = {
            "weights": [[0.1, 0.2], [0.3, 0.4]],
            "bias": [0.5, 0.6],
        }

    def test_participant_update_creation(self):
        """Test participant update creation."""
        update = ParticipantUpdate(
            participant_id="test_participant",
            model_params=self.test_params,
            is_byzantine=True,
            attack_type="gaussian_noise",
        )

        self.assertEqual(update.participant_id, "test_participant")
        self.assertEqual(update.model_params, self.test_params)
        self.assertTrue(update.is_byzantine)
        self.assertEqual(update.attack_type, "gaussian_noise")

    def test_get_flattened_params(self):
        """Test parameter flattening."""
        update = ParticipantUpdate(
            participant_id="test",
            model_params=self.test_params,
        )

        flattened = update.get_flattened_params()
        expected = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

        self.assertEqual(flattened, expected)

    def test_complex_param_flattening(self):
        """Test flattening of complex parameter structures."""
        complex_params = {
            "weights": [[0.1, 0.2], [0.3, 0.4]],
            "bias": [0.5, 0.6],
            "weights2": [[0.7], [0.8]],
            "bias2": [0.9],
        }

        update = ParticipantUpdate(
            participant_id="test",
            model_params=complex_params,
        )

        flattened = update.get_flattened_params()
        expected = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        self.assertEqual(flattened, expected)


class TestSimpleByzantineModel(unittest.TestCase):
    """Test simple Byzantine model."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = SimpleByzantineModel(input_dim=2, output_dim=1, hidden_dim=3)

    def test_model_initialization(self):
        """Test model parameter initialization."""
        self.assertEqual(self.model.input_dim, 2)
        self.assertEqual(self.model.output_dim, 1)
        self.assertEqual(self.model.hidden_dim, 3)

        # Check parameter shapes
        self.assertEqual(len(self.model.params["weights"]), 2)  # input_dim
        self.assertEqual(len(self.model.params["weights"][0]), 3)  # hidden_dim
        self.assertEqual(len(self.model.params["bias"]), 3)  # hidden_dim
        self.assertEqual(len(self.model.params["weights2"]), 3)  # hidden_dim
        self.assertEqual(len(self.model.params["weights2"][0]), 1)  # output_dim
        self.assertEqual(len(self.model.params["bias2"]), 1)  # output_dim

    def test_forward_pass(self):
        """Test forward pass computation."""
        x = [1.0, 2.0]
        output = self.model.forward(x)

        self.assertEqual(len(output), 1)
        self.assertIsInstance(output[0], (int, float))

    def test_loss_computation(self):
        """Test loss computation."""
        x = [1.0, 2.0]
        y_true = [0.5]

        loss = self.model.compute_loss(x, y_true)

        self.assertIsInstance(loss, (int, float))
        self.assertGreaterEqual(loss, 0)

    def test_gradient_computation(self):
        """Test gradient computation."""
        x = [1.0, 2.0]
        y_true = [0.5]

        gradients = self.model.compute_gradient(x, y_true)

        # Check gradient structure
        self.assertIn("weights", gradients)
        self.assertIn("bias", gradients)
        self.assertIn("weights2", gradients)
        self.assertIn("bias2", gradients)

        # Check gradient shapes match parameter shapes
        self.assertEqual(len(gradients["weights"]), 2)
        self.assertEqual(len(gradients["weights"][0]), 3)
        self.assertEqual(len(gradients["bias"]), 3)
        self.assertEqual(len(gradients["weights2"]), 3)
        self.assertEqual(len(gradients["weights2"][0]), 1)
        self.assertEqual(len(gradients["bias2"]), 1)


class TestByzantineAttackSimulator(unittest.TestCase):
    """Test Byzantine attack simulation."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_params = {
            "weights": [[0.1, 0.2], [0.3, 0.4]],
            "bias": [0.5, 0.6],
        }

    def test_gaussian_noise_attack(self):
        """Test Gaussian noise attack."""
        attacked = ByzantineAttackSimulator.apply_gaussian_noise_attack(
            self.test_params, intensity=1.0
        )

        # Structure should remain the same
        self.assertEqual(set(attacked.keys()), set(self.test_params.keys()))
        self.assertEqual(len(attacked["weights"]), len(self.test_params["weights"]))
        self.assertEqual(len(attacked["bias"]), len(self.test_params["bias"]))

        # Values should be different (with high probability)
        self.assertNotEqual(attacked["weights"], self.test_params["weights"])
        self.assertNotEqual(attacked["bias"], self.test_params["bias"])

    def test_sign_flip_attack(self):
        """Test sign flip attack."""
        attacked = ByzantineAttackSimulator.apply_sign_flip_attack(
            self.test_params, intensity=2.0
        )

        # Check that signs are flipped and scaled
        self.assertEqual(attacked["weights"][0][0], -0.1 * 2.0)
        self.assertEqual(attacked["weights"][0][1], -0.2 * 2.0)
        self.assertEqual(attacked["bias"][0], -0.5 * 2.0)

    def test_zero_update_attack(self):
        """Test zero update attack."""
        attacked = ByzantineAttackSimulator.apply_zero_update_attack(self.test_params)

        # All parameters should be zero
        self.assertEqual(attacked["weights"], [[0.0, 0.0], [0.0, 0.0]])
        self.assertEqual(attacked["bias"], [0.0, 0.0])

    def test_apply_attack_dispatcher(self):
        """Test attack application dispatcher."""
        # Test Gaussian noise
        attacked = ByzantineAttackSimulator.apply_attack(
            self.test_params, "gaussian_noise", 1.0
        )
        self.assertNotEqual(attacked, self.test_params)

        # Test unknown attack type (should return original)
        unchanged = ByzantineAttackSimulator.apply_attack(
            self.test_params, "unknown_attack", 1.0
        )
        self.assertEqual(unchanged, self.test_params)


class TestByzantineRobustAggregator(unittest.TestCase):
    """Test Byzantine-robust aggregation algorithms."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ByzantineConfig(byzantine_ratio=0.25, krum_f=2)
        self.aggregator = ByzantineRobustAggregator(self.config)

        # Create test updates
        self.updates = [
            ParticipantUpdate(
                participant_id=f"p{i}",
                model_params={
                    "weights": [[i * 0.1, i * 0.2], [i * 0.3, i * 0.4]],
                    "bias": [i * 0.5, i * 0.6],
                },
            )
            for i in range(5)
        ]

    def test_distance_computation(self):
        """Test distance computation between updates."""
        dist = self.aggregator.compute_distance(self.updates[0], self.updates[1])

        self.assertIsInstance(dist, (int, float))
        self.assertGreaterEqual(dist, 0)

        # Distance to self should be zero
        self_dist = self.aggregator.compute_distance(self.updates[0], self.updates[0])
        self.assertEqual(self_dist, 0)

    def test_krum_aggregation(self):
        """Test Krum aggregation."""
        result = self.aggregator.krum_aggregation(self.updates)

        # Should return parameters from one of the updates
        self.assertIn("weights", result)
        self.assertIn("bias", result)

        # Result should match structure
        self.assertEqual(len(result["weights"]), 2)
        self.assertEqual(len(result["weights"][0]), 2)
        self.assertEqual(len(result["bias"]), 2)

    def test_multi_krum_aggregation(self):
        """Test Multi-Krum aggregation."""
        self.config.selection_size = 3
        result = self.aggregator.multi_krum_aggregation(self.updates)

        self.assertIn("weights", result)
        self.assertIn("bias", result)

    def test_trimmed_mean_aggregation(self):
        """Test trimmed mean aggregation."""
        result = self.aggregator.trimmed_mean_aggregation(self.updates)

        self.assertIn("weights", result)
        self.assertIn("bias", result)

        # Values should be reasonable averages
        self.assertIsInstance(result["weights"][0][0], (int, float))
        self.assertIsInstance(result["bias"][0], (int, float))

    def test_median_aggregation(self):
        """Test median aggregation."""
        result = self.aggregator.median_aggregation(self.updates)

        self.assertIn("weights", result)
        self.assertIn("bias", result)

        # Check if median is correctly computed for a simple case
        expected_median_00 = statistics.median([i * 0.1 for i in range(5)])
        self.assertEqual(result["weights"][0][0], expected_median_00)

    def test_aggregate_dispatcher(self):
        """Test aggregation method dispatcher."""
        # Test each method
        methods = ["krum", "multi_krum", "trimmed_mean", "median"]

        for method in methods:
            self.config.aggregation_method = method
            result = self.aggregator.aggregate(self.updates)

            self.assertIn("weights", result)
            self.assertIn("bias", result)

    def test_empty_updates(self):
        """Test handling of empty update list."""
        result = self.aggregator.aggregate([])
        self.assertEqual(result, {})


class TestByzantineRobustFL(unittest.TestCase):
    """Test complete Byzantine-robust federated learning."""

    def test_basic_simulation(self):
        """Test basic Byzantine-robust FL simulation."""
        config = ByzantineConfig(
            aggregation_method="krum",
            byzantine_ratio=0.25,
            attack_intensity=1.0,
        )

        results = run_byzantine_robust_fl(
            num_participants=4,
            byzantine_ratio=0.25,
            num_rounds=3,
            local_steps=2,
            config=config,
            seed=42,
        )

        # Check result structure
        self.assertIn("training_history", results)
        self.assertIn("final_model_params", results)
        self.assertIn("config", results)
        self.assertIn("byzantine_participants", results)

        # Check training history
        history = results["training_history"]
        self.assertEqual(len(history), 3)  # 3 rounds

        for round_data in history:
            self.assertIn("round", round_data)
            self.assertIn("global_loss", round_data)
            self.assertIn("total_updates", round_data)
            self.assertIn("byzantine_updates", round_data)

    def test_different_aggregation_methods(self):
        """Test different aggregation methods."""
        methods = ["krum", "multi_krum", "trimmed_mean", "median"]

        for method in methods:
            config = ByzantineConfig(
                aggregation_method=method,
                byzantine_ratio=0.25,
                selection_size=2,  # For Multi-Krum
            )

            results = run_byzantine_robust_fl(
                num_participants=4,
                byzantine_ratio=0.25,
                num_rounds=2,
                local_steps=1,
                config=config,
                seed=42,
            )

            self.assertEqual(results["aggregation_method"], method)
            self.assertGreater(len(results["training_history"]), 0)

    def test_high_byzantine_ratio(self):
        """Test with high Byzantine participant ratio."""
        config = ByzantineConfig(
            aggregation_method="median",  # Use median for high Byzantine ratio
            byzantine_ratio=0.5,  # High ratio
        )

        results = run_byzantine_robust_fl(
            num_participants=6,
            byzantine_ratio=0.5,
            num_rounds=2,
            local_steps=1,
            config=config,
            seed=42,
        )

        # Should still complete without errors
        self.assertIn("training_history", results)
        self.assertEqual(len(results["byzantine_participants"]), 3)  # 50% of 6

    def test_attack_simulation(self):
        """Test attack simulation functionality."""
        config = ByzantineConfig(
            simulate_attacks=True,
            attack_types=["gaussian_noise", "sign_flip"],
            attack_intensity=2.0,
        )

        results = run_byzantine_robust_fl(
            num_participants=4,
            byzantine_ratio=0.5,
            num_rounds=3,
            local_steps=1,
            config=config,
            seed=42,
        )

        # Check that attacks were simulated
        history = results["training_history"]
        attacks_seen = set()
        for round_data in history:
            attacks_seen.update(round_data.get("attack_types_seen", []))

        # Should see some attacks
        self.assertGreater(len(attacks_seen), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
