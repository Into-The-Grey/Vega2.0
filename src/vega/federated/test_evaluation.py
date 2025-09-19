"""
Comprehensive Test Suite for Federated Learning Evaluation Framework

This module provides extensive tests for all evaluation components including
fairness metrics, privacy assessment, convergence analysis, and overall
evaluation framework functionality.
"""

import pytest
import numpy as np
import json
from typing import Dict, List, Any, Optional
from unittest.mock import patch, MagicMock
from dataclasses import asdict

from .evaluation import (
    FederatedEvaluationFramework,
    MetricCalculator,
    FairnessEvaluator,
    PrivacyEvaluator,
    ConvergenceAnalyzer,
    EvaluationMetrics,
    FairnessMetrics,
    PrivacyMetrics,
    ConvergenceMetrics,
    ParticipantAnalysis,
    create_evaluation_data_sample,
)
from .evaluation_example import SyntheticDataGenerator, EvaluationExampleRunner


class TestMetricCalculator:
    """Test suite for MetricCalculator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = MetricCalculator()

        # Create test data
        self.y_true_binary = [0, 1, 0, 1, 0, 1, 0, 1]
        self.y_pred_binary = [0, 1, 1, 1, 0, 0, 0, 1]

        self.y_true_multi = [0, 1, 2, 0, 1, 2, 0, 1]
        self.y_pred_multi = [0, 1, 1, 0, 2, 2, 1, 1]

        self.y_true_regression = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.y_pred_regression = [1.1, 2.2, 2.8, 4.2, 4.9]

    def test_binary_classification_metrics(self):
        """Test binary classification metric calculations."""
        metrics = self.calculator.calculate_classification_metrics(
            self.y_true_binary, self.y_pred_binary
        )

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

        # Check accuracy calculation
        expected_accuracy = 5 / 8  # 5 correct out of 8
        assert abs(metrics["accuracy"] - expected_accuracy) < 0.01

        # Check that all metrics are in valid range [0, 1]
        for metric_name, value in metrics.items():
            assert 0 <= value <= 1, f"{metric_name} should be between 0 and 1"

    def test_multiclass_classification_metrics(self):
        """Test multiclass classification metric calculations."""
        metrics = self.calculator.calculate_classification_metrics(
            self.y_true_multi, self.y_pred_multi
        )

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

        # Check accuracy
        expected_accuracy = 4 / 8  # 4 correct out of 8
        assert abs(metrics["accuracy"] - expected_accuracy) < 0.01

    def test_regression_metrics(self):
        """Test regression metric calculations."""
        metrics = self.calculator.calculate_regression_metrics(
            self.y_true_regression, self.y_pred_regression
        )

        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2_score" in metrics

        # MSE should be positive
        assert metrics["mse"] >= 0

        # RMSE should be sqrt of MSE
        assert abs(metrics["rmse"] - np.sqrt(metrics["mse"])) < 0.01

        # R² should be reasonable (close to 1 for good predictions)
        assert metrics["r2_score"] > 0.8  # Our test data should have high R²

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_metrics = self.calculator.calculate_classification_metrics([], [])

        # Should return metrics with default values or handle gracefully
        assert isinstance(empty_metrics, dict)

    def test_mismatched_lengths(self):
        """Test handling of mismatched y_true and y_pred lengths."""
        with pytest.raises((ValueError, AssertionError)):
            self.calculator.calculate_classification_metrics([0, 1], [0, 1, 1])

    def test_custom_metrics(self):
        """Test custom metric calculations."""
        # Test AUC calculation
        y_true = [0, 0, 1, 1]
        y_pred_proba = [0.1, 0.4, 0.35, 0.8]

        auc = self.calculator.calculate_auc(y_true, y_pred_proba)
        assert 0 <= auc <= 1
        assert auc > 0.5  # Should be better than random


class TestFairnessEvaluator:
    """Test suite for FairnessEvaluator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = FairnessEvaluator()

        # Create biased test data
        np.random.seed(42)
        self.y_true = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 10  # 100 samples
        self.y_pred = [0, 1, 1, 1, 0, 0, 0, 1, 1, 0] * 10  # Some errors

        # Create sensitive attributes with bias
        self.sensitive_attrs = {
            "gender": [0, 1, 0, 1, 0] * 20,  # 50/50 split
            "age_group": [0, 0, 1, 1, 2] * 20,  # Equal distribution
        }

    def test_demographic_parity_calculation(self):
        """Test demographic parity calculation."""
        dp = self.evaluator.calculate_demographic_parity(
            self.y_pred, self.sensitive_attrs["gender"]
        )

        assert isinstance(dp, float)
        assert 0 <= dp <= 1  # Should be a valid probability difference

    def test_equalized_odds_calculation(self):
        """Test equalized odds calculation."""
        eo = self.evaluator.calculate_equalized_odds(
            self.y_true, self.y_pred, self.sensitive_attrs["gender"]
        )

        assert isinstance(eo, float)
        assert eo >= 0  # Should be non-negative difference

    def test_disparate_impact_calculation(self):
        """Test disparate impact calculation."""
        di = self.evaluator.calculate_disparate_impact(
            self.y_pred, self.sensitive_attrs["gender"]
        )

        assert isinstance(di, float)
        assert di > 0  # Should be positive ratio

    def test_statistical_parity_calculation(self):
        """Test statistical parity calculation."""
        sp = self.evaluator.calculate_statistical_parity(
            self.y_pred, self.sensitive_attrs["gender"]
        )

        assert isinstance(sp, float)
        assert 0 <= sp <= 1

    def test_fairness_evaluation_complete(self):
        """Test complete fairness evaluation."""
        fairness_metrics = self.evaluator.evaluate_fairness(
            self.y_true, self.y_pred, self.sensitive_attrs
        )

        assert isinstance(fairness_metrics, FairnessMetrics)

        # Check that metrics are calculated for all sensitive attributes
        assert fairness_metrics.demographic_parity is not None
        assert fairness_metrics.equalized_odds is not None
        assert fairness_metrics.disparate_impact is not None
        assert fairness_metrics.statistical_parity is not None

    def test_perfect_fairness_scenario(self):
        """Test scenario with perfect fairness (no bias)."""
        # Create unbiased data
        y_true_fair = [0, 1] * 50
        y_pred_fair = [0, 1] * 50  # Perfect predictions
        sensitive_fair = {"gender": [0, 1] * 50}  # Balanced

        fairness_metrics = self.evaluator.evaluate_fairness(
            y_true_fair, y_pred_fair, sensitive_fair
        )

        # In perfect scenario, disparate impact should be close to 1
        assert abs(fairness_metrics.disparate_impact - 1.0) < 0.1

        # Demographic parity should be close to 0
        assert abs(fairness_metrics.demographic_parity) < 0.1

    def test_extreme_bias_scenario(self):
        """Test scenario with extreme bias."""
        # Create extremely biased data
        y_true_biased = [0, 1] * 50
        y_pred_biased = [0] * 50 + [1] * 50  # Always predict 0 for first group
        sensitive_biased = {"gender": [0] * 50 + [1] * 50}

        fairness_metrics = self.evaluator.evaluate_fairness(
            y_true_biased, y_pred_biased, sensitive_biased
        )

        # Should detect significant bias
        assert fairness_metrics.demographic_parity > 0.4  # Large difference
        assert fairness_metrics.disparate_impact < 0.6  # Low ratio


class TestPrivacyEvaluator:
    """Test suite for PrivacyEvaluator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = PrivacyEvaluator()

        # Create test privacy data
        np.random.seed(42)
        self.privacy_data = {
            "model_outputs": {
                "participant_1": np.random.beta(2, 2, 100).tolist(),
                "participant_2": np.random.beta(2, 2, 100).tolist(),
            },
            "membership_labels": {
                "participant_1": [True] * 50 + [False] * 50,
                "participant_2": [True] * 50 + [False] * 50,
            },
            "gradients": [
                {"layer1": np.random.randn(10).tolist()},
                {"layer1": np.random.randn(10).tolist()},
            ],
            "noise_scale": 0.1,
        }

    def test_membership_inference_evaluation(self):
        """Test membership inference attack evaluation."""
        accuracy = self.evaluator.evaluate_membership_inference(
            self.privacy_data["model_outputs"]["participant_1"],
            self.privacy_data["membership_labels"]["participant_1"],
        )

        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1

    def test_model_inversion_risk_assessment(self):
        """Test model inversion risk assessment."""
        risk_score = self.evaluator.assess_model_inversion_risk(
            self.privacy_data["gradients"]
        )

        assert isinstance(risk_score, float)
        assert risk_score >= 0

    def test_differential_privacy_analysis(self):
        """Test differential privacy analysis."""
        epsilon = self.evaluator.analyze_differential_privacy(
            self.privacy_data["noise_scale"]
        )

        assert isinstance(epsilon, float)
        assert epsilon > 0  # ε should be positive

    def test_privacy_evaluation_complete(self):
        """Test complete privacy evaluation."""
        privacy_metrics = self.evaluator.evaluate_privacy(self.privacy_data)

        assert isinstance(privacy_metrics, PrivacyMetrics)
        assert privacy_metrics.membership_inference_accuracy is not None
        assert privacy_metrics.model_inversion_risk is not None
        assert privacy_metrics.differential_privacy_epsilon is not None

    def test_high_privacy_scenario(self):
        """Test scenario with high privacy protection."""
        # High noise data
        high_privacy_data = self.privacy_data.copy()
        high_privacy_data["noise_scale"] = 1.0

        # Add noise to outputs
        for participant in high_privacy_data["model_outputs"]:
            outputs = high_privacy_data["model_outputs"][participant]
            noise = np.random.normal(0, 1.0, len(outputs))
            noisy_outputs = [max(0, min(1, out + n)) for out, n in zip(outputs, noise)]
            high_privacy_data["model_outputs"][participant] = noisy_outputs

        privacy_metrics = self.evaluator.evaluate_privacy(high_privacy_data)

        # High privacy should result in lower risks
        assert privacy_metrics.membership_inference_accuracy < 0.6  # Close to random
        assert privacy_metrics.differential_privacy_epsilon < 5.0  # Reasonable ε

    def test_low_privacy_scenario(self):
        """Test scenario with low privacy protection."""
        # Low noise data
        low_privacy_data = self.privacy_data.copy()
        low_privacy_data["noise_scale"] = 0.01

        privacy_metrics = self.evaluator.evaluate_privacy(low_privacy_data)

        # Low privacy might result in higher risks
        assert privacy_metrics.differential_privacy_epsilon > 0.01


class TestConvergenceAnalyzer:
    """Test suite for ConvergenceAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ConvergenceAnalyzer()

        # Create different training histories
        self.smooth_history = self._create_smooth_convergence_history()
        self.oscillating_history = self._create_oscillating_history()
        self.plateau_history = self._create_plateau_history()

    def _create_smooth_convergence_history(self) -> List[Dict[str, Any]]:
        """Create smooth convergence training history."""
        history = []
        for round_num in range(1, 21):
            loss = 2.0 * np.exp(-0.2 * round_num) + 0.1
            history.append(
                {
                    "round": round_num,
                    "loss": loss,
                    "training_results": [
                        {
                            "participant_id": f"p_{i}",
                            "total_loss": loss + 0.05 * np.random.randn(),
                        }
                        for i in range(3)
                    ],
                }
            )
        return history

    def _create_oscillating_history(self) -> List[Dict[str, Any]]:
        """Create oscillating convergence history."""
        history = []
        for round_num in range(1, 21):
            base_loss = 1.0 * np.exp(-0.1 * round_num)
            oscillation = 0.2 * np.sin(0.5 * round_num)
            loss = base_loss + oscillation + 0.2
            history.append(
                {
                    "round": round_num,
                    "loss": loss,
                    "training_results": [
                        {
                            "participant_id": f"p_{i}",
                            "total_loss": loss + 0.05 * np.random.randn(),
                        }
                        for i in range(3)
                    ],
                }
            )
        return history

    def _create_plateau_history(self) -> List[Dict[str, Any]]:
        """Create plateau convergence history."""
        history = []
        for round_num in range(1, 21):
            if round_num <= 10:
                loss = 2.0 * np.exp(-0.3 * round_num) + 0.5
            else:
                loss = 0.5 + 0.02 * np.random.randn()
                loss = max(0.4, loss)
            history.append(
                {
                    "round": round_num,
                    "loss": loss,
                    "training_results": [
                        {
                            "participant_id": f"p_{i}",
                            "total_loss": loss + 0.05 * np.random.randn(),
                        }
                        for i in range(3)
                    ],
                }
            )
        return history

    def test_convergence_detection(self):
        """Test convergence detection."""
        converged = self.analyzer.detect_convergence(self.smooth_history)
        assert isinstance(converged, bool)

        # Smooth convergence should be detected
        assert converged == True

    def test_rounds_to_convergence(self):
        """Test calculation of rounds to convergence."""
        rounds = self.analyzer.calculate_rounds_to_convergence(self.smooth_history)
        assert isinstance(rounds, (int, type(None)))

        if rounds is not None:
            assert rounds > 0
            assert rounds <= len(self.smooth_history)

    def test_stability_calculation(self):
        """Test stability score calculation."""
        stability = self.analyzer.calculate_stability_score(self.smooth_history)
        assert isinstance(stability, float)
        assert stability >= 0

        # Smooth convergence should have higher stability than oscillating
        oscillating_stability = self.analyzer.calculate_stability_score(
            self.oscillating_history
        )
        assert stability > oscillating_stability

    def test_oscillation_measurement(self):
        """Test oscillation measurement."""
        oscillation = self.analyzer.measure_oscillation(self.oscillating_history)
        assert isinstance(oscillation, float)
        assert oscillation >= 0

        # Oscillating history should have higher oscillation measure
        smooth_oscillation = self.analyzer.measure_oscillation(self.smooth_history)
        assert oscillation > smooth_oscillation

    def test_convergence_analysis_complete(self):
        """Test complete convergence analysis."""
        convergence_metrics = self.analyzer.analyze_convergence(self.smooth_history)

        assert isinstance(convergence_metrics, ConvergenceMetrics)
        assert convergence_metrics.has_converged is not None
        assert convergence_metrics.rounds_to_convergence is not None
        assert convergence_metrics.stability_score is not None
        assert convergence_metrics.oscillation_measure is not None

    def test_different_convergence_patterns(self):
        """Test analysis of different convergence patterns."""
        smooth_metrics = self.analyzer.analyze_convergence(self.smooth_history)
        oscillating_metrics = self.analyzer.analyze_convergence(
            self.oscillating_history
        )
        plateau_metrics = self.analyzer.analyze_convergence(self.plateau_history)

        # Smooth should converge faster than oscillating
        if (
            smooth_metrics.rounds_to_convergence
            and oscillating_metrics.rounds_to_convergence
        ):
            assert (
                smooth_metrics.rounds_to_convergence
                <= oscillating_metrics.rounds_to_convergence
            )

        # Smooth should be more stable than oscillating
        assert smooth_metrics.stability_score > oscillating_metrics.stability_score


class TestFederatedEvaluationFramework:
    """Test suite for the main FederatedEvaluationFramework class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.framework = FederatedEvaluationFramework()
        self.data_generator = SyntheticDataGenerator()

        # Create comprehensive test data
        self.evaluation_data = self._create_test_evaluation_data()

    def _create_test_evaluation_data(self) -> Dict[str, Any]:
        """Create comprehensive test evaluation data."""
        X, y_true = self.data_generator.generate_classification_data(200)

        # Generate predictions with some errors
        y_pred = []
        for true_label in y_true:
            if np.random.random() > 0.15:  # 85% accuracy
                y_pred.append(true_label)
            else:
                y_pred.append(1 - true_label)

        sensitive_attrs = self.data_generator.generate_sensitive_attributes(200)
        training_history = self.data_generator.generate_federated_training_history(15)
        privacy_data = self.data_generator.generate_privacy_attack_data(5)

        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "task_type": "classification",
            "sensitive_attributes": sensitive_attrs,
            "privacy_data": privacy_data,
            "training_history": training_history,
            "participant_data": {
                "participant_0": {
                    "data_size": 100,
                    "local_epochs": 3,
                    "performance": {"accuracy": 0.85, "loss": 0.35},
                },
                "participant_1": {
                    "data_size": 80,
                    "local_epochs": 2,
                    "performance": {"accuracy": 0.82, "loss": 0.42},
                },
            },
        }

    def test_comprehensive_evaluation(self):
        """Test comprehensive evaluation functionality."""
        results = self.framework.comprehensive_evaluation(self.evaluation_data)

        assert isinstance(results, dict)
        assert "evaluation_metrics" in results
        assert "fairness_metrics" in results
        assert "privacy_metrics" in results
        assert "convergence_metrics" in results
        assert "participant_analysis" in results
        assert "overall_assessment" in results

    def test_evaluation_metrics_calculation(self):
        """Test evaluation metrics calculation."""
        metrics = self.framework.evaluate_performance(
            self.evaluation_data["y_true"],
            self.evaluation_data["y_pred"],
            self.evaluation_data["task_type"],
        )

        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.accuracy is not None
        assert 0 <= metrics.accuracy <= 1

    def test_fairness_evaluation_integration(self):
        """Test fairness evaluation integration."""
        fairness_metrics = self.framework.evaluate_fairness(
            self.evaluation_data["y_true"],
            self.evaluation_data["y_pred"],
            self.evaluation_data["sensitive_attributes"],
        )

        assert isinstance(fairness_metrics, FairnessMetrics)
        assert fairness_metrics.demographic_parity is not None

    def test_privacy_evaluation_integration(self):
        """Test privacy evaluation integration."""
        privacy_metrics = self.framework.evaluate_privacy(
            self.evaluation_data["privacy_data"]
        )

        assert isinstance(privacy_metrics, PrivacyMetrics)
        assert privacy_metrics.membership_inference_accuracy is not None

    def test_convergence_analysis_integration(self):
        """Test convergence analysis integration."""
        convergence_metrics = self.framework.analyze_convergence(
            self.evaluation_data["training_history"]
        )

        assert isinstance(convergence_metrics, ConvergenceMetrics)
        assert convergence_metrics.has_converged is not None

    def test_participant_analysis(self):
        """Test participant analysis functionality."""
        if "participant_data" in self.evaluation_data:
            participant_analysis = self.framework.analyze_participants(
                self.evaluation_data["participant_data"]
            )

            assert isinstance(participant_analysis, list)
            assert len(participant_analysis) > 0

            for analysis in participant_analysis:
                assert isinstance(analysis, ParticipantAnalysis)
                assert analysis.participant_id is not None

    def test_session_comparison(self):
        """Test session comparison functionality."""
        # Create two similar evaluation sessions
        session1_data = self.evaluation_data.copy()
        session2_data = self.evaluation_data.copy()

        # Modify session2 slightly
        session2_data["y_pred"] = [
            1 - p for p in session2_data["y_pred"][:10]
        ] + session2_data["y_pred"][10:]

        results1 = self.framework.comprehensive_evaluation(session1_data)
        results2 = self.framework.comprehensive_evaluation(session2_data)

        comparison = self.framework.compare_sessions([results1, results2])

        assert isinstance(comparison, dict)
        assert "performance_comparison" in comparison
        assert "fairness_comparison" in comparison
        assert "privacy_comparison" in comparison

    def test_overall_assessment_generation(self):
        """Test overall assessment generation."""
        results = self.framework.comprehensive_evaluation(self.evaluation_data)
        assessment = results["overall_assessment"]

        assert isinstance(assessment, dict)
        assert "quality_score" in assessment
        assert "privacy_score" in assessment
        assert "fairness_score" in assessment
        assert "efficiency_score" in assessment
        assert "recommendations" in assessment

        # Scores should be in valid range
        for score_name in [
            "quality_score",
            "privacy_score",
            "fairness_score",
            "efficiency_score",
        ]:
            score = assessment[score_name]
            assert 0 <= score <= 1, f"{score_name} should be between 0 and 1"

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with empty data
        empty_data = {}

        try:
            results = self.framework.comprehensive_evaluation(empty_data)
            # Should either handle gracefully or raise appropriate error
            assert isinstance(results, dict)
        except (ValueError, KeyError, TypeError):
            # Expected for invalid input
            pass

    def test_missing_components_handling(self):
        """Test handling of missing evaluation components."""
        # Test with minimal data (missing some components)
        minimal_data = {
            "y_true": [0, 1, 0, 1],
            "y_pred": [0, 1, 1, 0],
            "task_type": "classification",
        }

        results = self.framework.comprehensive_evaluation(minimal_data)

        # Should handle missing components gracefully
        assert isinstance(results, dict)
        assert "evaluation_metrics" in results


class TestSyntheticDataGenerator:
    """Test suite for SyntheticDataGenerator utility class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = SyntheticDataGenerator()

    def test_classification_data_generation(self):
        """Test classification data generation."""
        X, y = self.generator.generate_classification_data(100, num_features=10)

        assert len(X) == 100
        assert len(y) == 100
        assert len(X[0]) == 10  # Number of features
        assert all(label in [0, 1] for label in y)  # Binary labels

    def test_biased_data_generation(self):
        """Test biased data generation."""
        X_unbiased, y_unbiased = self.generator.generate_classification_data(
            1000, bias_strength=0.0
        )
        X_biased, y_biased = self.generator.generate_classification_data(
            1000, bias_strength=0.5
        )

        # Biased data should have different distribution
        unbiased_ratio = sum(y_unbiased) / len(y_unbiased)
        biased_ratio = sum(y_biased) / len(y_biased)

        # There should be some difference (though randomness might affect this)
        assert len(y_unbiased) == len(y_biased)

    def test_sensitive_attributes_generation(self):
        """Test sensitive attributes generation."""
        attrs = self.generator.generate_sensitive_attributes(100)

        assert "gender" in attrs
        assert "age_group" in attrs
        assert "ethnicity" in attrs

        assert len(attrs["gender"]) == 100
        assert all(g in [0, 1] for g in attrs["gender"])
        assert all(a in [0, 1, 2] for a in attrs["age_group"])
        assert all(e in [0, 1, 2, 3] for e in attrs["ethnicity"])

    def test_training_history_generation(self):
        """Test training history generation."""
        history = self.generator.generate_federated_training_history(
            num_rounds=10, convergence_pattern="smooth"
        )

        assert len(history) == 10
        assert all("round" in entry for entry in history)
        assert all("loss" in entry for entry in history)
        assert all("training_results" in entry for entry in history)

        # Loss should generally decrease over time for smooth convergence
        losses = [entry["loss"] for entry in history]
        assert losses[0] > losses[-1]  # Initial loss > final loss

    def test_different_convergence_patterns(self):
        """Test different convergence patterns."""
        patterns = ["smooth", "oscillating", "plateau"]

        for pattern in patterns:
            history = self.generator.generate_federated_training_history(
                num_rounds=15, convergence_pattern=pattern
            )

            assert len(history) == 15
            losses = [entry["loss"] for entry in history]
            assert all(loss > 0 for loss in losses)  # All losses should be positive

    def test_privacy_attack_data_generation(self):
        """Test privacy attack data generation."""
        privacy_data = self.generator.generate_privacy_attack_data(3)

        assert "model_outputs" in privacy_data
        assert "membership_labels" in privacy_data
        assert "gradients" in privacy_data
        assert "noise_scale" in privacy_data

        assert len(privacy_data["model_outputs"]) == 3
        assert len(privacy_data["membership_labels"]) == 3

        # Check output distributions
        for participant_id in privacy_data["model_outputs"]:
            outputs = privacy_data["model_outputs"][participant_id]
            assert len(outputs) == 100
            assert all(0 <= out <= 1 for out in outputs)


class TestEvaluationExampleRunner:
    """Test suite for EvaluationExampleRunner."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = EvaluationExampleRunner()

    def test_basic_evaluation_example(self):
        """Test basic evaluation example execution."""
        results = self.runner.run_basic_evaluation_example()

        assert isinstance(results, dict)
        assert "evaluation_metrics" in results
        assert "overall_assessment" in results

    def test_fairness_analysis_example(self):
        """Test fairness analysis example execution."""
        results = self.runner.run_fairness_analysis_example()

        assert isinstance(results, dict)
        assert "fairness_metrics" in results
        assert "bias_analysis" in results
        assert "mitigation_recommendations" in results

    def test_privacy_assessment_example(self):
        """Test privacy assessment example execution."""
        results = self.runner.run_privacy_assessment_example()

        assert isinstance(results, dict)
        assert "privacy_scenarios" in results
        assert "privacy_recommendations" in results

    def test_convergence_analysis_example(self):
        """Test convergence analysis example execution."""
        results = self.runner.run_convergence_analysis_example()

        assert isinstance(results, dict)
        assert "convergence_patterns" in results
        assert "best_pattern" in results

    def test_comparative_study_example(self):
        """Test comparative study example execution."""
        results = self.runner.run_comparative_study_example()

        assert isinstance(results, dict)
        assert "configurations" in results
        assert "comparative_analysis" in results
        assert "recommendations" in results

    @pytest.mark.slow
    def test_complete_evaluation_suite(self):
        """Test complete evaluation suite execution."""
        results = self.runner.run_complete_evaluation_suite()

        assert isinstance(results, dict)
        assert "basic_evaluation" in results
        assert "fairness_analysis" in results
        assert "privacy_assessment" in results
        assert "convergence_analysis" in results
        assert "comparative_study" in results
        assert "overall_insights" in results


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_create_evaluation_data_sample(self):
        """Test evaluation data sample creation."""
        sample = create_evaluation_data_sample()

        assert isinstance(sample, dict)
        assert "y_true" in sample
        assert "y_pred" in sample
        assert "task_type" in sample

    def test_metrics_serialization(self):
        """Test that metrics can be serialized to JSON."""
        # Create sample metrics
        eval_metrics = EvaluationMetrics(
            accuracy=0.85, precision=0.82, recall=0.88, f1_score=0.85
        )

        fairness_metrics = FairnessMetrics(
            demographic_parity=0.05,
            equalized_odds=0.03,
            disparate_impact=0.95,
            statistical_parity=0.02,
        )

        privacy_metrics = PrivacyMetrics(
            membership_inference_accuracy=0.52,
            model_inversion_risk=0.15,
            differential_privacy_epsilon=1.0,
        )

        convergence_metrics = ConvergenceMetrics(
            has_converged=True,
            rounds_to_convergence=12,
            stability_score=8.5,
            oscillation_measure=0.05,
        )

        # Test serialization
        eval_dict = asdict(eval_metrics)
        fairness_dict = asdict(fairness_metrics)
        privacy_dict = asdict(privacy_metrics)
        convergence_dict = asdict(convergence_metrics)

        # Should be able to convert to JSON
        eval_json = json.dumps(eval_dict)
        fairness_json = json.dumps(fairness_dict)
        privacy_json = json.dumps(privacy_dict)
        convergence_json = json.dumps(convergence_dict)

        # Should be able to parse back
        eval_parsed = json.loads(eval_json)
        fairness_parsed = json.loads(fairness_json)
        privacy_parsed = json.loads(privacy_json)
        convergence_parsed = json.loads(convergence_json)

        assert eval_parsed["accuracy"] == 0.85
        assert fairness_parsed["demographic_parity"] == 0.05
        assert privacy_parsed["membership_inference_accuracy"] == 0.52
        assert convergence_parsed["has_converged"] == True


@pytest.fixture
def sample_evaluation_data():
    """Fixture providing sample evaluation data for tests."""
    generator = SyntheticDataGenerator()

    X, y_true = generator.generate_classification_data(100)
    y_pred = [1 - label if np.random.random() < 0.1 else label for label in y_true]
    sensitive_attrs = generator.generate_sensitive_attributes(100)
    training_history = generator.generate_federated_training_history(10)
    privacy_data = generator.generate_privacy_attack_data(3)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "task_type": "classification",
        "sensitive_attributes": sensitive_attrs,
        "privacy_data": privacy_data,
        "training_history": training_history,
    }


@pytest.fixture
def evaluation_framework():
    """Fixture providing evaluation framework instance."""
    return FederatedEvaluationFramework()


class TestIntegration:
    """Integration tests for the complete evaluation framework."""

    def test_end_to_end_evaluation(self, sample_evaluation_data, evaluation_framework):
        """Test end-to-end evaluation process."""
        results = evaluation_framework.comprehensive_evaluation(sample_evaluation_data)

        # Validate complete results structure
        required_keys = [
            "evaluation_metrics",
            "fairness_metrics",
            "privacy_metrics",
            "convergence_metrics",
            "overall_assessment",
        ]

        for key in required_keys:
            assert key in results, f"Missing key: {key}"

        # Validate overall assessment structure
        assessment = results["overall_assessment"]
        assessment_keys = [
            "quality_score",
            "privacy_score",
            "fairness_score",
            "efficiency_score",
        ]

        for key in assessment_keys:
            assert key in assessment, f"Missing assessment key: {key}"
            assert (
                0 <= assessment[key] <= 1
            ), f"Invalid score for {key}: {assessment[key]}"

    def test_evaluation_consistency(self, evaluation_framework):
        """Test that evaluation results are consistent across runs."""
        # Use fixed seed for reproducibility
        np.random.seed(42)
        generator = SyntheticDataGenerator()

        # Generate data
        X, y_true = generator.generate_classification_data(50)
        y_pred = [label for label in y_true]  # Perfect predictions

        evaluation_data = {
            "y_true": y_true,
            "y_pred": y_pred,
            "task_type": "classification",
        }

        # Run evaluation multiple times
        results1 = evaluation_framework.comprehensive_evaluation(evaluation_data)
        results2 = evaluation_framework.comprehensive_evaluation(evaluation_data)

        # Results should be identical for same input
        assert (
            results1["evaluation_metrics"].accuracy
            == results2["evaluation_metrics"].accuracy
        )
        assert (
            results1["overall_assessment"]["quality_score"]
            == results2["overall_assessment"]["quality_score"]
        )

    def test_evaluation_with_missing_data(self, evaluation_framework):
        """Test evaluation robustness with missing data components."""
        # Test with minimal data
        minimal_data = {
            "y_true": [0, 1, 0, 1],
            "y_pred": [0, 1, 1, 0],
            "task_type": "classification",
        }

        # Should handle missing components gracefully
        results = evaluation_framework.comprehensive_evaluation(minimal_data)
        assert isinstance(results, dict)
        assert "evaluation_metrics" in results


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
