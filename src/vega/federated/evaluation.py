"""
Federated Learning Evaluation Framework

Comprehensive evaluation tools for federated learning systems including
fairness metrics, convergence analysis, privacy leakage assessment,
model performance comparison, and bias detection.
"""

import numpy as np
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import statistics
from collections import defaultdict
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    loss: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class FairnessMetrics:
    """Container for fairness evaluation metrics."""

    demographic_parity: Optional[float] = None
    equalized_odds: Optional[float] = None
    equality_of_opportunity: Optional[float] = None
    statistical_parity: Optional[float] = None
    disparate_impact: Optional[float] = None
    calibration_difference: Optional[float] = None
    individual_fairness: Optional[float] = None


@dataclass
class PrivacyMetrics:
    """Container for privacy assessment metrics."""

    membership_inference_accuracy: Optional[float] = None
    attribute_inference_accuracy: Optional[float] = None
    model_inversion_risk: Optional[float] = None
    differential_privacy_epsilon: Optional[float] = None
    information_leakage_score: Optional[float] = None
    reconstruction_error: Optional[float] = None


@dataclass
class ConvergenceMetrics:
    """Container for convergence analysis metrics."""

    rounds_to_convergence: Optional[int] = None
    final_loss: Optional[float] = None
    convergence_rate: Optional[float] = None
    stability_score: Optional[float] = None
    oscillation_measure: Optional[float] = None
    early_stopping_round: Optional[int] = None


@dataclass
class ParticipantAnalysis:
    """Analysis of individual participant performance."""

    participant_id: str
    data_size: int
    local_epochs: int
    contribution_score: float
    performance_metrics: EvaluationMetrics
    fairness_impact: float
    privacy_risk: float
    communication_cost: float


class MetricCalculator:
    """Base class for metric calculations."""

    @staticmethod
    def calculate_accuracy(y_true: List, y_pred: List) -> float:
        """Calculate accuracy score."""
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")

        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        return correct / len(y_true)

    @staticmethod
    def calculate_precision_recall_f1(
        y_true: List, y_pred: List, average: str = "binary"
    ) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score."""
        if average == "binary":
            # Binary classification
            tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
            fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
            fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            return precision, recall, f1
        else:
            # Multi-class (simplified)
            unique_labels = list(set(y_true + y_pred))
            precisions, recalls, f1s = [], [], []

            for label in unique_labels:
                tp = sum(
                    1
                    for true, pred in zip(y_true, y_pred)
                    if true == label and pred == label
                )
                fp = sum(
                    1
                    for true, pred in zip(y_true, y_pred)
                    if true != label and pred == label
                )
                fn = sum(
                    1
                    for true, pred in zip(y_true, y_pred)
                    if true == label and pred != label
                )

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)

            return (
                statistics.mean(precisions),
                statistics.mean(recalls),
                statistics.mean(f1s),
            )

    @staticmethod
    def calculate_mse(y_true: List[float], y_pred: List[float]) -> float:
        """Calculate mean squared error."""
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")

        return statistics.mean(
            [(true - pred) ** 2 for true, pred in zip(y_true, y_pred)]
        )

    @staticmethod
    def calculate_mae(y_true: List[float], y_pred: List[float]) -> float:
        """Calculate mean absolute error."""
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")

        return statistics.mean([abs(true - pred) for true, pred in zip(y_true, y_pred)])

    @staticmethod
    def calculate_r2_score(y_true: List[float], y_pred: List[float]) -> float:
        """Calculate R-squared score."""
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")

        y_mean = statistics.mean(y_true)
        ss_tot = sum([(y - y_mean) ** 2 for y in y_true])
        ss_res = sum([(true - pred) ** 2 for true, pred in zip(y_true, y_pred)])

        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


class FairnessEvaluator:
    """Evaluator for fairness metrics in federated learning."""

    @staticmethod
    def calculate_demographic_parity(y_pred: List, sensitive_attribute: List) -> float:
        """Calculate demographic parity difference."""
        groups = defaultdict(list)
        for pred, attr in zip(y_pred, sensitive_attribute):
            groups[attr].append(pred)

        group_rates = {}
        for group, predictions in groups.items():
            positive_rate = sum(predictions) / len(predictions)
            group_rates[group] = positive_rate

        if len(group_rates) < 2:
            return 0.0

        rates = list(group_rates.values())
        return max(rates) - min(rates)

    @staticmethod
    def calculate_equalized_odds(
        y_true: List, y_pred: List, sensitive_attribute: List
    ) -> float:
        """Calculate equalized odds difference."""
        groups = defaultdict(lambda: {"true": [], "pred": []})
        for true, pred, attr in zip(y_true, y_pred, sensitive_attribute):
            groups[attr]["true"].append(true)
            groups[attr]["pred"].append(pred)

        group_tpr = {}  # True Positive Rate
        group_fpr = {}  # False Positive Rate

        for group, data in groups.items():
            tp = sum(1 for t, p in zip(data["true"], data["pred"]) if t == 1 and p == 1)
            fn = sum(1 for t, p in zip(data["true"], data["pred"]) if t == 1 and p == 0)
            fp = sum(1 for t, p in zip(data["true"], data["pred"]) if t == 0 and p == 1)
            tn = sum(1 for t, p in zip(data["true"], data["pred"]) if t == 0 and p == 0)

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            group_tpr[group] = tpr
            group_fpr[group] = fpr

        if len(group_tpr) < 2:
            return 0.0

        tpr_diff = max(group_tpr.values()) - min(group_tpr.values())
        fpr_diff = max(group_fpr.values()) - min(group_fpr.values())

        return max(tpr_diff, fpr_diff)

    @staticmethod
    def calculate_disparate_impact(y_pred: List, sensitive_attribute: List) -> float:
        """Calculate disparate impact ratio."""
        groups = defaultdict(list)
        for pred, attr in zip(y_pred, sensitive_attribute):
            groups[attr].append(pred)

        group_rates = {}
        for group, predictions in groups.items():
            positive_rate = sum(predictions) / len(predictions)
            group_rates[group] = positive_rate

        if len(group_rates) < 2:
            return 1.0

        rates = list(group_rates.values())
        return min(rates) / max(rates) if max(rates) > 0 else 1.0


class PrivacyEvaluator:
    """Evaluator for privacy leakage in federated learning."""

    @staticmethod
    def estimate_membership_inference_risk(
        model_outputs: Dict[str, List[float]], membership_labels: Dict[str, List[bool]]
    ) -> float:
        """Estimate membership inference attack success rate."""
        total_correct = 0
        total_samples = 0

        for participant_id, outputs in model_outputs.items():
            if participant_id not in membership_labels:
                continue

            labels = membership_labels[participant_id]
            if len(outputs) != len(labels):
                continue

            # Simple threshold-based membership inference
            threshold = statistics.median(outputs)

            for output, is_member in zip(outputs, labels):
                predicted_member = output > threshold
                if predicted_member == is_member:
                    total_correct += 1
                total_samples += 1

        return total_correct / total_samples if total_samples > 0 else 0.5

    @staticmethod
    def calculate_model_inversion_risk(
        model_gradients: List[Dict[str, Any]], data_sensitivity: float = 1.0
    ) -> float:
        """Estimate model inversion attack risk."""
        if not model_gradients:
            return 0.0

        # Calculate gradient magnitude as proxy for information leakage
        total_magnitude = 0.0
        param_count = 0

        for gradient_dict in model_gradients:
            for param_name, gradient_values in gradient_dict.items():
                if isinstance(gradient_values, (list, tuple)):
                    magnitude = sum([abs(g) for g in gradient_values])
                    total_magnitude += magnitude
                    param_count += len(gradient_values)
                elif isinstance(gradient_values, (int, float)):
                    total_magnitude += abs(gradient_values)
                    param_count += 1

        avg_magnitude = total_magnitude / param_count if param_count > 0 else 0.0

        # Normalize by data sensitivity
        risk_score = min(avg_magnitude * data_sensitivity, 1.0)
        return risk_score

    @staticmethod
    def estimate_differential_privacy_epsilon(
        noise_scale: float, sensitivity: float = 1.0
    ) -> float:
        """Estimate differential privacy epsilon parameter."""
        if noise_scale <= 0:
            return float("inf")

        # Approximate epsilon for Gaussian mechanism
        epsilon = sensitivity / noise_scale
        return epsilon


class ConvergenceAnalyzer:
    """Analyzer for federated learning convergence behavior."""

    @staticmethod
    def analyze_convergence(
        loss_history: List[float], patience: int = 5, min_delta: float = 1e-4
    ) -> ConvergenceMetrics:
        """Analyze convergence behavior from loss history."""
        if len(loss_history) < 2:
            return ConvergenceMetrics()

        # Detect convergence round
        converged_round = None
        for i in range(patience, len(loss_history)):
            recent_losses = loss_history[i - patience : i]
            current_loss = loss_history[i]

            # Check if improvement is below threshold
            improvements = [
                recent_losses[j] - recent_losses[j + 1]
                for j in range(len(recent_losses) - 1)
            ]
            avg_improvement = statistics.mean(improvements)

            if avg_improvement < min_delta:
                converged_round = i
                break

        # Calculate convergence rate
        if len(loss_history) > 1:
            initial_loss = loss_history[0]
            final_loss = loss_history[-1]
            convergence_rate = (initial_loss - final_loss) / len(loss_history)
        else:
            convergence_rate = 0.0

        # Calculate stability (inverse of variance in last 20% of training)
        stability_window = max(1, len(loss_history) // 5)
        recent_losses = loss_history[-stability_window:]
        stability_score = 1.0 / (statistics.variance(recent_losses) + 1e-8)

        # Calculate oscillation measure
        oscillation = 0.0
        for i in range(1, len(loss_history) - 1):
            if (
                loss_history[i] > loss_history[i - 1]
                and loss_history[i] > loss_history[i + 1]
            ) or (
                loss_history[i] < loss_history[i - 1]
                and loss_history[i] < loss_history[i + 1]
            ):
                oscillation += 1
        oscillation_measure = oscillation / max(1, len(loss_history) - 2)

        return ConvergenceMetrics(
            rounds_to_convergence=converged_round,
            final_loss=loss_history[-1],
            convergence_rate=convergence_rate,
            stability_score=stability_score,
            oscillation_measure=oscillation_measure,
            early_stopping_round=converged_round,
        )


class FederatedEvaluationFramework:
    """Comprehensive evaluation framework for federated learning."""

    def __init__(self):
        self.metric_calculator = MetricCalculator()
        self.fairness_evaluator = FairnessEvaluator()
        self.privacy_evaluator = PrivacyEvaluator()
        self.convergence_analyzer = ConvergenceAnalyzer()

        self.evaluation_history = []
        self.participant_analyses = {}

    def evaluate_model_performance(
        self, y_true: List, y_pred: List, task_type: str = "classification"
    ) -> EvaluationMetrics:
        """Evaluate overall model performance."""
        metrics = EvaluationMetrics()

        try:
            if task_type == "classification":
                metrics.accuracy = self.metric_calculator.calculate_accuracy(
                    y_true, y_pred
                )
                precision, recall, f1 = (
                    self.metric_calculator.calculate_precision_recall_f1(y_true, y_pred)
                )
                metrics.precision = precision
                metrics.recall = recall
                metrics.f1_score = f1

            elif task_type == "regression":
                # Convert to float if needed
                y_true_float = [float(y) for y in y_true]
                y_pred_float = [float(y) for y in y_pred]

                metrics.mse = self.metric_calculator.calculate_mse(
                    y_true_float, y_pred_float
                )
                metrics.mae = self.metric_calculator.calculate_mae(
                    y_true_float, y_pred_float
                )
                metrics.r2_score = self.metric_calculator.calculate_r2_score(
                    y_true_float, y_pred_float
                )

        except Exception as e:
            logger.warning(f"Error calculating performance metrics: {e}")

        return metrics

    def evaluate_fairness(
        self, y_true: List, y_pred: List, sensitive_attributes: Dict[str, List]
    ) -> FairnessMetrics:
        """Evaluate fairness across different groups."""
        metrics = FairnessMetrics()

        try:
            for attr_name, attr_values in sensitive_attributes.items():
                if len(attr_values) != len(y_pred):
                    logger.warning(f"Length mismatch for attribute {attr_name}")
                    continue

                # Calculate various fairness metrics
                demo_parity = self.fairness_evaluator.calculate_demographic_parity(
                    y_pred, attr_values
                )
                eq_odds = self.fairness_evaluator.calculate_equalized_odds(
                    y_true, y_pred, attr_values
                )
                disp_impact = self.fairness_evaluator.calculate_disparate_impact(
                    y_pred, attr_values
                )

                # Store the worst-case metrics across attributes
                if (
                    metrics.demographic_parity is None
                    or demo_parity > metrics.demographic_parity
                ):
                    metrics.demographic_parity = demo_parity

                if metrics.equalized_odds is None or eq_odds > metrics.equalized_odds:
                    metrics.equalized_odds = eq_odds

                if (
                    metrics.disparate_impact is None
                    or disp_impact < metrics.disparate_impact
                ):
                    metrics.disparate_impact = disp_impact

        except Exception as e:
            logger.warning(f"Error calculating fairness metrics: {e}")

        return metrics

    def evaluate_privacy(self, model_data: Dict[str, Any]) -> PrivacyMetrics:
        """Evaluate privacy risks and leakage."""
        metrics = PrivacyMetrics()

        try:
            # Membership inference risk
            if "model_outputs" in model_data and "membership_labels" in model_data:
                mi_risk = self.privacy_evaluator.estimate_membership_inference_risk(
                    model_data["model_outputs"], model_data["membership_labels"]
                )
                metrics.membership_inference_accuracy = mi_risk

            # Model inversion risk
            if "gradients" in model_data:
                inv_risk = self.privacy_evaluator.calculate_model_inversion_risk(
                    model_data["gradients"]
                )
                metrics.model_inversion_risk = inv_risk

            # Differential privacy analysis
            if "noise_scale" in model_data:
                epsilon = self.privacy_evaluator.estimate_differential_privacy_epsilon(
                    model_data["noise_scale"]
                )
                metrics.differential_privacy_epsilon = epsilon

        except Exception as e:
            logger.warning(f"Error calculating privacy metrics: {e}")

        return metrics

    def analyze_convergence(
        self, training_history: List[Dict[str, Any]]
    ) -> ConvergenceMetrics:
        """Analyze convergence behavior."""
        try:
            # Extract loss history
            loss_history = []
            for round_data in training_history:
                if "loss" in round_data:
                    loss_history.append(round_data["loss"])
                elif "training_results" in round_data:
                    # Average loss across participants
                    round_losses = []
                    for result in round_data["training_results"]:
                        if "total_loss" in result:
                            round_losses.append(result["total_loss"])
                    if round_losses:
                        loss_history.append(statistics.mean(round_losses))

            return self.convergence_analyzer.analyze_convergence(loss_history)

        except Exception as e:
            logger.warning(f"Error analyzing convergence: {e}")
            return ConvergenceMetrics()

    def analyze_participant_contribution(
        self, participant_data: Dict[str, Any]
    ) -> Dict[str, ParticipantAnalysis]:
        """Analyze individual participant contributions and characteristics."""
        analyses = {}

        try:
            for participant_id, data in participant_data.items():
                # Extract participant metrics
                data_size = data.get("data_size", 0)
                local_epochs = data.get("local_epochs", 1)

                # Calculate contribution score based on data size and performance
                performance = data.get("performance", {})
                contribution_score = data_size * performance.get("accuracy", 0.5)

                # Create performance metrics
                perf_metrics = EvaluationMetrics()
                if "accuracy" in performance:
                    perf_metrics.accuracy = performance["accuracy"]
                if "loss" in performance:
                    perf_metrics.loss = performance["loss"]

                # Estimate fairness impact and privacy risk
                fairness_impact = data.get("fairness_impact", 0.0)
                privacy_risk = data.get("privacy_risk", 0.5)
                communication_cost = data.get("communication_cost", 1.0)

                analysis = ParticipantAnalysis(
                    participant_id=participant_id,
                    data_size=data_size,
                    local_epochs=local_epochs,
                    contribution_score=contribution_score,
                    performance_metrics=perf_metrics,
                    fairness_impact=fairness_impact,
                    privacy_risk=privacy_risk,
                    communication_cost=communication_cost,
                )

                analyses[participant_id] = analysis

        except Exception as e:
            logger.warning(f"Error analyzing participant contributions: {e}")

        return analyses

    def comprehensive_evaluation(
        self, evaluation_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform comprehensive evaluation of federated learning session."""
        results = {
            "performance_metrics": None,
            "fairness_metrics": None,
            "privacy_metrics": None,
            "convergence_metrics": None,
            "participant_analyses": {},
            "overall_assessment": {},
        }

        try:
            # Performance evaluation
            if "y_true" in evaluation_data and "y_pred" in evaluation_data:
                task_type = evaluation_data.get("task_type", "classification")
                results["performance_metrics"] = self.evaluate_model_performance(
                    evaluation_data["y_true"], evaluation_data["y_pred"], task_type
                )

            # Fairness evaluation
            if (
                "y_true" in evaluation_data
                and "y_pred" in evaluation_data
                and "sensitive_attributes" in evaluation_data
            ):
                results["fairness_metrics"] = self.evaluate_fairness(
                    evaluation_data["y_true"],
                    evaluation_data["y_pred"],
                    evaluation_data["sensitive_attributes"],
                )

            # Privacy evaluation
            if "privacy_data" in evaluation_data:
                results["privacy_metrics"] = self.evaluate_privacy(
                    evaluation_data["privacy_data"]
                )

            # Convergence analysis
            if "training_history" in evaluation_data:
                results["convergence_metrics"] = self.analyze_convergence(
                    evaluation_data["training_history"]
                )

            # Participant analysis
            if "participant_data" in evaluation_data:
                results["participant_analyses"] = self.analyze_participant_contribution(
                    evaluation_data["participant_data"]
                )

            # Overall assessment
            results["overall_assessment"] = self._generate_overall_assessment(results)

            # Store in history
            self.evaluation_history.append(results)

        except Exception as e:
            logger.error(f"Error in comprehensive evaluation: {e}")

        return results

    def _generate_overall_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall assessment and recommendations."""
        assessment = {
            "quality_score": 0.0,
            "fairness_score": 0.0,
            "privacy_score": 0.0,
            "efficiency_score": 0.0,
            "recommendations": [],
            "risk_level": "UNKNOWN",
        }

        try:
            scores = []

            # Quality score from performance metrics
            perf_metrics = results.get("performance_metrics")
            if perf_metrics:
                if perf_metrics.accuracy is not None:
                    assessment["quality_score"] = perf_metrics.accuracy
                    scores.append(perf_metrics.accuracy)
                elif perf_metrics.r2_score is not None:
                    assessment["quality_score"] = max(0, perf_metrics.r2_score)
                    scores.append(max(0, perf_metrics.r2_score))

            # Fairness score
            fairness_metrics = results.get("fairness_metrics")
            if fairness_metrics:
                fairness_score = 1.0
                if fairness_metrics.demographic_parity is not None:
                    fairness_score *= max(0, 1 - fairness_metrics.demographic_parity)
                if fairness_metrics.equalized_odds is not None:
                    fairness_score *= max(0, 1 - fairness_metrics.equalized_odds)

                assessment["fairness_score"] = fairness_score
                scores.append(fairness_score)

            # Privacy score
            privacy_metrics = results.get("privacy_metrics")
            if privacy_metrics:
                privacy_score = 1.0
                if privacy_metrics.membership_inference_accuracy is not None:
                    # Lower membership inference accuracy is better for privacy
                    privacy_score *= max(
                        0, 1 - privacy_metrics.membership_inference_accuracy
                    )
                if privacy_metrics.model_inversion_risk is not None:
                    privacy_score *= max(0, 1 - privacy_metrics.model_inversion_risk)

                assessment["privacy_score"] = privacy_score
                scores.append(privacy_score)

            # Efficiency score from convergence
            conv_metrics = results.get("convergence_metrics")
            if conv_metrics:
                efficiency_score = 1.0
                if conv_metrics.rounds_to_convergence is not None:
                    # Faster convergence is better
                    efficiency_score = max(
                        0, 1 - conv_metrics.rounds_to_convergence / 100
                    )
                if conv_metrics.stability_score is not None:
                    efficiency_score *= min(1.0, conv_metrics.stability_score / 10)

                assessment["efficiency_score"] = efficiency_score
                scores.append(efficiency_score)

            # Generate recommendations
            if assessment["quality_score"] < 0.7:
                assessment["recommendations"].append(
                    "Consider increasing local training epochs or improving data quality"
                )

            if assessment["fairness_score"] < 0.8:
                assessment["recommendations"].append(
                    "Implement fairness-aware training techniques or data balancing"
                )

            if assessment["privacy_score"] < 0.7:
                assessment["recommendations"].append(
                    "Add stronger privacy protection (differential privacy, secure aggregation)"
                )

            if assessment["efficiency_score"] < 0.6:
                assessment["recommendations"].append(
                    "Optimize aggregation strategy or implement early stopping"
                )

            # Overall risk level
            overall_score = statistics.mean(scores) if scores else 0.5
            if overall_score >= 0.8:
                assessment["risk_level"] = "LOW"
            elif overall_score >= 0.6:
                assessment["risk_level"] = "MEDIUM"
            else:
                assessment["risk_level"] = "HIGH"

        except Exception as e:
            logger.warning(f"Error generating overall assessment: {e}")

        return assessment

    def save_evaluation_report(self, results: Dict[str, Any], filepath: str):
        """Save evaluation results to file."""
        try:
            # Convert results to JSON-serializable format
            serializable_results = self._make_serializable(results)

            with open(filepath, "w") as f:
                json.dump(serializable_results, f, indent=2, default=str)

            logger.info(f"Evaluation report saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving evaluation report: {e}")

    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if hasattr(obj, "__dict__"):
            return {k: self._make_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)


def create_evaluation_data_sample() -> Dict[str, Any]:
    """Create sample evaluation data for testing."""
    return {
        "y_true": [1, 0, 1, 1, 0, 1, 0, 0, 1, 0],
        "y_pred": [1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
        "task_type": "classification",
        "sensitive_attributes": {
            "gender": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "age_group": [0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
        },
        "privacy_data": {
            "model_outputs": {
                "participant_1": [0.8, 0.2, 0.9, 0.3, 0.1],
                "participant_2": [0.7, 0.4, 0.8, 0.2, 0.3],
            },
            "membership_labels": {
                "participant_1": [True, False, True, False, False],
                "participant_2": [True, True, False, False, True],
            },
            "noise_scale": 0.1,
        },
        "training_history": [
            {"loss": 1.0},
            {"loss": 0.8},
            {"loss": 0.6},
            {"loss": 0.5},
            {"loss": 0.45},
            {"loss": 0.44},
        ],
        "participant_data": {
            "participant_1": {
                "data_size": 1000,
                "local_epochs": 3,
                "performance": {"accuracy": 0.85, "loss": 0.3},
                "fairness_impact": 0.1,
                "privacy_risk": 0.2,
                "communication_cost": 1.2,
            },
            "participant_2": {
                "data_size": 800,
                "local_epochs": 2,
                "performance": {"accuracy": 0.78, "loss": 0.4},
                "fairness_impact": 0.15,
                "privacy_risk": 0.3,
                "communication_cost": 1.0,
            },
        },
    }
