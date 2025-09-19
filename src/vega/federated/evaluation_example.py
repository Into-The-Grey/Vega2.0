"""
Federated Learning Evaluation Examples

Comprehensive examples demonstrating the evaluation framework for federated learning
including fairness analysis, privacy assessment, convergence evaluation, and
comparative studies.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import random
from dataclasses import asdict

from .evaluation import (
    FederatedEvaluationFramework,
    EvaluationMetrics,
    FairnessMetrics,
    PrivacyMetrics,
    ConvergenceMetrics,
    ParticipantAnalysis,
    create_evaluation_data_sample,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generator for synthetic federated learning data for evaluation."""

    @staticmethod
    def generate_classification_data(
        num_samples: int = 1000,
        num_features: int = 20,
        num_classes: int = 2,
        bias_strength: float = 0.0,
    ) -> Tuple[List, List]:
        """Generate synthetic classification data with optional bias."""
        np.random.seed(42)

        # Generate features
        X = np.random.randn(num_samples, num_features)

        # Generate labels with optional bias
        if bias_strength > 0:
            # Create bias in first feature
            bias_factor = bias_strength * X[:, 0]
            probabilities = 1 / (1 + np.exp(-bias_factor))
            y = np.random.binomial(1, probabilities)
        else:
            # Random labels
            y = np.random.randint(0, num_classes, num_samples)

        return X.tolist(), y.tolist()

    @staticmethod
    def generate_sensitive_attributes(num_samples: int) -> Dict[str, List]:
        """Generate synthetic sensitive attributes."""
        np.random.seed(42)

        return {
            "gender": np.random.randint(0, 2, num_samples).tolist(),
            "age_group": np.random.randint(
                0, 3, num_samples
            ).tolist(),  # 0: young, 1: middle, 2: old
            "ethnicity": np.random.randint(
                0, 4, num_samples
            ).tolist(),  # 0-3: different ethnic groups
        }

    @staticmethod
    def generate_federated_training_history(
        num_rounds: int = 20,
        initial_loss: float = 2.0,
        convergence_pattern: str = "smooth",
    ) -> List[Dict[str, Any]]:
        """Generate synthetic training history."""
        history = []

        for round_num in range(1, num_rounds + 1):
            if convergence_pattern == "smooth":
                # Smooth exponential decay
                loss = initial_loss * np.exp(-0.15 * round_num) + 0.1
            elif convergence_pattern == "oscillating":
                # Oscillating convergence
                base_loss = initial_loss * np.exp(-0.1 * round_num)
                oscillation = 0.1 * np.sin(0.5 * round_num)
                loss = base_loss + oscillation + 0.1
            elif convergence_pattern == "plateau":
                # Early plateau
                if round_num <= 10:
                    loss = initial_loss * np.exp(-0.2 * round_num) + 0.3
                else:
                    loss = 0.3 + 0.02 * np.random.randn()
            else:
                # Default smooth
                loss = initial_loss * np.exp(-0.15 * round_num) + 0.1

            # Add some noise
            loss += 0.02 * np.random.randn()
            loss = max(0.05, loss)  # Ensure positive loss

            round_data = {
                "round": round_num,
                "loss": loss,
                "training_results": [
                    {
                        "participant_id": f"participant_{i}",
                        "total_loss": loss + 0.1 * np.random.randn(),
                        "local_epochs": 3,
                        "data_size": np.random.randint(500, 1500),
                    }
                    for i in range(5)  # 5 participants
                ],
            }
            history.append(round_data)

        return history

    @staticmethod
    def generate_privacy_attack_data(num_participants: int = 5) -> Dict[str, Any]:
        """Generate synthetic privacy attack data."""
        np.random.seed(42)

        privacy_data = {
            "model_outputs": {},
            "membership_labels": {},
            "gradients": [],
            "noise_scale": 0.1,
        }

        for i in range(num_participants):
            participant_id = f"participant_{i}"

            # Model outputs for membership inference
            num_samples = 100
            outputs = np.random.beta(
                2, 2, num_samples
            ).tolist()  # Beta distribution for realistic probabilities
            privacy_data["model_outputs"][participant_id] = outputs

            # Membership labels (50% members, 50% non-members)
            membership = [True] * 50 + [False] * 50
            np.random.shuffle(membership)
            privacy_data["membership_labels"][participant_id] = membership

        # Generate synthetic gradients
        for round_num in range(5):
            gradient_dict = {
                "layer1_weights": np.random.randn(100).tolist(),
                "layer1_bias": np.random.randn(10).tolist(),
                "layer2_weights": np.random.randn(50).tolist(),
                "layer2_bias": np.random.randn(5).tolist(),
            }
            privacy_data["gradients"].append(gradient_dict)

        return privacy_data


class EvaluationExampleRunner:
    """Runner for comprehensive evaluation examples."""

    def __init__(self):
        self.evaluator = FederatedEvaluationFramework()
        self.data_generator = SyntheticDataGenerator()
        self.results_history = []

    def run_basic_evaluation_example(self) -> Dict[str, Any]:
        """Run a basic evaluation example."""
        logger.info("Running basic evaluation example...")

        # Generate synthetic data
        X, y_true = self.data_generator.generate_classification_data(
            1000, bias_strength=0.1
        )

        # Simulate model predictions (with some errors)
        y_pred = []
        for true_label in y_true:
            if np.random.random() > 0.15:  # 85% accuracy
                y_pred.append(true_label)
            else:
                y_pred.append(1 - true_label)  # Flip label

        # Generate sensitive attributes
        sensitive_attrs = self.data_generator.generate_sensitive_attributes(1000)

        # Generate training history
        training_history = self.data_generator.generate_federated_training_history(15)

        # Generate privacy data
        privacy_data = self.data_generator.generate_privacy_attack_data(3)

        # Create evaluation data
        evaluation_data = {
            "y_true": y_true,
            "y_pred": y_pred,
            "task_type": "classification",
            "sensitive_attributes": sensitive_attrs,
            "privacy_data": privacy_data,
            "training_history": training_history,
            "participant_data": {
                "participant_0": {
                    "data_size": 800,
                    "local_epochs": 3,
                    "performance": {"accuracy": 0.87, "loss": 0.35},
                    "fairness_impact": 0.08,
                    "privacy_risk": 0.25,
                    "communication_cost": 1.1,
                },
                "participant_1": {
                    "data_size": 1200,
                    "local_epochs": 2,
                    "performance": {"accuracy": 0.83, "loss": 0.42},
                    "fairness_impact": 0.12,
                    "privacy_risk": 0.18,
                    "communication_cost": 0.9,
                },
                "participant_2": {
                    "data_size": 600,
                    "local_epochs": 4,
                    "performance": {"accuracy": 0.78, "loss": 0.48},
                    "fairness_impact": 0.15,
                    "privacy_risk": 0.32,
                    "communication_cost": 1.3,
                },
            },
        }

        # Perform comprehensive evaluation
        results = self.evaluator.comprehensive_evaluation(evaluation_data)
        self.results_history.append(results)

        return results

    def run_fairness_analysis_example(self) -> Dict[str, Any]:
        """Run detailed fairness analysis example."""
        logger.info("Running fairness analysis example...")

        # Generate biased data
        X, y_true = self.data_generator.generate_classification_data(
            1000, bias_strength=0.5
        )  # Higher bias

        # Generate predictions with fairness issues
        sensitive_attrs = self.data_generator.generate_sensitive_attributes(1000)

        y_pred = []
        for i, true_label in enumerate(y_true):
            # Introduce bias based on gender
            gender = sensitive_attrs["gender"][i]
            ethnicity = sensitive_attrs["ethnicity"][i]

            # Base accuracy
            if np.random.random() > 0.2:  # 80% base accuracy
                pred = true_label
            else:
                pred = 1 - true_label

            # Introduce unfairness: lower accuracy for certain groups
            if gender == 0 and ethnicity in [2, 3]:  # Specific demographic
                if np.random.random() < 0.3:  # Additional 30% error rate
                    pred = 1 - pred

            y_pred.append(pred)

        # Evaluate fairness
        fairness_metrics = self.evaluator.evaluate_fairness(
            y_true, y_pred, sensitive_attrs
        )

        # Create detailed analysis
        analysis = {
            "fairness_metrics": fairness_metrics,
            "bias_analysis": self._analyze_bias_patterns(
                y_true, y_pred, sensitive_attrs
            ),
            "mitigation_recommendations": self._generate_fairness_recommendations(
                fairness_metrics
            ),
        }

        return analysis

    def run_privacy_assessment_example(self) -> Dict[str, Any]:
        """Run comprehensive privacy assessment example."""
        logger.info("Running privacy assessment example...")

        # Generate data with various privacy risks
        privacy_scenarios = [
            {"noise_scale": 0.01, "name": "Low Privacy"},
            {"noise_scale": 0.1, "name": "Medium Privacy"},
            {"noise_scale": 1.0, "name": "High Privacy"},
        ]

        privacy_results = {}

        for scenario in privacy_scenarios:
            # Generate privacy data with different noise levels
            privacy_data = self.data_generator.generate_privacy_attack_data(5)
            privacy_data["noise_scale"] = scenario["noise_scale"]

            # Adjust attack success based on privacy level
            for participant_id in privacy_data["model_outputs"]:
                outputs = privacy_data["model_outputs"][participant_id]
                # Add noise to outputs
                noise = np.random.normal(0, scenario["noise_scale"], len(outputs))
                noisy_outputs = [
                    max(0, min(1, out + n)) for out, n in zip(outputs, noise)
                ]
                privacy_data["model_outputs"][participant_id] = noisy_outputs

            # Evaluate privacy
            privacy_metrics = self.evaluator.evaluate_privacy(privacy_data)

            privacy_results[scenario["name"]] = {
                "metrics": privacy_metrics,
                "scenario": scenario,
            }

        return {
            "privacy_scenarios": privacy_results,
            "privacy_recommendations": self._generate_privacy_recommendations(
                privacy_results
            ),
        }

    def run_convergence_analysis_example(self) -> Dict[str, Any]:
        """Run convergence analysis example."""
        logger.info("Running convergence analysis example...")

        # Generate different convergence patterns
        convergence_patterns = ["smooth", "oscillating", "plateau"]
        convergence_results = {}

        for pattern in convergence_patterns:
            training_history = self.data_generator.generate_federated_training_history(
                25, convergence_pattern=pattern
            )

            convergence_metrics = self.evaluator.analyze_convergence(training_history)

            convergence_results[pattern] = {
                "metrics": convergence_metrics,
                "training_history": training_history,
                "analysis": self._analyze_convergence_pattern(
                    convergence_metrics, pattern
                ),
            }

        return {
            "convergence_patterns": convergence_results,
            "best_pattern": self._select_best_convergence(convergence_results),
        }

    def run_comparative_study_example(self) -> Dict[str, Any]:
        """Run comparative study between different federated learning configurations."""
        logger.info("Running comparative study example...")

        # Define different FL configurations
        configurations = [
            {
                "name": "Standard FedAvg",
                "participants": 5,
                "local_epochs": 3,
                "privacy_level": "low",
            },
            {
                "name": "Secure FedAvg",
                "participants": 5,
                "local_epochs": 2,
                "privacy_level": "high",
            },
            {
                "name": "Large Scale FL",
                "participants": 20,
                "local_epochs": 1,
                "privacy_level": "medium",
            },
        ]

        comparison_results = {}

        for config in configurations:
            # Generate evaluation data for this configuration
            evaluation_data = self._generate_config_evaluation_data(config)

            # Evaluate
            results = self.evaluator.comprehensive_evaluation(evaluation_data)

            comparison_results[config["name"]] = {
                "results": results,
                "configuration": config,
            }

        # Perform comparative analysis
        comparative_analysis = self._perform_comparative_analysis(comparison_results)

        return {
            "configurations": comparison_results,
            "comparative_analysis": comparative_analysis,
            "recommendations": self._generate_configuration_recommendations(
                comparative_analysis
            ),
        }

    def run_complete_evaluation_suite(self) -> Dict[str, Any]:
        """Run the complete evaluation suite."""
        logger.info("Running complete evaluation suite...")

        suite_results = {
            "basic_evaluation": self.run_basic_evaluation_example(),
            "fairness_analysis": self.run_fairness_analysis_example(),
            "privacy_assessment": self.run_privacy_assessment_example(),
            "convergence_analysis": self.run_convergence_analysis_example(),
            "comparative_study": self.run_comparative_study_example(),
        }

        # Generate overall insights
        suite_results["overall_insights"] = self._generate_overall_insights(
            suite_results
        )

        return suite_results

    def _analyze_bias_patterns(
        self, y_true: List, y_pred: List, sensitive_attrs: Dict[str, List]
    ) -> Dict[str, Any]:
        """Analyze bias patterns in predictions."""
        bias_analysis = {}

        for attr_name, attr_values in sensitive_attrs.items():
            groups = {}
            for i, (true, pred, attr) in enumerate(zip(y_true, y_pred, attr_values)):
                if attr not in groups:
                    groups[attr] = {"correct": 0, "total": 0}

                groups[attr]["total"] += 1
                if true == pred:
                    groups[attr]["correct"] += 1

            # Calculate accuracy per group
            group_accuracies = {}
            for group, stats in groups.items():
                accuracy = (
                    stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                )
                group_accuracies[group] = accuracy

            # Find most and least advantaged groups
            accuracy_values = list(group_accuracies.values())
            max_accuracy = max(accuracy_values) if accuracy_values else 0
            min_accuracy = min(accuracy_values) if accuracy_values else 0

            most_advantaged = None
            least_advantaged = None
            for group, accuracy in group_accuracies.items():
                if accuracy == max_accuracy:
                    most_advantaged = group
                if accuracy == min_accuracy:
                    least_advantaged = group

            bias_analysis[attr_name] = {
                "group_accuracies": group_accuracies,
                "accuracy_gap": max_accuracy - min_accuracy,
                "most_advantaged_group": most_advantaged,
                "least_advantaged_group": least_advantaged,
            }

        return bias_analysis

    def _generate_fairness_recommendations(
        self, fairness_metrics: FairnessMetrics
    ) -> List[str]:
        """Generate fairness improvement recommendations."""
        recommendations = []

        if (
            hasattr(fairness_metrics, "demographic_parity")
            and fairness_metrics.demographic_parity
        ):
            if fairness_metrics.demographic_parity > 0.1:
                recommendations.append(
                    "High demographic parity gap detected. Consider data balancing or fairness constraints."
                )

        if (
            hasattr(fairness_metrics, "equalized_odds")
            and fairness_metrics.equalized_odds
        ):
            if fairness_metrics.equalized_odds > 0.1:
                recommendations.append(
                    "Equalized odds violation detected. Implement post-processing fairness techniques."
                )

        if (
            hasattr(fairness_metrics, "disparate_impact")
            and fairness_metrics.disparate_impact
        ):
            if fairness_metrics.disparate_impact < 0.8:
                recommendations.append(
                    "Disparate impact ratio below 0.8. Review model features and training data."
                )

        if not recommendations:
            recommendations.append("Fairness metrics are within acceptable ranges.")

        return recommendations

    def _generate_privacy_recommendations(
        self, privacy_results: Dict[str, Any]
    ) -> List[str]:
        """Generate privacy improvement recommendations."""
        recommendations = []

        # Analyze membership inference risks
        high_risk_scenarios = []
        for scenario_name, scenario_data in privacy_results.items():
            metrics = scenario_data["metrics"]
            if (
                hasattr(metrics, "membership_inference_accuracy")
                and metrics.membership_inference_accuracy
            ):
                if metrics.membership_inference_accuracy > 0.7:
                    high_risk_scenarios.append(scenario_name)

        if high_risk_scenarios:
            recommendations.append(
                f"High membership inference risk in {', '.join(high_risk_scenarios)}. "
                "Increase differential privacy noise or use secure aggregation."
            )

        recommendations.append(
            "Consider implementing federated learning with differential privacy."
        )
        recommendations.append(
            "Use secure multi-party computation for sensitive operations."
        )

        return recommendations

    def _analyze_convergence_pattern(
        self, metrics: ConvergenceMetrics, pattern: str
    ) -> Dict[str, Any]:
        """Analyze convergence pattern characteristics."""
        analysis = {"pattern_type": pattern, "convergence_quality": "unknown"}

        if hasattr(metrics, "rounds_to_convergence") and metrics.rounds_to_convergence:
            if metrics.rounds_to_convergence < 10:
                analysis["convergence_speed"] = "fast"
            elif metrics.rounds_to_convergence < 20:
                analysis["convergence_speed"] = "moderate"
            else:
                analysis["convergence_speed"] = "slow"

        if hasattr(metrics, "stability_score") and metrics.stability_score:
            if metrics.stability_score > 5:
                analysis["stability"] = "high"
            elif metrics.stability_score > 2:
                analysis["stability"] = "moderate"
            else:
                analysis["stability"] = "low"

        if hasattr(metrics, "oscillation_measure") and metrics.oscillation_measure:
            if metrics.oscillation_measure < 0.1:
                analysis["smoothness"] = "smooth"
            elif metrics.oscillation_measure < 0.3:
                analysis["smoothness"] = "moderate"
            else:
                analysis["smoothness"] = "oscillatory"

        return analysis

    def _select_best_convergence(self, convergence_results: Dict[str, Any]) -> str:
        """Select the best convergence pattern."""
        best_pattern = None
        best_score = -1

        for pattern, data in convergence_results.items():
            metrics = data["metrics"]
            score = 0

            # Score based on convergence speed and stability
            if (
                hasattr(metrics, "rounds_to_convergence")
                and metrics.rounds_to_convergence
            ):
                speed_score = max(0, 1 - metrics.rounds_to_convergence / 25)
                score += speed_score

            if hasattr(metrics, "stability_score") and metrics.stability_score:
                stability_score = min(1, metrics.stability_score / 10)
                score += stability_score

            if hasattr(metrics, "oscillation_measure") and metrics.oscillation_measure:
                smoothness_score = max(0, 1 - metrics.oscillation_measure)
                score += smoothness_score

            if score > best_score:
                best_score = score
                best_pattern = pattern

        return best_pattern or "smooth"

    def _generate_config_evaluation_data(
        self, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate evaluation data for a specific configuration."""
        # Adjust data generation based on configuration
        num_participants = config["participants"]
        local_epochs = config["local_epochs"]
        privacy_level = config["privacy_level"]

        # Generate basic data
        X, y_true = self.data_generator.generate_classification_data(1000)

        # Adjust predictions based on configuration
        if num_participants > 10:
            # More participants -> potentially lower individual accuracy
            base_accuracy = 0.75
        else:
            base_accuracy = 0.85

        y_pred = []
        for true_label in y_true:
            if np.random.random() > (1 - base_accuracy):
                y_pred.append(true_label)
            else:
                y_pred.append(1 - true_label)

        # Generate privacy data based on privacy level
        privacy_noise = {"low": 0.01, "medium": 0.1, "high": 0.5}
        privacy_data = self.data_generator.generate_privacy_attack_data(
            num_participants
        )
        privacy_data["noise_scale"] = privacy_noise[privacy_level]

        # Generate training history
        rounds = max(10, 20 - local_epochs * 2)  # More local epochs -> fewer rounds
        training_history = self.data_generator.generate_federated_training_history(
            rounds
        )

        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "task_type": "classification",
            "privacy_data": privacy_data,
            "training_history": training_history,
        }

    def _perform_comparative_analysis(
        self, comparison_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform comparative analysis between configurations."""
        analysis = {
            "performance_ranking": [],
            "privacy_ranking": [],
            "efficiency_ranking": [],
        }

        # Extract metrics for comparison
        config_metrics = {}
        for config_name, data in comparison_results.items():
            results = data["results"]
            overall_assessment = results.get("overall_assessment", {})

            config_metrics[config_name] = {
                "quality_score": overall_assessment.get("quality_score", 0),
                "privacy_score": overall_assessment.get("privacy_score", 0),
                "efficiency_score": overall_assessment.get("efficiency_score", 0),
            }

        # Rank configurations
        analysis["performance_ranking"] = sorted(
            config_metrics.keys(),
            key=lambda x: config_metrics[x]["quality_score"],
            reverse=True,
        )

        analysis["privacy_ranking"] = sorted(
            config_metrics.keys(),
            key=lambda x: config_metrics[x]["privacy_score"],
            reverse=True,
        )

        analysis["efficiency_ranking"] = sorted(
            config_metrics.keys(),
            key=lambda x: config_metrics[x]["efficiency_score"],
            reverse=True,
        )

        return analysis

    def _generate_configuration_recommendations(
        self, analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate configuration recommendations."""
        recommendations = []

        if "performance_ranking" in analysis:
            best_performance = analysis["performance_ranking"][0]
            recommendations.append(f"For best performance, use: {best_performance}")

        if "privacy_ranking" in analysis:
            best_privacy = analysis["privacy_ranking"][0]
            recommendations.append(f"For best privacy protection, use: {best_privacy}")

        if "efficiency_ranking" in analysis:
            best_efficiency = analysis["efficiency_ranking"][0]
            recommendations.append(f"For best efficiency, use: {best_efficiency}")

        recommendations.append(
            "Consider the trade-offs between performance, privacy, and efficiency."
        )
        recommendations.append(
            "Validate results with real-world data before deployment."
        )

        return recommendations

    def _generate_overall_insights(
        self, suite_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate overall insights from the complete evaluation suite."""
        insights = {
            "key_findings": [],
            "recommendations": [],
            "risk_assessment": "medium",
        }

        # Analyze basic evaluation
        basic_eval = suite_results.get("basic_evaluation", {})
        if "overall_assessment" in basic_eval:
            assessment = basic_eval["overall_assessment"]
            quality = assessment.get("quality_score", 0)

            if quality > 0.8:
                insights["key_findings"].append("High model quality achieved")
            elif quality < 0.6:
                insights["key_findings"].append("Model quality needs improvement")
                insights["recommendations"].append(
                    "Increase training epochs or improve data quality"
                )

        # Analyze fairness
        fairness_analysis = suite_results.get("fairness_analysis", {})
        if "fairness_metrics" in fairness_analysis:
            insights["key_findings"].append(
                "Fairness analysis completed - review bias patterns"
            )
            insights["recommendations"].extend(
                fairness_analysis.get("mitigation_recommendations", [])
            )

        # Analyze privacy
        privacy_assessment = suite_results.get("privacy_assessment", {})
        if "privacy_recommendations" in privacy_assessment:
            insights["recommendations"].extend(
                privacy_assessment["privacy_recommendations"]
            )

        # Overall risk assessment
        if len([r for r in insights["recommendations"] if "risk" in r.lower()]) > 2:
            insights["risk_assessment"] = "high"
        elif len(insights["recommendations"]) < 3:
            insights["risk_assessment"] = "low"

        return insights

    def save_results(self, results: Dict[str, Any], output_dir: str):
        """Save evaluation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save main results
        with open(output_path / "evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save summary
        summary = self._create_summary_report(results)
        with open(output_path / "evaluation_summary.txt", "w") as f:
            f.write(summary)

        logger.info(f"Evaluation results saved to {output_path}")

    def _create_summary_report(self, results: Dict[str, Any]) -> str:
        """Create a human-readable summary report."""
        report = "FEDERATED LEARNING EVALUATION SUMMARY\n"
        report += "=" * 50 + "\n\n"

        if "overall_insights" in results:
            insights = results["overall_insights"]

            report += "KEY FINDINGS:\n"
            for finding in insights.get("key_findings", []):
                report += f"  • {finding}\n"

            report += f"\nRISK ASSESSMENT: {insights.get('risk_assessment', 'unknown').upper()}\n"

            report += "\nRECOMMENDATIONS:\n"
            for rec in insights.get("recommendations", []):
                report += f"  • {rec}\n"

        return report


async def run_evaluation_examples():
    """Run all evaluation examples."""
    print("=" * 60)
    print("FEDERATED LEARNING EVALUATION EXAMPLES")
    print("=" * 60)

    runner = EvaluationExampleRunner()

    print("\n1. Running Basic Evaluation Example...")
    basic_results = runner.run_basic_evaluation_example()
    print(f"   ✅ Basic evaluation completed")

    print("\n2. Running Fairness Analysis Example...")
    fairness_results = runner.run_fairness_analysis_example()
    print(f"   ✅ Fairness analysis completed")

    print("\n3. Running Privacy Assessment Example...")
    privacy_results = runner.run_privacy_assessment_example()
    print(f"   ✅ Privacy assessment completed")

    print("\n4. Running Convergence Analysis Example...")
    convergence_results = runner.run_convergence_analysis_example()
    print(f"   ✅ Convergence analysis completed")

    print("\n5. Running Comparative Study Example...")
    comparative_results = runner.run_comparative_study_example()
    print(f"   ✅ Comparative study completed")

    print("\n6. Running Complete Evaluation Suite...")
    complete_results = runner.run_complete_evaluation_suite()
    print(f"   ✅ Complete evaluation suite completed")

    # Display summary of results
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    if "overall_insights" in complete_results:
        insights = complete_results["overall_insights"]

        print("\nKey Findings:")
        for finding in insights.get("key_findings", []):
            print(f"  • {finding}")

        print(
            f"\nOverall Risk Assessment: {insights.get('risk_assessment', 'unknown').upper()}"
        )

        print("\nTop Recommendations:")
        for i, rec in enumerate(insights.get("recommendations", [])[:3], 1):
            print(f"  {i}. {rec}")

    # Save complete results
    runner.save_results(complete_results, "evaluation_results")
    print(f"\n📁 Complete results saved to ./evaluation_results/")

    return complete_results


if __name__ == "__main__":
    asyncio.run(run_evaluation_examples())
