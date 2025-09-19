"""
Federated Learning Evaluation CLI

Command-line interface for comprehensive evaluation of federated learning systems.
Supports fairness analysis, convergence assessment, privacy evaluation, and
performance comparison.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import click
import csv

from .evaluation import (
    FederatedEvaluationFramework,
    create_evaluation_data_sample,
    EvaluationMetrics,
    FairnessMetrics,
    PrivacyMetrics,
    ConvergenceMetrics,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
def evaluation_cli():
    """Federated Learning Evaluation CLI"""
    pass


@evaluation_cli.command()
@click.option(
    "--session-file",
    "-s",
    required=True,
    type=click.Path(exists=True),
    help="Federated learning session file to evaluate",
)
@click.option(
    "--output", "-o", type=click.Path(), help="Output directory for evaluation results"
)
@click.option(
    "--task-type",
    default="classification",
    type=click.Choice(["classification", "regression"]),
    help="Type of ML task",
)
@click.option("--include-fairness", is_flag=True, help="Include fairness analysis")
@click.option("--include-privacy", is_flag=True, help="Include privacy assessment")
@click.option(
    "--sensitive-attrs",
    type=str,
    help="Comma-separated list of sensitive attribute files",
)
def evaluate_session(
    session_file: str,
    output: Optional[str],
    task_type: str,
    include_fairness: bool,
    include_privacy: bool,
    sensitive_attrs: Optional[str],
):
    """Evaluate a complete federated learning session."""

    click.echo(f"Evaluating federated learning session: {session_file}")

    try:
        # Load session data
        with open(session_file, "r") as f:
            session_data = json.load(f)

        # Initialize evaluation framework
        evaluator = FederatedEvaluationFramework()

        # Prepare evaluation data
        evaluation_data = prepare_evaluation_data(
            session_data, task_type, include_fairness, include_privacy, sensitive_attrs
        )

        # Perform comprehensive evaluation
        results = evaluator.comprehensive_evaluation(evaluation_data)

        # Display results summary
        display_evaluation_summary(results)

        # Save detailed results
        if output:
            output_path = Path(output)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save evaluation report
            report_file = output_path / "evaluation_report.json"
            evaluator.save_evaluation_report(results, str(report_file))

            # Save CSV summary
            csv_file = output_path / "evaluation_summary.csv"
            save_evaluation_csv(results, str(csv_file))

            click.echo(f"Detailed results saved to {output_path}")

    except Exception as e:
        click.echo(f"❌ Error evaluating session: {e}", err=True)


@evaluation_cli.command()
@click.option(
    "--true-labels",
    "-t",
    required=True,
    type=click.Path(exists=True),
    help="File containing true labels",
)
@click.option(
    "--predictions",
    "-p",
    required=True,
    type=click.Path(exists=True),
    help="File containing model predictions",
)
@click.option(
    "--task-type",
    default="classification",
    type=click.Choice(["classification", "regression"]),
    help="Type of ML task",
)
@click.option(
    "--output", "-o", type=click.Path(), help="Output file for performance metrics"
)
def evaluate_performance(
    true_labels: str, predictions: str, task_type: str, output: Optional[str]
):
    """Evaluate model performance metrics."""

    click.echo("Evaluating model performance...")

    try:
        # Load data
        y_true = load_data_file(true_labels)
        y_pred = load_data_file(predictions)

        if len(y_true) != len(y_pred):
            raise ValueError("True labels and predictions must have same length")

        # Initialize evaluator
        evaluator = FederatedEvaluationFramework()

        # Calculate performance metrics
        metrics = evaluator.evaluate_model_performance(y_true, y_pred, task_type)

        # Display results
        click.echo("\nPerformance Metrics:")
        display_performance_metrics(metrics)

        # Save results
        if output:
            with open(output, "w") as f:
                json.dump(metrics.__dict__, f, indent=2, default=str)
            click.echo(f"Performance metrics saved to {output}")

    except Exception as e:
        click.echo(f"❌ Error evaluating performance: {e}", err=True)


@evaluation_cli.command()
@click.option(
    "--true-labels",
    "-t",
    required=True,
    type=click.Path(exists=True),
    help="File containing true labels",
)
@click.option(
    "--predictions",
    "-p",
    required=True,
    type=click.Path(exists=True),
    help="File containing model predictions",
)
@click.option(
    "--sensitive-attrs",
    "-a",
    required=True,
    type=click.Path(exists=True),
    help="File containing sensitive attributes",
)
@click.option(
    "--output", "-o", type=click.Path(), help="Output file for fairness metrics"
)
def evaluate_fairness(
    true_labels: str, predictions: str, sensitive_attrs: str, output: Optional[str]
):
    """Evaluate fairness metrics."""

    click.echo("Evaluating fairness metrics...")

    try:
        # Load data
        y_true = load_data_file(true_labels)
        y_pred = load_data_file(predictions)

        # Load sensitive attributes
        sensitive_data = load_sensitive_attributes(sensitive_attrs)

        # Initialize evaluator
        evaluator = FederatedEvaluationFramework()

        # Calculate fairness metrics
        metrics = evaluator.evaluate_fairness(y_true, y_pred, sensitive_data)

        # Display results
        click.echo("\nFairness Metrics:")
        display_fairness_metrics(metrics)

        # Save results
        if output:
            with open(output, "w") as f:
                json.dump(metrics.__dict__, f, indent=2, default=str)
            click.echo(f"Fairness metrics saved to {output}")

    except Exception as e:
        click.echo(f"❌ Error evaluating fairness: {e}", err=True)


@evaluation_cli.command()
@click.option(
    "--model-data",
    "-m",
    required=True,
    type=click.Path(exists=True),
    help="File containing model data for privacy analysis",
)
@click.option(
    "--output", "-o", type=click.Path(), help="Output file for privacy metrics"
)
def evaluate_privacy(model_data: str, output: Optional[str]):
    """Evaluate privacy risks and leakage."""

    click.echo("Evaluating privacy risks...")

    try:
        # Load model data
        with open(model_data, "r") as f:
            privacy_data = json.load(f)

        # Initialize evaluator
        evaluator = FederatedEvaluationFramework()

        # Calculate privacy metrics
        metrics = evaluator.evaluate_privacy(privacy_data)

        # Display results
        click.echo("\nPrivacy Metrics:")
        display_privacy_metrics(metrics)

        # Save results
        if output:
            with open(output, "w") as f:
                json.dump(metrics.__dict__, f, indent=2, default=str)
            click.echo(f"Privacy metrics saved to {output}")

    except Exception as e:
        click.echo(f"❌ Error evaluating privacy: {e}", err=True)


@evaluation_cli.command()
@click.option(
    "--training-history",
    "-h",
    required=True,
    type=click.Path(exists=True),
    help="File containing training history",
)
@click.option(
    "--output", "-o", type=click.Path(), help="Output file for convergence analysis"
)
def analyze_convergence(training_history: str, output: Optional[str]):
    """Analyze convergence behavior."""

    click.echo("Analyzing convergence behavior...")

    try:
        # Load training history
        with open(training_history, "r") as f:
            history_data = json.load(f)

        # Initialize evaluator
        evaluator = FederatedEvaluationFramework()

        # Analyze convergence
        metrics = evaluator.analyze_convergence(history_data)

        # Display results
        click.echo("\nConvergence Analysis:")
        display_convergence_metrics(metrics)

        # Save results
        if output:
            with open(output, "w") as f:
                json.dump(metrics.__dict__, f, indent=2, default=str)
            click.echo(f"Convergence analysis saved to {output}")

    except Exception as e:
        click.echo(f"❌ Error analyzing convergence: {e}", err=True)


@evaluation_cli.command()
@click.option(
    "--session1",
    required=True,
    type=click.Path(exists=True),
    help="First session file to compare",
)
@click.option(
    "--session2",
    required=True,
    type=click.Path(exists=True),
    help="Second session file to compare",
)
@click.option(
    "--output", "-o", type=click.Path(), help="Output file for comparison results"
)
def compare_sessions(session1: str, session2: str, output: Optional[str]):
    """Compare two federated learning sessions."""

    click.echo(f"Comparing sessions: {session1} vs {session2}")

    try:
        # Load session data
        with open(session1, "r") as f:
            data1 = json.load(f)
        with open(session2, "r") as f:
            data2 = json.load(f)

        # Initialize evaluator
        evaluator = FederatedEvaluationFramework()

        # Evaluate both sessions
        eval_data1 = prepare_evaluation_data(
            data1, "classification", False, False, None
        )
        eval_data2 = prepare_evaluation_data(
            data2, "classification", False, False, None
        )

        results1 = evaluator.comprehensive_evaluation(eval_data1)
        results2 = evaluator.comprehensive_evaluation(eval_data2)

        # Compare results
        comparison = compare_evaluation_results(results1, results2)

        # Display comparison
        click.echo("\nSession Comparison:")
        display_comparison_results(comparison)

        # Save comparison
        if output:
            with open(output, "w") as f:
                json.dump(comparison, f, indent=2, default=str)
            click.echo(f"Comparison results saved to {output}")

    except Exception as e:
        click.echo(f"❌ Error comparing sessions: {e}", err=True)


@evaluation_cli.command()
@click.option(
    "--output", "-o", type=click.Path(), help="Output directory for example results"
)
def run_example(output: Optional[str]):
    """Run evaluation framework example."""

    click.echo("Running evaluation framework example...")

    try:
        # Create sample evaluation data
        evaluation_data = create_evaluation_data_sample()

        # Initialize evaluator
        evaluator = FederatedEvaluationFramework()

        # Perform comprehensive evaluation
        results = evaluator.comprehensive_evaluation(evaluation_data)

        # Display results
        click.echo("\nExample Evaluation Results:")
        display_evaluation_summary(results)

        # Save results if requested
        if output:
            output_path = Path(output)
            output_path.mkdir(parents=True, exist_ok=True)

            report_file = output_path / "example_evaluation.json"
            evaluator.save_evaluation_report(results, str(report_file))

            click.echo(f"Example results saved to {output_path}")

    except Exception as e:
        click.echo(f"❌ Error running example: {e}", err=True)


@evaluation_cli.command()
@click.option(
    "--config-file",
    required=True,
    type=click.Path(),
    help="Configuration file to create",
)
def create_config(config_file: str):
    """Create evaluation configuration template."""

    config = {
        "evaluation_settings": {
            "task_type": "classification",
            "include_fairness": True,
            "include_privacy": True,
            "convergence_patience": 5,
            "convergence_min_delta": 1e-4,
        },
        "fairness_settings": {
            "sensitive_attributes": ["gender", "age_group", "ethnicity"],
            "fairness_threshold": 0.1,
        },
        "privacy_settings": {
            "membership_inference_threshold": 0.6,
            "model_inversion_threshold": 0.5,
            "differential_privacy_epsilon": 1.0,
        },
        "performance_thresholds": {
            "minimum_accuracy": 0.7,
            "maximum_loss": 1.0,
            "minimum_r2_score": 0.5,
        },
    }

    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    click.echo(f"Evaluation configuration template created: {config_file}")


# Helper functions


def load_data_file(filepath: str) -> List:
    """Load data from various file formats."""
    path = Path(filepath)

    if path.suffix == ".json":
        with open(filepath, "r") as f:
            return json.load(f)
    elif path.suffix == ".csv":
        data = []
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 1:
                    try:
                        data.append(float(row[0]))
                    except ValueError:
                        data.append(row[0])
                else:
                    data.append(row)
        return data
    else:
        # Try to read as text file with one value per line
        with open(filepath, "r") as f:
            lines = f.read().strip().split("\n")
            data = []
            for line in lines:
                try:
                    data.append(float(line))
                except ValueError:
                    data.append(line)
            return data


def load_sensitive_attributes(filepath: str) -> Dict[str, List]:
    """Load sensitive attributes from file."""
    with open(filepath, "r") as f:
        if filepath.endswith(".json"):
            return json.load(f)
        else:
            # Assume CSV format with headers
            reader = csv.DictReader(f)
            attributes = {}
            for row in reader:
                for key, value in row.items():
                    if key not in attributes:
                        attributes[key] = []
                    try:
                        attributes[key].append(int(value))
                    except ValueError:
                        attributes[key].append(value)
            return attributes


def prepare_evaluation_data(
    session_data: Dict[str, Any],
    task_type: str,
    include_fairness: bool,
    include_privacy: bool,
    sensitive_attrs: Optional[str],
) -> Dict[str, Any]:
    """Prepare evaluation data from session data."""
    evaluation_data = {
        "task_type": task_type,
        "training_history": session_data.get("training_history", []),
    }

    # Add participant data if available
    if "participants" in session_data:
        evaluation_data["participant_data"] = session_data["participants"]

    # Add privacy data if available
    if include_privacy and "privacy_data" in session_data:
        evaluation_data["privacy_data"] = session_data["privacy_data"]

    # Add sensitive attributes if provided
    if include_fairness and sensitive_attrs:
        try:
            evaluation_data["sensitive_attributes"] = load_sensitive_attributes(
                sensitive_attrs
            )
        except Exception as e:
            logger.warning(f"Could not load sensitive attributes: {e}")

    return evaluation_data


def display_evaluation_summary(results: Dict[str, Any]):
    """Display evaluation results summary."""
    click.echo("\n" + "=" * 60)
    click.echo("FEDERATED LEARNING EVALUATION SUMMARY")
    click.echo("=" * 60)

    # Performance metrics
    if "performance_metrics" in results and results["performance_metrics"]:
        click.echo("\n📊 Performance Metrics:")
        display_performance_metrics(results["performance_metrics"])

    # Fairness metrics
    if "fairness_metrics" in results and results["fairness_metrics"]:
        click.echo("\n⚖️  Fairness Metrics:")
        display_fairness_metrics(results["fairness_metrics"])

    # Privacy metrics
    if "privacy_metrics" in results and results["privacy_metrics"]:
        click.echo("\n🔒 Privacy Metrics:")
        display_privacy_metrics(results["privacy_metrics"])

    # Convergence metrics
    if "convergence_metrics" in results and results["convergence_metrics"]:
        click.echo("\n📈 Convergence Analysis:")
        display_convergence_metrics(results["convergence_metrics"])

    # Overall assessment
    if "overall_assessment" in results:
        click.echo("\n🎯 Overall Assessment:")
        assessment = results["overall_assessment"]
        click.echo(f"  Quality Score: {assessment.get('quality_score', 0):.3f}")
        click.echo(f"  Fairness Score: {assessment.get('fairness_score', 0):.3f}")
        click.echo(f"  Privacy Score: {assessment.get('privacy_score', 0):.3f}")
        click.echo(f"  Efficiency Score: {assessment.get('efficiency_score', 0):.3f}")
        click.echo(f"  Risk Level: {assessment.get('risk_level', 'UNKNOWN')}")

        if "recommendations" in assessment and assessment["recommendations"]:
            click.echo("\n💡 Recommendations:")
            for rec in assessment["recommendations"]:
                click.echo(f"  • {rec}")


def display_performance_metrics(metrics):
    """Display performance metrics."""
    if hasattr(metrics, "__dict__"):
        metrics_dict = metrics.__dict__
    else:
        metrics_dict = metrics

    for key, value in metrics_dict.items():
        if value is not None and key != "custom_metrics":
            click.echo(f"  {key.replace('_', ' ').title()}: {value:.4f}")

    if "custom_metrics" in metrics_dict and metrics_dict["custom_metrics"]:
        click.echo("  Custom Metrics:")
        for key, value in metrics_dict["custom_metrics"].items():
            click.echo(f"    {key}: {value:.4f}")


def display_fairness_metrics(metrics):
    """Display fairness metrics."""
    if hasattr(metrics, "__dict__"):
        metrics_dict = metrics.__dict__
    else:
        metrics_dict = metrics

    for key, value in metrics_dict.items():
        if value is not None:
            click.echo(f"  {key.replace('_', ' ').title()}: {value:.4f}")


def display_privacy_metrics(metrics):
    """Display privacy metrics."""
    if hasattr(metrics, "__dict__"):
        metrics_dict = metrics.__dict__
    else:
        metrics_dict = metrics

    for key, value in metrics_dict.items():
        if value is not None:
            if key == "differential_privacy_epsilon" and value == float("inf"):
                click.echo(
                    f"  {key.replace('_', ' ').title()}: ∞ (no privacy protection)"
                )
            else:
                click.echo(f"  {key.replace('_', ' ').title()}: {value:.4f}")


def display_convergence_metrics(metrics):
    """Display convergence metrics."""
    if hasattr(metrics, "__dict__"):
        metrics_dict = metrics.__dict__
    else:
        metrics_dict = metrics

    for key, value in metrics_dict.items():
        if value is not None:
            if key in ["rounds_to_convergence", "early_stopping_round"]:
                click.echo(f"  {key.replace('_', ' ').title()}: {value}")
            else:
                click.echo(f"  {key.replace('_', ' ').title()}: {value:.4f}")


def compare_evaluation_results(
    results1: Dict[str, Any], results2: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare two evaluation results."""
    comparison = {
        "performance_comparison": {},
        "fairness_comparison": {},
        "privacy_comparison": {},
        "convergence_comparison": {},
        "overall_winner": None,
    }

    # Compare performance
    perf1 = results1.get("performance_metrics")
    perf2 = results2.get("performance_metrics")
    if perf1 and perf2:
        if hasattr(perf1, "accuracy") and hasattr(perf2, "accuracy"):
            if perf1.accuracy and perf2.accuracy:
                comparison["performance_comparison"]["accuracy_diff"] = (
                    perf1.accuracy - perf2.accuracy
                )
                comparison["performance_comparison"]["accuracy_winner"] = (
                    "1" if perf1.accuracy > perf2.accuracy else "2"
                )

    # Compare fairness
    fair1 = results1.get("fairness_metrics")
    fair2 = results2.get("fairness_metrics")
    if fair1 and fair2:
        if hasattr(fair1, "demographic_parity") and hasattr(
            fair2, "demographic_parity"
        ):
            if fair1.demographic_parity and fair2.demographic_parity:
                comparison["fairness_comparison"]["demographic_parity_diff"] = (
                    fair1.demographic_parity - fair2.demographic_parity
                )
                comparison["fairness_comparison"]["fairness_winner"] = (
                    "1" if fair1.demographic_parity < fair2.demographic_parity else "2"
                )

    # Determine overall winner
    scores1 = []
    scores2 = []

    if "overall_assessment" in results1:
        scores1.append(results1["overall_assessment"].get("quality_score", 0))
        scores1.append(results1["overall_assessment"].get("fairness_score", 0))
        scores1.append(results1["overall_assessment"].get("privacy_score", 0))

    if "overall_assessment" in results2:
        scores2.append(results2["overall_assessment"].get("quality_score", 0))
        scores2.append(results2["overall_assessment"].get("fairness_score", 0))
        scores2.append(results2["overall_assessment"].get("privacy_score", 0))

    if scores1 and scores2:
        avg1 = sum(scores1) / len(scores1)
        avg2 = sum(scores2) / len(scores2)
        comparison["overall_winner"] = "1" if avg1 > avg2 else "2"

    return comparison


def display_comparison_results(comparison: Dict[str, Any]):
    """Display comparison results."""
    click.echo("  Performance Comparison:")
    if "performance_comparison" in comparison:
        for key, value in comparison["performance_comparison"].items():
            click.echo(f"    {key}: {value}")

    click.echo("  Fairness Comparison:")
    if "fairness_comparison" in comparison:
        for key, value in comparison["fairness_comparison"].items():
            click.echo(f"    {key}: {value}")

    if "overall_winner" in comparison and comparison["overall_winner"]:
        click.echo(f"\n🏆 Overall Winner: Session {comparison['overall_winner']}")


def save_evaluation_csv(results: Dict[str, Any], filepath: str):
    """Save evaluation results to CSV format."""
    with open(filepath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(["Metric", "Value"])

        # Write performance metrics
        if "performance_metrics" in results and results["performance_metrics"]:
            perf = results["performance_metrics"]
            if hasattr(perf, "__dict__"):
                for key, value in perf.__dict__.items():
                    if value is not None and key != "custom_metrics":
                        writer.writerow([f"performance_{key}", value])

        # Write overall assessment
        if "overall_assessment" in results:
            assessment = results["overall_assessment"]
            for key, value in assessment.items():
                if key != "recommendations" and isinstance(value, (int, float, str)):
                    writer.writerow([f"overall_{key}", value])


if __name__ == "__main__":
    evaluation_cli()
