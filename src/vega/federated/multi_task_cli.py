"""
Multi-Task Federated Learning CLI

Command-line interface for managing multi-task federated learning sessions.
Supports task definition, participant management, and session coordination.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import click
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from .multi_task import (
    TaskDefinition,
    MultiTaskModelConfig,
    MultiTaskParticipant,
    MultiTaskCoordinator,
    create_synthetic_multitask_data,
)
from .multi_task_example import MultiTaskFederatedExample, run_multi_task_example

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
def multi_task_cli():
    """Multi-Task Federated Learning CLI"""
    pass


@multi_task_cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Configuration file for tasks and model",
)
@click.option("--participants", "-p", default=5, help="Number of participants")
@click.option("--rounds", "-r", default=10, help="Number of federated rounds")
@click.option("--epochs", "-e", default=2, help="Local epochs per round")
@click.option("--output", "-o", type=click.Path(), help="Output directory for results")
@click.option("--visualize", is_flag=True, help="Generate visualizations")
def run(
    config: Optional[str],
    participants: int,
    rounds: int,
    epochs: int,
    output: Optional[str],
    visualize: bool,
):
    """Run multi-task federated learning session."""

    click.echo(
        f"Starting multi-task federated learning with {participants} participants"
    )
    click.echo(f"Running {rounds} rounds with {epochs} local epochs each")

    if config:
        # Load configuration from file
        click.echo(f"Loading configuration from {config}")
        session_config = load_config_file(config)

        # Create session from config
        session = create_session_from_config(session_config)
    else:
        # Use default example configuration
        click.echo("Using default example configuration")
        session = MultiTaskFederatedExample(num_participants=participants)

        # Create data and participants
        participant_data = session.create_heterogeneous_data()
        session.create_participants(participant_data)

    # Run federated learning
    async def run_session():
        results = await session.run_federated_learning(
            num_rounds=rounds, local_epochs=epochs
        )

        # Analyze results
        analysis = session.analyze_results()

        # Output results
        if output:
            output_path = Path(output)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save session state
            session.save_results(str(output_path / "session_state.json"))

            # Save analysis
            with open(output_path / "analysis.json", "w") as f:
                json.dump(analysis, f, indent=2, default=str)

            # Generate visualizations
            if visualize:
                session.visualize_results(str(output_path / "results.png"))

            click.echo(f"Results saved to {output_path}")

        # Display summary
        display_results_summary(analysis)

        return results, analysis

    # Run the session
    asyncio.run(run_session())


@multi_task_cli.command()
@click.option("--task-id", required=True, help="Task identifier")
@click.option(
    "--task-type",
    required=True,
    type=click.Choice(["classification", "regression"]),
    help="Type of task",
)
@click.option("--input-dim", required=True, type=int, help="Input dimension")
@click.option("--output-dim", required=True, type=int, help="Output dimension")
@click.option(
    "--loss-function",
    required=True,
    type=click.Choice(["cross_entropy", "mse", "bce"]),
    help="Loss function",
)
@click.option("--weight", default=1.0, help="Task weight for multi-task learning")
@click.option(
    "--config-file", type=click.Path(), help="Configuration file to add task to"
)
def add_task(
    task_id: str,
    task_type: str,
    input_dim: int,
    output_dim: int,
    loss_function: str,
    weight: float,
    config_file: Optional[str],
):
    """Add a new task definition."""

    task = TaskDefinition(
        task_id=task_id,
        task_type=task_type,
        input_dim=input_dim,
        output_dim=output_dim,
        loss_function=loss_function,
        metric="accuracy" if task_type == "classification" else "mse",
        task_weight=weight,
    )

    if config_file:
        # Add to existing config file
        if Path(config_file).exists():
            with open(config_file, "r") as f:
                config = json.load(f)
        else:
            config = {"tasks": [], "model_config": {}}

        config["tasks"].append(task.__dict__)

        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        click.echo(f"Added task '{task_id}' to {config_file}")
    else:
        # Display task definition
        click.echo(f"Task Definition:")
        click.echo(f"  ID: {task.task_id}")
        click.echo(f"  Type: {task.task_type}")
        click.echo(f"  Input Dim: {task.input_dim}")
        click.echo(f"  Output Dim: {task.output_dim}")
        click.echo(f"  Loss: {task.loss_function}")
        click.echo(f"  Weight: {task.task_weight}")


@multi_task_cli.command()
@click.option(
    "--config-file",
    required=True,
    type=click.Path(),
    help="Configuration file to create",
)
@click.option(
    "--shared-layers",
    default="64,32",
    help="Comma-separated list of shared layer sizes",
)
@click.option(
    "--activation",
    default="relu",
    type=click.Choice(["relu", "tanh", "sigmoid"]),
    help="Activation function",
)
@click.option("--dropout", default=0.1, help="Dropout rate")
@click.option("--batch-norm", is_flag=True, help="Use batch normalization")
def create_config(
    config_file: str,
    shared_layers: str,
    activation: str,
    dropout: float,
    batch_norm: bool,
):
    """Create a new configuration file."""

    # Parse shared layers
    layer_sizes = [int(x.strip()) for x in shared_layers.split(",")]

    config = {
        "tasks": [],
        "model_config": {
            "shared_layers": layer_sizes,
            "task_specific_layers": {},
            "activation": activation,
            "dropout_rate": dropout,
            "use_batch_norm": batch_norm,
        },
        "training_config": {
            "num_participants": 5,
            "num_rounds": 10,
            "local_epochs": 2,
            "aggregation_strategy": "fedavg",
        },
    }

    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    click.echo(f"Created configuration file: {config_file}")
    click.echo("Use 'add-task' command to add tasks to this configuration.")


@multi_task_cli.command()
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True),
    help="Configuration file to validate",
)
def validate_config(config: str):
    """Validate a configuration file."""

    try:
        with open(config, "r") as f:
            config_data = json.load(f)

        errors = []

        # Validate required sections
        if "tasks" not in config_data:
            errors.append("Missing 'tasks' section")
        elif not config_data["tasks"]:
            errors.append("No tasks defined")

        if "model_config" not in config_data:
            errors.append("Missing 'model_config' section")

        # Validate tasks
        for i, task_data in enumerate(config_data.get("tasks", [])):
            required_fields = [
                "task_id",
                "task_type",
                "input_dim",
                "output_dim",
                "loss_function",
            ]
            for field in required_fields:
                if field not in task_data:
                    errors.append(f"Task {i}: Missing required field '{field}'")

        # Validate model config
        model_config = config_data.get("model_config", {})
        if "shared_layers" not in model_config:
            errors.append("Model config: Missing 'shared_layers'")

        if errors:
            click.echo("Configuration validation failed:")
            for error in errors:
                click.echo(f"  ❌ {error}")
        else:
            click.echo("✅ Configuration file is valid")

            # Show summary
            click.echo(f"\nConfiguration Summary:")
            click.echo(f"  Tasks: {len(config_data['tasks'])}")
            for task in config_data["tasks"]:
                click.echo(f"    - {task['task_id']} ({task['task_type']})")

            shared_layers = model_config.get("shared_layers", [])
            click.echo(f"  Shared layers: {shared_layers}")

    except json.JSONDecodeError as e:
        click.echo(f"❌ Invalid JSON in configuration file: {e}")
    except Exception as e:
        click.echo(f"❌ Error validating configuration: {e}")


@multi_task_cli.command()
@click.option(
    "--session-file",
    required=True,
    type=click.Path(exists=True),
    help="Saved session file to analyze",
)
@click.option(
    "--output", type=click.Path(), help="Output directory for analysis results"
)
def analyze(session_file: str, output: Optional[str]):
    """Analyze results from a saved session."""

    click.echo(f"Analyzing session file: {session_file}")

    try:
        with open(session_file, "r") as f:
            session_data = json.load(f)

        # Extract key metrics
        num_rounds = session_data.get("current_round", 0)
        tasks = session_data.get("tasks", [])

        click.echo(f"\nSession Summary:")
        click.echo(f"  Rounds completed: {num_rounds}")
        click.echo(f"  Tasks: {len(tasks)}")

        for task in tasks:
            click.echo(f"    - {task['task_id']} ({task['task_type']})")

        # Analyze training history if available
        history = session_data.get("training_history", [])
        if history:
            click.echo(f"\nTraining Progress:")
            click.echo(f"  Total training rounds: {len(history)}")

            # Show convergence for last few rounds
            recent_rounds = history[-3:] if len(history) >= 3 else history
            for round_data in recent_rounds:
                round_num = round_data.get("round", 0)
                participants = round_data.get("participants", [])
                click.echo(f"    Round {round_num}: {len(participants)} participants")

        if output:
            output_path = Path(output)
            output_path.mkdir(parents=True, exist_ok=True)

            # Create analysis report
            analysis_report = generate_analysis_report(session_data)

            with open(output_path / "analysis_report.json", "w") as f:
                json.dump(analysis_report, f, indent=2, default=str)

            click.echo(f"Analysis saved to {output_path}")

    except Exception as e:
        click.echo(f"❌ Error analyzing session: {e}")


@multi_task_cli.command()
def example():
    """Run the built-in multi-task federated learning example."""

    click.echo("Running built-in multi-task federated learning example...")
    click.echo("This will demonstrate:")
    click.echo(
        "  - Multi-task learning with sentiment analysis, spam detection, and rating prediction"
    )
    click.echo("  - Heterogeneous data distributions across participants")
    click.echo("  - Shared representations with task-specific heads")

    # Run the example
    asyncio.run(run_multi_task_example())


# Helper functions


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def create_session_from_config(config: Dict[str, Any]) -> MultiTaskFederatedExample:
    """Create a federated learning session from configuration."""

    # Parse tasks
    tasks = []
    for task_data in config["tasks"]:
        task = TaskDefinition(**task_data)
        tasks.append(task)

    # Parse model config
    model_config_data = config["model_config"]
    model_config = MultiTaskModelConfig(**model_config_data)

    # Get training config
    training_config = config.get("training_config", {})
    num_participants = training_config.get("num_participants", 5)

    # Create session
    session = MultiTaskFederatedExample(num_participants=num_participants)
    session.tasks = tasks
    session.model_config = model_config

    # Update coordinator
    session.coordinator = MultiTaskCoordinator(
        tasks=tasks,
        model_config=model_config,
        aggregation_strategy=training_config.get("aggregation_strategy", "fedavg"),
    )

    return session


def display_results_summary(analysis: Dict[str, Any]):
    """Display a summary of federated learning results."""

    click.echo("\n" + "=" * 50)
    click.echo("FEDERATED LEARNING RESULTS SUMMARY")
    click.echo("=" * 50)

    # Task performance
    if "task_performance" in analysis:
        click.echo("\nFinal Task Performance:")
        for task_id, metrics in analysis["task_performance"].items():
            click.echo(f"  {task_id}:")
            for metric_name, value in metrics.items():
                click.echo(f"    {metric_name}: {value:.4f}")

    # Participant contribution
    if "participant_contribution" in analysis:
        click.echo("\nParticipant Task Coverage:")
        for participant_id, tasks in analysis["participant_contribution"].items():
            task_list = list(tasks.keys())
            click.echo(f"  {participant_id}: {task_list}")

    # Convergence analysis
    if "convergence_analysis" in analysis:
        click.echo("\nConvergence Status:")
        convergence = analysis["convergence_analysis"]
        for metric_name, values in convergence.items():
            if len(values) >= 2:
                improvement = values[-1] - values[0]
                trend = "improving" if improvement > 0 else "declining"
                click.echo(f"  {metric_name}: {trend} ({improvement:+.4f})")


def generate_analysis_report(session_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive analysis report from session data."""

    report = {
        "session_summary": {
            "total_rounds": session_data.get("current_round", 0),
            "num_tasks": len(session_data.get("tasks", [])),
            "aggregation_strategy": session_data.get("aggregation_strategy", "unknown"),
        },
        "task_definitions": session_data.get("tasks", []),
        "model_configuration": session_data.get("model_config", {}),
        "training_summary": {
            "rounds_completed": len(session_data.get("training_history", [])),
            "total_participants": set(),
        },
    }

    # Analyze training history
    training_history = session_data.get("training_history", [])
    if training_history:
        all_participants = set()
        for round_data in training_history:
            participants = round_data.get("participants", [])
            all_participants.update(participants)

        report["training_summary"]["total_participants"] = list(all_participants)
        report["training_summary"]["unique_participant_count"] = len(all_participants)

    return report


if __name__ == "__main__":
    multi_task_cli()
