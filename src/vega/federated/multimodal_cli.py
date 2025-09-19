"""
CLI Interface for Multi-Modal Federated Learning

This module provides a comprehensive command-line interface for interacting
with the multi-modal federated learning system, including data preparation,
model training, evaluation, and demonstration capabilities.
"""

import typer
import torch
import json
import logging
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from datetime import datetime
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from .multimodal import DataModality, ModalityConfig, MultiModalDataManager
from .multimodal_aggregation import AggregationStrategy, MultiModalAggregationConfig
from .multimodal_examples import MultiModalFederatedLearningDemo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize CLI app and console
app = typer.Typer(help="Multi-Modal Federated Learning CLI")
console = Console()


# Global state
class CLIState:
    def __init__(self):
        self.demo = None
        self.config_file = None
        self.output_dir = "./multimodal_cli_results"


cli_state = CLIState()


@app.command()
def demo(
    demo_type: str = typer.Option(
        "full", help="Type of demo to run", case_sensitive=False
    ),
    participants: int = typer.Option(5, help="Number of federated participants"),
    output_dir: str = typer.Option(
        "./multimodal_demo_results", help="Output directory"
    ),
    device: str = typer.Option("cpu", help="Device to use (cpu/cuda)"),
    verbose: bool = typer.Option(False, help="Enable verbose logging"),
):
    """Run multi-modal federated learning demonstrations."""

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    console.print(f"[bold blue]Multi-Modal Federated Learning Demo[/bold blue]")
    console.print(f"Demo type: {demo_type}")
    console.print(f"Participants: {participants}")
    console.print(f"Output directory: {output_dir}")
    console.print(f"Device: {device}")
    console.print()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            # Initialize demo
            task = progress.add_task("Initializing demo...", total=1)
            demo = MultiModalFederatedLearningDemo(output_dir=output_dir, device=device)
            progress.update(task, completed=1)

            # Setup data
            task = progress.add_task("Setting up demo data...", total=1)
            demo.setup_demo_data()
            progress.update(task, completed=1)

            # Setup participants
            task = progress.add_task("Setting up participants...", total=1)
            demo.setup_federated_participants(num_participants=participants)
            progress.update(task, completed=1)

            # Setup models
            task = progress.add_task("Setting up models...", total=1)
            demo.setup_federated_models()
            progress.update(task, completed=1)

            # Setup aggregation
            task = progress.add_task("Setting up aggregation...", total=1)
            demo.setup_aggregation_strategy()
            progress.update(task, completed=1)

            # Run selected demo
            if demo_type.lower() == "full":
                task = progress.add_task("Running full demo...", total=1)
                demo.run_full_demo()
            elif demo_type.lower() == "vision_text":
                task = progress.add_task("Running vision+text demo...", total=1)
                demo.run_vision_text_demo()
            elif demo_type.lower() == "audio_sensor":
                task = progress.add_task("Running audio+sensor demo...", total=1)
                demo.run_audio_sensor_demo()
            elif demo_type.lower() == "cross_modal":
                task = progress.add_task(
                    "Running cross-modal learning demo...", total=1
                )
                demo.run_cross_modal_learning_demo()
            elif demo_type.lower() == "aggregation":
                task = progress.add_task(
                    "Running aggregation comparison demo...", total=1
                )
                demo.run_aggregation_comparison_demo()
            else:
                console.print(f"[red]Unknown demo type: {demo_type}[/red]")
                raise typer.Exit(1)

            progress.update(task, completed=1)

        console.print(f"[green]✓ Demo completed successfully![/green]")
        console.print(f"Results saved to: {output_dir}")

    except Exception as e:
        console.print(f"[red]✗ Demo failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list_modalities():
    """List all supported data modalities."""

    console.print("[bold blue]Supported Data Modalities[/bold blue]")
    console.print()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Modality", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Typical Use Cases", style="green")

    modality_info = {
        DataModality.VISION: {
            "description": "Image and video data",
            "use_cases": "Object detection, image classification, medical imaging",
        },
        DataModality.TEXT: {
            "description": "Natural language text data",
            "use_cases": "Sentiment analysis, language modeling, document classification",
        },
        DataModality.AUDIO: {
            "description": "Audio and speech data",
            "use_cases": "Speech recognition, music analysis, audio classification",
        },
        DataModality.SENSOR: {
            "description": "IoT sensor readings and time series",
            "use_cases": "Environmental monitoring, health tracking, industrial sensors",
        },
        DataModality.TABULAR: {
            "description": "Structured tabular data",
            "use_cases": "Financial analysis, customer analytics, business intelligence",
        },
        DataModality.TIME_SERIES: {
            "description": "Sequential time-dependent data",
            "use_cases": "Stock prediction, weather forecasting, signal processing",
        },
    }

    for modality, info in modality_info.items():
        table.add_row(modality.value, info["description"], info["use_cases"])

    console.print(table)


@app.command()
def list_aggregation_strategies():
    """List all available aggregation strategies."""

    console.print("[bold blue]Aggregation Strategies[/bold blue]")
    console.print()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Strategy", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Best For", style="green")

    strategy_info = {
        AggregationStrategy.FEDAVG_MULTIMODAL: {
            "description": "Weighted averaging based on data sizes",
            "best_for": "Balanced participants with similar data quality",
        },
        AggregationStrategy.MODALITY_WEIGHTED: {
            "description": "Considers modality-specific characteristics",
            "best_for": "Different modalities with varying importance",
        },
        AggregationStrategy.ADAPTIVE_MODALITY: {
            "description": "Learns optimal modality weights over time",
            "best_for": "Dynamic environments with changing data patterns",
        },
        AggregationStrategy.QUALITY_AWARE: {
            "description": "Emphasizes high-quality data over quantity",
            "best_for": "Participants with varying data quality",
        },
        AggregationStrategy.HIERARCHICAL_MULTIMODAL: {
            "description": "Two-stage aggregation: within then across modalities",
            "best_for": "Complex multi-modal scenarios",
        },
        AggregationStrategy.CONTRASTIVE_AGGREGATION: {
            "description": "Uses contrastive learning for alignment",
            "best_for": "Cross-modal alignment and representation learning",
        },
    }

    for strategy, info in strategy_info.items():
        table.add_row(strategy.value, info["description"], info["best_for"])

    console.print(table)


@app.command()
def create_config(
    output_file: str = typer.Option(
        "multimodal_config.json", help="Output configuration file"
    ),
    modalities: List[str] = typer.Option(
        ["vision", "text"], help="Modalities to include"
    ),
    aggregation_strategy: str = typer.Option(
        "adaptive_modality", help="Aggregation strategy"
    ),
    participants: int = typer.Option(5, help="Number of participants"),
    embedding_dim: int = typer.Option(512, help="Embedding dimension"),
):
    """Create a configuration file for multi-modal federated learning."""

    console.print(f"[bold blue]Creating Configuration File[/bold blue]")
    console.print()

    # Validate modalities
    valid_modalities = [m.value for m in DataModality]
    invalid_modalities = [m for m in modalities if m not in valid_modalities]

    if invalid_modalities:
        console.print(f"[red]Invalid modalities: {invalid_modalities}[/red]")
        console.print(f"Valid modalities: {valid_modalities}")
        raise typer.Exit(1)

    # Validate aggregation strategy
    valid_strategies = [s.value for s in AggregationStrategy]
    if aggregation_strategy not in valid_strategies:
        console.print(
            f"[red]Invalid aggregation strategy: {aggregation_strategy}[/red]"
        )
        console.print(f"Valid strategies: {valid_strategies}")
        raise typer.Exit(1)

    # Create configuration
    config = {
        "experiment_info": {
            "name": "multi_modal_federated_experiment",
            "description": "Multi-modal federated learning experiment",
            "created_at": datetime.now().isoformat(),
        },
        "modalities": modalities,
        "embedding_dim": embedding_dim,
        "participants": {"count": participants, "data_distribution": "balanced"},
        "aggregation": {
            "strategy": aggregation_strategy,
            "use_data_size_weighting": True,
            "use_quality_weighting": True,
            "quality_threshold": 0.6,
            "adaptation_rate": 0.1,
        },
        "training": {
            "rounds": 10,
            "local_epochs": 5,
            "learning_rate": 0.001,
            "batch_size": 32,
        },
        "model": {
            "transformer_layers": 6,
            "attention_heads": 8,
            "dropout": 0.1,
            "use_compression": True,
        },
    }

    # Save configuration
    output_path = Path(output_file)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    console.print(f"[green]✓ Configuration saved to: {output_path}[/green]")

    # Display configuration summary
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Modalities", ", ".join(modalities))
    table.add_row("Participants", str(participants))
    table.add_row("Aggregation Strategy", aggregation_strategy)
    table.add_row("Embedding Dimension", str(embedding_dim))
    table.add_row("Training Rounds", str(config["training"]["rounds"]))

    console.print()
    console.print(table)


@app.command()
def validate_config(
    config_file: str = typer.Argument(..., help="Configuration file to validate")
):
    """Validate a multi-modal federated learning configuration file."""

    console.print(f"[bold blue]Validating Configuration[/bold blue]")
    console.print(f"File: {config_file}")
    console.print()

    config_path = Path(config_file)
    if not config_path.exists():
        console.print(f"[red]Configuration file not found: {config_file}[/red]")
        raise typer.Exit(1)

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        validation_results = []

        # Validate required sections
        required_sections = [
            "modalities",
            "participants",
            "aggregation",
            "training",
            "model",
        ]
        for section in required_sections:
            if section in config:
                validation_results.append(
                    ("✓", f"Required section '{section}' present", "green")
                )
            else:
                validation_results.append(
                    ("✗", f"Missing required section '{section}'", "red")
                )

        # Validate modalities
        if "modalities" in config:
            valid_modalities = [m.value for m in DataModality]
            invalid_modalities = [
                m for m in config["modalities"] if m not in valid_modalities
            ]

            if not invalid_modalities:
                validation_results.append(
                    ("✓", f"All modalities valid: {config['modalities']}", "green")
                )
            else:
                validation_results.append(
                    ("✗", f"Invalid modalities: {invalid_modalities}", "red")
                )

        # Validate aggregation strategy
        if "aggregation" in config and "strategy" in config["aggregation"]:
            valid_strategies = [s.value for s in AggregationStrategy]
            strategy = config["aggregation"]["strategy"]

            if strategy in valid_strategies:
                validation_results.append(
                    ("✓", f"Aggregation strategy valid: {strategy}", "green")
                )
            else:
                validation_results.append(
                    ("✗", f"Invalid aggregation strategy: {strategy}", "red")
                )

        # Display validation results
        for status, message, color in validation_results:
            console.print(f"[{color}]{status} {message}[/{color}]")

        # Summary
        errors = [r for r in validation_results if r[0] == "✗"]
        if errors:
            console.print(f"\n[red]Validation failed with {len(errors)} errors[/red]")
            raise typer.Exit(1)
        else:
            console.print(f"\n[green]✓ Configuration is valid![/green]")

    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON format: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Validation error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def info():
    """Display information about the multi-modal federated learning system."""

    console.print("[bold blue]Multi-Modal Federated Learning System[/bold blue]")
    console.print()

    # System capabilities
    console.print("[bold yellow]Capabilities:[/bold yellow]")
    capabilities = [
        "Multi-modal data processing (vision, text, audio, sensor, tabular, time-series)",
        "Federated transformer architecture with cross-modal attention",
        "Multiple aggregation strategies for different scenarios",
        "Cross-modal learning and alignment techniques",
        "Quality-aware and adaptive aggregation",
        "Differential privacy support",
        "Comprehensive evaluation framework",
    ]

    for capability in capabilities:
        console.print(f"  • {capability}")

    console.print()

    # System statistics
    console.print("[bold yellow]System Statistics:[/bold yellow]")
    stats_table = Table(show_header=True, header_style="bold magenta")
    stats_table.add_column("Component", style="cyan")
    stats_table.add_column("Count", style="white")

    stats_table.add_row("Supported Modalities", str(len(DataModality)))
    stats_table.add_row("Aggregation Strategies", str(len(AggregationStrategy)))
    stats_table.add_row("Fusion Strategies", "4")
    stats_table.add_row("Alignment Methods", "3")

    console.print(stats_table)

    console.print()
    console.print("[bold yellow]For help with specific commands, use:[/bold yellow]")
    console.print("  multimodal-cli --help")
    console.print("  multimodal-cli demo --help")
    console.print("  multimodal-cli create-config --help")


@app.command()
def benchmark(
    config_file: Optional[str] = typer.Option(None, help="Configuration file"),
    output_dir: str = typer.Option("./benchmark_results", help="Output directory"),
    runs: int = typer.Option(3, help="Number of benchmark runs"),
    device: str = typer.Option("cpu", help="Device to use"),
):
    """Run performance benchmarks for different configurations."""

    console.print("[bold blue]Multi-Modal Federated Learning Benchmark[/bold blue]")
    console.print()

    # Create benchmark scenarios
    scenarios = [
        {
            "modalities": ["vision", "text"],
            "participants": 5,
            "strategy": "adaptive_modality",
        },
        {
            "modalities": ["audio", "sensor"],
            "participants": 3,
            "strategy": "quality_aware",
        },
        {
            "modalities": ["vision", "text", "audio"],
            "participants": 8,
            "strategy": "hierarchical_multimodal",
        },
        {
            "modalities": ["vision", "text", "audio", "sensor"],
            "participants": 10,
            "strategy": "modality_weighted",
        },
    ]

    results = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        for i, scenario in enumerate(scenarios):
            scenario_name = f"scenario_{i+1}"
            console.print(f"\n[yellow]Running Scenario {i+1}:[/yellow]")
            console.print(f"  Modalities: {scenario['modalities']}")
            console.print(f"  Participants: {scenario['participants']}")
            console.print(f"  Strategy: {scenario['strategy']}")

            scenario_results = []

            for run in range(runs):
                task = progress.add_task(f"Run {run+1}/{runs}...", total=1)

                try:
                    # Run demo with scenario parameters
                    demo = MultiModalFederatedLearningDemo(
                        output_dir=f"{output_dir}/{scenario_name}/run_{run+1}",
                        device=device,
                    )

                    start_time = datetime.now()

                    demo.setup_demo_data()
                    demo.setup_federated_participants(
                        num_participants=scenario["participants"]
                    )
                    demo.setup_federated_models()
                    demo.setup_aggregation_strategy()
                    demo.run_aggregation_comparison_demo()

                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()

                    scenario_results.append(
                        {
                            "run": run + 1,
                            "duration_seconds": duration,
                            "participants": scenario["participants"],
                            "modalities": scenario["modalities"],
                        }
                    )

                    progress.update(task, completed=1)

                except Exception as e:
                    console.print(f"[red]Run {run+1} failed: {e}[/red]")
                    scenario_results.append({"run": run + 1, "error": str(e)})

            results[scenario_name] = {"config": scenario, "results": scenario_results}

    # Save benchmark results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    benchmark_file = output_path / "benchmark_results.json"
    with open(benchmark_file, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[green]✓ Benchmark completed![/green]")
    console.print(f"Results saved to: {benchmark_file}")

    # Display summary
    console.print("\n[bold yellow]Benchmark Summary:[/bold yellow]")
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Scenario", style="cyan")
    summary_table.add_column("Avg Duration (s)", style="white")
    summary_table.add_column("Success Rate", style="green")

    for scenario_name, data in results.items():
        successful_runs = [r for r in data["results"] if "error" not in r]
        if successful_runs:
            avg_duration = sum(r["duration_seconds"] for r in successful_runs) / len(
                successful_runs
            )
            success_rate = f"{len(successful_runs)}/{len(data['results'])}"
        else:
            avg_duration = 0
            success_rate = "0/" + str(len(data["results"]))

        summary_table.add_row(scenario_name, f"{avg_duration:.2f}", success_rate)

    console.print(summary_table)


if __name__ == "__main__":
    app()
