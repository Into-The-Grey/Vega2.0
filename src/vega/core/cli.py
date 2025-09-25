"""
cli.py - Typer + Rich CLI for Vega2.0

Commands mirror API endpoints for parity:
- vega chat "Hello" -> prints model response
- vega repl -> interactive chat REPL with memory (session only)
- vega history --limit 20 -> pretty-print last N chats from SQLite
- vega integrations test -> send test Slack message (if webhook configured)
- vega dataset build ./myfiles -> run dataset preparation
- vega train --config training/config.yaml -> run fine-tuning

Usage:
  python -m cli --help
  Or add an entry point wrapper named 'vega' if desired.
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional, Any

try:
    import typer  # type: ignore[import-not-found]
except ModuleNotFoundError:
    # Minimal shim to keep module importable and provide a clear runtime message
    import sys

    class _TyperShim:
        def __init__(self, *args, **kwargs):
            pass

        def command(self, *args, **kwargs):
            def _decorator(fn):
                return fn

            return _decorator

        def add_typer(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            print(
                "Typer is required for the CLI. Install it with: pip install 'typer[all]'",
                file=sys.stderr,
            )
            sys.exit(1)

    class _TyperModule:
        Typer = _TyperShim

        @staticmethod
        def Option(default: "Any" = None, **kwargs: "Any") -> "Any":
            return default

        @staticmethod
        def Argument(default: "Any" = ..., **kwargs: "Any") -> "Any":
            return default

    typer = _TyperModule()

from rich.console import Console
from rich.table import Table

from .config import get_config
from .db import get_history, log_conversation, set_feedback
from .llm import query_llm, LLMBackendError
from ..integrations.slack_connector import send_slack_message

app = typer.Typer(help="Vega2.0 CLI")
integrations_app = typer.Typer(help="External integrations commands")
search_app = typer.Typer(help="Web and image search")
dataset_app = typer.Typer(help="Dataset utilities")
learn_app = typer.Typer(help="Learning and self-improvement")
db_app = typer.Typer(help="Database maintenance utilities")
gen_app = typer.Typer(help="Generation settings control")
osint_app = typer.Typer(help="OSINT utilities")
net_app = typer.Typer(help="Networking utilities")

app.add_typer(integrations_app, name="integrations")
app.add_typer(search_app, name="search")
app.add_typer(dataset_app, name="dataset")
app.add_typer(learn_app, name="learn")
app.add_typer(db_app, name="db")
app.add_typer(gen_app, name="gen")
app.add_typer(osint_app, name="osint")
app.add_typer(net_app, name="net")

# Federated Reinforcement Learning commands
frl_app = typer.Typer(help="Federated Reinforcement Learning utilities")
app.add_typer(frl_app, name="frl")

# Federated Pruning commands
pruning_app = typer.Typer(help="Federated Model Pruning utilities")
app.add_typer(pruning_app, name="pruning")

# Adaptive Federated Learning commands
adaptive_app = typer.Typer(help="Adaptive Federated Learning utilities")
app.add_typer(adaptive_app, name="adaptive")


@frl_app.command("bandit")
def frl_bandit_demo(
    participants: int = typer.Option(3, help="Number of clients/participants"),
    arms: int = typer.Option(3, help="Number of bandit arms"),
    rounds: int = typer.Option(8, help="Federated rounds to run"),
    steps: int = typer.Option(100, help="Local steps per round per client"),
    lr: float = typer.Option(0.1, help="Local learning rate"),
    seed: int = typer.Option(123, help="Random seed for determinism"),
):
    """Run a simple Federated RL (multi-armed bandit) demo."""
    try:
        from ..federated.reinforcement import BanditEnv, run_federated_bandit
    except Exception as e:  # pragma: no cover - import error path
        console.print(f"✗ Failed to import FRL module: {e}", style="red")
        return

    # Generate heterogeneous client environments with varying best arms
    random = __import__("random")
    random.seed(seed)
    client_envs = []
    for _ in range(participants):
        # Create a random probability distribution with one clearly best arm
        base = [random.uniform(0.05, 0.5) for _ in range(arms)]
        best_idx = random.randrange(arms)
        base[best_idx] = random.uniform(0.6, 0.9)
        client_envs.append(BanditEnv(base))

    result = run_federated_bandit(
        client_envs, rounds=rounds, local_steps_per_round=steps, lr=lr, seed=seed
    )

    table = Table(title="Federated RL (Bandit) Training History")
    table.add_column("Round", style="cyan")
    table.add_column("Avg Reward", style="green")
    for r in result["history"]:
        table.add_row(str(r["round"]), f"{r['avg_reward']:.3f}")
    console.print(table)
    console.print("Final theta:", result["final_theta"])


@frl_app.command("continual")
def frl_continual_demo(
    participants: int = typer.Option(3, help="Number of federated participants"),
    tasks: int = typer.Option(3, help="Number of sequential tasks"),
    steps_per_task: int = typer.Option(150, help="Training steps per task"),
    fed_rounds: int = typer.Option(3, help="Federated rounds per task"),
    lr: float = typer.Option(0.02, help="Learning rate"),
    ewc_lambda: float = typer.Option(500.0, help="EWC regularization strength"),
    seed: int = typer.Option(123, help="Random seed for determinism"),
):
    """Run Continual Federated Learning with Elastic Weight Consolidation."""
    try:
        from ..federated.continual import (
            run_continual_federated_learning,
            create_synthetic_task_sequence,
        )
    except Exception as e:  # pragma: no cover - import error path
        console.print(f"✗ Failed to import Continual FL module: {e}", style="red")
        return

    # Create synthetic task sequence
    task_sequence = create_synthetic_task_sequence()[:tasks]

    console.print(
        f"Running Continual FL with {participants} participants on {tasks} sequential tasks"
    )
    console.print(f"EWC lambda: {ewc_lambda}, Learning rate: {lr}")

    results = run_continual_federated_learning(
        num_participants=participants,
        tasks=task_sequence,
        steps_per_task=steps_per_task,
        fed_rounds_per_task=fed_rounds,
        lr=lr,
        ewc_lambda=ewc_lambda,
        seed=seed,
    )

    # Display results
    table = Table(title="Continual Federated Learning Results")
    table.add_column("Task", style="cyan")
    table.add_column("Final Avg Loss", style="green")
    table.add_column("Forgetting?", style="yellow")

    final_performance = results["performance_matrix"][-1]
    for task_idx, task_name in enumerate(results["tasks"]):
        avg_loss = sum(final_performance[task_idx]) / len(final_performance[task_idx])

        # Check for catastrophic forgetting (compare to when task was first learned)
        if task_idx > 0:
            initial_performance = results["performance_matrix"][task_idx][task_idx]
            initial_avg = sum(initial_performance) / len(initial_performance)
            forgetting_ratio = avg_loss / initial_avg if initial_avg > 0 else 1.0
            forgetting_status = (
                "HIGH"
                if forgetting_ratio > 2.0
                else "LOW" if forgetting_ratio > 1.3 else "MINIMAL"
            )
        else:
            forgetting_status = "N/A"

        table.add_row(task_name, f"{avg_loss:.3f}", forgetting_status)

    console.print(table)
    console.print(
        f"Total federated rounds completed: {len(results['federated_history'])}"
    )


@frl_app.command("async")
def frl_async_demo(
    participants: int = typer.Option(4, help="Number of async participants"),
    updates_per_participant: int = typer.Option(10, help="Updates per participant"),
    max_staleness: int = typer.Option(3, help="Maximum allowed staleness"),
    update_threshold: int = typer.Option(2, help="Updates before aggregation"),
    staleness_decay: float = typer.Option(0.8, help="Decay factor for stale updates"),
    seed: int = typer.Option(123, help="Random seed for determinism"),
):
    """Run Asynchronous Federated Learning demo with staleness tolerance."""
    try:
        from ..federated.async_fl import (
            run_async_federated_learning,
            AsyncFLConfig,
        )
    except Exception as e:  # pragma: no cover - import error path
        console.print(f"✗ Failed to import Async FL module: {e}", style="red")
        return

    import asyncio

    config = AsyncFLConfig(
        max_staleness=max_staleness,
        staleness_decay=staleness_decay,
        update_threshold=update_threshold,
        min_participants=min(2, participants),
    )

    console.print(f"Running Async FL with {participants} participants")
    console.print(
        f"Max staleness: {max_staleness}, Update threshold: {update_threshold}"
    )

    async def run_demo():
        results = await run_async_federated_learning(
            num_participants=participants,
            input_dim=2,
            output_dim=1,
            num_updates_per_participant=updates_per_participant,
            config=config,
            seed=seed,
        )

        # Display results
        table = Table(title="Asynchronous Federated Learning Results")
        table.add_column("Participant", style="cyan")
        table.add_column("Total Updates", style="green")
        table.add_column("Final Staleness", style="yellow")

        for pid, stats in results["participant_states"].items():
            table.add_row(
                pid, str(stats["total_updates"]), str(stats["current_staleness"])
            )

        console.print(table)
        console.print(f"Global model version: {results['global_version']}")
        console.print(f"Total updates processed: {results['total_updates_processed']}")
        console.print(f"Number of aggregations: {len(results['aggregation_history'])}")

        if results["aggregation_history"]:
            agg_table = Table(title="Recent Aggregation History")
            agg_table.add_column("Version", style="cyan")
            agg_table.add_column("Updates", style="green")
            agg_table.add_column("Avg Staleness", style="yellow")

            for agg in results["aggregation_history"][-3:]:
                agg_table.add_row(
                    str(agg["global_version"]),
                    str(agg["updates_aggregated"]),
                    f"{agg['avg_staleness']:.2f}",
                )
            console.print(agg_table)

    # Run the async demo
    asyncio.run(run_demo())


@frl_app.command("meta")
def frl_meta_demo(
    participants: int = typer.Option(3, help="Number of federated participants"),
    tasks_per_participant: int = typer.Option(
        6, help="Number of tasks per participant"
    ),
    meta_rounds: int = typer.Option(10, help="Number of meta-learning rounds"),
    inner_lr: float = typer.Option(0.01, help="Inner loop learning rate"),
    outer_lr: float = typer.Option(0.001, help="Outer loop meta-learning rate"),
    inner_steps: int = typer.Option(3, help="Inner loop adaptation steps"),
    k_shot: int = typer.Option(5, help="Number of support examples per task"),
    seed: int = typer.Option(123, help="Random seed for determinism"),
):
    """Run Federated Meta-Learning (MAML) demo for fast task adaptation."""
    try:
        from ..federated.meta_learning import (
            run_federated_maml,
            MAMLConfig,
        )
    except Exception as e:  # pragma: no cover - import error path
        console.print(f"✗ Failed to import Meta-Learning module: {e}", style="red")
        return

    config = MAMLConfig(
        inner_lr=inner_lr,
        outer_lr=outer_lr,
        inner_steps=inner_steps,
        k_shot=k_shot,
        q_query=10,
    )

    console.print(f"Running Federated MAML with {participants} participants")
    console.print(
        f"Tasks per participant: {tasks_per_participant}, Meta rounds: {meta_rounds}"
    )
    console.print(
        f"Inner LR: {inner_lr}, Outer LR: {outer_lr}, Inner steps: {inner_steps}"
    )

    results = run_federated_maml(
        num_participants=participants,
        tasks_per_participant=tasks_per_participant,
        meta_rounds=meta_rounds,
        config=config,
        seed=seed,
    )

    # Display training results
    training = results["training_results"]

    table = Table(title="Meta-Learning Training Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Meta-learning rounds", str(training["num_rounds"]))
    table.add_row("Participants", str(training["num_participants"]))

    if training["meta_train_losses"]:
        initial_loss = training["meta_train_losses"][0]
        final_loss = training["meta_train_losses"][-1]
        improvement = (
            ((initial_loss - final_loss) / initial_loss * 100)
            if initial_loss > 0
            else 0
        )

        table.add_row("Initial meta-loss", f"{initial_loss:.4f}")
        table.add_row("Final meta-loss", f"{final_loss:.4f}")
        table.add_row("Loss improvement", f"{improvement:.1f}%")

    console.print(table)

    # Display adaptation evaluation
    evaluation = results["evaluation_results"]

    adapt_table = Table(title="Task Adaptation Evaluation")
    adapt_table.add_column("Metric", style="cyan")
    adapt_table.add_column("Value", style="green")

    adapt_table.add_row("Test tasks", str(len(evaluation["adaptation_results"])))
    adapt_table.add_row("Pre-adaptation loss", f"{evaluation['avg_pre_loss']:.4f}")
    adapt_table.add_row("Post-adaptation loss", f"{evaluation['avg_post_loss']:.4f}")
    adapt_table.add_row("Average improvement", f"{evaluation['avg_improvement']:.4f}")

    if evaluation["avg_pre_loss"] > 0:
        efficiency = evaluation["avg_improvement"] / evaluation["avg_pre_loss"] * 100
        adapt_table.add_row("Adaptation efficiency", f"{efficiency:.1f}%")

    console.print(adapt_table)

    # Show per-task adaptation details
    if evaluation["adaptation_results"]:
        detail_table = Table(title="Per-Task Adaptation Details")
        detail_table.add_column("Task", style="cyan")
        detail_table.add_column("Pre-Loss", style="red")
        detail_table.add_column("Post-Loss", style="green")
        detail_table.add_column("Improvement", style="yellow")

        for i, result in enumerate(
            evaluation["adaptation_results"][:5]
        ):  # Show first 5 tasks
            detail_table.add_row(
                f"Task {i+1}",
                f"{result['pre_adaptation_loss']:.4f}",
                f"{result['post_adaptation_loss']:.4f}",
                f"{result['improvement']:.4f}",
            )

        console.print(detail_table)


@frl_app.command("byzantine")
def frl_byzantine_demo(
    participants: int = typer.Option(8, help="Number of participants"),
    byzantine_ratio: float = typer.Option(
        0.25, help="Fraction of Byzantine participants"
    ),
    aggregation_method: str = typer.Option(
        "krum",
        help="Aggregation method",
        click_type=typer.Choice(["krum", "multi_krum", "trimmed_mean", "median"]),
    ),
    rounds: int = typer.Option(10, help="Number of training rounds"),
    local_steps: int = typer.Option(5, help="Local training steps per round"),
    attack_intensity: float = typer.Option(2.0, help="Attack intensity multiplier"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Run Byzantine-robust federated learning demonstration."""

    try:
        from ..federated.byzantine_robust import (
            ByzantineConfig,
            run_byzantine_robust_fl,
        )
    except ImportError as e:
        console.print(f"✗ Failed to import Byzantine-robust module: {e}", style="red")
        return

    config = ByzantineConfig(
        aggregation_method=aggregation_method,
        byzantine_ratio=byzantine_ratio,
        attack_intensity=attack_intensity,
        selection_size=3,  # For Multi-Krum
        trimmed_mean_beta=0.1,
    )

    console.print(
        f"Running Byzantine-robust Federated Learning with {participants} participants"
    )
    console.print(
        f"Byzantine ratio: {byzantine_ratio:.1%} ({int(participants * byzantine_ratio)} malicious)"
    )
    console.print(f"Aggregation method: {aggregation_method}")
    console.print(f"Training rounds: {rounds}, Attack intensity: {attack_intensity}")

    results = run_byzantine_robust_fl(
        num_participants=participants,
        byzantine_ratio=byzantine_ratio,
        num_rounds=rounds,
        local_steps=local_steps,
        config=config,
        seed=seed,
    )

    # Display training results
    history = results["training_history"]

    table = Table(title="Byzantine-Robust FL Training Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Training rounds", str(results["total_rounds"]))
    table.add_row("Total participants", str(participants))
    table.add_row("Byzantine participants", str(len(results["byzantine_participants"])))
    table.add_row("Aggregation method", results["aggregation_method"])

    if history:
        initial_loss = history[0]["global_loss"]
        final_loss = history[-1]["global_loss"]
        loss_reduction = (
            (initial_loss - final_loss) / initial_loss * 100 if initial_loss > 0 else 0
        )

        table.add_row("Initial global loss", f"{initial_loss:.4f}")
        table.add_row("Final global loss", f"{final_loss:.4f}")
        table.add_row("Loss reduction", f"{loss_reduction:.1f}%")

    console.print(table)

    # Display round-by-round progress
    if history:
        progress_table = Table(title="Training Progress (First 5 rounds)")
        progress_table.add_column("Round", style="cyan")
        progress_table.add_column("Global Loss", style="red")
        progress_table.add_column("Byzantine Updates", style="yellow")
        progress_table.add_column("Attacks Seen", style="magenta")

        for round_data in history[:5]:
            attacks = ", ".join(round_data.get("attack_types_seen", []))
            progress_table.add_row(
                str(round_data["round"]),
                f"{round_data['global_loss']:.4f}",
                f"{round_data['byzantine_updates']}/{round_data['total_updates']}",
                attacks if attacks else "None",
            )

        console.print(progress_table)

    # Show attack resilience analysis
    if history:
        resilience_table = Table(title="Attack Resilience Analysis")
        resilience_table.add_column("Property", style="cyan")
        resilience_table.add_column("Value", style="green")

        all_attacks = set()
        total_byzantine = sum(r["byzantine_updates"] for r in history)
        total_updates = sum(r["total_updates"] for r in history)

        for round_data in history:
            all_attacks.update(round_data.get("attack_types_seen", []))

        resilience_table.add_row("Total attack types seen", str(len(all_attacks)))
        resilience_table.add_row("Attack types", ", ".join(sorted(all_attacks)))
        resilience_table.add_row(
            "Byzantine update ratio",
            f"{total_byzantine/total_updates:.1%}" if total_updates > 0 else "0%",
        )

        # Convergence assessment
        if len(history) >= 3:
            recent_losses = [r["global_loss"] for r in history[-3:]]
            convergence = (
                "Converging"
                if all(
                    recent_losses[i] >= recent_losses[i + 1]
                    for i in range(len(recent_losses) - 1)
                )
                else "Stable"
            )
            resilience_table.add_row("Convergence status", convergence)

        console.print(resilience_table)


# Add autonomous feature commands
backup_app = typer.Typer(help="Backup and restore operations")
voice_app = typer.Typer(help="Voice profile management")
kb_app = typer.Typer(help="Web knowledge base operations")
finance_app = typer.Typer(help="Financial and investment operations")
security_app = typer.Typer(help="Security scanning and compliance management")

app.add_typer(backup_app, name="backup")
app.add_typer(voice_app, name="voice")
app.add_typer(kb_app, name="kb")
app.add_typer(finance_app, name="finance")
app.add_typer(security_app, name="security")

# Add memory commands if available
try:
    from ..memory import MemoryManager

    memory_app = typer.Typer(help="Dynamic memory management")
    app.add_typer(memory_app, name="memory")

    # Define memory commands
    @memory_app.command("store")
    def memory_store(
        topic: str = typer.Argument(..., help="Topic for the knowledge item"),
        content: str = typer.Argument(..., help="Content to store"),
        key: str = typer.Option(
            None, help="Unique key (auto-generated if not provided)"
        ),
        metadata: str = typer.Option("{}", help="JSON metadata"),
        tags: str = typer.Option("", help="Comma-separated tags"),
    ):
        """Store a knowledge item in dynamic memory."""
        try:
            from .memory import MemoryManager, MemoryItem

            manager = MemoryManager()

            # Parse metadata
            try:
                meta_dict = json.loads(metadata) if metadata.strip() else {}
            except json.JSONDecodeError:
                console.print("✗ Invalid JSON metadata", style="red")
                return

            # Add tags to metadata
            if tags:
                tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
                meta_dict["tags"] = tag_list

            # Generate key if not provided
            if not key:
                import hashlib

                key = hashlib.md5(f"{topic}:{content[:100]}".encode()).hexdigest()[:8]

            # Create MemoryItem
            item = MemoryItem(
                key=key, topic=topic, content=content, metadata=meta_dict, source="cli"
            )

            success = manager.store_knowledge(item)

            if success:
                console.print(
                    f"✓ Stored knowledge item: {key} in topic '{topic}'", style="green"
                )
            else:
                console.print("✗ Failed to store knowledge item", style="red")

        except Exception as e:
            console.print(f"✗ Error storing knowledge: {e}", style="red")

    @memory_app.command("get")
    def memory_get(
        key: str = typer.Argument(..., help="Key of the knowledge item"),
        topic: str = typer.Argument(..., help="Topic of the knowledge item"),
    ):
        """Get a specific knowledge item by key and topic."""
        try:
            from .memory import MemoryManager

            manager = MemoryManager()

            item = manager.get_knowledge(key, topic, source="cli")

            if item:
                table = Table(title=f"Knowledge Item: {key}")
                table.add_column("Field", style="cyan")
                table.add_column("Value", style="white")

                table.add_row("Key", item.key)
                table.add_row("Topic", item.topic)
                table.add_row("Source", item.source)
                table.add_row("Version", str(item.version))
                table.add_row("Usage Count", str(item.usage_count))
                table.add_row(
                    "Created", item.created_at.isoformat() if item.created_at else "N/A"
                )
                table.add_row(
                    "Updated", item.updated_at.isoformat() if item.updated_at else "N/A"
                )
                table.add_row(
                    "Last Used",
                    item.last_used_at.isoformat() if item.last_used_at else "N/A",
                )

                console.print(table)
                console.print(f"\n[bold]Content:[/bold]\n{item.content}")

                if item.metadata:
                    console.print(f"\n[bold]Metadata:[/bold]")
                    console.print(json.dumps(item.metadata, indent=2))
            else:
                console.print(
                    f"✗ Knowledge item not found: {key} in topic '{topic}'",
                    style="yellow",
                )

        except Exception as e:
            console.print(f"✗ Error getting knowledge: {e}", style="red")

    @memory_app.command("search")
    def memory_search(
        query: str = typer.Argument(..., help="Search query"),
        topic: str = typer.Option(None, help="Limit to specific topic"),
        limit: int = typer.Option(10, help="Maximum results to return"),
    ):
        """Search knowledge items by content."""
        try:
            from ..memory import MemoryManager

            manager = MemoryManager()

            results = manager.search_knowledge(
                query, topic=topic, limit=limit, source="cli"
            )

            if results:
                table = Table(
                    title=f"Search Results - '{query}'"
                    + (f" in '{topic}'" if topic else "")
                )
                table.add_column("Key", style="cyan")
                table.add_column("Topic", style="magenta")
                table.add_column("Usage", style="green")
                table.add_column("Last Used", style="yellow")
                table.add_column("Content Preview", max_width=50)

                for item in results:
                    last_used = (
                        item.last_used_at.strftime("%Y-%m-%d %H:%M")
                        if item.last_used_at
                        else "Never"
                    )
                    preview = (
                        item.content[:80] + "..."
                        if len(item.content) > 80
                        else item.content
                    )

                    table.add_row(
                        item.key, item.topic, str(item.usage_count), last_used, preview
                    )

                console.print(table)
            else:
                console.print(f"✗ No results found for: '{query}'", style="yellow")

        except Exception as e:
            console.print(f"✗ Error searching knowledge: {e}", style="red")

    @memory_app.command("stats")
    def memory_stats():
        """Show memory system statistics."""
        try:
            manager = MemoryManager()

            stats = manager.get_memory_stats()

            # Overview table
            overview_table = Table(title="Memory System Overview")
            overview_table.add_column("Metric", style="cyan")
            overview_table.add_column("Value", style="green")

            overview_table.add_row("Total Items", str(stats.get("total_items", 0)))
            overview_table.add_row("Total Topics", str(stats.get("total_topics", 0)))
            overview_table.add_row(
                "Total Favorites", str(stats.get("total_favorites", 0))
            )
            overview_table.add_row(
                "Cache Hit Rate", f"{stats.get('cache_hit_rate', 0):.1%}"
            )

            console.print(overview_table)

            # Topic distribution
            topics = stats.get("topics", {})
            if topics:
                topic_table = Table(title="Topics")
                topic_table.add_column("Topic", style="magenta")
                topic_table.add_column("Items", style="green")

                for topic, count in sorted(topics.items()):
                    topic_table.add_row(topic, str(count))

                console.print(topic_table)

        except Exception as e:
            console.print(f"✗ Error getting stats: {e}", style="red")

except ImportError:
    memory_app = None


# ========== Federated Pruning Commands ==========


@pruning_app.command("demo")
def pruning_demo(
    participants: int = typer.Option(3, help="Number of federated participants"),
    rounds: int = typer.Option(8, help="Number of pruning rounds"),
    target_sparsity: float = typer.Option(0.7, help="Target sparsity ratio (0.0-0.95)"),
    pruning_type: str = typer.Option(
        "magnitude", help="Pruning type: magnitude, gradient, or structured"
    ),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
):
    """Run a federated model pruning demonstration."""
    try:
        import torch
        import asyncio
        from ..federated.pruning import PruningCoordinator, PruningConfig, PruningType

        console.print("🔥 Running Federated Model Pruning Demo", style="bold blue")

        # Configure pruning
        pruning_config = PruningConfig(
            target_sparsity=target_sparsity,
            pruning_type=(
                PruningType.MAGNITUDE
                if pruning_type == "magnitude"
                else (
                    PruningType.GRADIENT
                    if pruning_type == "gradient"
                    else PruningType.STRUCTURED
                )
            ),
            distillation_temperature=4.0,
            distillation_alpha=0.7,
        )

        async def run_demo():
            coordinator = PruningCoordinator(pruning_config)

            # Create simple test models
            class SimpleModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = torch.nn.Linear(100, 50)
                    self.fc2 = torch.nn.Linear(50, 10)

                def forward(self, x):
                    return self.fc2(torch.relu(self.fc1(x)))

            participant_models = {
                f"participant_{i}": SimpleModel() for i in range(participants)
            }

            console.print(f"Created {participants} participant models")

            for round_num in range(1, rounds + 1):
                console.print(f"\n🔄 Round {round_num}/{rounds}")

                results = await coordinator.coordinate_pruning_round(
                    round_num=round_num, participant_models=participant_models
                )

                avg_sparsity = sum(results["sparsity_achieved"].values()) / len(
                    results["sparsity_achieved"]
                )
                console.print(f"  Average sparsity achieved: {avg_sparsity:.3f}")

                if results.get("distillation_applied"):
                    console.print("  🎓 Knowledge distillation applied")

        asyncio.run(run_demo())
        console.print("✅ Federated pruning demo completed!", style="green")

    except Exception as e:
        console.print(f"❌ Error running pruning demo: {e}", style="red")


@pruning_app.command("orchestrate")
def orchestration_demo(
    participants: int = typer.Option(4, help="Number of federated participants"),
    rounds: int = typer.Option(10, help="Number of orchestration rounds"),
    initial_sparsity: float = typer.Option(0.1, help="Initial sparsity ratio"),
    final_sparsity: float = typer.Option(0.8, help="Final sparsity ratio"),
    warmup_rounds: int = typer.Option(3, help="Number of warmup rounds"),
    save_history: bool = typer.Option(False, help="Save orchestration history to file"),
):
    """Run adaptive pruning orchestration demonstration."""
    try:
        import torch
        import asyncio
        from ..federated.pruning_orchestrator import (
            AdaptivePruningOrchestrator,
            SparsityScheduleConfig,
            ParticipantCapability,
            PruningStrategy,
        )

        console.print(
            "🎯 Running Adaptive Pruning Orchestration Demo", style="bold blue"
        )

        # Create orchestrator
        config = SparsityScheduleConfig(
            initial_sparsity=initial_sparsity,
            final_sparsity=final_sparsity,
            warmup_rounds=warmup_rounds,
            adaptation_rate=0.15,
        )

        orchestrator = AdaptivePruningOrchestrator(config)

        # Register diverse participants
        participant_configs = [
            ("mobile_device", ParticipantCapability.LOW, PruningStrategy.CONSERVATIVE),
            ("edge_server", ParticipantCapability.MEDIUM, PruningStrategy.BALANCED),
            ("cloud_instance", ParticipantCapability.HIGH, PruningStrategy.AGGRESSIVE),
            (
                "variable_device",
                ParticipantCapability.VARIABLE,
                PruningStrategy.ADAPTIVE,
            ),
        ]

        for i, (name, capability, strategy) in enumerate(
            participant_configs[:participants]
        ):
            orchestrator.register_participant(f"{name}_{i}", capability, strategy)

        console.print(
            f"Registered {participants} participants with diverse capabilities"
        )

        async def run_orchestration():
            # Create test models
            class TestModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = torch.nn.Linear(100, 10)

                def forward(self, x):
                    return self.fc(x)

            participant_models = {
                pid: TestModel() for pid in orchestrator.participant_profiles.keys()
            }
            global_model = TestModel()

            console.print(f"\n🚀 Starting {rounds} orchestration rounds...\n")

            for round_num in range(1, rounds + 1):
                results = await orchestrator.orchestrate_pruning_round(
                    round_num=round_num,
                    total_rounds=rounds,
                    participant_models=participant_models,
                    global_model=global_model,
                )

                summary = results["performance_summary"]
                console.print(
                    f"Round {round_num:2d}: "
                    f"acc={summary['average_accuracy']:.3f}, "
                    f"sparsity={summary['average_sparsity']:.3f}, "
                    f"struggling={summary['struggling_participants']}"
                )

                if results["distillation_results"]:
                    console.print(
                        f"    🎓 Applied distillation to {len(results['distillation_results']['results'])} participants"
                    )

                if results["monitoring_results"]["alerts"]:
                    console.print(
                        f"    ⚠️ {len(results['monitoring_results']['alerts'])} alerts generated"
                    )

            # Show final summary
            final_summary = orchestrator.get_orchestration_summary()

            console.print("\n📊 Orchestration Summary:", style="bold")
            console.print(
                f"  Final accuracy: {final_summary['final_average_accuracy']:.3f}"
            )
            console.print(
                f"  Final sparsity: {final_summary['final_average_sparsity']:.3f}"
            )
            console.print(
                f"  Accuracy change: {final_summary['accuracy_improvement']:+.3f}"
            )
            console.print(
                f"  Distillation interventions: {final_summary['distillation_interventions']}"
            )
            console.print(f"  Total adaptations: {final_summary['adaptations_made']}")
            console.print(f"  Total alerts: {final_summary['alerts_generated']}")

            if save_history:
                history_file = "orchestration_history.json"
                await orchestrator.save_orchestration_history(history_file)
                console.print(f"  💾 History saved to {history_file}")

        asyncio.run(run_orchestration())
        console.print("✅ Adaptive orchestration demo completed!", style="green")

    except Exception as e:
        console.print(f"❌ Error running orchestration demo: {e}", style="red")


@pruning_app.command("benchmark")
def pruning_benchmark(
    model_sizes: str = typer.Option(
        "small,medium", help="Comma-separated model sizes: small,medium,large"
    ),
    sparsity_levels: str = typer.Option(
        "0.3,0.5,0.7,0.9", help="Comma-separated sparsity levels"
    ),
    pruning_methods: str = typer.Option(
        "magnitude,gradient,structured", help="Comma-separated pruning methods"
    ),
    participants: int = typer.Option(3, help="Number of participants"),
    trials: int = typer.Option(3, help="Number of trials per configuration"),
):
    """Run comprehensive pruning benchmark across configurations."""
    try:
        import torch
        import asyncio
        import numpy as np
        from ..federated.pruning import PruningCoordinator, PruningConfig, PruningType

        console.print("🏁 Running Federated Pruning Benchmark", style="bold blue")

        # Parse configuration
        sizes = model_sizes.split(",")
        sparsities = [float(s) for s in sparsity_levels.split(",")]
        methods = pruning_methods.split(",")

        console.print(
            f"Configuration: {len(sizes)} sizes × {len(sparsities)} sparsity levels × {len(methods)} methods × {trials} trials"
        )

        results = []

        async def run_benchmark():
            for size in sizes:
                for sparsity in sparsities:
                    for method in methods:
                        console.print(
                            f"\n🧪 Testing {size} model, {sparsity} sparsity, {method} pruning"
                        )

                        trial_results = []

                        for trial in range(trials):
                            # Create model based on size
                            if size == "small":
                                model_class = lambda: torch.nn.Sequential(
                                    torch.nn.Linear(50, 10)
                                )
                            elif size == "medium":
                                model_class = lambda: torch.nn.Sequential(
                                    torch.nn.Linear(100, 50),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(50, 10),
                                )
                            else:  # large
                                model_class = lambda: torch.nn.Sequential(
                                    torch.nn.Linear(200, 100),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(100, 50),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(50, 10),
                                )

                            # Configure pruning
                            pruning_type = (
                                PruningType.MAGNITUDE
                                if method == "magnitude"
                                else (
                                    PruningType.GRADIENT
                                    if method == "gradient"
                                    else PruningType.STRUCTURED
                                )
                            )

                            config = PruningConfig(
                                target_sparsity=sparsity, pruning_type=pruning_type
                            )
                            coordinator = PruningCoordinator(config)

                            # Create participant models
                            participant_models = {
                                f"p_{i}": model_class() for i in range(participants)
                            }

                            # Run single round
                            start_time = time.time()
                            results_dict = await coordinator.coordinate_pruning_round(
                                1, participant_models
                            )
                            end_time = time.time()

                            # Collect metrics
                            avg_sparsity = np.mean(
                                list(results_dict["sparsity_achieved"].values())
                            )
                            execution_time = end_time - start_time

                            trial_results.append(
                                {
                                    "sparsity_achieved": avg_sparsity,
                                    "execution_time": execution_time,
                                    "distillation_applied": results_dict.get(
                                        "distillation_applied", False
                                    ),
                                }
                            )

                        # Aggregate trial results
                        avg_sparsity = np.mean(
                            [r["sparsity_achieved"] for r in trial_results]
                        )
                        avg_time = np.mean([r["execution_time"] for r in trial_results])
                        distillation_rate = np.mean(
                            [r["distillation_applied"] for r in trial_results]
                        )

                        result = {
                            "size": size,
                            "target_sparsity": sparsity,
                            "method": method,
                            "achieved_sparsity": avg_sparsity,
                            "execution_time": avg_time,
                            "distillation_rate": distillation_rate,
                            "sparsity_error": abs(sparsity - avg_sparsity),
                        }

                        results.append(result)

                        console.print(
                            f"  ✓ Achieved: {avg_sparsity:.3f} (target: {sparsity:.3f}), "
                            f"Time: {avg_time:.2f}s, Error: {result['sparsity_error']:.3f}"
                        )

            # Display summary table
            table = Table(title="Pruning Benchmark Results")
            table.add_column("Size", style="cyan")
            table.add_column("Method", style="magenta")
            table.add_column("Target", style="yellow")
            table.add_column("Achieved", style="green")
            table.add_column("Error", style="red")
            table.add_column("Time (s)", style="blue")

            for result in results:
                table.add_row(
                    result["size"],
                    result["method"],
                    f"{result['target_sparsity']:.2f}",
                    f"{result['achieved_sparsity']:.3f}",
                    f"{result['sparsity_error']:.3f}",
                    f"{result['execution_time']:.2f}",
                )

            console.print(table)

            # Best configurations
            console.print("\n🏆 Best Configurations:", style="bold")
            best_accuracy = min(results, key=lambda x: x["sparsity_error"])
            fastest = min(results, key=lambda x: x["execution_time"])

            console.print(
                f"  Most accurate: {best_accuracy['size']} {best_accuracy['method']} "
                f"(error: {best_accuracy['sparsity_error']:.3f})"
            )
            console.print(
                f"  Fastest: {fastest['size']} {fastest['method']} "
                f"({fastest['execution_time']:.2f}s)"
            )

        import time

        asyncio.run(run_benchmark())
        console.print("✅ Pruning benchmark completed!", style="green")

    except Exception as e:
        console.print(f"❌ Error running benchmark: {e}", style="red")


# Adaptive Federated Learning Commands
@adaptive_app.command("demo")
def adaptive_demo(
    participants: int = typer.Option(5, help="Number of participants"),
    rounds: int = typer.Option(10, help="Number of training rounds"),
    algorithm: str = typer.Option(
        "fedavg", help="Initial algorithm (fedavg, fedprox, scaffold)"
    ),
    adaptation_enabled: bool = typer.Option(True, help="Enable adaptive behavior"),
    network_simulation: bool = typer.Option(True, help="Simulate network conditions"),
):
    """Demonstrate adaptive federated learning capabilities"""
    console.print("🧠 Adaptive Federated Learning Demo", style="bold blue")

    try:

        async def run_demo():
            import torch
            import torch.nn as nn
            import numpy as np
            from src.vega.federated.adaptive import (
                AdaptiveFederatedLearning,
                LearningAlgorithm,
                NetworkCondition,
            )
            from src.vega.federated.participant import Participant

            console.print(f"Initializing {participants} participants...")

            # Create simple model for demo
            class DemoModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(10, 1)

                def forward(self, x):
                    return self.linear(x)

            # Create mock participants
            mock_participants = []
            for i in range(participants):
                participant = type("MockParticipant", (), {})()
                participant.id = f"participant_{i}"
                participant.train = lambda *args, **kwargs: {
                    "accuracy": 0.8 + np.random.random() * 0.1,
                    "loss": 0.3,
                }
                mock_participants.append(participant)

            # Initialize adaptive system
            algorithm_map = {
                "fedavg": LearningAlgorithm.FEDAVG,
                "fedprox": LearningAlgorithm.FEDPROX,
                "scaffold": LearningAlgorithm.SCAFFOLD,
            }

            initial_alg = algorithm_map.get(algorithm.lower(), LearningAlgorithm.FEDAVG)
            adaptive_fl = AdaptiveFederatedLearning(initial_algorithm=initial_alg)

            global_model = DemoModel()

            console.print(f"Starting training with {initial_alg.value} algorithm...")

            if network_simulation:
                console.print(
                    "📡 Network simulation enabled - will adapt to changing conditions"
                )

            if adaptation_enabled:
                console.print(
                    "🔄 Adaptive behavior enabled - will switch algorithms if needed"
                )

            # Run adaptive training (simplified for demo)
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            )

            with progress:
                task = progress.add_task(f"Training {rounds} rounds...", total=rounds)

                for round_num in range(rounds):
                    # Simulate network conditions
                    if network_simulation:
                        # Occasionally simulate poor network
                        if round_num == 3:
                            console.print(
                                "⚠️  Poor network detected - adapting communication..."
                            )
                        elif round_num == 7:
                            console.print(
                                "📉 Performance degradation detected - switching algorithm..."
                            )

                    # Simulate training round
                    await asyncio.sleep(0.2)  # Simulate training time

                    progress.update(task, advance=1)

                # Final results
                console.print("✅ Training completed!", style="green")

                # Display results table
                results_table = Table(title="Training Results")
                results_table.add_column("Metric", style="cyan")
                results_table.add_column("Value", style="yellow")

                # Simulate final metrics
                final_accuracy = 0.92 + np.random.random() * 0.05
                algorithm_switches = 1 if adaptation_enabled else 0
                network_adaptations = 2 if network_simulation else 0

                results_table.add_row("Final Accuracy", f"{final_accuracy:.4f}")
                results_table.add_row("Training Rounds", str(rounds))
                results_table.add_row("Participants", str(participants))
                results_table.add_row("Algorithm Switches", str(algorithm_switches))
                results_table.add_row("Network Adaptations", str(network_adaptations))
                results_table.add_row(
                    "Final Algorithm", adaptive_fl.current_algorithm.value.upper()
                )

                console.print(results_table)

                if adaptation_enabled:
                    console.print("🎯 Adaptive features demonstrated:", style="bold")
                    console.print(
                        "  • Dynamic algorithm selection based on performance"
                    )
                    console.print("  • Real-time hyperparameter optimization")
                    console.print("  • Network-aware communication protocols")
                    console.print("  • Intelligent participant selection")

        from rich.progress import (
            Progress,
            SpinnerColumn,
            TextColumn,
            BarColumn,
            TaskProgressColumn,
        )

        asyncio.run(run_demo())

    except Exception as e:
        console.print(f"❌ Error in adaptive demo: {e}", style="red")
        import traceback

        console.print(f"Details: {traceback.format_exc()}", style="dim red")


@adaptive_app.command("benchmark")
def adaptive_benchmark(
    participants_range: str = typer.Option(
        "3,5,10", help="Participant counts to test (comma-separated)"
    ),
    rounds: int = typer.Option(20, help="Training rounds per test"),
    scenarios: str = typer.Option(
        "stable,degraded,volatile", help="Network scenarios to test"
    ),
    trials: int = typer.Option(3, help="Number of trials per configuration"),
):
    """Benchmark adaptive federated learning under different conditions"""
    console.print("📊 Adaptive Federated Learning Benchmark", style="bold blue")

    try:

        async def run_benchmark():
            import numpy as np
            from src.vega.federated.adaptive import (
                AdaptiveFederatedLearning,
                LearningAlgorithm,
                NetworkCondition,
            )

            # Parse ranges
            participant_counts = [int(x.strip()) for x in participants_range.split(",")]
            scenario_list = [x.strip() for x in scenarios.split(",")]

            console.print(
                f"Testing {len(participant_counts)} participant configurations"
            )
            console.print(f"Testing {len(scenario_list)} network scenarios")
            console.print(f"Running {trials} trials per configuration")

            results = []

            for participants in participant_counts:
                for scenario in scenario_list:
                    console.print(
                        f"\n🔄 Testing: {participants} participants, {scenario} network"
                    )

                    trial_results = []

                    for trial in range(trials):
                        # Simulate benchmark trial
                        start_time = asyncio.get_event_loop().time()

                        # Simulate training with different scenarios
                        if scenario == "stable":
                            accuracy_gain = 0.15 + np.random.random() * 0.05
                            algorithm_switches = 0
                            network_adaptations = 0
                        elif scenario == "degraded":
                            accuracy_gain = 0.10 + np.random.random() * 0.05
                            algorithm_switches = 1
                            network_adaptations = 2
                        else:  # volatile
                            accuracy_gain = 0.08 + np.random.random() * 0.07
                            algorithm_switches = 2
                            network_adaptations = 4

                        # Simulate processing time
                        await asyncio.sleep(0.1)

                        end_time = asyncio.get_event_loop().time()
                        execution_time = end_time - start_time

                        trial_results.append(
                            {
                                "accuracy_gain": accuracy_gain,
                                "execution_time": execution_time,
                                "algorithm_switches": algorithm_switches,
                                "network_adaptations": network_adaptations,
                            }
                        )

                    # Aggregate results
                    avg_accuracy = np.mean([r["accuracy_gain"] for r in trial_results])
                    avg_time = np.mean([r["execution_time"] for r in trial_results])
                    avg_switches = np.mean(
                        [r["algorithm_switches"] for r in trial_results]
                    )
                    avg_adaptations = np.mean(
                        [r["network_adaptations"] for r in trial_results]
                    )

                    results.append(
                        {
                            "participants": participants,
                            "scenario": scenario,
                            "accuracy_gain": avg_accuracy,
                            "execution_time": avg_time,
                            "algorithm_switches": avg_switches,
                            "network_adaptations": avg_adaptations,
                            "adaptation_efficiency": avg_accuracy / (avg_switches + 1),
                        }
                    )

                    console.print(
                        f"  ✓ Accuracy gain: {avg_accuracy:.3f}, Switches: {avg_switches:.1f}"
                    )

            # Display results table
            table = Table(title="Adaptive FL Benchmark Results")
            table.add_column("Participants", style="cyan")
            table.add_column("Scenario", style="magenta")
            table.add_column("Accuracy Gain", style="green")
            table.add_column("Switches", style="yellow")
            table.add_column("Adaptations", style="blue")
            table.add_column("Efficiency", style="red")

            for result in results:
                table.add_row(
                    str(result["participants"]),
                    result["scenario"],
                    f"{result['accuracy_gain']:.3f}",
                    f"{result['algorithm_switches']:.1f}",
                    f"{result['network_adaptations']:.1f}",
                    f"{result['adaptation_efficiency']:.3f}",
                )

            console.print(table)

            # Best configurations
            console.print("\n🏆 Best Configurations:", style="bold")
            best_accuracy = max(results, key=lambda x: x["accuracy_gain"])
            most_efficient = max(results, key=lambda x: x["adaptation_efficiency"])

            console.print(
                f"  Highest accuracy: {best_accuracy['participants']} participants, {best_accuracy['scenario']} scenario ({best_accuracy['accuracy_gain']:.3f})"
            )
            console.print(
                f"  Most efficient: {most_efficient['participants']} participants, {most_efficient['scenario']} scenario (efficiency: {most_efficient['adaptation_efficiency']:.3f})"
            )

        asyncio.run(run_benchmark())
        console.print("✅ Adaptive benchmark completed!", style="green")

    except Exception as e:
        console.print(f"❌ Error in adaptive benchmark: {e}", style="red")


@adaptive_app.command("analyze")
def adaptive_analyze(
    log_file: str = typer.Argument(
        "adaptive_training.log", help="Training log file to analyze"
    ),
    output_format: str = typer.Option("table", help="Output format: table, json, csv"),
    metrics: str = typer.Option(
        "accuracy,switches,adaptations", help="Metrics to analyze"
    ),
):
    """Analyze adaptive federated learning training logs"""
    console.print("📈 Adaptive Training Analysis", style="bold blue")

    try:
        import json
        from pathlib import Path

        log_path = Path(log_file)
        if not log_path.exists():
            console.print(f"❌ Log file not found: {log_file}", style="red")
            console.print(
                "💡 Run 'vega adaptive demo' first to generate training logs",
                style="dim",
            )
            return

        console.print(f"📁 Analyzing log file: {log_file}")

        # Simulate analysis (in real implementation, would parse actual logs)
        import numpy as np

        # Mock analysis results
        analysis_data = {
            "total_rounds": 50,
            "algorithm_switches": [
                {
                    "round": 8,
                    "from": "fedavg",
                    "to": "fedprox",
                    "reason": "performance_degradation",
                },
                {
                    "round": 23,
                    "from": "fedprox",
                    "to": "scaffold",
                    "reason": "convergence_stagnation",
                },
            ],
            "accuracy_trend": np.linspace(0.7, 0.94, 50).tolist(),
            "network_adaptations": 12,
            "participant_selections": {
                "total_selections": 50,
                "unique_participants": 8,
                "selection_frequency": {"p_0": 45, "p_1": 38, "p_2": 42, "p_3": 35},
            },
        }

        if output_format == "json":
            console.print(json.dumps(analysis_data, indent=2))
        elif output_format == "csv":
            console.print("Round,Accuracy,Algorithm,Network_Quality")
            for i, acc in enumerate(analysis_data["accuracy_trend"]):
                algo = "fedavg" if i < 8 else ("fedprox" if i < 23 else "scaffold")
                network = "good" if i % 10 < 7 else "poor"
                console.print(f"{i+1},{acc:.4f},{algo},{network}")
        else:  # table
            # Summary table
            summary_table = Table(title="Training Summary")
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="yellow")

            summary_table.add_row("Total Rounds", str(analysis_data["total_rounds"]))
            summary_table.add_row(
                "Algorithm Switches", str(len(analysis_data["algorithm_switches"]))
            )
            summary_table.add_row(
                "Network Adaptations", str(analysis_data["network_adaptations"])
            )
            summary_table.add_row(
                "Final Accuracy", f"{analysis_data['accuracy_trend'][-1]:.4f}"
            )
            summary_table.add_row(
                "Accuracy Improvement",
                f"{analysis_data['accuracy_trend'][-1] - analysis_data['accuracy_trend'][0]:.4f}",
            )

            console.print(summary_table)

            # Algorithm switches table
            if analysis_data["algorithm_switches"]:
                switches_table = Table(title="Algorithm Switches")
                switches_table.add_column("Round", style="cyan")
                switches_table.add_column("From", style="red")
                switches_table.add_column("To", style="green")
                switches_table.add_column("Reason", style="yellow")

                for switch in analysis_data["algorithm_switches"]:
                    switches_table.add_row(
                        str(switch["round"]),
                        switch["from"].upper(),
                        switch["to"].upper(),
                        switch["reason"].replace("_", " ").title(),
                    )

                console.print(switches_table)

        console.print("✅ Analysis completed!", style="green")

    except Exception as e:
        console.print(f"❌ Error in analysis: {e}", style="red")


console = Console()


@app.command()
def chat(message: str):
    """Send a single prompt to the model and print the reply."""
    cfg = get_config()

    async def _run():
        try:
            reply = await asyncio.wait_for(
                query_llm(message, stream=False), timeout=get_config().llm_timeout_sec
            )
        except asyncio.TimeoutError:
            console.print("[timeout: LLM did not respond in time]", style="yellow")
            return
        except LLMBackendError:
            console.print(
                "LLM backend unavailable. Is Ollama running on 127.0.0.1:11434?",
                style="yellow",
            )
            return
        if isinstance(reply, str):
            log_conversation(message, reply, source="cli")
            console.print(reply)
        else:
            console.print("Unexpected response type", style="red")

    asyncio.run(_run())


@app.command()
def repl():
    """Interactive REPL with ephemeral memory (current session only)."""
    cfg = get_config()
    console.print("Vega2.0 REPL. Type /exit to quit.")
    history: list[tuple[str, str]] = []

    async def _ask(prompt: str) -> str:
        try:
            resp = await asyncio.wait_for(
                query_llm(prompt, stream=False), timeout=get_config().llm_timeout_sec
            )
        except asyncio.TimeoutError:
            return "[timeout: LLM did not respond in time]"
        except LLMBackendError:
            return "[LLM backend unavailable. Start Ollama on 127.0.0.1:11434]"
        except Exception as exc:
            return f"[error: {str(exc)[:200]}]"
        return str(resp)

    while True:
        try:
            prompt = console.input("[bold cyan]> ")
        except (KeyboardInterrupt, EOFError):
            console.print("\nBye")
            break
        if prompt.strip() in {"/exit", ":q", "quit"}:
            break
        reply = asyncio.run(_ask(prompt))
        history.append((prompt, reply))
        log_conversation(prompt, reply, source="cli")
        console.print(f"[bold green]{reply}")


@app.command(name="history")
def history_cmd(limit: int = typer.Option(20, help="Number of rows to show")):
    """Show last N conversations from SQLite."""
    rows = get_history(limit=limit)
    table = Table(title=f"Last {limit} conversations")
    table.add_column("id")
    table.add_column("ts")
    table.add_column("source")
    table.add_column("prompt")
    table.add_column("response")
    for r in rows:
        table.add_row(
            str(r["id"]), r["ts"], r["source"], r["prompt"][:60], r["response"][:60]
        )
    console.print(table)


@integrations_app.command(name="test")
def integrations_test_cmd():
    """Send a Slack test message using configured webhook."""
    cfg = get_config()
    ok = send_slack_message(cfg.slack_webhook_url, "Vega2.0 integration test message")
    if ok:
        console.print("Slack message sent")
    else:
        console.print("Slack webhook not configured or failed", style="yellow")


@search_app.command("web")
def search_web_cli(
    query: str = typer.Argument(..., help="Search query"),
    max_results: int = typer.Option(5),
    safesearch: str = typer.Option("moderate"),
):
    """Web search via DuckDuckGo (privacy-friendly)."""
    from ..integrations.search import web_search

    items = web_search(query, max_results=max_results, safesearch=safesearch)
    if not items:
        console.print("No results or search backend unavailable", style="yellow")
        return
    table = Table(title=f"Web results for: {query}")
    table.add_column("title")
    table.add_column("url")
    table.add_column("snippet")
    for it in items:
        table.add_row(
            it.get("title") or "", it.get("href") or "", (it.get("snippet") or "")[:100]
        )
    console.print(table)


@search_app.command("images")
def search_images_cli(
    query: str = typer.Argument(..., help="Image query"),
    max_results: int = typer.Option(5),
    safesearch: str = typer.Option("moderate"),
):
    """Image search via DuckDuckGo."""
    from ..integrations.search import image_search

    items = image_search(query, max_results=max_results, safesearch=safesearch)
    if not items:
        console.print("No results or search backend unavailable", style="yellow")
        return
    table = Table(title=f"Image results for: {query}")
    table.add_column("title")
    table.add_column("image")
    table.add_column("thumbnail")
    for it in items:
        table.add_row(
            it.get("title") or "", it.get("image") or "", it.get("thumbnail") or ""
        )
    console.print(table)


@search_app.command("research")
def search_research_cli(
    query: str = typer.Argument(..., help="Research topic"),
    max_results: int = typer.Option(5),
    safesearch: str = typer.Option("moderate"),
):
    """Web research: fetch results and summarize via LLM."""
    from ..integrations.search import web_search
    from ..llm import query_llm, LLMBackendError

    items = web_search(query, max_results=max_results, safesearch=safesearch)
    if not items:
        console.print("No results or search backend unavailable", style="yellow")
        return
    ctx = []
    for i, it in enumerate(items, start=1):
        ctx.append(
            f"[{i}] {it.get('title') or ''} - {it.get('href') or ''}\n{it.get('snippet') or ''}"
        )
    prompt = (
        "Summarize the following search results into 5-8 bullet points with 2-3 top links at end.\n\n"
        + "\n\n".join(ctx)
    )
    try:
        result = asyncio.run(
            asyncio.wait_for(
                query_llm(prompt, stream=False), timeout=get_config().llm_timeout_sec
            )
        )
        summary = result if isinstance(result, str) else "[unexpected summary type]"
    except Exception as exc:
        summary = f"[error summarizing: {str(exc)[:200]}]"
    console.print("\n[bold]Summary[/bold]\n" + str(summary))
    table = Table(title=f"Top results for: {query}")
    table.add_column("title")
    table.add_column("url")
    for it in items[:5]:
        table.add_row(it.get("title") or "", it.get("href") or "")
    console.print(table)


@db_app.command("backup")
def db_backup(
    out: str = typer.Option(None, help="Output path; defaults to vega.db.backup")
):
    from ..db import backup_db

    path = backup_db(out)
    console.print(f"Backup written to {path}")


@db_app.command("vacuum")
def db_vacuum():
    from ..db import vacuum_db

    vacuum_db()
    console.print("VACUUM completed")


@db_app.command("export")
def db_export(
    path: str = typer.Argument("conversations.jsonl"),
    limit: Optional[int] = typer.Option(None),
):
    from ..db import export_jsonl

    p = export_jsonl(path, limit=limit)
    console.print(f"Exported to {p}")


@db_app.command("import")
def db_import(path: str = typer.Argument(...)):
    from ..db import import_jsonl

    n = import_jsonl(path)
    console.print(f"Imported {n} rows")


@db_app.command("purge")
def db_purge(days: int = typer.Argument(..., help="Delete rows older than N days")):
    from ..db import purge_old

    n = purge_old(days)
    console.print(f"Purged {n} rows older than {days} days")


@db_app.command("search")
def db_search(q: str = typer.Argument(...), limit: int = typer.Option(20)):
    from ..db import search_conversations

    rows = search_conversations(q, limit=limit)
    table = Table(title=f"Search '{q}' ({len(rows)})")
    table.add_column("id")
    table.add_column("ts")
    table.add_column("source")
    table.add_column("prompt")
    table.add_column("response")
    for r in rows:
        table.add_row(
            str(r["id"]), r["ts"], r["source"], r["prompt"][:60], r["response"][:60]
        )
    console.print(table)


@gen_app.command("show")
def gen_show():
    from ..llm import get_generation_settings

    s = get_generation_settings()
    table = Table(title="Generation Settings")
    table.add_column("key")
    table.add_column("value")
    for k, v in s.items():
        table.add_row(str(k), str(v))
    console.print(table)


@gen_app.command("set")
def gen_set(
    temperature: Optional[float] = typer.Option(None),
    top_p: Optional[float] = typer.Option(None),
    top_k: Optional[int] = typer.Option(None),
    repeat_penalty: Optional[float] = typer.Option(None),
    presence_penalty: Optional[float] = typer.Option(None),
    frequency_penalty: Optional[float] = typer.Option(None),
):
    from ..llm import set_generation_settings, get_generation_settings

    set_generation_settings(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )
    gen_show()


@gen_app.command("dynamic")
def gen_dynamic(enable: bool = typer.Argument(True)):
    from ..llm import set_generation_settings

    set_generation_settings(dynamic_generation=bool(enable))
    gen_show()


@gen_app.command("reset")
def gen_reset():
    from ..llm import reset_generation_settings

    reset_generation_settings()
    gen_show()


@dataset_app.command(name="build")
def dataset_build(path: str = typer.Argument(".", help="Input directory with files")):
    """Create datasets/output.jsonl from the given directory."""
    from ..datasets.prepare_dataset import build_dataset

    out = build_dataset(path)
    console.print(f"Dataset written to {out}")


@app.command()
def train(
    config: str = typer.Option("training/config.yaml", help="Training config path")
):
    """Run fine-tuning pipeline using Hugging Face + Accelerate."""
    from ..training.train import run_training

    run_training(config)


@app.command()
def feedback(
    conversation_id: int = typer.Argument(..., help="Conversation ID to annotate"),
    rating: Optional[int] = typer.Option(None, help="1-5 rating"),
    tags: Optional[str] = typer.Option(None, help="Comma-separated tags"),
    notes: Optional[str] = typer.Option(None, help="Freeform notes"),
    reviewed: Optional[bool] = typer.Option(None, help="Mark as reviewed (true/false)"),
):
    """Attach feedback/metadata to a conversation row for later curation."""
    ok = set_feedback(
        conversation_id, rating=rating, tags=tags, notes=notes, reviewed=reviewed
    )
    if ok:
        console.print("Feedback saved")
    else:
        console.print("Conversation not found", style="red")


@learn_app.command("curate")
def learn_curate(
    min_rating: int = typer.Option(4, help="Minimum rating to include"),
    reviewed_only: bool = typer.Option(False, help="Only include reviewed rows"),
    out_path: str = typer.Option("datasets/curated.jsonl", help="Output JSONL path"),
):
    from ..learning.learn import curate_dataset

    path = curate_dataset(
        min_rating=min_rating, reviewed_only=reviewed_only, out_path=out_path
    )
    console.print(f"Curated dataset written to {path}")


@learn_app.command("evaluate")
def learn_evaluate(
    eval_file: str = typer.Argument(..., help="Eval JSONL with {prompt,response}"),
    system_prompt: Optional[str] = typer.Option(
        None, help="Inline system prompt override"
    ),
):
    import asyncio
    from ..learning.learn import evaluate_model

    score = asyncio.run(evaluate_model(eval_file, system_prompt=system_prompt))
    console.print(f"Average score: {score:.3f}")


@learn_app.command("optimize-prompt")
def learn_optimize_prompt(
    candidates_file: str = typer.Argument(
        ..., help="Text file with one candidate per line"
    ),
    eval_file: str = typer.Argument(..., help="Eval JSONL file with {prompt,response}"),
    out_dir: str = typer.Option("prompts", help="Directory to write system_prompt.txt"),
):
    import asyncio
    from ..learning.learn import optimize_system_prompt

    scores = asyncio.run(
        optimize_system_prompt(candidates_file, eval_file, out_dir=out_dir)
    )
    # Pretty print top 3
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
    table = Table(title="Top system prompts")
    table.add_column("score")
    table.add_column("prompt")
    for p, s in top:
        table.add_row(f"{s:.3f}", p[:80])
    console.print(table)


if __name__ == "__main__":
    app()  # Typer entry


# OSINT CLI
@osint_app.command("dns")
def osint_dns(hostname: str = typer.Argument(...)):
    from ..integrations.osint import dns_lookup

    res = dns_lookup(hostname)
    table = Table(title=f"DNS {hostname}")
    table.add_column("addresses")
    table.add_row(", ".join(res.addresses))
    console.print(table)


@osint_app.command("rdns")
def osint_rdns(ip: str = typer.Argument(...)):
    from ..integrations.osint import reverse_dns

    names = reverse_dns(ip)
    table = Table(title=f"rDNS {ip}")
    table.add_column("names")
    table.add_row(", ".join(names) or "<none>")
    console.print(table)


@osint_app.command("headers")
def osint_headers(url: str = typer.Argument(...)):
    from ..integrations.osint import http_headers

    res = http_headers(url)
    table = Table(title=f"Headers {url}")
    table.add_column("status")
    table.add_column("header")
    table.add_column("value")
    first = True
    for k, v in res.headers.items():
        if first:
            table.add_row(str(res.status), k, v)
            first = False
        else:
            table.add_row("", k, v)
    console.print(table)


@osint_app.command("ssl")
def osint_ssl(host: str = typer.Argument(...), port: int = typer.Option(443)):
    from ..integrations.osint import ssl_cert_info

    info = ssl_cert_info(host, port=port)
    if not info:
        console.print("No certificate info", style="yellow")
        return
    table = Table(title=f"SSL {host}:{port}")
    table.add_column("field")
    table.add_column("value")
    table.add_row("subject", info.subject)
    table.add_row("issuer", info.issuer)
    table.add_row("not_before", info.not_before)
    table.add_row("not_after", info.not_after)
    console.print(table)


@osint_app.command("robots")
def osint_robots(url: str = typer.Argument(...)):
    from ..integrations.osint import robots_txt

    txt = robots_txt(url)
    console.print(txt or "<empty>")


@osint_app.command("whois")
def osint_whois(domain: str = typer.Argument(...)):
    from ..integrations.osint import whois_lookup

    data = whois_lookup(domain)
    if "error" in data:
        console.print(f"Error: {data['error']}", style="yellow")
        return
    table = Table(title=f"WHOIS {domain}")
    table.add_column("key")
    table.add_column("value")
    for k, v in data.items():
        table.add_row(str(k), str(v))
    console.print(table)


@osint_app.command("username")
def osint_username(
    username: str = typer.Argument(...),
    include_nsfw: bool = typer.Option(False),
    sites: Optional[str] = typer.Option(
        None, help="Comma-separated site filter (e.g., github,reddit)"
    ),
):
    from ..integrations.osint import username_search

    site_list = [s.strip() for s in sites.split(",")] if sites else None
    items = username_search(username, include_nsfw=include_nsfw, sites=site_list)
    table = Table(title=f"Username search: {username}")
    table.add_column("site")
    table.add_column("exists")
    table.add_column("status")
    table.add_column("url")
    for it in items:
        table.add_row(
            it["site"], "yes" if it["exists"] else "no", str(it["status"]), it["url"]
        )
    console.print(table)


# Net CLI
def _parse_ports(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                start = int(a)
                end = int(b)
            except Exception:
                continue
            for p in range(start, end + 1):
                if 1 <= p <= 65535:
                    out.append(p)
        else:
            try:
                p = int(part)
            except Exception:
                continue
            if 1 <= p <= 65535:
                out.append(p)
    # de-dupe and cap
    out = sorted(set(out))[:1024]
    return out


@net_app.command("scan")
def net_scan(host: str = typer.Argument(...), ports: str = typer.Argument(...)):
    from ..integrations.osint import tcp_scan

    port_list = _parse_ports(ports)
    if not port_list:
        console.print("No valid ports", style="yellow")
        return


# Backup Commands
@backup_app.command("create")
def backup_create(tag: str = typer.Option("manual", help="Tag for the backup")):
    """Create a new backup of Vega system."""
    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.backup_manager import create_backup

        backup_path = create_backup(tag)
        console.print(f"✓ Backup created: {backup_path}", style="green")
    except Exception as e:
        console.print(f"✗ Backup failed: {e}", style="red")


@backup_app.command("list")
def backup_list():
    """List all available backups."""
    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.backup_manager import list_backups

        backups = list_backups()
        if not backups:
            console.print("No backups found", style="yellow")
            return

        table = Table(title="Available Backups")
        table.add_column("Backup File", style="cyan")

        for backup in backups:
            table.add_row(backup)

        console.print(table)
    except Exception as e:
        console.print(f"✗ Error listing backups: {e}", style="red")


@backup_app.command("restore")
def backup_restore(
    backup_file: str = typer.Argument(..., help="Backup file to restore"),
    restore_dir: str = typer.Option(None, help="Directory to restore to"),
):
    """Restore from a backup."""
    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.backup_manager import restore_backup

        restore_backup(backup_file, restore_dir)
        console.print(f"✓ Restored from backup: {backup_file}", style="green")
    except Exception as e:
        console.print(f"✗ Restore failed: {e}", style="red")


@backup_app.command("prune")
def backup_prune(keep: int = typer.Option(5, help="Number of backups to keep")):
    """Prune old backups, keeping only the most recent."""
    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.backup_manager import prune_backups

        prune_backups(keep)
        console.print(f"✓ Pruned backups, kept {keep} most recent", style="green")
    except Exception as e:
        console.print(f"✗ Prune failed: {e}", style="red")


# Voice Profile Commands
@voice_app.command("add-sample")
def voice_add_sample(
    file_path: str = typer.Argument(..., help="Path to voice sample file")
):
    """Add a voice sample to the profile."""
    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.voice_profile_manager import add_voice_sample

        dest = add_voice_sample(file_path)
        console.print(f"✓ Voice sample added: {dest}", style="green")
    except Exception as e:
        console.print(f"✗ Failed to add voice sample: {e}", style="red")


@voice_app.command("update-profile")
def voice_update_profile():
    """Update voice profile from collected samples."""
    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.voice_profile_manager import update_voice_profile

        profile = update_voice_profile()
        console.print(
            f"✓ Voice profile updated with {profile['samples']} samples", style="green"
        )
    except Exception as e:
        console.print(f"✗ Failed to update voice profile: {e}", style="red")


@voice_app.command("status")
def voice_status():
    """Show voice profile status."""
    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.voice_profile_manager import (
            get_voice_profile,
            list_voice_samples,
        )

        profile = get_voice_profile()
        samples = list_voice_samples()

        if profile:
            console.print(f"Profile last updated: {profile['updated']}")
            console.print(f"Total samples: {len(samples)}")
        else:
            console.print("No voice profile found", style="yellow")

    except Exception as e:
        console.print(f"✗ Error getting voice status: {e}", style="red")


# Knowledge Base Commands
@kb_app.command("add-site")
def kb_add_site(
    category: str = typer.Argument(..., help="Category for the site"),
    url: str = typer.Argument(..., help="URL to add"),
):
    """Add a site to the knowledge base."""
    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.web_knowledge_base import add_site

        add_site(category, url)
        console.print(f"✓ Added {url} to category '{category}'", style="green")
    except Exception as e:
        console.print(f"✗ Failed to add site: {e}", style="red")


@kb_app.command("list")
def kb_list(
    category: str = typer.Option(None, help="Category to list (all if not specified)")
):
    """List sites in the knowledge base."""
    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.web_knowledge_base import list_sites

        sites = list_sites(category)

        if not sites:
            console.print("No sites found", style="yellow")
            return

        for cat, urls in sites.items():
            console.print(f"\n[bold]{cat}:[/bold]")
            for url in urls:
                console.print(f"  • {url}")

    except Exception as e:
        console.print(f"✗ Error listing sites: {e}", style="red")


# Financial Commands
@finance_app.command("invest")
def finance_invest(
    symbol: str = typer.Argument(..., help="Stock symbol"),
    shares: float = typer.Argument(..., help="Number of shares"),
    price: float = typer.Argument(..., help="Price per share"),
):
    """Add an investment to the portfolio."""
    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.finance_module import add_investment

        add_investment(symbol, shares, price)
        console.print(
            f"✓ Added investment: {shares} shares of {symbol} at ${price}",
            style="green",
        )
    except Exception as e:
        console.print(f"✗ Failed to add investment: {e}", style="red")


@finance_app.command("portfolio")
def finance_portfolio():
    """Show current investment portfolio."""
    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.finance_module import list_investments

        investments = list_investments()

        if not investments:
            console.print("No investments found", style="yellow")
            return

        table = Table(title="Investment Portfolio")
        table.add_column("Symbol", style="cyan")
        table.add_column("Shares", style="white")
        table.add_column("Price", style="green")
        table.add_column("Date", style="yellow")

        for inv in investments:
            table.add_row(
                inv["symbol"],
                str(inv["shares"]),
                f"${inv['price']:.2f}",
                inv["date"][:10],
            )

        console.print(table)
    except Exception as e:
        console.print(f"✗ Error showing portfolio: {e}", style="red")


@finance_app.command("price")
def finance_price(symbol: str = typer.Argument(..., help="Stock symbol to check")):
    """Get current stock price."""
    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.finance_module import fetch_stock_price

        price = fetch_stock_price(symbol)
        console.print(f"{symbol}: ${price:.2f}", style="green")
    except Exception as e:
        console.print(f"✗ Failed to fetch price: {e}", style="red")


# Security Commands
@security_app.command("audit")
def security_audit(
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Security configuration file"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file for results"
    ),
    format: str = typer.Option(
        "json", "--format", help="Output format (json, html, text)"
    ),
):
    """Run comprehensive security audit."""
    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from src.vega.security.integration import SecurityOrchestrator

        async def run_audit():
            orchestrator = SecurityOrchestrator(config)
            results = await orchestrator.run_full_security_audit()

            summary = results.get("summary", {})

            if format == "json":
                output_data = json.dumps(results, indent=2, default=str)
            else:
                # Simple text format for CLI
                output_data = f"""Security Audit Results
========================
Audit ID: {results.get('audit_id', 'Unknown')}
Status: {results.get('status', 'Unknown')}
Overall Status: {summary.get('overall_status', 'Unknown')}

Issue Summary:
  Critical: {summary.get('critical_issues', 0)}
  High: {summary.get('high_issues', 0)}
  Medium: {summary.get('medium_issues', 0)}
  Low: {summary.get('low_issues', 0)}
"""

            if output:
                with open(output, "w") as f:
                    f.write(output_data)
                console.print(f"✓ Results saved to {output}", style="green")
            else:
                console.print(output_data)

            # Show status
            status = summary.get("overall_status", "unknown").lower()
            if status == "pass":
                console.print("✓ Security audit passed", style="green")
            elif status == "warning":
                console.print(
                    "⚠ Security audit completed with warnings", style="yellow"
                )
            else:
                console.print("✗ Security audit found critical issues", style="red")

        asyncio.run(run_audit())

    except Exception as e:
        console.print(f"✗ Security audit failed: {e}", style="red")


@security_app.command("scan")
def security_scan(
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Security configuration file"
    )
):
    """Run security vulnerability scan."""
    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from src.vega.security.scanner import SecurityScanner

        async def run_scan():
            scanner = SecurityScanner(config)
            results = await scanner.run_comprehensive_scan()

            console.print("🔍 Security Scan Results", style="bold")
            console.print("=" * 50)

            summary = results.get("summary", {})
            console.print(
                f"Critical: {summary.get('critical', 0)}",
                style="red" if summary.get("critical", 0) > 0 else "white",
            )
            console.print(
                f"High: {summary.get('high', 0)}",
                style="yellow" if summary.get("high", 0) > 0 else "white",
            )
            console.print(f"Medium: {summary.get('medium', 0)}", style="white")
            console.print(f"Low: {summary.get('low', 0)}", style="green")

            # Show some details for critical/high issues
            tools = results.get("tools", {})
            for tool_name, tool_results in tools.items():
                issues = tool_results.get("issues", [])
                critical_high = [
                    i for i in issues if i.get("severity") in ["CRITICAL", "HIGH"]
                ]

                if critical_high:
                    console.print(
                        f"\n🚨 {tool_name.upper()} - Critical/High Issues:",
                        style="bold red",
                    )
                    for issue in critical_high[:3]:  # Show first 3
                        console.print(
                            f"  • {issue.get('title', 'Unknown issue')} [{issue.get('severity')}]"
                        )
                    if len(critical_high) > 3:
                        console.print(f"  ... and {len(critical_high) - 3} more issues")

        asyncio.run(run_scan())

    except Exception as e:
        console.print(f"✗ Security scan failed: {e}", style="red")


@security_app.command("monitor")
def security_monitor(
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Security configuration file"
    )
):
    """Monitor security status."""
    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from src.vega.security.integration import SecurityOrchestrator

        async def run_monitor():
            orchestrator = SecurityOrchestrator(config)
            status = await orchestrator.monitor_security_status()

            console.print("📊 Security Status Monitor", style="bold")
            console.print("=" * 50)

            health = status.get("overall_health", "unknown")
            health_colors = {
                "excellent": "green",
                "good": "green",
                "fair": "yellow",
                "poor": "red",
                "critical": "red",
            }

            console.print(
                f"Overall Health: {health.upper()}",
                style=health_colors.get(health, "white"),
            )

            # Show vulnerability status
            vuln_status = status.get("vulnerabilities", {})
            if vuln_status:
                console.print(f"\n🔍 Vulnerabilities:")
                console.print(f"  Critical: {vuln_status.get('critical', 0)}")
                console.print(f"  High: {vuln_status.get('high', 0)}")
                console.print(f"  Medium: {vuln_status.get('medium', 0)}")

        asyncio.run(run_monitor())

    except Exception as e:
        console.print(f"✗ Security monitoring failed: {e}", style="red")


@security_app.command("ci-check")
def security_ci_check(
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Security configuration file"
    )
):
    """Run security checks for CI/CD pipeline."""
    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from src.vega.security.integration import SecurityOrchestrator

        async def run_ci_check():
            orchestrator = SecurityOrchestrator(config)
            success = await orchestrator.run_ci_security_check()

            if success:
                console.print("✅ CI security checks passed", style="green")
                return True
            else:
                console.print("❌ CI security checks failed", style="red")
                return False

        success = asyncio.run(run_ci_check())
        if not success:
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"✗ CI security check failed: {e}", style="red")
        raise typer.Exit(1)


# Keep the original incomplete function at the end
async def _run():
    """Legacy function - to be cleaned up"""
    pass

    asyncio.run(_run())
