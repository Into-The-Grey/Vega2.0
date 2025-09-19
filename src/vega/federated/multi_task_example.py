"""
Multi-Task Federated Learning Example

This example demonstrates how to set up and run multi-task federated learning
where participants train on multiple tasks simultaneously (e.g., image classification
and sentiment analysis) while sharing knowledge through shared representations.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import asyncio
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Optional visualization dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print(
        "Warning: matplotlib and/or seaborn not available. Visualization will be disabled."
    )

from .multi_task import (
    TaskDefinition,
    MultiTaskModelConfig,
    MultiTaskParticipant,
    MultiTaskCoordinator,
    create_synthetic_multitask_data,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MultiTaskFederatedExample:
    """Example implementation of multi-task federated learning."""

    def __init__(self, num_participants: int = 5, input_dim: int = 20):
        self.num_participants = num_participants
        self.input_dim = input_dim

        # Define multiple tasks
        self.tasks = [
            TaskDefinition(
                task_id="sentiment_analysis",
                task_type="classification",
                input_dim=input_dim,
                output_dim=3,  # positive, negative, neutral
                loss_function="cross_entropy",
                metric="accuracy",
                task_weight=1.0,
            ),
            TaskDefinition(
                task_id="spam_detection",
                task_type="classification",
                input_dim=input_dim,
                output_dim=2,  # spam, not spam
                loss_function="cross_entropy",
                metric="accuracy",
                task_weight=1.0,
            ),
            TaskDefinition(
                task_id="rating_prediction",
                task_type="regression",
                input_dim=input_dim,
                output_dim=1,  # rating score
                loss_function="mse",
                metric="mse",
                task_weight=0.8,
            ),
        ]

        # Model configuration
        self.model_config = MultiTaskModelConfig(
            shared_layers=[64, 32],
            task_specific_layers={
                "sentiment_analysis": [16],
                "spam_detection": [16],
                "rating_prediction": [16],
            },
            activation="relu",
            dropout_rate=0.1,
            use_batch_norm=True,
        )

        # Initialize coordinator
        self.coordinator = MultiTaskCoordinator(
            tasks=self.tasks,
            model_config=self.model_config,
            aggregation_strategy="fedavg",
        )

        # Participants will be created later
        self.participants = []

        # Results tracking
        self.results = {"training_history": [], "evaluation_history": []}

    def create_heterogeneous_data(self) -> Dict[str, Dict[str, DataLoader]]:
        """Create heterogeneous data distributions across participants."""
        logger.info("Creating heterogeneous multi-task data distributions...")

        participant_data = {}

        for i in range(self.num_participants):
            participant_id = f"participant_{i}"
            participant_data[participant_id] = {}

            # Each participant has different task combinations
            if i < 2:
                # Participants 0-1: All tasks
                available_tasks = [
                    "sentiment_analysis",
                    "spam_detection",
                    "rating_prediction",
                ]
            elif i < 4:
                # Participants 2-3: Text tasks only
                available_tasks = ["sentiment_analysis", "spam_detection"]
            else:
                # Participants 4+: Sentiment and rating only
                available_tasks = ["sentiment_analysis", "rating_prediction"]

            # Generate data for available tasks
            for task_id in available_tasks:
                task = next(task for task in self.tasks if task.task_id == task_id)

                # Create task-specific data with some correlation
                num_samples = np.random.randint(800, 1200)

                if task_id == "sentiment_analysis":
                    # Text features for sentiment
                    X = (
                        torch.randn(num_samples, self.input_dim) + i * 0.1
                    )  # Small participant bias
                    y = torch.randint(0, 3, (num_samples,))

                elif task_id == "spam_detection":
                    # Text features for spam detection (correlated with sentiment)
                    X = torch.randn(num_samples, self.input_dim) + i * 0.05
                    y = torch.randint(0, 2, (num_samples,))

                elif task_id == "rating_prediction":
                    # Continuous rating prediction
                    X = torch.randn(num_samples, self.input_dim) + i * 0.1
                    y = torch.randn(num_samples, 1) * 2 + 3  # Ratings around 3

                # Create data loader
                dataset = TensorDataset(X, y)
                data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
                participant_data[participant_id][task_id] = data_loader

        return participant_data

    def create_participants(self, participant_data: Dict[str, Dict[str, DataLoader]]):
        """Create multi-task federated participants."""
        logger.info("Creating multi-task federated participants...")

        for participant_id, data_loaders in participant_data.items():
            # Determine which tasks this participant has
            participant_tasks = [
                task for task in self.tasks if task.task_id in data_loaders
            ]

            # Create participant
            participant = MultiTaskParticipant(
                participant_id=participant_id,
                tasks=participant_tasks,
                model_config=self.model_config,
                data_loaders=data_loaders,
            )

            self.participants.append(participant)
            self.coordinator.register_participant(participant)

            logger.info(
                f"Created {participant_id} with tasks: {list(data_loaders.keys())}"
            )

    async def run_federated_learning(
        self, num_rounds: int = 10, local_epochs: int = 2
    ) -> Dict[str, List]:
        """Run multi-task federated learning simulation."""
        logger.info(
            f"Starting multi-task federated learning for {num_rounds} rounds..."
        )

        training_history = []
        evaluation_history = []

        for round_num in range(num_rounds):
            logger.info(f"\n=== ROUND {round_num + 1}/{num_rounds} ===")

            # Run federated round
            round_result = await self.coordinator.run_federated_round(
                local_epochs=local_epochs
            )
            training_history.append(round_result)

            # Evaluate global model
            evaluation_result = await self.coordinator.evaluate_global_model()
            evaluation_history.append(evaluation_result)

            # Log progress
            self._log_round_progress(round_num + 1, evaluation_result)

        self.results["training_history"] = training_history
        self.results["evaluation_history"] = evaluation_history

        logger.info("Multi-task federated learning completed!")
        return self.results

    def _log_round_progress(self, round_num: int, evaluation_result: Dict):
        """Log progress for current round."""
        aggregated_metrics = evaluation_result["aggregated_metrics"]

        logger.info(f"Round {round_num} Results:")
        for task_id, metrics in aggregated_metrics.items():
            if "accuracy" in metrics:
                acc_mean = metrics["accuracy"]["mean"]
                logger.info(f"  {task_id}: Accuracy = {acc_mean:.4f}")
            elif "loss" in metrics:
                loss_mean = metrics["loss"]["mean"]
                logger.info(f"  {task_id}: Loss = {loss_mean:.4f}")

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze and visualize federated learning results."""
        logger.info("Analyzing multi-task federated learning results...")

        analysis = {
            "convergence_analysis": self._analyze_convergence(),
            "task_performance": self._analyze_task_performance(),
            "participant_contribution": self._analyze_participant_contribution(),
        }

        return analysis

    def _analyze_convergence(self) -> Dict[str, List[float]]:
        """Analyze convergence for each task."""
        convergence = {}

        for task in self.tasks:
            task_id = task.task_id
            metric_name = "accuracy" if task.task_type == "classification" else "loss"

            task_metrics = []
            for eval_result in self.results["evaluation_history"]:
                if task_id in eval_result["aggregated_metrics"]:
                    metrics = eval_result["aggregated_metrics"][task_id]
                    if metric_name in metrics:
                        task_metrics.append(metrics[metric_name]["mean"])

            convergence[f"{task_id}_{metric_name}"] = task_metrics

        return convergence

    def _analyze_task_performance(self) -> Dict[str, Dict[str, float]]:
        """Analyze final performance for each task."""
        if not self.results["evaluation_history"]:
            return {}

        final_evaluation = self.results["evaluation_history"][-1]
        task_performance = {}

        for task_id, metrics in final_evaluation["aggregated_metrics"].items():
            task_performance[task_id] = {}
            for metric_name, metric_stats in metrics.items():
                task_performance[task_id][metric_name] = metric_stats["mean"]

        return task_performance

    def _analyze_participant_contribution(self) -> Dict[str, Dict[str, float]]:
        """Analyze each participant's contribution to different tasks."""
        if not self.results["evaluation_history"]:
            return {}

        final_evaluation = self.results["evaluation_history"][-1]
        participant_metrics = final_evaluation["participant_metrics"]

        contribution = {}
        for participant_id, tasks_metrics in participant_metrics.items():
            contribution[participant_id] = {}
            for task_id, metrics in tasks_metrics.items():
                # Use accuracy for classification, negative loss for regression
                if "accuracy" in metrics:
                    contribution[participant_id][task_id] = metrics["accuracy"]
                elif "loss" in metrics:
                    contribution[participant_id][task_id] = -metrics[
                        "loss"
                    ]  # Negative loss

        return contribution

    def visualize_results(self, save_path: Optional[str] = None):
        """Create visualizations of the federated learning results."""
        if not HAS_PLOTTING:
            logger.warning("Matplotlib/seaborn not available. Skipping visualization.")
            return

        logger.info("Creating visualizations...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Multi-Task Federated Learning Results", fontsize=16)

        # 1. Convergence plot
        ax1 = axes[0, 0]
        convergence = self._analyze_convergence()

        for metric_name, values in convergence.items():
            ax1.plot(
                range(1, len(values) + 1),
                values,
                marker="o",
                label=metric_name,
                linewidth=2,
            )

        ax1.set_xlabel("Federated Round")
        ax1.set_ylabel("Metric Value")
        ax1.set_title("Task Convergence Over Rounds")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Final task performance
        ax2 = axes[0, 1]
        task_performance = self._analyze_task_performance()

        tasks = list(task_performance.keys())
        accuracies = [task_performance[task].get("accuracy", 0) for task in tasks]
        losses = [task_performance[task].get("loss", 0) for task in tasks]

        x_pos = np.arange(len(tasks))
        bars = ax2.bar(x_pos, accuracies, alpha=0.7)
        ax2.set_xlabel("Tasks")
        ax2.set_ylabel("Final Accuracy")
        ax2.set_title("Final Task Performance")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(tasks, rotation=45)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                )

        # 3. Participant contribution heatmap
        ax3 = axes[1, 0]
        contribution = self._analyze_participant_contribution()

        if contribution:
            # Create matrix for heatmap
            participants = list(contribution.keys())
            all_tasks = set()
            for tasks in contribution.values():
                all_tasks.update(tasks.keys())
            all_tasks = sorted(list(all_tasks))

            matrix = np.zeros((len(participants), len(all_tasks)))
            for i, participant in enumerate(participants):
                for j, task in enumerate(all_tasks):
                    if task in contribution[participant]:
                        matrix[i, j] = contribution[participant][task]

            sns.heatmap(
                matrix,
                annot=True,
                fmt=".3f",
                xticklabels=all_tasks,
                yticklabels=participants,
                ax=ax3,
                cmap="YlOrRd",
            )
            ax3.set_title("Participant Performance by Task")
            ax3.set_xlabel("Tasks")
            ax3.set_ylabel("Participants")

        # 4. Training loss over rounds
        ax4 = axes[1, 1]

        round_losses = []
        for round_result in self.results["training_history"]:
            round_loss = 0.0
            count = 0
            for training_result in round_result["training_results"]:
                round_loss += training_result["total_loss"]
                count += 1
            if count > 0:
                round_losses.append(round_loss / count)

        if round_losses:
            ax4.plot(
                range(1, len(round_losses) + 1),
                round_losses,
                marker="s",
                color="red",
                linewidth=2,
            )
            ax4.set_xlabel("Federated Round")
            ax4.set_ylabel("Average Training Loss")
            ax4.set_title("Training Loss Convergence")
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved visualization to {save_path}")

        plt.show()

    def save_results(self, filepath: str):
        """Save complete results to file."""
        self.coordinator.save_session_state(filepath)
        logger.info(f"Saved complete results to {filepath}")


async def run_multi_task_example():
    """Run a complete multi-task federated learning example."""
    print("=" * 60)
    print("MULTI-TASK FEDERATED LEARNING EXAMPLE")
    print("=" * 60)

    # Create example
    example = MultiTaskFederatedExample(num_participants=5, input_dim=20)

    # Create heterogeneous data
    participant_data = example.create_heterogeneous_data()

    # Create participants
    example.create_participants(participant_data)

    # Run federated learning
    results = await example.run_federated_learning(num_rounds=15, local_epochs=3)

    # Analyze results
    analysis = example.analyze_results()

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    print("\nTask Performance:")
    for task_id, performance in analysis["task_performance"].items():
        print(f"  {task_id}:")
        for metric, value in performance.items():
            print(f"    {metric}: {value:.4f}")

    print("\nParticipant Contribution:")
    for participant_id, tasks in analysis["participant_contribution"].items():
        print(f"  {participant_id}: {list(tasks.keys())}")

    # Visualize results
    example.visualize_results("multi_task_federated_results.png")

    # Save complete results
    example.save_results("multi_task_federated_session.json")

    return example, results, analysis


class CrossDomainExample:
    """Example of cross-domain multi-task federated learning."""

    def __init__(self):
        # Define tasks from different domains
        self.tasks = [
            TaskDefinition(
                task_id="image_classification",
                task_type="classification",
                input_dim=784,  # 28x28 flattened images
                output_dim=10,  # 10 classes
                loss_function="cross_entropy",
                metric="accuracy",
                task_weight=1.0,
            ),
            TaskDefinition(
                task_id="text_sentiment",
                task_type="classification",
                input_dim=300,  # text embeddings
                output_dim=2,  # positive/negative
                loss_function="cross_entropy",
                metric="accuracy",
                task_weight=1.0,
            ),
            TaskDefinition(
                task_id="sensor_prediction",
                task_type="regression",
                input_dim=50,  # sensor readings
                output_dim=1,  # predicted value
                loss_function="mse",
                metric="mse",
                task_weight=0.8,
            ),
        ]

    def create_domain_specific_participants(self) -> List[MultiTaskParticipant]:
        """Create participants specializing in different domains."""
        participants = []

        # Image specialist
        image_participant = self._create_image_participant()
        participants.append(image_participant)

        # Text specialist
        text_participant = self._create_text_participant()
        participants.append(text_participant)

        # Sensor specialist
        sensor_participant = self._create_sensor_participant()
        participants.append(sensor_participant)

        # Multi-domain participant
        multi_participant = self._create_multi_domain_participant()
        participants.append(multi_participant)

        return participants

    def _create_image_participant(self) -> MultiTaskParticipant:
        """Create participant specializing in image tasks."""
        image_task = [
            task for task in self.tasks if task.task_id == "image_classification"
        ][0]

        # Generate synthetic image data
        num_samples = 1000
        X = torch.randn(num_samples, 784)  # Simulated flattened images
        y = torch.randint(0, 10, (num_samples,))

        dataset = TensorDataset(X, y)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        model_config = MultiTaskModelConfig(
            shared_layers=[128, 64],
            task_specific_layers={"image_classification": [32]},
            activation="relu",
            dropout_rate=0.2,
        )

        return MultiTaskParticipant(
            participant_id="image_specialist",
            tasks=[image_task],
            model_config=model_config,
            data_loaders={"image_classification": data_loader},
        )

    def _create_text_participant(self) -> MultiTaskParticipant:
        """Create participant specializing in text tasks."""
        text_task = [task for task in self.tasks if task.task_id == "text_sentiment"][0]

        # Generate synthetic text embeddings
        num_samples = 800
        X = torch.randn(num_samples, 300)  # Simulated text embeddings
        y = torch.randint(0, 2, (num_samples,))

        dataset = TensorDataset(X, y)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        model_config = MultiTaskModelConfig(
            shared_layers=[128, 64],
            task_specific_layers={"text_sentiment": [32]},
            activation="relu",
            dropout_rate=0.1,
        )

        return MultiTaskParticipant(
            participant_id="text_specialist",
            tasks=[text_task],
            model_config=model_config,
            data_loaders={"text_sentiment": data_loader},
        )

    def _create_sensor_participant(self) -> MultiTaskParticipant:
        """Create participant specializing in sensor tasks."""
        sensor_task = [
            task for task in self.tasks if task.task_id == "sensor_prediction"
        ][0]

        # Generate synthetic sensor data
        num_samples = 1200
        X = torch.randn(num_samples, 50)  # Simulated sensor readings
        y = torch.randn(num_samples, 1) * 5 + 10  # Predicted values

        dataset = TensorDataset(X, y)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        model_config = MultiTaskModelConfig(
            shared_layers=[64, 32],
            task_specific_layers={"sensor_prediction": [16]},
            activation="relu",
            dropout_rate=0.1,
        )

        return MultiTaskParticipant(
            participant_id="sensor_specialist",
            tasks=[sensor_task],
            model_config=model_config,
            data_loaders={"sensor_prediction": data_loader},
        )

    def _create_multi_domain_participant(self) -> MultiTaskParticipant:
        """Create participant with data from multiple domains."""
        # This participant has smaller amounts of data from all domains
        data_loaders = {}

        # Image data
        X_img = torch.randn(300, 784)
        y_img = torch.randint(0, 10, (300,))
        data_loaders["image_classification"] = DataLoader(
            TensorDataset(X_img, y_img), batch_size=16, shuffle=True
        )

        # Text data
        X_text = torch.randn(250, 300)
        y_text = torch.randint(0, 2, (250,))
        data_loaders["text_sentiment"] = DataLoader(
            TensorDataset(X_text, y_text), batch_size=16, shuffle=True
        )

        # Sensor data
        X_sensor = torch.randn(400, 50)
        y_sensor = torch.randn(400, 1) * 5 + 10
        data_loaders["sensor_prediction"] = DataLoader(
            TensorDataset(X_sensor, y_sensor), batch_size=16, shuffle=True
        )

        model_config = MultiTaskModelConfig(
            shared_layers=[128, 64, 32],
            task_specific_layers={
                "image_classification": [16],
                "text_sentiment": [16],
                "sensor_prediction": [16],
            },
            activation="relu",
            dropout_rate=0.15,
        )

        return MultiTaskParticipant(
            participant_id="multi_domain",
            tasks=self.tasks,
            model_config=model_config,
            data_loaders=data_loaders,
        )


async def run_cross_domain_example():
    """Run cross-domain multi-task federated learning example."""
    print("\n" + "=" * 60)
    print("CROSS-DOMAIN MULTI-TASK FEDERATED LEARNING")
    print("=" * 60)

    # Create cross-domain example
    cross_domain = CrossDomainExample()

    # Create domain-specific participants
    participants = cross_domain.create_domain_specific_participants()

    # Create coordinator
    coordinator = MultiTaskCoordinator(
        tasks=cross_domain.tasks,
        model_config=MultiTaskModelConfig(shared_layers=[128, 64]),
        aggregation_strategy="fedavg",
    )

    # Register participants
    for participant in participants:
        coordinator.register_participant(participant)

    print(
        f"\nCreated {len(participants)} participants with different domain specializations"
    )

    # Run federated learning
    results = []
    for round_num in range(10):
        round_result = await coordinator.run_federated_round(local_epochs=2)
        eval_result = await coordinator.evaluate_global_model()
        results.append((round_result, eval_result))

        print(f"Round {round_num + 1}: Cross-domain knowledge sharing in progress...")

    print("\nCross-domain federated learning completed!")
    return coordinator, results


if __name__ == "__main__":
    # Run basic multi-task example
    asyncio.run(run_multi_task_example())

    # Run cross-domain example
    asyncio.run(run_cross_domain_example())
