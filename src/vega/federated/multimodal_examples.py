"""
Multi-Modal Federated Learning Examples

This module provides comprehensive examples demonstrating various multi-modal
federated learning scenarios including vision+text, audio+sensor data,
and cross-modal learning workflows.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from .multimodal import (
    DataModality,
    ModalityConfig,
    MultiModalSample,
    MultiModalBatch,
    VisionDataHandler,
    TextDataHandler,
    AudioDataHandler,
    SensorDataHandler,
    MultiModalDataManager,
)
from .federated_transformer import (
    FederatedMultiModalTransformer,
    FederatedTransformerConfig,
)
from .cross_modal import CrossModalLearningFramework, FusionStrategy, AlignmentStrategy
from .multimodal_aggregation import (
    MultiModalAggregationCoordinator,
    MultiModalAggregationConfig,
    AggregationStrategy,
    ParticipantModalityInfo,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalFederatedLearningDemo:
    """Comprehensive demo of multi-modal federated learning capabilities."""

    def __init__(
        self, output_dir: str = "./multimodal_demo_results", device: str = "cpu"
    ):
        """Initialize the demo environment."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)

        # Initialize components
        self.data_manager = None
        self.transformer = None
        self.cross_modal_framework = None
        self.aggregation_coordinator = None

        # Demo data
        self.demo_samples = []
        self.participant_data = {}

        logger.info(f"Initialized Multi-Modal Federated Learning Demo")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: {self.device}")

    def setup_demo_data(self):
        """Create synthetic demo data for different modalities."""
        logger.info("Setting up demo data...")

        # Create modality configurations
        vision_config = ModalityConfig(
            modality=DataModality.VISION,
            input_dim=224 * 224 * 3,  # RGB image
            embedding_dim=512,
            preprocessing_params={
                "image_size": (224, 224),
                "normalize": True,
                "augment": True,
            },
        )

        text_config = ModalityConfig(
            modality=DataModality.TEXT,
            input_dim=768,  # BERT-like embeddings
            embedding_dim=512,
            preprocessing_params={
                "max_length": 128,
                "tokenizer": "bert-base-uncased",
                "lowercase": True,
            },
        )

        audio_config = ModalityConfig(
            modality=DataModality.AUDIO,
            input_dim=1024,  # Audio features
            embedding_dim=512,
            preprocessing_params={
                "sample_rate": 16000,
                "n_mfcc": 13,
                "hop_length": 512,
            },
        )

        sensor_config = ModalityConfig(
            modality=DataModality.SENSOR,
            input_dim=128,  # Sensor readings
            embedding_dim=512,
            preprocessing_params={
                "normalize": True,
                "window_size": 100,
                "overlap": 0.5,
            },
        )

        # Initialize data manager
        modality_configs = [vision_config, text_config, audio_config, sensor_config]
        self.data_manager = MultiModalDataManager(modality_configs)

        # Generate synthetic samples
        num_samples = 100
        for i in range(num_samples):
            # Create multi-modal sample
            vision_data = torch.randn(3, 224, 224)  # RGB image
            text_data = torch.randn(768)  # Text embedding
            audio_data = torch.randn(1024)  # Audio features
            sensor_data = torch.randn(128)  # Sensor readings

            sample = MultiModalSample(
                sample_id=f"sample_{i}",
                modality_data={
                    DataModality.VISION: vision_data,
                    DataModality.TEXT: text_data,
                    DataModality.AUDIO: audio_data,
                    DataModality.SENSOR: sensor_data,
                },
                labels={
                    "classification": torch.randint(0, 10, (1,)).item(),
                    "regression": torch.randn(1).item(),
                },
                metadata={
                    "source": f"synthetic_generator_{i % 5}",
                    "timestamp": datetime.now().isoformat(),
                    "quality_score": np.random.uniform(0.7, 1.0),
                },
            )

            self.demo_samples.append(sample)

        logger.info(f"Created {num_samples} synthetic multi-modal samples")

    def setup_federated_participants(self, num_participants: int = 5):
        """Create federated learning participants with different modality distributions."""
        logger.info(f"Setting up {num_participants} federated participants...")

        # Define modality distributions for different participant types
        participant_types = [
            {"primary": DataModality.VISION, "secondary": [DataModality.TEXT]},
            {"primary": DataModality.TEXT, "secondary": [DataModality.AUDIO]},
            {"primary": DataModality.AUDIO, "secondary": [DataModality.SENSOR]},
            {"primary": DataModality.SENSOR, "secondary": [DataModality.VISION]},
            {
                "primary": DataModality.VISION,
                "secondary": [DataModality.TEXT, DataModality.AUDIO],
            },
        ]

        samples_per_participant = len(self.demo_samples) // num_participants

        for i in range(num_participants):
            participant_id = f"participant_{i}"
            participant_type = participant_types[i % len(participant_types)]

            # Assign samples to participant
            start_idx = i * samples_per_participant
            end_idx = (
                (i + 1) * samples_per_participant
                if i < num_participants - 1
                else len(self.demo_samples)
            )
            participant_samples = self.demo_samples[start_idx:end_idx]

            # Filter samples by modality availability
            available_modalities = [participant_type["primary"]] + participant_type[
                "secondary"
            ]
            filtered_samples = []

            for sample in participant_samples:
                # Keep only available modalities
                filtered_modality_data = {
                    modality: data
                    for modality, data in sample.modality_data.items()
                    if modality in available_modalities
                }

                if filtered_modality_data:  # At least one modality available
                    filtered_sample = MultiModalSample(
                        sample_id=sample.sample_id,
                        modality_data=filtered_modality_data,
                        labels=sample.labels,
                        metadata=sample.metadata,
                    )
                    filtered_samples.append(filtered_sample)

            # Create participant info
            participant_info = ParticipantModalityInfo(
                participant_id=participant_id,
                modality=participant_type["primary"],
                data_size=len(filtered_samples),
                data_quality_score=np.random.uniform(0.7, 0.95),
                modality_coverage=len(available_modalities) / len(DataModality),
                local_performance={
                    "accuracy": np.random.uniform(0.7, 0.9),
                    "loss": np.random.uniform(0.1, 0.5),
                },
            )

            self.participant_data[participant_id] = {
                "samples": filtered_samples,
                "info": participant_info,
                "modalities": available_modalities,
            }

        logger.info(
            f"Created {len(self.participant_data)} participants with modality distributions"
        )

        # Log participant statistics
        for pid, data in self.participant_data.items():
            modalities = [m.value for m in data["modalities"]]
            logger.info(
                f"  {pid}: {data['info'].data_size} samples, modalities: {modalities}"
            )

    def setup_federated_models(self):
        """Initialize federated transformer and cross-modal learning framework."""
        logger.info("Setting up federated models...")

        # Transformer configuration
        transformer_config = FederatedTransformerConfig(
            d_model=512,
            n_heads=8,
            n_layers=6,
            d_ff=2048,
            dropout=0.1,
            max_seq_length=128,
            use_compression=True,
            compression_ratio=0.5,
            parameter_sharing_strategy="adapter",
        )

        # Initialize federated transformer
        modality_configs = self.data_manager.modality_configs
        self.transformer = FederatedMultiModalTransformer(
            modality_configs=modality_configs, config=transformer_config
        ).to(self.device)

        # Initialize cross-modal learning framework
        self.cross_modal_framework = CrossModalLearningFramework(
            modality_configs=modality_configs,
            fusion_strategy=FusionStrategy.ATTENTION_FUSION,
            alignment_strategy=AlignmentStrategy.CONTRASTIVE_ALIGNMENT,
            hidden_dim=512,
        ).to(self.device)

        logger.info("Federated models initialized successfully")

    def setup_aggregation_strategy(
        self, strategy: AggregationStrategy = AggregationStrategy.ADAPTIVE_MODALITY
    ):
        """Initialize aggregation coordinator."""
        logger.info(f"Setting up aggregation strategy: {strategy.value}")

        aggregation_config = MultiModalAggregationConfig(
            strategy=strategy,
            use_data_size_weighting=True,
            use_quality_weighting=True,
            use_performance_weighting=True,
            quality_threshold=0.6,
            adaptation_rate=0.1,
            momentum=0.9,
        )

        self.aggregation_coordinator = MultiModalAggregationCoordinator(
            aggregation_config
        )

        logger.info("Aggregation strategy configured")

    def run_vision_text_demo(self):
        """Demonstrate vision + text multi-modal federated learning."""
        logger.info("Running Vision + Text Demo")
        logger.info("=" * 50)

        # Filter participants with vision and text modalities
        vision_text_participants = {
            pid: data
            for pid, data in self.participant_data.items()
            if DataModality.VISION in data["modalities"]
            and DataModality.TEXT in data["modalities"]
        }

        if not vision_text_participants:
            logger.warning("No participants with both vision and text modalities")
            return

        # Simulate federated training round
        participant_updates = {}
        participant_info = {}

        for pid, data in vision_text_participants.items():
            # Simulate local training (create dummy parameter updates)
            updates = {}
            for name, param in self.transformer.named_parameters():
                if "vision" in name or "text" in name or "cross_modal" in name:
                    # Simulate gradient update
                    update = torch.randn_like(param) * 0.01
                    updates[name] = update

            participant_updates[pid] = updates
            participant_info[pid] = data["info"]

        # Perform aggregation
        aggregated_params = self.aggregation_coordinator.aggregate_round(
            participant_updates=participant_updates,
            participant_info=participant_info,
            round_number=1,
        )

        logger.info(
            f"Vision + Text demo completed with {len(vision_text_participants)} participants"
        )
        logger.info(f"Aggregated {len(aggregated_params)} parameters")

        # Save results
        demo_results = {
            "demo_type": "vision_text",
            "participants": list(vision_text_participants.keys()),
            "num_parameters": len(aggregated_params),
            "aggregation_stats": self.aggregation_coordinator.get_aggregation_stats(),
        }

        results_file = self.output_dir / "vision_text_demo_results.json"
        with open(results_file, "w") as f:
            json.dump(demo_results, f, indent=2)

        logger.info(f"Results saved to {results_file}")

    def run_audio_sensor_demo(self):
        """Demonstrate audio + sensor multi-modal federated learning."""
        logger.info("Running Audio + Sensor Demo")
        logger.info("=" * 50)

        # Filter participants with audio and sensor modalities
        audio_sensor_participants = {
            pid: data
            for pid, data in self.participant_data.items()
            if DataModality.AUDIO in data["modalities"]
            and DataModality.SENSOR in data["modalities"]
        }

        if not audio_sensor_participants:
            logger.warning("No participants with both audio and sensor modalities")
            return

        # Test cross-modal learning
        sample_batch = self._create_sample_batch(
            audio_sensor_participants, [DataModality.AUDIO, DataModality.SENSOR]
        )

        if sample_batch:
            # Forward pass through cross-modal framework
            with torch.no_grad():
                cross_modal_output = self.cross_modal_framework(sample_batch)
                fusion_output = cross_modal_output["fused_representation"]
                alignment_loss = cross_modal_output["alignment_loss"]

            logger.info(f"Cross-modal fusion output shape: {fusion_output.shape}")
            logger.info(f"Alignment loss: {alignment_loss.item():.4f}")

        # Simulate aggregation
        participant_updates = {}
        participant_info = {}

        for pid, data in audio_sensor_participants.items():
            updates = {}
            for name, param in self.cross_modal_framework.named_parameters():
                if "audio" in name or "sensor" in name:
                    update = torch.randn_like(param) * 0.01
                    updates[name] = update

            participant_updates[pid] = updates
            participant_info[pid] = data["info"]

        aggregated_params = self.aggregation_coordinator.aggregate_round(
            participant_updates=participant_updates,
            participant_info=participant_info,
            round_number=2,
        )

        logger.info(
            f"Audio + Sensor demo completed with {len(audio_sensor_participants)} participants"
        )

        # Save results
        demo_results = {
            "demo_type": "audio_sensor",
            "participants": list(audio_sensor_participants.keys()),
            "fusion_output_shape": list(fusion_output.shape) if sample_batch else None,
            "alignment_loss": alignment_loss.item() if sample_batch else None,
            "num_parameters": len(aggregated_params),
        }

        results_file = self.output_dir / "audio_sensor_demo_results.json"
        with open(results_file, "w") as f:
            json.dump(demo_results, f, indent=2)

        logger.info(f"Results saved to {results_file}")

    def run_cross_modal_learning_demo(self):
        """Demonstrate cross-modal learning capabilities."""
        logger.info("Running Cross-Modal Learning Demo")
        logger.info("=" * 50)

        # Test different fusion strategies
        fusion_strategies = [
            FusionStrategy.EARLY_FUSION,
            FusionStrategy.LATE_FUSION,
            FusionStrategy.ATTENTION_FUSION,
            FusionStrategy.CONTRASTIVE_FUSION,
        ]

        results = {}

        for strategy in fusion_strategies:
            logger.info(f"Testing fusion strategy: {strategy.value}")

            # Reconfigure cross-modal framework
            modality_configs = self.data_manager.modality_configs
            framework = CrossModalLearningFramework(
                modality_configs=modality_configs,
                fusion_strategy=strategy,
                alignment_strategy=AlignmentStrategy.CONTRASTIVE_ALIGNMENT,
                hidden_dim=512,
            ).to(self.device)

            # Create test batch with all modalities
            test_batch = self._create_sample_batch(
                self.participant_data,
                [
                    DataModality.VISION,
                    DataModality.TEXT,
                    DataModality.AUDIO,
                    DataModality.SENSOR,
                ],
            )

            if test_batch:
                with torch.no_grad():
                    output = framework(test_batch)
                    fusion_output = output["fused_representation"]

                results[strategy.value] = {
                    "output_shape": list(fusion_output.shape),
                    "output_mean": fusion_output.mean().item(),
                    "output_std": fusion_output.std().item(),
                }

                logger.info(f"  Output shape: {fusion_output.shape}")
                logger.info(
                    f"  Output statistics: mean={fusion_output.mean().item():.4f}, std={fusion_output.std().item():.4f}"
                )

        # Save cross-modal learning results
        demo_results = {
            "demo_type": "cross_modal_learning",
            "fusion_strategies": results,
            "test_modalities": ["vision", "text", "audio", "sensor"],
        }

        results_file = self.output_dir / "cross_modal_demo_results.json"
        with open(results_file, "w") as f:
            json.dump(demo_results, f, indent=2)

        logger.info(f"Cross-modal learning results saved to {results_file}")

    def run_aggregation_comparison_demo(self):
        """Compare different aggregation strategies."""
        logger.info("Running Aggregation Strategy Comparison Demo")
        logger.info("=" * 50)

        strategies = [
            AggregationStrategy.FEDAVG_MULTIMODAL,
            AggregationStrategy.MODALITY_WEIGHTED,
            AggregationStrategy.ADAPTIVE_MODALITY,
            AggregationStrategy.QUALITY_AWARE,
            AggregationStrategy.HIERARCHICAL_MULTIMODAL,
        ]

        comparison_results = {}

        # Create consistent participant updates for fair comparison
        participant_updates = {}
        participant_info = {}

        for pid, data in self.participant_data.items():
            updates = {}
            for name, param in self.transformer.named_parameters():
                # Simulate gradient updates with some variation based on participant quality
                quality_factor = data["info"].data_quality_score
                update = torch.randn_like(param) * 0.01 * quality_factor
                updates[name] = update

            participant_updates[pid] = updates
            participant_info[pid] = data["info"]

        # Test each aggregation strategy
        for strategy in strategies:
            logger.info(f"Testing aggregation strategy: {strategy.value}")

            # Create aggregation coordinator
            config = MultiModalAggregationConfig(strategy=strategy)
            coordinator = MultiModalAggregationCoordinator(config)

            # Perform aggregation
            aggregated_params = coordinator.aggregate_round(
                participant_updates=participant_updates,
                participant_info=participant_info,
                round_number=1,
            )

            # Compute aggregation statistics
            param_stats = {}
            for name, param in aggregated_params.items():
                param_stats[name] = {
                    "mean": param.mean().item(),
                    "std": param.std().item(),
                    "norm": param.norm().item(),
                }

            comparison_results[strategy.value] = {
                "num_parameters": len(aggregated_params),
                "aggregation_stats": coordinator.get_aggregation_stats(),
                "parameter_statistics": {
                    "total_norm": sum(stats["norm"] for stats in param_stats.values()),
                    "avg_mean": np.mean(
                        [stats["mean"] for stats in param_stats.values()]
                    ),
                    "avg_std": np.mean(
                        [stats["std"] for stats in param_stats.values()]
                    ),
                },
            }

            logger.info(f"  Parameters aggregated: {len(aggregated_params)}")
            logger.info(
                f"  Total parameter norm: {comparison_results[strategy.value]['parameter_statistics']['total_norm']:.4f}"
            )

        # Save comparison results
        demo_results = {
            "demo_type": "aggregation_comparison",
            "strategies_compared": [s.value for s in strategies],
            "results": comparison_results,
            "participants": list(participant_info.keys()),
        }

        results_file = self.output_dir / "aggregation_comparison_results.json"
        with open(results_file, "w") as f:
            json.dump(demo_results, f, indent=2)

        logger.info(f"Aggregation comparison results saved to {results_file}")

    def _create_sample_batch(
        self,
        participants: Dict[str, Any],
        required_modalities: List[DataModality],
        batch_size: int = 8,
    ) -> Optional[MultiModalBatch]:
        """Create a sample batch from participant data."""
        valid_samples = []

        for pid, data in participants.items():
            for sample in data["samples"]:
                # Check if sample has all required modalities
                if all(
                    modality in sample.modality_data for modality in required_modalities
                ):
                    valid_samples.append(sample)
                    if len(valid_samples) >= batch_size:
                        break
            if len(valid_samples) >= batch_size:
                break

        if len(valid_samples) < 2:
            logger.warning(
                f"Not enough samples with required modalities: {[m.value for m in required_modalities]}"
            )
            return None

        # Create batch
        batch = self.data_manager.create_batch(valid_samples[:batch_size])
        return batch

    def generate_demo_report(self):
        """Generate a comprehensive demo report."""
        logger.info("Generating comprehensive demo report...")

        # Collect all results
        report_data = {
            "demo_info": {
                "timestamp": datetime.now().isoformat(),
                "total_participants": len(self.participant_data),
                "total_samples": len(self.demo_samples),
                "output_directory": str(self.output_dir),
            },
            "participant_statistics": {},
            "aggregation_statistics": (
                self.aggregation_coordinator.get_aggregation_stats()
                if self.aggregation_coordinator
                else {}
            ),
        }

        # Participant statistics
        for pid, data in self.participant_data.items():
            report_data["participant_statistics"][pid] = {
                "modalities": [m.value for m in data["modalities"]],
                "data_size": data["info"].data_size,
                "data_quality": data["info"].data_quality_score,
                "modality_coverage": data["info"].modality_coverage,
                "performance": data["info"].local_performance,
            }

        # Save comprehensive report
        report_file = self.output_dir / "comprehensive_demo_report.json"
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)

        # Generate summary text report
        summary_file = self.output_dir / "demo_summary.txt"
        with open(summary_file, "w") as f:
            f.write("Multi-Modal Federated Learning Demo Summary\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Demo completed at: {report_data['demo_info']['timestamp']}\n")
            f.write(
                f"Total participants: {report_data['demo_info']['total_participants']}\n"
            )
            f.write(f"Total samples: {report_data['demo_info']['total_samples']}\n\n")

            f.write("Participant Distribution by Modality:\n")
            modality_counts = {}
            for pid, stats in report_data["participant_statistics"].items():
                for modality in stats["modalities"]:
                    modality_counts[modality] = modality_counts.get(modality, 0) + 1

            for modality, count in modality_counts.items():
                f.write(f"  {modality}: {count} participants\n")

            f.write(f"\nResults saved to: {self.output_dir}\n")

        logger.info(f"Demo report generated: {report_file}")
        logger.info(f"Demo summary generated: {summary_file}")

    def run_full_demo(self):
        """Run the complete multi-modal federated learning demo."""
        logger.info("Starting Full Multi-Modal Federated Learning Demo")
        logger.info("=" * 60)

        try:
            # Setup
            self.setup_demo_data()
            self.setup_federated_participants()
            self.setup_federated_models()
            self.setup_aggregation_strategy()

            # Run individual demos
            self.run_vision_text_demo()
            self.run_audio_sensor_demo()
            self.run_cross_modal_learning_demo()
            self.run_aggregation_comparison_demo()

            # Generate report
            self.generate_demo_report()

            logger.info("Full demo completed successfully!")
            logger.info(f"All results saved to: {self.output_dir}")

        except Exception as e:
            logger.error(f"Demo failed with error: {e}")
            raise


def main():
    """Main function to run the multi-modal federated learning demo."""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Modal Federated Learning Demo")
    parser.add_argument(
        "--output-dir",
        default="./multimodal_demo_results",
        help="Output directory for demo results",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"], help="Device to run demo on"
    )
    parser.add_argument(
        "--participants", type=int, default=5, help="Number of federated participants"
    )
    parser.add_argument(
        "--demo-type",
        choices=["full", "vision_text", "audio_sensor", "cross_modal", "aggregation"],
        default="full",
        help="Type of demo to run",
    )

    args = parser.parse_args()

    # Create and run demo
    demo = MultiModalFederatedLearningDemo(
        output_dir=args.output_dir, device=args.device
    )

    # Setup common components
    demo.setup_demo_data()
    demo.setup_federated_participants(num_participants=args.participants)
    demo.setup_federated_models()
    demo.setup_aggregation_strategy()

    # Run selected demo
    if args.demo_type == "full":
        demo.run_full_demo()
    elif args.demo_type == "vision_text":
        demo.run_vision_text_demo()
    elif args.demo_type == "audio_sensor":
        demo.run_audio_sensor_demo()
    elif args.demo_type == "cross_modal":
        demo.run_cross_modal_learning_demo()
    elif args.demo_type == "aggregation":
        demo.run_aggregation_comparison_demo()

    print(f"\nDemo completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
