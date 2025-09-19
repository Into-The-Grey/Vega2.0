"""
Comprehensive Test Suite for Multi-Modal Federated Learning Components

This module provides extensive test coverage for all multi-modal federated
learning components including data handling, model architectures, aggregation
strategies, and cross-modal learning capabilities.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import tempfile
import json

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
    ModalityEmbedding,
    CrossModalAttention,
    AdapterLayer,
)
from .cross_modal import (
    CrossModalLearningFramework,
    FusionStrategy,
    AlignmentStrategy,
    AttentionFusion,
    ContrastiveLearning,
    KnowledgeDistillation,
)
from .multimodal_aggregation import (
    MultiModalAggregationCoordinator,
    MultiModalAggregationConfig,
    AggregationStrategy,
    ParticipantModalityInfo,
    FedAvgMultiModalAggregator,
    AdaptiveModalityAggregator,
)


class TestMultiModalDataHandling:
    """Test suite for multi-modal data handling components."""

    @pytest.fixture
    def sample_configs(self):
        """Create sample modality configurations."""
        return {
            DataModality.VISION: ModalityConfig(
                modality=DataModality.VISION,
                input_dim=224 * 224 * 3,
                embedding_dim=512,
                preprocessing_params={"image_size": (224, 224)},
            ),
            DataModality.TEXT: ModalityConfig(
                modality=DataModality.TEXT,
                input_dim=768,
                embedding_dim=512,
                preprocessing_params={"max_length": 128},
            ),
            DataModality.AUDIO: ModalityConfig(
                modality=DataModality.AUDIO,
                input_dim=1024,
                embedding_dim=512,
                preprocessing_params={"sample_rate": 16000},
            ),
            DataModality.SENSOR: ModalityConfig(
                modality=DataModality.SENSOR,
                input_dim=128,
                embedding_dim=512,
                preprocessing_params={"normalize": True},
            ),
        }

    @pytest.fixture
    def sample_data(self):
        """Create sample multi-modal data."""
        return {
            DataModality.VISION: torch.randn(3, 224, 224),
            DataModality.TEXT: torch.randn(768),
            DataModality.AUDIO: torch.randn(1024),
            DataModality.SENSOR: torch.randn(128),
        }

    def test_modality_config_creation(self, sample_configs):
        """Test modality configuration creation."""
        for modality, config in sample_configs.items():
            assert config.modality == modality
            assert config.embedding_dim == 512
            assert isinstance(config.preprocessing_params, dict)

    def test_vision_data_handler(self, sample_configs, sample_data):
        """Test vision data handler."""
        handler = VisionDataHandler(sample_configs[DataModality.VISION])

        # Test single sample processing
        processed = handler.process(sample_data[DataModality.VISION])
        assert processed.shape == (3, 224, 224)

        # Test batch processing
        batch_data = torch.stack([sample_data[DataModality.VISION]] * 4)
        batch_processed = handler.process_batch(batch_data)
        assert batch_processed.shape == (4, 3, 224, 224)

        # Test validation
        assert handler.validate(sample_data[DataModality.VISION])
        assert not handler.validate(torch.randn(2, 224, 224))  # Wrong channels

    def test_text_data_handler(self, sample_configs, sample_data):
        """Test text data handler."""
        handler = TextDataHandler(sample_configs[DataModality.TEXT])

        processed = handler.process(sample_data[DataModality.TEXT])
        assert processed.shape == (768,)

        batch_data = torch.stack([sample_data[DataModality.TEXT]] * 4)
        batch_processed = handler.process_batch(batch_data)
        assert batch_processed.shape == (4, 768)

    def test_audio_data_handler(self, sample_configs, sample_data):
        """Test audio data handler."""
        handler = AudioDataHandler(sample_configs[DataModality.AUDIO])

        processed = handler.process(sample_data[DataModality.AUDIO])
        assert processed.shape == (1024,)

        # Test validation
        assert handler.validate(sample_data[DataModality.AUDIO])

    def test_sensor_data_handler(self, sample_configs, sample_data):
        """Test sensor data handler."""
        handler = SensorDataHandler(sample_configs[DataModality.SENSOR])

        processed = handler.process(sample_data[DataModality.SENSOR])
        assert processed.shape == (128,)

        # Test normalization
        normalized = handler.normalize(sample_data[DataModality.SENSOR])
        assert abs(normalized.mean().item()) < 0.1  # Should be close to 0
        assert abs(normalized.std().item() - 1.0) < 0.1  # Should be close to 1

    def test_multimodal_sample_creation(self, sample_data):
        """Test multi-modal sample creation."""
        sample = MultiModalSample(
            sample_id="test_sample",
            modality_data=sample_data,
            labels={"classification": 1, "regression": 0.5},
            metadata={"source": "test", "quality": 0.9},
        )

        assert sample.sample_id == "test_sample"
        assert len(sample.modality_data) == 4
        assert sample.labels["classification"] == 1
        assert sample.metadata["quality"] == 0.9

        # Test get_modality method
        vision_data = sample.get_modality(DataModality.VISION)
        assert torch.equal(vision_data, sample_data[DataModality.VISION])

    def test_multimodal_data_manager(self, sample_configs, sample_data):
        """Test multi-modal data manager."""
        manager = MultiModalDataManager(list(sample_configs.values()))

        # Test single sample processing
        sample = MultiModalSample(
            sample_id="test", modality_data=sample_data, labels={"class": 1}
        )

        processed_sample = manager.process_sample(sample)
        assert processed_sample.sample_id == "test"
        assert len(processed_sample.modality_data) == 4

        # Test batch creation
        samples = [
            MultiModalSample(f"sample_{i}", sample_data, {"class": i}) for i in range(4)
        ]

        batch = manager.create_batch(samples)
        assert isinstance(batch, MultiModalBatch)
        assert batch.batch_size == 4
        assert len(batch.modality_data) == 4

        # Check batch data shapes
        for modality, data in batch.modality_data.items():
            assert data.shape[0] == 4  # Batch dimension


class TestFederatedTransformer:
    """Test suite for federated transformer components."""

    @pytest.fixture
    def transformer_config(self):
        """Create transformer configuration."""
        return FederatedTransformerConfig(
            d_model=512,
            n_heads=8,
            n_layers=6,
            d_ff=2048,
            dropout=0.1,
            max_seq_length=128,
        )

    @pytest.fixture
    def modality_configs(self):
        """Create modality configurations for transformer."""
        return [
            ModalityConfig(DataModality.VISION, 512, 512),
            ModalityConfig(DataModality.TEXT, 768, 512),
            ModalityConfig(DataModality.AUDIO, 1024, 512),
        ]

    def test_modality_embedding(self, modality_configs):
        """Test modality embedding layer."""
        config = modality_configs[0]  # Vision config
        embedding = ModalityEmbedding(config, d_model=512)

        # Test embedding
        input_data = torch.randn(4, 512)  # Batch of 4, dim 512
        embedded = embedding(input_data)

        assert embedded.shape == (4, 512)
        assert not torch.equal(embedded, input_data)  # Should be transformed

    def test_cross_modal_attention(self):
        """Test cross-modal attention mechanism."""
        attention = CrossModalAttention(d_model=512, n_heads=8)

        # Create query, key, value tensors
        batch_size, seq_len, d_model = 4, 16, 512
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)

        output, attention_weights = attention(query, key, value)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, 8, seq_len, seq_len)

        # Check attention weights sum to 1
        assert torch.allclose(
            attention_weights.sum(dim=-1),
            torch.ones_like(attention_weights.sum(dim=-1)),
        )

    def test_adapter_layer(self):
        """Test adapter layer for parameter-efficient training."""
        adapter = AdapterLayer(d_model=512, adapter_dim=64)

        input_tensor = torch.randn(4, 16, 512)
        output = adapter(input_tensor)

        assert output.shape == input_tensor.shape

        # Test that adapter has fewer parameters than full layer
        adapter_params = sum(p.numel() for p in adapter.parameters())
        full_layer_params = 512 * 512  # Equivalent full linear layer
        assert adapter_params < full_layer_params

    def test_federated_multimodal_transformer(
        self, modality_configs, transformer_config
    ):
        """Test complete federated multi-modal transformer."""
        transformer = FederatedMultiModalTransformer(
            modality_configs=modality_configs, config=transformer_config
        )

        # Create sample batch
        batch_data = {
            DataModality.VISION: torch.randn(4, 512),
            DataModality.TEXT: torch.randn(4, 768),
            DataModality.AUDIO: torch.randn(4, 1024),
        }

        batch = MultiModalBatch(
            batch_size=4,
            modality_data=batch_data,
            labels=torch.randint(0, 10, (4,)),
            sample_ids=[f"sample_{i}" for i in range(4)],
        )

        # Forward pass
        output = transformer(batch)

        assert "representations" in output
        assert "attention_weights" in output
        assert "modality_embeddings" in output

        representations = output["representations"]
        assert representations.shape == (4, 512)  # Batch size, d_model

    def test_transformer_compression(self, modality_configs, transformer_config):
        """Test transformer compression capabilities."""
        transformer_config.use_compression = True
        transformer_config.compression_ratio = 0.5

        transformer = FederatedMultiModalTransformer(
            modality_configs=modality_configs, config=transformer_config
        )

        # Test parameter compression
        original_params = {}
        for name, param in transformer.named_parameters():
            original_params[name] = param.clone()

        compressed_params = transformer.compress_parameters(original_params)

        # Check that compressed parameters are smaller
        original_size = sum(p.numel() for p in original_params.values())
        compressed_size = sum(p.numel() for p in compressed_params.values())

        assert compressed_size < original_size


class TestCrossModalLearning:
    """Test suite for cross-modal learning components."""

    @pytest.fixture
    def modality_configs(self):
        """Create modality configurations."""
        return [
            ModalityConfig(DataModality.VISION, 512, 256),
            ModalityConfig(DataModality.TEXT, 768, 256),
            ModalityConfig(DataModality.AUDIO, 1024, 256),
        ]

    def test_attention_fusion(self, modality_configs):
        """Test attention-based fusion strategy."""
        fusion = AttentionFusion(modality_configs, hidden_dim=256)

        # Create modality representations
        modality_reps = {
            DataModality.VISION: torch.randn(4, 256),
            DataModality.TEXT: torch.randn(4, 256),
            DataModality.AUDIO: torch.randn(4, 256),
        }

        fused_rep = fusion(modality_reps)

        assert fused_rep.shape == (4, 256)

        # Test that fusion weights sum to 1
        attention_weights = fusion.get_attention_weights(modality_reps)
        assert torch.allclose(attention_weights.sum(dim=-1), torch.ones(4))

    def test_contrastive_learning(self, modality_configs):
        """Test contrastive learning for cross-modal alignment."""
        contrastive = ContrastiveLearning(modality_configs, temperature=0.1)

        # Create paired modality representations
        vision_reps = torch.randn(4, 256)
        text_reps = torch.randn(4, 256)

        loss = contrastive.compute_loss(vision_reps, text_reps)

        assert loss.item() >= 0  # Contrastive loss should be non-negative
        assert loss.requires_grad  # Should be differentiable

    def test_knowledge_distillation(self, modality_configs):
        """Test knowledge distillation between modalities."""
        kd = KnowledgeDistillation(modality_configs, temperature=3.0, alpha=0.7)

        # Create teacher and student outputs
        teacher_output = torch.randn(4, 10)  # 10 classes
        student_output = torch.randn(4, 10)
        true_labels = torch.randint(0, 10, (4,))

        kd_loss = kd.compute_loss(student_output, teacher_output, true_labels)

        assert kd_loss.item() >= 0
        assert kd_loss.requires_grad

    def test_cross_modal_framework(self, modality_configs):
        """Test complete cross-modal learning framework."""
        framework = CrossModalLearningFramework(
            modality_configs=modality_configs,
            fusion_strategy=FusionStrategy.ATTENTION_FUSION,
            alignment_strategy=AlignmentStrategy.CONTRASTIVE_ALIGNMENT,
            hidden_dim=256,
        )

        # Create sample batch
        batch_data = {
            DataModality.VISION: torch.randn(4, 512),
            DataModality.TEXT: torch.randn(4, 768),
            DataModality.AUDIO: torch.randn(4, 1024),
        }

        batch = MultiModalBatch(
            batch_size=4,
            modality_data=batch_data,
            labels=torch.randint(0, 10, (4,)),
            sample_ids=[f"sample_{i}" for i in range(4)],
        )

        # Forward pass
        output = framework(batch)

        assert "fused_representation" in output
        assert "modality_representations" in output
        assert "alignment_loss" in output
        assert "fusion_weights" in output

        fused_rep = output["fused_representation"]
        assert fused_rep.shape == (4, 256)

    def test_fusion_strategies(self, modality_configs):
        """Test different fusion strategies."""
        strategies = [
            FusionStrategy.EARLY_FUSION,
            FusionStrategy.LATE_FUSION,
            FusionStrategy.ATTENTION_FUSION,
            FusionStrategy.CONTRASTIVE_FUSION,
        ]

        for strategy in strategies:
            framework = CrossModalLearningFramework(
                modality_configs=modality_configs,
                fusion_strategy=strategy,
                alignment_strategy=AlignmentStrategy.CONTRASTIVE_ALIGNMENT,
                hidden_dim=256,
            )

            # Test that framework can be created without errors
            assert framework.fusion_strategy == strategy


class TestMultiModalAggregation:
    """Test suite for multi-modal aggregation strategies."""

    @pytest.fixture
    def participant_info(self):
        """Create sample participant information."""
        return {
            "participant_1": ParticipantModalityInfo(
                participant_id="participant_1",
                modality=DataModality.VISION,
                data_size=100,
                data_quality_score=0.9,
                modality_coverage=0.8,
                local_performance={"accuracy": 0.85, "loss": 0.3},
            ),
            "participant_2": ParticipantModalityInfo(
                participant_id="participant_2",
                modality=DataModality.TEXT,
                data_size=150,
                data_quality_score=0.8,
                modality_coverage=0.9,
                local_performance={"accuracy": 0.82, "loss": 0.35},
            ),
            "participant_3": ParticipantModalityInfo(
                participant_id="participant_3",
                modality=DataModality.AUDIO,
                data_size=80,
                data_quality_score=0.95,
                modality_coverage=0.7,
                local_performance={"accuracy": 0.88, "loss": 0.25},
            ),
        }

    @pytest.fixture
    def participant_updates(self):
        """Create sample participant parameter updates."""
        return {
            "participant_1": {
                "layer1.weight": torch.randn(512, 256),
                "layer1.bias": torch.randn(512),
                "layer2.weight": torch.randn(256, 128),
            },
            "participant_2": {
                "layer1.weight": torch.randn(512, 256),
                "layer1.bias": torch.randn(512),
                "layer2.weight": torch.randn(256, 128),
            },
            "participant_3": {
                "layer1.weight": torch.randn(512, 256),
                "layer1.bias": torch.randn(512),
                "layer2.weight": torch.randn(256, 128),
            },
        }

    def test_fedavg_aggregator(self, participant_updates, participant_info):
        """Test FedAvg-style multi-modal aggregator."""
        config = MultiModalAggregationConfig(
            strategy=AggregationStrategy.FEDAVG_MULTIMODAL
        )
        aggregator = FedAvgMultiModalAggregator(config)

        # Test weight computation
        weights = aggregator.compute_weights(participant_info)

        # Weights should sum to 1
        assert abs(sum(weights.values()) - 1.0) < 1e-6

        # Participant with more data should have higher weight
        assert weights["participant_2"] > weights["participant_3"]

        # Test aggregation
        aggregated = aggregator.aggregate(participant_updates, participant_info)

        assert len(aggregated) == 3  # Three parameters
        assert "layer1.weight" in aggregated
        assert aggregated["layer1.weight"].shape == (512, 256)

    def test_adaptive_aggregator(self, participant_updates, participant_info):
        """Test adaptive modality aggregator."""
        config = MultiModalAggregationConfig(
            strategy=AggregationStrategy.ADAPTIVE_MODALITY,
            adaptation_rate=0.1,
            momentum=0.9,
        )
        aggregator = AdaptiveModalityAggregator(config)

        # Simulate multiple rounds to test adaptation
        for round_num in range(3):
            # Update performance history
            for pid, info in participant_info.items():
                info.local_performance["accuracy"] += np.random.uniform(-0.05, 0.05)

            aggregated = aggregator.aggregate(participant_updates, participant_info)

            assert len(aggregated) == 3
            assert all(param.requires_grad for param in aggregated.values())

        # Check that adaptive weights have been learned
        assert len(aggregator.adaptive_weights) > 0

    def test_aggregation_coordinator(self, participant_updates, participant_info):
        """Test multi-modal aggregation coordinator."""
        config = MultiModalAggregationConfig(
            strategy=AggregationStrategy.QUALITY_AWARE, quality_threshold=0.7
        )
        coordinator = MultiModalAggregationCoordinator(config)

        # Test aggregation round
        aggregated = coordinator.aggregate_round(
            participant_updates=participant_updates,
            participant_info=participant_info,
            round_number=1,
        )

        assert len(aggregated) == 3
        assert len(coordinator.aggregation_history) == 1

        # Test aggregation statistics
        stats = coordinator.get_aggregation_stats()
        assert stats["total_rounds"] == 1
        assert stats["avg_participants_per_round"] == 3

    def test_differential_privacy(self, participant_updates, participant_info):
        """Test differential privacy in aggregation."""
        config = MultiModalAggregationConfig(
            strategy=AggregationStrategy.FEDAVG_MULTIMODAL,
            enable_differential_privacy=True,
            noise_multiplier=1.0,
            clipping_threshold=1.0,
        )
        coordinator = MultiModalAggregationCoordinator(config)

        # Store original updates
        original_updates = {}
        for pid, updates in participant_updates.items():
            original_updates[pid] = {
                name: param.clone() for name, param in updates.items()
            }

        # Apply differential privacy
        noisy_updates = coordinator._apply_differential_privacy(participant_updates)

        # Check that updates have been modified (due to noise)
        for pid in participant_updates.keys():
            for param_name in participant_updates[pid].keys():
                original_param = original_updates[pid][param_name]
                noisy_param = noisy_updates[pid][param_name]

                # Should be different due to added noise
                assert not torch.equal(original_param, noisy_param)

    def test_aggregation_strategies(self, participant_updates, participant_info):
        """Test all aggregation strategies."""
        strategies = [
            AggregationStrategy.FEDAVG_MULTIMODAL,
            AggregationStrategy.MODALITY_WEIGHTED,
            AggregationStrategy.ADAPTIVE_MODALITY,
            AggregationStrategy.QUALITY_AWARE,
            AggregationStrategy.HIERARCHICAL_MULTIMODAL,
        ]

        for strategy in strategies:
            config = MultiModalAggregationConfig(strategy=strategy)
            coordinator = MultiModalAggregationCoordinator(config)

            # Test that aggregation works for all strategies
            aggregated = coordinator.aggregate_round(
                participant_updates=participant_updates,
                participant_info=participant_info,
                round_number=1,
            )

            assert len(aggregated) == 3
            assert all(isinstance(param, torch.Tensor) for param in aggregated.values())


class TestIntegration:
    """Integration tests for the complete multi-modal federated learning system."""

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create modality configurations
        modality_configs = [
            ModalityConfig(DataModality.VISION, 512, 256),
            ModalityConfig(DataModality.TEXT, 768, 256),
        ]

        # Create data manager
        data_manager = MultiModalDataManager(modality_configs)

        # Create sample data
        samples = []
        for i in range(8):
            sample = MultiModalSample(
                sample_id=f"sample_{i}",
                modality_data={
                    DataModality.VISION: torch.randn(512),
                    DataModality.TEXT: torch.randn(768),
                },
                labels={"class": i % 2},
            )
            samples.append(sample)

        # Create batch
        batch = data_manager.create_batch(samples)

        # Create models
        transformer_config = FederatedTransformerConfig(
            d_model=256, n_heads=4, n_layers=2
        )
        transformer = FederatedMultiModalTransformer(
            modality_configs, transformer_config
        )

        cross_modal_framework = CrossModalLearningFramework(
            modality_configs=modality_configs,
            fusion_strategy=FusionStrategy.ATTENTION_FUSION,
            alignment_strategy=AlignmentStrategy.CONTRASTIVE_ALIGNMENT,
            hidden_dim=256,
        )

        # Forward passes
        transformer_output = transformer(batch)
        cross_modal_output = cross_modal_framework(batch)

        # Check outputs
        assert "representations" in transformer_output
        assert "fused_representation" in cross_modal_output

        # Test aggregation
        participant_updates = {}
        participant_info = {}

        for i in range(3):
            pid = f"participant_{i}"
            updates = {}
            for name, param in transformer.named_parameters():
                updates[name] = torch.randn_like(param) * 0.01

            participant_updates[pid] = updates
            participant_info[pid] = ParticipantModalityInfo(
                participant_id=pid,
                modality=DataModality.VISION if i % 2 == 0 else DataModality.TEXT,
                data_size=50 + i * 10,
                data_quality_score=0.8 + i * 0.05,
            )

        # Aggregation
        config = MultiModalAggregationConfig(
            strategy=AggregationStrategy.ADAPTIVE_MODALITY
        )
        coordinator = MultiModalAggregationCoordinator(config)

        aggregated = coordinator.aggregate_round(
            participant_updates=participant_updates,
            participant_info=participant_info,
            round_number=1,
        )

        assert len(aggregated) > 0

    def test_error_handling(self):
        """Test error handling in various components."""
        # Test empty batch handling
        modality_configs = [ModalityConfig(DataModality.VISION, 512, 256)]
        data_manager = MultiModalDataManager(modality_configs)

        with pytest.raises(ValueError):
            data_manager.create_batch([])  # Empty samples list

        # Test mismatched configurations
        with pytest.raises(ValueError):
            FederatedMultiModalTransformer(
                modality_configs=[],  # Empty configs
                config=FederatedTransformerConfig(),
            )

    def test_performance_characteristics(self):
        """Test performance characteristics of components."""
        # Create large-scale test
        modality_configs = [
            ModalityConfig(DataModality.VISION, 2048, 512),
            ModalityConfig(DataModality.TEXT, 1024, 512),
            ModalityConfig(DataModality.AUDIO, 1536, 512),
        ]

        # Large batch test
        batch_data = {
            DataModality.VISION: torch.randn(32, 2048),
            DataModality.TEXT: torch.randn(32, 1024),
            DataModality.AUDIO: torch.randn(32, 1536),
        }

        batch = MultiModalBatch(
            batch_size=32,
            modality_data=batch_data,
            labels=torch.randint(0, 10, (32,)),
            sample_ids=[f"sample_{i}" for i in range(32)],
        )

        # Test transformer performance
        transformer_config = FederatedTransformerConfig(
            d_model=512, n_heads=8, n_layers=6, use_compression=True
        )
        transformer = FederatedMultiModalTransformer(
            modality_configs, transformer_config
        )

        # Measure forward pass time
        import time

        start_time = time.time()

        with torch.no_grad():
            output = transformer(batch)

        end_time = time.time()
        forward_time = end_time - start_time

        # Should complete within reasonable time (less than 5 seconds on CPU)
        assert forward_time < 5.0

        # Check memory usage is reasonable
        assert output["representations"].shape == (32, 512)


def run_tests():
    """Run all tests with pytest."""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
