"""
Cross-Modal Learning Capabilities for Federated Learning

This module implements cross-modal knowledge transfer, multi-modal fusion strategies,
and techniques for learning shared representations across different data modalities
in federated environments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from enum import Enum

from .multimodal import DataModality, MultiModalBatch, MultiModalSample
from .federated_transformer import FederatedMultiModalTransformer

# Configure logging
logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Cross-modal fusion strategies."""

    EARLY_FUSION = "early_fusion"
    LATE_FUSION = "late_fusion"
    INTERMEDIATE_FUSION = "intermediate_fusion"
    ATTENTION_FUSION = "attention_fusion"
    CONTRASTIVE_FUSION = "contrastive_fusion"


class AlignmentStrategy(Enum):
    """Cross-modal alignment strategies."""

    CANONICAL_CORRELATION = "cca"
    SHARED_SUBSPACE = "shared_subspace"
    ADVERSARIAL_ALIGNMENT = "adversarial"
    CONTRASTIVE_ALIGNMENT = "contrastive"
    MUTUAL_INFORMATION = "mutual_info"


@dataclass
class CrossModalConfig:
    """Configuration for cross-modal learning."""

    fusion_strategy: FusionStrategy = FusionStrategy.ATTENTION_FUSION
    alignment_strategy: AlignmentStrategy = AlignmentStrategy.CONTRASTIVE_ALIGNMENT
    shared_representation_dim: int = 512
    modality_specific_dim: int = 256

    # Contrastive learning parameters
    temperature: float = 0.1
    margin: float = 0.2
    negative_samples: int = 32

    # Adversarial alignment parameters
    discriminator_hidden_dim: int = 256
    adversarial_weight: float = 0.1

    # Knowledge transfer parameters
    transfer_weight: float = 0.5
    distillation_temperature: float = 4.0

    # Regularization
    orthogonality_weight: float = 0.01
    diversity_weight: float = 0.01


class SharedRepresentationLearner(nn.Module):
    """Learns shared representations across modalities."""

    def __init__(
        self,
        input_dims: Dict[DataModality, int],
        shared_dim: int,
        modality_specific_dim: int,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.shared_dim = shared_dim
        self.modality_specific_dim = modality_specific_dim

        # Shared encoders
        self.shared_encoders = nn.ModuleDict()

        # Modality-specific encoders
        self.modality_encoders = nn.ModuleDict()

        for modality, input_dim in input_dims.items():
            # Shared representation branch
            self.shared_encoders[modality.value] = nn.Sequential(
                nn.Linear(input_dim, shared_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(shared_dim * 2, shared_dim),
                nn.LayerNorm(shared_dim),
            )

            # Modality-specific branch
            self.modality_encoders[modality.value] = nn.Sequential(
                nn.Linear(input_dim, modality_specific_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(modality_specific_dim * 2, modality_specific_dim),
                nn.LayerNorm(modality_specific_dim),
            )

    def forward(
        self, modality_features: Dict[DataModality, torch.Tensor]
    ) -> Tuple[Dict[DataModality, torch.Tensor], Dict[DataModality, torch.Tensor]]:
        """Extract shared and modality-specific representations."""
        shared_reps = {}
        specific_reps = {}

        for modality, features in modality_features.items():
            # Flatten features if needed
            if features.dim() > 2:
                batch_size = features.shape[0]
                features = features.view(batch_size, -1)

            shared_reps[modality] = self.shared_encoders[modality.value](features)
            specific_reps[modality] = self.modality_encoders[modality.value](features)

        return shared_reps, specific_reps


class ContrastiveLearning(nn.Module):
    """Contrastive learning for cross-modal alignment."""

    def __init__(self, feature_dim: int, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim // 2),
        )

    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss between two modalities."""
        # Project features
        proj1 = F.normalize(self.projection_head(features1), dim=-1)
        proj2 = F.normalize(self.projection_head(features2), dim=-1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(proj1, proj2.T) / self.temperature

        # Create labels (positive pairs are on the diagonal)
        batch_size = features1.shape[0]
        labels = torch.arange(batch_size, device=features1.device)

        # Compute contrastive loss
        loss1 = F.cross_entropy(similarity_matrix, labels)
        loss2 = F.cross_entropy(similarity_matrix.T, labels)

        return (loss1 + loss2) / 2


class ModalityDiscriminator(nn.Module):
    """Discriminator for adversarial modality alignment."""

    def __init__(self, feature_dim: int, hidden_dim: int, num_modalities: int):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_modalities),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict modality from features."""
        return self.discriminator(features)


class AttentionFusion(nn.Module):
    """Attention-based fusion mechanism."""

    def __init__(self, feature_dim: int, num_modalities: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities

        # Multi-head attention for fusion
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=8, batch_first=True
        )

        # Modality-aware attention weights
        self.modality_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(
        self, modality_features: Dict[DataModality, torch.Tensor]
    ) -> torch.Tensor:
        """Fuse features from multiple modalities using attention."""
        if len(modality_features) == 1:
            return next(iter(modality_features.values()))

        # Stack modality features
        feature_list = list(modality_features.values())
        stacked_features = torch.stack(
            feature_list, dim=1
        )  # (batch, num_modalities, feature_dim)

        # Apply multi-head attention
        attended_features, attention_weights = self.multihead_attention(
            stacked_features, stacked_features, stacked_features
        )

        # Compute modality-specific attention weights
        modality_weights = []
        for features in feature_list:
            weight = self.modality_attention(features)
            modality_weights.append(weight)

        modality_weights = torch.stack(
            modality_weights, dim=1
        )  # (batch, num_modalities, 1)

        # Weighted fusion
        weighted_features = attended_features * modality_weights
        fused_features = weighted_features.sum(dim=1)

        # Layer normalization
        fused_features = self.layer_norm(fused_features)

        return fused_features


class KnowledgeDistillation(nn.Module):
    """Knowledge distillation between modalities."""

    def __init__(self, feature_dim: int, temperature: float = 4.0):
        super().__init__()
        self.temperature = temperature
        self.teacher_projector = nn.Linear(feature_dim, feature_dim)
        self.student_projector = nn.Linear(feature_dim, feature_dim)

    def forward(
        self, teacher_features: torch.Tensor, student_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute knowledge distillation loss."""
        # Project features
        teacher_logits = self.teacher_projector(teacher_features) / self.temperature
        student_logits = self.student_projector(student_features) / self.temperature

        # Compute soft targets
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(student_logits, dim=-1)

        # KL divergence loss
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

        return kl_loss * (self.temperature**2)


class CrossModalLearningFramework(nn.Module):
    """Complete framework for cross-modal learning in federated settings."""

    def __init__(
        self, config: CrossModalConfig, modality_dims: Dict[DataModality, int]
    ):
        super().__init__()
        self.config = config
        self.modality_dims = modality_dims
        self.num_modalities = len(modality_dims)

        # Shared representation learner
        self.shared_learner = SharedRepresentationLearner(
            modality_dims,
            config.shared_representation_dim,
            config.modality_specific_dim,
        )

        # Fusion mechanism
        if config.fusion_strategy == FusionStrategy.ATTENTION_FUSION:
            self.fusion_module = AttentionFusion(
                config.shared_representation_dim, self.num_modalities
            )
        else:
            self.fusion_module = None

        # Contrastive learning
        if config.alignment_strategy == AlignmentStrategy.CONTRASTIVE_ALIGNMENT:
            self.contrastive_learner = ContrastiveLearning(
                config.shared_representation_dim, config.temperature
            )

        # Adversarial alignment
        if config.alignment_strategy == AlignmentStrategy.ADVERSARIAL_ALIGNMENT:
            self.discriminator = ModalityDiscriminator(
                config.shared_representation_dim,
                config.discriminator_hidden_dim,
                self.num_modalities,
            )

        # Knowledge distillation
        self.knowledge_distiller = KnowledgeDistillation(
            config.shared_representation_dim, config.distillation_temperature
        )

        # Output layers
        self.output_projection = nn.Linear(
            config.shared_representation_dim, config.shared_representation_dim
        )

    def forward(
        self,
        modality_features: Dict[DataModality, torch.Tensor],
        return_intermediate: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through cross-modal learning framework."""
        # Extract shared and specific representations
        shared_reps, specific_reps = self.shared_learner(modality_features)

        # Apply fusion
        if self.fusion_module is not None:
            fused_features = self.fusion_module(shared_reps)
        else:
            # Simple concatenation or averaging
            if self.config.fusion_strategy == FusionStrategy.EARLY_FUSION:
                feature_list = list(shared_reps.values())
                fused_features = torch.cat(feature_list, dim=-1)
                fused_features = self.output_projection(fused_features)
            else:  # Late fusion (averaging)
                feature_list = list(shared_reps.values())
                fused_features = torch.stack(feature_list, dim=0).mean(dim=0)

        # Final output projection
        output = self.output_projection(fused_features)

        if return_intermediate:
            return {
                "output": output,
                "shared_representations": shared_reps,
                "specific_representations": specific_reps,
                "fused_features": fused_features,
            }

        return output

    def compute_alignment_loss(
        self, shared_reps: Dict[DataModality, torch.Tensor]
    ) -> torch.Tensor:
        """Compute cross-modal alignment loss."""
        if self.config.alignment_strategy == AlignmentStrategy.CONTRASTIVE_ALIGNMENT:
            # Pairwise contrastive loss
            modality_list = list(shared_reps.keys())
            total_loss = 0.0
            num_pairs = 0

            for i, modality1 in enumerate(modality_list):
                for j, modality2 in enumerate(modality_list[i + 1 :], i + 1):
                    contrastive_loss = self.contrastive_learner(
                        shared_reps[modality1], shared_reps[modality2]
                    )
                    total_loss += contrastive_loss
                    num_pairs += 1

            return total_loss / max(num_pairs, 1)

        elif self.config.alignment_strategy == AlignmentStrategy.ADVERSARIAL_ALIGNMENT:
            # Adversarial loss for modality invariance
            all_features = torch.cat(list(shared_reps.values()), dim=0)
            modality_labels = []

            for i, (modality, features) in enumerate(shared_reps.items()):
                batch_size = features.shape[0]
                modality_labels.extend([i] * batch_size)

            modality_labels = torch.tensor(modality_labels, device=all_features.device)

            # Discriminator loss (maximize)
            discriminator_logits = self.discriminator(all_features.detach())
            discriminator_loss = F.cross_entropy(discriminator_logits, modality_labels)

            # Generator loss (minimize discriminator accuracy)
            generator_logits = self.discriminator(all_features)
            uniform_target = torch.full_like(
                generator_logits, 1.0 / self.num_modalities
            )
            generator_loss = F.kl_div(
                F.log_softmax(generator_logits, dim=-1),
                uniform_target,
                reduction="batchmean",
            )

            return self.config.adversarial_weight * (
                discriminator_loss + generator_loss
            )

        else:
            return torch.tensor(0.0, device=next(iter(shared_reps.values())).device)

    def compute_knowledge_transfer_loss(
        self, shared_reps: Dict[DataModality, torch.Tensor]
    ) -> torch.Tensor:
        """Compute knowledge transfer loss between modalities."""
        if len(shared_reps) < 2:
            return torch.tensor(0.0, device=next(iter(shared_reps.values())).device)

        modality_list = list(shared_reps.keys())
        total_loss = 0.0
        num_pairs = 0

        # Pairwise knowledge distillation
        for i, modality1 in enumerate(modality_list):
            for j, modality2 in enumerate(modality_list):
                if i != j:
                    # modality1 as teacher, modality2 as student
                    kd_loss = self.knowledge_distiller(
                        shared_reps[modality1].detach(),  # Teacher (detached)
                        shared_reps[modality2],  # Student
                    )
                    total_loss += kd_loss
                    num_pairs += 1

        return self.config.transfer_weight * total_loss / max(num_pairs, 1)

    def compute_regularization_loss(
        self,
        shared_reps: Dict[DataModality, torch.Tensor],
        specific_reps: Dict[DataModality, torch.Tensor],
    ) -> torch.Tensor:
        """Compute regularization losses for representation learning."""
        total_loss = 0.0

        # Orthogonality constraint between shared and specific representations
        if self.config.orthogonality_weight > 0:
            orthogonality_loss = 0.0
            num_modalities = 0

            for modality in shared_reps.keys():
                if modality in specific_reps:
                    shared = shared_reps[modality]
                    specific = specific_reps[modality]

                    # Compute correlation between shared and specific
                    correlation = torch.matmul(shared.T, specific)
                    orthogonality_loss += torch.norm(correlation, p="fro") ** 2
                    num_modalities += 1

            if num_modalities > 0:
                total_loss += (
                    self.config.orthogonality_weight
                    * orthogonality_loss
                    / num_modalities
                )

        # Diversity constraint for shared representations
        if self.config.diversity_weight > 0:
            all_shared = torch.cat(list(shared_reps.values()), dim=0)

            # Encourage diversity in shared representations
            correlation_matrix = torch.corrcoef(all_shared.T)
            off_diagonal_sum = torch.sum(torch.abs(correlation_matrix)) - torch.trace(
                torch.abs(correlation_matrix)
            )
            diversity_loss = off_diagonal_sum / (
                correlation_matrix.numel() - correlation_matrix.shape[0]
            )

            total_loss += self.config.diversity_weight * diversity_loss

        return total_loss

    def compute_total_loss(
        self,
        modality_features: Dict[DataModality, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        task_loss_fn: Optional[Callable] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute total training loss including all cross-modal objectives."""
        # Forward pass
        results = self.forward(modality_features, return_intermediate=True)

        losses = {}

        # Task-specific loss
        if labels is not None and task_loss_fn is not None:
            task_loss = task_loss_fn(results["output"], labels)
            losses["task_loss"] = task_loss

        # Alignment loss
        alignment_loss = self.compute_alignment_loss(results["shared_representations"])
        losses["alignment_loss"] = alignment_loss

        # Knowledge transfer loss
        transfer_loss = self.compute_knowledge_transfer_loss(
            results["shared_representations"]
        )
        losses["transfer_loss"] = transfer_loss

        # Regularization loss
        regularization_loss = self.compute_regularization_loss(
            results["shared_representations"], results["specific_representations"]
        )
        losses["regularization_loss"] = regularization_loss

        # Total loss
        total_loss = sum(losses.values())
        losses["total_loss"] = total_loss

        return losses


class FederatedCrossModalTrainer:
    """Trainer for federated cross-modal learning."""

    def __init__(self, framework: CrossModalLearningFramework):
        self.framework = framework
        self.optimizer = None
        self.discriminator_optimizer = None

    def setup_optimizers(self, lr: float = 1e-4, discriminator_lr: float = 1e-3):
        """Setup optimizers for training."""
        # Main framework optimizer
        self.optimizer = torch.optim.AdamW(
            [
                p
                for name, p in self.framework.named_parameters()
                if "discriminator" not in name
            ],
            lr=lr,
        )

        # Discriminator optimizer (if using adversarial alignment)
        if hasattr(self.framework, "discriminator"):
            self.discriminator_optimizer = torch.optim.AdamW(
                self.framework.discriminator.parameters(), lr=discriminator_lr
            )

    def train_step(
        self,
        modality_features: Dict[DataModality, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        task_loss_fn: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """Perform a single training step."""
        if self.optimizer is None:
            raise ValueError("Optimizers not set up. Call setup_optimizers() first.")

        self.framework.train()

        # Compute losses
        losses = self.framework.compute_total_loss(
            modality_features, labels, task_loss_fn
        )

        # Update discriminator (if using adversarial training)
        if self.discriminator_optimizer is not None and "alignment_loss" in losses:
            self.discriminator_optimizer.zero_grad()
            # Note: Discriminator loss is included in alignment_loss
            losses["alignment_loss"].backward(retain_graph=True)
            self.discriminator_optimizer.step()

        # Update main framework
        self.optimizer.zero_grad()
        losses["total_loss"].backward()
        self.optimizer.step()

        # Convert to float for logging
        float_losses = {
            k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()
        }

        return float_losses

    def evaluate(
        self,
        modality_features: Dict[DataModality, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        task_loss_fn: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """Evaluate the framework."""
        self.framework.eval()

        with torch.no_grad():
            losses = self.framework.compute_total_loss(
                modality_features, labels, task_loss_fn
            )

        # Convert to float for logging
        float_losses = {
            k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()
        }

        return float_losses


def create_cross_modal_framework(
    modality_dims: Dict[DataModality, int], config: Optional[CrossModalConfig] = None
) -> CrossModalLearningFramework:
    """Factory function to create cross-modal learning framework."""

    if config is None:
        config = CrossModalConfig()

    framework = CrossModalLearningFramework(config, modality_dims)

    logger.info(f"Created cross-modal framework for {len(modality_dims)} modalities")
    logger.info(f"Fusion strategy: {config.fusion_strategy.value}")
    logger.info(f"Alignment strategy: {config.alignment_strategy.value}")

    return framework


if __name__ == "__main__":
    # Example usage
    print("Cross-Modal Learning Framework")
    print("=" * 35)

    # Define modality dimensions
    modality_dims = {
        DataModality.VISION: 2048,
        DataModality.TEXT: 768,
        DataModality.AUDIO: 1024,
    }

    # Create framework
    framework = create_cross_modal_framework(modality_dims)

    # Print framework info
    total_params = sum(p.numel() for p in framework.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Number of modalities: {len(modality_dims)}")
    print(f"Shared representation dim: {framework.config.shared_representation_dim}")
