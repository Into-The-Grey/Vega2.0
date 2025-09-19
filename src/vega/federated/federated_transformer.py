"""
Federated Transformer Architecture for Multi-Modal Learning

This module implements federated transformer models supporting multi-modal inputs,
cross-modal attention mechanisms, and efficient parameter sharing for large-scale
transformer training in federated settings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging

from .multimodal import (
    DataModality,
    MultiModalBatch,
    MultiModalDataManager,
    ModalityConfig,
    create_sample_multimodal_config,
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Configuration for federated transformer architecture."""

    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    activation: str = "relu"
    layer_norm_eps: float = 1e-5
    batch_first: bool = True
    max_sequence_length: int = 512

    # Multi-modal specific
    enable_cross_modal_attention: bool = True
    cross_modal_fusion_strategy: str = (
        "concatenate"  # "concatenate", "add", "attention"
    )
    modality_embedding_dim: int = 64
    shared_encoder_layers: int = 3
    modality_specific_layers: int = 3


@dataclass
class FederatedTransformerConfig:
    """Configuration for federated training of transformers."""

    transformer_config: TransformerConfig
    federated_strategy: str = "fedavg"  # "fedavg", "fedprox", "scaffold"
    compression_strategy: str = "none"  # "none", "quantization", "sparsification"
    parameter_sharing_strategy: str = "full"  # "full", "partial", "adapter"

    # Adapter-based learning
    use_adapters: bool = False
    adapter_hidden_dim: int = 64
    adapter_layers: List[str] = field(
        default_factory=lambda: ["attention", "feedforward"]
    )

    # Differential privacy
    enable_differential_privacy: bool = False
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0


class ModalityEmbedding(nn.Module):
    """Learnable embeddings for different modalities."""

    def __init__(self, num_modalities: int, d_model: int):
        super().__init__()
        self.modality_embeddings = nn.Embedding(num_modalities, d_model)
        self.num_modalities = num_modalities

    def forward(self, modality_indices: torch.Tensor) -> torch.Tensor:
        """Forward pass for modality embeddings."""
        return self.modality_embeddings(modality_indices)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_length: int = 5000):
        super().__init__()

        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[: x.size(0), :]


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for cross-modal attention."""
        # Multi-head attention
        attn_output, _ = self.multihead_attn(query, key, value, attn_mask=mask)

        # Residual connection and layer norm
        x = self.layer_norm1(query + self.dropout(attn_output))

        # Feedforward network
        ffn_output = self.ffn(x)

        # Residual connection and layer norm
        output = self.layer_norm2(x + self.dropout(ffn_output))

        return output


class AdapterLayer(nn.Module):
    """Adapter layer for parameter-efficient fine-tuning."""

    def __init__(self, d_model: int, adapter_hidden_dim: int):
        super().__init__()
        self.down_project = nn.Linear(d_model, adapter_hidden_dim)
        self.up_project = nn.Linear(adapter_hidden_dim, d_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for adapter layer."""
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_project(x)
        return residual + x


class MultiModalTransformerEncoder(nn.Module):
    """Multi-modal transformer encoder."""

    def __init__(self, config: TransformerConfig, num_modalities: int):
        super().__init__()
        self.config = config
        self.num_modalities = num_modalities

        # Modality embeddings
        self.modality_embedding = ModalityEmbedding(num_modalities, config.d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            config.d_model, config.max_sequence_length
        )

        # Shared transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            layer_norm_eps=config.layer_norm_eps,
            batch_first=config.batch_first,
        )

        self.shared_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.shared_encoder_layers
        )

        # Modality-specific encoders
        self.modality_encoders = nn.ModuleDict()
        for i in range(num_modalities):
            modality_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=config.modality_specific_layers
            )
            self.modality_encoders[f"modality_{i}"] = modality_encoder

        # Cross-modal attention layers
        if config.enable_cross_modal_attention:
            self.cross_modal_attention = nn.ModuleList(
                [
                    CrossModalAttention(config.d_model, config.nhead, config.dropout)
                    for _ in range(config.shared_encoder_layers)
                ]
            )

        # Fusion layer
        if config.cross_modal_fusion_strategy == "attention":
            self.fusion_attention = nn.MultiheadAttention(
                config.d_model, config.nhead, batch_first=True
            )
        elif config.cross_modal_fusion_strategy == "concatenate":
            self.fusion_projection = nn.Linear(
                config.d_model * num_modalities, config.d_model
            )

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        modality_features: Dict[DataModality, torch.Tensor],
        modality_masks: Optional[Dict[DataModality, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass for multi-modal transformer encoder."""
        batch_size = next(iter(modality_features.values())).shape[0]

        # Process each modality
        modality_outputs = {}
        modality_indices = {}

        for i, (modality, features) in enumerate(modality_features.items()):
            # Add modality embeddings
            modality_emb = self.modality_embedding(
                torch.full(
                    (batch_size, features.shape[1]),
                    i,
                    device=features.device,
                    dtype=torch.long,
                )
            )

            # Add positional encoding
            enhanced_features = features + modality_emb
            enhanced_features = self.pos_encoding(enhanced_features)
            enhanced_features = self.dropout(enhanced_features)

            # Shared encoding
            shared_output = self.shared_encoder(enhanced_features)

            # Modality-specific encoding
            modality_specific_output = self.modality_encoders[f"modality_{i}"](
                shared_output
            )

            modality_outputs[modality] = modality_specific_output
            modality_indices[modality] = i

        # Cross-modal attention (if enabled)
        if self.config.enable_cross_modal_attention and len(modality_outputs) > 1:
            modality_list = list(modality_outputs.keys())

            for layer_idx, cross_attn_layer in enumerate(self.cross_modal_attention):
                updated_outputs = {}

                for modality in modality_list:
                    # Use current modality as query, others as key/value
                    query = modality_outputs[modality]

                    # Concatenate other modalities for key/value
                    other_modalities = [m for m in modality_list if m != modality]
                    if other_modalities:
                        key_value = torch.cat(
                            [modality_outputs[m] for m in other_modalities], dim=1
                        )

                        updated_output = cross_attn_layer(query, key_value, key_value)
                        updated_outputs[modality] = updated_output
                    else:
                        updated_outputs[modality] = query

                modality_outputs = updated_outputs

        # Fusion
        fused_output = self._fuse_modalities(modality_outputs)

        return fused_output

    def _fuse_modalities(
        self, modality_outputs: Dict[DataModality, torch.Tensor]
    ) -> torch.Tensor:
        """Fuse outputs from different modalities."""
        if len(modality_outputs) == 1:
            return next(iter(modality_outputs.values()))

        if self.config.cross_modal_fusion_strategy == "concatenate":
            # Concatenate along feature dimension
            concatenated = torch.cat(list(modality_outputs.values()), dim=-1)
            fused = self.fusion_projection(concatenated)

        elif self.config.cross_modal_fusion_strategy == "add":
            # Element-wise addition
            fused = sum(modality_outputs.values())

        elif self.config.cross_modal_fusion_strategy == "attention":
            # Attention-based fusion
            stacked = torch.stack(list(modality_outputs.values()), dim=1)
            query = stacked.mean(dim=1, keepdim=True)  # Global representation as query

            fused_output, _ = self.fusion_attention(query, stacked, stacked)
            fused = fused_output.squeeze(1)

        else:
            raise ValueError(
                f"Unknown fusion strategy: {self.config.cross_modal_fusion_strategy}"
            )

        return fused


class FederatedMultiModalTransformer(nn.Module):
    """Federated multi-modal transformer model."""

    def __init__(
        self,
        config: FederatedTransformerConfig,
        modality_configs: Dict[DataModality, ModalityConfig],
    ):
        super().__init__()
        self.config = config
        self.modality_configs = modality_configs
        self.num_modalities = len(modality_configs)

        # Input projection layers for each modality
        self.input_projections = nn.ModuleDict()
        for modality, modality_config in modality_configs.items():
            input_dim = modality_config.feature_dim or 512
            self.input_projections[modality.value] = nn.Linear(
                input_dim, config.transformer_config.d_model
            )

        # Multi-modal transformer encoder
        self.encoder = MultiModalTransformerEncoder(
            config.transformer_config, self.num_modalities
        )

        # Adapter layers (if enabled)
        if config.use_adapters:
            self.adapters = nn.ModuleDict()
            for layer_name in config.adapter_layers:
                self.adapters[layer_name] = AdapterLayer(
                    config.transformer_config.d_model, config.adapter_hidden_dim
                )

        # Output layers
        self.output_projection = nn.Linear(
            config.transformer_config.d_model, config.transformer_config.d_model
        )

        self.layer_norm = nn.LayerNorm(config.transformer_config.d_model)

    def forward(self, batch: MultiModalBatch) -> torch.Tensor:
        """Forward pass for federated multi-modal transformer."""
        # Project inputs to transformer dimension
        projected_features = {}

        for modality, features in batch.modality_tensors.items():
            # Flatten features if needed
            if features.dim() > 3:
                batch_size, seq_len = features.shape[:2]
                features = features.view(batch_size, seq_len, -1)
            elif features.dim() == 2:
                features = features.unsqueeze(1)  # Add sequence dimension

            # Project to transformer dimension
            projected = self.input_projections[modality.value](features)
            projected_features[modality] = projected

        # Apply transformer encoder
        encoded_output = self.encoder(projected_features, batch.masks)

        # Apply adapters if enabled
        if self.config.use_adapters:
            for layer_name, adapter in self.adapters.items():
                encoded_output = adapter(encoded_output)

        # Output projection and normalization
        output = self.output_projection(encoded_output)
        output = self.layer_norm(output)

        return output

    def get_federated_parameters(self) -> Dict[str, torch.Tensor]:
        """Get parameters for federated aggregation."""
        if self.config.parameter_sharing_strategy == "full":
            return {name: param for name, param in self.named_parameters()}

        elif self.config.parameter_sharing_strategy == "partial":
            # Only share encoder parameters
            shared_params = {}
            for name, param in self.named_parameters():
                if "encoder" in name:
                    shared_params[name] = param
            return shared_params

        elif self.config.parameter_sharing_strategy == "adapter":
            # Only share adapter parameters
            adapter_params = {}
            for name, param in self.named_parameters():
                if "adapter" in name:
                    adapter_params[name] = param
            return adapter_params

        else:
            raise ValueError(
                f"Unknown parameter sharing strategy: {self.config.parameter_sharing_strategy}"
            )

    def update_federated_parameters(
        self, new_parameters: Dict[str, torch.Tensor]
    ) -> None:
        """Update model parameters from federated aggregation."""
        current_params = dict(self.named_parameters())

        for name, new_param in new_parameters.items():
            if name in current_params:
                current_params[name].data.copy_(new_param.data)
            else:
                logger.warning(f"Parameter {name} not found in model")

    def compress_parameters(
        self, parameters: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compress parameters for efficient communication."""
        if self.config.compression_strategy == "none":
            return parameters

        elif self.config.compression_strategy == "quantization":
            # Simple 8-bit quantization
            compressed = {}
            for name, param in parameters.items():
                # Quantize to 8-bit
                param_min, param_max = param.min(), param.max()
                scale = (param_max - param_min) / 255.0
                quantized = (
                    torch.round((param - param_min) / scale).clamp(0, 255).byte()
                )

                compressed[name] = {
                    "quantized": quantized,
                    "min": param_min,
                    "max": param_max,
                    "scale": scale,
                }
            return compressed

        elif self.config.compression_strategy == "sparsification":
            # Top-k sparsification
            compressed = {}
            sparsity_ratio = 0.9  # Keep top 10% of parameters

            for name, param in parameters.items():
                flat_param = param.flatten()
                k = max(1, int(len(flat_param) * (1 - sparsity_ratio)))

                # Get top-k indices
                _, top_indices = torch.topk(torch.abs(flat_param), k)

                # Create sparse representation
                sparse_values = flat_param[top_indices]
                compressed[name] = {
                    "indices": top_indices,
                    "values": sparse_values,
                    "shape": param.shape,
                }
            return compressed

        else:
            raise ValueError(
                f"Unknown compression strategy: {self.config.compression_strategy}"
            )

    def decompress_parameters(
        self, compressed_params: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Decompress parameters received from federated aggregation."""
        if self.config.compression_strategy == "none":
            return compressed_params

        elif self.config.compression_strategy == "quantization":
            decompressed = {}
            for name, compressed_data in compressed_params.items():
                quantized = compressed_data["quantized"].float()
                param_min = compressed_data["min"]
                scale = compressed_data["scale"]

                # Dequantize
                decompressed_param = quantized * scale + param_min
                decompressed[name] = decompressed_param

            return decompressed

        elif self.config.compression_strategy == "sparsification":
            decompressed = {}
            for name, compressed_data in compressed_params.items():
                indices = compressed_data["indices"]
                values = compressed_data["values"]
                shape = compressed_data["shape"]

                # Reconstruct sparse tensor
                flat_param = torch.zeros(torch.prod(torch.tensor(shape)))
                flat_param[indices] = values
                decompressed_param = flat_param.view(shape)
                decompressed[name] = decompressed_param

            return decompressed

        else:
            raise ValueError(
                f"Unknown compression strategy: {self.config.compression_strategy}"
            )


class MultiModalFederatedTrainer:
    """Trainer for federated multi-modal transformers."""

    def __init__(
        self, model: FederatedMultiModalTransformer, data_manager: MultiModalDataManager
    ):
        self.model = model
        self.data_manager = data_manager
        self.optimizer = None
        self.loss_fn = nn.MSELoss()  # Can be customized

    def setup_optimizer(self, lr: float = 1e-4, weight_decay: float = 1e-5):
        """Setup optimizer for training."""
        if self.model.config.use_adapters:
            # Only train adapter parameters
            adapter_params = []
            for name, param in self.model.named_parameters():
                if "adapter" in name:
                    adapter_params.append(param)
            self.optimizer = torch.optim.AdamW(
                adapter_params, lr=lr, weight_decay=weight_decay
            )
        else:
            # Train all parameters
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )

    def train_step(self, batch: MultiModalBatch, labels: torch.Tensor) -> float:
        """Perform a single training step with optional differential privacy (DP-SGD)."""
        if self.optimizer is None:
            raise ValueError("Optimizer not set up. Call setup_optimizer() first.")

        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(batch)

        # Compute loss
        if outputs.dim() > 2:
            outputs = outputs.mean(dim=1)  # Pool over sequence dimension

        loss = self.loss_fn(outputs, labels)

        # Backward pass
        loss.backward()

        # Differential Privacy: Gradient clipping and noise addition
        if self.model.config.enable_differential_privacy:
            max_norm = getattr(self.model.config, "max_grad_norm", 1.0)
            noise_multiplier = getattr(self.model.config, "noise_multiplier", 1.0)
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            # Add Gaussian noise to each parameter's gradient
            for p in self.model.parameters():
                if p.grad is not None:
                    noise = torch.normal(
                        mean=0.0,
                        std=noise_multiplier * max_norm,
                        size=p.grad.shape,
                        device=p.grad.device,
                    )
                    p.grad.add_(noise)

        self.optimizer.step()

        return loss.item()

    def evaluate(self, batch: MultiModalBatch, labels: torch.Tensor) -> float:
        """Evaluate model on a batch."""
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(batch)

            if outputs.dim() > 2:
                outputs = outputs.mean(dim=1)

            loss = self.loss_fn(outputs, labels)

        return loss.item()


def create_federated_multimodal_transformer(
    modality_configs: Optional[Dict[DataModality, ModalityConfig]] = None,
    transformer_config: Optional[TransformerConfig] = None,
    federated_config: Optional[FederatedTransformerConfig] = None,
) -> FederatedMultiModalTransformer:
    """Factory function to create federated multi-modal transformer."""

    if modality_configs is None:
        modality_configs = create_sample_multimodal_config()

    if transformer_config is None:
        transformer_config = TransformerConfig()

    if federated_config is None:
        federated_config = FederatedTransformerConfig(transformer_config)

    model = FederatedMultiModalTransformer(federated_config, modality_configs)

    logger.info(
        f"Created federated multi-modal transformer with {len(modality_configs)} modalities"
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


if __name__ == "__main__":
    # Example usage
    print("Federated Multi-Modal Transformer")
    print("=" * 40)

    # Create model
    model = create_federated_multimodal_transformer()

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Number of modalities: {model.num_modalities}")
    print(f"Model dimension: {model.config.transformer_config.d_model}")
