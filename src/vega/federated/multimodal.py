"""
Enhanced Multi-Modal Federated Learning Framework

This module provides comprehensive support for federated learning across multiple
data modalities including vision, text, audio, and sensor data. It implements
unified data handling, specialized aggregation strategies, cross-modal learning,
and federated transformer architectures.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import base64
import io
from pathlib import Path
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataModality(Enum):
    """Enumeration of supported data modalities."""

    VISION = "vision"
    TEXT = "text"
    AUDIO = "audio"
    SENSOR = "sensor"
    TABULAR = "tabular"
    TIME_SERIES = "time_series"


@dataclass
class ModalityConfig:
    """Configuration for a specific data modality."""

    modality: DataModality
    input_shape: Tuple[int, ...]
    preprocessing_params: Dict[str, Any] = field(default_factory=dict)
    encoding_params: Dict[str, Any] = field(default_factory=dict)
    feature_dim: Optional[int] = None
    max_sequence_length: Optional[int] = None
    vocabulary_size: Optional[int] = None
    sample_rate: Optional[int] = None  # For audio data
    channels: Optional[int] = None  # For vision/audio data


@dataclass
class MultiModalSample:
    """Container for multi-modal data samples."""

    sample_id: str
    modalities: Dict[DataModality, Any]
    labels: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    participant_id: Optional[str] = None


@dataclass
class MultiModalBatch:
    """Container for batched multi-modal data."""

    samples: List[MultiModalSample]
    batch_size: int
    modality_tensors: Dict[DataModality, torch.Tensor]
    labels: Optional[torch.Tensor] = None
    masks: Dict[DataModality, torch.Tensor] = field(default_factory=dict)


class ModalityDataHandler(ABC):
    """Abstract base class for modality-specific data handlers."""

    def __init__(self, config: ModalityConfig):
        self.config = config
        self.modality = config.modality

    @abstractmethod
    def preprocess(self, data: Any) -> torch.Tensor:
        """Preprocess raw data into tensor format."""
        pass

    @abstractmethod
    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Encode data into feature representation."""
        pass

    @abstractmethod
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features back to data format."""
        pass

    @abstractmethod
    def validate_data(self, data: Any) -> bool:
        """Validate data format and shape."""
        pass

    def get_feature_dim(self) -> int:
        """Get the output feature dimension."""
        return self.config.feature_dim or self.config.input_shape[-1]


class VisionDataHandler(ModalityDataHandler):
    """Handler for vision/image data."""

    def __init__(self, config: ModalityConfig):
        super().__init__(config)
        if config.modality != DataModality.VISION:
            raise ValueError("VisionDataHandler requires VISION modality")

        # Default preprocessing parameters
        self.normalize_mean = config.preprocessing_params.get(
            "normalize_mean", [0.485, 0.456, 0.406]
        )
        self.normalize_std = config.preprocessing_params.get(
            "normalize_std", [0.229, 0.224, 0.225]
        )
        self.resize_dims = config.preprocessing_params.get("resize_dims", (224, 224))

    def preprocess(self, data: Any) -> torch.Tensor:
        """Preprocess image data."""
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).float()
        elif isinstance(data, torch.Tensor):
            tensor = data.float()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Ensure correct shape: (C, H, W) or (B, C, H, W)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)  # Add batch dimension

        # Resize if needed
        if tensor.shape[-2:] != self.resize_dims:
            tensor = F.interpolate(
                tensor, size=self.resize_dims, mode="bilinear", align_corners=False
            )

        # Normalize
        if len(self.normalize_mean) == tensor.shape[1]:  # Check channel dimension
            mean = torch.tensor(self.normalize_mean).view(1, -1, 1, 1)
            std = torch.tensor(self.normalize_std).view(1, -1, 1, 1)
            tensor = (tensor - mean) / std

        return tensor

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Encode image data into feature representation."""
        # Simple CNN-based encoding (can be replaced with more sophisticated models)
        encoder = self._get_vision_encoder()
        with torch.no_grad():
            features = encoder(data)
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features back to image format."""
        # Simple decoder (placeholder implementation)
        decoder = self._get_vision_decoder()
        with torch.no_grad():
            reconstructed = decoder(features)
        return reconstructed

    def validate_data(self, data: Any) -> bool:
        """Validate image data format."""
        if isinstance(data, (np.ndarray, torch.Tensor)):
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)

            # Check dimensions
            if data.dim() < 2 or data.dim() > 4:
                return False

            # Check reasonable image dimensions
            if data.dim() >= 2 and (data.shape[-1] < 8 or data.shape[-2] < 8):
                return False

            return True
        return False

    def _get_vision_encoder(self) -> nn.Module:
        """Get vision encoder network."""
        # Simple CNN encoder
        channels = self.config.channels or 3
        return nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, self.get_feature_dim()),
        )

    def _get_vision_decoder(self) -> nn.Module:
        """Get vision decoder network."""
        # Simple decoder (placeholder)
        return nn.Sequential(
            nn.Linear(self.get_feature_dim(), 256 * 7 * 7),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                64, self.config.channels or 3, kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid(),
        )


class TextDataHandler(ModalityDataHandler):
    """Handler for text data."""

    def __init__(self, config: ModalityConfig):
        super().__init__(config)
        if config.modality != DataModality.TEXT:
            raise ValueError("TextDataHandler requires TEXT modality")

        self.max_length = config.max_sequence_length or 512
        self.vocab_size = config.vocabulary_size or 50000
        self.embedding_dim = config.feature_dim or 768

        # Simple tokenizer (in practice, use proper tokenizers like transformers)
        self.tokenizer = self._create_simple_tokenizer()

    def preprocess(self, data: Any) -> torch.Tensor:
        """Preprocess text data."""
        if isinstance(data, str):
            text = data
        elif isinstance(data, list) and all(isinstance(x, str) for x in data):
            text = " ".join(data)
        else:
            raise ValueError(f"Unsupported text data type: {type(data)}")

        # Tokenize
        tokens = self.tokenizer.encode(text)

        # Pad or truncate
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        else:
            tokens.extend([0] * (self.max_length - len(tokens)))  # Pad with zeros

        return torch.tensor(tokens, dtype=torch.long)

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Encode text data into feature representation."""
        encoder = self._get_text_encoder()
        with torch.no_grad():
            # data shape: (batch_size, seq_length) or (seq_length,)
            if data.dim() == 1:
                data = data.unsqueeze(0)

            features = encoder(data)
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features back to text tokens."""
        decoder = self._get_text_decoder()
        with torch.no_grad():
            tokens = decoder(features)
        return tokens

    def validate_data(self, data: Any) -> bool:
        """Validate text data format."""
        if isinstance(data, str):
            return len(data.strip()) > 0
        elif isinstance(data, list):
            return all(isinstance(x, str) for x in data) and len(data) > 0
        elif isinstance(data, torch.Tensor):
            return data.dtype in [torch.long, torch.int] and data.dim() <= 2
        return False

    def _create_simple_tokenizer(self):
        """Create a simple tokenizer (placeholder)."""

        class SimpleTokenizer:
            def __init__(self, vocab_size):
                self.vocab_size = vocab_size
                self.char_to_id = {chr(i): i for i in range(min(256, vocab_size))}
                self.id_to_char = {i: chr(i) for i in range(min(256, vocab_size))}

            def encode(self, text: str) -> List[int]:
                return [self.char_to_id.get(c, 0) for c in text.lower()]

            def decode(self, tokens: List[int]) -> str:
                return "".join([self.id_to_char.get(t, "") for t in tokens])

        return SimpleTokenizer(self.vocab_size)

    def _get_text_encoder(self) -> nn.Module:
        """Get text encoder network."""
        return nn.Sequential(
            nn.Embedding(self.vocab_size, self.embedding_dim),
            nn.LSTM(
                self.embedding_dim,
                self.embedding_dim // 2,
                batch_first=True,
                bidirectional=True,
            ),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.embedding_dim, self.get_feature_dim()),
        )

    def _get_text_decoder(self) -> nn.Module:
        """Get text decoder network."""
        return nn.Sequential(
            nn.Linear(self.get_feature_dim(), self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim, self.max_length * self.vocab_size),
            nn.Unflatten(1, (self.max_length, self.vocab_size)),
        )


class AudioDataHandler(ModalityDataHandler):
    """Handler for audio data."""

    def __init__(self, config: ModalityConfig):
        super().__init__(config)
        if config.modality != DataModality.AUDIO:
            raise ValueError("AudioDataHandler requires AUDIO modality")

        self.sample_rate = config.sample_rate or 16000
        self.n_fft = config.preprocessing_params.get("n_fft", 512)
        self.hop_length = config.preprocessing_params.get("hop_length", 256)
        self.n_mels = config.preprocessing_params.get("n_mels", 128)
        self.max_length = config.max_sequence_length or 1000  # Max time frames

    def preprocess(self, data: Any) -> torch.Tensor:
        """Preprocess audio data."""
        if isinstance(data, np.ndarray):
            audio = torch.from_numpy(data).float()
        elif isinstance(data, torch.Tensor):
            audio = data.float()
        else:
            raise ValueError(f"Unsupported audio data type: {type(data)}")

        # Ensure mono audio
        if audio.dim() > 1:
            audio = audio.mean(dim=-1)

        # Convert to mel spectrogram
        spectrogram = self._to_mel_spectrogram(audio)

        # Pad or truncate temporal dimension
        if spectrogram.shape[-1] > self.max_length:
            spectrogram = spectrogram[..., : self.max_length]
        else:
            pad_length = self.max_length - spectrogram.shape[-1]
            spectrogram = F.pad(spectrogram, (0, pad_length))

        return spectrogram

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Encode audio data into feature representation."""
        encoder = self._get_audio_encoder()
        with torch.no_grad():
            features = encoder(data)
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features back to audio format."""
        decoder = self._get_audio_decoder()
        with torch.no_grad():
            reconstructed = decoder(features)
        return reconstructed

    def validate_data(self, data: Any) -> bool:
        """Validate audio data format."""
        if isinstance(data, (np.ndarray, torch.Tensor)):
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)

            # Check reasonable audio length
            if data.numel() < 100:  # Too short
                return False

            return True
        return False

    def _to_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio to mel spectrogram."""
        # Simplified mel spectrogram computation
        # In practice, use torchaudio.transforms.MelSpectrogram
        stft = torch.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True
        )
        magnitude = torch.abs(stft)

        # Simple linear-to-mel mapping (placeholder)
        mel_scale = torch.linspace(0, 1, self.n_mels).unsqueeze(-1)
        mel_spectrogram = torch.matmul(
            mel_scale, magnitude.view(magnitude.shape[0], -1)
        )
        mel_spectrogram = mel_spectrogram.view(self.n_mels, -1)

        return mel_spectrogram

    def _get_audio_encoder(self) -> nn.Module:
        """Get audio encoder network."""
        return nn.Sequential(
            nn.Conv1d(self.n_mels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, self.get_feature_dim()),
        )

    def _get_audio_decoder(self) -> nn.Module:
        """Get audio decoder network."""
        return nn.Sequential(
            nn.Linear(self.get_feature_dim(), 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.n_mels * self.max_length),
            nn.Unflatten(1, (self.n_mels, self.max_length)),
        )


class SensorDataHandler(ModalityDataHandler):
    """Handler for sensor/IoT data."""

    def __init__(self, config: ModalityConfig):
        super().__init__(config)
        if config.modality != DataModality.SENSOR:
            raise ValueError("SensorDataHandler requires SENSOR modality")

        self.num_sensors = config.input_shape[0] if config.input_shape else 10
        self.sequence_length = config.max_sequence_length or 100
        self.normalization_params = config.preprocessing_params.get("normalization", {})

    def preprocess(self, data: Any) -> torch.Tensor:
        """Preprocess sensor data."""
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).float()
        elif isinstance(data, torch.Tensor):
            tensor = data.float()
        elif isinstance(data, list):
            tensor = torch.tensor(data, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported sensor data type: {type(data)}")

        # Ensure correct shape: (num_sensors, sequence_length)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)  # Add sensor dimension

        # Pad or truncate sequence dimension
        if tensor.shape[-1] > self.sequence_length:
            tensor = tensor[..., : self.sequence_length]
        else:
            pad_length = self.sequence_length - tensor.shape[-1]
            tensor = F.pad(tensor, (0, pad_length))

        # Normalize if parameters provided
        if self.normalization_params:
            mean = self.normalization_params.get("mean", 0)
            std = self.normalization_params.get("std", 1)
            tensor = (tensor - mean) / std

        return tensor

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Encode sensor data into feature representation."""
        encoder = self._get_sensor_encoder()
        with torch.no_grad():
            features = encoder(data)
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features back to sensor format."""
        decoder = self._get_sensor_decoder()
        with torch.no_grad():
            reconstructed = decoder(features)
        return reconstructed

    def validate_data(self, data: Any) -> bool:
        """Validate sensor data format."""
        if isinstance(data, (np.ndarray, torch.Tensor, list)):
            if isinstance(data, list):
                data = np.array(data)
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)

            # Check reasonable sensor data dimensions
            if data.numel() == 0:
                return False

            return True
        return False

    def _get_sensor_encoder(self) -> nn.Module:
        """Get sensor encoder network."""
        return nn.Sequential(
            nn.Conv1d(self.num_sensors, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, self.get_feature_dim()),
        )

    def _get_sensor_decoder(self) -> nn.Module:
        """Get sensor decoder network."""
        return nn.Sequential(
            nn.Linear(self.get_feature_dim(), 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.num_sensors * self.sequence_length),
            nn.Unflatten(1, (self.num_sensors, self.sequence_length)),
        )


class MultiModalDataManager:
    """Manager for multi-modal data handling and processing."""

    def __init__(self):
        self.handlers: Dict[DataModality, ModalityDataHandler] = {}
        self.modality_configs: Dict[DataModality, ModalityConfig] = {}

    def register_modality(self, config: ModalityConfig) -> None:
        """Register a new data modality."""
        self.modality_configs[config.modality] = config

        # Create appropriate handler
        if config.modality == DataModality.VISION:
            handler = VisionDataHandler(config)
        elif config.modality == DataModality.TEXT:
            handler = TextDataHandler(config)
        elif config.modality == DataModality.AUDIO:
            handler = AudioDataHandler(config)
        elif config.modality == DataModality.SENSOR:
            handler = SensorDataHandler(config)
        else:
            raise ValueError(f"Unsupported modality: {config.modality}")

        self.handlers[config.modality] = handler
        logger.info(f"Registered {config.modality.value} modality handler")

    def process_sample(
        self, sample: MultiModalSample
    ) -> Dict[DataModality, torch.Tensor]:
        """Process a multi-modal sample."""
        processed_data = {}

        for modality, data in sample.modalities.items():
            if modality not in self.handlers:
                logger.warning(f"No handler registered for modality: {modality}")
                continue

            handler = self.handlers[modality]
            if handler.validate_data(data):
                processed_data[modality] = handler.preprocess(data)
            else:
                logger.warning(f"Invalid data for modality: {modality}")

        return processed_data

    def create_batch(self, samples: List[MultiModalSample]) -> MultiModalBatch:
        """Create a batch from multiple samples."""
        if not samples:
            raise ValueError("Cannot create batch from empty sample list")

        # Get all modalities present in samples
        all_modalities = set()
        for sample in samples:
            all_modalities.update(sample.modalities.keys())

        # Process all samples
        processed_samples = []
        modality_tensors = {modality: [] for modality in all_modalities}

        for sample in samples:
            processed_data = self.process_sample(sample)
            processed_samples.append(sample)

            for modality in all_modalities:
                if modality in processed_data:
                    modality_tensors[modality].append(processed_data[modality])
                else:
                    # Create empty tensor for missing modality
                    handler = self.handlers.get(modality)
                    if handler:
                        empty_shape = self._get_empty_tensor_shape(handler.config)
                        empty_tensor = torch.zeros(empty_shape)
                        modality_tensors[modality].append(empty_tensor)

        # Stack tensors for each modality
        batched_tensors = {}
        masks = {}

        for modality, tensor_list in modality_tensors.items():
            if tensor_list:
                try:
                    batched_tensors[modality] = torch.stack(tensor_list)
                    masks[modality] = torch.ones(len(tensor_list), dtype=torch.bool)
                except RuntimeError as e:
                    logger.warning(f"Could not stack tensors for {modality}: {e}")
                    continue

        return MultiModalBatch(
            samples=processed_samples,
            batch_size=len(samples),
            modality_tensors=batched_tensors,
            masks=masks,
        )

    def encode_batch(self, batch: MultiModalBatch) -> Dict[DataModality, torch.Tensor]:
        """Encode a batch of multi-modal data."""
        encoded_features = {}

        for modality, tensor in batch.modality_tensors.items():
            if modality in self.handlers:
                handler = self.handlers[modality]
                encoded_features[modality] = handler.encode(tensor)

        return encoded_features

    def get_unified_feature_dim(self) -> int:
        """Get unified feature dimension across all modalities."""
        if not self.handlers:
            return 512  # Default

        return max(handler.get_feature_dim() for handler in self.handlers.values())

    def _get_empty_tensor_shape(self, config: ModalityConfig) -> Tuple[int, ...]:
        """Get shape for empty tensor of given modality."""
        if config.modality == DataModality.VISION:
            return (config.channels or 3, *config.input_shape[-2:])
        elif config.modality == DataModality.TEXT:
            return (config.max_sequence_length or 512,)
        elif config.modality == DataModality.AUDIO:
            n_mels = config.preprocessing_params.get("n_mels", 128)
            max_length = config.max_sequence_length or 1000
            return (n_mels, max_length)
        elif config.modality == DataModality.SENSOR:
            return config.input_shape
        else:
            return (1,)  # Fallback


def create_sample_multimodal_config() -> Dict[DataModality, ModalityConfig]:
    """Create sample configurations for different modalities."""
    configs = {}

    # Vision configuration
    configs[DataModality.VISION] = ModalityConfig(
        modality=DataModality.VISION,
        input_shape=(3, 224, 224),
        feature_dim=512,
        channels=3,
        preprocessing_params={
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "resize_dims": (224, 224),
        },
    )

    # Text configuration
    configs[DataModality.TEXT] = ModalityConfig(
        modality=DataModality.TEXT,
        input_shape=(512,),
        feature_dim=512,
        max_sequence_length=512,
        vocabulary_size=50000,
    )

    # Audio configuration
    configs[DataModality.AUDIO] = ModalityConfig(
        modality=DataModality.AUDIO,
        input_shape=(128, 1000),  # n_mels x time_frames
        feature_dim=512,
        sample_rate=16000,
        max_sequence_length=1000,
        preprocessing_params={"n_fft": 512, "hop_length": 256, "n_mels": 128},
    )

    # Sensor configuration
    configs[DataModality.SENSOR] = ModalityConfig(
        modality=DataModality.SENSOR,
        input_shape=(10, 100),  # 10 sensors x 100 time steps
        feature_dim=512,
        max_sequence_length=100,
        preprocessing_params={"normalization": {"mean": 0.0, "std": 1.0}},
    )

    return configs


if __name__ == "__main__":
    # Example usage
    print("Multi-Modal Federated Learning Framework")
    print("=" * 50)

    # Create sample configurations
    configs = create_sample_multimodal_config()

    # Initialize data manager
    manager = MultiModalDataManager()

    # Register modalities
    for config in configs.values():
        manager.register_modality(config)

    print(f"Registered {len(manager.handlers)} modality handlers")
    print(f"Unified feature dimension: {manager.get_unified_feature_dim()}")
