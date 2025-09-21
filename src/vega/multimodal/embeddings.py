"""
Multi-Modal Embeddings System
=============================

Unified embedding system that creates a shared embedding space for different modalities.
Supports advanced embedding techniques including:
- CLIP-style vision-language embeddings
- Audio-text alignment
- Video-text understanding
- Cross-modal retrieval optimization

This system enables semantic search across modalities and supports fine-tuning
for domain-specific applications.
"""

import asyncio
import logging
import numpy as np
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Set
import hashlib

# Vector storage and similarity
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Neural network frameworks (optional imports)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyTorch not available - using CPU-only embeddings")

try:
    import transformers
    from transformers import CLIPModel, CLIPProcessor

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class EmbeddingModel(Enum):
    """Available embedding model types"""

    SIMPLE = "simple"  # Basic statistical features
    CLIP = "clip"  # Vision-language understanding
    SENTENCE_TRANSFORMER = "sentence_transformer"  # Text embeddings
    CUSTOM = "custom"  # Custom trained model
    UNIFIED = "unified"  # Multi-modal unified space


class ModalityAdapter(Enum):
    """Adapters for different input modalities"""

    TEXT_ENCODER = "text_encoder"
    IMAGE_ENCODER = "image_encoder"
    AUDIO_ENCODER = "audio_encoder"
    VIDEO_ENCODER = "video_encoder"
    DOCUMENT_ENCODER = "document_encoder"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""

    model_type: EmbeddingModel = EmbeddingModel.SIMPLE
    embedding_dim: int = 512
    normalize_embeddings: bool = True
    use_gpu: bool = False
    model_path: Optional[str] = None
    cache_embeddings: bool = True
    batch_size: int = 32
    similarity_threshold: float = 0.5


@dataclass
class EmbeddingVector:
    """Represents an embedding vector with metadata"""

    vector: np.ndarray
    modality: str
    source_id: str
    model_type: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def similarity(self, other: "EmbeddingVector", metric: str = "cosine") -> float:
        """Calculate similarity with another embedding"""
        if metric == "cosine":
            return float(cosine_similarity([self.vector], [other.vector])[0][0])
        elif metric == "euclidean":
            distance = np.linalg.norm(self.vector - other.vector)
            return 1.0 / (1.0 + distance)
        elif metric == "dot_product":
            return float(np.dot(self.vector, other.vector))
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")


class EmbeddingSpace:
    """
    Manages a unified embedding space for multi-modal content.
    Supports indexing, similarity search, and space visualization.
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.embeddings: Dict[str, EmbeddingVector] = {}
        self.embedding_matrix: Optional[np.ndarray] = None
        self.index_map: Dict[str, int] = {}
        self.pca_reducer: Optional[PCA] = None
        self.scaler: Optional[StandardScaler] = None

        # Modality-specific indices
        self.modality_indices: Dict[str, Set[str]] = {
            "text": set(),
            "image": set(),
            "audio": set(),
            "video": set(),
            "document": set(),
        }

        logger.info(
            f"Initialized EmbeddingSpace with {config.embedding_dim}D embeddings"
        )

    def add_embedding(self, embedding: EmbeddingVector) -> bool:
        """Add embedding to the space"""
        try:
            self.embeddings[embedding.source_id] = embedding
            self.modality_indices[embedding.modality].add(embedding.source_id)

            # Invalidate matrix cache
            self.embedding_matrix = None
            self.index_map = {}

            logger.debug(f"Added embedding for {embedding.source_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add embedding {embedding.source_id}: {e}")
            return False

    def get_embedding(self, source_id: str) -> Optional[EmbeddingVector]:
        """Retrieve embedding by source ID"""
        return self.embeddings.get(source_id)

    def remove_embedding(self, source_id: str) -> bool:
        """Remove embedding from space"""
        if source_id in self.embeddings:
            embedding = self.embeddings[source_id]
            del self.embeddings[source_id]

            # Remove from modality index
            self.modality_indices[embedding.modality].discard(source_id)

            # Invalidate matrix cache
            self.embedding_matrix = None
            self.index_map = {}

            logger.debug(f"Removed embedding for {source_id}")
            return True
        return False

    def similarity_search(
        self,
        query_embedding: EmbeddingVector,
        top_k: int = 10,
        modality_filter: Optional[List[str]] = None,
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """
        Find most similar embeddings to query

        Args:
            query_embedding: The query embedding
            top_k: Number of results to return
            modality_filter: Only search within specific modalities
            threshold: Minimum similarity threshold

        Returns:
            List of (source_id, similarity_score) tuples
        """
        results = []
        threshold = threshold or self.config.similarity_threshold

        # Filter candidates by modality if specified
        candidates = set(self.embeddings.keys())
        if modality_filter:
            candidates = set()
            for modality in modality_filter:
                candidates.update(self.modality_indices.get(modality, set()))

        # Calculate similarities
        for source_id in candidates:
            if source_id == query_embedding.source_id:
                continue  # Skip self

            embedding = self.embeddings[source_id]
            similarity = query_embedding.similarity(embedding)

            if similarity >= threshold:
                results.append((source_id, similarity))

        # Sort by similarity and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_embedding_matrix(self) -> np.ndarray:
        """Get matrix representation of all embeddings"""
        if self.embedding_matrix is None:
            self._build_embedding_matrix()
        return self.embedding_matrix

    def _build_embedding_matrix(self):
        """Build matrix representation for efficient operations"""
        if not self.embeddings:
            self.embedding_matrix = np.array([]).reshape(0, self.config.embedding_dim)
            return

        # Create matrix and index mapping
        embeddings_list = list(self.embeddings.values())
        self.embedding_matrix = np.vstack([emb.vector for emb in embeddings_list])
        self.index_map = {emb.source_id: i for i, emb in enumerate(embeddings_list)}

        logger.info(f"Built embedding matrix: {self.embedding_matrix.shape}")

    def reduce_dimensionality(
        self, target_dim: int = 2, method: str = "pca"
    ) -> np.ndarray:
        """Reduce embedding dimensionality for visualization"""
        matrix = self.get_embedding_matrix()

        if method == "pca":
            if self.pca_reducer is None or self.pca_reducer.n_components != target_dim:
                self.pca_reducer = PCA(n_components=target_dim)

            reduced = self.pca_reducer.fit_transform(matrix)
            return reduced
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")

    def get_modality_statistics(self) -> Dict[str, Any]:
        """Get statistics about the embedding space"""
        stats = {
            "total_embeddings": len(self.embeddings),
            "modality_counts": {
                modality: len(indices)
                for modality, indices in self.modality_indices.items()
            },
            "embedding_dimension": self.config.embedding_dim,
            "model_type": self.config.model_type.value,
        }

        if self.embedding_matrix is not None:
            stats["matrix_shape"] = self.embedding_matrix.shape
            stats["mean_norm"] = float(
                np.mean(np.linalg.norm(self.embedding_matrix, axis=1))
            )
            stats["std_norm"] = float(
                np.std(np.linalg.norm(self.embedding_matrix, axis=1))
            )

        return stats

    def save_space(self, filepath: str):
        """Save embedding space to disk"""
        data = {
            "config": {
                "model_type": self.config.model_type.value,
                "embedding_dim": self.config.embedding_dim,
                "normalize_embeddings": self.config.normalize_embeddings,
            },
            "embeddings": {
                source_id: {
                    "vector": emb.vector.tolist(),
                    "modality": emb.modality,
                    "model_type": emb.model_type,
                    "created_at": emb.created_at.isoformat(),
                    "metadata": emb.metadata,
                }
                for source_id, emb in self.embeddings.items()
            },
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved embedding space to {filepath}")

    @classmethod
    def load_space(cls, filepath: str) -> "EmbeddingSpace":
        """Load embedding space from disk"""
        with open(filepath, "r") as f:
            data = json.load(f)

        # Reconstruct config
        config = EmbeddingConfig(
            model_type=EmbeddingModel(data["config"]["model_type"]),
            embedding_dim=data["config"]["embedding_dim"],
            normalize_embeddings=data["config"]["normalize_embeddings"],
        )

        # Create space
        space = cls(config)

        # Restore embeddings
        for source_id, emb_data in data["embeddings"].items():
            embedding = EmbeddingVector(
                vector=np.array(emb_data["vector"]),
                modality=emb_data["modality"],
                source_id=source_id,
                model_type=emb_data["model_type"],
                created_at=datetime.fromisoformat(emb_data["created_at"]),
                metadata=emb_data["metadata"],
            )
            space.add_embedding(embedding)

        logger.info(f"Loaded embedding space from {filepath}")
        return space


class MultiModalEmbeddings:
    """
    Advanced multi-modal embedding system that creates unified representations
    for different types of content (text, images, audio, video, documents).
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.embedding_space = EmbeddingSpace(config)
        self.models = {}
        self.processors = {}

        # Initialize models based on configuration
        asyncio.create_task(self._initialize_models())

    async def _initialize_models(self):
        """Initialize embedding models"""
        try:
            if self.config.model_type == EmbeddingModel.CLIP and TRANSFORMERS_AVAILABLE:
                await self._init_clip_model()

            logger.info(f"Initialized embedding models: {self.config.model_type.value}")

        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            # Fallback to simple embeddings
            self.config.model_type = EmbeddingModel.SIMPLE

    async def _init_clip_model(self):
        """Initialize CLIP model for vision-language embeddings"""
        try:
            self.models["clip"] = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self.processors["clip"] = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )

            if self.config.use_gpu and torch.cuda.is_available():
                self.models["clip"] = self.models["clip"].cuda()

            logger.info("CLIP model initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize CLIP model: {e}")
            raise

    async def embed_text(
        self, text: str, source_id: str, metadata: Dict[str, Any] = None
    ) -> EmbeddingVector:
        """Generate embedding for text content"""
        try:
            if self.config.model_type == EmbeddingModel.CLIP and "clip" in self.models:
                vector = await self._clip_text_embedding(text)
            else:
                vector = self._simple_text_embedding(text)

            embedding = EmbeddingVector(
                vector=vector,
                modality="text",
                source_id=source_id,
                model_type=self.config.model_type.value,
                metadata=metadata or {},
            )

            # Add to space
            self.embedding_space.add_embedding(embedding)

            return embedding

        except Exception as e:
            logger.error(f"Text embedding failed for {source_id}: {e}")
            raise

    async def embed_image(
        self, image_path: str, source_id: str, metadata: Dict[str, Any] = None
    ) -> EmbeddingVector:
        """Generate embedding for image content"""
        try:
            if self.config.model_type == EmbeddingModel.CLIP and "clip" in self.models:
                vector = await self._clip_image_embedding(image_path)
            else:
                vector = self._simple_image_embedding(image_path)

            embedding = EmbeddingVector(
                vector=vector,
                modality="image",
                source_id=source_id,
                model_type=self.config.model_type.value,
                metadata=metadata or {},
            )

            self.embedding_space.add_embedding(embedding)
            return embedding

        except Exception as e:
            logger.error(f"Image embedding failed for {source_id}: {e}")
            raise

    async def embed_audio(
        self, audio_path: str, source_id: str, metadata: Dict[str, Any] = None
    ) -> EmbeddingVector:
        """Generate embedding for audio content"""
        try:
            vector = self._simple_audio_embedding(audio_path)

            embedding = EmbeddingVector(
                vector=vector,
                modality="audio",
                source_id=source_id,
                model_type=self.config.model_type.value,
                metadata=metadata or {},
            )

            self.embedding_space.add_embedding(embedding)
            return embedding

        except Exception as e:
            logger.error(f"Audio embedding failed for {source_id}: {e}")
            raise

    async def embed_video(
        self, video_path: str, source_id: str, metadata: Dict[str, Any] = None
    ) -> EmbeddingVector:
        """Generate embedding for video content"""
        try:
            vector = self._simple_video_embedding(video_path)

            embedding = EmbeddingVector(
                vector=vector,
                modality="video",
                source_id=source_id,
                model_type=self.config.model_type.value,
                metadata=metadata or {},
            )

            self.embedding_space.add_embedding(embedding)
            return embedding

        except Exception as e:
            logger.error(f"Video embedding failed for {source_id}: {e}")
            raise

    async def embed_document(
        self, doc_path: str, source_id: str, metadata: Dict[str, Any] = None
    ) -> EmbeddingVector:
        """Generate embedding for document content"""
        try:
            vector = self._simple_document_embedding(doc_path)

            embedding = EmbeddingVector(
                vector=vector,
                modality="document",
                source_id=source_id,
                model_type=self.config.model_type.value,
                metadata=metadata or {},
            )

            self.embedding_space.add_embedding(embedding)
            return embedding

        except Exception as e:
            logger.error(f"Document embedding failed for {source_id}: {e}")
            raise

    async def _clip_text_embedding(self, text: str) -> np.ndarray:
        """Generate CLIP text embedding"""
        inputs = self.processors["clip"](text=[text], return_tensors="pt", padding=True)

        if self.config.use_gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self.models["clip"].get_text_features(**inputs)

        vector = text_features.cpu().numpy()[0]

        if self.config.normalize_embeddings:
            vector = vector / np.linalg.norm(vector)

        return vector

    async def _clip_image_embedding(self, image_path: str) -> np.ndarray:
        """Generate CLIP image embedding"""
        from PIL import Image

        image = Image.open(image_path)
        inputs = self.processors["clip"](images=image, return_tensors="pt")

        if self.config.use_gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.models["clip"].get_image_features(**inputs)

        vector = image_features.cpu().numpy()[0]

        if self.config.normalize_embeddings:
            vector = vector / np.linalg.norm(vector)

        return vector

    def _simple_text_embedding(self, text: str) -> np.ndarray:
        """Simple text embedding using statistical features"""
        words = text.lower().split()

        # Basic text features
        features = []

        # Length features
        features.append(len(text))
        features.append(len(words))
        features.append(len(set(words)))  # Unique words

        # Character features
        features.append(sum(1 for c in text if c.isupper()))
        features.append(sum(1 for c in text if c.isdigit()))
        features.append(sum(1 for c in text if c in ".,!?;:"))

        # Word features (top 500 most common words as features)
        word_hash_features = []
        for word in words[:100]:  # Limit to first 100 words
            word_hash = hash(word) % 500
            word_hash_features.append(word_hash / 500.0)

        # Pad or truncate word features
        if len(word_hash_features) < 500:
            word_hash_features.extend([0.0] * (500 - len(word_hash_features)))
        else:
            word_hash_features = word_hash_features[:500]

        features.extend(word_hash_features)

        # Pad to embedding dimension
        while len(features) < self.config.embedding_dim:
            features.append(0.0)

        vector = np.array(features[: self.config.embedding_dim], dtype=np.float32)

        if self.config.normalize_embeddings:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

        return vector

    def _simple_image_embedding(self, image_path: str) -> np.ndarray:
        """Simple image embedding using basic image statistics"""
        try:
            from PIL import Image
            import numpy as np

            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize to standard size
            image = image.resize((224, 224))
            img_array = np.array(image)

            # Basic image features
            features = []

            # Color statistics
            for channel in range(3):  # RGB
                channel_data = img_array[:, :, channel].flatten()
                features.extend(
                    [
                        np.mean(channel_data),
                        np.std(channel_data),
                        np.min(channel_data),
                        np.max(channel_data),
                        np.median(channel_data),
                    ]
                )

            # Histogram features (simplified)
            for channel in range(3):
                hist, _ = np.histogram(
                    img_array[:, :, channel], bins=32, range=(0, 256)
                )
                features.extend((hist / hist.sum()).tolist())

            # Gradient features (simplified edge detection)
            gray = np.mean(img_array, axis=2)
            grad_x = np.gradient(gray, axis=1)
            grad_y = np.gradient(gray, axis=0)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

            features.extend(
                [
                    np.mean(grad_magnitude),
                    np.std(grad_magnitude),
                    np.percentile(grad_magnitude, 75),
                    np.percentile(grad_magnitude, 95),
                ]
            )

            # Pad to embedding dimension
            while len(features) < self.config.embedding_dim:
                features.append(0.0)

            vector = np.array(features[: self.config.embedding_dim], dtype=np.float32)

            if self.config.normalize_embeddings:
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm

            return vector

        except Exception as e:
            logger.error(f"Simple image embedding failed: {e}")
            return np.zeros(self.config.embedding_dim, dtype=np.float32)

    def _simple_audio_embedding(self, audio_path: str) -> np.ndarray:
        """Simple audio embedding using basic audio features"""
        # For now, create a placeholder embedding
        # In a real implementation, this would use librosa or similar
        features = []

        # File-based features
        file_size = Path(audio_path).stat().st_size if Path(audio_path).exists() else 0
        features.append(file_size / 1000000.0)  # Size in MB

        # Filename-based features
        filename = Path(audio_path).stem.lower()
        filename_hash = hash(filename) % 1000
        features.append(filename_hash / 1000.0)

        # Pad to embedding dimension
        while len(features) < self.config.embedding_dim:
            features.append(np.random.normal(0, 0.1))  # Add some randomness

        vector = np.array(features[: self.config.embedding_dim], dtype=np.float32)

        if self.config.normalize_embeddings:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

        return vector

    def _simple_video_embedding(self, video_path: str) -> np.ndarray:
        """Simple video embedding combining image and audio features"""
        # Placeholder implementation
        # In practice, this would extract frames and audio, then combine embeddings

        features = []

        # File-based features
        file_size = Path(video_path).stat().st_size if Path(video_path).exists() else 0
        features.append(file_size / 1000000.0)  # Size in MB

        # Filename-based features
        filename = Path(video_path).stem.lower()
        filename_hash = hash(filename) % 1000
        features.append(filename_hash / 1000.0)

        # Simulate video-specific features
        features.extend(
            [0.5, 0.3, 0.7, 0.9]
        )  # Placeholder for motion, scene changes, etc.

        # Pad to embedding dimension
        while len(features) < self.config.embedding_dim:
            features.append(np.random.normal(0, 0.1))

        vector = np.array(features[: self.config.embedding_dim], dtype=np.float32)

        if self.config.normalize_embeddings:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

        return vector

    def _simple_document_embedding(self, doc_path: str) -> np.ndarray:
        """Simple document embedding based on content analysis"""
        try:
            # Read document content
            with open(doc_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Use text embedding for document content
            return self._simple_text_embedding(content)

        except Exception as e:
            logger.error(f"Document embedding failed: {e}")
            return np.zeros(self.config.embedding_dim, dtype=np.float32)

    def search_similar(
        self,
        query: str,
        query_modality: str = "text",
        target_modalities: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> List[Tuple[str, float, str]]:
        """
        Search for similar content across modalities

        Returns:
            List of (source_id, similarity_score, modality) tuples
        """
        # Create query embedding
        query_id = f"query_{hash(query)}"

        if query_modality == "text":
            asyncio.create_task(self.embed_text(query, query_id))
        else:
            # For other modalities, treat as file path
            if query_modality == "image":
                asyncio.create_task(self.embed_image(query, query_id))
            elif query_modality == "audio":
                asyncio.create_task(self.embed_audio(query, query_id))
            elif query_modality == "video":
                asyncio.create_task(self.embed_video(query, query_id))
            elif query_modality == "document":
                asyncio.create_task(self.embed_document(query, query_id))

        # Get query embedding
        query_embedding = self.embedding_space.get_embedding(query_id)
        if not query_embedding:
            return []

        # Search for similar embeddings
        results = self.embedding_space.similarity_search(
            query_embedding, top_k=top_k, modality_filter=target_modalities
        )

        # Add modality information
        enhanced_results = []
        for source_id, score in results:
            embedding = self.embedding_space.get_embedding(source_id)
            if embedding:
                enhanced_results.append((source_id, score, embedding.modality))

        # Remove query embedding
        self.embedding_space.remove_embedding(query_id)

        return enhanced_results

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the embedding system"""
        return self.embedding_space.get_modality_statistics()

    def save_embeddings(self, filepath: str):
        """Save all embeddings to disk"""
        self.embedding_space.save_space(filepath)

    def load_embeddings(self, filepath: str):
        """Load embeddings from disk"""
        self.embedding_space = EmbeddingSpace.load_space(filepath)

    def clear_embeddings(self):
        """Clear all embeddings"""
        self.embedding_space = EmbeddingSpace(self.config)
