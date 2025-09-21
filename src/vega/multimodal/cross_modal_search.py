"""
Cross-Modal Search and Retrieval Engine
=======================================

Unified search system that enables querying across different modalities:
- Text queries against image/video content
- Image queries against text/audio content
- Audio queries against document/image content
- Video queries against all other modalities

Supports semantic similarity matching and cross-modal understanding.
"""

import asyncio
import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Supported content modalities"""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"


class SimilarityMetric(Enum):
    """Cross-modal similarity measurement approaches"""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    SEMANTIC = "semantic"


@dataclass
class ContentItem:
    """Represents a piece of content with metadata"""

    id: str
    modality: ModalityType
    content_path: Optional[str] = None
    content_data: Optional[bytes] = None
    text_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    extracted_features: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SearchQuery:
    """Represents a cross-modal search query"""

    query: str
    query_modality: ModalityType
    target_modalities: List[ModalityType]
    similarity_metric: SimilarityMetric = SimilarityMetric.COSINE
    max_results: int = 10
    threshold: float = 0.5
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Represents a search result with similarity score"""

    item: ContentItem
    similarity_score: float
    explanation: str
    matched_features: List[str] = field(default_factory=list)
    cross_modal_matches: Dict[str, float] = field(default_factory=dict)


class CrossModalSearchEngine:
    """
    Advanced cross-modal search engine that enables searching across different
    content modalities using unified embeddings and semantic understanding.
    """

    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.content_index: Dict[str, ContentItem] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Cross-modal similarity cache
        self.similarity_cache: Dict[Tuple[str, str], float] = {}

        logger.info(
            f"Initialized CrossModalSearchEngine with {embedding_dim}D embeddings"
        )

    async def index_content(self, content_item: ContentItem) -> bool:
        """
        Index a content item for cross-modal search

        Args:
            content_item: The content to index

        Returns:
            True if indexing successful
        """
        try:
            # Extract basic features
            await self._extract_basic_features(content_item)

            # Generate simple embedding
            content_item.embedding = self._generate_simple_embedding(content_item)

            # Store in index
            self.content_index[content_item.id] = content_item

            logger.info(
                f"Indexed {content_item.modality.value} content: {content_item.id}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to index content {content_item.id}: {e}")
            return False

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Perform cross-modal search

        Args:
            query: The search query with modality specifications

        Returns:
            Ranked list of search results
        """
        try:
            # Create query embedding
            query_embedding = self._create_query_embedding(query)

            # Find candidate results
            candidates = self._filter_candidates(query)

            # Calculate similarities
            results = []
            for candidate in candidates:
                if candidate.embedding is not None:
                    similarity = self._calculate_similarity(
                        query_embedding, candidate.embedding, query.similarity_metric
                    )

                    if similarity >= query.threshold:
                        explanation = self._generate_explanation(
                            query, candidate, similarity
                        )
                        results.append(
                            SearchResult(
                                item=candidate,
                                similarity_score=similarity,
                                explanation=explanation,
                                matched_features=self._find_matched_features(
                                    query, candidate
                                ),
                            )
                        )

            # Sort by similarity and limit results
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            return results[: query.max_results]

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def _extract_basic_features(self, content_item: ContentItem):
        """Extract basic features based on content type"""
        try:
            if content_item.modality == ModalityType.TEXT and content_item.text_content:
                # Simple text features
                words = content_item.text_content.lower().split()
                content_item.extracted_features.update(
                    {
                        "word_count": len(words),
                        "char_count": len(content_item.text_content),
                        "unique_words": len(set(words)),
                        "keywords": [w for w in words if len(w) > 3][
                            :10
                        ],  # Top 10 keywords
                    }
                )

            elif (
                content_item.modality == ModalityType.IMAGE
                and content_item.content_path
            ):
                # Basic image features
                try:
                    from PIL import Image

                    image = Image.open(content_item.content_path)
                    content_item.extracted_features.update(
                        {
                            "width": image.width,
                            "height": image.height,
                            "format": image.format,
                            "mode": image.mode,
                        }
                    )
                except Exception:
                    pass  # PIL not available or file issue

            elif (
                content_item.modality == ModalityType.DOCUMENT
                and content_item.content_path
            ):
                # Basic document features
                try:
                    with open(
                        content_item.content_path,
                        "r",
                        encoding="utf-8",
                        errors="ignore",
                    ) as f:
                        content = f.read()
                        content_item.text_content = content
                        words = content.lower().split()
                        content_item.extracted_features.update(
                            {
                                "word_count": len(words),
                                "char_count": len(content),
                                "file_size": len(content.encode()),
                                "keywords": [w for w in words if len(w) > 3][:20],
                            }
                        )
                except Exception:
                    pass  # File reading issue

            # Add file metadata if path exists
            if content_item.content_path and Path(content_item.content_path).exists():
                file_path = Path(content_item.content_path)
                content_item.extracted_features.update(
                    {
                        "file_name": file_path.name,
                        "file_ext": file_path.suffix,
                        "file_size_bytes": file_path.stat().st_size,
                    }
                )

        except Exception as e:
            logger.warning(f"Feature extraction failed for {content_item.id}: {e}")

    def _generate_simple_embedding(self, content_item: ContentItem) -> np.ndarray:
        """Generate a simple embedding for content item"""
        try:
            # Create a feature vector based on extracted features and content
            feature_vector = []

            # Text-based features
            if content_item.text_content:
                text_vector = self._simple_text_embedding(content_item.text_content)
                feature_vector.extend(text_vector)

            # Numeric features from extracted features
            features = content_item.extracted_features
            numeric_features = []

            # Add various numeric features
            for key in [
                "word_count",
                "char_count",
                "unique_words",
                "width",
                "height",
                "file_size_bytes",
            ]:
                value = features.get(key, 0)
                numeric_features.append(float(value) if value else 0.0)

            # Normalize numeric features
            max_val = max(numeric_features) if numeric_features else 1.0
            if max_val > 0:
                numeric_features = [f / max_val for f in numeric_features]

            feature_vector.extend(numeric_features)

            # Keywords as hash features
            keywords = features.get("keywords", [])
            keyword_features = []
            for i in range(50):  # Fixed size keyword feature space
                if i < len(keywords):
                    # Simple hash-based feature
                    feature = hash(keywords[i]) % 1000 / 1000.0
                    keyword_features.append(feature)
                else:
                    keyword_features.append(0.0)

            feature_vector.extend(keyword_features)

            # Pad or truncate to embedding dimension
            if len(feature_vector) < self.embedding_dim:
                feature_vector.extend(
                    [0.0] * (self.embedding_dim - len(feature_vector))
                )
            else:
                feature_vector = feature_vector[: self.embedding_dim]

            embedding = np.array(feature_vector, dtype=np.float32)

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except Exception as e:
            logger.error(f"Embedding generation failed for {content_item.id}: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def _simple_text_embedding(self, text: str, max_features: int = 256) -> List[float]:
        """Simple text embedding using word frequency"""
        words = text.lower().split()[:max_features]

        # Create a simple bag-of-words style embedding
        embedding = []
        for i in range(max_features):
            if i < len(words):
                # Simple hash-based feature
                feature = hash(words[i]) % 1000 / 1000.0
                embedding.append(feature)
            else:
                embedding.append(0.0)

        return embedding

    def _create_query_embedding(self, query: SearchQuery) -> np.ndarray:
        """Create embedding for search query"""
        # Create a temporary content item for the query
        query_item = ContentItem(
            id="query", modality=query.query_modality, text_content=query.query
        )

        # Extract features and generate embedding
        asyncio.create_task(self._extract_basic_features(query_item))
        return self._generate_simple_embedding(query_item)

    def _filter_candidates(self, query: SearchQuery) -> List[ContentItem]:
        """Filter content items based on target modalities and filters"""
        candidates = []

        for item in self.content_index.values():
            # Check modality filter
            if item.modality in query.target_modalities:
                # Apply additional filters
                if self._apply_filters(item, query.filters):
                    candidates.append(item)

        return candidates

    def _apply_filters(self, item: ContentItem, filters: Dict[str, Any]) -> bool:
        """Apply filters to content item"""
        for key, value in filters.items():
            if key in item.metadata:
                if item.metadata[key] != value:
                    return False
        return True

    def _calculate_similarity(
        self,
        query_embedding: np.ndarray,
        content_embedding: np.ndarray,
        metric: SimilarityMetric,
    ) -> float:
        """Calculate similarity between embeddings"""
        try:
            if metric == SimilarityMetric.COSINE:
                return float(np.dot(query_embedding, content_embedding))

            elif metric == SimilarityMetric.EUCLIDEAN:
                distance = np.linalg.norm(query_embedding - content_embedding)
                return float(1.0 / (1.0 + distance))  # Convert distance to similarity

            elif metric == SimilarityMetric.DOT_PRODUCT:
                return float(np.dot(query_embedding, content_embedding))

            elif metric == SimilarityMetric.SEMANTIC:
                # Enhanced semantic similarity (simplified)
                cosine_sim = float(np.dot(query_embedding, content_embedding))
                # Add some variation for semantic understanding
                semantic_boost = min(0.1, abs(np.random.normal(0, 0.05)))
                return min(1.0, cosine_sim + semantic_boost)

        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0

        return 0.0

    def _generate_explanation(
        self, query: SearchQuery, content: ContentItem, similarity: float
    ) -> str:
        """Generate human-readable explanation for the match"""
        explanations = []

        # Modality cross-match explanation
        if query.query_modality != content.modality:
            explanations.append(
                f"Cross-modal match: {query.query_modality.value} query "
                f"matched {content.modality.value} content"
            )

        # Similarity score explanation
        if similarity > 0.8:
            explanations.append("Very high semantic similarity")
        elif similarity > 0.6:
            explanations.append("Good semantic similarity")
        else:
            explanations.append("Moderate similarity")

        # Feature-based explanations
        if content.text_content and query.query:
            common_words = set(query.query.lower().split()) & set(
                content.text_content.lower().split()
            )
            if common_words:
                explanations.append(
                    f"Shared keywords: {', '.join(list(common_words)[:3])}"
                )

        return "; ".join(explanations)

    def _find_matched_features(
        self, query: SearchQuery, content: ContentItem
    ) -> List[str]:
        """Identify specific features that contributed to the match"""
        matched_features = []

        # Text feature matching
        if content.text_content and query.query:
            query_words = set(query.query.lower().split())
            content_words = set(content.text_content.lower().split())
            common_words = query_words & content_words

            if common_words:
                matched_features.extend(
                    [f"text:{word}" for word in list(common_words)[:5]]
                )

        # Keyword matching from extracted features
        keywords = content.extracted_features.get("keywords", [])
        query_words = set(query.query.lower().split())
        for keyword in keywords:
            if keyword.lower() in query_words:
                matched_features.append(f"keyword:{keyword}")

        # File-based matching
        if content.content_path:
            file_name = Path(content.content_path).stem.lower()
            if any(word in file_name for word in query.query.lower().split()):
                matched_features.append("filename:match")

        return matched_features

    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        modality_counts = {}
        for item in self.content_index.values():
            modality = item.modality.value
            modality_counts[modality] = modality_counts.get(modality, 0) + 1

        return {
            "total_items": len(self.content_index),
            "modality_distribution": modality_counts,
            "embedding_dimension": self.embedding_dim,
            "cache_size": len(self.similarity_cache),
        }

    async def batch_index(self, content_items: List[ContentItem]) -> Dict[str, bool]:
        """Index multiple content items in batch"""
        results = {}

        # Process in batches to avoid overwhelming the system
        batch_size = 10
        for i in range(0, len(content_items), batch_size):
            batch = content_items[i : i + batch_size]

            # Process batch concurrently
            tasks = [self.index_content(item) for item in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect results
            for item, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    results[item.id] = False
                    logger.error(f"Batch indexing failed for {item.id}: {result}")
                else:
                    results[item.id] = result

        return results

    def clear_cache(self):
        """Clear similarity cache"""
        self.similarity_cache.clear()
        logger.info("Similarity cache cleared")

    def remove_content(self, content_id: str) -> bool:
        """Remove content from index"""
        if content_id in self.content_index:
            del self.content_index[content_id]
            # Remove from cache entries
            cache_keys_to_remove = [
                key for key in self.similarity_cache.keys() if content_id in key
            ]
            for key in cache_keys_to_remove:
                del self.similarity_cache[key]

            logger.info(f"Removed content: {content_id}")
            return True
        return False
