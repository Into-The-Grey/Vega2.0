"""
Vega2.0 Multi-Modal Integration
==============================

Cross-modal search, retrieval, and unified embedding system for text, image, audio, and video content.
This module provides the foundation for Phase 4 of the Multi-Modal Support roadmap.

Enhanced Features:
- Advanced CLIP model integration with multiple variants
- Enhanced zero-shot classification with confidence scoring
- Cross-modal retrieval with sophisticated ranking
- Multi-scale image processing and analysis
- Vector database integration (FAISS/Pinecone)
- Performance monitoring and optimization
"""

from .cross_modal_search import CrossModalSearchEngine, SearchResult, ModalityType
from .embeddings import MultiModalEmbeddings, EmbeddingSpace
from .retrieval import UnifiedRetrieval, RetrievalConfig
from .vision_language import VisionLanguageModel, CLIPIntegration
from .clip_advanced import (
    AdvancedCLIPIntegration,
    CLIPEnhancedConfig,
    CLIPModelType,
    ProcessingMode,
    EnhancedClassificationResult,
    RetrievalResult,
    create_advanced_clip_integration,
)

from .vector_database import (
    UnifiedVectorDB,
    VectorDBConfig,
    VectorDBType,
    IndexType,
    DistanceMetric,
    VectorRecord,
    SearchQuery as VectorSearchQuery,
    FAISSVectorDB,
    PineconeVectorDB,
    create_faiss_db,
    create_pinecone_db,
)

__all__ = [
    "CrossModalSearchEngine",
    "SearchResult",
    "ModalityType",
    "MultiModalEmbeddings",
    "EmbeddingSpace",
    "UnifiedRetrieval",
    "RetrievalConfig",
    "VisionLanguageModel",
    "CLIPIntegration",
    # Advanced CLIP integration
    "AdvancedCLIPIntegration",
    "CLIPEnhancedConfig",
    "CLIPModelType",
    "ProcessingMode",
    "EnhancedClassificationResult",
    "RetrievalResult",
    "create_advanced_clip_integration",
    # Vector database
    "UnifiedVectorDB",
    "VectorDBConfig",
    "VectorDBType",
    "IndexType",
    "DistanceMetric",
    "VectorRecord",
    "VectorSearchQuery",
    "FAISSVectorDB",
    "PineconeVectorDB",
    "create_faiss_db",
    "create_pinecone_db",
]
