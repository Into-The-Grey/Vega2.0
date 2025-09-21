"""
Vega2.0 Multi-Modal Integration
==============================

Cross-modal search, retrieval, and unified embedding system for text, image, audio, and video content.
This module provides the foundation for Phase 4 of the Multi-Modal Support roadmap.
"""

from .cross_modal_search import CrossModalSearchEngine, SearchResult, ModalityType
from .embeddings import MultiModalEmbeddings, EmbeddingSpace
from .retrieval import UnifiedRetrieval, RetrievalConfig
from .vision_language import VisionLanguageModel, CLIPIntegration

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
]
