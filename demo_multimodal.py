#!/usr/bin/env python3
"""
Multi-Modal Cross-Search Demo
============================

Demonstration of Vega2.0's multi-modal search capabilities across text, image,
audio, video, and document content using the unified cross-modal search engine.

This demo showcases:
- Indexing content from different modalities
- Cross-modal similarity search
- Unified embedding generation
- Search result ranking and explanation
"""

import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import List

# Import the multi-modal components
from src.vega.multimodal import (
    CrossModalSearchEngine,
    MultiModalEmbeddings,
    UnifiedRetrieval,
    VisionLanguageModel,
    CLIPIntegration,
)
from src.vega.multimodal.cross_modal_search import (
    ContentItem,
    SearchQuery,
    ModalityType,
    SimilarityMetric,
)
from src.vega.multimodal.embeddings import EmbeddingConfig, EmbeddingModel
from src.vega.multimodal.retrieval import RetrievalConfig, RetrievalStrategy

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def create_sample_content() -> List[ContentItem]:
    """Create sample content items for demonstration"""
    content_items = []

    # Sample text content
    text_items = [
        ContentItem(
            id="text_001",
            modality=ModalityType.TEXT,
            text_content="A beautiful mountain landscape with snow-capped peaks and a crystal-clear lake reflecting the sky. The scene is peaceful and serene.",
            metadata={"category": "nature", "author": "demo"},
        ),
        ContentItem(
            id="text_002",
            modality=ModalityType.TEXT,
            text_content="Advanced machine learning algorithms for computer vision and natural language processing. Deep neural networks enable sophisticated AI applications.",
            metadata={"category": "technology", "author": "demo"},
        ),
        ContentItem(
            id="text_003",
            modality=ModalityType.TEXT,
            text_content="Classical music concert featuring Mozart's Symphony No. 40. The orchestra performed brilliantly with exceptional violin and piano solos.",
            metadata={"category": "music", "author": "demo"},
        ),
    ]

    # Sample document content (simulated)
    doc_items = [
        ContentItem(
            id="doc_001",
            modality=ModalityType.DOCUMENT,
            text_content="Research Paper: Computer Vision in Autonomous Vehicles\n\nAbstract: This paper explores the application of deep learning techniques in autonomous vehicle navigation systems. We present novel approaches for object detection, lane recognition, and traffic sign classification using convolutional neural networks.",
            metadata={"type": "research_paper", "field": "computer_vision"},
        ),
        ContentItem(
            id="doc_002",
            modality=ModalityType.DOCUMENT,
            text_content="Travel Guide: Mountain Hiking Adventures\n\nExplore breathtaking mountain trails with stunning panoramic views. Our comprehensive guide covers the best hiking routes, essential gear, and safety tips for mountain enthusiasts.",
            metadata={"type": "travel_guide", "category": "outdoor"},
        ),
    ]

    # Sample image content (simulated - no actual image files)
    image_items = [
        ContentItem(
            id="img_001",
            modality=ModalityType.IMAGE,
            metadata={
                "description": "Mountain landscape photograph",
                "width": 1920,
                "height": 1080,
                "format": "JPG",
                "tags": ["mountain", "landscape", "nature", "snow"],
            },
            extracted_features={
                "width": 1920,
                "height": 1080,
                "keywords": ["mountain", "landscape", "snow", "peak", "nature"],
            },
        ),
        ContentItem(
            id="img_002",
            modality=ModalityType.IMAGE,
            metadata={
                "description": "Technology conference presentation",
                "width": 1280,
                "height": 720,
                "format": "PNG",
                "tags": ["technology", "presentation", "AI", "conference"],
            },
            extracted_features={
                "width": 1280,
                "height": 720,
                "keywords": ["technology", "AI", "machine", "learning", "conference"],
            },
        ),
    ]

    # Sample audio content (simulated)
    audio_items = [
        ContentItem(
            id="audio_001",
            modality=ModalityType.AUDIO,
            metadata={
                "description": "Classical music recording - Mozart Symphony",
                "duration": 1800,  # 30 minutes
                "format": "MP3",
                "genre": "classical",
            },
            extracted_features={
                "keywords": ["classical", "music", "mozart", "symphony", "orchestra"]
            },
        ),
        ContentItem(
            id="audio_002",
            modality=ModalityType.AUDIO,
            metadata={
                "description": "Nature sounds - Mountain stream",
                "duration": 600,  # 10 minutes
                "format": "WAV",
                "genre": "ambient",
            },
            extracted_features={
                "keywords": ["nature", "water", "stream", "mountain", "peaceful"]
            },
        ),
    ]

    # Sample video content (simulated)
    video_items = [
        ContentItem(
            id="video_001",
            modality=ModalityType.VIDEO,
            metadata={
                "description": "Mountain hiking documentary",
                "duration": 3600,  # 1 hour
                "format": "MP4",
                "resolution": "1080p",
            },
            extracted_features={
                "keywords": ["mountain", "hiking", "documentary", "nature", "adventure"]
            },
        ),
        ContentItem(
            id="video_002",
            modality=ModalityType.VIDEO,
            metadata={
                "description": "AI technology presentation",
                "duration": 1200,  # 20 minutes
                "format": "MP4",
                "resolution": "720p",
            },
            extracted_features={
                "keywords": ["AI", "technology", "presentation", "machine", "learning"]
            },
        ),
    ]

    content_items.extend(text_items)
    content_items.extend(doc_items)
    content_items.extend(image_items)
    content_items.extend(audio_items)
    content_items.extend(video_items)

    return content_items


async def demo_cross_modal_search():
    """Demonstrate cross-modal search capabilities"""
    print("🚀 Starting Multi-Modal Cross-Search Demo")
    print("=" * 50)

    # Initialize the search engine
    search_engine = CrossModalSearchEngine(embedding_dim=512)

    # Create and index sample content
    print("\n📥 Creating and indexing sample content...")
    content_items = await create_sample_content()

    # Index all content
    indexing_results = await search_engine.batch_index(content_items)
    successful_indexes = sum(1 for success in indexing_results.values() if success)
    print(
        f"✅ Successfully indexed {successful_indexes}/{len(content_items)} content items"
    )

    # Display search engine statistics
    stats = search_engine.get_stats()
    print(f"\n📊 Search Engine Statistics:")
    print(f"   Total Items: {stats['total_items']}")
    print(f"   Modality Distribution: {stats['modality_distribution']}")
    print(f"   Embedding Dimension: {stats['embedding_dimension']}")

    # Demo search scenarios
    search_scenarios = [
        {
            "name": "Text → All Modalities",
            "query": "beautiful mountain landscape with snow",
            "query_modality": ModalityType.TEXT,
            "target_modalities": [
                ModalityType.TEXT,
                ModalityType.IMAGE,
                ModalityType.VIDEO,
                ModalityType.AUDIO,
                ModalityType.DOCUMENT,
            ],
        },
        {
            "name": "Text → Technology Content",
            "query": "machine learning artificial intelligence",
            "query_modality": ModalityType.TEXT,
            "target_modalities": [
                ModalityType.TEXT,
                ModalityType.DOCUMENT,
                ModalityType.IMAGE,
                ModalityType.VIDEO,
            ],
        },
        {
            "name": "Text → Audio/Video",
            "query": "classical music symphony orchestra",
            "query_modality": ModalityType.TEXT,
            "target_modalities": [
                ModalityType.AUDIO,
                ModalityType.VIDEO,
                ModalityType.TEXT,
            ],
        },
        {
            "name": "Cross-Modal Nature Search",
            "query": "nature peaceful mountain water",
            "query_modality": ModalityType.TEXT,
            "target_modalities": [
                ModalityType.IMAGE,
                ModalityType.AUDIO,
                ModalityType.VIDEO,
                ModalityType.DOCUMENT,
            ],
        },
    ]

    print(f"\n🔍 Running Cross-Modal Search Scenarios")
    print("=" * 50)

    for i, scenario in enumerate(search_scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   Query: '{scenario['query']}'")
        print(f"   Query Modality: {scenario['query_modality'].value}")
        print(
            f"   Target Modalities: {[m.value for m in scenario['target_modalities']]}"
        )

        # Create search query
        search_query = SearchQuery(
            query=scenario["query"],
            query_modality=scenario["query_modality"],
            target_modalities=scenario["target_modalities"],
            similarity_metric=SimilarityMetric.COSINE,
            max_results=5,
            threshold=0.1,  # Lower threshold for demo
        )

        # Perform search
        results = await search_engine.search(search_query)

        print(f"   📋 Found {len(results)} results:")

        for j, result in enumerate(results[:3], 1):  # Show top 3 results
            print(f"      {j}. [{result.item.modality.value.upper()}] {result.item.id}")
            print(f"         Similarity: {result.similarity_score:.3f}")
            print(f"         Explanation: {result.explanation}")
            if result.matched_features:
                print(
                    f"         Matched Features: {', '.join(result.matched_features[:3])}"
                )
            print()


async def demo_multimodal_embeddings():
    """Demonstrate multi-modal embeddings system"""
    print("\n🎯 Multi-Modal Embeddings Demo")
    print("=" * 30)

    # Initialize embeddings system
    config = EmbeddingConfig(
        model_type=EmbeddingModel.SIMPLE, embedding_dim=256, normalize_embeddings=True
    )

    embeddings = MultiModalEmbeddings(config)

    # Create sample embeddings
    sample_texts = [
        "mountain landscape photography",
        "artificial intelligence research",
        "classical music performance",
    ]

    print("📝 Creating text embeddings...")
    for i, text in enumerate(sample_texts):
        embedding = await embeddings.embed_text(text, f"text_{i}")
        print(f"   ✅ Created embedding for: '{text[:30]}...'")

    # Demonstrate similarity search
    print("\n🔍 Similarity Search Demo:")
    search_results = embeddings.search_similar(
        query="beautiful nature photography",
        query_modality="text",
        target_modalities=["text"],
        top_k=3,
    )

    print("   Query: 'beautiful nature photography'")
    for source_id, score, modality in search_results:
        print(f"   ✅ {source_id} (similarity: {score:.3f})")

    # Display statistics
    stats = embeddings.get_statistics()
    print(f"\n📊 Embeddings Statistics: {stats}")


async def demo_unified_retrieval():
    """Demonstrate unified retrieval system"""
    print("\n🎯 Unified Retrieval System Demo")
    print("=" * 35)

    # Initialize retrieval system
    config = RetrievalConfig(
        strategy=RetrievalStrategy.HYBRID,
        max_results=5,
        similarity_threshold=0.2,
        enable_query_expansion=True,
        enable_reranking=True,
    )

    retrieval = UnifiedRetrieval(config)

    print("⚙️  Initialized Unified Retrieval System")
    print(f"   Strategy: {config.strategy.value}")
    print(f"   Max Results: {config.max_results}")
    print(f"   Query Expansion: {config.enable_query_expansion}")
    print(f"   Re-ranking: {config.enable_reranking}")

    # Display system statistics
    stats = retrieval.get_statistics()
    print(f"\n📊 Retrieval System Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_value in value.items():
                print(f"     {sub_key}: {sub_value}")
        else:
            print(f"   {key}: {value}")


async def main():
    """Main demo function"""
    print("🌟 Vega2.0 Multi-Modal Integration Demo")
    print("=" * 60)
    print("Phase 4: Cross-Modal Search & Retrieval")
    print(f"Demo Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    try:
        # Run cross-modal search demo
        await demo_cross_modal_search()

        # Run embeddings demo
        await demo_multimodal_embeddings()

        # Run retrieval demo
        await demo_unified_retrieval()

        print("\n" + "=" * 60)
        print("✨ Multi-Modal Integration Demo Completed Successfully!")
        print("🎉 Phase 4 foundational components are working correctly.")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logger.exception("Demo execution failed")


if __name__ == "__main__":
    asyncio.run(main())
