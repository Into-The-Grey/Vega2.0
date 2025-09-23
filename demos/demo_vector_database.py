#!/usr/bin/env python3
"""
Vector Database Integration Demo
===============================

Demonstration of Vega2.0's vector database integration for large-scale
similarity search and retrieval across multi-modal embeddings.

This demo showcases:
- FAISS local vector database setup and operations
- Pinecone cloud vector database integration
- Unified vector database interface
- Batch operations and performance optimization
- Metadata filtering and hybrid search
- Cross-modal vector search and retrieval
"""

import asyncio
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import time

# Import vector database components
from src.vega.multimodal import (
    UnifiedVectorDB,
    VectorDBConfig,
    VectorDBType,
    IndexType,
    DistanceMetric,
    VectorRecord,
    VectorSearchQuery,
    create_faiss_db,
    create_pinecone_db,
)

# Import multimodal components for embedding generation
from src.vega.multimodal import (
    AdvancedCLIPIntegration,
    CLIPModelType,
    ProcessingMode,
    create_advanced_clip_integration,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def generate_sample_embeddings(
    clip_integration: AdvancedCLIPIntegration,
) -> List[VectorRecord]:
    """Generate sample embeddings for testing"""

    # Sample content for embedding generation
    sample_content = [
        {
            "id": "text_001",
            "type": "text",
            "content": "beautiful mountain landscape with snow-capped peaks",
            "metadata": {"category": "nature", "source": "description"},
        },
        {
            "id": "text_002",
            "type": "text",
            "content": "artificial intelligence and machine learning research",
            "metadata": {"category": "technology", "source": "description"},
        },
        {
            "id": "text_003",
            "type": "text",
            "content": "classical music symphony orchestra performance",
            "metadata": {"category": "music", "source": "description"},
        },
        {
            "id": "img_001",
            "type": "image",
            "content": "mountain_landscape.jpg",
            "metadata": {"category": "nature", "source": "image", "format": "jpg"},
        },
        {
            "id": "img_002",
            "type": "image",
            "content": "technology_conference.jpg",
            "metadata": {"category": "technology", "source": "image", "format": "jpg"},
        },
        {
            "id": "img_003",
            "type": "image",
            "content": "music_concert.jpg",
            "metadata": {"category": "music", "source": "image", "format": "jpg"},
        },
        {
            "id": "doc_001",
            "type": "document",
            "content": "Research paper on computer vision applications in autonomous vehicles",
            "metadata": {
                "category": "technology",
                "source": "document",
                "type": "research",
            },
        },
        {
            "id": "doc_002",
            "type": "document",
            "content": "Travel guide for mountain hiking and outdoor adventures",
            "metadata": {"category": "nature", "source": "document", "type": "guide"},
        },
        {
            "id": "audio_001",
            "type": "audio",
            "content": "classical_symphony.mp3",
            "metadata": {"category": "music", "source": "audio", "format": "mp3"},
        },
        {
            "id": "video_001",
            "type": "video",
            "content": "nature_documentary.mp4",
            "metadata": {"category": "nature", "source": "video", "format": "mp4"},
        },
    ]

    vector_records = []

    for item in sample_content:
        try:
            # Generate embedding based on content type
            if item["type"] == "text" or item["type"] == "document":
                analysis = await clip_integration.encode_text_advanced(item["content"])
                if analysis and analysis.text_embedding is not None:
                    embedding = analysis.text_embedding
                else:
                    # Fallback: generate consistent embedding based on content
                    seed = hash(item["content"]) % 10000
                    np.random.seed(seed)
                    embedding = np.random.normal(0, 1, 512).astype(np.float32)
                    embedding = embedding / np.linalg.norm(embedding)

            elif item["type"] in ["image", "audio", "video"]:
                analysis = await clip_integration.encode_image_advanced(item["content"])
                if analysis and analysis.image_embedding is not None:
                    embedding = analysis.image_embedding
                else:
                    # Fallback: generate consistent embedding based on content
                    seed = hash(item["content"]) % 10000
                    np.random.seed(seed)
                    embedding = np.random.normal(0, 1, 512).astype(np.float32)
                    embedding = embedding / np.linalg.norm(embedding)

            else:
                # Generate random but consistent embedding
                seed = hash(item["content"]) % 10000
                np.random.seed(seed)
                embedding = np.random.normal(0, 1, 512).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)

            # Create vector record
            record = VectorRecord(
                id=item["id"],
                vector=embedding,
                metadata=item["metadata"],
                modality=item["type"],
                source=item["content"],
            )

            vector_records.append(record)

        except Exception as e:
            logger.error(f"Failed to generate embedding for {item['id']}: {e}")

    return vector_records


async def demo_faiss_database():
    """Demonstrate FAISS vector database operations"""
    print("🔍 FAISS Vector Database Demo")
    print("=" * 30)

    # Test different FAISS configurations
    faiss_configs = [
        {
            "name": "Flat Index (Exact Search)",
            "config": VectorDBConfig(
                db_type=VectorDBType.FAISS,
                index_type=IndexType.FLAT,
                distance_metric=DistanceMetric.COSINE,
                dimension=512,
            ),
        },
        {
            "name": "IVF Flat Index (Fast Approximate)",
            "config": VectorDBConfig(
                db_type=VectorDBType.FAISS,
                index_type=IndexType.IVF_FLAT,
                distance_metric=DistanceMetric.COSINE,
                dimension=512,
                nlist=10,
                nprobe=3,
            ),
        },
        {
            "name": "HNSW Index (Memory Efficient)",
            "config": VectorDBConfig(
                db_type=VectorDBType.FAISS,
                index_type=IndexType.HNSW,
                distance_metric=DistanceMetric.COSINE,
                dimension=512,
            ),
        },
    ]

    # Initialize CLIP for embedding generation
    clip_integration = create_advanced_clip_integration(
        model_type=CLIPModelType.VIT_B_32, processing_mode=ProcessingMode.STANDARD
    )
    await clip_integration.initialize()

    # Generate sample embeddings
    print("📊 Generating sample embeddings...")
    vector_records = await generate_sample_embeddings(clip_integration)
    print(f"   Generated {len(vector_records)} vector records")

    for config_info in faiss_configs:
        print(f"\n🧪 Testing: {config_info['name']}")

        # Create database
        db = UnifiedVectorDB(config_info["config"])
        success = await db.initialize()

        if success:
            print("   ✅ Database initialized successfully")

            # Add vectors
            start_time = time.time()
            add_success = await db.add_vectors(vector_records)
            add_time = time.time() - start_time

            if add_success:
                print(f"   ✅ Added {len(vector_records)} vectors in {add_time:.3f}s")

                # Test searches
                await _test_vector_searches(db, clip_integration)

                # Get statistics
                stats = db.get_stats()
                print(f"   📈 Database stats:")
                print(f"      Total vectors: {stats.get('total_vectors', 0)}")
                print(f"      Index type: {stats.get('index_type', 'unknown')}")
                print(
                    f"      Average search time: {stats.get('avg_search_time', 0):.4f}s"
                )

            else:
                print("   ❌ Failed to add vectors")
        else:
            print("   ❌ Database initialization failed")

    await clip_integration.shutdown()


async def demo_pinecone_database():
    """Demonstrate Pinecone vector database operations"""
    print("\n☁️  Pinecone Vector Database Demo")
    print("=" * 35)

    # Note: This demo uses mock Pinecone since we don't have real API keys
    config = VectorDBConfig(
        db_type=VectorDBType.PINECONE,
        dimension=512,
        pinecone_api_key="demo-api-key",
        pinecone_index_name="vega-demo",
        pinecone_environment="us-west1-gcp",
    )

    # Initialize CLIP for embedding generation
    clip_integration = create_advanced_clip_integration(
        model_type=CLIPModelType.VIT_B_32, processing_mode=ProcessingMode.STANDARD
    )
    await clip_integration.initialize()

    # Generate sample embeddings
    print("📊 Generating sample embeddings...")
    vector_records = await generate_sample_embeddings(clip_integration)

    # Create Pinecone database
    db = UnifiedVectorDB(config)
    success = await db.initialize()

    if success:
        print("   ✅ Pinecone database initialized (mock)")

        # Add vectors
        start_time = time.time()
        add_success = await db.add_vectors(vector_records)
        add_time = time.time() - start_time

        if add_success:
            print(f"   ✅ Upserted {len(vector_records)} vectors in {add_time:.3f}s")

            # Test searches
            await _test_vector_searches(db, clip_integration)

            # Test metadata filtering
            await _test_metadata_filtering(db, clip_integration)

            # Get statistics
            stats = db.get_stats()
            print(f"   📈 Pinecone stats:")
            print(f"      Database type: {stats.get('db_type', 'unknown')}")
            print(f"      Index name: {stats.get('index_name', 'unknown')}")
            print(f"      Search count: {stats.get('search_count', 0)}")
            print(f"      Average search time: {stats.get('avg_search_time', 0):.4f}s")

        else:
            print("   ❌ Failed to upsert vectors")
    else:
        print("   ❌ Pinecone database initialization failed")

    await clip_integration.shutdown()


async def _test_vector_searches(
    db: UnifiedVectorDB, clip_integration: AdvancedCLIPIntegration
):
    """Test various vector search scenarios"""

    search_scenarios = [
        {
            "name": "Nature Content Search",
            "query": "mountain landscape nature scenery",
            "top_k": 3,
        },
        {
            "name": "Technology Content Search",
            "query": "artificial intelligence machine learning technology",
            "top_k": 3,
        },
        {
            "name": "Music Content Search",
            "query": "classical music symphony orchestra",
            "top_k": 3,
        },
    ]

    print(f"   🔍 Testing vector searches:")

    for scenario in search_scenarios:
        # Generate query embedding
        text_analysis = await clip_integration.encode_text_advanced(scenario["query"])
        if text_analysis and text_analysis.text_embedding is not None:
            query_vector = text_analysis.text_embedding
        else:
            # Fallback query vector
            seed = hash(scenario["query"]) % 1000
            np.random.seed(seed)
            query_vector = np.random.normal(0, 1, 512).astype(np.float32)
            query_vector = query_vector / np.linalg.norm(query_vector)

        # Create search query
        search_query = VectorSearchQuery(
            vector=query_vector,
            top_k=scenario["top_k"],
            include_metadata=True,
            threshold=0.1,
        )

        # Perform search
        start_time = time.time()
        results = await db.search(search_query)
        search_time = time.time() - start_time

        print(
            f"      📋 {scenario['name']}: {len(results)} results in {search_time:.3f}s"
        )

        for i, result in enumerate(results):
            modality = result.metadata.get("source", "unknown")
            category = result.metadata.get("category", "unknown")
            print(
                f"         {i+1}. {result.id} ({modality}/{category}) - Score: {result.score:.3f}"
            )


async def _test_metadata_filtering(
    db: UnifiedVectorDB, clip_integration: AdvancedCLIPIntegration
):
    """Test metadata filtering capabilities"""

    print(f"   🔽 Testing metadata filtering:")

    # Generate query embedding
    query_text = "search across all content"
    text_analysis = await clip_integration.encode_text_advanced(query_text)
    if text_analysis and text_analysis.text_embedding is not None:
        query_vector = text_analysis.text_embedding
    else:
        seed = hash(query_text) % 1000
        np.random.seed(seed)
        query_vector = np.random.normal(0, 1, 512).astype(np.float32)
        query_vector = query_vector / np.linalg.norm(query_vector)

    # Test different filters
    filters = [
        {"category": "nature"},
        {"category": "technology"},
        {"source": "image"},
        {"source": "document"},
    ]

    for filter_dict in filters:
        search_query = VectorSearchQuery(
            vector=query_vector, top_k=5, filters=filter_dict, include_metadata=True
        )

        results = await db.search(search_query)
        filter_str = ", ".join([f"{k}={v}" for k, v in filter_dict.items()])
        print(f"      🔍 Filter [{filter_str}]: {len(results)} results")

        for result in results[:2]:  # Show top 2
            category = result.metadata.get("category", "unknown")
            source = result.metadata.get("source", "unknown")
            print(
                f"         {result.id} ({source}/{category}) - Score: {result.score:.3f}"
            )


async def demo_batch_operations():
    """Demonstrate batch operations for performance"""
    print("\n⚡ Batch Operations Demo")
    print("=" * 25)

    # Create larger dataset for batch testing
    print("📊 Generating large dataset...")

    # Initialize CLIP
    clip_integration = create_advanced_clip_integration(
        model_type=CLIPModelType.VIT_B_32, batch_size=32
    )
    await clip_integration.initialize()

    # Generate larger dataset
    large_dataset = []
    categories = ["nature", "technology", "music", "art", "sports"]
    modalities = ["text", "image", "document", "audio", "video"]

    for i in range(100):  # Generate 100 records
        category = categories[i % len(categories)]
        modality = modalities[i % len(modalities)]

        # Generate consistent content based on index
        content = f"{category} {modality} content example {i:03d}"

        # Generate embedding
        seed = hash(f"{content}_{i}") % 10000
        np.random.seed(seed)
        embedding = np.random.normal(0, 1, 512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        record = VectorRecord(
            id=f"{modality}_{i:03d}",
            vector=embedding,
            metadata={"category": category, "modality": modality, "index": i},
            modality=modality,
            source=content,
        )

        large_dataset.append(record)

    print(f"   Generated {len(large_dataset)} records")

    # Test with FAISS
    print("\n🔧 Testing FAISS batch operations:")

    config = VectorDBConfig(
        db_type=VectorDBType.FAISS,
        index_type=IndexType.IVF_FLAT,
        distance_metric=DistanceMetric.COSINE,
        dimension=512,
        batch_size=50,
        nlist=20,
    )

    db = UnifiedVectorDB(config)
    await db.initialize()

    # Batch add
    start_time = time.time()
    success = await db.add_vectors(large_dataset)
    add_time = time.time() - start_time

    if success:
        print(f"   ✅ Batch added {len(large_dataset)} vectors in {add_time:.3f}s")
        print(f"   ⚡ Rate: {len(large_dataset)/add_time:.1f} vectors/second")

        # Batch search test
        print("\n   🔍 Batch search performance:")

        # Generate multiple queries
        query_texts = [
            "nature landscape mountain",
            "technology computer AI",
            "music classical symphony",
            "art painting creative",
            "sports athletic competition",
        ]

        total_search_time = 0
        total_results = 0

        for query_text in query_texts:
            # Generate query vector
            seed = hash(query_text) % 1000
            np.random.seed(seed)
            query_vector = np.random.normal(0, 1, 512).astype(np.float32)
            query_vector = query_vector / np.linalg.norm(query_vector)

            search_query = VectorSearchQuery(
                vector=query_vector, top_k=10, include_metadata=True
            )

            search_start = time.time()
            results = await db.search(search_query)
            search_time = time.time() - search_start

            total_search_time += search_time
            total_results += len(results)

            print(
                f"      Query '{query_text[:20]}...': {len(results)} results in {search_time:.3f}s"
            )

        avg_search_time = total_search_time / len(query_texts)
        print(f"   📊 Average search time: {avg_search_time:.3f}s")
        print(f"   📊 Average results per query: {total_results/len(query_texts):.1f}")

        # Database statistics
        stats = db.get_stats()
        print(f"\n   📈 Final database statistics:")
        print(f"      Total vectors: {stats.get('total_vectors', 0)}")
        print(f"      Search operations: {stats.get('search_count', 0)}")
        print(f"      Add operations: {stats.get('add_count', 0)}")
        print(f"      Average search time: {stats.get('avg_search_time', 0):.4f}s")

    else:
        print("   ❌ Batch add failed")

    await clip_integration.shutdown()


async def demo_performance_comparison():
    """Compare performance between different vector database configurations"""
    print("\n🏁 Performance Comparison Demo")
    print("=" * 30)

    # Test configurations
    test_configs = [
        {
            "name": "FAISS Flat (Exact)",
            "config": VectorDBConfig(
                db_type=VectorDBType.FAISS,
                index_type=IndexType.FLAT,
                distance_metric=DistanceMetric.COSINE,
                dimension=512,
            ),
        },
        {
            "name": "FAISS IVF (Approximate)",
            "config": VectorDBConfig(
                db_type=VectorDBType.FAISS,
                index_type=IndexType.IVF_FLAT,
                distance_metric=DistanceMetric.COSINE,
                dimension=512,
                nlist=10,
                nprobe=3,
            ),
        },
        {
            "name": "Pinecone (Mock)",
            "config": VectorDBConfig(
                db_type=VectorDBType.PINECONE,
                dimension=512,
                pinecone_api_key="demo-key",
            ),
        },
    ]

    # Generate test dataset
    print("📊 Preparing test dataset...")
    test_size = 50
    test_vectors = []

    for i in range(test_size):
        seed = i * 42
        np.random.seed(seed)
        embedding = np.random.normal(0, 1, 512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        record = VectorRecord(
            id=f"test_vector_{i:03d}",
            vector=embedding,
            metadata={"index": i, "category": f"cat_{i % 5}"},
            modality="test",
            source=f"test_content_{i}",
        )
        test_vectors.append(record)

    # Test each configuration
    performance_results = []

    for config_info in test_configs:
        print(f"\n🧪 Testing: {config_info['name']}")

        try:
            db = UnifiedVectorDB(config_info["config"])
            success = await db.initialize()

            if success:
                # Measure add performance
                add_start = time.time()
                add_success = await db.add_vectors(test_vectors)
                add_time = time.time() - add_start

                if add_success:
                    # Measure search performance
                    search_times = []
                    for i in range(10):  # 10 search queries
                        np.random.seed(i)
                        query_vector = np.random.normal(0, 1, 512).astype(np.float32)
                        query_vector = query_vector / np.linalg.norm(query_vector)

                        search_query = VectorSearchQuery(
                            vector=query_vector, top_k=5, include_metadata=True
                        )

                        search_start = time.time()
                        results = await db.search(search_query)
                        search_time = time.time() - search_start
                        search_times.append(search_time)

                    avg_search_time = sum(search_times) / len(search_times)

                    result = {
                        "name": config_info["name"],
                        "add_time": add_time,
                        "avg_search_time": avg_search_time,
                        "vectors_per_sec": test_size / add_time,
                        "searches_per_sec": 1 / avg_search_time,
                    }

                    performance_results.append(result)

                    print(
                        f"   ✅ Add time: {add_time:.3f}s ({result['vectors_per_sec']:.1f} vectors/sec)"
                    )
                    print(
                        f"   ✅ Search time: {avg_search_time:.4f}s ({result['searches_per_sec']:.1f} searches/sec)"
                    )
                else:
                    print("   ❌ Add operation failed")
            else:
                print("   ❌ Initialization failed")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    # Performance summary
    if performance_results:
        print(f"\n📊 Performance Summary:")
        print(f"{'Configuration':<25} {'Add (vec/s)':<12} {'Search (qps)':<15}")
        print("-" * 52)

        for result in performance_results:
            print(
                f"{result['name']:<25} {result['vectors_per_sec']:<12.1f} {result['searches_per_sec']:<15.1f}"
            )


async def main():
    """Main demo function"""
    print("🌟 Vega2.0 Vector Database Integration Demo")
    print("=" * 60)
    print("Large-Scale Similarity Search & Retrieval")
    print(f"Demo Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    try:
        # Run all vector database demos
        await demo_faiss_database()
        await demo_pinecone_database()
        await demo_batch_operations()
        await demo_performance_comparison()

        print("\n" + "=" * 60)
        print("✨ Vector Database Integration Demo Completed Successfully!")
        print("🚀 Large-scale similarity search capabilities are ready.")
        print("📈 Both local (FAISS) and cloud (Pinecone) options available.")
        print("⚡ Optimized for batch operations and high performance.")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logger.exception("Vector database demo execution failed")


if __name__ == "__main__":
    asyncio.run(main())
