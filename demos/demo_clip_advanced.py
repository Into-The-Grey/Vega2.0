#!/usr/bin/env python3
"""
Advanced CLIP Integration Demo
=============================

Demonstration of Vega2.0's enhanced CLIP model integration with advanced
zero-shot classification, cross-modal retrieval, and sophisticated features.

This demo showcases:
- Multiple CLIP model variants (ViT, ResNet architectures)
- Enhanced zero-shot classification with confidence scoring
- Advanced image processing modes (high-res, multi-crop, ensemble)
- Cross-modal retrieval with sophisticated ranking
- Performance monitoring and optimization
- Batch processing capabilities
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any

# Import enhanced CLIP integration
from src.vega.multimodal import (
    AdvancedCLIPIntegration,
    CLIPEnhancedConfig,
    CLIPModelType,
    ProcessingMode,
    create_advanced_clip_integration,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def demo_advanced_clip_models():
    """Demonstrate different CLIP model variants"""
    print("🔬 Advanced CLIP Model Variants Demo")
    print("=" * 40)

    # Test different model types
    model_variants = [
        CLIPModelType.VIT_B_32,
        CLIPModelType.VIT_L_14,
        CLIPModelType.RN50,
        CLIPModelType.RN50x4,
    ]

    sample_image = "demo_mountain_landscape.jpg"
    sample_text = "a beautiful mountain landscape with snow-capped peaks"

    for variant in model_variants:
        print(f"\n🔍 Testing {variant.value}:")

        # Create advanced CLIP integration
        clip_integration = create_advanced_clip_integration(
            model_type=variant,
            processing_mode=ProcessingMode.STANDARD,
            batch_size=16,
            temperature=0.05,
            enable_caching=True,
        )

        # Initialize and test
        success = await clip_integration.initialize()
        if success:
            # Test image encoding
            image_analysis = await clip_integration.encode_image_advanced(sample_image)
            if image_analysis:
                print(
                    f"   ✅ Image encoded successfully ({image_analysis.image_embedding.shape})"
                )
                print(f"   🎯 Confidence: {image_analysis.confidence_scores}")
                print(f"   ⚡ Processing time: {image_analysis.processing_time:.3f}s")

                # Test text encoding
                text_analysis = await clip_integration.encode_text_advanced(sample_text)
                if text_analysis:
                    print(
                        f"   ✅ Text encoded successfully ({text_analysis.text_embedding.shape})"
                    )
                    print(
                        f"   📊 Semantic features: {len(text_analysis.semantic_features)} categories"
                    )
                    print(f"   🔑 Keywords: {text_analysis.keywords[:5]}")
            else:
                print(f"   ❌ Image encoding failed")
        else:
            print(f"   ❌ Model initialization failed")

        # Cleanup
        await clip_integration.shutdown()


async def demo_processing_modes():
    """Demonstrate different image processing modes"""
    print("\n🖼️  Image Processing Modes Demo")
    print("=" * 35)

    processing_modes = [
        ProcessingMode.STANDARD,
        ProcessingMode.HIGH_RESOLUTION,
        ProcessingMode.MULTI_CROP,
        ProcessingMode.ENSEMBLE,
    ]

    sample_images = ["demo_mountain.jpg", "demo_technology.jpg", "demo_nature.jpg"]

    for mode in processing_modes:
        print(f"\n🔧 Processing Mode: {mode.value}")

        # Create integration with specific processing mode
        clip_integration = create_advanced_clip_integration(
            model_type=CLIPModelType.VIT_B_32, processing_mode=mode, enable_caching=True
        )

        await clip_integration.initialize()

        processing_times = []
        for image in sample_images:
            analysis = await clip_integration.encode_image_advanced(image)
            if analysis:
                processing_times.append(analysis.processing_time)
                print(f"   📸 {image}: {analysis.processing_time:.3f}s")
                print(
                    f"      Visual features: {len(analysis.visual_features)} detected"
                )

        avg_time = (
            sum(processing_times) / len(processing_times) if processing_times else 0
        )
        print(f"   ⚡ Average processing time: {avg_time:.3f}s")

        await clip_integration.shutdown()


async def demo_enhanced_zero_shot_classification():
    """Demonstrate enhanced zero-shot classification"""
    print("\n🎯 Enhanced Zero-Shot Classification Demo")
    print("=" * 45)

    # Create advanced CLIP integration
    clip_integration = create_advanced_clip_integration(
        model_type=CLIPModelType.VIT_L_14,
        processing_mode=ProcessingMode.ENSEMBLE,
        temperature=0.1,
        confidence_threshold=0.15,
    )

    await clip_integration.initialize()

    # Classification scenarios
    classification_scenarios = [
        {
            "image": "demo_mountain_landscape.jpg",
            "classes": ["mountain", "forest", "city", "ocean", "desert", "building"],
            "description": "Natural landscape classification",
        },
        {
            "image": "demo_technology_scene.jpg",
            "classes": ["computer", "smartphone", "car", "book", "plant", "animal"],
            "description": "Technology object detection",
        },
        {
            "image": "demo_concert_scene.jpg",
            "classes": [
                "music concert",
                "sports event",
                "conference",
                "restaurant",
                "library",
                "park",
            ],
            "description": "Scene type classification",
        },
    ]

    for i, scenario in enumerate(classification_scenarios, 1):
        print(f"\n{i}. {scenario['description']}")
        print(f"   Image: {scenario['image']}")
        print(f"   Classes: {scenario['classes']}")

        # Perform enhanced classification
        result = await clip_integration.zero_shot_classify_enhanced(
            scenario["image"], scenario["classes"]
        )

        if result:
            print(
                f"   🏆 Top Prediction: {result.top_prediction} ({result.top_confidence:.3f})"
            )
            print(f"   ⚡ Processing time: {result.processing_time:.3f}s")
            print(f"   📊 All predictions:")

            for j, (class_name, confidence, metadata) in enumerate(
                result.predictions[:3]
            ):
                print(f"      {j+1}. {class_name}: {confidence:.3f}")
                if "confidence_factors" in metadata:
                    factors = metadata["confidence_factors"]
                    print(
                        f"         Factors: embedding={factors.get('embedding_quality', 0):.2f}, "
                        f"semantic={factors.get('semantic_alignment', False)}"
                    )
        else:
            print(f"   ❌ Classification failed")

    await clip_integration.shutdown()


async def demo_cross_modal_retrieval():
    """Demonstrate advanced cross-modal retrieval"""
    print("\n🔍 Cross-Modal Retrieval Demo")
    print("=" * 30)

    # Create advanced CLIP integration
    clip_integration = create_advanced_clip_integration(
        model_type=CLIPModelType.VIT_B_16,
        processing_mode=ProcessingMode.MULTI_CROP,
        enable_caching=True,
    )

    await clip_integration.initialize()

    # Retrieval scenarios
    retrieval_scenarios = [
        {
            "query": "beautiful mountain landscape with snow",
            "candidates": [
                "demo_mountain_winter.jpg",
                "demo_city_skyline.jpg",
                "demo_forest_summer.jpg",
                "demo_ocean_sunset.jpg",
                "mountain hiking adventure guide",
                "urban photography techniques",
                "nature conservation article",
            ],
            "description": "Nature landscape retrieval",
        },
        {
            "query": "artificial intelligence and machine learning",
            "candidates": [
                "demo_computer_lab.jpg",
                "demo_robot_demo.jpg",
                "demo_library_books.jpg",
                "AI research paper abstract",
                "machine learning tutorial",
                "cooking recipe instructions",
                "travel photography tips",
            ],
            "description": "Technology content retrieval",
        },
    ]

    for i, scenario in enumerate(retrieval_scenarios, 1):
        print(f"\n{i}. {scenario['description']}")
        print(f"   Query: '{scenario['query']}'")
        print(f"   Candidates: {len(scenario['candidates'])} items")

        # Perform cross-modal retrieval
        result = await clip_integration.cross_modal_retrieval(
            scenario["query"], scenario["candidates"], modality_type="mixed"
        )

        print(f"   ⚡ Retrieval time: {result.retrieval_time:.3f}s")
        print(f"   📊 Top results:")

        for j, (candidate, score, metadata) in enumerate(result.results[:3]):
            content_type = metadata.get("type", "unknown")
            print(f"      {j+1}. [{content_type.upper()}] {candidate}")
            print(f"         Similarity: {score:.3f}")
            if "confidence" in metadata:
                print(f"         Confidence: {metadata['confidence']:.3f}")

    await clip_integration.shutdown()


async def demo_batch_processing():
    """Demonstrate batch processing capabilities"""
    print("\n⚡ Batch Processing Demo")
    print("=" * 25)

    # Create advanced CLIP integration optimized for batch processing
    clip_integration = create_advanced_clip_integration(
        model_type=CLIPModelType.VIT_B_32,
        processing_mode=ProcessingMode.STANDARD,
        batch_size=8,
        enable_caching=True,
        num_workers=4,
    )

    await clip_integration.initialize()

    # Batch image classification
    batch_images = [f"demo_image_{i:03d}.jpg" for i in range(1, 13)]  # 12 demo images

    class_labels = [
        "nature",
        "technology",
        "people",
        "architecture",
        "transportation",
        "art",
    ]

    print(f"🖼️  Batch processing {len(batch_images)} images")
    print(f"📋 Classification with {len(class_labels)} classes")

    start_time = asyncio.get_event_loop().time()

    # Process all images in batches
    batch_results = []
    batch_size = 4

    for i in range(0, len(batch_images), batch_size):
        batch = batch_images[i : i + batch_size]
        print(f"   Processing batch {i//batch_size + 1}: {len(batch)} images")

        # Process batch
        batch_classifications = []
        for image in batch:
            result = await clip_integration.zero_shot_classify_enhanced(
                image, class_labels
            )
            batch_classifications.append(result)

        batch_results.extend(batch_classifications)

    total_time = asyncio.get_event_loop().time() - start_time

    # Analyze results
    successful_classifications = [r for r in batch_results if r is not None]

    print(f"\n📊 Batch Processing Results:")
    print(f"   Total images: {len(batch_images)}")
    print(f"   Successfully processed: {len(successful_classifications)}")
    print(f"   Total processing time: {total_time:.3f}s")
    print(f"   Average time per image: {total_time/len(batch_images):.3f}s")

    if successful_classifications:
        avg_confidence = sum(
            r.top_confidence for r in successful_classifications
        ) / len(successful_classifications)
        print(f"   Average top confidence: {avg_confidence:.3f}")

        # Class distribution
        class_counts = {}
        for result in successful_classifications:
            top_class = result.top_prediction
            class_counts[top_class] = class_counts.get(top_class, 0) + 1

        print(f"   📈 Class distribution:")
        for class_name, count in sorted(
            class_counts.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"      {class_name}: {count} images")

    await clip_integration.shutdown()


async def demo_performance_monitoring():
    """Demonstrate performance monitoring and statistics"""
    print("\n📈 Performance Monitoring Demo")
    print("=" * 30)

    # Create integration with performance monitoring
    clip_integration = create_advanced_clip_integration(
        model_type=CLIPModelType.VIT_B_32, enable_caching=True, cache_size=100
    )

    await clip_integration.initialize()

    print("🔄 Running mixed workload to generate performance data...")

    # Mixed workload
    tasks = [
        # Image encoding tasks
        clip_integration.encode_image_advanced("test_image_1.jpg"),
        clip_integration.encode_image_advanced("test_image_2.jpg"),
        clip_integration.encode_image_advanced("test_image_1.jpg"),  # Cache hit
        # Text encoding tasks
        clip_integration.encode_text_advanced("mountain landscape"),
        clip_integration.encode_text_advanced("technology innovation"),
        clip_integration.encode_text_advanced("mountain landscape"),  # Cache hit
        # Classification task
        clip_integration.zero_shot_classify_enhanced(
            "test_image_3.jpg", ["nature", "technology", "art"]
        ),
        # Retrieval task
        clip_integration.cross_modal_retrieval(
            "beautiful scenery", ["image1.jpg", "image2.jpg", "text_doc.txt"]
        ),
    ]

    # Execute all tasks
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Get performance statistics
    stats = clip_integration.get_performance_stats()

    print(f"\n📊 Performance Statistics:")
    print(f"   Model: {stats['model_info']['type']}")
    print(f"   Processing Mode: {stats['model_info']['processing_mode']}")
    print(f"   Device: {stats['model_info']['device']}")
    print(f"   Embedding Dimension: {stats['model_info']['embedding_dim']}")

    print(f"\n⚡ Performance Metrics:")
    print(f"   Images Processed: {stats['performance']['total_images_processed']}")
    print(f"   Texts Processed: {stats['performance']['total_texts_processed']}")
    print(f"   Cache Hit Rate: {stats['performance']['cache_hit_rate']:.1%}")
    print(f"   Cache Size: {stats['performance']['cache_size']}")
    print(
        f"   Average Processing Time: {stats['performance']['average_processing_time']:.3f}s"
    )

    print(f"\n⚙️  Configuration:")
    print(f"   Batch Size: {stats['configuration']['batch_size']}")
    print(f"   Temperature: {stats['configuration']['temperature']}")
    print(f"   Confidence Threshold: {stats['configuration']['confidence_threshold']}")

    await clip_integration.shutdown()


async def main():
    """Main demo function"""
    print("🌟 Vega2.0 Advanced CLIP Integration Demo")
    print("=" * 60)
    print("Enhanced Multi-Modal Understanding with CLIP")
    print(f"Demo Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    try:
        # Run all demos
        await demo_advanced_clip_models()
        await demo_processing_modes()
        await demo_enhanced_zero_shot_classification()
        await demo_cross_modal_retrieval()
        await demo_batch_processing()
        await demo_performance_monitoring()

        print("\n" + "=" * 60)
        print("✨ Advanced CLIP Integration Demo Completed Successfully!")
        print("🚀 Enhanced multi-modal capabilities are ready for deployment.")
        print("📈 Performance optimizations and monitoring active.")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logger.exception("Advanced CLIP demo execution failed")


if __name__ == "__main__":
    asyncio.run(main())
