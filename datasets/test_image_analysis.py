import pytest
import tempfile
import os
import numpy as np
from PIL import Image
from datasets.image_analysis import (
    ImageFeatureExtractor,
    ImageSimilarityAnalyzer,
    ImageClusteringAnalyzer,
    AutoTaggingSystem,
    ImageQualityAssessment,
    ContentBasedImageRetrieval,
    find_similar_images,
    cluster_images,
    assess_image_quality,
    auto_tag_image,
)


def create_test_image(size=(224, 224), color=(255, 0, 0)):
    """Create a test image with specified color."""
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)

    img = Image.new("RGB", size, color)
    img.save(path)
    return path


def create_test_images(count=5):
    """Create multiple test images with different colors."""
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    paths = []

    for i in range(count):
        color = colors[i % len(colors)]
        path = create_test_image(color=color)
        paths.append(path)

    return paths


def cleanup_test_images(paths):
    """Clean up test image files."""
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def test_feature_extractor():
    """Test image feature extraction."""
    image_path = create_test_image()

    try:
        extractor = ImageFeatureExtractor()
        features = extractor.extract_features(image_path)

        assert isinstance(features, np.ndarray)
        assert len(features) > 0

    finally:
        os.remove(image_path)


def test_similarity_analyzer():
    """Test image similarity analysis."""
    test_paths = create_test_images(3)

    try:
        analyzer = ImageSimilarityAnalyzer()

        # Add images to database
        for path in test_paths:
            image_id = analyzer.add_image(path)
            assert isinstance(image_id, str)

        # Test similarity search
        similar = analyzer.find_similar_images(test_paths[0], top_k=2)
        assert isinstance(similar, list)

        # Test direct similarity calculation
        similarity = analyzer.calculate_similarity(test_paths[0], test_paths[1])
        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1

    finally:
        cleanup_test_images(test_paths)


def test_clustering_analyzer():
    """Test image clustering."""
    test_paths = create_test_images(4)

    try:
        analyzer = ImageClusteringAnalyzer()
        results = analyzer.cluster_images(test_paths, n_clusters=2)

        assert isinstance(results, dict)
        assert "clusters" in results
        assert "n_clusters" in results

    finally:
        cleanup_test_images(test_paths)


def test_auto_tagging():
    """Test automated tagging system."""
    image_path = create_test_image()

    try:
        tagger = AutoTaggingSystem()

        # Test without detection results
        tags = tagger.generate_tags(image_path)
        assert isinstance(tags, dict)
        assert "tags" in tags
        assert "categories" in tags

        # Test with mock detection results
        detection_results = [
            {"class": "dog", "confidence": 0.9},
            {"class": "person", "confidence": 0.8},
        ]
        tags_with_detection = tagger.generate_tags(image_path, detection_results)
        assert isinstance(tags_with_detection, dict)
        assert len(tags_with_detection["tags"]) >= 2

    finally:
        os.remove(image_path)


def test_quality_assessment():
    """Test image quality assessment."""
    image_path = create_test_image()

    try:
        assessor = ImageQualityAssessment()
        quality = assessor.assess_quality(image_path)

        assert isinstance(quality, dict)
        assert "overall_score" in quality
        assert "metrics" in quality
        assert "assessment" in quality
        assert "recommendations" in quality

        assert 0 <= quality["overall_score"] <= 100

    finally:
        os.remove(image_path)


def test_cbir_system():
    """Test complete CBIR system."""
    test_paths = create_test_images(3)

    try:
        cbir = ContentBasedImageRetrieval()

        # Test indexing
        index_result = cbir.index_image_collection(test_paths)
        assert isinstance(index_result, dict)
        assert "indexed_successfully" in index_result

        # Test search
        search_result = cbir.search_by_image(test_paths[0], top_k=2)
        assert isinstance(search_result, dict)
        assert "similar_images" in search_result

        # Test collection analysis
        analysis_result = cbir.analyze_image_collection(test_paths, n_clusters=2)
        assert isinstance(analysis_result, dict)
        assert "collection_statistics" in analysis_result
        assert "clustering_results" in analysis_result

    finally:
        cleanup_test_images(test_paths)


def test_convenience_functions():
    """Test convenience functions."""
    test_paths = create_test_images(3)

    try:
        # Test similarity search
        similar = find_similar_images(test_paths[0], test_paths[1:], top_k=2)
        assert isinstance(similar, list)

        # Test clustering
        clusters = cluster_images(test_paths, n_clusters=2)
        assert isinstance(clusters, dict)

        # Test quality assessment
        quality = assess_image_quality(test_paths[0])
        assert isinstance(quality, dict)

        # Test auto-tagging
        tags = auto_tag_image(test_paths[0])
        assert isinstance(tags, dict)

    finally:
        cleanup_test_images(test_paths)


def test_error_handling():
    """Test error handling with invalid inputs."""
    # Test with non-existent file
    extractor = ImageFeatureExtractor()
    features = extractor.extract_features("nonexistent.jpg")
    assert len(features) == 0

    # Test quality assessment with invalid file
    assessor = ImageQualityAssessment()
    quality = assessor.assess_quality("nonexistent.jpg")
    assert "error" in quality


def test_component_initialization():
    """Test that all components can be initialized without errors."""
    extractor = ImageFeatureExtractor()
    similarity_analyzer = ImageSimilarityAnalyzer()
    clustering_analyzer = ImageClusteringAnalyzer()
    auto_tagger = AutoTaggingSystem()
    quality_assessor = ImageQualityAssessment()
    cbir = ContentBasedImageRetrieval()

    assert extractor is not None
    assert similarity_analyzer is not None
    assert clustering_analyzer is not None
    assert auto_tagger is not None
    assert quality_assessor is not None
    assert cbir is not None
