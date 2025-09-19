import pytest
import tempfile
import os
from PIL import Image
import numpy as np
from datasets.computer_vision import (
    ImageClassificationModel,
    ObjectDetectionModel,
    FaceDetectionModel,
    OCRModel,
    ImageSegmentationModel,
    ComputerVisionPipeline,
    classify_image,
    detect_objects,
    detect_faces,
    extract_text,
    segment_image,
)


def create_test_image(size=(224, 224), color="RGB"):
    """Create a test image for testing."""
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)

    # Create a simple test image
    if color == "RGB":
        img = Image.new("RGB", size, color=(255, 0, 0))  # Red image
    else:
        img = Image.new("L", size, 128)  # Gray image

    img.save(path)
    return path


def test_image_classification():
    """Test image classification functionality."""
    image_path = create_test_image()

    try:
        classifier = ImageClassificationModel(model_name="resnet50")
        results = classifier.predict(image_path, top_k=3)

        assert isinstance(results, list)
        assert len(results) == 3
        assert all("class" in r and "confidence" in r for r in results)

    finally:
        os.remove(image_path)


def test_object_detection():
    """Test object detection functionality."""
    image_path = create_test_image(size=(640, 480))

    try:
        detector = ObjectDetectionModel()
        results = detector.detect(image_path)

        assert isinstance(results, list)
        # Results might be empty for a simple test image, which is fine

    finally:
        os.remove(image_path)


def test_face_detection():
    """Test face detection functionality."""
    image_path = create_test_image(size=(300, 300))

    try:
        face_detector = FaceDetectionModel()
        results = face_detector.detect_faces(image_path)

        assert isinstance(results, list)
        # Results might be empty for a simple test image, which is fine

    finally:
        os.remove(image_path)


def test_ocr():
    """Test OCR functionality."""
    image_path = create_test_image()

    try:
        ocr = OCRModel()
        result = ocr.extract_text(image_path)

        assert isinstance(result, dict)
        assert "text" in result
        assert "confidence" in result

    finally:
        os.remove(image_path)


def test_image_segmentation():
    """Test image segmentation functionality."""
    image_path = create_test_image()

    try:
        segmenter = ImageSegmentationModel()
        result = segmenter.segment(image_path)

        assert isinstance(result, dict)
        if "error" not in result:
            assert "segmentation_mask" in result
            assert "detected_objects" in result

    finally:
        os.remove(image_path)


def test_cv_pipeline():
    """Test the complete computer vision pipeline."""
    image_path = create_test_image()

    try:
        pipeline = ComputerVisionPipeline()
        results = pipeline.analyze_image(image_path, tasks=["classification"])

        assert isinstance(results, dict)
        assert "results" in results
        assert "status" in results

    finally:
        os.remove(image_path)


def test_convenience_functions():
    """Test convenience functions."""
    image_path = create_test_image()

    try:
        # Test classification convenience function
        results = classify_image(image_path)
        assert isinstance(results, list)

        # Test detection convenience function
        results = detect_objects(image_path)
        assert isinstance(results, list)

        # Test face detection convenience function
        results = detect_faces(image_path)
        assert isinstance(results, list)

        # Test OCR convenience function
        result = extract_text(image_path)
        assert isinstance(result, dict)

        # Test segmentation convenience function
        result = segment_image(image_path)
        assert isinstance(result, dict)

    finally:
        os.remove(image_path)


def test_model_initialization():
    """Test that models can be initialized without errors."""
    # These should not raise exceptions
    classifier = ImageClassificationModel()
    detector = ObjectDetectionModel()
    face_detector = FaceDetectionModel()
    ocr = OCRModel()
    segmenter = ImageSegmentationModel()
    pipeline = ComputerVisionPipeline()

    assert classifier is not None
    assert detector is not None
    assert face_detector is not None
    assert ocr is not None
    assert segmenter is not None
    assert pipeline is not None
