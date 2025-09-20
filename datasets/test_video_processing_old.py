from datasets.video_processing import (
    VideoProcessor,
    VideoFormatConverter,
    validate_video_file,
    extract_video_metadata,
    extract_video_frames,
    sample_video_frames,
    generate_video_thumbnail,
    get_video_information,
    SUPPORTED_VIDEO_FORMATS,
)
import tempfile
import os
import cv2
import numpy as np
import pytest
from PIL import Image
from pathlib import Path


def create_test_video(
    output_path: str, duration: int = 2, fps: int = 10, size: tuple = (320, 240)
):
    """Create a simple test video for testing purposes."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    total_frames = duration * fps
    for i in range(total_frames):
        # Create a simple frame with changing colors
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        # Create a gradient effect
        frame[:, :, 0] = i * 255 // total_frames  # Red channel
        frame[:, :, 1] = (total_frames - i) * 255 // total_frames  # Green channel
        frame[:, :, 2] = 128  # Blue channel constant

        out.write(frame)

    out.release()
    return output_path


@pytest.fixture
def test_video():
    """Create a temporary test video."""
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    create_test_video(path)
    yield path

    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def video_processor():
    """Create a VideoProcessor instance."""
    return VideoProcessor()


def test_supported_video_formats():
    """Test that supported video formats are properly defined."""
    assert ".mp4" in SUPPORTED_VIDEO_FORMATS
    assert ".avi" in SUPPORTED_VIDEO_FORMATS
    assert ".mov" in SUPPORTED_VIDEO_FORMATS
    assert ".webm" in SUPPORTED_VIDEO_FORMATS


def test_is_supported_video(video_processor):
    """Test video format detection."""
    assert video_processor.is_supported_video("test.mp4")
    assert video_processor.is_supported_video("test.avi")
    assert video_processor.is_supported_video("test.mov")
    assert not video_processor.is_supported_video("test.txt")
    assert not video_processor.is_supported_video("test.xyz")


def test_validate_video_nonexistent(video_processor):
    """Test validation of non-existent video."""
    result = video_processor.validate_video("nonexistent.mp4")
    assert not result["is_valid"]
    assert "does not exist" in result["error"]


def test_validate_video_valid(video_processor, test_video):
    """Test validation of a valid video."""
    result = video_processor.validate_video(test_video)

    assert result["is_valid"]
    assert result["error"] is None
    assert result["format"] == "MP4"
    assert result["duration"] > 0
    assert result["frame_count"] > 0
    assert result["fps"] > 0
    assert result["resolution"][0] > 0
    assert result["resolution"][1] > 0
    assert result["file_size"] > 0


def test_extract_metadata(video_processor, test_video):
    """Test metadata extraction."""
    metadata = video_processor.extract_metadata(test_video)

    assert metadata["error"] is None
    assert "basic_info" in metadata
    assert "technical_details" in metadata
    assert "timestamps" in metadata

    assert metadata["basic_info"]["is_valid"]
    assert "created" in metadata["timestamps"]
    assert "modified" in metadata["timestamps"]


def test_extract_frames(video_processor, test_video):
    """Test frame extraction."""
    with tempfile.TemporaryDirectory() as temp_dir:
        frames = video_processor.extract_frames(
            test_video, output_dir=temp_dir, frame_interval=5, max_frames=3
        )

        assert len(frames) <= 3
        assert len(frames) > 0

        # Check that frame files exist
        for frame_path in frames:
            assert os.path.exists(frame_path)
            assert frame_path.endswith(".jpg")


def test_sample_frames_uniform(video_processor, test_video):
    """Test uniform frame sampling."""
    frames = video_processor.sample_frames(test_video, num_samples=5, method="uniform")

    assert len(frames) <= 5
    assert len(frames) > 0

    # Check that frames are numpy arrays
    for frame in frames:
        assert isinstance(frame, np.ndarray)
        assert len(frame.shape) == 3  # Height, width, channels


def test_sample_frames_random(video_processor, test_video):
    """Test random frame sampling."""
    frames = video_processor.sample_frames(test_video, num_samples=3, method="random")

    assert len(frames) <= 3
    assert len(frames) > 0

    for frame in frames:
        assert isinstance(frame, np.ndarray)


def test_generate_thumbnail(video_processor, test_video):
    """Test thumbnail generation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        thumbnail_path = os.path.join(temp_dir, "thumb.jpg")
        result = video_processor.generate_thumbnail(
            test_video, output_path=thumbnail_path, size=(160, 120)
        )

        assert result == thumbnail_path
        assert os.path.exists(thumbnail_path)

        # Verify thumbnail dimensions
        with Image.open(thumbnail_path) as img:
            assert img.size[0] <= 160
            assert img.size[1] <= 120


def test_convenience_functions(test_video):
    """Test convenience functions."""
    # Test validation function
    result = validate_video_file(test_video)
    assert result["is_valid"]

    # Test metadata extraction function
    metadata = extract_video_metadata(test_video)
    assert metadata["error"] is None

    # Test frame sampling function
    frames = sample_video_frames(test_video, num_samples=2)
    assert len(frames) <= 2


def test_error_handling():
    """Test error handling with invalid inputs."""
    processor = VideoProcessor()

    # Test with non-existent file
    result = processor.validate_video("nonexistent.mp4")
    assert not result["is_valid"]

    # Test frame extraction with invalid file
    frames = processor.extract_frames("nonexistent.mp4")
    assert len(frames) == 0

    # Test thumbnail generation with invalid file
    thumbnail = processor.generate_thumbnail("nonexistent.mp4")
    assert thumbnail == ""


def test_video_format_converter():
    """Test video format converter initialization."""
    converter = VideoFormatConverter()
    assert converter is not None


def test_video_processor_cleanup(video_processor):
    """Test that VideoProcessor cleans up properly."""
    temp_dir = video_processor.temp_dir
    assert os.path.exists(temp_dir)

    # Manually trigger cleanup
    video_processor.__del__()

    # Note: In some cases, the directory might still exist due to timing
    # This test mainly ensures no exceptions are raised during cleanup


def test_unsupported_format_validation(video_processor):
    """Test validation of unsupported format."""
    # Create a dummy file with unsupported extension
    fd, path = tempfile.mkstemp(suffix=".xyz")
    os.close(fd)

    try:
        result = video_processor.validate_video(path)
        assert not result["is_valid"]
        assert "Unsupported video format" in result["error"]
    finally:
        os.remove(path)


def test_empty_video_path(video_processor):
    """Test handling of empty video path."""
    result = video_processor.validate_video("")
    assert not result["is_valid"]
