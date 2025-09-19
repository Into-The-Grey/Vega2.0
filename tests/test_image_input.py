import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageChops

from src.vega.federated.image_input import (
    ImageInputConfig,
    ImageInputHandler,
    ImageInputError,
    process_image,
)


@pytest.fixture(scope="module")
def sample_images(tmp_path_factory):
    base = tmp_path_factory.mktemp("images")
    formats = {"JPEG": "RGB", "PNG": "RGBA", "TIFF": "RGB", "WEBP": "RGB"}
    files = {}
    for fmt, mode in formats.items():
        img = Image.new(mode, (128, 96), color=(120, 80, 200))
        buffer = io.BytesIO()
        img.save(buffer, format=fmt)
        buffer.seek(0)
        path = base / f"sample.{fmt.lower()}"
        path.write_bytes(buffer.getvalue())
        files[fmt] = path
    return files


def test_process_supported_formats(sample_images):
    handler = ImageInputHandler()
    for fmt, path in sample_images.items():
        processed = handler.process(path)
        assert processed.format == handler.config.target_format
        assert processed.width == 128 and processed.height == 96
        assert processed.metadata["hash"]
        assert processed.metadata["mean"]
        thumb = Image.open(io.BytesIO(processed.thumbnail_bytes))
        assert thumb.size == handler.config.thumbnail_size


def test_process_bytes(sample_images):
    jpeg_path = sample_images["JPEG"]
    data = jpeg_path.read_bytes()
    processed = process_image(data)
    assert isinstance(processed.array, np.ndarray)
    assert processed.array.shape[-1] == 3


def test_reject_invalid_format(tmp_path):
    txt = tmp_path / "bad.txt"
    txt.write_text("not an image")
    handler = ImageInputHandler()
    with pytest.raises(ImageInputError):
        handler.process(txt)


def test_exif_removed(tmp_path):
    img = Image.new("RGB", (32, 32), color=(255, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", exif=b"Exif\x00\x00")
    buffer.seek(0)

    handler = ImageInputHandler()
    processed = handler.process(buffer.getvalue())
    sanitized = Image.open(io.BytesIO(processed.sanitized_bytes))
    assert sanitized.getexif() == {}
    assert processed.metadata["exif_removed"] is True


def test_normalization_range(sample_images):
    handler = ImageInputHandler()
    processed = handler.process(sample_images["PNG"])
    expected_max = processed.array.max()
    assert 0.0 <= expected_max <= 1.0
