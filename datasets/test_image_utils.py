import os
import tempfile
from datasets.image_utils import (
    is_supported_image,
    validate_image,
    sanitize_image,
    resize_image,
    generate_thumbnail,
    extract_exif,
    scrub_exif,
)
from PIL import Image
import pytest


def create_test_image(fmt="JPEG", exif=False):
    """Create a simple image file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    img = Image.new("RGB", (100, 100), color="red")
    if exif:
        # Add minimal EXIF data
        exif_bytes = b"Exif\x00\x00" + b"\x00" * 100
        img.save(path, fmt, exif=exif_bytes)
    else:
        img.save(path, fmt)
    return path


def test_is_supported_image():
    path = create_test_image("JPEG")
    assert is_supported_image(path)
    os.remove(path)


def test_validate_image():
    path = create_test_image("JPEG")
    assert validate_image(path)
    os.remove(path)


def test_sanitize_image():
    path = create_test_image("JPEG")
    out = sanitize_image(path)
    assert os.path.exists(out)
    os.remove(path)


def test_resize_image():
    path = create_test_image("JPEG")
    out = resize_image(path, (50, 50))
    with Image.open(out) as img:
        assert img.size == (50, 50)
    os.remove(path)


def test_generate_thumbnail():
    path = create_test_image("JPEG")
    out = generate_thumbnail(path, (32, 32))
    with Image.open(out) as img:
        assert max(img.size) <= 32
    os.remove(path)
    if os.path.exists(out):
        os.remove(out)


def test_extract_and_scrub_exif():
    path = create_test_image("JPEG", exif=True)
    exif = extract_exif(path)
    assert isinstance(exif, dict)
    out = scrub_exif(path)
    # After scrubbing, EXIF should be gone
    exif2 = extract_exif(out)
    assert not exif2
    os.remove(path)
    if os.path.exists(out):
        os.remove(out)
