"""
Image utilities for Vega 2.0: format support, validation, resizing, thumbnail generation, EXIF extraction/scrubbing.
"""

import io
from typing import Tuple, Optional
from PIL import Image, ExifTags
import os

# Supported formats
SUPPORTED_FORMATS = ("JPEG", "PNG", "TIFF", "WEBP")


def is_supported_image(path: str) -> bool:
    """Check if file is a supported image format."""
    try:
        with Image.open(path) as img:
            return img.format in SUPPORTED_FORMATS
    except Exception:
        return False


def validate_image(path: str, max_size_mb: float = 10.0) -> bool:
    """Validate image: format, size, and integrity."""
    if not os.path.isfile(path):
        return False
    if os.path.getsize(path) > max_size_mb * 1024 * 1024:
        return False
    try:
        with Image.open(path) as img:
            img.verify()  # Check for corruption
            if img.format not in SUPPORTED_FORMATS:
                return False
        return True
    except Exception:
        return False


def sanitize_image(path: str, output_path: Optional[str] = None) -> str:
    """Sanitize image by re-saving to strip metadata and ensure valid format."""
    with Image.open(path) as img:
        data = img.convert("RGB") if img.mode != "RGB" else img.copy()
        fmt = img.format if img.format in SUPPORTED_FORMATS else "JPEG"
        out_path = output_path or path
        data.save(out_path, fmt)
    return out_path


def resize_image(
    path: str, size: Tuple[int, int], output_path: Optional[str] = None
) -> str:
    """Resize image to (width, height)."""
    with Image.open(path) as img:
        resized = img.resize(size, Image.LANCZOS)
        out_path = output_path or path
        resized.save(out_path, img.format)
    return out_path


def generate_thumbnail(
    path: str,
    thumb_size: Tuple[int, int] = (128, 128),
    output_path: Optional[str] = None,
) -> str:
    """Generate thumbnail for image."""
    with Image.open(path) as img:
        img.thumbnail(thumb_size)
        out_path = output_path or (
            os.path.splitext(path)[0] + "_thumb." + img.format.lower()
        )
        img.save(out_path, img.format)
    return out_path


def extract_exif(path: str) -> dict:
    """Extract EXIF metadata from image (if present)."""
    with Image.open(path) as img:
        exif = img._getexif()
        if not exif:
            return {}
        return {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}


def scrub_exif(path: str, output_path: Optional[str] = None) -> str:
    """Remove EXIF metadata from image by re-saving without EXIF."""
    with Image.open(path) as img:
        data = img.convert("RGB") if img.mode != "RGB" else img.copy()
        fmt = img.format if img.format in SUPPORTED_FORMATS else "JPEG"
        out_path = output_path or path
        data.save(out_path, fmt)
    return out_path
