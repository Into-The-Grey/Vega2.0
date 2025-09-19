"""Image input handling utilities for federated multi-modal pipelines."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps, ImageStat, ExifTags

try:  # Optional torch integration
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch optional at runtime
    TORCH_AVAILABLE = False


SAFE_EXIF_FIELDS: set[str] = {
    "Orientation",
    "ImageWidth",
    "ImageLength",
    "ColorSpace",
    "PixelXDimension",
    "PixelYDimension",
}


@dataclass
class ImageInputConfig:
    """Configuration for :class:`ImageInputHandler`."""

    allowed_formats: Tuple[str, ...] = ("JPEG", "PNG", "TIFF", "WEBP")
    color_mode: str = "RGB"
    max_pixels: int = 50_000_000
    max_dimension: int = 8192
    thumbnail_size: Tuple[int, int] = (256, 256)
    thumbnail_mode: str = "RGB"
    normalize: bool = True
    dtype: str = "float32"
    hash_algorithm: str = "sha256"
    target_format: str = "PNG"
    quality: int = 95


@dataclass
class ProcessedImage:
    """Container with sanitized image artifacts."""

    sanitized_bytes: bytes
    format: str
    width: int
    height: int
    color_mode: str
    array: np.ndarray
    thumbnail_bytes: bytes
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_tensor(self):  # pragma: no cover - dependent on torch availability
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch not available")
        return torch.from_numpy(self.array.transpose(2, 0, 1))


class ImageInputError(ValueError):
    """Raised when image processing fails."""


class ImageInputHandler:
    """Processes image inputs for the federated multi-modal pipeline."""

    def __init__(self, config: Optional[ImageInputConfig] = None):
        self.config = config or ImageInputConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self, source: Any) -> ProcessedImage:
        image = self._load_image(source)
        self._validate_image(image)

        exif_summary = self._extract_exif_summary(image)
        had_exif = bool(exif_summary) or bool(image.info.get("exif"))

        sanitized_image = self._sanitize_image(image)
        sanitized_bytes = self._encode_image(sanitized_image)

        thumb = self._generate_thumbnail(sanitized_image)
        thumbnail_bytes = self._encode_image(thumb, format=self.config.target_format)

        array = self._to_numpy(sanitized_image)
        metadata = self._compute_metadata(sanitized_image, sanitized_bytes)
        metadata["exif_removed"] = had_exif
        if exif_summary:
            metadata["exif_summary"] = exif_summary

        return ProcessedImage(
            sanitized_bytes=sanitized_bytes,
            format=sanitized_image.format or self.config.target_format,
            width=sanitized_image.width,
            height=sanitized_image.height,
            color_mode=sanitized_image.mode,
            array=array,
            thumbnail_bytes=thumbnail_bytes,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_image(self, source: Any) -> Image.Image:
        if isinstance(source, (bytes, bytearray)):
            stream = io.BytesIO(source)
        elif hasattr(source, "read"):
            stream = io.BytesIO(source.read())
        else:
            path = Path(source)
            if not path.exists():
                raise ImageInputError(f"Image path not found: {source}")
            stream = io.BytesIO(path.read_bytes())

        try:
            image = Image.open(stream)
            image.load()
        except Exception as exc:
            raise ImageInputError(f"Unable to load image: {exc}") from exc
        return image

    def _validate_image(self, image: Image.Image) -> None:
        fmt = (image.format or "").upper()
        if fmt not in self.config.allowed_formats:
            raise ImageInputError(f"Unsupported format: {fmt}")
        if image.width <= 0 or image.height <= 0:
            raise ImageInputError("Invalid image dimensions")
        if image.width > self.config.max_dimension or image.height > self.config.max_dimension:
            raise ImageInputError("Image dimensions exceed limit")
        if image.width * image.height > self.config.max_pixels:
            raise ImageInputError("Image contains too many pixels")

    def _sanitize_image(self, image: Image.Image) -> Image.Image:
        converted = image.convert(self.config.color_mode)
        sanitized = Image.new(self.config.color_mode, converted.size)
        sanitized.putdata(list(converted.getdata()))
        sanitized.info.clear()
        if "icc_profile" in image.info:
            sanitized.info.pop("icc_profile", None)
        sanitized.format = self.config.target_format
        return sanitized

    def _extract_exif_summary(self, image: Image.Image) -> dict[str, Any]:
        raw = image.getexif()
        if not raw:
            return {}
        summary: dict[str, Any] = {}
        for tag, value in raw.items():
            name = ExifTags.TAGS.get(tag, str(tag))
            if name in SAFE_EXIF_FIELDS:
                summary[name] = value
        return summary

    def _generate_thumbnail(self, image: Image.Image) -> Image.Image:
        thumb = ImageOps.fit(
            image.copy(),
            self.config.thumbnail_size,
            Image.Resampling.LANCZOS,
        )
        return thumb.convert(self.config.thumbnail_mode)

    def _encode_image(self, image: Image.Image, *, format: Optional[str] = None) -> bytes:
        output = io.BytesIO()
        save_format = format or self.config.target_format
        params = {}
        if save_format in {"JPEG", "WEBP"}:
            params["quality"] = self.config.quality
            if save_format == "JPEG":
                params["optimize"] = True
        image.save(output, format=save_format, **params)
        return output.getvalue()

    def _to_numpy(self, image: Image.Image) -> np.ndarray:
        array = np.asarray(image, dtype=self.config.dtype)
        if self.config.normalize:
            array /= 255.0
        if array.ndim == 2:
            array = np.expand_dims(array, axis=2)
        return array

    def _compute_metadata(self, image: Image.Image, data: bytes) -> dict[str, Any]:
        stat = ImageStat.Stat(image)
        hash_digest = hashlib.new(self.config.hash_algorithm, data).hexdigest()
        return {
            "hash": hash_digest,
            "channels": image.getbands(),
            "mean": tuple(round(v, 4) for v in stat.mean),
            "stddev": tuple(round(v, 4) for v in stat.stddev),
        }


def process_image(source: Any, config: Optional[ImageInputConfig] = None) -> ProcessedImage:
    handler = ImageInputHandler(config)
    return handler.process(source)
