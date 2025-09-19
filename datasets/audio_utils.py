"""
Audio extraction and processing utilities for Vega 2.0 multi-modal learning.

This module provides:
- Audio extraction from video files (FFmpeg, pydub)
- Audio format validation and conversion
- Audio loading and saving utilities
"""

import os
import ffmpeg
from pydub import AudioSegment
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging
import tempfile
import shutil
import soundfile as sf
import librosa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}


class AudioExtractor:
    """Extract audio tracks from video files and validate audio formats."""

    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()

    def __del__(self):
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def extract_audio(
        self,
        video_path: str,
        output_format: str = "wav",
        output_path: Optional[str] = None,
    ) -> str:
        """Extract audio from video using ffmpeg and save as output_format."""
        try:
            if output_path is None:
                video_name = Path(video_path).stem
                output_path = os.path.join(
                    self.temp_dir, f"{video_name}_audio.{output_format}"
                )
            (
                ffmpeg.input(video_path)
                .output(
                    output_path,
                    acodec="pcm_s16le" if output_format == "wav" else None,
                    ac=1,
                    ar="16000",
                )
                .overwrite_output()
                .run(quiet=True)
            )
            logger.info(f"Extracted audio: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            return ""

    def validate_audio(self, audio_path: str) -> Dict[str, Any]:
        """Validate audio file and return info."""
        result = {
            "is_valid": False,
            "error": None,
            "format": None,
            "duration": 0,
            "sample_rate": 0,
            "channels": 0,
            "file_size": 0,
        }
        try:
            if not os.path.exists(audio_path):
                result["error"] = "Audio file does not exist"
                return result
            ext = Path(audio_path).suffix.lower()
            if ext not in SUPPORTED_AUDIO_FORMATS:
                result["error"] = f"Unsupported audio format: {ext}"
                return result
            info = sf.info(audio_path)
            result["format"] = ext[1:]
            result["duration"] = info.duration
            result["sample_rate"] = info.samplerate
            result["channels"] = info.channels
            result["file_size"] = os.path.getsize(audio_path)
            result["is_valid"] = True
        except Exception as e:
            result["error"] = f"Audio validation failed: {e}"
        return result

    def convert_audio_format(
        self,
        input_path: str,
        output_format: str = "wav",
        output_path: Optional[str] = None,
    ) -> str:
        """Convert audio file to another format using pydub."""
        try:
            if output_path is None:
                input_name = Path(input_path).stem
                output_path = os.path.join(
                    self.temp_dir, f"{input_name}_converted.{output_format}"
                )
            audio = AudioSegment.from_file(input_path)
            audio.export(output_path, format=output_format)
            logger.info(f"Converted audio: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return ""

    def load_audio(self, audio_path: str, sr: int = 16000) -> Tuple[Any, int]:
        """Load audio as numpy array and return (audio, sample_rate)."""
        try:
            audio, sample_rate = librosa.load(audio_path, sr=sr, mono=True)
            return audio, sample_rate
        except Exception as e:
            logger.error(f"Audio loading failed: {e}")
            return None, 0

    def save_audio(self, audio: Any, sample_rate: int, output_path: str) -> bool:
        """Save numpy audio array to file."""
        try:
            sf.write(output_path, audio, sample_rate)
            logger.info(f"Saved audio: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Audio saving failed: {e}")
            return False


# Convenience functions
def extract_audio_from_video(video_path: str, output_format: str = "wav") -> str:
    extractor = AudioExtractor()
    return extractor.extract_audio(video_path, output_format)


def validate_audio_file(audio_path: str) -> Dict[str, Any]:
    extractor = AudioExtractor()
    return extractor.validate_audio(audio_path)


def convert_audio(input_path: str, output_format: str = "wav") -> str:
    extractor = AudioExtractor()
    return extractor.convert_audio_format(input_path, output_format)


def load_audio_file(audio_path: str, sr: int = 16000) -> Tuple[Any, int]:
    extractor = AudioExtractor()
    return extractor.load_audio(audio_path, sr)


def save_audio_file(audio: Any, sample_rate: int, output_path: str) -> bool:
    extractor = AudioExtractor()
    return extractor.save_audio(audio, sample_rate, output_path)


if __name__ == "__main__":
    test_video = "sample_video.mp4"
    if os.path.exists(test_video):
        audio_path = extract_audio_from_video(test_video)
        print(f"Extracted audio: {audio_path}")
        info = validate_audio_file(audio_path)
        print(f"Audio info: {info}")
