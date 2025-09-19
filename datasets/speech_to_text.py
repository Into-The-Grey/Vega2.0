"""
Speech-to-text transcription utilities for Vega 2.0 multi-modal learning.

This module provides:
- Speech-to-text transcription using OpenAI Whisper
- Alternative transcription using SpeechRecognition
- Audio preprocessing for optimal transcription
- Batch transcription capabilities
"""

import os
import tempfile
import shutil
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path
import logging
import time
from dataclasses import dataclass, asdict
import json

# Whisper imports
try:
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logging.warning(
        "OpenAI Whisper not available. Install with: pip install openai-whisper"
    )

# SpeechRecognition imports
try:
    import speech_recognition as sr

    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    logging.warning(
        "SpeechRecognition not available. Install with: pip install SpeechRecognition"
    )

# Audio processing imports
import librosa
import soundfile as sf
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Represents a speech-to-text transcription result."""

    text: str
    confidence: float
    language: Optional[str] = None
    segments: Optional[List[Dict]] = None
    processing_time: float = 0.0
    model_used: str = "unknown"
    audio_duration: float = 0.0


class WhisperTranscriber:
    """OpenAI Whisper-based speech-to-text transcription."""

    def __init__(self, model_name: str = "base"):
        if not WHISPER_AVAILABLE:
            raise ImportError(
                "OpenAI Whisper not available. Install with: pip install openai-whisper"
            )

        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the Whisper model."""
        try:
            self.model = whisper.load_model(self.model_name)
            logger.info(f"Loaded Whisper model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model {self.model_name}: {e}")
            raise

    def transcribe(
        self, audio_path: str, language: Optional[str] = None
    ) -> TranscriptionResult:
        """Transcribe audio file using Whisper."""
        start_time = time.time()

        try:
            # Get audio duration
            audio_duration = librosa.get_duration(path=audio_path)

            # Transcribe with Whisper
            result = self.model.transcribe(audio_path, language=language)

            processing_time = time.time() - start_time

            # Extract segments if available
            segments = []
            if "segments" in result:
                for seg in result["segments"]:
                    segments.append(
                        {
                            "start": seg.get("start", 0),
                            "end": seg.get("end", 0),
                            "text": seg.get("text", ""),
                            "confidence": seg.get("avg_logprob", 0),
                        }
                    )

            return TranscriptionResult(
                text=result["text"].strip(),
                confidence=1.0,  # Whisper doesn't provide confidence scores
                language=result.get("language"),
                segments=segments if segments else None,
                processing_time=processing_time,
                model_used=f"whisper-{self.model_name}",
                audio_duration=audio_duration,
            )

        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return TranscriptionResult(
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                model_used=f"whisper-{self.model_name}-error",
            )


class SpeechRecognitionTranscriber:
    """SpeechRecognition-based speech-to-text transcription."""

    def __init__(self, engine: str = "google"):
        if not SPEECH_RECOGNITION_AVAILABLE:
            raise ImportError(
                "SpeechRecognition not available. Install with: pip install SpeechRecognition"
            )

        self.engine = engine
        self.recognizer = sr.Recognizer()

    def transcribe(
        self, audio_path: str, language: str = "en-US"
    ) -> TranscriptionResult:
        """Transcribe audio file using SpeechRecognition."""
        start_time = time.time()

        try:
            # Get audio duration
            audio_duration = librosa.get_duration(path=audio_path)

            # Load audio file
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)

            # Perform recognition
            if self.engine == "google":
                text = self.recognizer.recognize_google(audio, language=language)
                confidence = 1.0  # Google doesn't provide confidence
            elif self.engine == "sphinx":
                text = self.recognizer.recognize_sphinx(audio)
                confidence = 0.8  # Estimated confidence for offline recognition
            else:
                raise ValueError(f"Unsupported engine: {self.engine}")

            processing_time = time.time() - start_time

            return TranscriptionResult(
                text=text.strip(),
                confidence=confidence,
                language=language,
                processing_time=processing_time,
                model_used=f"speechrecognition-{self.engine}",
                audio_duration=audio_duration,
            )

        except sr.UnknownValueError:
            logger.warning("Speech recognition could not understand audio")
            return TranscriptionResult(
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                model_used=f"speechrecognition-{self.engine}-no-speech",
            )
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {e}")
            return TranscriptionResult(
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                model_used=f"speechrecognition-{self.engine}-error",
            )
        except Exception as e:
            logger.error(f"SpeechRecognition transcription failed: {e}")
            return TranscriptionResult(
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                model_used=f"speechrecognition-{self.engine}-error",
            )


class AudioPreprocessor:
    """Preprocess audio for optimal transcription."""

    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()

    def __del__(self):
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def preprocess_for_transcription(
        self, audio_path: str, target_sr: int = 16000
    ) -> str:
        """Preprocess audio file for better transcription results."""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None, mono=True)

            # Resample to target sample rate
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                sr = target_sr

            # Normalize audio
            y = librosa.util.normalize(y)

            # Remove silence (basic)
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)

            # Save preprocessed audio
            output_path = os.path.join(self.temp_dir, "preprocessed_audio.wav")
            sf.write(output_path, y_trimmed, sr)

            logger.info(f"Preprocessed audio saved to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            return audio_path  # Return original path if preprocessing fails


class SpeechToTextPipeline:
    """Complete speech-to-text pipeline with multiple backends."""

    def __init__(self, preferred_engine: str = "whisper", whisper_model: str = "base"):
        self.preferred_engine = preferred_engine
        self.whisper_model = whisper_model
        self.preprocessor = AudioPreprocessor()

        # Initialize transcribers
        self.whisper_transcriber = None
        self.sr_transcriber = None

        if preferred_engine == "whisper" and WHISPER_AVAILABLE:
            try:
                self.whisper_transcriber = WhisperTranscriber(whisper_model)
            except Exception as e:
                logger.warning(f"Failed to initialize Whisper: {e}")

        if SPEECH_RECOGNITION_AVAILABLE:
            try:
                self.sr_transcriber = SpeechRecognitionTranscriber("google")
            except Exception as e:
                logger.warning(f"Failed to initialize SpeechRecognition: {e}")

    def transcribe(
        self, audio_path: str, preprocess: bool = True, language: Optional[str] = None
    ) -> TranscriptionResult:
        """Transcribe audio using the best available method."""

        # Preprocess audio if requested
        if preprocess:
            processed_path = self.preprocessor.preprocess_for_transcription(audio_path)
        else:
            processed_path = audio_path

        # Try preferred engine first
        if self.preferred_engine == "whisper" and self.whisper_transcriber:
            result = self.whisper_transcriber.transcribe(processed_path, language)
            if result.text:  # If we got a result
                return result

        # Fallback to SpeechRecognition
        if self.sr_transcriber:
            sr_language = language if language else "en-US"
            result = self.sr_transcriber.transcribe(processed_path, sr_language)
            if result.text:
                return result

        # If neither worked, return empty result
        return TranscriptionResult(
            text="", confidence=0.0, model_used="no-engine-available"
        )

    def batch_transcribe(
        self, audio_paths: List[str], preprocess: bool = True
    ) -> List[TranscriptionResult]:
        """Transcribe multiple audio files."""
        results = []
        for audio_path in audio_paths:
            result = self.transcribe(audio_path, preprocess)
            results.append(result)
        return results


# Convenience functions
def transcribe_audio(
    audio_path: str,
    engine: str = "whisper",
    model: str = "base",
    language: Optional[str] = None,
) -> TranscriptionResult:
    """Transcribe an audio file using the specified engine."""
    pipeline = SpeechToTextPipeline(preferred_engine=engine, whisper_model=model)
    return pipeline.transcribe(audio_path, language=language)


def transcribe_from_video(
    video_path: str, engine: str = "whisper", model: str = "base"
) -> TranscriptionResult:
    """Extract audio from video and transcribe it."""
    from datasets import audio_utils

    # Extract audio from video
    audio_path = audio_utils.extract_audio_from_video(video_path)

    if not audio_path or not os.path.exists(audio_path):
        return TranscriptionResult(
            text="", confidence=0.0, model_used="video-extraction-failed"
        )

    # Transcribe the extracted audio
    return transcribe_audio(audio_path, engine, model)


if __name__ == "__main__":
    # Example usage
    test_audio = "sample_audio.wav"
    if os.path.exists(test_audio):
        result = transcribe_audio(test_audio)
        print(f"Transcription: {result.text}")
        print(f"Confidence: {result.confidence}")
        print(f"Model: {result.model_used}")
        print(f"Processing time: {result.processing_time:.2f}s")
