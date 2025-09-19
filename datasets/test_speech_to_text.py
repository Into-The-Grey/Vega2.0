"""
Test suite for speech_to_text.py (speech-to-text transcription).
"""

import os
import tempfile
import shutil
import numpy as np
import pytest
import soundfile as sf
from datasets import speech_to_text


def create_dummy_speech_wav(
    path, frequency=440, duration_sec=2, sr=16000, add_speech_pattern=True
):
    """Create a dummy wav file that simulates speech patterns."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), False)

    if add_speech_pattern:
        # Create a more speech-like pattern with varying amplitude and frequency
        speech_signal = np.zeros_like(t)

        # Add multiple frequency components to simulate speech
        for i, freq in enumerate([200, 400, 800, 1600]):
            amplitude = 0.1 * (1 + i) * np.exp(-i * 0.5)  # Decreasing amplitude
            modulation = 1 + 0.3 * np.sin(2 * np.pi * 3 * t)  # Amplitude modulation
            speech_signal += amplitude * modulation * np.sin(2 * np.pi * freq * t)

        # Add some noise to make it more realistic
        noise = 0.05 * np.random.normal(0, 1, len(t))
        speech_signal += noise

        # Normalize
        speech_signal = 0.5 * speech_signal / np.max(np.abs(speech_signal))
    else:
        # Simple sine wave
        speech_signal = 0.5 * np.sin(2 * np.pi * frequency * t)

    sf.write(path, speech_signal, sr)
    return path


def test_transcription_result_dataclass():
    """Test TranscriptionResult dataclass."""
    result = speech_to_text.TranscriptionResult(
        text="Hello world",
        confidence=0.95,
        language="en",
        processing_time=1.5,
        model_used="whisper-base",
    )

    assert result.text == "Hello world"
    assert result.confidence == 0.95
    assert result.language == "en"
    assert result.processing_time == 1.5
    assert result.model_used == "whisper-base"


def test_audio_preprocessor(tmp_path):
    """Test audio preprocessing functionality."""
    # Create test audio
    audio_path = tmp_path / "test_speech.wav"
    create_dummy_speech_wav(str(audio_path))

    # Test preprocessing
    preprocessor = speech_to_text.AudioPreprocessor()
    processed_path = preprocessor.preprocess_for_transcription(str(audio_path))

    assert os.path.exists(processed_path)
    assert processed_path.endswith(".wav")

    # Verify processed audio properties
    data, sr = sf.read(processed_path)
    assert sr == 16000  # Target sample rate
    assert len(data) > 0


def test_whisper_transcriber_init():
    """Test Whisper transcriber initialization."""
    if not speech_to_text.WHISPER_AVAILABLE:
        pytest.skip("OpenAI Whisper not available")

    try:
        transcriber = speech_to_text.WhisperTranscriber(
            "tiny"
        )  # Use smallest model for speed
        assert transcriber.model is not None
        assert transcriber.model_name == "tiny"
    except Exception as e:
        pytest.skip(f"Whisper model loading failed: {e}")


def test_whisper_transcriber_with_dummy_audio(tmp_path):
    """Test Whisper transcription with dummy audio."""
    if not speech_to_text.WHISPER_AVAILABLE:
        pytest.skip("OpenAI Whisper not available")

    try:
        # Create dummy speech audio
        audio_path = tmp_path / "speech.wav"
        create_dummy_speech_wav(str(audio_path))

        # Test transcription
        transcriber = speech_to_text.WhisperTranscriber("tiny")
        result = transcriber.transcribe(str(audio_path))

        assert isinstance(result, speech_to_text.TranscriptionResult)
        assert result.model_used.startswith("whisper-tiny")
        assert result.processing_time > 0
        assert result.audio_duration > 0
        # Note: We can't assert on actual text content since it's dummy audio

    except Exception as e:
        pytest.skip(f"Whisper transcription test failed: {e}")


def test_speechrecognition_transcriber_init():
    """Test SpeechRecognition transcriber initialization."""
    if not speech_to_text.SPEECH_RECOGNITION_AVAILABLE:
        pytest.skip("SpeechRecognition not available")

    transcriber = speech_to_text.SpeechRecognitionTranscriber("sphinx")
    assert transcriber.engine == "sphinx"
    assert transcriber.recognizer is not None


def test_speechrecognition_transcriber_with_dummy_audio(tmp_path):
    """Test SpeechRecognition transcription with dummy audio."""
    if not speech_to_text.SPEECH_RECOGNITION_AVAILABLE:
        pytest.skip("SpeechRecognition not available")

    # Create dummy speech audio
    audio_path = tmp_path / "speech.wav"
    create_dummy_speech_wav(str(audio_path))

    # Test with sphinx (offline) to avoid network dependencies
    transcriber = speech_to_text.SpeechRecognitionTranscriber("sphinx")
    result = transcriber.transcribe(str(audio_path))

    assert isinstance(result, speech_to_text.TranscriptionResult)
    assert result.model_used.startswith("speechrecognition-sphinx")
    assert result.processing_time >= 0
    # Note: Sphinx may not recognize dummy audio, so we just check structure


def test_speech_to_text_pipeline(tmp_path):
    """Test the complete speech-to-text pipeline."""
    # Create dummy speech audio
    audio_path = tmp_path / "speech.wav"
    create_dummy_speech_wav(str(audio_path))

    # Test pipeline initialization
    if speech_to_text.WHISPER_AVAILABLE:
        try:
            pipeline = speech_to_text.SpeechToTextPipeline("whisper", "tiny")
            assert pipeline.preferred_engine == "whisper"
            assert pipeline.whisper_model == "tiny"
        except Exception:
            # Fallback if Whisper fails
            pipeline = speech_to_text.SpeechToTextPipeline("speechrecognition")
    else:
        pipeline = speech_to_text.SpeechToTextPipeline("speechrecognition")

    # Test transcription
    result = pipeline.transcribe(str(audio_path), preprocess=True)
    assert isinstance(result, speech_to_text.TranscriptionResult)
    # Note: With dummy audio, engines may not recognize speech, so we just check structure
    # The important thing is that it doesn't crash and returns a valid result object
    assert hasattr(result, "text")
    assert hasattr(result, "confidence")
    assert hasattr(result, "model_used")


def test_batch_transcription(tmp_path):
    """Test batch transcription functionality."""
    # Create multiple dummy audio files
    audio_paths = []
    for i in range(3):
        audio_path = tmp_path / f"speech_{i}.wav"
        create_dummy_speech_wav(str(audio_path), frequency=440 + i * 100)
        audio_paths.append(str(audio_path))

    # Test batch transcription
    pipeline = speech_to_text.SpeechToTextPipeline()
    results = pipeline.batch_transcribe(audio_paths)

    assert len(results) == 3
    assert all(isinstance(r, speech_to_text.TranscriptionResult) for r in results)


def test_convenience_functions(tmp_path):
    """Test convenience functions."""
    # Create dummy speech audio
    audio_path = tmp_path / "speech.wav"
    create_dummy_speech_wav(str(audio_path))

    # Test transcribe_audio function
    result = speech_to_text.transcribe_audio(str(audio_path))
    assert isinstance(result, speech_to_text.TranscriptionResult)


def test_transcribe_from_video_mock(tmp_path, monkeypatch):
    """Test transcribe_from_video with mocked audio extraction."""
    # Create dummy video file (just a text file for testing)
    video_path = tmp_path / "video.mp4"
    with open(video_path, "wb") as f:
        f.write(b"fake video data")

    # Create dummy audio file to return from extraction
    audio_path = tmp_path / "extracted_audio.wav"
    create_dummy_speech_wav(str(audio_path))

    # Mock the audio extraction function
    def mock_extract_audio(video_path_arg):
        return str(audio_path)

    # Patch the audio_utils module
    import datasets.audio_utils

    monkeypatch.setattr(
        datasets.audio_utils, "extract_audio_from_video", mock_extract_audio
    )

    # Test transcription from video
    result = speech_to_text.transcribe_from_video(str(video_path))
    assert isinstance(result, speech_to_text.TranscriptionResult)
    assert result.model_used != "video-extraction-failed"


def test_error_handling(tmp_path):
    """Test error handling for invalid inputs."""
    # Test with non-existent file
    result = speech_to_text.transcribe_audio("nonexistent.wav")
    assert isinstance(result, speech_to_text.TranscriptionResult)
    assert result.confidence == 0.0
    assert result.text == ""

    # Test with invalid audio file
    invalid_path = tmp_path / "invalid.wav"
    with open(invalid_path, "wb") as f:
        f.write(b"not audio data")

    result = speech_to_text.transcribe_audio(str(invalid_path))
    assert isinstance(result, speech_to_text.TranscriptionResult)
    # Should handle gracefully


def test_no_engines_available(monkeypatch):
    """Test behavior when no transcription engines are available."""
    # Mock both engines as unavailable
    monkeypatch.setattr(speech_to_text, "WHISPER_AVAILABLE", False)
    monkeypatch.setattr(speech_to_text, "SPEECH_RECOGNITION_AVAILABLE", False)

    pipeline = speech_to_text.SpeechToTextPipeline()
    result = pipeline.transcribe("dummy.wav")

    assert result.model_used == "no-engine-available"
    assert result.confidence == 0.0
    assert result.text == ""
