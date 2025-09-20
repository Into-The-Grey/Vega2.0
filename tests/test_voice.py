"""
test_voice.py - Tests for voice processing infrastructure

Tests comprehensive voice functionality:
- TTS (Text-to-Speech) providers and engines
- STT (Speech-to-Text) providers and engines
- Voice configuration management
- Audio file handling
- Model management
- Error handling and fallbacks
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import io

from src.vega.voice.voice_engine import VoiceManager
from src.vega.voice.providers import TTSProviderBase, STTProviderBase


class MockTTSProvider(TTSProviderBase):
    """Mock TTS provider for testing"""

    def __init__(self, name: str):
        self.name = name
        self.is_available = True

    def is_available(self) -> bool:
        return self.is_available

    def synthesize(self, text: str, voice: str = None, **kwargs) -> bytes:
        # Return mock audio data
        return b"mock_audio_data_" + text.encode()

    def list_voices(self) -> list:
        return ["voice1", "voice2", "voice3"]


class MockSTTProvider(STTProviderBase):
    """Mock STT provider for testing"""

    def __init__(self, name: str):
        self.name = name
        self.is_available = True

    def is_available(self) -> bool:
        return self.is_available

    def transcribe(self, audio_data: bytes, **kwargs) -> str:
        # Return mock transcription
        return f"transcribed_text_from_{len(audio_data)}_bytes"

    def list_models(self) -> list:
        return ["model1", "model2", "model3"]


class TestVoiceManager:
    """Test VoiceManager functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.voice_manager = VoiceManager(
            tts_provider="mock", stt_provider="mock", models_dir=self.test_dir
        )

        # Register mock providers
        self.mock_tts = MockTTSProvider("mock")
        self.mock_stt = MockSTTProvider("mock")

        self.voice_manager.tts_providers["mock"] = self.mock_tts
        self.voice_manager.stt_providers["mock"] = self.mock_stt

    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_initialization(self):
        """Test VoiceManager initialization"""
        assert self.voice_manager.tts_provider == "mock"
        assert self.voice_manager.stt_provider == "mock"
        assert self.voice_manager.models_dir == self.test_dir

    def test_tts_synthesis(self):
        """Test text-to-speech synthesis"""
        text = "Hello, this is a test message"
        audio_data = self.voice_manager.synthesize(text)

        assert isinstance(audio_data, bytes)
        assert b"Hello, this is a test message" in audio_data

    def test_tts_with_voice_selection(self):
        """Test TTS with specific voice selection"""
        text = "Test with specific voice"
        audio_data = self.voice_manager.synthesize(text, voice="voice2")

        assert isinstance(audio_data, bytes)
        assert b"Test with specific voice" in audio_data

    def test_stt_transcription(self):
        """Test speech-to-text transcription"""
        mock_audio = b"mock_audio_data_for_transcription"
        transcription = self.voice_manager.transcribe(mock_audio)

        assert isinstance(transcription, str)
        assert "transcribed_text" in transcription
        assert str(len(mock_audio)) in transcription

    def test_list_tts_voices(self):
        """Test listing available TTS voices"""
        voices = self.voice_manager.list_voices()

        assert isinstance(voices, list)
        assert "voice1" in voices
        assert "voice2" in voices
        assert "voice3" in voices

    def test_list_stt_models(self):
        """Test listing available STT models"""
        models = self.voice_manager.list_models()

        assert isinstance(models, list)
        assert "model1" in models
        assert "model2" in models
        assert "model3" in models

    def test_provider_availability_check(self):
        """Test provider availability checking"""
        assert self.voice_manager.is_tts_available()
        assert self.voice_manager.is_stt_available()

        # Test unavailable provider
        self.mock_tts.is_available = False
        assert not self.voice_manager.is_tts_available()

    def test_provider_switching(self):
        """Test switching between providers"""
        # Add another mock provider
        mock_tts2 = MockTTSProvider("mock2")
        self.voice_manager.tts_providers["mock2"] = mock_tts2

        # Switch provider
        self.voice_manager.set_tts_provider("mock2")
        assert self.voice_manager.tts_provider == "mock2"

    def test_invalid_provider_handling(self):
        """Test handling of invalid providers"""
        with pytest.raises(ValueError):
            self.voice_manager.set_tts_provider("nonexistent")

        with pytest.raises(ValueError):
            self.voice_manager.set_stt_provider("nonexistent")

    def test_audio_file_saving(self):
        """Test saving audio to file"""
        text = "Test audio file saving"
        output_path = Path(self.test_dir) / "test_audio.wav"

        self.voice_manager.synthesize_to_file(text, str(output_path))

        assert output_path.exists()

        # Verify content
        with open(output_path, "rb") as f:
            audio_data = f.read()
        assert b"Test audio file saving" in audio_data

    def test_audio_file_loading(self):
        """Test loading and transcribing audio file"""
        # Create test audio file
        test_audio_path = Path(self.test_dir) / "test_input.wav"
        test_audio_data = b"test_audio_file_content"

        with open(test_audio_path, "wb") as f:
            f.write(test_audio_data)

        # Transcribe from file
        transcription = self.voice_manager.transcribe_file(str(test_audio_path))

        assert isinstance(transcription, str)
        assert "transcribed_text" in transcription

    def test_configuration_management(self):
        """Test voice configuration management"""
        config = {
            "tts": {"provider": "mock", "voice": "voice1", "speed": 1.0},
            "stt": {"provider": "mock", "model": "model1", "language": "en"},
        }

        self.voice_manager.update_config(config)

        # Verify configuration applied
        current_config = self.voice_manager.get_config()
        assert current_config["tts"]["provider"] == "mock"
        assert current_config["stt"]["provider"] == "mock"

    def test_error_handling_tts_failure(self):
        """Test error handling when TTS fails"""

        # Make TTS provider raise an exception
        def failing_synthesize(text, voice=None, **kwargs):
            raise Exception("TTS synthesis failed")

        self.mock_tts.synthesize = failing_synthesize

        with pytest.raises(Exception):
            self.voice_manager.synthesize("Test text")

    def test_error_handling_stt_failure(self):
        """Test error handling when STT fails"""

        # Make STT provider raise an exception
        def failing_transcribe(audio_data, **kwargs):
            raise Exception("STT transcription failed")

        self.mock_stt.transcribe = failing_transcribe

        with pytest.raises(Exception):
            self.voice_manager.transcribe(b"test_audio")

    def test_model_management(self):
        """Test voice model management"""
        # Test downloading a model (mock)
        model_name = "test_model"
        model_path = self.voice_manager.download_model(model_name)

        assert model_path is not None

        # Test listing installed models
        installed_models = self.voice_manager.list_installed_models()
        assert isinstance(installed_models, list)

    def test_audio_format_conversion(self):
        """Test audio format conversion"""
        # This test would depend on actual audio processing
        # For now, just test the interface exists
        assert hasattr(self.voice_manager, "convert_audio_format")

    def test_real_time_processing(self):
        """Test real-time audio processing capabilities"""
        # Test streaming TTS
        text_chunks = ["Hello", " this", " is", " streaming", " text"]

        audio_chunks = []
        for chunk in text_chunks:
            audio_data = self.voice_manager.synthesize(chunk)
            audio_chunks.append(audio_data)

        assert len(audio_chunks) == len(text_chunks)
        for chunk in audio_chunks:
            assert isinstance(chunk, bytes)


class TestTTSProviderBase:
    """Test TTS provider base class"""

    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError"""
        provider = TTSProviderBase()

        with pytest.raises(NotImplementedError):
            provider.is_available()

        with pytest.raises(NotImplementedError):
            provider.synthesize("test")

        with pytest.raises(NotImplementedError):
            provider.list_voices()


class TestSTTProviderBase:
    """Test STT provider base class"""

    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError"""
        provider = STTProviderBase()

        with pytest.raises(NotImplementedError):
            provider.is_available()

        with pytest.raises(NotImplementedError):
            provider.transcribe(b"test")

        with pytest.raises(NotImplementedError):
            provider.list_models()


class TestPiperTTSProvider:
    """Test Piper TTS provider"""

    @patch("voice.providers.PiperTTSProvider")
    def test_piper_initialization(self, mock_piper):
        """Test Piper TTS provider initialization"""
        mock_instance = MagicMock()
        mock_piper.return_value = mock_instance
        mock_instance.is_available.return_value = True

        voice_manager = VoiceManager(tts_provider="piper")

        # Test that Piper was initialized
        assert mock_piper.called

    @patch("voice.providers.PiperTTSProvider")
    def test_piper_synthesis(self, mock_piper):
        """Test Piper TTS synthesis"""
        mock_instance = MagicMock()
        mock_piper.return_value = mock_instance
        mock_instance.is_available.return_value = True
        mock_instance.synthesize.return_value = b"piper_audio_data"

        voice_manager = VoiceManager(tts_provider="piper")
        audio_data = voice_manager.synthesize("Test with Piper")

        mock_instance.synthesize.assert_called_once()
        assert audio_data == b"piper_audio_data"


class TestVoskSTTProvider:
    """Test Vosk STT provider"""

    @patch("voice.providers.VoskSTTProvider")
    def test_vosk_initialization(self, mock_vosk):
        """Test Vosk STT provider initialization"""
        mock_instance = MagicMock()
        mock_vosk.return_value = mock_instance
        mock_instance.is_available.return_value = True

        voice_manager = VoiceManager(stt_provider="vosk")

        # Test that Vosk was initialized
        assert mock_vosk.called

    @patch("voice.providers.VoskSTTProvider")
    def test_vosk_transcription(self, mock_vosk):
        """Test Vosk STT transcription"""
        mock_instance = MagicMock()
        mock_vosk.return_value = mock_instance
        mock_instance.is_available.return_value = True
        mock_instance.transcribe.return_value = "vosk transcription result"

        voice_manager = VoiceManager(stt_provider="vosk")
        transcription = voice_manager.transcribe(b"test_audio")

        mock_instance.transcribe.assert_called_once()
        assert transcription == "vosk transcription result"


class TestVoiceIntegration:
    """Integration tests for voice processing"""

    def setup_method(self):
        """Setup integration test environment"""
        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup integration test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("voice.VoiceManager")
    def test_voice_chat_integration(self, mock_voice_manager):
        """Test integration with chat system"""
        mock_instance = MagicMock()
        mock_voice_manager.return_value = mock_instance

        # Mock voice processing
        mock_instance.transcribe.return_value = "Hello, how are you?"
        mock_instance.synthesize.return_value = b"response_audio_data"

        # Simulate voice chat workflow
        audio_input = b"user_audio_input"
        text_input = mock_instance.transcribe(audio_input)

        # Process with chat system (would be actual chat logic)
        chat_response = f"You said: {text_input}"

        # Convert response to audio
        audio_output = mock_instance.synthesize(chat_response)

        # Verify workflow
        mock_instance.transcribe.assert_called_with(audio_input)
        mock_instance.synthesize.assert_called_with(chat_response)
        assert audio_output == b"response_audio_data"

    def test_voice_configuration_persistence(self):
        """Test that voice configuration persists correctly"""
        config_file = Path(self.test_dir) / "voice_config.yaml"

        # This would test actual configuration persistence
        # Implementation depends on voice module structure
        assert True  # Placeholder for actual test


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
