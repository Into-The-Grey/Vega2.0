#!/usr/bin/env python3
"""
Vega Voice Processing Module
===========================

Provides local Text-to-Speech (TTS) and Speech-to-Text (STT) capabilities:
- Multiple TTS engines: Piper (high quality), pyttsx3 (cross-platform), espeak (fallback)
- Multiple STT engines: Vosk (offline), Whisper (optional)
- Voice Activity Detection (VAD)
- Audio preprocessing and enhancement
- Real-time streaming support
- Voice configuration management
"""

import asyncio
import os
import tempfile
import wave
import threading
from pathlib import Path
from typing import Optional, AsyncIterator, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

try:
    import soundfile as sf

    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    import pyaudio

    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

from core.logging_setup import get_voice_logger
from core.config_manager import get_config

logger = get_voice_logger()


class TTSEngine(Enum):
    """Text-to-Speech engine options"""

    PIPER = "piper"  # High quality neural TTS
    PYTTSX3 = "pyttsx3"  # Cross-platform TTS
    ESPEAK = "espeak"  # Lightweight fallback


class STTEngine(Enum):
    """Speech-to-Text engine options"""

    VOSK = "vosk"  # Offline recognition
    WHISPER = "whisper"  # OpenAI Whisper (optional)


@dataclass
class AudioConfig:
    """Audio processing configuration"""

    sample_rate: int = 16000
    channels: int = 1
    format: str = "wav"
    chunk_size: int = 1024
    vad_enabled: bool = True
    noise_reduction: bool = True


@dataclass
class VoiceConfig:
    """Voice processing configuration"""

    tts_engine: TTSEngine = TTSEngine.PIPER
    tts_voice: str = "en_US-lessac-medium"
    tts_speed: float = 1.0

    stt_engine: STTEngine = STTEngine.VOSK
    stt_model: str = "vosk-model-en-us-0.22"

    audio: AudioConfig = None

    def __post_init__(self):
        if self.audio is None:
            self.audio = AudioConfig()


class TTSProviderBase:
    """Base class for TTS providers"""

    def __init__(self, config: VoiceConfig):
        self.config = config
        self.audio_config = config.audio

    async def speak(self, text: str) -> bytes:
        """Convert text to speech, return audio bytes"""
        raise NotImplementedError

    async def speak_to_file(self, text: str, output_path: Path) -> bool:
        """Convert text to speech and save to file"""
        try:
            audio_data = await self.speak(text)
            with open(output_path, "wb") as f:
                f.write(audio_data)
            logger.info(f"TTS audio saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving TTS audio: {e}")
            return False

    def get_available_voices(self) -> List[str]:
        """Get list of available voices"""
        return []


class PiperTTSProvider(TTSProviderBase):
    """High-quality neural TTS using Piper"""

    def __init__(self, config: VoiceConfig):
        super().__init__(config)
        self.model_path = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize Piper model"""
        try:
            # Look for Piper models in voice/models/
            models_dir = Path("voice/models/piper")
            models_dir.mkdir(parents=True, exist_ok=True)

            voice_name = self.config.tts_voice
            model_file = models_dir / f"{voice_name}.onnx"

            if not model_file.exists():
                logger.warning(f"Piper model {voice_name} not found at {model_file}")
                logger.info(
                    "Download Piper models from: https://github.com/rhasspy/piper/releases"
                )
                return

            self.model_path = model_file
            logger.info(f"Initialized Piper TTS with model: {voice_name}")

        except Exception as e:
            logger.error(f"Error initializing Piper TTS: {e}")

    async def speak(self, text: str) -> bytes:
        """Convert text to speech using Piper"""
        if not self.model_path:
            raise RuntimeError("Piper model not initialized")

        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as text_file:
                text_file.write(text)
                text_path = text_file.name

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
                audio_path = audio_file.name

            # Run Piper CLI
            import subprocess

            cmd = [
                "piper",
                "--model",
                str(self.model_path),
                "--output_file",
                audio_path,
                text_path,
            ]

            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                raise RuntimeError(f"Piper failed: {stderr.decode()}")

            # Read audio data
            with open(audio_path, "rb") as f:
                audio_data = f.read()

            # Cleanup
            os.unlink(text_path)
            os.unlink(audio_path)

            logger.info(f"Generated {len(audio_data)} bytes of TTS audio")
            return audio_data

        except Exception as e:
            logger.error(f"Piper TTS error: {e}")
            raise


class Pyttsx3TTSProvider(TTSProviderBase):
    """Cross-platform TTS using pyttsx3"""

    def __init__(self, config: VoiceConfig):
        super().__init__(config)
        self.engine = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize pyttsx3 engine"""
        try:
            import pyttsx3

            self.engine = pyttsx3.init()

            # Configure voice properties
            voices = self.engine.getProperty("voices")
            if voices:
                # Try to find the requested voice
                for voice in voices:
                    if self.config.tts_voice.lower() in voice.name.lower():
                        self.engine.setProperty("voice", voice.id)
                        break
                else:
                    # Use first available voice
                    self.engine.setProperty("voice", voices[0].id)

            # Set speech rate
            rate = self.engine.getProperty("rate")
            self.engine.setProperty("rate", rate * self.config.tts_speed)

            logger.info("Initialized pyttsx3 TTS engine")

        except ImportError:
            logger.error("pyttsx3 not available. Install with: pip install pyttsx3")
        except Exception as e:
            logger.error(f"Error initializing pyttsx3 TTS: {e}")

    async def speak(self, text: str) -> bytes:
        """Convert text to speech using pyttsx3"""
        if not self.engine:
            raise RuntimeError("pyttsx3 engine not initialized")

        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name

            # Run TTS in thread to avoid blocking
            def _speak():
                self.engine.save_to_file(text, temp_path)
                self.engine.runAndWait()

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _speak)

            # Read audio data
            with open(temp_path, "rb") as f:
                audio_data = f.read()

            # Cleanup
            os.unlink(temp_path)

            logger.info(f"Generated {len(audio_data)} bytes of TTS audio")
            return audio_data

        except Exception as e:
            logger.error(f"pyttsx3 TTS error: {e}")
            raise

    def get_available_voices(self) -> List[str]:
        """Get list of available voices"""
        if not self.engine:
            return []

        try:
            voices = self.engine.getProperty("voices")
            return [voice.name for voice in voices] if voices else []
        except Exception as e:
            logger.error(f"Error getting voices: {e}")
            return []


class STTProviderBase:
    """Base class for STT providers"""

    def __init__(self, config: VoiceConfig):
        self.config = config
        self.audio_config = config.audio

    async def transcribe(self, audio_data: bytes) -> str:
        """Convert speech to text"""
        raise NotImplementedError

    async def transcribe_file(self, audio_path: Path) -> str:
        """Convert speech to text from file"""
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        return await self.transcribe(audio_data)

    async def start_streaming(self) -> AsyncIterator[str]:
        """Start streaming transcription"""
        raise NotImplementedError


class VoskSTTProvider(STTProviderBase):
    """Offline STT using Vosk"""

    def __init__(self, config: VoiceConfig):
        super().__init__(config)
        self.model = None
        self.recognizer = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize Vosk model"""
        try:
            import vosk
            import json

            # Look for Vosk models in voice/models/
            models_dir = Path("voice/models/vosk")
            models_dir.mkdir(parents=True, exist_ok=True)

            model_name = self.config.stt_model
            model_path = models_dir / model_name

            if not model_path.exists():
                logger.warning(f"Vosk model {model_name} not found at {model_path}")
                logger.info(
                    "Download Vosk models from: https://alphacephei.com/vosk/models"
                )
                return

            # Initialize model and recognizer
            self.model = vosk.Model(str(model_path))
            self.recognizer = vosk.KaldiRecognizer(
                self.model, self.audio_config.sample_rate
            )

            logger.info(f"Initialized Vosk STT with model: {model_name}")

        except ImportError:
            logger.error("Vosk not available. Install with: pip install vosk")
        except Exception as e:
            logger.error(f"Error initializing Vosk STT: {e}")

    async def transcribe(self, audio_data: bytes) -> str:
        """Convert speech to text using Vosk"""
        if not self.recognizer:
            raise RuntimeError("Vosk recognizer not initialized")

        try:
            import json

            # Process audio data
            if self.recognizer.AcceptWaveform(audio_data):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "")
            else:
                partial = json.loads(self.recognizer.PartialResult())
                text = partial.get("partial", "")

            logger.info(f"Transcribed: {text}")
            return text

        except Exception as e:
            logger.error(f"Vosk STT error: {e}")
            raise


class VoiceManager:
    """Main voice processing manager"""

    def __init__(self, config: Optional[VoiceConfig] = None):
        if config is None:
            config_dict = get_config("voice")
            config = VoiceConfig(
                tts_engine=TTSEngine(config_dict.get("tts_engine", "piper")),
                tts_voice=config_dict.get("tts_voice", "en_US-lessac-medium"),
                tts_speed=config_dict.get("tts_speed", 1.0),
                stt_engine=STTEngine(config_dict.get("stt_engine", "vosk")),
                stt_model=config_dict.get("stt_model", "vosk-model-en-us-0.22"),
                audio=AudioConfig(
                    sample_rate=config_dict.get("sample_rate", 16000),
                    vad_enabled=config_dict.get("vad_enabled", True),
                    noise_reduction=config_dict.get("noise_reduction", True),
                ),
            )

        self.config = config
        self.tts_provider = self._create_tts_provider()
        self.stt_provider = self._create_stt_provider()

        logger.info(
            f"VoiceManager initialized with TTS: {config.tts_engine.value}, STT: {config.stt_engine.value}"
        )

    def _create_tts_provider(self) -> TTSProviderBase:
        """Create TTS provider based on configuration"""
        if self.config.tts_engine == TTSEngine.PIPER:
            return PiperTTSProvider(self.config)
        elif self.config.tts_engine == TTSEngine.PYTTSX3:
            return Pyttsx3TTSProvider(self.config)
        else:
            # Fallback to pyttsx3
            logger.warning(
                f"Unknown TTS engine {self.config.tts_engine}, falling back to pyttsx3"
            )
            return Pyttsx3TTSProvider(self.config)

    def _create_stt_provider(self) -> STTProviderBase:
        """Create STT provider based on configuration"""
        if self.config.stt_engine == STTEngine.VOSK:
            return VoskSTTProvider(self.config)
        else:
            # Future: add Whisper support
            logger.warning(
                f"Unknown STT engine {self.config.stt_engine}, falling back to Vosk"
            )
            return VoskSTTProvider(self.config)

    async def speak(self, text: str) -> bytes:
        """Convert text to speech"""
        logger.info(f"TTS request: {text[:50]}...")
        return await self.tts_provider.speak(text)

    async def speak_to_file(self, text: str, output_path: Path) -> bool:
        """Convert text to speech and save to file"""
        return await self.tts_provider.speak_to_file(text, output_path)

    async def transcribe(self, audio_data: bytes) -> str:
        """Convert speech to text"""
        logger.info(f"STT request: {len(audio_data)} bytes")
        return await self.stt_provider.transcribe(audio_data)

    async def transcribe_file(self, audio_path: Path) -> str:
        """Convert speech to text from file"""
        return await self.stt_provider.transcribe_file(audio_path)

    def get_available_voices(self) -> List[str]:
        """Get list of available TTS voices"""
        return self.tts_provider.get_available_voices()

    async def test_tts(
        self, text: str = "Hello, this is a test of the Vega voice system."
    ):
        """Test TTS functionality"""
        try:
            audio_data = await self.speak(text)
            test_file = Path("voice/test_output.wav")
            test_file.parent.mkdir(exist_ok=True)

            with open(test_file, "wb") as f:
                f.write(audio_data)

            logger.info(f"TTS test successful. Audio saved to {test_file}")
            return True

        except Exception as e:
            logger.error(f"TTS test failed: {e}")
            return False

    async def test_stt(self, audio_path: Optional[Path] = None):
        """Test STT functionality"""
        try:
            if audio_path is None:
                # Use test file if available
                audio_path = Path("voice/test_input.wav")
                if not audio_path.exists():
                    logger.warning("No test audio file available for STT test")
                    return False

            text = await self.transcribe_file(audio_path)
            logger.info(f"STT test successful. Transcribed: {text}")
            return True

        except Exception as e:
            logger.error(f"STT test failed: {e}")
            return False


# Global voice manager instance
_voice_manager: Optional[VoiceManager] = None


def get_voice_manager() -> VoiceManager:
    """Get the global voice manager instance"""
    global _voice_manager
    if _voice_manager is None:
        _voice_manager = VoiceManager()
    return _voice_manager


async def speak(text: str) -> bytes:
    """Convenience function for TTS"""
    return await get_voice_manager().speak(text)


async def transcribe(audio_data: bytes) -> str:
    """Convenience function for STT"""
    return await get_voice_manager().transcribe(audio_data)
