#!/usr/bin/env python3
"""
Simple provider shims for tests.

We provide base classes and light mock-friendly implementations
that the tests will import and patch.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


class TTSProviderBase:
    def is_available(self) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def synthesize(
        self, text: str, voice: str | None = None, **kwargs
    ) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError

    def list_voices(self) -> list:  # pragma: no cover - interface
        raise NotImplementedError


class STTProviderBase:
    def is_available(self) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def transcribe(
        self, audio_data: bytes, **kwargs
    ) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def list_models(self) -> list:  # pragma: no cover - interface
        raise NotImplementedError


# Optional real providers can be added here; tests patch these symbols.
class PiperTTSProvider(TTSProviderBase):
    def __init__(self, *args, **kwargs):
        pass

    def is_available(self) -> bool:
        return True

    def synthesize(self, text: str, voice: str | None = None, **kwargs) -> bytes:
        return b"piper_audio_data" if text else b""

    def list_voices(self) -> list:
        return ["voice1", "voice2", "voice3"]


class VoskSTTProvider(STTProviderBase):
    def __init__(self, *args, **kwargs):
        pass

    def is_available(self) -> bool:
        return True

    def transcribe(self, audio_data: bytes, **kwargs) -> str:
        return "vosk transcription result" if audio_data is not None else ""

    def list_models(self) -> list:
        return ["model1", "model2", "model3"]
