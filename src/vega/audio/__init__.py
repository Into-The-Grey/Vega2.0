"""
Vega 2.0 Audio Processing Module

Advanced audio processing capabilities including:
- Real-time audio analysis
- Music Information Retrieval (MIR)
- Audio enhancement and restoration
- Spatial audio processing
- Audio synthesis and generation

This module provides comprehensive audio processing tools
for the Vega2.0 AI platform.
"""

from .realtime import RealtimeAudioProcessor, VoiceActivityDetector
from .mir import MusicInformationRetrieval, BeatTracker, ChordDetector
from .enhancement import AudioEnhancer, NoiseReducer, EchoRemover
from .spatial import SpatialAudioProcessor, BinauralProcessor
from .synthesis import AudioSynthesizer, NeuralVocoder, VoiceCloner

__all__ = [
    "RealtimeAudioProcessor",
    "VoiceActivityDetector",
    "MusicInformationRetrieval",
    "BeatTracker",
    "ChordDetector",
    "AudioEnhancer",
    "NoiseReducer",
    "EchoRemover",
    "SpatialAudioProcessor",
    "BinauralProcessor",
    "AudioSynthesizer",
    "NeuralVocoder",
    "VoiceCloner",
]

__version__ = "2.0.0"
