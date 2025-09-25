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

# Export all audio processing classes and functions
from .realtime import (
    RealtimeAudioProcessor,
    VoiceActivityDetector,
    NoiseReducer,
    AcousticFingerprinter,
)

from .mir import (
    BeatTracker,
    ChordDetector,
    GenreClassifier,
    MoodAnalyzer,
    MusicInformationRetrieval,
)

from .enhancement import (
    SpectralNoiseReducer,
    EchoRemover,
    AudioRestorer,
    DynamicRangeProcessor,
    AudioEqualizer,
    AudioEnhancer,
)

from .spatial import (
    SpatialAudioProcessor,
    HRTFProcessor,
    AmbisonicProcessor,
    BinauralProcessor,
    Position3D,
    SpatialSource,
    Listener,
    RoomAcoustics,
)

from .synthesis import (
    AudioSynthesizer,
    WaveformGenerator,
    FMSynthesizer,
    GranularSynthesizer,
    PhysicalModelingSynthesizer,
    WavetableSynthesizer,
    EnvelopeGenerator,
    AudioFilter,
    Oscillator,
    ModulationSource,
    FilterConfig,
    ADSREnvelope,
    GranularConfig,
    SynthConfig,
    WaveformType,
    FilterType,
    EnvelopeType,
    ModulationType,
    SynthesisMethod,
    SynthesisError,
)

__all__ = [
    # Real-time Audio Processing
    "RealtimeAudioProcessor",
    "VoiceActivityDetector",
    "NoiseReducer",
    "AcousticFingerprinter",
    # Music Information Retrieval
    "MusicInformationRetrieval",
    "BeatTracker",
    "ChordDetector",
    "GenreClassifier",
    "MoodAnalyzer",
    # Audio Enhancement
    "AudioEnhancer",
    "SpectralNoiseReducer",
    "EchoRemover",
    "AudioRestorer",
    "DynamicRangeProcessor",
    "AudioEqualizer",
    # Spatial Audio Processing
    "SpatialAudioProcessor",
    "HRTFProcessor",
    "AmbisonicProcessor",
    "BinauralProcessor",
    "Position3D",
    "SpatialSource",
    "Listener",
    "RoomAcoustics",
    # Audio Synthesis & Generation
    "AudioSynthesizer",
    "WaveformGenerator",
    "FMSynthesizer",
    "GranularSynthesizer",
    "PhysicalModelingSynthesizer",
    "WavetableSynthesizer",
    "EnvelopeGenerator",
    "AudioFilter",
    # Configuration Classes
    "Oscillator",
    "ModulationSource",
    "FilterConfig",
    "ADSREnvelope",
    "GranularConfig",
    "SynthConfig",
    # Enums
    "WaveformType",
    "FilterType",
    "EnvelopeType",
    "ModulationType",
    "SynthesisMethod",
    # Exceptions
    "SynthesisError",
]

__version__ = "2.0.0"

__version__ = "2.0.0"
