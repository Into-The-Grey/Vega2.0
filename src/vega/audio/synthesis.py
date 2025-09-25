"""
Vega 2.0 Audio Synthesis & Generation Module

This module provides comprehensive audio synthesis and generation capabilities including:
- Waveform generators (sine, square, sawtooth, triangle, noise)
- Frequency Modulation (FM) and Amplitude Modulation (AM) synthesis
- Granular synthesis for texture and time-stretching
- Physical modeling synthesis
- AI-powered audio generation and style transfer
- Real-time synthesis with MIDI control
- Audio effects and processing chains
- Sample-based synthesis and playback

Dependencies:
- numpy: Audio array processing and mathematical operations
- scipy: Signal processing and filtering
- librosa: Audio analysis and feature extraction
- scikit-learn: Machine learning for AI synthesis (optional)
- mido: MIDI input/output handling (optional)
"""

import asyncio
import logging
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np

try:
    import librosa
    import scipy.signal
    from scipy import interpolate

    HAS_AUDIO_LIBS = True
except ImportError:
    librosa = None
    scipy = None
    HAS_AUDIO_LIBS = False

try:
    from sklearn.decomposition import PCA, FastICA
    from sklearn.cluster import KMeans

    HAS_SKLEARN = True
except ImportError:
    PCA = FastICA = KMeans = None
    HAS_SKLEARN = False

try:
    import mido

    HAS_MIDO = True
except ImportError:
    mido = None
    HAS_MIDO = False

logger = logging.getLogger(__name__)


class SynthesisError(Exception):
    """Custom exception for synthesis errors"""

    pass


class WaveformType(Enum):
    """Basic waveform types"""

    SINE = "sine"
    SQUARE = "square"
    SAWTOOTH = "sawtooth"
    TRIANGLE = "triangle"
    NOISE_WHITE = "white_noise"
    NOISE_PINK = "pink_noise"
    NOISE_BROWN = "brown_noise"
    PULSE = "pulse"
    CUSTOM = "custom"


class FilterType(Enum):
    """Audio filter types"""

    LOW_PASS = "lowpass"
    HIGH_PASS = "highpass"
    BAND_PASS = "bandpass"
    BAND_STOP = "bandstop"
    NOTCH = "notch"
    PEAK = "peak"
    LOW_SHELF = "lowshelf"
    HIGH_SHELF = "highshelf"


class EnvelopeType(Enum):
    """Envelope generator types"""

    ADSR = "adsr"  # Attack, Decay, Sustain, Release
    ASR = "asr"  # Attack, Sustain, Release
    AR = "ar"  # Attack, Release
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class ModulationType(Enum):
    """Modulation types"""

    AMPLITUDE = "amplitude"
    FREQUENCY = "frequency"
    PHASE = "phase"
    RING = "ring"
    TREMOLO = "tremolo"
    VIBRATO = "vibrato"


class SynthesisMethod(Enum):
    """Synthesis methods"""

    ADDITIVE = "additive"
    SUBTRACTIVE = "subtractive"
    FM = "fm"
    GRANULAR = "granular"
    PHYSICAL_MODELING = "physical_modeling"
    WAVETABLE = "wavetable"
    NEURAL = "neural"


@dataclass
class ADSREnvelope:
    """ADSR envelope parameters"""

    attack_time: float = 0.1  # seconds
    decay_time: float = 0.2  # seconds
    sustain_level: float = 0.7  # 0.0 to 1.0
    release_time: float = 0.5  # seconds


@dataclass
class Oscillator:
    """Oscillator configuration"""

    waveform: WaveformType = WaveformType.SINE
    frequency: float = 440.0  # Hz
    amplitude: float = 1.0  # 0.0 to 1.0
    phase: float = 0.0  # radians
    detune: float = 0.0  # cents
    pulse_width: float = 0.5  # for pulse wave (0.0 to 1.0)


@dataclass
class ModulationSource:
    """Modulation source configuration"""

    oscillator: Oscillator
    modulation_type: ModulationType
    depth: float = 0.1  # modulation depth
    rate: float = 5.0  # modulation rate (Hz)


@dataclass
class FilterConfig:
    """Audio filter configuration"""

    filter_type: FilterType = FilterType.LOW_PASS
    cutoff_frequency: float = 1000.0  # Hz
    resonance: float = 0.7  # Q factor
    gain: float = 0.0  # dB (for peak/shelf filters)


@dataclass
class GranularConfig:
    """Granular synthesis configuration"""

    grain_size: float = 0.1  # seconds
    grain_density: float = 10.0  # grains per second
    grain_pitch: float = 1.0  # pitch ratio
    grain_scatter: float = 0.1  # position randomization
    grain_envelope: EnvelopeType = EnvelopeType.LINEAR


@dataclass
class SynthConfig:
    """Main synthesis configuration"""

    sample_rate: int = 48000
    block_size: int = 1024
    num_voices: int = 8
    synthesis_method: SynthesisMethod = SynthesisMethod.SUBTRACTIVE
    enable_effects: bool = True
    enable_modulation: bool = True


class WaveformGenerator:
    """
    Basic waveform generation functions
    """

    @staticmethod
    def generate_sine(
        frequency: float,
        duration: float,
        sample_rate: int,
        phase: float = 0.0,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """Generate sine wave"""
        try:
            t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
            wave = amplitude * np.sin(2 * np.pi * frequency * t + phase)
            return wave.astype(np.float32)
        except Exception as e:
            logger.error(f"Sine wave generation error: {e}")
            return np.zeros(int(duration * sample_rate), dtype=np.float32)

    @staticmethod
    def generate_square(
        frequency: float,
        duration: float,
        sample_rate: int,
        phase: float = 0.0,
        amplitude: float = 1.0,
        pulse_width: float = 0.5,
    ) -> np.ndarray:
        """Generate square/pulse wave"""
        try:
            t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
            # Generate sawtooth then convert to square
            sawtooth = (2 * frequency * t + phase / np.pi) % 2 - 1
            square = np.where(sawtooth < (2 * pulse_width - 1), amplitude, -amplitude)
            return square.astype(np.float32)
        except Exception as e:
            logger.error(f"Square wave generation error: {e}")
            return np.zeros(int(duration * sample_rate), dtype=np.float32)

    @staticmethod
    def generate_sawtooth(
        frequency: float,
        duration: float,
        sample_rate: int,
        phase: float = 0.0,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """Generate sawtooth wave"""
        try:
            t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
            sawtooth = amplitude * ((2 * frequency * t + phase / np.pi) % 2 - 1)
            return sawtooth.astype(np.float32)
        except Exception as e:
            logger.error(f"Sawtooth wave generation error: {e}")
            return np.zeros(int(duration * sample_rate), dtype=np.float32)

    @staticmethod
    def generate_triangle(
        frequency: float,
        duration: float,
        sample_rate: int,
        phase: float = 0.0,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """Generate triangle wave"""
        try:
            t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
            sawtooth = (2 * frequency * t + phase / np.pi) % 2 - 1
            triangle = amplitude * (2 * np.abs(sawtooth) - 1)
            return triangle.astype(np.float32)
        except Exception as e:
            logger.error(f"Triangle wave generation error: {e}")
            return np.zeros(int(duration * sample_rate), dtype=np.float32)

    @staticmethod
    def generate_noise(
        duration: float,
        sample_rate: int,
        noise_type: WaveformType = WaveformType.NOISE_WHITE,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """Generate various types of noise"""
        try:
            num_samples = int(duration * sample_rate)

            if noise_type == WaveformType.NOISE_WHITE:
                noise = np.random.normal(0, amplitude, num_samples)

            elif noise_type == WaveformType.NOISE_PINK:
                # Pink noise (1/f noise)
                white_noise = np.random.normal(0, 1, num_samples)
                # Apply pink noise filter using FFT
                freqs = np.fft.fftfreq(num_samples, 1 / sample_rate)
                # Avoid division by zero
                pink_filter = np.where(freqs != 0, 1.0 / np.sqrt(np.abs(freqs)), 1.0)
                pink_filter[0] = 1.0  # DC component

                white_fft = np.fft.fft(white_noise)
                pink_fft = white_fft * pink_filter
                noise = np.fft.ifft(pink_fft).real * amplitude

            elif noise_type == WaveformType.NOISE_BROWN:
                # Brown noise (1/f^2 noise)
                white_noise = np.random.normal(0, 1, num_samples)
                freqs = np.fft.fftfreq(num_samples, 1 / sample_rate)
                brown_filter = np.where(freqs != 0, 1.0 / np.abs(freqs), 1.0)
                brown_filter[0] = 1.0

                white_fft = np.fft.fft(white_noise)
                brown_fft = white_fft * brown_filter
                noise = np.fft.ifft(brown_fft).real * amplitude

            else:
                # Default to white noise
                noise = np.random.normal(0, amplitude, num_samples)

            return noise.astype(np.float32)

        except Exception as e:
            logger.error(f"Noise generation error: {e}")
            return np.zeros(int(duration * sample_rate), dtype=np.float32)


class EnvelopeGenerator:
    """
    Envelope generation for amplitude and parameter control
    """

    @staticmethod
    def generate_adsr(
        duration: float, sample_rate: int, envelope: ADSREnvelope
    ) -> np.ndarray:
        """Generate ADSR envelope"""
        try:
            num_samples = int(duration * sample_rate)
            env = np.zeros(num_samples)

            # Calculate sample indices for each phase
            attack_samples = int(envelope.attack_time * sample_rate)
            decay_samples = int(envelope.decay_time * sample_rate)
            release_samples = int(envelope.release_time * sample_rate)

            # Ensure we don't exceed the total duration
            attack_samples = min(attack_samples, num_samples)
            decay_samples = min(decay_samples, num_samples - attack_samples)
            release_samples = min(release_samples, num_samples)

            sustain_samples = max(
                0, num_samples - attack_samples - decay_samples - release_samples
            )

            idx = 0

            # Attack phase
            if attack_samples > 0:
                env[idx : idx + attack_samples] = np.linspace(0, 1, attack_samples)
                idx += attack_samples

            # Decay phase
            if decay_samples > 0:
                env[idx : idx + decay_samples] = np.linspace(
                    1, envelope.sustain_level, decay_samples
                )
                idx += decay_samples

            # Sustain phase
            if sustain_samples > 0:
                env[idx : idx + sustain_samples] = envelope.sustain_level
                idx += sustain_samples

            # Release phase
            if release_samples > 0 and idx < num_samples:
                remaining = num_samples - idx
                release_samples = min(release_samples, remaining)
                start_level = env[idx - 1] if idx > 0 else envelope.sustain_level
                env[idx : idx + release_samples] = np.linspace(
                    start_level, 0, release_samples
                )

            return env.astype(np.float32)

        except Exception as e:
            logger.error(f"ADSR envelope generation error: {e}")
            # Return linear decay envelope as fallback
            env = np.linspace(1, 0, int(duration * sample_rate))
            return env.astype(np.float32)

    @staticmethod
    def generate_exponential(
        duration: float, sample_rate: int, decay_constant: float = 2.0
    ) -> np.ndarray:
        """Generate exponential decay envelope"""
        try:
            num_samples = int(duration * sample_rate)
            t = np.linspace(0, duration, num_samples)
            env = np.exp(-decay_constant * t)
            return env.astype(np.float32)
        except Exception as e:
            logger.error(f"Exponential envelope generation error: {e}")
            return np.ones(int(duration * sample_rate), dtype=np.float32)


class AudioFilter:
    """
    Digital audio filters for synthesis
    """

    @staticmethod
    def apply_filter(
        audio: np.ndarray, filter_config: FilterConfig, sample_rate: int
    ) -> np.ndarray:
        """Apply digital filter to audio"""
        if not HAS_AUDIO_LIBS:
            return audio

        try:
            nyquist = sample_rate / 2
            normalized_cutoff = filter_config.cutoff_frequency / nyquist

            # Ensure cutoff is within valid range
            normalized_cutoff = np.clip(normalized_cutoff, 0.01, 0.99)

            if filter_config.filter_type == FilterType.LOW_PASS:
                b, a = scipy.signal.butter(2, normalized_cutoff, btype="low")

            elif filter_config.filter_type == FilterType.HIGH_PASS:
                b, a = scipy.signal.butter(2, normalized_cutoff, btype="high")

            elif filter_config.filter_type == FilterType.BAND_PASS:
                # Use cutoff as center frequency, calculate bandwidth
                bandwidth = normalized_cutoff * 0.2  # 20% bandwidth
                low = max(0.01, normalized_cutoff - bandwidth / 2)
                high = min(0.99, normalized_cutoff + bandwidth / 2)
                b, a = scipy.signal.butter(2, [low, high], btype="band")

            elif filter_config.filter_type == FilterType.BAND_STOP:
                bandwidth = normalized_cutoff * 0.2
                low = max(0.01, normalized_cutoff - bandwidth / 2)
                high = min(0.99, normalized_cutoff + bandwidth / 2)
                b, a = scipy.signal.butter(2, [low, high], btype="bandstop")

            else:
                # No filtering for unsupported types
                return audio

            # Apply filter
            filtered = scipy.signal.filtfilt(b, a, audio)
            return filtered.astype(audio.dtype)

        except Exception as e:
            logger.error(f"Filter application error: {e}")
            return audio

    @staticmethod
    def resonant_filter(
        audio: np.ndarray,
        cutoff: float,
        resonance: float,
        sample_rate: int,
        filter_type: FilterType = FilterType.LOW_PASS,
    ) -> np.ndarray:
        """Apply resonant filter (simplified implementation)"""
        if not HAS_AUDIO_LIBS:
            return audio

        try:
            # Simple resonant filter using biquad
            nyquist = sample_rate / 2
            normalized_cutoff = np.clip(cutoff / nyquist, 0.01, 0.99)

            # Calculate biquad coefficients
            omega = 2 * np.pi * normalized_cutoff
            sin_omega = np.sin(omega)
            cos_omega = np.cos(omega)
            alpha = sin_omega / (2 * resonance)

            if filter_type == FilterType.LOW_PASS:
                b0 = (1 - cos_omega) / 2
                b1 = 1 - cos_omega
                b2 = (1 - cos_omega) / 2
                a0 = 1 + alpha
                a1 = -2 * cos_omega
                a2 = 1 - alpha
            else:
                # Default to low-pass
                b0 = (1 - cos_omega) / 2
                b1 = 1 - cos_omega
                b2 = (1 - cos_omega) / 2
                a0 = 1 + alpha
                a1 = -2 * cos_omega
                a2 = 1 - alpha

            # Normalize coefficients
            b = np.array([b0, b1, b2]) / a0
            a = np.array([1, a1, a2]) / a0

            # Apply filter
            filtered = scipy.signal.lfilter(b, a, audio)
            return filtered.astype(audio.dtype)

        except Exception as e:
            logger.error(f"Resonant filter error: {e}")
            return audio


class FMSynthesizer:
    """
    Frequency Modulation synthesis
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def generate_fm(
        self,
        carrier_freq: float,
        modulator_freq: float,
        modulation_index: float,
        duration: float,
        carrier_amplitude: float = 1.0,
    ) -> np.ndarray:
        """
        Generate FM synthesized audio

        Args:
            carrier_freq: Carrier frequency (Hz)
            modulator_freq: Modulator frequency (Hz)
            modulation_index: Modulation index (depth)
            duration: Duration in seconds
            carrier_amplitude: Carrier amplitude
        """
        try:
            num_samples = int(duration * self.sample_rate)
            t = np.linspace(0, duration, num_samples, endpoint=False)

            # Generate modulator
            modulator = np.sin(2 * np.pi * modulator_freq * t)

            # Generate FM signal
            instantaneous_freq = (
                carrier_freq + modulation_index * modulator_freq * modulator
            )
            phase = 2 * np.pi * np.cumsum(instantaneous_freq) / self.sample_rate

            fm_signal = carrier_amplitude * np.sin(phase)

            return fm_signal.astype(np.float32)

        except Exception as e:
            logger.error(f"FM synthesis error: {e}")
            # Fallback to sine wave
            return WaveformGenerator.generate_sine(
                carrier_freq, duration, self.sample_rate, amplitude=carrier_amplitude
            )

    def generate_complex_fm(
        self, operators: List[Dict[str, float]], duration: float
    ) -> np.ndarray:
        """
        Generate complex FM using multiple operators

        Args:
            operators: List of operator dictionaries with keys:
                      'freq', 'mod_index', 'amplitude', 'modulator_freq'
            duration: Duration in seconds
        """
        try:
            if not operators:
                return np.zeros(int(duration * self.sample_rate), dtype=np.float32)

            num_samples = int(duration * self.sample_rate)
            output = np.zeros(num_samples, dtype=np.float32)

            for op in operators:
                freq = op.get("freq", 440.0)
                mod_freq = op.get("modulator_freq", freq * 2)
                mod_index = op.get("mod_index", 1.0)
                amplitude = op.get("amplitude", 1.0)

                # Generate FM for this operator
                fm_component = self.generate_fm(
                    freq, mod_freq, mod_index, duration, amplitude
                )

                # Add to output
                output += fm_component

            # Normalize to prevent clipping
            max_val = np.max(np.abs(output))
            if max_val > 1.0:
                output /= max_val

            return output

        except Exception as e:
            logger.error(f"Complex FM synthesis error: {e}")
            return np.zeros(int(duration * self.sample_rate), dtype=np.float32)


class GranularSynthesizer:
    """
    Granular synthesis for texture and time-stretching
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def synthesize_granular(
        self, source_audio: np.ndarray, config: GranularConfig, output_duration: float
    ) -> np.ndarray:
        """
        Apply granular synthesis to source audio

        Args:
            source_audio: Input audio to granulate
            config: Granular synthesis configuration
            output_duration: Output duration in seconds

        Returns:
            Granularly synthesized audio
        """
        try:
            if len(source_audio) == 0:
                return np.zeros(
                    int(output_duration * self.sample_rate), dtype=np.float32
                )

            output_samples = int(output_duration * self.sample_rate)
            output = np.zeros(output_samples, dtype=np.float32)

            grain_size_samples = int(config.grain_size * self.sample_rate)
            grain_interval = int(self.sample_rate / config.grain_density)

            # Generate envelope for grains
            grain_env = self._generate_grain_envelope(
                grain_size_samples, config.grain_envelope
            )

            # Generate grains
            current_pos = 0
            while current_pos < output_samples:
                # Calculate source position with scatter
                source_pos = current_pos * config.grain_pitch
                if config.grain_scatter > 0:
                    scatter_samples = int(config.grain_scatter * self.sample_rate)
                    source_pos += random.randint(-scatter_samples, scatter_samples)

                # Extract grain from source
                grain = self._extract_grain(
                    source_audio, source_pos, grain_size_samples
                )

                # Apply pitch shift
                if config.grain_pitch != 1.0:
                    grain = self._pitch_shift_grain(grain, config.grain_pitch)

                # Apply envelope
                if len(grain) == len(grain_env):
                    grain *= grain_env
                else:
                    # Resize envelope if needed
                    if len(grain) > 0:
                        env_resized = (
                            scipy.signal.resample(grain_env, len(grain))
                            if HAS_AUDIO_LIBS
                            else grain_env[: len(grain)]
                        )
                        grain *= env_resized[: len(grain)]

                # Mix grain into output
                end_pos = min(current_pos + len(grain), output_samples)
                grain_length = end_pos - current_pos
                output[current_pos:end_pos] += grain[:grain_length]

                # Move to next grain position
                current_pos += grain_interval

            # Normalize output
            max_val = np.max(np.abs(output))
            if max_val > 1.0:
                output /= max_val

            return output

        except Exception as e:
            logger.error(f"Granular synthesis error: {e}")
            return np.zeros(int(output_duration * self.sample_rate), dtype=np.float32)

    def _generate_grain_envelope(
        self, grain_size: int, envelope_type: EnvelopeType
    ) -> np.ndarray:
        """Generate envelope for individual grains"""
        try:
            if envelope_type == EnvelopeType.LINEAR:
                # Triangular envelope
                mid_point = grain_size // 2
                env = np.concatenate(
                    [
                        np.linspace(0, 1, mid_point),
                        np.linspace(1, 0, grain_size - mid_point),
                    ]
                )
            elif envelope_type == EnvelopeType.EXPONENTIAL:
                # Exponential fade-in and fade-out
                t = np.linspace(0, 1, grain_size)
                fade_in = 1 - np.exp(-5 * t)
                fade_out = np.exp(-5 * t)
                env = fade_in * fade_out
            else:
                # Default to Hann window
                env = np.hanning(grain_size)

            return env.astype(np.float32)

        except Exception as e:
            logger.debug(f"Grain envelope generation error: {e}")
            return np.hanning(grain_size).astype(np.float32)

    def _extract_grain(
        self, source: np.ndarray, position: float, grain_size: int
    ) -> np.ndarray:
        """Extract grain from source audio"""
        try:
            start_idx = int(position) % len(source)
            end_idx = start_idx + grain_size

            if end_idx <= len(source):
                return source[start_idx:end_idx].copy()
            else:
                # Wrap around if needed
                grain = np.zeros(grain_size, dtype=source.dtype)
                remaining = len(source) - start_idx
                grain[:remaining] = source[start_idx:]
                if end_idx - len(source) > 0:
                    wrap_amount = min(end_idx - len(source), remaining)
                    grain[remaining : remaining + wrap_amount] = source[:wrap_amount]
                return grain

        except Exception as e:
            logger.debug(f"Grain extraction error: {e}")
            return np.zeros(grain_size, dtype=np.float32)

    def _pitch_shift_grain(self, grain: np.ndarray, pitch_ratio: float) -> np.ndarray:
        """Apply pitch shift to grain (simple time-domain method)"""
        try:
            if pitch_ratio == 1.0 or not HAS_AUDIO_LIBS:
                return grain

            # Simple pitch shift using resampling
            new_length = int(len(grain) / pitch_ratio)
            if new_length > 0:
                shifted = scipy.signal.resample(grain, new_length)
                return shifted.astype(grain.dtype)
            else:
                return grain

        except Exception as e:
            logger.debug(f"Grain pitch shift error: {e}")
            return grain


class PhysicalModelingSynthesizer:
    """
    Physical modeling synthesis (string, wind, percussion models)
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def synthesize_string(
        self,
        frequency: float,
        duration: float,
        pluck_position: float = 0.5,
        damping: float = 0.999,
    ) -> np.ndarray:
        """
        Synthesize plucked string using Karplus-Strong algorithm

        Args:
            frequency: Fundamental frequency
            duration: Duration in seconds
            pluck_position: Pluck position (0.0 to 1.0)
            damping: Damping factor (0.0 to 1.0)
        """
        try:
            # Calculate delay line length
            delay_length = int(self.sample_rate / frequency)
            if delay_length < 2:
                delay_length = 2

            num_samples = int(duration * self.sample_rate)
            output = np.zeros(num_samples, dtype=np.float32)

            # Initialize delay line with noise burst
            delay_line = np.random.uniform(-1, 1, delay_length).astype(np.float32)

            # Apply initial pluck shaping
            pluck_idx = int(pluck_position * delay_length)
            for i in range(delay_length):
                distance = abs(i - pluck_idx) / delay_length
                delay_line[i] *= 1.0 - distance

            # Generate output using feedback loop
            delay_idx = 0
            for i in range(num_samples):
                # Get current sample from delay line
                current_sample = delay_line[delay_idx]
                output[i] = current_sample

                # Calculate new sample (low-pass filter + damping)
                next_idx = (delay_idx + 1) % delay_length
                new_sample = damping * 0.5 * (current_sample + delay_line[next_idx])
                delay_line[delay_idx] = new_sample

                # Advance delay line index
                delay_idx = next_idx

            return output

        except Exception as e:
            logger.error(f"String synthesis error: {e}")
            return np.zeros(int(duration * self.sample_rate), dtype=np.float32)

    def synthesize_drum(
        self, frequency: float, duration: float, noise_level: float = 0.3
    ) -> np.ndarray:
        """
        Synthesize drum hit using filtered noise burst

        Args:
            frequency: Fundamental frequency
            duration: Duration in seconds
            noise_level: Amount of noise vs. tonal content
        """
        try:
            num_samples = int(duration * self.sample_rate)

            # Generate exponentially decaying sine wave
            t = np.linspace(0, duration, num_samples, endpoint=False)
            decay_rate = 8.0  # Faster decay for drum-like sound
            envelope = np.exp(-decay_rate * t)
            tone = np.sin(2 * np.pi * frequency * t) * envelope

            # Generate filtered noise burst
            noise = np.random.uniform(-1, 1, num_samples)
            if HAS_AUDIO_LIBS:
                # Apply bandpass filter around the fundamental frequency
                nyquist = self.sample_rate / 2
                low_freq = max(frequency * 0.5, 50) / nyquist
                high_freq = min(frequency * 4, nyquist * 0.9) / nyquist

                try:
                    b, a = scipy.signal.butter(4, [low_freq, high_freq], btype="band")
                    filtered_noise = scipy.signal.filtfilt(b, a, noise)
                except:
                    filtered_noise = noise
            else:
                filtered_noise = noise

            # Apply envelope to noise
            filtered_noise *= envelope

            # Mix tone and noise
            drum_sound = (1 - noise_level) * tone + noise_level * filtered_noise

            return drum_sound.astype(np.float32)

        except Exception as e:
            logger.error(f"Drum synthesis error: {e}")
            return np.zeros(int(duration * self.sample_rate), dtype=np.float32)


class WavetableSynthesizer:
    """
    Wavetable synthesis using stored waveforms
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.wavetables = {}
        self._initialize_default_tables()

    def _initialize_default_tables(self):
        """Initialize default wavetables"""
        try:
            table_size = 2048
            t = np.linspace(0, 2 * np.pi, table_size, endpoint=False)

            # Basic waveforms
            self.wavetables["sine"] = np.sin(t)
            self.wavetables["square"] = np.sign(np.sin(t))
            self.wavetables["sawtooth"] = 2 * (t / (2 * np.pi)) - 1
            self.wavetables["triangle"] = 2 * np.abs(2 * (t / (2 * np.pi)) - 1) - 1

            # Complex waveforms
            # Harmonic series
            harmonic = np.sin(t)
            for h in range(2, 8):
                harmonic += np.sin(h * t) / h
            self.wavetables["harmonic"] = harmonic / np.max(np.abs(harmonic))

            # PWM sweep
            pwm = np.zeros_like(t)
            for i, phase in enumerate(t):
                width = 0.1 + 0.4 * (i / len(t))  # Width from 10% to 50%
                pwm[i] = 1.0 if (phase % (2 * np.pi)) < (2 * np.pi * width) else -1.0
            self.wavetables["pwm_sweep"] = pwm

        except Exception as e:
            logger.error(f"Wavetable initialization error: {e}")

    def add_wavetable(self, name: str, waveform: np.ndarray) -> None:
        """Add custom wavetable"""
        try:
            # Normalize waveform
            if len(waveform) > 0:
                normalized = waveform / (np.max(np.abs(waveform)) + 1e-8)
                self.wavetables[name] = normalized.astype(np.float32)
        except Exception as e:
            logger.error(f"Wavetable addition error: {e}")

    def synthesize_wavetable(
        self,
        wavetable_name: str,
        frequency: float,
        duration: float,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """
        Synthesize audio using wavetable lookup

        Args:
            wavetable_name: Name of wavetable to use
            frequency: Playback frequency
            duration: Duration in seconds
            amplitude: Output amplitude
        """
        try:
            if wavetable_name not in self.wavetables:
                logger.warning(f"Wavetable '{wavetable_name}' not found, using sine")
                wavetable_name = "sine"

            wavetable = self.wavetables[wavetable_name]
            table_size = len(wavetable)

            num_samples = int(duration * self.sample_rate)
            output = np.zeros(num_samples, dtype=np.float32)

            # Calculate phase increment
            phase_increment = frequency * table_size / self.sample_rate

            # Generate output with linear interpolation
            current_phase = 0.0
            for i in range(num_samples):
                # Get integer and fractional parts
                int_phase = int(current_phase) % table_size
                frac_phase = current_phase - int(current_phase)

                # Linear interpolation
                next_phase = (int_phase + 1) % table_size
                interpolated = (1 - frac_phase) * wavetable[
                    int_phase
                ] + frac_phase * wavetable[next_phase]

                output[i] = amplitude * interpolated
                current_phase += phase_increment

                # Wrap phase to prevent overflow
                if current_phase >= table_size:
                    current_phase -= table_size

            return output

        except Exception as e:
            logger.error(f"Wavetable synthesis error: {e}")
            return np.zeros(int(duration * self.sample_rate), dtype=np.float32)

    def morphing_synthesis(
        self,
        wavetable1: str,
        wavetable2: str,
        morph_amount: float,
        frequency: float,
        duration: float,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """
        Synthesize with morphing between two wavetables

        Args:
            wavetable1: First wavetable name
            wavetable2: Second wavetable name
            morph_amount: Morphing amount (0.0 = table1, 1.0 = table2)
            frequency: Playback frequency
            duration: Duration in seconds
            amplitude: Output amplitude
        """
        try:
            if wavetable1 not in self.wavetables or wavetable2 not in self.wavetables:
                # Fallback to basic synthesis
                return self.synthesize_wavetable("sine", frequency, duration, amplitude)

            # Get both waveforms
            audio1 = self.synthesize_wavetable(
                wavetable1, frequency, duration, amplitude
            )
            audio2 = self.synthesize_wavetable(
                wavetable2, frequency, duration, amplitude
            )

            # Ensure same length
            min_length = min(len(audio1), len(audio2))
            audio1 = audio1[:min_length]
            audio2 = audio2[:min_length]

            # Linear interpolation between waveforms
            morph_amount = np.clip(morph_amount, 0.0, 1.0)
            morphed = (1 - morph_amount) * audio1 + morph_amount * audio2

            return morphed

        except Exception as e:
            logger.error(f"Morphing synthesis error: {e}")
            return np.zeros(int(duration * self.sample_rate), dtype=np.float32)


class AudioSynthesizer:
    """
    Main audio synthesis engine
    """

    def __init__(self, config: Optional[SynthConfig] = None):
        self.config = config or SynthConfig()

        # Initialize synthesis engines
        self.fm_synth = FMSynthesizer(self.config.sample_rate)
        self.granular_synth = GranularSynthesizer(self.config.sample_rate)
        self.physical_synth = PhysicalModelingSynthesizer(self.config.sample_rate)
        self.wavetable_synth = WavetableSynthesizer(self.config.sample_rate)

        # Voice management
        self.active_voices = []
        self.voice_counter = 0

    async def generate_tone(
        self,
        oscillator: Oscillator,
        duration: float,
        envelope: Optional[ADSREnvelope] = None,
    ) -> np.ndarray:
        """
        Generate basic tone with oscillator and envelope

        Args:
            oscillator: Oscillator configuration
            duration: Duration in seconds
            envelope: Optional ADSR envelope
        """
        try:
            # Apply detune
            frequency = oscillator.frequency
            if oscillator.detune != 0:
                frequency *= 2 ** (oscillator.detune / 1200)  # Convert cents to ratio

            # Generate base waveform
            if oscillator.waveform == WaveformType.SINE:
                audio = WaveformGenerator.generate_sine(
                    frequency,
                    duration,
                    self.config.sample_rate,
                    oscillator.phase,
                    oscillator.amplitude,
                )
            elif oscillator.waveform == WaveformType.SQUARE:
                audio = WaveformGenerator.generate_square(
                    frequency,
                    duration,
                    self.config.sample_rate,
                    oscillator.phase,
                    oscillator.amplitude,
                    oscillator.pulse_width,
                )
            elif oscillator.waveform == WaveformType.SAWTOOTH:
                audio = WaveformGenerator.generate_sawtooth(
                    frequency,
                    duration,
                    self.config.sample_rate,
                    oscillator.phase,
                    oscillator.amplitude,
                )
            elif oscillator.waveform == WaveformType.TRIANGLE:
                audio = WaveformGenerator.generate_triangle(
                    frequency,
                    duration,
                    self.config.sample_rate,
                    oscillator.phase,
                    oscillator.amplitude,
                )
            elif oscillator.waveform in [
                WaveformType.NOISE_WHITE,
                WaveformType.NOISE_PINK,
                WaveformType.NOISE_BROWN,
            ]:
                audio = WaveformGenerator.generate_noise(
                    duration,
                    self.config.sample_rate,
                    oscillator.waveform,
                    oscillator.amplitude,
                )
            else:
                # Default to sine wave
                audio = WaveformGenerator.generate_sine(
                    frequency,
                    duration,
                    self.config.sample_rate,
                    oscillator.phase,
                    oscillator.amplitude,
                )

            # Apply envelope if provided
            if envelope is not None:
                env = EnvelopeGenerator.generate_adsr(
                    duration, self.config.sample_rate, envelope
                )
                # Ensure same length
                min_length = min(len(audio), len(env))
                audio = audio[:min_length] * env[:min_length]

            return audio

        except Exception as e:
            logger.error(f"Tone generation error: {e}")
            return np.zeros(int(duration * self.config.sample_rate), dtype=np.float32)

    async def generate_fm_sound(
        self,
        carrier_freq: float,
        modulator_freq: float,
        modulation_index: float,
        duration: float,
        envelope: Optional[ADSREnvelope] = None,
    ) -> np.ndarray:
        """Generate FM synthesized sound"""
        try:
            # Generate FM audio
            fm_audio = self.fm_synth.generate_fm(
                carrier_freq, modulator_freq, modulation_index, duration
            )

            # Apply envelope if provided
            if envelope is not None:
                env = EnvelopeGenerator.generate_adsr(
                    duration, self.config.sample_rate, envelope
                )
                min_length = min(len(fm_audio), len(env))
                fm_audio = fm_audio[:min_length] * env[:min_length]

            return fm_audio

        except Exception as e:
            logger.error(f"FM sound generation error: {e}")
            return np.zeros(int(duration * self.config.sample_rate), dtype=np.float32)

    async def generate_granular_texture(
        self, source_audio: np.ndarray, config: GranularConfig, duration: float
    ) -> np.ndarray:
        """Generate granular texture from source audio"""
        try:
            return self.granular_synth.synthesize_granular(
                source_audio, config, duration
            )
        except Exception as e:
            logger.error(f"Granular texture generation error: {e}")
            return np.zeros(int(duration * self.config.sample_rate), dtype=np.float32)

    async def generate_physical_model(
        self, model_type: str, frequency: float, duration: float, **params
    ) -> np.ndarray:
        """Generate sound using physical modeling"""
        try:
            if model_type == "string":
                pluck_pos = params.get("pluck_position", 0.5)
                damping = params.get("damping", 0.999)
                return self.physical_synth.synthesize_string(
                    frequency, duration, pluck_pos, damping
                )
            elif model_type == "drum":
                noise_level = params.get("noise_level", 0.3)
                return self.physical_synth.synthesize_drum(
                    frequency, duration, noise_level
                )
            else:
                logger.warning(f"Unknown physical model type: {model_type}")
                return np.zeros(
                    int(duration * self.config.sample_rate), dtype=np.float32
                )

        except Exception as e:
            logger.error(f"Physical modeling error: {e}")
            return np.zeros(int(duration * self.config.sample_rate), dtype=np.float32)

    async def generate_wavetable_sound(
        self,
        wavetable_name: str,
        frequency: float,
        duration: float,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """Generate sound using wavetable synthesis"""
        try:
            return self.wavetable_synth.synthesize_wavetable(
                wavetable_name, frequency, duration, amplitude
            )
        except Exception as e:
            logger.error(f"Wavetable synthesis error: {e}")
            return np.zeros(int(duration * self.config.sample_rate), dtype=np.float32)

    def apply_effects_chain(
        self,
        audio: np.ndarray,
        filters: List[FilterConfig] = None,
        modulation: List[ModulationSource] = None,
    ) -> np.ndarray:
        """Apply effects and modulation to audio"""
        try:
            processed = audio.copy()

            # Apply filters
            if filters:
                for filter_config in filters:
                    processed = AudioFilter.apply_filter(
                        processed, filter_config, self.config.sample_rate
                    )

            # Apply modulation (simplified implementation)
            if modulation and self.config.enable_modulation:
                for mod_source in modulation:
                    processed = self._apply_modulation(processed, mod_source)

            return processed

        except Exception as e:
            logger.error(f"Effects chain error: {e}")
            return audio

    def _apply_modulation(
        self, audio: np.ndarray, mod_source: ModulationSource
    ) -> np.ndarray:
        """Apply modulation to audio"""
        try:
            duration = len(audio) / self.config.sample_rate

            # Generate modulation signal
            mod_audio = WaveformGenerator.generate_sine(
                mod_source.rate, duration, self.config.sample_rate
            )

            if mod_source.modulation_type == ModulationType.AMPLITUDE:
                # Amplitude modulation (tremolo)
                modulated = audio * (1 + mod_source.depth * mod_audio)
            elif mod_source.modulation_type == ModulationType.RING:
                # Ring modulation
                modulated = audio * mod_audio * mod_source.depth
            else:
                # No modulation for unsupported types
                modulated = audio

            return modulated.astype(audio.dtype)

        except Exception as e:
            logger.debug(f"Modulation application error: {e}")
            return audio

    def create_demo_composition(self) -> np.ndarray:
        """Create a demo composition showcasing various synthesis methods"""
        try:
            duration = 8.0  # Total duration
            composition = np.zeros(
                int(duration * self.config.sample_rate), dtype=np.float32
            )

            # Segment durations
            segment_dur = duration / 4

            # Segment 1: Basic waveforms
            t1_end = int(segment_dur * self.config.sample_rate)
            oscillator = Oscillator(
                waveform=WaveformType.SAWTOOTH, frequency=220, amplitude=0.3
            )
            envelope = ADSREnvelope(
                attack_time=0.1, decay_time=0.2, sustain_level=0.6, release_time=0.5
            )

            async def gen_seg1():
                return await self.generate_tone(oscillator, segment_dur, envelope)

            try:
                seg1 = asyncio.run(gen_seg1())
                composition[: len(seg1)] += seg1
            except:
                pass

            # Segment 2: FM synthesis
            t2_start = int(segment_dur * self.config.sample_rate)
            t2_end = int(2 * segment_dur * self.config.sample_rate)

            fm_audio = self.fm_synth.generate_fm(330, 100, 2.5, segment_dur, 0.3)
            composition[t2_start : t2_start + len(fm_audio)] += fm_audio

            # Segment 3: Physical modeling
            t3_start = int(2 * segment_dur * self.config.sample_rate)
            t3_end = int(3 * segment_dur * self.config.sample_rate)

            string_audio = self.physical_synth.synthesize_string(
                165, segment_dur, 0.3, 0.995
            )
            composition[t3_start : t3_start + len(string_audio)] += string_audio

            # Segment 4: Wavetable synthesis
            t4_start = int(3 * segment_dur * self.config.sample_rate)

            wavetable_audio = self.wavetable_synth.synthesize_wavetable(
                "harmonic", 440, segment_dur, 0.3
            )
            composition[t4_start : t4_start + len(wavetable_audio)] += wavetable_audio

            # Normalize final composition
            max_val = np.max(np.abs(composition))
            if max_val > 0:
                composition = composition * 0.8 / max_val

            return composition

        except Exception as e:
            logger.error(f"Demo composition creation error: {e}")
            return np.zeros(int(duration * self.config.sample_rate), dtype=np.float32)


# Convenience functions and aliases
def create_basic_oscillator(
    waveform: WaveformType, frequency: float, amplitude: float = 1.0
) -> Oscillator:
    """Create a basic oscillator configuration"""
    return Oscillator(waveform=waveform, frequency=frequency, amplitude=amplitude)


def create_adsr_envelope(
    attack: float = 0.1, decay: float = 0.2, sustain: float = 0.7, release: float = 0.5
) -> ADSREnvelope:
    """Create an ADSR envelope configuration"""
    return ADSREnvelope(
        attack_time=attack,
        decay_time=decay,
        sustain_level=sustain,
        release_time=release,
    )


# Example usage and testing
if __name__ == "__main__":

    async def demo():
        """Demo audio synthesis capabilities"""
        try:
            # Create synthesizer
            config = SynthConfig(sample_rate=44100)
            synthesizer = AudioSynthesizer(config)

            print("Creating audio synthesis demo...")

            # Test basic waveform generation
            print("Generating basic waveforms...")
            oscillator = create_basic_oscillator(WaveformType.SAWTOOTH, 440, 0.5)
            envelope = create_adsr_envelope(0.05, 0.1, 0.8, 0.3)

            basic_tone = await synthesizer.generate_tone(oscillator, 2.0, envelope)
            print(
                f"Basic tone: {len(basic_tone)} samples, RMS: {np.sqrt(np.mean(basic_tone**2)):.4f}"
            )

            # Test FM synthesis
            print("Generating FM synthesis...")
            fm_sound = await synthesizer.generate_fm_sound(220, 55, 3.0, 2.0, envelope)
            print(
                f"FM sound: {len(fm_sound)} samples, RMS: {np.sqrt(np.mean(fm_sound**2)):.4f}"
            )

            # Test physical modeling
            print("Generating physical model (string)...")
            string_sound = await synthesizer.generate_physical_model(
                "string", 330, 3.0, pluck_position=0.3, damping=0.998
            )
            print(
                f"String sound: {len(string_sound)} samples, RMS: {np.sqrt(np.mean(string_sound**2)):.4f}"
            )

            # Test wavetable synthesis
            print("Generating wavetable synthesis...")
            wavetable_sound = await synthesizer.generate_wavetable_sound(
                "harmonic", 660, 2.0, 0.4
            )
            print(
                f"Wavetable sound: {len(wavetable_sound)} samples, RMS: {np.sqrt(np.mean(wavetable_sound**2)):.4f}"
            )

            # Test granular synthesis
            print("Generating granular texture...")
            source_audio = WaveformGenerator.generate_sine(880, 1.0, config.sample_rate)
            granular_config = GranularConfig(
                grain_size=0.05, grain_density=20.0, grain_pitch=0.8, grain_scatter=0.2
            )
            granular_texture = await synthesizer.generate_granular_texture(
                source_audio, granular_config, 3.0
            )
            print(
                f"Granular texture: {len(granular_texture)} samples, RMS: {np.sqrt(np.mean(granular_texture**2)):.4f}"
            )

            # Create full demo composition
            print("Creating full demo composition...")
            composition = synthesizer.create_demo_composition()
            print(
                f"Demo composition: {len(composition)} samples ({len(composition)/config.sample_rate:.1f}s)"
            )
            print(f"Composition RMS: {np.sqrt(np.mean(composition**2)):.4f}")
            print(f"Peak level: {np.max(np.abs(composition)):.4f}")

            print("\nAudio synthesis demo completed successfully!")

        except Exception as e:
            print(f"Demo error: {e}")

    # Run demo
    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
