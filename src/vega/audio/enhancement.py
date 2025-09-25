"""
Vega 2.0 Audio Enhancement Suite

This module provides comprehensive audio enhancement capabilities including:
- Advanced noise cancellation using multiple algorithms
- Echo and reverberation removal
- Audio restoration for damaged recordings
- Quality enhancement and upsampling
- Dynamic range compression/expansion
- Equalization and spectral shaping
- Multi-channel audio processing

Dependencies:
- numpy: Audio array processing
- scipy: Signal processing and filters
- librosa: Audio analysis and processing
- soundfile: Audio I/O operations
"""

import asyncio
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import librosa
    import librosa.display
    import scipy.signal
    import scipy.ndimage
    import scipy.fftpack
    from scipy.signal import butter, sosfilt, hilbert

    HAS_AUDIO_LIBS = True
except ImportError:
    librosa = None
    scipy = None
    butter = None
    sosfilt = None
    hilbert = None
    HAS_AUDIO_LIBS = False

try:
    import soundfile as sf

    HAS_SOUNDFILE = True
except ImportError:
    sf = None
    HAS_SOUNDFILE = False

logger = logging.getLogger(__name__)


class AudioEnhancementError(Exception):
    """Custom exception for audio enhancement errors"""

    pass


class NoiseReductionAlgorithm(Enum):
    """Available noise reduction algorithms"""

    SPECTRAL_SUBTRACTION = "spectral_subtraction"
    WIENER_FILTER = "wiener_filter"
    RNNoise = "rnnoise"
    KALMAN_FILTER = "kalman_filter"
    ADAPTIVE_FILTER = "adaptive_filter"
    STATIONARY_WAVELET = "stationary_wavelet"


class EchoRemovalMethod(Enum):
    """Echo removal algorithms"""

    ADAPTIVE_ECHO_CANCELLATION = "aec"
    SPECTRAL_SUBTRACTION = "spectral"
    CEPSTRAL_FILTERING = "cepstral"
    BLIND_SOURCE_SEPARATION = "bss"


class CompressionType(Enum):
    """Dynamic range compression types"""

    SOFT_KNEE = "soft_knee"
    HARD_KNEE = "hard_knee"
    MULTIBAND = "multiband"
    UPWARD = "upward"
    DOWNWARD = "downward"


@dataclass
class EnhancementConfig:
    """Configuration for audio enhancement pipeline"""

    sample_rate: int = 44100

    # Noise reduction settings
    noise_reduction: bool = True
    noise_algorithm: NoiseReductionAlgorithm = (
        NoiseReductionAlgorithm.SPECTRAL_SUBTRACTION
    )
    noise_reduction_strength: float = 0.8  # 0.0 to 1.0

    # Echo removal settings
    echo_removal: bool = True
    echo_method: EchoRemovalMethod = EchoRemovalMethod.ADAPTIVE_ECHO_CANCELLATION
    echo_suppression_db: float = 20.0  # dB of suppression

    # Restoration settings
    restoration: bool = True
    declipping: bool = True
    denoising: bool = True
    bandwidth_extension: bool = False

    # Dynamic range settings
    compression: bool = False
    compression_type: CompressionType = CompressionType.SOFT_KNEE
    threshold_db: float = -12.0
    ratio: float = 4.0
    attack_ms: float = 5.0
    release_ms: float = 100.0

    # Equalization
    equalization: bool = False
    eq_bands: List[Tuple[float, float]] = field(
        default_factory=list
    )  # (frequency, gain_db)

    # Quality enhancement
    upsampling: bool = False
    target_sample_rate: int = 48000
    stereo_enhancement: bool = False
    harmonic_enhancement: bool = False


@dataclass
class ProcessingResult:
    """Result of audio enhancement processing"""

    enhanced_audio: np.ndarray
    original_audio: np.ndarray
    sample_rate: int
    processing_stats: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    processing_time: float = 0.0


class SpectralNoiseReducer:
    """
    Advanced spectral noise reduction using multiple algorithms
    """

    def __init__(self, config: EnhancementConfig):
        self.config = config
        self.noise_profile = None
        self.adaptation_rate = 0.05

    def estimate_noise_profile(
        self, audio_data: np.ndarray, speech_mask: Optional[np.ndarray] = None
    ) -> None:
        """
        Estimate noise profile from audio data

        Args:
            audio_data: Input audio signal
            speech_mask: Binary mask indicating speech regions (True) vs noise (False)
        """
        if not HAS_AUDIO_LIBS:
            return

        try:
            # Use short-time Fourier transform
            stft = librosa.stft(audio_data, hop_length=512, n_fft=2048)
            magnitude = np.abs(stft)

            if speech_mask is not None and len(speech_mask) == magnitude.shape[1]:
                # Use only noise regions
                noise_frames = magnitude[:, ~speech_mask]
                if noise_frames.shape[1] > 0:
                    noise_psd = np.mean(noise_frames**2, axis=1)
                else:
                    noise_psd = np.mean(magnitude**2, axis=1) * 0.1  # Fallback
            else:
                # Use minimum statistics to estimate noise
                noise_psd = np.percentile(magnitude**2, 10, axis=1)  # 10th percentile

            if self.noise_profile is None:
                self.noise_profile = noise_psd
            else:
                # Adaptive update
                self.noise_profile = (
                    1 - self.adaptation_rate
                ) * self.noise_profile + self.adaptation_rate * noise_psd

        except Exception as e:
            logger.error(f"Noise profile estimation error: {e}")

    def spectral_subtraction(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Enhanced spectral subtraction with over-subtraction and spectral floor
        """
        if not HAS_AUDIO_LIBS or self.noise_profile is None:
            return audio_data

        try:
            # Compute STFT
            stft = librosa.stft(audio_data, hop_length=512, n_fft=2048)
            magnitude = np.abs(stft)
            phase = np.angle(stft)

            # Expand noise profile to match STFT dimensions
            noise_magnitude = np.sqrt(self.noise_profile)
            noise_magnitude = np.expand_dims(noise_magnitude, axis=1)
            noise_magnitude = np.broadcast_to(noise_magnitude, magnitude.shape)

            # Spectral subtraction with adaptive over-subtraction
            strength = self.config.noise_reduction_strength
            alpha = 1.5 + strength  # Over-subtraction factor
            beta = 0.1 + (0.1 * strength)  # Spectral floor factor

            # Enhanced magnitude
            enhanced_magnitude = magnitude - alpha * noise_magnitude

            # Apply spectral floor
            enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)

            # Spectral smoothing to reduce musical noise
            enhanced_magnitude = scipy.ndimage.median_filter(
                enhanced_magnitude, size=(3, 3)
            )

            # Reconstruct signal
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(
                enhanced_stft, hop_length=512, length=len(audio_data)
            )

            return enhanced_audio.astype(audio_data.dtype)

        except Exception as e:
            logger.error(f"Spectral subtraction error: {e}")
            return audio_data

    def wiener_filter(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Wiener filtering with adaptive gain
        """
        if not HAS_AUDIO_LIBS or self.noise_profile is None:
            return audio_data

        try:
            # Compute STFT
            stft = librosa.stft(audio_data, hop_length=512, n_fft=2048)
            power_spectrum = np.abs(stft) ** 2

            # Expand noise profile
            noise_power = np.expand_dims(self.noise_profile, axis=1)
            noise_power = np.broadcast_to(noise_power, power_spectrum.shape)

            # Wiener gain calculation
            signal_power = np.maximum(
                power_spectrum - noise_power, 0.1 * power_spectrum
            )
            wiener_gain = signal_power / (signal_power + noise_power)

            # Apply strength parameter
            strength = self.config.noise_reduction_strength
            wiener_gain = 1 - strength * (1 - wiener_gain)

            # Apply gain
            enhanced_stft = stft * wiener_gain
            enhanced_audio = librosa.istft(
                enhanced_stft, hop_length=512, length=len(audio_data)
            )

            return enhanced_audio.astype(audio_data.dtype)

        except Exception as e:
            logger.error(f"Wiener filter error: {e}")
            return audio_data

    def adaptive_filter(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Adaptive noise reduction using LMS algorithm
        """
        if not HAS_AUDIO_LIBS:
            return audio_data

        try:
            # Simple LMS adaptive filter
            filter_length = 32
            step_size = 0.01

            # Initialize filter weights
            weights = np.zeros(filter_length)
            enhanced_audio = np.zeros_like(audio_data)

            for n in range(filter_length, len(audio_data)):
                # Input vector
                x = audio_data[n - filter_length : n]

                # Filter output
                y = np.dot(weights, x)

                # Error signal (assume some reference)
                error = audio_data[n] - y

                # Update weights
                weights += step_size * error * x

                # Enhanced output
                enhanced_audio[n] = y

            # Copy early samples
            enhanced_audio[:filter_length] = audio_data[:filter_length]

            return enhanced_audio

        except Exception as e:
            logger.error(f"Adaptive filter error: {e}")
            return audio_data

    def reduce_noise(
        self, audio_data: np.ndarray, speech_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply selected noise reduction algorithm
        """
        if not self.config.noise_reduction:
            return audio_data

        # Update noise profile
        self.estimate_noise_profile(audio_data, speech_mask)

        algorithm = self.config.noise_algorithm

        if algorithm == NoiseReductionAlgorithm.SPECTRAL_SUBTRACTION:
            return self.spectral_subtraction(audio_data)
        elif algorithm == NoiseReductionAlgorithm.WIENER_FILTER:
            return self.wiener_filter(audio_data)
        elif algorithm == NoiseReductionAlgorithm.ADAPTIVE_FILTER:
            return self.adaptive_filter(audio_data)
        else:
            # Fallback to spectral subtraction
            return self.spectral_subtraction(audio_data)


class EchoRemover:
    """
    Advanced echo and reverberation removal
    """

    def __init__(self, config: EnhancementConfig):
        self.config = config
        self.echo_filter = None

    def adaptive_echo_cancellation(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Adaptive Echo Cancellation using LMS algorithm
        """
        if not HAS_AUDIO_LIBS:
            return audio_data

        try:
            # Parameters
            filter_length = 1024  # Echo filter length
            step_size = 0.001

            # Initialize
            if self.echo_filter is None:
                self.echo_filter = np.zeros(filter_length)

            enhanced_audio = np.zeros_like(audio_data)

            for n in range(filter_length, len(audio_data)):
                # Reference signal (delayed input)
                reference = audio_data[n - filter_length : n]

                # Estimate echo
                echo_estimate = np.dot(self.echo_filter, reference)

                # Error signal (echo-cancelled)
                enhanced_audio[n] = audio_data[n] - echo_estimate

                # Update filter weights
                self.echo_filter += step_size * enhanced_audio[n] * reference

            # Copy early samples
            enhanced_audio[:filter_length] = audio_data[:filter_length]

            return enhanced_audio

        except Exception as e:
            logger.error(f"Adaptive echo cancellation error: {e}")
            return audio_data

    def spectral_echo_removal(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Spectral domain echo removal
        """
        if not HAS_AUDIO_LIBS:
            return audio_data

        try:
            # Compute STFT
            stft = librosa.stft(audio_data, hop_length=512, n_fft=2048)
            magnitude = np.abs(stft)
            phase = np.angle(stft)

            # Detect echo patterns in magnitude spectrogram
            # Simple approach: suppress periodic patterns
            for freq_bin in range(magnitude.shape[0]):
                freq_data = magnitude[freq_bin, :]

                # Find periodic components using autocorrelation
                autocorr = np.correlate(freq_data, freq_data, mode="full")
                autocorr = autocorr[len(autocorr) // 2 :]

                # Find dominant period (skip lag 0)
                if len(autocorr) > 10:
                    dominant_lag = np.argmax(autocorr[10:]) + 10

                    # Suppress echo if significant periodicity found
                    if autocorr[dominant_lag] > 0.7 * autocorr[0]:
                        # Apply suppression
                        suppression_factor = 0.3
                        magnitude[freq_bin, :] *= suppression_factor

            # Reconstruct
            enhanced_stft = magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(
                enhanced_stft, hop_length=512, length=len(audio_data)
            )

            return enhanced_audio.astype(audio_data.dtype)

        except Exception as e:
            logger.error(f"Spectral echo removal error: {e}")
            return audio_data

    def cepstral_echo_removal(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Cepstral domain echo removal
        """
        if not HAS_AUDIO_LIBS:
            return audio_data

        try:
            # Frame-based processing
            frame_size = 2048
            hop_size = 512
            enhanced_frames = []

            for i in range(0, len(audio_data) - frame_size, hop_size):
                frame = audio_data[i : i + frame_size]

                # Compute cepstrum
                spectrum = np.fft.fft(frame)
                log_spectrum = np.log(np.abs(spectrum) + 1e-8)
                cepstrum = np.fft.ifft(log_spectrum).real

                # Lifter cepstrum to remove echo
                liftered_cepstrum = cepstrum.copy()
                # Zero out high quefrency components (echo)
                quefrency_cutoff = len(cepstrum) // 4
                liftered_cepstrum[quefrency_cutoff:] = 0
                liftered_cepstrum[-quefrency_cutoff:] = 0

                # Reconstruct spectrum
                log_spectrum_enhanced = np.fft.fft(liftered_cepstrum).real
                spectrum_enhanced = np.exp(log_spectrum_enhanced) * np.exp(
                    1j * np.angle(spectrum)
                )

                # Convert back to time domain
                enhanced_frame = np.fft.ifft(spectrum_enhanced).real
                enhanced_frames.append(enhanced_frame[:hop_size])

            # Concatenate frames
            enhanced_audio = np.concatenate(enhanced_frames)

            # Pad to original length
            if len(enhanced_audio) < len(audio_data):
                enhanced_audio = np.pad(
                    enhanced_audio, (0, len(audio_data) - len(enhanced_audio))
                )
            else:
                enhanced_audio = enhanced_audio[: len(audio_data)]

            return enhanced_audio.astype(audio_data.dtype)

        except Exception as e:
            logger.error(f"Cepstral echo removal error: {e}")
            return audio_data

    def remove_echo(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply selected echo removal method
        """
        if not self.config.echo_removal:
            return audio_data

        method = self.config.echo_method

        if method == EchoRemovalMethod.ADAPTIVE_ECHO_CANCELLATION:
            return self.adaptive_echo_cancellation(audio_data)
        elif method == EchoRemovalMethod.SPECTRAL_SUBTRACTION:
            return self.spectral_echo_removal(audio_data)
        elif method == EchoRemovalMethod.CEPSTRAL_FILTERING:
            return self.cepstral_echo_removal(audio_data)
        else:
            return self.adaptive_echo_cancellation(audio_data)


class AudioRestorer:
    """
    Audio restoration for damaged or degraded recordings
    """

    def __init__(self, config: EnhancementConfig):
        self.config = config

    def declip_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Repair clipped/distorted audio using interpolation
        """
        if not HAS_AUDIO_LIBS:
            return audio_data

        try:
            # Detect clipping
            threshold = 0.95  # Clipping detection threshold
            clipped_mask = np.abs(audio_data) > threshold

            if not np.any(clipped_mask):
                return audio_data  # No clipping detected

            enhanced_audio = audio_data.copy()

            # Find clipped regions
            clipped_regions = self._find_clipped_regions(clipped_mask)

            for start, end in clipped_regions:
                if end - start < 100:  # Only repair short clips
                    # Cubic interpolation
                    if start > 0 and end < len(audio_data) - 1:
                        x = np.array([start - 1, end + 1])
                        y = np.array([audio_data[start - 1], audio_data[end + 1]])

                        # Interpolate
                        interp_indices = np.arange(start, end + 1)
                        interpolated = np.interp(interp_indices, x, y)
                        enhanced_audio[start : end + 1] = interpolated

            return enhanced_audio

        except Exception as e:
            logger.error(f"Declipping error: {e}")
            return audio_data

    def _find_clipped_regions(self, clipped_mask: np.ndarray) -> List[Tuple[int, int]]:
        """Find contiguous clipped regions"""
        regions = []
        in_region = False
        start = 0

        for i, is_clipped in enumerate(clipped_mask):
            if is_clipped and not in_region:
                start = i
                in_region = True
            elif not is_clipped and in_region:
                regions.append((start, i - 1))
                in_region = False

        # Handle region extending to end
        if in_region:
            regions.append((start, len(clipped_mask) - 1))

        return regions

    def bandwidth_extension(self, audio_data: np.ndarray, target_sr: int) -> np.ndarray:
        """
        Extend bandwidth of audio using harmonic generation
        """
        if not HAS_AUDIO_LIBS:
            return audio_data

        try:
            current_sr = self.config.sample_rate

            if target_sr <= current_sr:
                return audio_data

            # Resample to higher rate
            upsampled = librosa.resample(
                audio_data, orig_sr=current_sr, target_sr=target_sr
            )

            # Generate high-frequency content using nonlinear processing
            # Simple approach: add harmonics
            stft = librosa.stft(upsampled, hop_length=512, n_fft=4096)
            magnitude = np.abs(stft)
            phase = np.angle(stft)

            # Extend high frequencies
            nyquist_orig = current_sr // 2
            nyquist_new = target_sr // 2

            # Find frequency bins
            freqs = librosa.fft_frequencies(sr=target_sr, n_fft=4096)
            orig_mask = freqs <= nyquist_orig
            extend_mask = (freqs > nyquist_orig) & (freqs <= nyquist_new)

            # Copy and attenuate existing content to high frequencies
            if np.any(extend_mask) and np.any(orig_mask):
                # Simple harmonic extension
                orig_bins = np.where(orig_mask)[0]
                extend_bins = np.where(extend_mask)[0]

                if len(orig_bins) > 0 and len(extend_bins) > 0:
                    # Map high frequencies from lower frequencies
                    for i, high_bin in enumerate(extend_bins):
                        if i < len(orig_bins):
                            low_bin = orig_bins[-(i + 1)]  # Use high-frequency content
                            magnitude[high_bin, :] = (
                                magnitude[low_bin, :] * 0.3
                            )  # Attenuated
                            phase[high_bin, :] = phase[low_bin, :] + np.random.uniform(
                                -0.1, 0.1, size=phase.shape[1]
                            )

            # Reconstruct
            extended_stft = magnitude * np.exp(1j * phase)
            extended_audio = librosa.istft(
                extended_stft, hop_length=512, length=len(upsampled)
            )

            return extended_audio.astype(audio_data.dtype)

        except Exception as e:
            logger.error(f"Bandwidth extension error: {e}")
            return audio_data

    def restore_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply complete audio restoration pipeline
        """
        if not self.config.restoration:
            return audio_data

        enhanced_audio = audio_data.copy()

        if self.config.declipping:
            enhanced_audio = self.declip_audio(enhanced_audio)

        if self.config.bandwidth_extension:
            enhanced_audio = self.bandwidth_extension(
                enhanced_audio, self.config.target_sample_rate
            )

        return enhanced_audio


class DynamicRangeProcessor:
    """
    Dynamic range compression and expansion
    """

    def __init__(self, config: EnhancementConfig):
        self.config = config
        self.envelope_follower = None

    def soft_knee_compressor(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Soft-knee compressor with attack and release
        """
        if not HAS_AUDIO_LIBS:
            return audio_data

        try:
            # Parameters
            threshold_linear = 10 ** (self.config.threshold_db / 20)
            ratio = self.config.ratio
            attack_coeff = np.exp(
                -1 / (self.config.attack_ms * 0.001 * self.config.sample_rate)
            )
            release_coeff = np.exp(
                -1 / (self.config.release_ms * 0.001 * self.config.sample_rate)
            )

            # Initialize envelope follower
            envelope = 0.0
            compressed_audio = np.zeros_like(audio_data)

            for i, sample in enumerate(audio_data):
                # Envelope detection
                input_level = abs(sample)

                if input_level > envelope:
                    envelope = (
                        attack_coeff * envelope + (1 - attack_coeff) * input_level
                    )
                else:
                    envelope = (
                        release_coeff * envelope + (1 - release_coeff) * input_level
                    )

                # Compression calculation
                if envelope > threshold_linear:
                    # Soft knee calculation
                    knee_width = 0.1  # 10% soft knee
                    over_threshold = envelope - threshold_linear

                    if over_threshold < knee_width * threshold_linear:
                        # Soft knee region
                        compression_ratio = (
                            1
                            + (ratio - 1)
                            * (over_threshold / (knee_width * threshold_linear)) ** 2
                        )
                    else:
                        # Full compression
                        compression_ratio = ratio

                    # Calculate gain reduction
                    gain_reduction = over_threshold / compression_ratio
                    output_level = threshold_linear + gain_reduction
                    gain = output_level / envelope if envelope > 0 else 1.0
                else:
                    gain = 1.0

                # Apply gain
                compressed_audio[i] = sample * gain

            return compressed_audio

        except Exception as e:
            logger.error(f"Compression error: {e}")
            return audio_data

    def multiband_compressor(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Multi-band compressor with frequency-specific compression
        """
        if not HAS_AUDIO_LIBS:
            return audio_data

        try:
            # Define frequency bands
            bands = [
                (20, 250),  # Low
                (250, 2000),  # Mid
                (2000, 8000),  # High
                (8000, 20000),  # Very High
            ]

            compressed_bands = []

            for low_freq, high_freq in bands:
                # Bandpass filter
                nyquist = self.config.sample_rate / 2
                low = low_freq / nyquist
                high = min(high_freq / nyquist, 0.99)

                if low < high:
                    if low <= 0:
                        # Low-pass filter
                        sos = butter(4, high, btype="low", output="sos")
                    elif high >= 0.99:
                        # High-pass filter
                        sos = butter(4, low, btype="high", output="sos")
                    else:
                        # Bandpass filter
                        sos = butter(4, [low, high], btype="band", output="sos")

                    # Apply filter
                    band_audio = sosfilt(sos, audio_data)

                    # Compress band
                    compressed_band = self.soft_knee_compressor(band_audio)
                    compressed_bands.append(compressed_band)
                else:
                    compressed_bands.append(np.zeros_like(audio_data))

            # Sum bands
            compressed_audio = np.sum(compressed_bands, axis=0)

            return compressed_audio.astype(audio_data.dtype)

        except Exception as e:
            logger.error(f"Multiband compression error: {e}")
            return audio_data

    def apply_compression(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply selected compression type
        """
        if not self.config.compression:
            return audio_data

        comp_type = self.config.compression_type

        if comp_type == CompressionType.MULTIBAND:
            return self.multiband_compressor(audio_data)
        else:
            return self.soft_knee_compressor(audio_data)


class AudioEqualizer:
    """
    Parametric equalizer with multiple band types
    """

    def __init__(self, config: EnhancementConfig):
        self.config = config

    def apply_eq(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply equalization based on configured bands
        """
        if (
            not self.config.equalization
            or not self.config.eq_bands
            or not HAS_AUDIO_LIBS
        ):
            return audio_data

        try:
            equalized_audio = audio_data.copy()

            for freq, gain_db in self.config.eq_bands:
                if gain_db != 0:  # Skip bands with no gain change
                    # Apply peaking EQ filter
                    equalized_audio = self._apply_peaking_filter(
                        equalized_audio, freq, gain_db, 1.0
                    )

            return equalized_audio

        except Exception as e:
            logger.error(f"Equalization error: {e}")
            return audio_data

    def _apply_peaking_filter(
        self, audio_data: np.ndarray, freq: float, gain_db: float, q: float
    ) -> np.ndarray:
        """
        Apply peaking EQ filter at specified frequency
        """
        try:
            # Convert parameters
            gain_linear = 10 ** (gain_db / 20)
            omega = 2 * np.pi * freq / self.config.sample_rate

            # Peaking filter coefficients
            alpha = np.sin(omega) / (2 * q)
            a0 = 1 + alpha / gain_linear
            a1 = -2 * np.cos(omega)
            a2 = 1 - alpha / gain_linear
            b0 = 1 + alpha * gain_linear
            b1 = -2 * np.cos(omega)
            b2 = 1 - alpha * gain_linear

            # Normalize
            b = [b0 / a0, b1 / a0, b2 / a0]
            a = [1, a1 / a0, a2 / a0]

            # Apply filter
            filtered_audio = scipy.signal.lfilter(b, a, audio_data)

            return filtered_audio.astype(audio_data.dtype)

        except Exception as e:
            logger.debug(f"Peaking filter error at {freq}Hz: {e}")
            return audio_data


class AudioEnhancer:
    """
    Main audio enhancement system combining all processing modules
    """

    def __init__(self, config: Optional[EnhancementConfig] = None):
        self.config = config or EnhancementConfig()

        # Initialize processing modules
        self.noise_reducer = SpectralNoiseReducer(self.config)
        self.echo_remover = EchoRemover(self.config)
        self.audio_restorer = AudioRestorer(self.config)
        self.compressor = DynamicRangeProcessor(self.config)
        self.equalizer = AudioEqualizer(self.config)

    async def enhance_audio(
        self, audio_data: np.ndarray, speech_mask: Optional[np.ndarray] = None
    ) -> ProcessingResult:
        """
        Apply complete audio enhancement pipeline
        """
        if not HAS_AUDIO_LIBS:
            raise AudioEnhancementError("Required audio libraries not available")

        import time

        start_time = time.time()

        original_audio = audio_data.copy()
        enhanced_audio = audio_data.copy()
        processing_stats = {}

        try:
            # Step 1: Noise Reduction
            if self.config.noise_reduction:
                logger.debug("Applying noise reduction...")
                step_start = time.time()
                enhanced_audio = self.noise_reducer.reduce_noise(
                    enhanced_audio, speech_mask
                )
                processing_stats["noise_reduction_time"] = time.time() - step_start

            # Step 2: Echo Removal
            if self.config.echo_removal:
                logger.debug("Removing echo...")
                step_start = time.time()
                enhanced_audio = self.echo_remover.remove_echo(enhanced_audio)
                processing_stats["echo_removal_time"] = time.time() - step_start

            # Step 3: Audio Restoration
            if self.config.restoration:
                logger.debug("Restoring audio...")
                step_start = time.time()
                enhanced_audio = self.audio_restorer.restore_audio(enhanced_audio)
                processing_stats["restoration_time"] = time.time() - step_start

            # Step 4: Dynamic Range Processing
            if self.config.compression:
                logger.debug("Applying compression...")
                step_start = time.time()
                enhanced_audio = self.compressor.apply_compression(enhanced_audio)
                processing_stats["compression_time"] = time.time() - step_start

            # Step 5: Equalization
            if self.config.equalization:
                logger.debug("Applying equalization...")
                step_start = time.time()
                enhanced_audio = self.equalizer.apply_eq(enhanced_audio)
                processing_stats["equalization_time"] = time.time() - step_start

            # Step 6: Quality Enhancement
            enhanced_audio = await self._apply_quality_enhancements(enhanced_audio)

            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(
                original_audio, enhanced_audio
            )

            total_time = time.time() - start_time

            return ProcessingResult(
                enhanced_audio=enhanced_audio,
                original_audio=original_audio,
                sample_rate=self.config.sample_rate,
                processing_stats=processing_stats,
                quality_metrics=quality_metrics,
                processing_time=total_time,
            )

        except Exception as e:
            logger.error(f"Audio enhancement error: {e}")
            raise AudioEnhancementError(f"Enhancement pipeline failed: {e}")

    async def _apply_quality_enhancements(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply additional quality enhancements"""
        enhanced_audio = audio_data.copy()

        try:
            # Stereo enhancement (if applicable)
            if self.config.stereo_enhancement and audio_data.ndim > 1:
                enhanced_audio = self._enhance_stereo(enhanced_audio)

            # Harmonic enhancement
            if self.config.harmonic_enhancement:
                enhanced_audio = self._enhance_harmonics(enhanced_audio)

            # Upsampling
            if (
                self.config.upsampling
                and self.config.target_sample_rate > self.config.sample_rate
            ):
                enhanced_audio = librosa.resample(
                    enhanced_audio,
                    orig_sr=self.config.sample_rate,
                    target_sr=self.config.target_sample_rate,
                )

        except Exception as e:
            logger.debug(f"Quality enhancement error: {e}")

        return enhanced_audio

    def _enhance_stereo(self, audio_data: np.ndarray) -> np.ndarray:
        """Enhance stereo width and separation"""
        if audio_data.ndim != 2 or audio_data.shape[1] != 2:
            return audio_data

        try:
            # M/S processing for stereo enhancement
            left = audio_data[:, 0]
            right = audio_data[:, 1]

            # Convert to Mid/Side
            mid = (left + right) / 2
            side = (left - right) / 2

            # Enhance side signal
            enhanced_side = side * 1.3  # Increase stereo width

            # Convert back to L/R
            enhanced_left = mid + enhanced_side
            enhanced_right = mid - enhanced_side

            return np.column_stack([enhanced_left, enhanced_right])

        except Exception as e:
            logger.debug(f"Stereo enhancement error: {e}")
            return audio_data

    def _enhance_harmonics(self, audio_data: np.ndarray) -> np.ndarray:
        """Enhance harmonic content for warmth"""
        try:
            # Generate harmonics using soft saturation
            enhanced_audio = audio_data.copy()

            # Soft clipping to generate harmonics
            drive = 0.1
            enhanced_audio = np.tanh(enhanced_audio * (1 + drive)) / (1 + drive)

            # Mix with original
            mix = 0.2  # 20% enhancement
            enhanced_audio = (1 - mix) * audio_data + mix * enhanced_audio

            return enhanced_audio

        except Exception as e:
            logger.debug(f"Harmonic enhancement error: {e}")
            return audio_data

    def _calculate_quality_metrics(
        self, original: np.ndarray, enhanced: np.ndarray
    ) -> Dict[str, float]:
        """Calculate audio quality metrics"""
        metrics = {}

        try:
            # Signal-to-noise ratio improvement
            original_noise = np.std(original)
            enhanced_noise = np.std(enhanced - original)

            if enhanced_noise > 0:
                snr_improvement = 20 * np.log10(original_noise / enhanced_noise)
                metrics["snr_improvement_db"] = float(snr_improvement)

            # RMS level change
            original_rms = np.sqrt(np.mean(original**2))
            enhanced_rms = np.sqrt(np.mean(enhanced**2))

            if original_rms > 0:
                rms_change_db = 20 * np.log10(enhanced_rms / original_rms)
                metrics["rms_change_db"] = float(rms_change_db)

            # Peak level change
            original_peak = np.max(np.abs(original))
            enhanced_peak = np.max(np.abs(enhanced))

            if original_peak > 0:
                peak_change_db = 20 * np.log10(enhanced_peak / original_peak)
                metrics["peak_change_db"] = float(peak_change_db)

            # Spectral similarity
            if HAS_AUDIO_LIBS:
                orig_spec = np.abs(librosa.stft(original))
                enh_spec = np.abs(librosa.stft(enhanced))

                # Correlation coefficient
                correlation = np.corrcoef(orig_spec.flatten(), enh_spec.flatten())[0, 1]
                metrics["spectral_correlation"] = float(correlation)

        except Exception as e:
            logger.debug(f"Quality metrics calculation error: {e}")

        return metrics

    async def enhance_file(self, input_path: str, output_path: str) -> ProcessingResult:
        """
        Enhance audio file and save result
        """
        try:
            # Load audio
            if HAS_SOUNDFILE:
                audio_data, sr = sf.read(input_path)
            elif HAS_AUDIO_LIBS:
                audio_data, sr = librosa.load(input_path, sr=None, mono=False)
            else:
                raise AudioEnhancementError("No audio loading library available")

            # Update config sample rate
            self.config.sample_rate = sr

            # Process audio
            result = await self.enhance_audio(audio_data)

            # Save enhanced audio
            if HAS_SOUNDFILE:
                sf.write(output_path, result.enhanced_audio, sr)
            else:
                raise AudioEnhancementError("SoundFile not available for saving")

            logger.info(f"Enhanced audio saved to {output_path}")
            return result

        except Exception as e:
            logger.error(f"File enhancement error: {e}")
            raise AudioEnhancementError(f"Failed to enhance file {input_path}: {e}")


# Convenience functions


def create_enhancement_config(**kwargs) -> EnhancementConfig:
    """Create enhancement configuration with custom parameters"""
    return EnhancementConfig(**kwargs)


async def enhance_audio_file(
    input_path: str, output_path: str, config: Optional[EnhancementConfig] = None
) -> ProcessingResult:
    """Convenience function to enhance audio file"""
    enhancer = AudioEnhancer(config)
    return await enhancer.enhance_file(input_path, output_path)


class NoiseReducer(SpectralNoiseReducer):
    """Alias for backward compatibility"""

    pass


class EchoRemover(EchoRemover):
    """Alias for backward compatibility"""

    pass


# Example usage and testing
if __name__ == "__main__":

    async def demo():
        """Demo audio enhancement"""
        try:
            # Create configuration
            config = EnhancementConfig(
                noise_reduction=True,
                echo_removal=True,
                restoration=True,
                compression=False,
                equalization=True,
                eq_bands=[(1000, 3.0), (3000, -2.0)],  # Boost 1kHz, cut 3kHz
            )

            # Create enhancer
            enhancer = AudioEnhancer(config)

            print("Creating synthetic test audio...")

            # Generate test audio with noise
            duration = 3  # seconds
            sr = 44100
            t = np.linspace(0, duration, duration * sr)

            # Clean signal: sine wave + harmonics
            clean_signal = (
                np.sin(2 * np.pi * 440 * t) * 0.5
                + np.sin(2 * np.pi * 880 * t) * 0.25
                + np.sin(2 * np.pi * 1320 * t) * 0.125
            )

            # Add noise and distortion
            noise = np.random.normal(0, 0.1, len(clean_signal))
            echo_delay = int(0.2 * sr)  # 200ms echo
            echo_signal = np.roll(clean_signal, echo_delay) * 0.3

            # Combine
            degraded_signal = clean_signal + noise + echo_signal

            # Add some clipping
            degraded_signal = np.clip(degraded_signal * 1.5, -1.0, 1.0)

            print("Enhancing audio...")
            result = await enhancer.enhance_audio(degraded_signal)

            print(f"\n=== ENHANCEMENT RESULTS ===")
            print(f"Processing time: {result.processing_time:.3f}s")
            print(f"Quality metrics: {result.quality_metrics}")
            print(f"Processing stats: {result.processing_stats}")

            # Calculate improvement metrics
            original_rms = np.sqrt(np.mean(result.original_audio**2))
            enhanced_rms = np.sqrt(np.mean(result.enhanced_audio**2))
            print(
                f"RMS level change: {20 * np.log10(enhanced_rms / original_rms):.2f} dB"
            )

            print("\nAudio enhancement demo completed successfully!")

        except Exception as e:
            print(f"Demo error: {e}")

    # Run demo
    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
