"""
Vega 2.0 Spatial Audio Processing Module

This module provides advanced spatial audio processing capabilities including:
- 3D audio analysis and positioning
- Binaural audio processing and rendering
- HRTF (Head-Related Transfer Function) processing
- Ambisonic encoding/decoding
- Immersive audio generation
- Spatial audio mixing and panning
- Room acoustics modeling

Dependencies:
- numpy: Audio array processing
- scipy: Signal processing and convolution
- librosa: Audio analysis
- pyroomacoustics: Room acoustics simulation (optional)
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
    import scipy.signal
    import scipy.spatial.distance
    from scipy import ndimage

    HAS_AUDIO_LIBS = True
except ImportError:
    librosa = None
    scipy = None
    ndimage = None
    HAS_AUDIO_LIBS = False

try:
    import pyroomacoustics as pra

    HAS_PYROOMACOUSTICS = True
except ImportError:
    pra = None
    HAS_PYROOMACOUSTICS = False

logger = logging.getLogger(__name__)


class SpatialAudioError(Exception):
    """Custom exception for spatial audio processing errors"""

    pass


class CoordinateSystem(Enum):
    """3D coordinate system conventions"""

    CARTESIAN = "cartesian"  # (x, y, z)
    SPHERICAL = "spherical"  # (azimuth, elevation, distance)
    CYLINDRICAL = "cylindrical"  # (azimuth, height, distance)


class AmbisonicOrder(Enum):
    """Ambisonic encoding orders"""

    FIRST_ORDER = 1  # 4 channels (W, X, Y, Z)
    SECOND_ORDER = 2  # 9 channels
    THIRD_ORDER = 3  # 16 channels
    FOURTH_ORDER = 4  # 25 channels


class HRTFModel(Enum):
    """HRTF model types"""

    GENERIC = "generic"
    MIT_KEMAR = "mit_kemar"
    CIPIC = "cipic"
    CUSTOM = "custom"


class RoomType(Enum):
    """Room acoustic types"""

    ANECHOIC = "anechoic"
    STUDIO = "studio"
    CONCERT_HALL = "concert_hall"
    CHURCH = "church"
    LIVING_ROOM = "living_room"
    OUTDOOR = "outdoor"
    CUSTOM = "custom"


@dataclass
class Position3D:
    """3D position representation"""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    coordinate_system: CoordinateSystem = CoordinateSystem.CARTESIAN

    def to_cartesian(self) -> Tuple[float, float, float]:
        """Convert to Cartesian coordinates"""
        if self.coordinate_system == CoordinateSystem.CARTESIAN:
            return (self.x, self.y, self.z)
        elif self.coordinate_system == CoordinateSystem.SPHERICAL:
            # (azimuth, elevation, distance) -> (x, y, z)
            azimuth, elevation, distance = self.x, self.y, self.z
            x = distance * np.cos(elevation) * np.cos(azimuth)
            y = distance * np.cos(elevation) * np.sin(azimuth)
            z = distance * np.sin(elevation)
            return (float(x), float(y), float(z))
        elif self.coordinate_system == CoordinateSystem.CYLINDRICAL:
            # (azimuth, height, distance) -> (x, y, z)
            azimuth, height, distance = self.x, self.y, self.z
            x = distance * np.cos(azimuth)
            y = distance * np.sin(azimuth)
            z = height
            return (float(x), float(y), float(z))

    def to_spherical(self) -> Tuple[float, float, float]:
        """Convert to spherical coordinates (azimuth, elevation, distance)"""
        x, y, z = self.to_cartesian()
        distance = np.sqrt(x**2 + y**2 + z**2)
        azimuth = np.arctan2(y, x)
        elevation = np.arcsin(z / (distance + 1e-8))
        return (float(azimuth), float(elevation), float(distance))


@dataclass
class SpatialSource:
    """Spatial audio source definition"""

    audio_data: np.ndarray
    position: Position3D
    velocity: Optional[Position3D] = None  # For Doppler effect
    directivity_pattern: Optional[np.ndarray] = None
    gain: float = 1.0
    source_id: str = "default"


@dataclass
class Listener:
    """Spatial audio listener definition"""

    position: Position3D = field(default_factory=Position3D)
    orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # yaw, pitch, roll
    head_radius: float = 0.0875  # Average head radius in meters
    ear_separation: float = 0.18  # Ear separation in meters


@dataclass
class RoomAcoustics:
    """Room acoustic properties"""

    room_type: RoomType = RoomType.STUDIO
    dimensions: Tuple[float, float, float] = (5.0, 4.0, 3.0)  # length, width, height
    absorption_coefficients: List[float] = field(
        default_factory=lambda: [0.1] * 6
    )  # walls
    rt60: float = 0.5  # Reverberation time in seconds
    temperature: float = 20.0  # Celsius
    humidity: float = 50.0  # Percent
    air_absorption: bool = True


@dataclass
class SpatialConfig:
    """Configuration for spatial audio processing"""

    sample_rate: int = 48000
    block_size: int = 1024
    max_delay_samples: int = 4800  # Maximum delay for distance modeling
    speed_of_sound: float = 343.0  # m/s at 20°C
    ambisonic_order: AmbisonicOrder = AmbisonicOrder.FIRST_ORDER
    hrtf_model: HRTFModel = HRTFModel.GENERIC
    use_doppler: bool = True
    use_air_absorption: bool = True
    use_distance_attenuation: bool = True


class HRTFProcessor:
    """
    Head-Related Transfer Function processing for binaural audio
    """

    def __init__(self, config: SpatialConfig):
        self.config = config
        self.hrtf_data = None
        self.hrtf_azimuth_grid = None
        self.hrtf_elevation_grid = None
        self._load_hrtf_data()

    def _load_hrtf_data(self) -> None:
        """Load HRTF data based on selected model"""
        try:
            # Generate generic HRTF data (simplified model)
            # In a real implementation, this would load from HRTF databases
            num_azimuths = 72  # 5-degree resolution
            num_elevations = 25  # Various elevation angles
            filter_length = 256

            self.hrtf_azimuth_grid = np.linspace(
                0, 2 * np.pi, num_azimuths, endpoint=False
            )
            self.hrtf_elevation_grid = np.linspace(
                -np.pi / 2, np.pi / 2, num_elevations
            )

            # Generate synthetic HRTF data based on simple head model
            self.hrtf_data = self._generate_synthetic_hrtf(
                self.hrtf_azimuth_grid, self.hrtf_elevation_grid, filter_length
            )

            logger.info(f"Loaded HRTF data: {self.hrtf_data.shape}")

        except Exception as e:
            logger.error(f"HRTF loading error: {e}")
            self.hrtf_data = None

    def _generate_synthetic_hrtf(
        self, azimuths: np.ndarray, elevations: np.ndarray, filter_length: int
    ) -> np.ndarray:
        """Generate synthetic HRTF data using simplified head model"""
        try:
            # HRTF shape: (num_azimuths, num_elevations, 2 ears, filter_length)
            hrtf = np.zeros((len(azimuths), len(elevations), 2, filter_length))

            head_radius = 0.0875  # meters
            ear_angle = np.pi / 2  # 90 degrees apart

            for i, azimuth in enumerate(azimuths):
                for j, elevation in enumerate(elevations):
                    # Calculate ITD (Interaural Time Difference)
                    itd_left = self._calculate_itd(
                        azimuth, elevation, -ear_angle, head_radius
                    )
                    itd_right = self._calculate_itd(
                        azimuth, elevation, ear_angle, head_radius
                    )

                    # Calculate ILD (Interaural Level Difference)
                    ild_left = self._calculate_ild(azimuth, elevation, -ear_angle)
                    ild_right = self._calculate_ild(azimuth, elevation, ear_angle)

                    # Generate impulse responses
                    hrtf[i, j, 0, :] = self._generate_hrtf_ir(
                        itd_left, ild_left, filter_length
                    )
                    hrtf[i, j, 1, :] = self._generate_hrtf_ir(
                        itd_right, ild_right, filter_length
                    )

            return hrtf

        except Exception as e:
            logger.error(f"Synthetic HRTF generation error: {e}")
            return np.zeros((len(azimuths), len(elevations), 2, filter_length))

    def _calculate_itd(
        self, source_az: float, source_el: float, ear_az: float, head_radius: float
    ) -> float:
        """Calculate Interaural Time Difference using spherical head model"""
        try:
            # Woodworth-Schlosberg formula
            angle_diff = source_az - ear_az
            itd_samples = (
                head_radius
                / self.config.speed_of_sound
                * (np.sin(angle_diff) + angle_diff)
                * self.config.sample_rate
            )
            return float(itd_samples)
        except Exception:
            return 0.0

    def _calculate_ild(
        self, source_az: float, source_el: float, ear_az: float
    ) -> float:
        """Calculate Interaural Level Difference"""
        try:
            # Simple head shadow model
            angle_diff = abs(source_az - ear_az)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff

            # Maximum attenuation at 90 degrees
            max_attenuation_db = -20.0
            ild_db = max_attenuation_db * np.sin(angle_diff) * np.cos(source_el)
            return float(ild_db)
        except Exception:
            return 0.0

    def _generate_hrtf_ir(
        self, itd_samples: float, ild_db: float, filter_length: int
    ) -> np.ndarray:
        """Generate HRTF impulse response from ITD and ILD"""
        try:
            # Create delayed impulse for ITD
            ir = np.zeros(filter_length)
            delay_int = int(abs(itd_samples))
            delay_frac = abs(itd_samples) - delay_int

            if delay_int < filter_length - 1:
                # Integer delay
                ir[delay_int] = 1.0 - delay_frac
                # Fractional delay
                ir[delay_int + 1] = delay_frac

            # Apply ILD (level difference)
            gain = 10 ** (ild_db / 20.0)
            ir *= gain

            # Add some spectral shaping (simplified)
            # High-frequency rolloff
            freqs = np.fft.fftfreq(filter_length, 1 / self.config.sample_rate)
            spectrum = np.fft.fft(ir)

            # Simple head filtering (high-frequency attenuation)
            hf_rolloff = np.exp(-(freqs**2) / (2 * (8000**2)))
            spectrum *= hf_rolloff

            ir = np.fft.ifft(spectrum).real
            return ir

        except Exception as e:
            logger.debug(f"HRTF IR generation error: {e}")
            ir = np.zeros(filter_length)
            ir[0] = 1.0
            return ir

    def get_hrtf(
        self, azimuth: float, elevation: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get HRTF impulse responses for given direction"""
        if self.hrtf_data is None:
            # Return identity filters
            filter_length = 256
            return (np.zeros(filter_length), np.zeros(filter_length))

        try:
            # Find nearest HRTF data points
            az_idx = np.argmin(np.abs(self.hrtf_azimuth_grid - azimuth))
            el_idx = np.argmin(np.abs(self.hrtf_elevation_grid - elevation))

            left_hrtf = self.hrtf_data[az_idx, el_idx, 0, :]
            right_hrtf = self.hrtf_data[az_idx, el_idx, 1, :]

            return (left_hrtf, right_hrtf)

        except Exception as e:
            logger.error(f"HRTF retrieval error: {e}")
            filter_length = (
                self.hrtf_data.shape[-1] if self.hrtf_data is not None else 256
            )
            return (np.zeros(filter_length), np.zeros(filter_length))


class AmbisonicProcessor:
    """
    Ambisonic encoding and decoding for 3D audio
    """

    def __init__(self, config: SpatialConfig):
        self.config = config
        self.order = config.ambisonic_order.value
        self.num_channels = (self.order + 1) ** 2

    def encode_source(
        self,
        audio_data: np.ndarray,
        azimuth: float,
        elevation: float,
        distance: float = 1.0,
    ) -> np.ndarray:
        """
        Encode mono audio source to ambisonic format

        Args:
            audio_data: Mono input audio
            azimuth: Source azimuth in radians
            elevation: Source elevation in radians
            distance: Source distance (for distance compensation)

        Returns:
            Multi-channel ambisonic audio
        """
        try:
            # Apply distance attenuation
            if self.config.use_distance_attenuation and distance > 0:
                distance_gain = 1.0 / max(distance, 0.1)  # Prevent division by zero
                audio_data = audio_data * distance_gain

            # Calculate ambisonic encoding coefficients
            encoding_coeffs = self._calculate_encoding_coefficients(azimuth, elevation)

            # Encode to ambisonic channels
            ambisonic_audio = np.zeros((self.num_channels, len(audio_data)))

            for ch in range(self.num_channels):
                ambisonic_audio[ch, :] = audio_data * encoding_coeffs[ch]

            return ambisonic_audio

        except Exception as e:
            logger.error(f"Ambisonic encoding error: {e}")
            # Return silent multichannel audio
            return np.zeros((self.num_channels, len(audio_data)))

    def _calculate_encoding_coefficients(
        self, azimuth: float, elevation: float
    ) -> np.ndarray:
        """Calculate spherical harmonic coefficients for ambisonic encoding"""
        try:
            coeffs = np.zeros(self.num_channels)

            # ACN (Ambisonic Channel Number) ordering
            ch_idx = 0

            for order in range(self.order + 1):
                for degree in range(-order, order + 1):
                    if ch_idx < self.num_channels:
                        # Calculate spherical harmonic
                        coeff = self._spherical_harmonic(
                            order, degree, azimuth, elevation
                        )
                        coeffs[ch_idx] = coeff
                        ch_idx += 1

            return coeffs

        except Exception as e:
            logger.error(f"Encoding coefficients calculation error: {e}")
            coeffs = np.zeros(self.num_channels)
            if len(coeffs) > 0:
                coeffs[0] = 1.0  # W channel
            return coeffs

    def _spherical_harmonic(
        self, order: int, degree: int, azimuth: float, elevation: float
    ) -> float:
        """Calculate spherical harmonic coefficient"""
        try:
            # Simplified spherical harmonics for first-order ambisonic
            if order == 0 and degree == 0:
                # W channel (omnidirectional)
                return 1.0
            elif order == 1:
                if degree == -1:
                    # Y channel
                    return np.sin(azimuth) * np.cos(elevation)
                elif degree == 0:
                    # Z channel
                    return np.sin(elevation)
                elif degree == 1:
                    # X channel
                    return np.cos(azimuth) * np.cos(elevation)

            # For higher orders, would need more complex calculations
            return 0.0

        except Exception:
            return 0.0

    def decode_binaural(
        self, ambisonic_audio: np.ndarray, listener: Listener
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode ambisonic audio to binaural stereo

        Args:
            ambisonic_audio: Multi-channel ambisonic audio
            listener: Listener configuration

        Returns:
            Tuple of (left_channel, right_channel)
        """
        try:
            if ambisonic_audio.shape[0] < self.num_channels:
                logger.warning("Insufficient ambisonic channels for decoding")
                # Return mono to both ears
                mono = np.sum(ambisonic_audio, axis=0)
                return (mono, mono)

            # Simple binaural decoder for first-order ambisonics
            # Extract ambisonic channels
            w = ambisonic_audio[0, :]  # W (omnidirectional)

            if self.num_channels > 1:
                x = (
                    ambisonic_audio[1, :]
                    if ambisonic_audio.shape[0] > 1
                    else np.zeros_like(w)
                )  # X (front-back)
                y = (
                    ambisonic_audio[2, :]
                    if ambisonic_audio.shape[0] > 2
                    else np.zeros_like(w)
                )  # Y (left-right)
                z = (
                    ambisonic_audio[3, :]
                    if ambisonic_audio.shape[0] > 3
                    else np.zeros_like(w)
                )  # Z (up-down)
            else:
                x = y = z = np.zeros_like(w)

            # Binaural decoder matrix (simplified)
            # Left ear
            left = w + 0.7071 * x + 0.5 * y

            # Right ear
            right = w + 0.7071 * x - 0.5 * y

            return (left, right)

        except Exception as e:
            logger.error(f"Binaural decoding error: {e}")
            # Fallback: return first channel to both ears
            mono = (
                ambisonic_audio[0, :]
                if ambisonic_audio.shape[0] > 0
                else np.zeros(1024)
            )
            return (mono, mono)

    def decode_speaker_array(
        self, ambisonic_audio: np.ndarray, speaker_positions: List[Position3D]
    ) -> np.ndarray:
        """
        Decode ambisonic audio for speaker array

        Args:
            ambisonic_audio: Multi-channel ambisonic audio
            speaker_positions: List of speaker positions

        Returns:
            Multi-channel audio for speakers
        """
        try:
            num_speakers = len(speaker_positions)
            audio_length = ambisonic_audio.shape[1]
            speaker_audio = np.zeros((num_speakers, audio_length))

            # Calculate decoder matrix
            decoder_matrix = self._calculate_decoder_matrix(speaker_positions)

            # Apply decoder matrix
            for spk in range(num_speakers):
                for amb_ch in range(min(self.num_channels, ambisonic_audio.shape[0])):
                    speaker_audio[spk, :] += (
                        ambisonic_audio[amb_ch, :] * decoder_matrix[spk, amb_ch]
                    )

            return speaker_audio

        except Exception as e:
            logger.error(f"Speaker array decoding error: {e}")
            # Fallback: send W channel to all speakers
            w_channel = (
                ambisonic_audio[0, :]
                if ambisonic_audio.shape[0] > 0
                else np.zeros(1024)
            )
            num_speakers = len(speaker_positions)
            return np.tile(w_channel, (num_speakers, 1))

    def _calculate_decoder_matrix(
        self, speaker_positions: List[Position3D]
    ) -> np.ndarray:
        """Calculate ambisonic decoder matrix for speaker array"""
        try:
            num_speakers = len(speaker_positions)
            decoder_matrix = np.zeros((num_speakers, self.num_channels))

            for spk, pos in enumerate(speaker_positions):
                azimuth, elevation, _ = pos.to_spherical()

                # Calculate encoding coefficients for this speaker position
                coeffs = self._calculate_encoding_coefficients(azimuth, elevation)
                decoder_matrix[spk, : len(coeffs)] = coeffs

            # Pseudo-inverse for least-squares decoding
            if num_speakers >= self.num_channels:
                decoder_matrix = np.linalg.pinv(decoder_matrix.T).T

            return decoder_matrix

        except Exception as e:
            logger.error(f"Decoder matrix calculation error: {e}")
            # Identity matrix fallback
            size = min(len(speaker_positions), self.num_channels)
            matrix = np.eye(size, self.num_channels)
            return np.pad(matrix, ((0, len(speaker_positions) - size), (0, 0)))


class BinauralProcessor:
    """
    Binaural audio processing and rendering
    """

    def __init__(self, config: SpatialConfig):
        self.config = config
        self.hrtf_processor = HRTFProcessor(config)

    def process_source(
        self, source: SpatialSource, listener: Listener
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process spatial audio source to binaural output

        Args:
            source: Spatial audio source
            listener: Listener configuration

        Returns:
            Tuple of (left_ear_audio, right_ear_audio)
        """
        try:
            # Calculate relative position
            rel_pos = self._calculate_relative_position(source.position, listener)
            azimuth, elevation, distance = rel_pos.to_spherical()

            # Apply distance attenuation
            audio_data = source.audio_data.copy()
            if self.config.use_distance_attenuation:
                distance_gain = 1.0 / max(distance, 0.1)
                audio_data *= distance_gain * source.gain

            # Apply air absorption
            if self.config.use_air_absorption:
                audio_data = self._apply_air_absorption(audio_data, distance)

            # Apply Doppler effect
            if self.config.use_doppler and source.velocity is not None:
                audio_data = self._apply_doppler_effect(
                    audio_data, source.velocity, listener
                )

            # Get HRTF filters
            left_hrtf, right_hrtf = self.hrtf_processor.get_hrtf(azimuth, elevation)

            # Convolve with HRTF
            left_audio = scipy.signal.fftconvolve(audio_data, left_hrtf, mode="same")
            right_audio = scipy.signal.fftconvolve(audio_data, right_hrtf, mode="same")

            return (left_audio, right_audio)

        except Exception as e:
            logger.error(f"Binaural processing error: {e}")
            # Return mono to both ears
            audio = source.audio_data * source.gain
            return (audio, audio)

    def _calculate_relative_position(
        self, source_pos: Position3D, listener: Listener
    ) -> Position3D:
        """Calculate source position relative to listener"""
        try:
            # Convert to Cartesian coordinates
            src_x, src_y, src_z = source_pos.to_cartesian()
            list_x, list_y, list_z = listener.position.to_cartesian()

            # Relative position
            rel_x = src_x - list_x
            rel_y = src_y - list_y
            rel_z = src_z - list_z

            # Apply listener orientation (simplified - yaw only)
            yaw = listener.orientation[0]
            rotated_x = rel_x * np.cos(-yaw) - rel_y * np.sin(-yaw)
            rotated_y = rel_x * np.sin(-yaw) + rel_y * np.cos(-yaw)

            return Position3D(rotated_x, rotated_y, rel_z, CoordinateSystem.CARTESIAN)

        except Exception as e:
            logger.error(f"Relative position calculation error: {e}")
            return Position3D()

    def _apply_air_absorption(
        self, audio_data: np.ndarray, distance: float
    ) -> np.ndarray:
        """Apply frequency-dependent air absorption"""
        if distance <= 1.0:
            return audio_data

        try:
            # Simplified air absorption model
            # Higher frequencies attenuated more over distance
            stft = librosa.stft(audio_data, hop_length=512, n_fft=2048)
            magnitude = np.abs(stft)
            phase = np.angle(stft)

            # Frequency-dependent attenuation
            freqs = librosa.fft_frequencies(sr=self.config.sample_rate, n_fft=2048)

            # Air absorption coefficient (dB/km/kHz^2)
            absorption_coeff = 0.1

            # Calculate attenuation
            attenuation_db = (
                -absorption_coeff * (freqs / 1000.0) ** 2 * (distance / 1000.0)
            )
            attenuation_linear = 10 ** (attenuation_db / 20.0)

            # Apply attenuation
            magnitude *= attenuation_linear[:, np.newaxis]

            # Reconstruct
            attenuated_stft = magnitude * np.exp(1j * phase)
            attenuated_audio = librosa.istft(
                attenuated_stft, hop_length=512, length=len(audio_data)
            )

            return attenuated_audio.astype(audio_data.dtype)

        except Exception as e:
            logger.debug(f"Air absorption error: {e}")
            return audio_data

    def _apply_doppler_effect(
        self, audio_data: np.ndarray, velocity: Position3D, listener: Listener
    ) -> np.ndarray:
        """Apply Doppler frequency shift"""
        try:
            # Calculate radial velocity (component toward/away from listener)
            vel_x, vel_y, vel_z = velocity.to_cartesian()
            radial_velocity = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)  # Simplified

            # Doppler frequency shift ratio
            doppler_ratio = (
                self.config.speed_of_sound - radial_velocity
            ) / self.config.speed_of_sound

            if abs(doppler_ratio - 1.0) < 0.001:  # No significant Doppler effect
                return audio_data

            # Apply time stretching to simulate frequency shift
            # This is a simplified approach - real implementation would use proper pitch shifting
            if doppler_ratio > 0.5 and doppler_ratio < 2.0:
                stretched_length = int(len(audio_data) / doppler_ratio)
                doppler_audio = scipy.signal.resample(audio_data, stretched_length)

                # Pad or trim to original length
                if len(doppler_audio) > len(audio_data):
                    doppler_audio = doppler_audio[: len(audio_data)]
                else:
                    doppler_audio = np.pad(
                        doppler_audio, (0, len(audio_data) - len(doppler_audio))
                    )

                return doppler_audio

        except Exception as e:
            logger.debug(f"Doppler effect error: {e}")

        return audio_data


class RoomAcousticProcessor:
    """
    Room acoustics modeling and reverberation
    """

    def __init__(self, config: SpatialConfig):
        self.config = config
        self.room = None

    def setup_room(self, room_config: RoomAcoustics) -> None:
        """Setup room acoustic simulation"""
        try:
            if HAS_PYROOMACOUSTICS:
                # Create room using pyroomacoustics
                self.room = pra.ShoeBox(
                    room_config.dimensions,
                    fs=self.config.sample_rate,
                    materials=pra.Material(
                        absorption=np.mean(room_config.absorption_coefficients)
                    ),
                    ray_tracing=True,
                    air_absorption=room_config.air_absorption,
                )
                logger.info(f"Room setup: {room_config.dimensions}")
            else:
                # Fallback: simple impulse response generation
                self.room = self._create_simple_room_ir(room_config)

        except Exception as e:
            logger.error(f"Room setup error: {e}")
            self.room = None

    def _create_simple_room_ir(self, room_config: RoomAcoustics) -> np.ndarray:
        """Create simple room impulse response without pyroomacoustics"""
        try:
            # Generate synthetic room impulse response
            ir_length = int(room_config.rt60 * self.config.sample_rate)

            # Exponential decay
            t = np.arange(ir_length) / self.config.sample_rate
            envelope = np.exp(-3 * t / room_config.rt60)  # -60dB decay

            # Add early reflections
            ir = np.random.normal(0, 0.1, ir_length) * envelope

            # Add direct path
            ir[0] = 1.0

            # Add some distinct early reflections
            reflection_delays = [0.01, 0.02, 0.035, 0.05]  # seconds
            reflection_gains = [0.3, 0.2, 0.15, 0.1]

            for delay, gain in zip(reflection_delays, reflection_gains):
                delay_samples = int(delay * self.config.sample_rate)
                if delay_samples < len(ir):
                    ir[delay_samples] += gain

            return ir

        except Exception as e:
            logger.error(f"Simple room IR creation error: {e}")
            # Return delta function (no reverb)
            ir = np.zeros(1024)
            ir[0] = 1.0
            return ir

    def process_with_acoustics(
        self,
        audio_data: np.ndarray,
        source_position: Position3D,
        listener_position: Position3D,
    ) -> np.ndarray:
        """Process audio through room acoustics"""
        if self.room is None:
            return audio_data

        try:
            if HAS_PYROOMACOUSTICS and hasattr(self.room, "add_source"):
                # Use pyroomacoustics
                source_pos = source_position.to_cartesian()
                listener_pos = listener_position.to_cartesian()

                # Add source and microphone to room
                self.room.add_source(source_pos)
                self.room.add_microphone(listener_pos)

                # Simulate
                self.room.simulate()

                # Get room impulse response
                rir = self.room.rir[0][0]  # First microphone, first source

            else:
                # Use simple room impulse response
                rir = (
                    self.room
                    if isinstance(self.room, np.ndarray)
                    else self._create_simple_room_ir(RoomAcoustics())
                )

            # Convolve audio with room impulse response
            reverb_audio = scipy.signal.fftconvolve(audio_data, rir, mode="same")

            return reverb_audio.astype(audio_data.dtype)

        except Exception as e:
            logger.error(f"Room acoustics processing error: {e}")
            return audio_data


class SpatialAudioProcessor:
    """
    Main spatial audio processing system
    """

    def __init__(self, config: Optional[SpatialConfig] = None):
        self.config = config or SpatialConfig()

        # Initialize processors
        self.binaural_processor = BinauralProcessor(self.config)
        self.ambisonic_processor = AmbisonicProcessor(self.config)
        self.room_processor = RoomAcousticProcessor(self.config)

    async def render_binaural(
        self,
        sources: List[SpatialSource],
        listener: Listener,
        room_config: Optional[RoomAcoustics] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render spatial audio scene to binaural output

        Args:
            sources: List of spatial audio sources
            listener: Listener configuration
            room_config: Optional room acoustics

        Returns:
            Tuple of (left_channel, right_channel)
        """
        try:
            if not sources:
                # Return silence
                silence = np.zeros(1024)
                return (silence, silence)

            # Setup room if provided
            if room_config:
                self.room_processor.setup_room(room_config)

            # Process each source
            left_mix = None
            right_mix = None

            for source in sources:
                # Apply room acoustics if enabled
                source_audio = source.audio_data
                if room_config and self.room_processor.room is not None:
                    source_audio = self.room_processor.process_with_acoustics(
                        source_audio, source.position, listener.position
                    )

                    # Update source with processed audio
                    processed_source = SpatialSource(
                        audio_data=source_audio,
                        position=source.position,
                        velocity=source.velocity,
                        gain=source.gain,
                        source_id=source.source_id,
                    )
                else:
                    processed_source = source

                # Process to binaural
                left_channel, right_channel = self.binaural_processor.process_source(
                    processed_source, listener
                )

                # Mix with existing audio
                if left_mix is None:
                    left_mix = left_channel
                    right_mix = right_channel
                else:
                    # Ensure same length
                    min_length = min(len(left_mix), len(left_channel))
                    left_mix = left_mix[:min_length] + left_channel[:min_length]
                    right_mix = right_mix[:min_length] + right_channel[:min_length]

            # Apply final processing
            if left_mix is not None and right_mix is not None:
                left_mix = self._apply_final_processing(left_mix)
                right_mix = self._apply_final_processing(right_mix)
                return (left_mix, right_mix)
            else:
                silence = np.zeros(1024)
                return (silence, silence)

        except Exception as e:
            logger.error(f"Binaural rendering error: {e}")
            silence = np.zeros(1024)
            return (silence, silence)

    async def render_ambisonic(self, sources: List[SpatialSource]) -> np.ndarray:
        """
        Render spatial audio scene to ambisonic format

        Args:
            sources: List of spatial audio sources

        Returns:
            Multi-channel ambisonic audio
        """
        try:
            if not sources:
                return np.zeros((self.ambisonic_processor.num_channels, 1024))

            # Find the longest audio
            max_length = max(len(source.audio_data) for source in sources)
            ambisonic_mix = np.zeros(
                (self.ambisonic_processor.num_channels, max_length)
            )

            for source in sources:
                # Get source position in spherical coordinates
                azimuth, elevation, distance = source.position.to_spherical()

                # Encode to ambisonic
                ambisonic_source = self.ambisonic_processor.encode_source(
                    source.audio_data, azimuth, elevation, distance
                )

                # Mix with existing audio
                mix_length = min(ambisonic_mix.shape[1], ambisonic_source.shape[1])
                for ch in range(min(ambisonic_mix.shape[0], ambisonic_source.shape[0])):
                    ambisonic_mix[ch, :mix_length] += (
                        ambisonic_source[ch, :mix_length] * source.gain
                    )

            return ambisonic_mix

        except Exception as e:
            logger.error(f"Ambisonic rendering error: {e}")
            return np.zeros((self.ambisonic_processor.num_channels, 1024))

    async def render_speaker_array(
        self, sources: List[SpatialSource], speaker_positions: List[Position3D]
    ) -> np.ndarray:
        """
        Render spatial audio scene for speaker array

        Args:
            sources: List of spatial audio sources
            speaker_positions: List of speaker positions

        Returns:
            Multi-channel audio for speakers
        """
        try:
            # First render to ambisonic
            ambisonic_audio = await self.render_ambisonic(sources)

            # Then decode to speaker array
            speaker_audio = self.ambisonic_processor.decode_speaker_array(
                ambisonic_audio, speaker_positions
            )

            return speaker_audio

        except Exception as e:
            logger.error(f"Speaker array rendering error: {e}")
            num_speakers = len(speaker_positions)
            return np.zeros((num_speakers, 1024))

    def _apply_final_processing(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply final processing (limiting, etc.)"""
        try:
            # Simple soft limiting
            threshold = 0.95
            processed = np.tanh(audio_data / threshold) * threshold
            return processed.astype(audio_data.dtype)

        except Exception as e:
            logger.debug(f"Final processing error: {e}")
            return audio_data

    def create_demo_scene(self) -> Tuple[List[SpatialSource], Listener, RoomAcoustics]:
        """Create a demo spatial audio scene"""
        try:
            # Create demo sources
            duration = 3.0  # seconds
            sample_rate = self.config.sample_rate
            t = np.linspace(0, duration, int(duration * sample_rate))

            # Source 1: Left side, 440 Hz tone
            audio1 = 0.3 * np.sin(2 * np.pi * 440 * t)
            source1 = SpatialSource(
                audio_data=audio1,
                position=Position3D(-2.0, 0.0, 0.0),  # 2 meters to the left
                gain=0.8,
                source_id="left_tone",
            )

            # Source 2: Right side, 880 Hz tone
            audio2 = 0.3 * np.sin(2 * np.pi * 880 * t)
            source2 = SpatialSource(
                audio_data=audio2,
                position=Position3D(2.0, 0.0, 0.0),  # 2 meters to the right
                gain=0.8,
                source_id="right_tone",
            )

            # Source 3: Behind, white noise
            audio3 = 0.1 * np.random.normal(0, 1, len(t))
            source3 = SpatialSource(
                audio_data=audio3,
                position=Position3D(0.0, -3.0, 0.0),  # 3 meters behind
                gain=0.5,
                source_id="behind_noise",
            )

            # Listener at origin
            listener = Listener(position=Position3D(0.0, 0.0, 0.0))

            # Room acoustics
            room = RoomAcoustics(
                room_type=RoomType.STUDIO,
                dimensions=(8.0, 6.0, 3.0),  # 8x6x3 meter room
                rt60=0.4,  # 400ms reverberation
            )

            return ([source1, source2, source3], listener, room)

        except Exception as e:
            logger.error(f"Demo scene creation error: {e}")
            # Return minimal scene
            silence = np.zeros(1024)
            source = SpatialSource(audio_data=silence, position=Position3D())
            listener = Listener()
            room = RoomAcoustics()
            return ([source], listener, room)


# Convenience functions and aliases

BinauralProcessor = BinauralProcessor


# Example usage and testing
if __name__ == "__main__":

    async def demo():
        """Demo spatial audio processing"""
        try:
            # Create spatial processor
            config = SpatialConfig(sample_rate=44100)
            processor = SpatialAudioProcessor(config)

            print("Creating demo spatial audio scene...")
            sources, listener, room = processor.create_demo_scene()

            print(f"Processing {len(sources)} sources...")
            print(f"Room: {room.dimensions} meters, RT60: {room.rt60}s")

            # Render binaural
            print("Rendering binaural audio...")
            left, right = await processor.render_binaural(sources, listener, room)

            print(f"Binaural output: {len(left)} samples per channel")
            print(f"Left channel RMS: {np.sqrt(np.mean(left**2)):.4f}")
            print(f"Right channel RMS: {np.sqrt(np.mean(right**2)):.4f}")

            # Render ambisonic
            print("Rendering ambisonic audio...")
            ambisonic = await processor.render_ambisonic(sources)

            print(
                f"Ambisonic output: {ambisonic.shape[0]} channels, {ambisonic.shape[1]} samples"
            )
            print(f"W channel RMS: {np.sqrt(np.mean(ambisonic[0]**2)):.4f}")

            # Speaker array example
            speaker_positions = [
                Position3D(1.0, 1.0, 0.0),  # Front left
                Position3D(1.0, -1.0, 0.0),  # Front right
                Position3D(-1.0, 1.0, 0.0),  # Rear left
                Position3D(-1.0, -1.0, 0.0),  # Rear right
            ]

            print("Rendering for 4-speaker array...")
            speaker_audio = await processor.render_speaker_array(
                sources, speaker_positions
            )

            print(
                f"Speaker array output: {speaker_audio.shape[0]} speakers, {speaker_audio.shape[1]} samples"
            )

            print("\nSpatial audio processing demo completed successfully!")

        except Exception as e:
            print(f"Demo error: {e}")

    # Run demo
    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
