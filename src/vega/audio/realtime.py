"""
Vega 2.0 Real-time Audio Analysis Module

This module provides live audio stream processing capabilities including:
- Voice Activity Detection (VAD)
- Real-time noise reduction
- Acoustic fingerprinting
- Live audio feature extraction
- Stream processing pipeline

Dependencies:
- numpy: Audio array processing
- scipy: Signal processing algorithms
- librosa: Audio analysis and feature extraction
- pyaudio: Real-time audio I/O
- webrtcvad: Google's WebRTC VAD
"""

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import librosa
    import librosa.display
    import scipy.signal
    import scipy.fftpack

    HAS_AUDIO_LIBS = True
except ImportError:
    librosa = None
    scipy = None
    HAS_AUDIO_LIBS = False

try:
    import pyaudio

    HAS_PYAUDIO = True
except ImportError:
    pyaudio = None
    HAS_PYAUDIO = False

try:
    import webrtcvad

    HAS_WEBRTC_VAD = True
except ImportError:
    webrtcvad = None
    HAS_WEBRTC_VAD = False

logger = logging.getLogger(__name__)


class AudioProcessingError(Exception):
    """Custom exception for audio processing errors"""

    pass


class VADMode(Enum):
    """Voice Activity Detection sensitivity modes"""

    QUALITY = 0  # Most sensitive, best quality
    LOW_BITRATE = 1  # Less sensitive
    AGGRESSIVE = 2  # Least sensitive
    VERY_AGGRESSIVE = 3  # Most aggressive filtering


class NoiseReductionMethod(Enum):
    """Noise reduction algorithm types"""

    SPECTRAL_SUBTRACTION = "spectral_subtraction"
    WIENER_FILTER = "wiener_filter"
    ADAPTIVE_FILTER = "adaptive_filter"
    STATIONARY_WAVELET = "stationary_wavelet"


@dataclass
class AudioConfig:
    """Configuration for real-time audio processing"""

    sample_rate: int = 16000
    chunk_size: int = 1024
    channels: int = 1
    format: int = None  # Will be set to pyaudio.paInt16 if available
    vad_mode: VADMode = VADMode.QUALITY
    noise_reduction: bool = True
    noise_method: NoiseReductionMethod = NoiseReductionMethod.SPECTRAL_SUBTRACTION
    frame_duration_ms: int = 30  # 10, 20, or 30ms for WebRTC VAD
    silence_threshold: float = 0.01
    min_speech_duration: float = 0.5  # seconds
    max_silence_duration: float = 2.0  # seconds


@dataclass
class AudioFrame:
    """Container for processed audio frame data"""

    data: np.ndarray
    timestamp: float
    sample_rate: int
    is_speech: bool = False
    confidence: float = 0.0
    features: Dict[str, Any] = field(default_factory=dict)
    fingerprint: Optional[str] = None


class VoiceActivityDetector:
    """
    Advanced Voice Activity Detection using multiple algorithms
    """

    def __init__(self, config: AudioConfig):
        self.config = config
        self.vad = None

        if HAS_WEBRTC_VAD:
            self.vad = webrtcvad.Vad(config.vad_mode.value)

        # Initialize buffers for smoothing
        self.speech_buffer = deque(maxlen=10)
        self.energy_history = deque(maxlen=50)

    def is_speech(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """
        Determine if audio contains speech using multiple detection methods

        Args:
            audio_data: Audio samples as numpy array

        Returns:
            Tuple of (is_speech_detected, confidence_score)
        """
        if not HAS_AUDIO_LIBS:
            return False, 0.0

        confidence = 0.0
        speech_detected = False

        # Method 1: WebRTC VAD (if available)
        if self.vad and len(audio_data) > 0:
            try:
                # Convert float to int16 for WebRTC VAD
                audio_int16 = (audio_data * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
                webrtc_result = self.vad.is_speech(audio_bytes, self.config.sample_rate)
                if webrtc_result:
                    confidence += 0.4
                    speech_detected = True
            except Exception as e:
                logger.warning(f"WebRTC VAD error: {e}")

        # Method 2: Energy-based detection
        energy = np.sum(audio_data**2) / len(audio_data)
        self.energy_history.append(energy)

        if len(self.energy_history) > 10:
            avg_energy = np.mean(list(self.energy_history))
            if energy > avg_energy * 2 and energy > self.config.silence_threshold:
                confidence += 0.3
                speech_detected = True

        # Method 3: Zero crossing rate
        zcr = np.sum(np.abs(np.diff(np.sign(audio_data)))) / (2 * len(audio_data))
        if 0.01 < zcr < 0.3:  # Typical speech range
            confidence += 0.2

        # Method 4: Spectral features (if librosa available)
        if HAS_AUDIO_LIBS:
            try:
                # Spectral centroid
                spectral_centroid = librosa.feature.spectral_centroid(
                    y=audio_data, sr=self.config.sample_rate
                )[0]
                if np.mean(spectral_centroid) > 1000:  # Typical speech range
                    confidence += 0.1
            except Exception as e:
                logger.debug(f"Spectral analysis error: {e}")

        # Smooth the decision using history
        self.speech_buffer.append(speech_detected)
        smoothed_decision = sum(self.speech_buffer) > len(self.speech_buffer) // 2

        return smoothed_decision, min(confidence, 1.0)


class NoiseReducer:
    """
    Multi-algorithm noise reduction for real-time audio
    """

    def __init__(self, config: AudioConfig):
        self.config = config
        self.noise_profile = None
        self.adaptation_rate = 0.1
        self.wiener_coeffs = None

    def estimate_noise_profile(self, audio_data: np.ndarray) -> None:
        """
        Estimate noise characteristics from silent/non-speech audio
        """
        if not HAS_AUDIO_LIBS:
            return

        try:
            # Compute power spectral density
            freqs, psd = scipy.signal.welch(audio_data, self.config.sample_rate)

            if self.noise_profile is None:
                self.noise_profile = psd
            else:
                # Adaptive noise profile update
                self.noise_profile = (
                    1 - self.adaptation_rate
                ) * self.noise_profile + self.adaptation_rate * psd
        except Exception as e:
            logger.error(f"Noise profile estimation error: {e}")

    def spectral_subtraction(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Reduce noise using spectral subtraction method
        """
        if not HAS_AUDIO_LIBS or self.noise_profile is None:
            return audio_data

        try:
            # Compute STFT
            stft = librosa.stft(audio_data)
            magnitude = np.abs(stft)
            phase = np.angle(stft)

            # Estimate noise power spectrum
            noise_magnitude = np.sqrt(self.noise_profile)
            noise_magnitude = np.expand_dims(
                noise_magnitude[: magnitude.shape[0]], axis=1
            )

            # Spectral subtraction with over-subtraction factor
            alpha = 2.0  # Over-subtraction factor
            enhanced_magnitude = magnitude - alpha * noise_magnitude

            # Half-wave rectification with spectral floor
            beta = 0.1  # Spectral floor factor
            enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)

            # Reconstruct signal
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, length=len(audio_data))

            return enhanced_audio.astype(audio_data.dtype)

        except Exception as e:
            logger.error(f"Spectral subtraction error: {e}")
            return audio_data

    def wiener_filter(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply Wiener filtering for noise reduction
        """
        if not HAS_AUDIO_LIBS or self.noise_profile is None:
            return audio_data

        try:
            # Compute power spectral density of signal
            freqs, signal_psd = scipy.signal.welch(audio_data, self.config.sample_rate)

            # Wiener filter coefficients
            wiener_gain = signal_psd / (signal_psd + self.noise_profile)

            # Apply in frequency domain
            audio_fft = scipy.fftpack.fft(audio_data)
            enhanced_fft = audio_fft * wiener_gain[: len(audio_fft)]
            enhanced_audio = scipy.fftpack.ifft(enhanced_fft).real

            return enhanced_audio.astype(audio_data.dtype)

        except Exception as e:
            logger.error(f"Wiener filter error: {e}")
            return audio_data

    def reduce_noise(
        self, audio_data: np.ndarray, is_speech: bool = True
    ) -> np.ndarray:
        """
        Apply noise reduction using configured method
        """
        if not is_speech and self.config.noise_reduction:
            # Update noise profile during silence
            self.estimate_noise_profile(audio_data)

        if not self.config.noise_reduction or self.noise_profile is None:
            return audio_data

        method = self.config.noise_method

        if method == NoiseReductionMethod.SPECTRAL_SUBTRACTION:
            return self.spectral_subtraction(audio_data)
        elif method == NoiseReductionMethod.WIENER_FILTER:
            return self.wiener_filter(audio_data)
        else:
            # Fallback to simple high-pass filter
            try:
                if HAS_AUDIO_LIBS:
                    sos = scipy.signal.butter(
                        4, 300, btype="high", fs=self.config.sample_rate, output="sos"
                    )
                    return scipy.signal.sosfilt(sos, audio_data)
            except Exception as e:
                logger.error(f"Simple filter error: {e}")

        return audio_data


class AcousticFingerprinter:
    """
    Generate acoustic fingerprints for audio identification
    """

    def __init__(self, config: AudioConfig):
        self.config = config

    def extract_fingerprint(self, audio_data: np.ndarray) -> Optional[str]:
        """
        Extract acoustic fingerprint from audio data
        """
        if not HAS_AUDIO_LIBS or len(audio_data) == 0:
            return None

        try:
            # Extract spectral features
            mfccs = librosa.feature.mfcc(
                y=audio_data, sr=self.config.sample_rate, n_mfcc=13
            )
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_data, sr=self.config.sample_rate
            )
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=self.config.sample_rate
            )
            chroma = librosa.feature.chroma_stft(
                y=audio_data, sr=self.config.sample_rate
            )

            # Combine features
            features = np.concatenate(
                [
                    np.mean(mfccs, axis=1),
                    [np.mean(spectral_centroid)],
                    [np.mean(spectral_rolloff)],
                    np.mean(chroma, axis=1),
                ]
            )

            # Create simple hash-based fingerprint
            fingerprint_hash = hash(tuple(features.round(3)))
            return f"fp_{abs(fingerprint_hash):016x}"

        except Exception as e:
            logger.error(f"Fingerprint extraction error: {e}")
            return None


class RealtimeAudioProcessor:
    """
    Main real-time audio processing pipeline
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()

        if HAS_PYAUDIO:
            self.config.format = self.config.format or pyaudio.paInt16

        # Initialize components
        self.vad = VoiceActivityDetector(self.config)
        self.noise_reducer = NoiseReducer(self.config)
        self.fingerprinter = AcousticFingerprinter(self.config)

        # Processing state
        self.is_recording = False
        self.audio_stream = None
        self.processing_thread = None
        self.frame_queue = asyncio.Queue()
        self.callbacks: List[Callable[[AudioFrame], None]] = []

        # Speech detection state
        self.speech_start_time = None
        self.silence_start_time = None
        self.current_speech_buffer = []

    def add_callback(self, callback: Callable[[AudioFrame], None]) -> None:
        """Add callback function to receive processed audio frames"""
        self.callbacks.append(callback)

    def remove_callback(self, callback: Callable[[AudioFrame], None]) -> None:
        """Remove callback function"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    async def start_processing(self) -> None:
        """
        Start real-time audio processing pipeline
        """
        if not HAS_PYAUDIO:
            raise AudioProcessingError("PyAudio not available for real-time processing")

        if self.is_recording:
            return

        try:
            # Initialize PyAudio
            p = pyaudio.PyAudio()

            # Open audio stream
            self.audio_stream = p.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size,
                stream_callback=self._audio_callback,
            )

            self.is_recording = True
            self.audio_stream.start_stream()

            logger.info("Real-time audio processing started")

            # Start processing loop
            await self._processing_loop()

        except Exception as e:
            logger.error(f"Failed to start audio processing: {e}")
            raise AudioProcessingError(f"Audio processing startup failed: {e}")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio stream callback"""
        try:
            # Convert bytes to numpy array
            audio_data = (
                np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            # Create frame object
            frame = AudioFrame(
                data=audio_data,
                timestamp=time.time(),
                sample_rate=self.config.sample_rate,
            )

            # Add to processing queue (non-blocking)
            try:
                self.frame_queue.put_nowait(frame)
            except asyncio.QueueFull:
                logger.warning("Audio processing queue full, dropping frame")

        except Exception as e:
            logger.error(f"Audio callback error: {e}")

        return (None, pyaudio.paContinue)

    async def _processing_loop(self):
        """Main audio processing loop"""
        while self.is_recording:
            try:
                # Get frame from queue with timeout
                frame = await asyncio.wait_for(self.frame_queue.get(), timeout=1.0)

                # Process frame
                processed_frame = await self._process_frame(frame)

                # Send to callbacks
                for callback in self.callbacks:
                    try:
                        callback(processed_frame)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Processing loop error: {e}")

    async def _process_frame(self, frame: AudioFrame) -> AudioFrame:
        """
        Process individual audio frame
        """
        try:
            # Voice Activity Detection
            is_speech, confidence = self.vad.is_speech(frame.data)
            frame.is_speech = is_speech
            frame.confidence = confidence

            # Noise reduction
            if self.config.noise_reduction:
                frame.data = self.noise_reducer.reduce_noise(frame.data, is_speech)

            # Extract audio features
            if HAS_AUDIO_LIBS:
                try:
                    # Basic features
                    frame.features["energy"] = float(np.sum(frame.data**2))
                    frame.features["zcr"] = float(
                        np.sum(np.abs(np.diff(np.sign(frame.data))))
                    )
                    frame.features["rms"] = float(np.sqrt(np.mean(frame.data**2)))

                    # Spectral features (if frame is long enough)
                    if len(frame.data) >= 512:
                        centroid = librosa.feature.spectral_centroid(
                            y=frame.data, sr=frame.sample_rate
                        )
                        frame.features["spectral_centroid"] = float(np.mean(centroid))

                        rolloff = librosa.feature.spectral_rolloff(
                            y=frame.data, sr=frame.sample_rate
                        )
                        frame.features["spectral_rolloff"] = float(np.mean(rolloff))

                except Exception as e:
                    logger.debug(f"Feature extraction error: {e}")

            # Speech boundary detection
            self._update_speech_boundaries(frame)

            # Generate fingerprint for speech segments
            if is_speech and len(self.current_speech_buffer) > 0:
                # Combine recent speech frames
                speech_audio = np.concatenate(
                    [f.data for f in self.current_speech_buffer[-10:]]
                )
                frame.fingerprint = self.fingerprinter.extract_fingerprint(speech_audio)

            return frame

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return frame

    def _update_speech_boundaries(self, frame: AudioFrame):
        """
        Update speech/silence boundary detection
        """
        current_time = frame.timestamp

        if frame.is_speech:
            if self.speech_start_time is None:
                self.speech_start_time = current_time
                logger.debug("Speech started")

            self.silence_start_time = None
            self.current_speech_buffer.append(frame)

            # Limit buffer size
            if len(self.current_speech_buffer) > 100:
                self.current_speech_buffer = self.current_speech_buffer[-50:]

        else:  # Silence
            if self.silence_start_time is None:
                self.silence_start_time = current_time

            # Check for end of speech
            if (
                self.speech_start_time is not None
                and current_time - self.silence_start_time
                > self.config.max_silence_duration
            ):

                speech_duration = self.silence_start_time - self.speech_start_time
                if speech_duration >= self.config.min_speech_duration:
                    logger.debug(f"Speech ended, duration: {speech_duration:.2f}s")

                # Reset speech detection
                self.speech_start_time = None
                self.current_speech_buffer = []

    async def stop_processing(self):
        """
        Stop real-time audio processing
        """
        if not self.is_recording:
            return

        self.is_recording = False

        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None

        logger.info("Real-time audio processing stopped")

    async def process_audio_file(self, file_path: str) -> List[AudioFrame]:
        """
        Process audio file and return frames
        """
        if not HAS_AUDIO_LIBS:
            raise AudioProcessingError("Librosa not available for file processing")

        try:
            # Load audio file
            audio_data, sr = librosa.load(
                file_path, sr=self.config.sample_rate, mono=True
            )

            # Process in chunks
            frames = []
            chunk_samples = self.config.chunk_size

            for i in range(0, len(audio_data), chunk_samples):
                chunk = audio_data[i : i + chunk_samples]

                if len(chunk) < chunk_samples:
                    # Pad last chunk
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

                frame = AudioFrame(data=chunk, timestamp=i / sr, sample_rate=sr)

                processed_frame = await self._process_frame(frame)
                frames.append(processed_frame)

            return frames

        except Exception as e:
            logger.error(f"File processing error: {e}")
            raise AudioProcessingError(f"Failed to process audio file: {e}")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current processing status
        """
        return {
            "is_recording": self.is_recording,
            "sample_rate": self.config.sample_rate,
            "chunk_size": self.config.chunk_size,
            "vad_mode": self.config.vad_mode.name,
            "noise_reduction": self.config.noise_reduction,
            "has_pyaudio": HAS_PYAUDIO,
            "has_audio_libs": HAS_AUDIO_LIBS,
            "has_webrtc_vad": HAS_WEBRTC_VAD,
            "speech_active": self.speech_start_time is not None,
            "queue_size": self.frame_queue.qsize() if self.frame_queue else 0,
        }


# Convenience functions for easy usage


async def create_realtime_processor(
    sample_rate: int = 16000,
    vad_mode: VADMode = VADMode.QUALITY,
    noise_reduction: bool = True,
) -> RealtimeAudioProcessor:
    """
    Create and configure a real-time audio processor
    """
    config = AudioConfig(
        sample_rate=sample_rate, vad_mode=vad_mode, noise_reduction=noise_reduction
    )

    return RealtimeAudioProcessor(config)


async def process_audio_stream(
    processor: RealtimeAudioProcessor,
    duration: float,
    callback: Optional[Callable[[AudioFrame], None]] = None,
) -> List[AudioFrame]:
    """
    Process audio stream for specified duration
    """
    frames = []

    def frame_collector(frame: AudioFrame):
        frames.append(frame)
        if callback:
            callback(frame)

    processor.add_callback(frame_collector)

    try:
        await processor.start_processing()
        await asyncio.sleep(duration)
    finally:
        await processor.stop_processing()
        processor.remove_callback(frame_collector)

    return frames


# Example usage and testing
if __name__ == "__main__":

    async def demo_callback(frame: AudioFrame):
        """Example callback for processing audio frames"""
        status = "SPEECH" if frame.is_speech else "SILENCE"
        energy = frame.features.get("energy", 0)
        print(
            f"[{frame.timestamp:.2f}s] {status} (confidence: {frame.confidence:.2f}, energy: {energy:.4f})"
        )

        if frame.fingerprint:
            print(f"  Fingerprint: {frame.fingerprint}")

    async def main():
        """Demo real-time audio processing"""
        try:
            # Create processor
            processor = await create_realtime_processor()

            print("Starting real-time audio processing demo...")
            print("Speak into the microphone. Press Ctrl+C to stop.")
            print(f"Status: {processor.get_status()}")

            # Process for 10 seconds
            frames = await process_audio_stream(processor, 10.0, demo_callback)

            print(f"\nProcessed {len(frames)} audio frames")
            speech_frames = [f for f in frames if f.is_speech]
            print(
                f"Detected speech in {len(speech_frames)} frames ({len(speech_frames)/len(frames)*100:.1f}%)"
            )

        except KeyboardInterrupt:
            print("\nDemo stopped by user")
        except Exception as e:
            print(f"Demo error: {e}")

    # Run demo
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
