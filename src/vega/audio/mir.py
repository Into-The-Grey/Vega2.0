"""
Vega 2.0 Music Information Retrieval (MIR) Module

This module provides comprehensive music analysis capabilities including:
- Beat tracking and tempo detection
- Chord detection and harmonic analysis
- Genre classification using machine learning
- Mood and emotion analysis
- Key signature and musical structure analysis
- Audio similarity and music recommendation

Dependencies:
- librosa: Music and audio analysis
- numpy: Numerical computations
- scipy: Signal processing
- scikit-learn: Machine learning for classification
- essentia: Advanced music analysis (optional)
"""

import asyncio
import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import librosa
    import librosa.display
    import scipy.signal
    import scipy.stats
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    HAS_MIR_LIBS = True
except ImportError:
    librosa = None
    scipy = None
    RandomForestClassifier = None
    StandardScaler = None
    HAS_MIR_LIBS = False

try:
    import essentia.standard as es

    HAS_ESSENTIA = True
except ImportError:
    es = None
    HAS_ESSENTIA = False

logger = logging.getLogger(__name__)


class MusicProcessingError(Exception):
    """Custom exception for music processing errors"""

    pass


class ChordType(Enum):
    """Common chord types"""

    MAJOR = "major"
    MINOR = "minor"
    DIMINISHED = "diminished"
    AUGMENTED = "augmented"
    MAJOR_7 = "major7"
    MINOR_7 = "minor7"
    DOMINANT_7 = "dominant7"
    SUSPENDED_2 = "sus2"
    SUSPENDED_4 = "sus4"
    NO_CHORD = "N"


class Genre(Enum):
    """Music genre classifications"""

    CLASSICAL = "classical"
    JAZZ = "jazz"
    ROCK = "rock"
    POP = "pop"
    ELECTRONIC = "electronic"
    HIP_HOP = "hip_hop"
    COUNTRY = "country"
    BLUES = "blues"
    REGGAE = "reggae"
    FOLK = "folk"
    METAL = "metal"
    PUNK = "punk"
    UNKNOWN = "unknown"


class MoodCategory(Enum):
    """Emotional mood categories"""

    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    CALM = "calm"
    ENERGETIC = "energetic"
    ROMANTIC = "romantic"
    MYSTERIOUS = "mysterious"
    TRIUMPHANT = "triumphant"
    MELANCHOLIC = "melancholic"
    NEUTRAL = "neutral"


@dataclass
class BeatInfo:
    """Container for beat tracking information"""

    tempo: float
    beats: np.ndarray
    beat_times: np.ndarray
    downbeats: Optional[np.ndarray] = None
    time_signature: Optional[Tuple[int, int]] = None
    rhythm_stability: float = 0.0
    tempo_confidence: float = 0.0


@dataclass
class ChordProgression:
    """Container for chord analysis results"""

    chords: List[str]
    chord_times: np.ndarray
    chord_confidence: np.ndarray
    key_signature: Optional[str] = None
    mode: Optional[str] = None  # major/minor
    harmonic_complexity: float = 0.0


@dataclass
class GenreClassification:
    """Container for genre classification results"""

    primary_genre: Genre
    confidence: float
    genre_probabilities: Dict[Genre, float] = field(default_factory=dict)
    features_used: List[str] = field(default_factory=list)


@dataclass
class MoodAnalysis:
    """Container for mood analysis results"""

    primary_mood: MoodCategory
    confidence: float
    mood_probabilities: Dict[MoodCategory, float] = field(default_factory=dict)
    valence: float = 0.0  # Positive/negative emotion (-1 to 1)
    arousal: float = 0.0  # Energy level (0 to 1)
    dominance: float = 0.0  # Control/dominance (0 to 1)


@dataclass
class MusicAnalysis:
    """Complete music analysis results"""

    beat_info: Optional[BeatInfo] = None
    chord_progression: Optional[ChordProgression] = None
    genre_classification: Optional[GenreClassification] = None
    mood_analysis: Optional[MoodAnalysis] = None
    structural_analysis: Dict[str, Any] = field(default_factory=dict)
    audio_features: Dict[str, Any] = field(default_factory=dict)
    similarity_features: Optional[np.ndarray] = None


class BeatTracker:
    """
    Advanced beat tracking and tempo analysis
    """

    def __init__(self):
        self.hop_length = 512
        self.sr = 22050

    def analyze_beats(self, audio_data: np.ndarray, sr: int) -> BeatInfo:
        """
        Analyze beats, tempo, and rhythmic information
        """
        if not HAS_MIR_LIBS:
            raise MusicProcessingError(
                "Required libraries not available for beat tracking"
            )

        try:
            # Tempo and beat tracking
            tempo, beats = librosa.beat.beat_track(
                y=audio_data, sr=sr, hop_length=self.hop_length
            )
            beat_times = librosa.frames_to_time(
                beats, sr=sr, hop_length=self.hop_length
            )

            # Onset detection for more precise beat tracking
            onset_frames = librosa.onset.onset_detect(
                y=audio_data, sr=sr, hop_length=self.hop_length
            )
            onset_times = librosa.frames_to_time(
                onset_frames, sr=sr, hop_length=self.hop_length
            )

            # Rhythm stability analysis
            if len(beat_times) > 3:
                beat_intervals = np.diff(beat_times)
                rhythm_stability = 1.0 - (
                    np.std(beat_intervals) / np.mean(beat_intervals)
                )
                rhythm_stability = max(0.0, min(1.0, rhythm_stability))
            else:
                rhythm_stability = 0.0

            # Tempo confidence based on beat strength
            tempo_confidence = self._calculate_tempo_confidence(
                audio_data, sr, tempo, beats
            )

            # Downbeat detection (simplified)
            downbeats = self._detect_downbeats(audio_data, sr, beats)

            # Time signature estimation
            time_signature = self._estimate_time_signature(beat_times, downbeats)

            return BeatInfo(
                tempo=float(tempo),
                beats=beats,
                beat_times=beat_times,
                downbeats=downbeats,
                time_signature=time_signature,
                rhythm_stability=rhythm_stability,
                tempo_confidence=tempo_confidence,
            )

        except Exception as e:
            logger.error(f"Beat tracking error: {e}")
            return BeatInfo(tempo=0.0, beats=np.array([]), beat_times=np.array([]))

    def _calculate_tempo_confidence(
        self, audio_data: np.ndarray, sr: int, tempo: float, beats: np.ndarray
    ) -> float:
        """Calculate confidence in tempo detection"""
        try:
            # Analyze tempo consistency
            tempogram = librosa.feature.tempogram(
                y=audio_data, sr=sr, hop_length=self.hop_length
            )
            tempo_strength = np.max(tempogram, axis=0)
            return float(np.mean(tempo_strength))
        except Exception:
            return 0.5

    def _detect_downbeats(
        self, audio_data: np.ndarray, sr: int, beats: np.ndarray
    ) -> Optional[np.ndarray]:
        """Detect downbeats (first beat of each measure)"""
        try:
            if len(beats) < 8:
                return None

            # Simple downbeat detection based on spectral content
            chroma = librosa.feature.chroma_stft(
                y=audio_data, sr=sr, hop_length=self.hop_length
            )
            beat_chroma = chroma[:, beats]

            # Downbeats often have stronger harmonic content
            harmonic_strength = np.sum(beat_chroma, axis=0)

            # Find local maxima that could be downbeats
            peaks, _ = scipy.signal.find_peaks(harmonic_strength, distance=4)

            if len(peaks) > 0:
                return beats[peaks]

        except Exception as e:
            logger.debug(f"Downbeat detection error: {e}")

        return None

    def _estimate_time_signature(
        self, beat_times: np.ndarray, downbeats: Optional[np.ndarray]
    ) -> Optional[Tuple[int, int]]:
        """Estimate time signature"""
        if downbeats is None or len(downbeats) < 2:
            return (4, 4)  # Default

        try:
            # Calculate beats between downbeats
            downbeat_intervals = []
            for i in range(len(downbeats) - 1):
                start_time = librosa.frames_to_time(
                    downbeats[i], sr=self.sr, hop_length=self.hop_length
                )
                end_time = librosa.frames_to_time(
                    downbeats[i + 1], sr=self.sr, hop_length=self.hop_length
                )

                beats_in_measure = np.sum(
                    (beat_times >= start_time) & (beat_times < end_time)
                )
                if beats_in_measure > 0:
                    downbeat_intervals.append(beats_in_measure)

            if downbeat_intervals:
                most_common = max(set(downbeat_intervals), key=downbeat_intervals.count)
                return (int(most_common), 4)  # Assume quarter note denominator

        except Exception as e:
            logger.debug(f"Time signature estimation error: {e}")

        return (4, 4)


class ChordDetector:
    """
    Chord detection and harmonic analysis
    """

    def __init__(self):
        self.sr = 22050
        self.hop_length = 512
        self.chord_templates = self._create_chord_templates()

    def _create_chord_templates(self) -> Dict[str, np.ndarray]:
        """Create chord templates for matching"""
        templates = {}

        # Major chord template (root, major third, fifth)
        major_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])

        # Minor chord template (root, minor third, fifth)
        minor_template = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])

        # Create templates for all 12 keys
        chord_notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        for i, note in enumerate(chord_notes):
            # Major chords
            templates[f"{note}:maj"] = np.roll(major_template, i)
            # Minor chords
            templates[f"{note}:min"] = np.roll(minor_template, i)

        # No chord template
        templates["N"] = np.zeros(12)

        return templates

    def detect_chords(self, audio_data: np.ndarray, sr: int) -> ChordProgression:
        """
        Detect chord progressions in audio
        """
        if not HAS_MIR_LIBS:
            raise MusicProcessingError(
                "Required libraries not available for chord detection"
            )

        try:
            # Extract chromagram
            chroma = librosa.feature.chroma_stft(
                y=audio_data, sr=sr, hop_length=self.hop_length
            )

            # Normalize chroma
            chroma = librosa.util.normalize(chroma, norm=2, axis=0)

            # Template matching
            chord_labels = []
            chord_confidence = []

            for frame in range(chroma.shape[1]):
                chroma_frame = chroma[:, frame]
                best_chord = "N"
                best_score = 0.0

                for chord_name, template in self.chord_templates.items():
                    # Cosine similarity
                    score = np.dot(chroma_frame, template) / (
                        np.linalg.norm(chroma_frame) * np.linalg.norm(template) + 1e-8
                    )

                    if score > best_score:
                        best_score = score
                        best_chord = chord_name

                chord_labels.append(best_chord)
                chord_confidence.append(best_score)

            # Convert frame indices to time
            times = librosa.frames_to_time(
                np.arange(len(chord_labels)), sr=sr, hop_length=self.hop_length
            )

            # Smooth chord sequence and remove short segments
            smoothed_chords, smoothed_times, smoothed_confidence = self._smooth_chords(
                chord_labels, times, chord_confidence
            )

            # Key detection
            key_signature, mode = self._detect_key(chroma)

            # Harmonic complexity
            harmonic_complexity = self._calculate_harmonic_complexity(smoothed_chords)

            return ChordProgression(
                chords=smoothed_chords,
                chord_times=smoothed_times,
                chord_confidence=np.array(smoothed_confidence),
                key_signature=key_signature,
                mode=mode,
                harmonic_complexity=harmonic_complexity,
            )

        except Exception as e:
            logger.error(f"Chord detection error: {e}")
            return ChordProgression(
                chords=[], chord_times=np.array([]), chord_confidence=np.array([])
            )

    def _smooth_chords(
        self,
        chord_labels: List[str],
        times: np.ndarray,
        confidence: List[float],
        min_duration: float = 0.5,
    ) -> Tuple[List[str], np.ndarray, List[float]]:
        """Smooth chord sequence by removing short segments"""
        if not chord_labels:
            return [], np.array([]), []

        smoothed_chords = []
        smoothed_times = []
        smoothed_confidence = []

        current_chord = chord_labels[0]
        current_start = times[0]
        current_confidences = [confidence[0]]

        for i in range(1, len(chord_labels)):
            if chord_labels[i] == current_chord:
                current_confidences.append(confidence[i])
            else:
                # End of current chord segment
                duration = times[i] - current_start

                if duration >= min_duration:
                    smoothed_chords.append(current_chord)
                    smoothed_times.append(current_start)
                    smoothed_confidence.append(np.mean(current_confidences))

                # Start new segment
                current_chord = chord_labels[i]
                current_start = times[i]
                current_confidences = [confidence[i]]

        # Add final segment
        if len(times) > 0:
            duration = times[-1] - current_start
            if duration >= min_duration:
                smoothed_chords.append(current_chord)
                smoothed_times.append(current_start)
                smoothed_confidence.append(np.mean(current_confidences))

        return smoothed_chords, np.array(smoothed_times), smoothed_confidence

    def _detect_key(self, chroma: np.ndarray) -> Tuple[Optional[str], Optional[str]]:
        """Detect key signature using Krumhansl-Schmuckler algorithm"""
        try:
            # Key profiles (Krumhansl-Schmuckler)
            major_profile = np.array(
                [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
            )
            minor_profile = np.array(
                [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
            )

            # Average chroma across time
            mean_chroma = np.mean(chroma, axis=1)

            key_names = [
                "C",
                "C#",
                "D",
                "D#",
                "E",
                "F",
                "F#",
                "G",
                "G#",
                "A",
                "A#",
                "B",
            ]

            best_key = None
            best_mode = None
            best_correlation = -1.0

            for shift in range(12):
                # Major key correlation
                major_corr = scipy.stats.pearsonr(
                    mean_chroma, np.roll(major_profile, shift)
                )[0]
                if major_corr > best_correlation:
                    best_correlation = major_corr
                    best_key = key_names[shift]
                    best_mode = "major"

                # Minor key correlation
                minor_corr = scipy.stats.pearsonr(
                    mean_chroma, np.roll(minor_profile, shift)
                )[0]
                if minor_corr > best_correlation:
                    best_correlation = minor_corr
                    best_key = key_names[shift]
                    best_mode = "minor"

            return best_key, best_mode

        except Exception as e:
            logger.debug(f"Key detection error: {e}")
            return None, None

    def _calculate_harmonic_complexity(self, chords: List[str]) -> float:
        """Calculate harmonic complexity based on chord progressions"""
        if not chords:
            return 0.0

        try:
            # Count unique chords
            unique_chords = set(chord for chord in chords if chord != "N")
            chord_diversity = len(unique_chords) / max(len(chords), 1)

            # Analyze chord transitions
            transitions = set()
            for i in range(len(chords) - 1):
                if chords[i] != "N" and chords[i + 1] != "N":
                    transitions.add((chords[i], chords[i + 1]))

            transition_diversity = len(transitions) / max(len(chords) - 1, 1)

            # Combine measures
            complexity = (chord_diversity + transition_diversity) / 2
            return min(complexity, 1.0)

        except Exception:
            return 0.0


class GenreClassifier:
    """
    Machine learning-based genre classification
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_trained = False

    def extract_genre_features(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """Extract features relevant for genre classification"""
        if not HAS_MIR_LIBS:
            raise MusicProcessingError(
                "Required libraries not available for genre classification"
            )

        try:
            features = []

            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
            features.extend([np.mean(spectral_centroid), np.std(spectral_centroid)])

            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
            features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])

            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
            features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])

            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            for i in range(13):
                features.extend([np.mean(mfccs[i]), np.std(mfccs[i])])

            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
            for i in range(12):
                features.extend([np.mean(chroma[i]), np.std(chroma[i])])

            # Tempo and rhythm features
            tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
            features.append(tempo)

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)
            features.extend([np.mean(zcr), np.std(zcr)])

            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
            for i in range(contrast.shape[0]):
                features.extend([np.mean(contrast[i]), np.std(contrast[i])])

            return np.array(features)

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return np.array([])

    def train_model(self, training_data: List[Tuple[np.ndarray, int, Genre]]) -> None:
        """
        Train genre classification model

        Args:
            training_data: List of (audio_data, sample_rate, genre) tuples
        """
        if not HAS_MIR_LIBS:
            raise MusicProcessingError(
                "Required libraries not available for model training"
            )

        try:
            logger.info(f"Training genre classifier with {len(training_data)} samples")

            # Extract features for all samples
            X = []
            y = []

            for audio_data, sr, genre in training_data:
                features = self.extract_genre_features(audio_data, sr)
                if len(features) > 0:
                    X.append(features)
                    y.append(genre.value)

            if not X:
                raise MusicProcessingError(
                    "No valid features extracted from training data"
                )

            X = np.array(X)

            # Store feature names for reference
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Train model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_scaled, y)

            self.is_trained = True
            logger.info("Genre classifier training completed")

        except Exception as e:
            logger.error(f"Model training error: {e}")
            raise MusicProcessingError(f"Failed to train genre classifier: {e}")

    def classify_genre(self, audio_data: np.ndarray, sr: int) -> GenreClassification:
        """Classify genre of audio"""
        if not self.is_trained:
            # Use heuristic classification if no model is trained
            return self._heuristic_classification(audio_data, sr)

        try:
            features = self.extract_genre_features(audio_data, sr)
            if len(features) == 0:
                return GenreClassification(Genre.UNKNOWN, 0.0)

            features_scaled = self.scaler.transform(features.reshape(1, -1))

            # Get prediction probabilities
            probabilities = self.model.predict_proba(features_scaled)[0]
            class_names = self.model.classes_

            # Find best prediction
            best_idx = np.argmax(probabilities)
            primary_genre = Genre(class_names[best_idx])
            confidence = probabilities[best_idx]

            # Create probability dictionary
            genre_probs = {}
            for i, class_name in enumerate(class_names):
                try:
                    genre_probs[Genre(class_name)] = probabilities[i]
                except ValueError:
                    continue

            return GenreClassification(
                primary_genre=primary_genre,
                confidence=confidence,
                genre_probabilities=genre_probs,
                features_used=self.feature_names,
            )

        except Exception as e:
            logger.error(f"Genre classification error: {e}")
            return GenreClassification(Genre.UNKNOWN, 0.0)

    def _heuristic_classification(
        self, audio_data: np.ndarray, sr: int
    ) -> GenreClassification:
        """Heuristic genre classification based on audio characteristics"""
        try:
            # Extract basic features
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
            spectral_centroid = np.mean(
                librosa.feature.spectral_centroid(y=audio_data, sr=sr)
            )

            # Simple heuristics
            if tempo < 70:
                return GenreClassification(Genre.CLASSICAL, 0.6)
            elif tempo > 140 and spectral_centroid > 3000:
                return GenreClassification(Genre.ELECTRONIC, 0.6)
            elif 80 < tempo < 120:
                return GenreClassification(Genre.POP, 0.5)
            else:
                return GenreClassification(Genre.ROCK, 0.4)

        except Exception:
            return GenreClassification(Genre.UNKNOWN, 0.0)


class MoodAnalyzer:
    """
    Audio mood and emotion analysis
    """

    def __init__(self):
        pass

    def analyze_mood(self, audio_data: np.ndarray, sr: int) -> MoodAnalysis:
        """
        Analyze emotional content and mood of audio
        """
        if not HAS_MIR_LIBS:
            raise MusicProcessingError(
                "Required libraries not available for mood analysis"
            )

        try:
            # Extract mood-relevant features
            mood_features = self._extract_mood_features(audio_data, sr)

            # Calculate valence (positive/negative emotion)
            valence = self._calculate_valence(mood_features)

            # Calculate arousal (energy level)
            arousal = self._calculate_arousal(mood_features)

            # Calculate dominance
            dominance = self._calculate_dominance(mood_features)

            # Map to mood categories
            primary_mood, confidence, mood_probs = self._map_to_mood_categories(
                valence, arousal, dominance
            )

            return MoodAnalysis(
                primary_mood=primary_mood,
                confidence=confidence,
                mood_probabilities=mood_probs,
                valence=valence,
                arousal=arousal,
                dominance=dominance,
            )

        except Exception as e:
            logger.error(f"Mood analysis error: {e}")
            return MoodAnalysis(MoodCategory.NEUTRAL, 0.0)

    def _extract_mood_features(
        self, audio_data: np.ndarray, sr: int
    ) -> Dict[str, float]:
        """Extract features relevant for mood analysis"""
        features = {}

        try:
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
            features["tempo"] = tempo

            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
            features["spectral_centroid_mean"] = np.mean(spectral_centroid)
            features["spectral_centroid_std"] = np.std(spectral_centroid)

            # Energy and dynamics
            rms = librosa.feature.rms(y=audio_data)
            features["energy_mean"] = np.mean(rms)
            features["energy_std"] = np.std(rms)

            # Harmonic content
            harmonic, percussive = librosa.effects.hpss(audio_data)
            features["harmonic_ratio"] = np.sum(harmonic**2) / np.sum(audio_data**2)

            # Tonal features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
            features["tonal_stability"] = np.std(np.mean(chroma, axis=1))

            # MFCC for timbral characteristics
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=5)
            for i in range(5):
                features[f"mfcc_{i}_mean"] = np.mean(mfccs[i])

        except Exception as e:
            logger.debug(f"Mood feature extraction error: {e}")

        return features

    def _calculate_valence(self, features: Dict[str, float]) -> float:
        """Calculate valence (positive/negative emotion) from features"""
        try:
            # Higher spectral centroid and energy often indicate positive valence
            valence_score = 0.0

            if "spectral_centroid_mean" in features:
                # Normalize and weight spectral centroid
                centroid_norm = min(features["spectral_centroid_mean"] / 4000.0, 1.0)
                valence_score += 0.3 * centroid_norm

            if "energy_mean" in features:
                # Energy contribution
                valence_score += 0.3 * min(features["energy_mean"] * 100, 1.0)

            if "tempo" in features:
                # Tempo contribution (moderate tempos often positive)
                tempo_norm = 1.0 - abs(features["tempo"] - 120) / 120.0
                valence_score += 0.2 * max(tempo_norm, 0.0)

            if "harmonic_ratio" in features:
                # Harmonic content contribution
                valence_score += 0.2 * features["harmonic_ratio"]

            # Convert to -1 to 1 range
            return (valence_score * 2) - 1

        except Exception:
            return 0.0

    def _calculate_arousal(self, features: Dict[str, float]) -> float:
        """Calculate arousal (energy level) from features"""
        try:
            arousal_score = 0.0

            if "tempo" in features:
                # Tempo strongly influences arousal
                arousal_score += 0.4 * min(features["tempo"] / 180.0, 1.0)

            if "energy_mean" in features:
                # Energy level
                arousal_score += 0.3 * min(features["energy_mean"] * 100, 1.0)

            if "energy_std" in features:
                # Energy variation
                arousal_score += 0.3 * min(features["energy_std"] * 100, 1.0)

            return min(arousal_score, 1.0)

        except Exception:
            return 0.5

    def _calculate_dominance(self, features: Dict[str, float]) -> float:
        """Calculate dominance from features"""
        try:
            dominance_score = 0.5  # Neutral baseline

            if "energy_mean" in features and "spectral_centroid_mean" in features:
                # Strong, bright sounds often indicate dominance
                energy_factor = min(features["energy_mean"] * 100, 1.0)
                brightness_factor = min(
                    features["spectral_centroid_mean"] / 3000.0, 1.0
                )
                dominance_score = (energy_factor + brightness_factor) / 2

            return dominance_score

        except Exception:
            return 0.5

    def _map_to_mood_categories(
        self, valence: float, arousal: float, dominance: float
    ) -> Tuple[MoodCategory, float, Dict[MoodCategory, float]]:
        """Map valence/arousal/dominance to mood categories"""
        mood_scores = {}

        # Define mood regions in valence-arousal space
        if valence > 0.3 and arousal > 0.6:
            mood_scores[MoodCategory.HAPPY] = 0.8
            mood_scores[MoodCategory.ENERGETIC] = 0.7
        elif valence > 0.3 and arousal < 0.4:
            mood_scores[MoodCategory.CALM] = 0.7
            mood_scores[MoodCategory.ROMANTIC] = 0.6
        elif valence < -0.3 and arousal > 0.6:
            mood_scores[MoodCategory.ANGRY] = 0.8
        elif valence < -0.3 and arousal < 0.4:
            mood_scores[MoodCategory.SAD] = 0.8
            mood_scores[MoodCategory.MELANCHOLIC] = 0.7
        elif arousal > 0.7:
            mood_scores[MoodCategory.ENERGETIC] = 0.6
        elif arousal < 0.3:
            mood_scores[MoodCategory.CALM] = 0.6
        else:
            mood_scores[MoodCategory.NEUTRAL] = 0.5

        # Add some randomness based on dominance
        if dominance > 0.7:
            mood_scores[MoodCategory.TRIUMPHANT] = (
                mood_scores.get(MoodCategory.TRIUMPHANT, 0.0) + 0.3
            )
        elif dominance < 0.3:
            mood_scores[MoodCategory.MYSTERIOUS] = (
                mood_scores.get(MoodCategory.MYSTERIOUS, 0.0) + 0.3
            )

        # Find primary mood
        if mood_scores:
            primary_mood = max(mood_scores.keys(), key=lambda k: mood_scores[k])
            confidence = mood_scores[primary_mood]
        else:
            primary_mood = MoodCategory.NEUTRAL
            confidence = 0.5
            mood_scores[MoodCategory.NEUTRAL] = 0.5

        # Normalize probabilities
        total = sum(mood_scores.values())
        if total > 0:
            mood_probabilities = {
                mood: score / total for mood, score in mood_scores.items()
            }
        else:
            mood_probabilities = {MoodCategory.NEUTRAL: 1.0}

        return primary_mood, confidence, mood_probabilities


class MusicInformationRetrieval:
    """
    Main MIR system combining all analysis components
    """

    def __init__(self):
        self.beat_tracker = BeatTracker()
        self.chord_detector = ChordDetector()
        self.genre_classifier = GenreClassifier()
        self.mood_analyzer = MoodAnalyzer()

    async def analyze_music(
        self,
        audio_data: np.ndarray,
        sr: int,
        include_beats: bool = True,
        include_chords: bool = True,
        include_genre: bool = True,
        include_mood: bool = True,
    ) -> MusicAnalysis:
        """
        Perform comprehensive music analysis
        """
        if not HAS_MIR_LIBS:
            raise MusicProcessingError(
                "Required libraries not available for music analysis"
            )

        analysis = MusicAnalysis()

        try:
            # Beat tracking
            if include_beats:
                logger.debug("Analyzing beats and tempo...")
                analysis.beat_info = self.beat_tracker.analyze_beats(audio_data, sr)

            # Chord detection
            if include_chords:
                logger.debug("Detecting chords...")
                analysis.chord_progression = self.chord_detector.detect_chords(
                    audio_data, sr
                )

            # Genre classification
            if include_genre:
                logger.debug("Classifying genre...")
                analysis.genre_classification = self.genre_classifier.classify_genre(
                    audio_data, sr
                )

            # Mood analysis
            if include_mood:
                logger.debug("Analyzing mood...")
                analysis.mood_analysis = self.mood_analyzer.analyze_mood(audio_data, sr)

            # Extract general audio features
            analysis.audio_features = await self._extract_audio_features(audio_data, sr)

            # Extract similarity features
            analysis.similarity_features = self._extract_similarity_features(
                audio_data, sr
            )

            # Structural analysis
            analysis.structural_analysis = await self._analyze_structure(audio_data, sr)

            logger.info("Music analysis completed successfully")
            return analysis

        except Exception as e:
            logger.error(f"Music analysis error: {e}")
            raise MusicProcessingError(f"Music analysis failed: {e}")

    async def _extract_audio_features(
        self, audio_data: np.ndarray, sr: int
    ) -> Dict[str, Any]:
        """Extract general audio features"""
        features = {}

        try:
            # Duration
            features["duration"] = len(audio_data) / sr

            # RMS energy
            rms = librosa.feature.rms(y=audio_data)
            features["rms_mean"] = float(np.mean(rms))
            features["rms_std"] = float(np.std(rms))

            # Spectral statistics
            stft = librosa.stft(audio_data)
            magnitude = np.abs(stft)

            features["spectral_mean"] = float(np.mean(magnitude))
            features["spectral_std"] = float(np.std(magnitude))

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)
            features["zcr_mean"] = float(np.mean(zcr))
            features["zcr_std"] = float(np.std(zcr))

        except Exception as e:
            logger.debug(f"Audio features extraction error: {e}")

        return features

    def _extract_similarity_features(
        self, audio_data: np.ndarray, sr: int
    ) -> Optional[np.ndarray]:
        """Extract features for similarity comparison"""
        try:
            # Combine multiple feature types for similarity
            features = []

            # MFCCs (13 coefficients)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            features.extend(np.mean(mfccs, axis=1))

            # Chroma (12 pitch classes)
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
            features.extend(np.mean(chroma, axis=1))

            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
            features.append(np.mean(spectral_centroid))

            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
            features.append(np.mean(spectral_bandwidth))

            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
            features.append(np.mean(spectral_rolloff))

            # Tempo
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
            features.append(tempo)

            return np.array(features)

        except Exception as e:
            logger.debug(f"Similarity features extraction error: {e}")
            return None

    async def _analyze_structure(
        self, audio_data: np.ndarray, sr: int
    ) -> Dict[str, Any]:
        """Analyze musical structure and form"""
        structure = {}

        try:
            # Segment boundary detection
            boundaries = librosa.segment.agglomerative(audio_data, sr)
            structure["segment_boundaries"] = boundaries.tolist()
            structure["num_segments"] = len(boundaries)

            # Repetition analysis
            recurrence_matrix = librosa.segment.recurrence_matrix(audio_data, sr)
            structure["repetition_score"] = float(np.mean(recurrence_matrix))

        except Exception as e:
            logger.debug(f"Structural analysis error: {e}")

        return structure

    async def analyze_file(self, file_path: str, **kwargs) -> MusicAnalysis:
        """
        Analyze music from file
        """
        try:
            audio_data, sr = librosa.load(file_path, sr=None, mono=True)
            return await self.analyze_music(audio_data, sr, **kwargs)

        except Exception as e:
            logger.error(f"File analysis error: {e}")
            raise MusicProcessingError(f"Failed to analyze file {file_path}: {e}")

    def save_model(self, file_path: str) -> None:
        """Save trained models to file"""
        if self.genre_classifier.is_trained:
            with open(file_path, "wb") as f:
                pickle.dump(
                    {
                        "genre_model": self.genre_classifier.model,
                        "genre_scaler": self.genre_classifier.scaler,
                        "feature_names": self.genre_classifier.feature_names,
                    },
                    f,
                )

    def load_model(self, file_path: str) -> None:
        """Load trained models from file"""
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                self.genre_classifier.model = data["genre_model"]
                self.genre_classifier.scaler = data["genre_scaler"]
                self.genre_classifier.feature_names = data["feature_names"]
                self.genre_classifier.is_trained = True
                logger.info("MIR models loaded successfully")
        except Exception as e:
            logger.error(f"Model loading error: {e}")


# Convenience functions


async def analyze_music_file(file_path: str, **kwargs) -> MusicAnalysis:
    """Convenience function to analyze music file"""
    mir = MusicInformationRetrieval()
    return await mir.analyze_file(file_path, **kwargs)


def calculate_similarity(features1: np.ndarray, features2: np.ndarray) -> float:
    """Calculate similarity between two feature vectors"""
    if len(features1) != len(features2):
        return 0.0

    try:
        # Cosine similarity
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

    except Exception:
        return 0.0


# Example usage and testing
if __name__ == "__main__":

    async def demo():
        """Demo music information retrieval"""
        try:
            # Create MIR system
            mir = MusicInformationRetrieval()

            # Example: analyze a sample audio file (would need actual file)
            # analysis = await mir.analyze_file("sample_music.wav")

            # Generate synthetic audio for demo
            duration = 5  # seconds
            sr = 22050
            t = np.linspace(0, duration, duration * sr)

            # Create a synthetic musical signal
            fundamental = 440  # A4
            audio_data = (
                np.sin(2 * np.pi * fundamental * t) * 0.3
                + np.sin(2 * np.pi * fundamental * 1.5 * t) * 0.2
                + np.sin(2 * np.pi * fundamental * 2 * t) * 0.1
            )

            # Add some noise
            audio_data += np.random.normal(0, 0.05, len(audio_data))

            print("Analyzing synthetic audio...")
            analysis = await mir.analyze_music(audio_data, sr)

            # Print results
            print(f"\n=== MUSIC ANALYSIS RESULTS ===")

            if analysis.beat_info:
                print(f"Tempo: {analysis.beat_info.tempo:.1f} BPM")
                print(f"Time Signature: {analysis.beat_info.time_signature}")
                print(f"Rhythm Stability: {analysis.beat_info.rhythm_stability:.2f}")

            if analysis.chord_progression:
                print(
                    f"Key: {analysis.chord_progression.key_signature} {analysis.chord_progression.mode}"
                )
                print(
                    f"Chords: {analysis.chord_progression.chords[:10]}..."
                )  # First 10 chords
                print(
                    f"Harmonic Complexity: {analysis.chord_progression.harmonic_complexity:.2f}"
                )

            if analysis.genre_classification:
                print(f"Genre: {analysis.genre_classification.primary_genre.value}")
                print(
                    f"Genre Confidence: {analysis.genre_classification.confidence:.2f}"
                )

            if analysis.mood_analysis:
                print(f"Mood: {analysis.mood_analysis.primary_mood.value}")
                print(f"Valence: {analysis.mood_analysis.valence:.2f}")
                print(f"Arousal: {analysis.mood_analysis.arousal:.2f}")

            print(f"Audio Features: {list(analysis.audio_features.keys())}")

            if analysis.similarity_features is not None:
                print(
                    f"Similarity Features Shape: {analysis.similarity_features.shape}"
                )

        except Exception as e:
            print(f"Demo error: {e}")

    # Run demo
    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
