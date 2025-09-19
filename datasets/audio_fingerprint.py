"""
Audio fingerprinting utilities for Vega 2.0 multi-modal learning.

This module provides:
- Audio fingerprint generation using chromagram and spectral features
- Fingerprint matching and similarity computation
- Database storage and retrieval of fingerprints
- Audio identification from fingerprints
"""

import os
import numpy as np
import librosa
import hashlib
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging
import sqlite3
import json
import pickle
from dataclasses import dataclass, asdict
from scipy.spatial.distance import cosine, euclidean
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioFingerprint:
    """Represents an audio fingerprint with metadata."""

    file_path: str
    duration: float
    sample_rate: int
    chroma_features: List[float]
    mfcc_features: List[float]
    spectral_centroid: List[float]
    zero_crossing_rate: List[float]
    hash_signature: str
    created_at: str


class AudioFingerprintExtractor:
    """Extract and compute audio fingerprints from audio files."""

    def __init__(self, n_chroma: int = 12, n_mfcc: int = 13):
        self.n_chroma = n_chroma
        self.n_mfcc = n_mfcc

    def extract_fingerprint(
        self, audio_path: str, sr: int = 22050
    ) -> Optional[AudioFingerprint]:
        """Extract audio fingerprint from file."""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=sr, mono=True)
            duration = len(y) / sr

            # Extract features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=self.n_chroma)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)

            # Compute mean features for fingerprint
            chroma_mean = np.mean(chroma, axis=1).tolist()
            mfcc_mean = np.mean(mfcc, axis=1).tolist()
            spectral_centroid_mean = np.mean(spectral_centroid, axis=1).tolist()
            zcr_mean = np.mean(zcr, axis=1).tolist()

            # Create hash signature
            feature_string = (
                f"{chroma_mean}{mfcc_mean}{spectral_centroid_mean}{zcr_mean}"
            )
            hash_signature = hashlib.sha256(feature_string.encode()).hexdigest()

            from datetime import datetime

            created_at = datetime.now().isoformat()

            fingerprint = AudioFingerprint(
                file_path=audio_path,
                duration=duration,
                sample_rate=sr,
                chroma_features=chroma_mean,
                mfcc_features=mfcc_mean,
                spectral_centroid=spectral_centroid_mean,
                zero_crossing_rate=zcr_mean,
                hash_signature=hash_signature,
                created_at=created_at,
            )

            logger.info(f"Extracted fingerprint for {audio_path}")
            return fingerprint

        except Exception as e:
            logger.error(f"Fingerprint extraction failed for {audio_path}: {e}")
            return None


class FingerprintDatabase:
    """Database for storing and retrieving audio fingerprints."""

    def __init__(self, db_path: str = "audio_fingerprints.db"):
        self.db_path = db_path
        self._db_available = True  # Initialize flag
        self._init_database()

    def _init_database(self):
        """Initialize the fingerprint database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS fingerprints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE,
                    duration REAL,
                    sample_rate INTEGER,
                    chroma_features TEXT,
                    mfcc_features TEXT,
                    spectral_centroid TEXT,
                    zero_crossing_rate TEXT,
                    hash_signature TEXT UNIQUE,
                    created_at TEXT
                )
            """
            )

            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Database initialization failed: {e}")
            self._db_available = False
        except Exception as e:
            logger.error(f"Unexpected error during database initialization: {e}")
            self._db_available = False

    def store_fingerprint(self, fingerprint: AudioFingerprint) -> bool:
        """Store a fingerprint in the database."""
        if not self._db_available:
            return False

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO fingerprints 
                (file_path, duration, sample_rate, chroma_features, mfcc_features, 
                 spectral_centroid, zero_crossing_rate, hash_signature, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    fingerprint.file_path,
                    fingerprint.duration,
                    fingerprint.sample_rate,
                    json.dumps(fingerprint.chroma_features),
                    json.dumps(fingerprint.mfcc_features),
                    json.dumps(fingerprint.spectral_centroid),
                    json.dumps(fingerprint.zero_crossing_rate),
                    fingerprint.hash_signature,
                    fingerprint.created_at,
                ),
            )

            conn.commit()
            conn.close()
            logger.info(f"Stored fingerprint for {fingerprint.file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to store fingerprint: {e}")
            return False

    def get_fingerprint(self, file_path: str) -> Optional[AudioFingerprint]:
        """Retrieve a fingerprint by file path."""
        if not self._db_available:
            return None

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM fingerprints WHERE file_path = ?", (file_path,)
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                return AudioFingerprint(
                    file_path=row[1],
                    duration=row[2],
                    sample_rate=row[3],
                    chroma_features=json.loads(row[4]),
                    mfcc_features=json.loads(row[5]),
                    spectral_centroid=json.loads(row[6]),
                    zero_crossing_rate=json.loads(row[7]),
                    hash_signature=row[8],
                    created_at=row[9],
                )
            return None

        except Exception as e:
            logger.error(f"Failed to retrieve fingerprint: {e}")
            return None

    def get_all_fingerprints(self) -> List[AudioFingerprint]:
        """Retrieve all fingerprints from database."""
        if not self._db_available:
            return []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM fingerprints")
            rows = cursor.fetchall()
            conn.close()

            fingerprints = []
            for row in rows:
                fingerprints.append(
                    AudioFingerprint(
                        file_path=row[1],
                        duration=row[2],
                        sample_rate=row[3],
                        chroma_features=json.loads(row[4]),
                        mfcc_features=json.loads(row[5]),
                        spectral_centroid=json.loads(row[6]),
                        zero_crossing_rate=json.loads(row[7]),
                        hash_signature=row[8],
                        created_at=row[9],
                    )
                )

            return fingerprints

        except Exception as e:
            logger.error(f"Failed to retrieve all fingerprints: {e}")
            return []


class FingerprintMatcher:
    """Match and compare audio fingerprints."""

    def __init__(self, database: FingerprintDatabase):
        self.database = database

    def compute_similarity(self, fp1: AudioFingerprint, fp2: AudioFingerprint) -> float:
        """Compute similarity between two fingerprints (0-1, higher = more similar)."""
        try:
            # Combine all features into vectors
            features1 = np.array(
                fp1.chroma_features
                + fp1.mfcc_features
                + fp1.spectral_centroid
                + fp1.zero_crossing_rate
            )
            features2 = np.array(
                fp2.chroma_features
                + fp2.mfcc_features
                + fp2.spectral_centroid
                + fp2.zero_crossing_rate
            )

            # Compute cosine similarity (1 - cosine distance)
            similarity = 1 - cosine(features1, features2)
            return max(0.0, similarity)  # Ensure non-negative

        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            return 0.0

    def find_matches(
        self,
        query_fingerprint: AudioFingerprint,
        threshold: float = 0.8,
        max_results: int = 10,
    ) -> List[Tuple[AudioFingerprint, float]]:
        """Find matching fingerprints in database."""
        try:
            all_fingerprints = self.database.get_all_fingerprints()
            matches = []

            for fp in all_fingerprints:
                if (
                    fp.hash_signature != query_fingerprint.hash_signature
                ):  # Skip identical
                    similarity = self.compute_similarity(query_fingerprint, fp)
                    if similarity >= threshold:
                        matches.append((fp, similarity))

            # Sort by similarity (descending) and limit results
            matches.sort(key=lambda x: x[1], reverse=True)
            return matches[:max_results]

        except Exception as e:
            logger.error(f"Match finding failed: {e}")
            return []

    def identify_audio(
        self, audio_path: str, threshold: float = 0.8
    ) -> Optional[Tuple[AudioFingerprint, float]]:
        """Identify an audio file by finding its best match."""
        try:
            extractor = AudioFingerprintExtractor()
            query_fp = extractor.extract_fingerprint(audio_path)

            if not query_fp:
                return None

            matches = self.find_matches(query_fp, threshold, max_results=1)
            return matches[0] if matches else None

        except Exception as e:
            logger.error(f"Audio identification failed: {e}")
            return None


# Convenience functions
def create_fingerprint(audio_path: str) -> Optional[AudioFingerprint]:
    """Create fingerprint for an audio file."""
    extractor = AudioFingerprintExtractor()
    return extractor.extract_fingerprint(audio_path)


def store_audio_fingerprint(
    audio_path: str, db_path: str = "audio_fingerprints.db"
) -> bool:
    """Extract and store fingerprint for an audio file."""
    fingerprint = create_fingerprint(audio_path)
    if fingerprint:
        db = FingerprintDatabase(db_path)
        return db.store_fingerprint(fingerprint)
    return False


def find_similar_audio(
    audio_path: str, db_path: str = "audio_fingerprints.db", threshold: float = 0.8
) -> List[Tuple[str, float]]:
    """Find similar audio files in database."""
    db = FingerprintDatabase(db_path)
    matcher = FingerprintMatcher(db)

    result = matcher.identify_audio(audio_path, threshold)
    if result:
        fp, similarity = result
        return [(fp.file_path, similarity)]
    return []


if __name__ == "__main__":
    # Example usage
    test_audio = "sample_audio.wav"
    if os.path.exists(test_audio):
        # Create and store fingerprint
        success = store_audio_fingerprint(test_audio)
        print(f"Fingerprint stored: {success}")

        # Find similar audio
        matches = find_similar_audio(test_audio)
        print(f"Similar audio found: {matches}")
