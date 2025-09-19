"""
Test suite for audio_fingerprint.py (audio fingerprinting and matching).
"""

import os
import tempfile
import shutil
import numpy as np
import pytest
import sqlite3
from datasets import audio_fingerprint


def create_dummy_wav(path, frequency=440, duration_sec=1, sr=22050):
    """Create a dummy wav file with a sine wave."""
    import soundfile as sf

    t = np.linspace(0, duration_sec, int(sr * duration_sec), False)
    tone = 0.5 * np.sin(2 * np.pi * frequency * t)
    sf.write(path, tone, sr)
    return path


def test_fingerprint_extraction(tmp_path):
    """Test audio fingerprint extraction."""
    # Create test audio files
    audio1 = tmp_path / "test1.wav"
    audio2 = tmp_path / "test2.wav"
    create_dummy_wav(str(audio1), frequency=440)  # A4 note
    create_dummy_wav(str(audio2), frequency=880)  # A5 note

    # Extract fingerprints
    extractor = audio_fingerprint.AudioFingerprintExtractor()
    fp1 = extractor.extract_fingerprint(str(audio1))
    fp2 = extractor.extract_fingerprint(str(audio2))

    assert fp1 is not None
    assert fp2 is not None
    assert fp1.file_path == str(audio1)
    assert fp2.file_path == str(audio2)
    assert fp1.duration > 0
    assert fp2.duration > 0
    assert len(fp1.chroma_features) == 12  # Default n_chroma
    assert len(fp1.mfcc_features) == 13  # Default n_mfcc
    assert (
        fp1.hash_signature != fp2.hash_signature
    )  # Different audio should have different fingerprints


def test_fingerprint_database(tmp_path):
    """Test fingerprint database operations."""
    # Create test database
    db_path = tmp_path / "test_fingerprints.db"
    db = audio_fingerprint.FingerprintDatabase(str(db_path))

    # Create test audio and fingerprint
    audio_path = tmp_path / "test.wav"
    create_dummy_wav(str(audio_path))

    extractor = audio_fingerprint.AudioFingerprintExtractor()
    fingerprint = extractor.extract_fingerprint(str(audio_path))
    assert fingerprint is not None

    # Store fingerprint
    success = db.store_fingerprint(fingerprint)
    assert success

    # Retrieve fingerprint
    retrieved = db.get_fingerprint(str(audio_path))
    assert retrieved is not None
    assert retrieved.file_path == fingerprint.file_path
    assert retrieved.hash_signature == fingerprint.hash_signature
    assert retrieved.chroma_features == fingerprint.chroma_features

    # Get all fingerprints
    all_fps = db.get_all_fingerprints()
    assert len(all_fps) == 1
    assert all_fps[0].file_path == fingerprint.file_path


def test_fingerprint_matching(tmp_path):
    """Test fingerprint matching and similarity."""
    # Create test database
    db_path = tmp_path / "test_fingerprints.db"
    db = audio_fingerprint.FingerprintDatabase(str(db_path))
    matcher = audio_fingerprint.FingerprintMatcher(db)

    # Create test audio files
    audio1 = tmp_path / "similar1.wav"
    audio2 = tmp_path / "similar2.wav"
    audio3 = tmp_path / "different.wav"

    create_dummy_wav(str(audio1), frequency=440, duration_sec=1)
    create_dummy_wav(
        str(audio2), frequency=442, duration_sec=1
    )  # Very similar frequency
    create_dummy_wav(str(audio3), frequency=1000, duration_sec=1)  # Very different

    # Extract and store fingerprints
    extractor = audio_fingerprint.AudioFingerprintExtractor()
    fp1 = extractor.extract_fingerprint(str(audio1))
    fp2 = extractor.extract_fingerprint(str(audio2))
    fp3 = extractor.extract_fingerprint(str(audio3))

    assert all(fp is not None for fp in [fp1, fp2, fp3])

    # Store in database
    for fp in [fp1, fp2, fp3]:
        db.store_fingerprint(fp)

    # Test similarity computation
    sim_12 = matcher.compute_similarity(fp1, fp2)  # Similar frequencies
    sim_13 = matcher.compute_similarity(fp1, fp3)  # Different frequencies

    assert 0 <= sim_12 <= 1
    assert 0 <= sim_13 <= 1
    assert sim_12 > sim_13  # Similar frequencies should be more similar

    # Test finding matches
    matches = matcher.find_matches(fp1, threshold=0.5, max_results=5)
    assert len(matches) >= 1  # Should find at least fp2 and/or fp3

    # Test audio identification
    result = matcher.identify_audio(str(audio1), threshold=0.5)
    assert result is not None  # Should find a match


def test_convenience_functions(tmp_path):
    """Test convenience functions."""
    # Create test audio
    audio_path = tmp_path / "test.wav"
    create_dummy_wav(str(audio_path))

    db_path = tmp_path / "test_fingerprints.db"

    # Test create_fingerprint
    fp = audio_fingerprint.create_fingerprint(str(audio_path))
    assert fp is not None
    assert fp.file_path == str(audio_path)

    # Test store_audio_fingerprint
    success = audio_fingerprint.store_audio_fingerprint(str(audio_path), str(db_path))
    assert success

    # Verify database was created and has data
    assert os.path.exists(db_path)

    # Test find_similar_audio
    matches = audio_fingerprint.find_similar_audio(
        str(audio_path), str(db_path), threshold=0.5
    )
    # Should not find itself as similar (excluded by hash comparison)
    assert isinstance(matches, list)


def test_invalid_audio_handling(tmp_path):
    """Test handling of invalid audio files."""
    # Non-existent file
    extractor = audio_fingerprint.AudioFingerprintExtractor()
    fp = extractor.extract_fingerprint("nonexistent.wav")
    assert fp is None

    # Invalid file
    invalid_path = tmp_path / "invalid.wav"
    with open(invalid_path, "wb") as f:
        f.write(b"not audio data")

    fp = extractor.extract_fingerprint(str(invalid_path))
    assert fp is None

    # Test convenience function with invalid file
    success = audio_fingerprint.store_audio_fingerprint(str(invalid_path))
    assert not success


def test_database_error_handling(tmp_path):
    """Test database error handling."""
    # Test with invalid database path (read-only directory)
    if hasattr(os, "chmod"):  # Unix-like systems
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        os.chmod(readonly_dir, 0o444)  # Read-only

        try:
            db_path = readonly_dir / "test.db"
            db = audio_fingerprint.FingerprintDatabase(str(db_path))

            # Create a dummy fingerprint
            audio_path = tmp_path / "test.wav"
            create_dummy_wav(str(audio_path))
            extractor = audio_fingerprint.AudioFingerprintExtractor()
            fp = extractor.extract_fingerprint(str(audio_path))

            # This should fail gracefully
            success = db.store_fingerprint(fp)
            assert not success

        finally:
            # Restore permissions for cleanup
            os.chmod(readonly_dir, 0o755)
