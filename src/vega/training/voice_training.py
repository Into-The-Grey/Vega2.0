"""
VEGA Voice Training System
==========================

Train VEGA to recognize and synthesize user's voice for personalized
voice interactions. Supports both voice recognition (STT) and voice
synthesis/cloning (TTS) training.

Features:
- Voice sample collection and preprocessing
- Voice feature extraction and analysis
- Model fine-tuning for voice recognition
- Voice cloning for TTS personalization
- Quality assessment and validation
- Progressive training with multiple samples

Training Modes:
1. Voice Recognition: Train to recognize your voice for STT
2. Voice Cloning: Train to synthesize speech in your voice
3. Voice Adaptation: Adapt both recognition and synthesis
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Audio processing
try:
    import librosa
    import soundfile as sf
    import torch
    import torchaudio

    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: Audio processing libraries not available")
    print("Install with: pip install librosa soundfile torch torchaudio")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vega.personality.vega_core import get_vega_personality

logger = logging.getLogger(__name__)


class VoiceTrainingConfig:
    """Configuration for voice training"""

    def __init__(
        self,
        training_mode: str = "both",  # recognition, cloning, both
        sample_rate: int = 16000,
        min_duration: float = 2.0,  # seconds
        max_duration: float = 30.0,  # seconds
        quality_threshold: float = 0.7,
        batch_size: int = 8,
        num_epochs: int = 10,
        learning_rate: float = 1e-4,
        output_dir: str = "data/voice_training",
    ):
        self.training_mode = training_mode
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.quality_threshold = quality_threshold
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


class VoiceTrainer:
    """Main voice training system"""

    def __init__(self, config: Optional[VoiceTrainingConfig] = None):
        self.config = config or VoiceTrainingConfig()
        self.personality = get_vega_personality()
        self.samples: List[Dict[str, Any]] = []
        self.voice_profile: Dict[str, Any] = {}

        logger.info(f"Initialized VoiceTrainer (mode: {self.config.training_mode})")

    def load_audio_file(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load and preprocess audio file"""
        if not AUDIO_AVAILABLE:
            raise RuntimeError("Audio processing libraries not available")

        logger.info(f"Loading audio file: {file_path}")

        # Load audio
        audio, sr = librosa.load(file_path, sr=self.config.sample_rate, mono=True)

        # Validate duration
        duration = len(audio) / sr
        if duration < self.config.min_duration:
            raise ValueError(
                f"Audio too short: {duration:.1f}s < {self.config.min_duration}s"
            )
        if duration > self.config.max_duration:
            logger.warning(
                f"Audio too long: {duration:.1f}s > {self.config.max_duration}s, truncating"
            )
            audio = audio[: int(self.config.max_duration * sr)]

        return audio, sr

    def extract_voice_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract voice characteristics from audio"""
        if not AUDIO_AVAILABLE:
            raise RuntimeError("Audio processing libraries not available")

        features = {}

        try:
            # Mel-frequency cepstral coefficients (voice timbre)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            features["mfcc_mean"] = np.mean(mfccs, axis=1).tolist()
            features["mfcc_std"] = np.std(mfccs, axis=1).tolist()

            # Pitch/fundamental frequency
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            if pitch_values:
                features["pitch_mean"] = float(np.mean(pitch_values))
                features["pitch_std"] = float(np.std(pitch_values))
                features["pitch_range"] = float(np.ptp(pitch_values))
            else:
                features["pitch_mean"] = 0.0
                features["pitch_std"] = 0.0
                features["pitch_range"] = 0.0

            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))

            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))

            # Zero crossing rate (voice quality)
            zcr = librosa.feature.zero_crossing_rate(audio)
            features["zero_crossing_rate"] = float(np.mean(zcr))

            # Energy
            rms = librosa.feature.rms(y=audio)
            features["rms_energy_mean"] = float(np.mean(rms))

            # Duration
            features["duration_seconds"] = float(len(audio) / sr)

            logger.info(f"Extracted {len(features)} voice features")

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise

        return features

    def assess_audio_quality(
        self, audio: np.ndarray, sr: int, features: Dict[str, Any]
    ) -> float:
        """Assess the quality of an audio sample"""
        quality_score = 1.0

        # Check signal-to-noise ratio (simplified)
        rms = np.sqrt(np.mean(audio**2))
        if rms < 0.01:
            quality_score *= 0.5  # Very quiet audio

        # Check for clipping
        if np.max(np.abs(audio)) > 0.99:
            quality_score *= 0.8  # Audio might be clipped

        # Check spectral characteristics
        if features.get("spectral_centroid_mean", 0) < 1000:
            quality_score *= 0.9  # Might be too muffled

        # Check duration
        duration = features.get("duration_seconds", 0)
        if duration < 3.0:
            quality_score *= 0.9  # Prefer longer samples

        logger.info(f"Audio quality score: {quality_score:.2f}")
        return quality_score

    def add_voice_sample(
        self,
        file_path: str,
        transcription: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add a voice sample to the training set"""
        # Load and process audio
        audio, sr = self.load_audio_file(file_path)

        # Extract features
        features = self.extract_voice_features(audio, sr)

        # Assess quality
        quality_score = self.assess_audio_quality(audio, sr, features)

        if quality_score < self.config.quality_threshold:
            logger.warning(
                f"Low quality sample: {quality_score:.2f} < {self.config.quality_threshold}"
            )
            logger.warning("Consider re-recording with better audio quality")

        # Save processed audio
        processed_filename = f"sample_{len(self.samples):04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        processed_path = self.config.output_dir / "samples" / processed_filename
        processed_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(str(processed_path), audio, sr)

        # Create sample metadata
        sample = {
            "id": len(self.samples),
            "original_path": file_path,
            "processed_path": str(processed_path),
            "transcription": transcription,
            "context": context,
            "features": features,
            "quality_score": quality_score,
            "timestamp": datetime.now().isoformat(),
        }

        self.samples.append(sample)

        # Log to personality system
        self.personality.log_voice_training_session(
            session_type="sample_collection",
            audio_file_path=str(processed_path),
            duration_seconds=features["duration_seconds"],
            quality_score=quality_score,
            features=features,
            notes=f"Sample {len(self.samples)}: {transcription or 'no transcription'}",
        )

        logger.info(
            f"Added voice sample {len(self.samples)}: "
            f"{features['duration_seconds']:.1f}s, quality={quality_score:.2f}"
        )

        return sample

    def analyze_voice_profile(self) -> Dict[str, Any]:
        """Analyze collected samples to create voice profile"""
        if not self.samples:
            raise ValueError("No samples available for analysis")

        logger.info(f"Analyzing voice profile from {len(self.samples)} samples...")

        # Aggregate features from all samples
        profile = {
            "sample_count": len(self.samples),
            "total_duration": sum(
                s["features"]["duration_seconds"] for s in self.samples
            ),
            "avg_quality_score": np.mean([s["quality_score"] for s in self.samples]),
            "created_at": datetime.now().isoformat(),
        }

        # Average pitch characteristics
        pitch_means = [
            s["features"]["pitch_mean"]
            for s in self.samples
            if s["features"]["pitch_mean"] > 0
        ]
        if pitch_means:
            profile["voice_pitch"] = {
                "mean": float(np.mean(pitch_means)),
                "std": float(np.std(pitch_means)),
                "min": float(np.min(pitch_means)),
                "max": float(np.max(pitch_means)),
            }

        # Average spectral characteristics
        profile["spectral_profile"] = {
            "centroid_mean": float(
                np.mean([s["features"]["spectral_centroid_mean"] for s in self.samples])
            ),
            "rolloff_mean": float(
                np.mean([s["features"]["spectral_rolloff_mean"] for s in self.samples])
            ),
        }

        # MFCC profile (voice timbre)
        all_mfcc_means = np.array([s["features"]["mfcc_mean"] for s in self.samples])
        profile["mfcc_profile"] = {
            "mean": np.mean(all_mfcc_means, axis=0).tolist(),
            "std": np.std(all_mfcc_means, axis=0).tolist(),
        }

        # Energy profile
        profile["energy_profile"] = {
            "mean": float(
                np.mean([s["features"]["rms_energy_mean"] for s in self.samples])
            ),
            "std": float(
                np.std([s["features"]["rms_energy_mean"] for s in self.samples])
            ),
        }

        self.voice_profile = profile

        # Save profile
        profile_path = self.config.output_dir / "voice_profile.json"
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=2)

        logger.info(f"Voice profile saved to: {profile_path}")
        logger.info(f"  Samples: {profile['sample_count']}")
        logger.info(f"  Total duration: {profile['total_duration']:.1f}s")
        logger.info(f"  Avg quality: {profile['avg_quality_score']:.2f}")
        if "voice_pitch" in profile:
            logger.info(
                f"  Voice pitch: {profile['voice_pitch']['mean']:.1f}Hz "
                f"(range: {profile['voice_pitch']['min']:.0f}-{profile['voice_pitch']['max']:.0f}Hz)"
            )

        return profile

    def train_voice_model(self, model_type: str = "recognition") -> Dict[str, Any]:
        """
        Train voice model (placeholder for actual model training)

        Args:
            model_type: "recognition" (STT) or "cloning" (TTS)

        Returns:
            Training results and metrics
        """
        if not self.samples:
            raise ValueError("No samples available for training")

        logger.info(f"Starting {model_type} model training...")
        logger.info(f"Training data: {len(self.samples)} samples")

        # This is a placeholder for actual model training
        # In a full implementation, this would:
        # 1. Prepare training data from samples
        # 2. Load/initialize the appropriate model (Whisper for STT, TTS model for cloning)
        # 3. Fine-tune on the voice samples
        # 4. Validate performance
        # 5. Save the trained model

        results = {
            "model_type": model_type,
            "training_status": "placeholder",
            "message": (
                f"Voice model training placeholder. "
                f"Collected {len(self.samples)} samples for {model_type}. "
                f"Full training implementation requires integration with specific "
                f"TTS/STT models (e.g., Coqui TTS, Whisper fine-tuning)."
            ),
            "samples_used": len(self.samples),
            "total_duration": sum(
                s["features"]["duration_seconds"] for s in self.samples
            ),
            "avg_quality": np.mean([s["quality_score"] for s in self.samples]),
            "timestamp": datetime.now().isoformat(),
        }

        # Log training session
        self.personality.log_voice_training_session(
            session_type=f"model_training_{model_type}",
            audio_file_path=str(self.config.output_dir),
            duration_seconds=results["total_duration"],
            quality_score=results["avg_quality"],
            notes=results["message"],
        )

        logger.info(f"Training completed: {results['message']}")

        return results

    def save_training_session(self, filename: Optional[str] = None):
        """Save training session data"""
        if filename is None:
            filename = (
                f"training_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        session_path = self.config.output_dir / filename

        session_data = {
            "config": {
                "training_mode": self.config.training_mode,
                "sample_rate": self.config.sample_rate,
                "quality_threshold": self.config.quality_threshold,
            },
            "samples": self.samples,
            "voice_profile": self.voice_profile,
            "timestamp": datetime.now().isoformat(),
        }

        with open(session_path, "w") as f:
            json.dump(session_data, f, indent=2)

        logger.info(f"Training session saved to: {session_path}")

    def load_training_session(self, filename: str):
        """Load previous training session"""
        session_path = Path(filename)
        if not session_path.exists():
            session_path = self.config.output_dir / filename

        with open(session_path, "r") as f:
            session_data = json.load(f)

        self.samples = session_data.get("samples", [])
        self.voice_profile = session_data.get("voice_profile", {})

        logger.info(f"Loaded training session with {len(self.samples)} samples")


def main():
    """CLI interface for voice training"""
    parser = argparse.ArgumentParser(
        description="VEGA Voice Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add voice samples
  python voice_training.py add --file recording1.wav --text "Hello, I am training VEGA"
  python voice_training.py add --file recording2.wav --text "VEGA voice recognition"
  
  # Analyze voice profile
  python voice_training.py analyze
  
  # Train model
  python voice_training.py train --mode recognition
  python voice_training.py train --mode cloning
  python voice_training.py train --mode both
  
  # View status
  python voice_training.py status
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Add sample command
    add_parser = subparsers.add_parser("add", help="Add a voice sample")
    add_parser.add_argument("--file", "-f", required=True, help="Audio file path")
    add_parser.add_argument("--text", "-t", help="Transcription of the audio")
    add_parser.add_argument("--context", "-c", help="Context or notes about the sample")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze voice profile")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train voice model")
    train_parser.add_argument(
        "--mode",
        choices=["recognition", "cloning", "both"],
        default="both",
        help="Training mode",
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Show training status")

    # Batch add command
    batch_parser = subparsers.add_parser(
        "batch", help="Add multiple samples from directory"
    )
    batch_parser.add_argument(
        "--dir", "-d", required=True, help="Directory with audio files"
    )
    batch_parser.add_argument(
        "--pattern", "-p", default="*.wav", help="File pattern to match"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize trainer
    trainer = VoiceTrainer()

    # Try to load existing session
    session_files = list(trainer.config.output_dir.glob("training_session_*.json"))
    if session_files:
        latest_session = max(session_files, key=lambda p: p.stat().st_mtime)
        try:
            trainer.load_training_session(str(latest_session))
            logger.info(f"Loaded existing session: {latest_session.name}")
        except Exception as e:
            logger.warning(f"Could not load previous session: {e}")

    # Execute command
    if args.command == "add":
        try:
            sample = trainer.add_voice_sample(
                args.file, transcription=args.text, context=args.context
            )
            print(f"\n✅ Added voice sample #{sample['id']}")
            print(f"   Duration: {sample['features']['duration_seconds']:.1f}s")
            print(f"   Quality: {sample['quality_score']:.2f}")
            print(f"   Total samples: {len(trainer.samples)}")
            trainer.save_training_session()
        except Exception as e:
            print(f"\n❌ Error adding sample: {e}")
            sys.exit(1)

    elif args.command == "analyze":
        try:
            profile = trainer.analyze_voice_profile()
            print(f"\n✅ Voice Profile Analysis")
            print(f"   Samples: {profile['sample_count']}")
            print(f"   Total duration: {profile['total_duration']:.1f}s")
            print(f"   Average quality: {profile['avg_quality_score']:.2f}")
            if "voice_pitch" in profile:
                print(f"   Voice pitch: {profile['voice_pitch']['mean']:.1f}Hz")
            trainer.save_training_session()
        except Exception as e:
            print(f"\n❌ Error analyzing profile: {e}")
            sys.exit(1)

    elif args.command == "train":
        try:
            # First analyze if not done
            if not trainer.voice_profile:
                print("Analyzing voice profile first...")
                trainer.analyze_voice_profile()

            print(f"\nTraining voice model (mode: {args.mode})...")
            results = trainer.train_voice_model(args.mode)
            print(f"\n✅ Training completed")
            print(f"   {results['message']}")
            trainer.save_training_session()
        except Exception as e:
            print(f"\n❌ Error training model: {e}")
            sys.exit(1)

    elif args.command == "status":
        print(f"\n📊 VEGA Voice Training Status")
        print(f"\nSamples collected: {len(trainer.samples)}")
        if trainer.samples:
            total_duration = sum(
                s["features"]["duration_seconds"] for s in trainer.samples
            )
            avg_quality = np.mean([s["quality_score"] for s in trainer.samples])
            print(f"Total duration: {total_duration:.1f}s")
            print(f"Average quality: {avg_quality:.2f}")

        if trainer.voice_profile:
            print(f"\n✅ Voice profile created")
            print(f"   Profile samples: {trainer.voice_profile['sample_count']}")
            if "voice_pitch" in trainer.voice_profile:
                pitch = trainer.voice_profile["voice_pitch"]
                print(f"   Voice pitch: {pitch['mean']:.1f}Hz (±{pitch['std']:.1f}Hz)")
        else:
            print(f"\n⚠️  No voice profile yet (run 'analyze' command)")

        # Get personality stats
        personality = get_vega_personality()
        stats = personality.get_personality_stats()
        voice_stats = stats.get("voice_training", {})

        print(f"\nVoice training history:")
        print(f"   Total sessions: {voice_stats.get('sessions', 0)}")
        print(
            f"   Total training time: {voice_stats.get('total_duration_seconds', 0):.0f}s"
        )

    elif args.command == "batch":
        from pathlib import Path
        import glob

        directory = Path(args.dir)
        if not directory.exists():
            print(f"❌ Directory not found: {directory}")
            sys.exit(1)

        pattern = directory / args.pattern
        files = glob.glob(str(pattern))

        if not files:
            print(f"⚠️  No files found matching: {pattern}")
            sys.exit(1)

        print(f"\nFound {len(files)} audio files")
        successful = 0
        failed = 0

        for file_path in files:
            try:
                print(f"\nProcessing: {Path(file_path).name}")
                sample = trainer.add_voice_sample(file_path)
                print(
                    f"  ✅ Added (duration: {sample['features']['duration_seconds']:.1f}s, "
                    f"quality: {sample['quality_score']:.2f})"
                )
                successful += 1
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                failed += 1

        print(f"\n{'='*50}")
        print(f"Batch processing complete:")
        print(f"  Successful: {successful}/{len(files)}")
        print(f"  Failed: {failed}/{len(files)}")

        if successful > 0:
            trainer.save_training_session()
            print(f"  Total samples: {len(trainer.samples)}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)8s | %(message)s"
    )
    main()
