#!/usr/bin/env python3
"""
VEGA Dynamic Training Script
Flexible training system supporting:
- GPU-only, CPU-only, or Mixed (GPU with CPU fallback) modes
- Voice and non-voice training modes
- Automatic resource detection and optimization
- Progress tracking and session management
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Literal
import json
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  Warning: PyTorch not available. CPU-only mode will be limited.")

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  Warning: psutil not available. Resource monitoring disabled.")


class ResourceDetector:
    """Detect and validate available compute resources"""

    def __init__(self):
        self.has_cuda = False
        self.cuda_devices = []
        self.cpu_cores = 0
        self.ram_gb = 0

        self._detect_resources()

    def _detect_resources(self):
        """Detect available hardware resources"""
        # CPU detection
        if PSUTIL_AVAILABLE:
            self.cpu_cores = psutil.cpu_count(logical=False) or 1
            self.ram_gb = psutil.virtual_memory().total / (1024**3)
        else:
            self.cpu_cores = os.cpu_count() or 1
            self.ram_gb = 0

        # GPU detection
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.has_cuda = True
            self.cuda_devices = list(range(torch.cuda.device_count()))

    def get_available_devices(self) -> list:
        """Get list of available compute devices"""
        devices = []
        if self.has_cuda:
            for i in self.cuda_devices:
                props = torch.cuda.get_device_properties(i)
                devices.append(
                    {
                        "type": "cuda",
                        "id": i,
                        "name": props.name,
                        "memory_gb": props.total_memory / (1024**3),
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                )

        devices.append(
            {
                "type": "cpu",
                "id": 0,
                "name": "CPU",
                "cores": self.cpu_cores,
                "memory_gb": self.ram_gb,
            }
        )

        return devices

    def get_device_memory(self, device_id: int = 0) -> tuple[float, float]:
        """Get free and total memory for GPU device (in GB)"""
        if not self.has_cuda or device_id >= len(self.cuda_devices):
            return (0, 0)

        torch.cuda.set_device(device_id)
        free = torch.cuda.mem_get_info()[0] / (1024**3)
        total = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
        return (free, total)

    def select_best_device(self, mode: str, min_memory_gb: float = 5.0) -> str:
        """
        Select best device based on mode and memory requirements

        Args:
            mode: 'gpu', 'cpu', or 'mixed'
            min_memory_gb: Minimum GPU memory required (default 5GB for Mistral)

        Returns:
            Device string ('cuda:0', 'cuda:1', 'cpu')
        """
        if mode == "cpu":
            return "cpu"

        if mode == "gpu" or mode == "mixed":
            if not self.has_cuda:
                if mode == "gpu":
                    raise RuntimeError(
                        "GPU mode requested but no CUDA devices available"
                    )
                print("⚠️  No CUDA devices found, falling back to CPU")
                return "cpu"

            # Find GPU with sufficient memory
            for device_id in self.cuda_devices:
                free, total = self.get_device_memory(device_id)
                if free >= min_memory_gb:
                    return f"cuda:{device_id}"

            if mode == "gpu":
                raise RuntimeError(
                    f"GPU mode requested but no device has {min_memory_gb}GB free"
                )

            print(f"⚠️  No GPU with {min_memory_gb}GB free, falling back to CPU")
            return "cpu"

        return "cpu"


class VoiceDataDetector:
    """Detect and validate voice training data"""

    def __init__(self, voice_data_dir: Optional[Path] = None):
        self.voice_data_dir = voice_data_dir or Path("recordings")
        self.supported_formats = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]

    def has_voice_data(self) -> bool:
        """Check if any voice data is available"""
        if not self.voice_data_dir.exists():
            return False

        for fmt in self.supported_formats:
            if list(self.voice_data_dir.rglob(f"*{fmt}")):
                return True

        return False

    def get_voice_files(self) -> list[Path]:
        """Get list of all voice training files"""
        files = []
        if not self.voice_data_dir.exists():
            return files

        for fmt in self.supported_formats:
            files.extend(self.voice_data_dir.rglob(f"*{fmt}"))

        return sorted(files)

    def get_stats(self) -> dict:
        """Get statistics about available voice data"""
        files = self.get_voice_files()

        if not files:
            return {"total_files": 0, "total_size_mb": 0, "formats": {}}

        stats = {"total_files": len(files), "total_size_mb": 0, "formats": {}}

        for file in files:
            size_mb = file.stat().st_size / (1024**2)
            stats["total_size_mb"] += size_mb

            ext = file.suffix.lower()
            stats["formats"][ext] = stats["formats"].get(ext, 0) + 1

        return stats


class TrainingSession:
    """Manage training session configuration and execution"""

    def __init__(
        self,
        mode: Literal["gpu", "cpu", "mixed"],
        voice_mode: Literal["voice", "text", "auto"],
        device: str,
        config_path: Optional[Path] = None,
        voice_data_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        self.mode = mode
        self.voice_mode = voice_mode
        self.device = device
        self.config_path = config_path or Path("training/config.yaml")
        self.voice_data_dir = voice_data_dir or Path("recordings")
        self.output_dir = output_dir or Path("training/output")

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log = []

    def log(self, message: str, level: str = "INFO"):
        """Log message to session"""
        timestamp = datetime.now().isoformat()
        log_entry = {"timestamp": timestamp, "level": level, "message": message}
        self.session_log.append(log_entry)

        # Print to console
        prefix = {"INFO": "ℹ️ ", "SUCCESS": "✅", "WARNING": "⚠️ ", "ERROR": "❌"}.get(
            level, ""
        )

        print(f"{prefix} [{timestamp}] {message}")

    def save_session_log(self):
        """Save session log to file"""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"session_{self.session_id}.json"
        with open(log_file, "w") as f:
            json.dump(
                {
                    "session_id": self.session_id,
                    "mode": self.mode,
                    "voice_mode": self.voice_mode,
                    "device": self.device,
                    "log": self.session_log,
                },
                f,
                indent=2,
            )

        return log_file

    def run_text_training(self):
        """Run text-based model training"""
        self.log("Starting text-based training")
        self.log(f"Device: {self.device}")
        self.log(f"Mode: {self.mode}")

        # Check for training config
        if not self.config_path.exists():
            self.log(f"Training config not found: {self.config_path}", "ERROR")
            return False

        # Import training module
        try:
            from src.vega.training.train import main as train_main

            self.log("Loaded training module")
        except ImportError as e:
            self.log(f"Failed to import training module: {e}", "ERROR")
            return False

        # Run training
        try:
            self.log("Starting HuggingFace training pipeline...")
            # TODO: Pass device configuration to training module
            # This would require updating train.py to accept device parameter
            train_main()
            self.log("Training completed successfully", "SUCCESS")
            return True
        except Exception as e:
            self.log(f"Training failed: {e}", "ERROR")
            return False

    def run_voice_training(self):
        """Run voice-based model training"""
        self.log("Starting voice training")
        self.log(f"Device: {self.device}")
        self.log(f"Voice data directory: {self.voice_data_dir}")

        # Detect voice data
        detector = VoiceDataDetector(self.voice_data_dir)
        stats = detector.get_stats()

        if stats["total_files"] == 0:
            self.log("No voice data found", "ERROR")
            return False

        self.log(
            f"Found {stats['total_files']} voice files ({stats['total_size_mb']:.2f} MB)"
        )
        self.log(f"Formats: {stats['formats']}")

        # Import voice training module
        try:
            from src.vega.training.voice_training import (
                VoiceTrainer,
                VoiceTrainingConfig,
            )

            self.log("Loaded voice training module")
        except ImportError as e:
            self.log(f"Failed to import voice training module: {e}", "ERROR")
            return False

        # Setup voice training
        try:
            config = VoiceTrainingConfig()
            trainer = VoiceTrainer(config)

            # Add voice samples
            voice_files = detector.get_voice_files()
            self.log(f"Processing {len(voice_files)} voice samples...")

            successful = 0
            failed = 0

            for file in voice_files:
                try:
                    session_id = trainer.add_voice_sample(str(file), "training")
                    successful += 1

                    if successful % 10 == 0:
                        self.log(
                            f"Processed {successful}/{len(voice_files)} samples..."
                        )
                except Exception as e:
                    self.log(f"Failed to process {file.name}: {e}", "WARNING")
                    failed += 1

            self.log(
                f"Voice sample processing complete: {successful} successful, {failed} failed"
            )

            # Analyze voice profile
            self.log("Analyzing voice profile...")
            profile = trainer.analyze_voice_profile()

            if profile:
                self.log("Voice profile analysis:", "SUCCESS")
                self.log(f"  Total samples: {profile['total_samples']}")
                self.log(f"  Total duration: {profile['total_duration']:.2f}s")

                if "average_features" in profile:
                    features = profile["average_features"]
                    self.log(f"  Avg pitch: {features.get('pitch_mean', 0):.2f} Hz")

            # Train voice model
            self.log("Training voice model...")
            # TODO: Implement actual voice model training (TTS/STT)
            # This is a placeholder for future Coqui TTS/Whisper integration
            model_path = trainer.train_voice_model(self.output_dir / "voice_model")

            if model_path:
                self.log(f"Voice model saved to: {model_path}", "SUCCESS")
                return True
            else:
                self.log("Voice model training not yet implemented", "WARNING")
                self.log(
                    "Voice features extracted and analyzed successfully", "SUCCESS"
                )
                return True

        except Exception as e:
            self.log(f"Voice training failed: {e}", "ERROR")
            return False

    def run(self):
        """Run training session"""
        self.log("=" * 80)
        self.log("VEGA Dynamic Training Session")
        self.log(f"Session ID: {self.session_id}")
        self.log("=" * 80)

        start_time = time.time()
        success = False

        try:
            if self.voice_mode == "voice":
                success = self.run_voice_training()
            elif self.voice_mode == "text":
                success = self.run_text_training()
            elif self.voice_mode == "auto":
                # Auto-detect based on available data
                detector = VoiceDataDetector(self.voice_data_dir)
                if detector.has_voice_data():
                    self.log("Auto-detected voice data, starting voice training")
                    success = self.run_voice_training()
                else:
                    self.log("No voice data found, starting text training")
                    success = self.run_text_training()
        except KeyboardInterrupt:
            self.log("Training interrupted by user", "WARNING")
            success = False
        except Exception as e:
            self.log(f"Training session failed: {e}", "ERROR")
            success = False

        # Calculate duration
        duration = time.time() - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)

        self.log("=" * 80)
        if success:
            self.log("Training session completed successfully", "SUCCESS")
        else:
            self.log("Training session failed or incomplete", "ERROR")

        self.log(f"Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
        self.log("=" * 80)

        # Save session log
        log_file = self.save_session_log()
        self.log(f"Session log saved to: {log_file}")

        return success


def main():
    parser = argparse.ArgumentParser(
        description="VEGA Dynamic Training Script - Flexible training with GPU/CPU and Voice/Text modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GPU-only training with voice data
  python train_dynamic.py --mode gpu --voice-mode voice
  
  # CPU-only training with text data
  python train_dynamic.py --mode cpu --voice-mode text
  
  # Mixed mode with automatic voice detection
  python train_dynamic.py --mode mixed --voice-mode auto
  
  # Show available resources
  python train_dynamic.py --show-resources
  
  # Check for voice data
  python train_dynamic.py --check-voice-data
        """,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["gpu", "cpu", "mixed"],
        default="mixed",
        help="Training mode: gpu (GPU-only), cpu (CPU-only), mixed (GPU with CPU fallback)",
    )

    parser.add_argument(
        "--voice-mode",
        choices=["voice", "text", "auto"],
        default="auto",
        help="Training data mode: voice (voice training), text (text training), auto (detect automatically)",
    )

    # Resource options
    parser.add_argument(
        "--device",
        type=str,
        help="Force specific device (e.g., 'cuda:0', 'cuda:1', 'cpu'). Overrides --mode selection",
    )

    parser.add_argument(
        "--min-gpu-memory",
        type=float,
        default=5.0,
        help="Minimum GPU memory required in GB (default: 5.0 for Mistral 7B)",
    )

    # Data options
    parser.add_argument(
        "--voice-data-dir",
        type=Path,
        default=Path("recordings"),
        help="Directory containing voice training data (default: recordings/)",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("training/config.yaml"),
        help="Path to training config file (default: training/config.yaml)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training/output"),
        help="Output directory for trained models (default: training/output/)",
    )

    # Utility options
    parser.add_argument(
        "--show-resources",
        action="store_true",
        help="Show available compute resources and exit",
    )

    parser.add_argument(
        "--check-voice-data",
        action="store_true",
        help="Check for voice training data and exit",
    )

    args = parser.parse_args()

    # Show resources
    if args.show_resources:
        print("\n" + "=" * 80)
        print("Available Compute Resources")
        print("=" * 80 + "\n")

        detector = ResourceDetector()
        devices = detector.get_available_devices()

        for device in devices:
            if device["type"] == "cuda":
                free, total = detector.get_device_memory(device["id"])
                print(f"GPU {device['id']}: {device['name']}")
                print(f"  Memory: {free:.2f} GB free / {total:.2f} GB total")
                print(f"  Compute: {device['compute_capability']}")
            else:
                print(f"CPU: {device['cores']} cores")
                if device["memory_gb"] > 0:
                    print(f"  RAM: {device['memory_gb']:.2f} GB")
            print()

        return 0

    # Check voice data
    if args.check_voice_data:
        print("\n" + "=" * 80)
        print("Voice Training Data Check")
        print("=" * 80 + "\n")

        detector = VoiceDataDetector(args.voice_data_dir)

        if detector.has_voice_data():
            stats = detector.get_stats()
            print(f"✅ Voice data found in: {args.voice_data_dir}")
            print(f"\nStatistics:")
            print(f"  Total files: {stats['total_files']}")
            print(f"  Total size: {stats['total_size_mb']:.2f} MB")
            print(f"  Formats:")
            for fmt, count in stats["formats"].items():
                print(f"    {fmt}: {count} files")
        else:
            print(f"❌ No voice data found in: {args.voice_data_dir}")
            print(f"\nSupported formats: {', '.join(detector.supported_formats)}")

        print()
        return 0

    # Initialize resource detector
    detector = ResourceDetector()

    # Select device
    if args.device:
        device = args.device
        print(f"Using forced device: {device}")
    else:
        device = detector.select_best_device(args.mode, args.min_gpu_memory)
        print(f"Selected device: {device} (mode: {args.mode})")

    # Check voice mode
    if args.voice_mode == "voice":
        voice_detector = VoiceDataDetector(args.voice_data_dir)
        if not voice_detector.has_voice_data():
            print(
                f"\n❌ Error: Voice mode requested but no voice data found in {args.voice_data_dir}"
            )
            print("Use --check-voice-data to see supported formats")
            return 1

    # Create and run training session
    session = TrainingSession(
        mode=args.mode,
        voice_mode=args.voice_mode,
        device=device,
        config_path=args.config,
        voice_data_dir=args.voice_data_dir,
        output_dir=args.output_dir,
    )

    success = session.run()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
