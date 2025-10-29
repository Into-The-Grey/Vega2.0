# VEGA Dynamic Training System

Comprehensive training system with flexible GPU/CPU and Voice/Text modes.

## Table of Contents

- [Overview](#overview)
- [Training Modes](#training-modes)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Voice Training](#voice-training)
- [Text Training](#text-training)
- [Resource Management](#resource-management)
- [Troubleshooting](#troubleshooting)

## Overview

The VEGA dynamic training system provides:

✅ **3 Compute Modes:**

- **GPU-only**: Maximum performance, requires sufficient VRAM
- **CPU-only**: Compatibility mode, works anywhere
- **Mixed**: Smart fallback from GPU to CPU when needed

✅ **3 Training Modes:**

- **Voice**: Train on recorded voice samples for TTS/STT
- **Text**: Train on text datasets for language model
- **Auto**: Automatically detect and use available data

✅ **Smart Features:**

- Automatic resource detection
- Memory requirement validation
- Progress tracking and logging
- Session management

## Training Modes

### GPU-Only Mode

**When to use:**

- You have sufficient GPU VRAM (6GB+ for Mistral 7B)
- Maximum training speed required
- Willing to fail if GPU unavailable

**Command:**

```bash
# Auto-detect data type
./scripts/train_gpu_only.sh

# Force voice training
./scripts/train_gpu_only.sh voice

# Force text training
./scripts/train_gpu_only.sh text
```

**Characteristics:**

- ⚡ Fastest training speed (10-20x faster than CPU)
- 🎮 Requires NVIDIA GPU with CUDA support
- 💾 ~5-6GB VRAM for Mistral 7B
- ❌ Fails immediately if GPU unavailable or out of memory

### CPU-Only Mode

**When to use:**

- No GPU available
- GPU memory insufficient
- Testing on limited hardware
- Long-running background training

**Command:**

```bash
# Auto-detect data type
./scripts/train_cpu_only.sh

# Force voice training
./scripts/train_cpu_only.sh voice

# Force text training
./scripts/train_cpu_only.sh text
```

**Characteristics:**

- 💻 Works on any system with sufficient RAM
- 🐌 Slower training (10-20x slower than GPU)
- 💾 ~8-16GB RAM recommended
- ✅ Most compatible option

### Mixed Mode (Recommended)

**When to use:**

- Default choice for most users
- GPU available but memory uncertain
- Want automatic optimization
- Running other GPU tasks simultaneously

**Command:**

```bash
# Auto-detect data type
./scripts/train_mixed.sh

# Force voice training
./scripts/train_mixed.sh voice

# Force text training
./scripts/train_mixed.sh text
```

**Characteristics:**

- 🔄 Tries GPU first, falls back to CPU if needed
- 🎯 Checks memory availability before allocation
- ⚡ Fast when GPU available, reliable always
- 🛡️ Most robust option

## Quick Start

### 1. Check Your Resources

```bash
# See available GPUs, CPUs, memory
python src/vega/training/train_dynamic.py --show-resources
```

Expected output:

```
================================================================================
Available Compute Resources
================================================================================

GPU 0: NVIDIA GeForce GTX 1660 SUPER
  Memory: 5.2 GB free / 6.0 GB total
  Compute: 7.5

GPU 1: Quadro P1000
  Memory: 3.8 GB free / 4.0 GB total
  Compute: 6.1

CPU: 12 cores
  RAM: 125.71 GB
```

### 2. Check Voice Data (if doing voice training)

```bash
# Check for voice recordings
python src/vega/training/train_dynamic.py --check-voice-data
```

Expected output:

```
================================================================================
Voice Training Data Check
================================================================================

✅ Voice data found in: recordings

Statistics:
  Total files: 150
  Total size: 245.67 MB
  Formats:
    .wav: 150 files
```

### 3. Start Training

**Option A: Automatic (Recommended)**

```bash
# Detects GPU availability and data type automatically
./scripts/train_mixed.sh
```

**Option B: Specific Mode**

```bash
# GPU voice training
./scripts/train_gpu_only.sh voice

# CPU text training
./scripts/train_cpu_only.sh text

# Mixed auto-detect
./scripts/train_mixed.sh
```

## Detailed Usage

### Direct Python Script Usage

The core script is `src/vega/training/train_dynamic.py`. Shell scripts are convenience wrappers.

```bash
# Full control with all options
python src/vega/training/train_dynamic.py \
  --mode mixed \
  --voice-mode auto \
  --min-gpu-memory 5.0 \
  --voice-data-dir recordings \
  --output-dir training/output
```

### Command-Line Options

```
--mode {gpu,cpu,mixed}
    Training mode selection
    - gpu: GPU-only, fails if unavailable
    - cpu: CPU-only, never uses GPU
    - mixed: Try GPU, fallback to CPU (default)

--voice-mode {voice,text,auto}
    Training data mode
    - voice: Use voice recordings
    - text: Use text datasets
    - auto: Automatically detect (default)

--device DEVICE
    Force specific device (overrides --mode)
    Examples: 'cuda:0', 'cuda:1', 'cpu'

--min-gpu-memory GB
    Minimum GPU memory required (default: 5.0)
    Adjust for different model sizes

--voice-data-dir DIR
    Directory containing voice recordings (default: recordings/)

--config PATH
    Training config file (default: training/config.yaml)

--output-dir DIR
    Output directory for models (default: training/output/)

--show-resources
    Display available resources and exit

--check-voice-data
    Check for voice data and exit
```

## Voice Training

### Prerequisites

1. **Record Voice Samples**

   ```bash
   # Create recordings directory
   mkdir -p recordings/session_001
   
   # Record using your preferred tool (Audacity, etc.)
   # Save as WAV, MP3, FLAC, OGG, or M4A
   ```

2. **Supported Formats**
   - WAV (recommended, highest quality)
   - MP3 (good compression)
   - FLAC (lossless)
   - OGG (open format)
   - M4A (Apple format)

3. **Recommended Recording Settings**
   - Sample rate: 44.1 kHz or 48 kHz
   - Bit depth: 16-bit or 24-bit
   - Channels: Mono (stereo will be converted)
   - Environment: Quiet, minimal echo
   - Duration: 2-30 seconds per clip

### Voice Training Process

The system will:

1. **Scan** for voice files in `recordings/` directory
2. **Validate** audio quality (duration, format, clarity)
3. **Extract** features (MFCC, pitch, timbre, energy)
4. **Analyze** voice profile (pitch range, speaking style)
5. **Train** voice model (TTS/STT integration)
6. **Save** model to `training/output/voice_model/`

### Voice Training Example

```bash
# Check if you have voice data
python src/vega/training/train_dynamic.py --check-voice-data

# Train on GPU with voice data
./scripts/train_gpu_only.sh voice

# Or use direct command
python src/vega/training/train_dynamic.py \
  --mode gpu \
  --voice-mode voice \
  --voice-data-dir recordings \
  --output-dir training/output
```

### Voice Training Output

```
================================================================================
VEGA Dynamic Training Session
Session ID: 20251029_143022
================================================================================

ℹ️  [2025-10-29T14:30:22] Starting voice training
ℹ️  [2025-10-29T14:30:22] Device: cuda:1
ℹ️  [2025-10-29T14:30:22] Voice data directory: recordings
ℹ️  [2025-10-29T14:30:22] Found 150 voice files (245.67 MB)
ℹ️  [2025-10-29T14:30:22] Formats: {'.wav': 150}
ℹ️  [2025-10-29T14:30:23] Processing 150 voice samples...
ℹ️  [2025-10-29T14:30:25] Processed 10/150 samples...
...
✅ [2025-10-29T14:32:15] Voice sample processing complete: 150 successful, 0 failed
ℹ️  [2025-10-29T14:32:15] Analyzing voice profile...
✅ [2025-10-29T14:32:16] Voice profile analysis:
ℹ️  [2025-10-29T14:32:16]   Total samples: 150
ℹ️  [2025-10-29T14:32:16]   Total duration: 450.23s
ℹ️  [2025-10-29T14:32:16]   Avg pitch: 142.50 Hz
...
```

## Text Training

### Prerequisites

1. **Prepare Dataset**

   ```bash
   # Place text files in datasets/samples/
   cp your_data.txt datasets/samples/
   
   # Build JSONL dataset
   python main.py cli dataset build ./datasets/samples
   ```

2. **Configure Training**
   Edit `training/config.yaml`:

   ```yaml
   model_name: "mistralai/Mistral-7B-v0.1"
   output_dir: "./training/output"
   num_train_epochs: 3
   per_device_train_batch_size: 4
   learning_rate: 2e-5
   ```

### Text Training Example

```bash
# Check resources
python src/vega/training/train_dynamic.py --show-resources

# Train on GPU with text data
./scripts/train_gpu_only.sh text

# Or use direct command
python src/vega/training/train_dynamic.py \
  --mode gpu \
  --voice-mode text \
  --config training/config.yaml \
  --output-dir training/output
```

### Text Training Output

The system uses HuggingFace Transformers for text training:

```
================================================================================
VEGA Dynamic Training Session
Session ID: 20251029_150045
================================================================================

ℹ️  [2025-10-29T15:00:45] Starting text-based training
ℹ️  [2025-10-29T15:00:45] Device: cuda:0
ℹ️  [2025-10-29T15:00:45] Mode: gpu
ℹ️  [2025-10-29T15:00:46] Loaded training module
ℹ️  [2025-10-29T15:00:47] Starting HuggingFace training pipeline...
...
[HuggingFace training output]
...
✅ [2025-10-29T16:45:22] Training completed successfully
```

## Resource Management

### Memory Requirements

**Voice Training:**

- CPU: 8-16 GB RAM
- GPU: 2-4 GB VRAM (feature extraction + small models)
- Disk: ~1 GB per hour of audio

**Text Training (Mistral 7B):**

- CPU: 16-32 GB RAM
- GPU: 6-8 GB VRAM (Q4 quantized), 14-16 GB (full precision)
- Disk: 4-8 GB for model checkpoints

### GPU Memory Management

```bash
# Check GPU memory before training
nvidia-smi

# Set minimum GPU memory requirement
python src/vega/training/train_dynamic.py \
  --mode mixed \
  --min-gpu-memory 6.0  # Require 6GB free

# Force specific GPU
python src/vega/training/train_dynamic.py \
  --device cuda:1  # Use second GPU
```

### Multi-GPU Systems

The system automatically selects the best GPU:

1. Scans all available GPUs
2. Checks free memory on each
3. Selects first GPU with sufficient memory
4. Falls back to CPU if none suitable

Manual GPU selection:

```bash
# Use GPU 0
python src/vega/training/train_dynamic.py --device cuda:0

# Use GPU 1
python src/vega/training/train_dynamic.py --device cuda:1

# Force CPU
python src/vega/training/train_dynamic.py --device cpu
```

## Session Logging

Every training session creates a detailed log:

```
training/output/logs/session_YYYYMMDD_HHMMSS.json
```

Log contents:

```json
{
  "session_id": "20251029_143022",
  "mode": "gpu",
  "voice_mode": "voice",
  "device": "cuda:1",
  "log": [
    {
      "timestamp": "2025-10-29T14:30:22.123456",
      "level": "INFO",
      "message": "Starting voice training"
    },
    ...
  ]
}
```

View logs:

```bash
# Latest session
ls -t training/output/logs/ | head -1 | xargs -I {} cat training/output/logs/{}

# Pretty print with jq
cat training/output/logs/session_20251029_143022.json | jq .
```

## Troubleshooting

### GPU Not Detected

**Symptom:** System falls back to CPU despite having GPU

**Solutions:**

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU visibility
nvidia-smi

# Reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory

**Symptom:** CUDA out of memory error

**Solutions:**

```bash
# Use CPU mode
./scripts/train_cpu_only.sh

# Reduce batch size in training/config.yaml
per_device_train_batch_size: 2  # Instead of 4

# Use smaller model or more aggressive quantization

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"
```

### No Voice Data Found

**Symptom:** Voice mode requested but no data detected

**Solutions:**

```bash
# Check voice data location
python src/vega/training/train_dynamic.py --check-voice-data

# Ensure recordings in correct directory
ls recordings/

# Check file formats
ls recordings/*.wav recordings/*.mp3

# Create recordings directory
mkdir -p recordings
```

### Training Very Slow

**Symptom:** Training taking much longer than expected

**Solutions:**

```bash
# Verify GPU is being used
nvidia-smi  # Should show python process

# Check if accidentally using CPU
# Look for "Selected device: cpu" in output

# Force GPU mode
./scripts/train_gpu_only.sh

# Check CPU usage (should be <50% if GPU working)
top
```

### Import Errors

**Symptom:** Module import failures

**Solutions:**

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install audio processing dependencies
pip install librosa soundfile torchaudio

# Verify imports
python -c "import torch, librosa, soundfile; print('OK')"
```

## Advanced Usage

### Custom Training Configuration

Create custom config file:

```yaml
# custom_config.yaml
model_name: "mistralai/Mistral-7B-v0.1"
output_dir: "./custom_output"
num_train_epochs: 5
per_device_train_batch_size: 2
gradient_accumulation_steps: 2
learning_rate: 1e-5
warmup_steps: 100
logging_steps: 10
save_steps: 500
```

Use custom config:

```bash
python src/vega/training/train_dynamic.py \
  --config custom_config.yaml \
  --output-dir custom_output
```

### Combining Voice and Text Training

```bash
# Train voice model first
./scripts/train_mixed.sh voice

# Then train text model
./scripts/train_mixed.sh text

# Or use auto mode to train both if data available
./scripts/train_mixed.sh auto
```

### Continuous Training

```bash
# Loop until interrupted
while true; do
  ./scripts/train_mixed.sh auto
  echo "Session complete, waiting 60s..."
  sleep 60
done
```

### Background Training

```bash
# Run in background
nohup ./scripts/train_mixed.sh > training.log 2>&1 &

# Check progress
tail -f training.log

# Stop training
pkill -f train_dynamic.py
```

## Performance Benchmarks

Approximate training speeds (Mistral 7B):

| Hardware | Mode | Tokens/sec | Epoch Time (1K samples) |
|----------|------|------------|------------------------|
| RTX 4090 | GPU | ~180 | 15 min |
| RTX 3080 | GPU | ~120 | 22 min |
| GTX 1660 SUPER | GPU | ~50 | 55 min |
| Ryzen 9 5950X | CPU | ~8 | 6 hours |
| Intel i7-12700K | CPU | ~6 | 8 hours |

Voice training is typically faster (2-4 hours for 500 samples).

## Next Steps

1. ✅ Set up training environment
2. ✅ Check resources with `--show-resources`
3. 📝 Record voice samples (for voice training)
4. 📝 Prepare text dataset (for text training)
5. ▶️ Start training with appropriate script
6. 📊 Monitor progress in session logs
7. ✅ Verify trained model output

## Related Documentation

- [Voice Training Guide](VOICE_TRAINING_GUIDE.md) - Recording and voice training details
- [Dataset Preparation](../docs/DATASETS.md) - Text dataset creation
- [Training Config](config.yaml) - HuggingFace training parameters
- [Voice Training Workflow](VOICE_TRAINING_WORKFLOW.md) - Step-by-step voice training
