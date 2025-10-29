# VEGA Training Scripts

This directory contains convenient wrapper scripts for the dynamic training system.

## Quick Start

```bash
# Check system status
../check_training_setup.sh

# Run training (auto-detect everything)
./training/train_mixed.sh
```

## Available Scripts

### `train_mixed.sh` ⭐ Recommended

Tries GPU first, falls back to CPU if unavailable or out of memory.

```bash
# Auto-detect data type
./training/train_mixed.sh

# Voice training
./training/train_mixed.sh voice

# Text training
./training/train_mixed.sh text
```

**Use when:**

- You want the most reliable option
- GPU availability is uncertain
- Running alongside other GPU tasks
- First time training

### `train_gpu_only.sh`

Forces GPU-only training, fails if GPU unavailable.

```bash
# Auto-detect data type
./train_gpu_only.sh

# Voice training
./train_gpu_only.sh voice

# Text training
./train_gpu_only.sh text
```

**Use when:**

- Maximum speed required
- GPU definitely available
- Willing to fail fast if GPU issues

### `train_cpu_only.sh`

Forces CPU-only training, never uses GPU.

```bash
# Auto-detect data type
./train_cpu_only.sh

# Voice training
./train_cpu_only.sh voice

# Text training
./train_cpu_only.sh text
```

**Use when:**

- No GPU available
- GPU reserved for other tasks
- Long-running background training
- Testing on limited hardware

## Arguments

All scripts accept the same arguments:

- `voice` - Force voice training mode
- `text` - Force text training mode
- (no argument) - Auto-detect data type

Additional arguments are passed through to the underlying Python script.

## Examples

```bash
# Simple usage
./train_mixed.sh                    # Auto everything
./train_gpu_only.sh voice          # GPU voice training
./train_cpu_only.sh text           # CPU text training

# With custom options (pass-through)
./train_mixed.sh voice --output-dir /custom/path
./train_gpu_only.sh --voice-data-dir /recordings/session1
```

## Output

All scripts produce:

1. **Console output** - Real-time progress and status
2. **Session log** - JSON file in `training/output/logs/session_YYYYMMDD_HHMMSS.json`
3. **Model files** - Trained models in `training/output/`

## Troubleshooting

### Script won't run

```bash
# Make executable
chmod +x train_*.sh

# Check permissions
ls -l train_*.sh
```

### GPU not detected

```bash
# Check GPU availability
nvidia-smi

# Use CPU mode
./train_cpu_only.sh
```

### No voice/text data found

```bash
# Check for voice data
python ../src/vega/training/train_dynamic.py --check-voice-data

# For voice: create recordings/
mkdir -p ../recordings

# For text: build dataset
python ../main.py cli dataset build ../datasets/samples
```

## Documentation

- **Quick Reference:** `../TRAINING_QUICK_REFERENCE.md`
- **Full Guide:** `../docs/TRAINING_MODES.md`
- **System Summary:** `../TRAINING_SYSTEM_SUMMARY.md`

## Advanced Usage

For full control, use the Python script directly:

```bash
python ../src/vega/training/train_dynamic.py --help

# Example: Force specific GPU
python ../src/vega/training/train_dynamic.py \
  --device cuda:0 \
  --voice-mode voice \
  --output-dir training/output
```

## System Check

Before training, check your setup:

```bash
# Run system check
../check_training_setup.sh

# Or check resources directly
python ../src/vega/training/train_dynamic.py --show-resources
```
