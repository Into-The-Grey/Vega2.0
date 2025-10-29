# VEGA Training Quick Reference

## 🚀 Quick Commands

### Check System

```bash
# See available GPUs, CPUs, RAM
python src/vega/training/train_dynamic.py --show-resources

# Check for voice recordings
python src/vega/training/train_dynamic.py --check-voice-data
```

### Simple Training (Recommended)

```bash
# Auto-detect everything (GPU/CPU, voice/text)
./scripts/train_mixed.sh

# GPU-only, auto-detect data
./scripts/train_gpu_only.sh

# CPU-only, auto-detect data
./scripts/train_cpu_only.sh
```

### Specific Training Modes

```bash
# GPU with voice data
./scripts/train_gpu_only.sh voice

# GPU with text data
./scripts/train_gpu_only.sh text

# CPU with voice data
./scripts/train_cpu_only.sh voice

# CPU with text data
./scripts/train_cpu_only.sh text

# Mixed with voice data
./scripts/train_mixed.sh voice

# Mixed with text data
./scripts/train_mixed.sh text
```

## 📊 Training Mode Matrix

| Script | Compute | Data | Fallback | When to Use |
|--------|---------|------|----------|-------------|
| `train_mixed.sh` | GPU→CPU | Auto | Yes | **Default, most reliable** |
| `train_mixed.sh voice` | GPU→CPU | Voice | Yes | Voice training, any hardware |
| `train_mixed.sh text` | GPU→CPU | Text | Yes | Text training, any hardware |
| `train_gpu_only.sh` | GPU only | Auto | No | Maximum speed, GPU required |
| `train_gpu_only.sh voice` | GPU only | Voice | No | Fast voice training |
| `train_gpu_only.sh text` | GPU only | Text | No | Fast text training |
| `train_cpu_only.sh` | CPU only | Auto | N/A | No GPU available |
| `train_cpu_only.sh voice` | CPU only | Voice | N/A | Voice training, CPU only |
| `train_cpu_only.sh text` | CPU only | Text | N/A | Text training, CPU only |

## 🎛️ Advanced Options

### Direct Python Command

```bash
python src/vega/training/train_dynamic.py \
  --mode {gpu|cpu|mixed} \
  --voice-mode {voice|text|auto} \
  --device {cuda:0|cuda:1|cpu} \
  --min-gpu-memory 5.0 \
  --voice-data-dir recordings \
  --output-dir training/output
```

### Force Specific GPU

```bash
# Use GPU 0 (GTX 1660 SUPER)
python src/vega/training/train_dynamic.py --device cuda:0 --voice-mode voice

# Use GPU 1 (Quadro P1000)
python src/vega/training/train_dynamic.py --device cuda:1 --voice-mode voice

# Force CPU
python src/vega/training/train_dynamic.py --device cpu --voice-mode text
```

### Custom Directories

```bash
python src/vega/training/train_dynamic.py \
  --voice-data-dir /path/to/recordings \
  --output-dir /path/to/output \
  --config /path/to/config.yaml
```

### Memory Management

```bash
# Require at least 6GB GPU memory
python src/vega/training/train_dynamic.py --mode mixed --min-gpu-memory 6.0

# Require at least 4GB GPU memory (tight fit)
python src/vega/training/train_dynamic.py --mode mixed --min-gpu-memory 4.0
```

## 🎯 Common Scenarios

### Scenario 1: First Time Training

```bash
# Check your system
python src/vega/training/train_dynamic.py --show-resources

# Check if you have voice data
python src/vega/training/train_dynamic.py --check-voice-data

# Start with auto mode
./scripts/train_mixed.sh
```

### Scenario 2: Voice Training with Recordings

```bash
# 1. Record voice samples to recordings/
mkdir -p recordings/session_001
# ... record files ...

# 2. Verify recordings detected
python src/vega/training/train_dynamic.py --check-voice-data

# 3. Train on GPU (or fallback to CPU)
./scripts/train_mixed.sh voice
```

### Scenario 3: Text Training on GPU

```bash
# 1. Prepare dataset
python main.py cli dataset build ./datasets/samples

# 2. Configure training
vim training/config.yaml

# 3. Train on GPU only (maximum speed)
./scripts/train_gpu_only.sh text
```

### Scenario 4: CPU-Only System

```bash
# All training on CPU
./scripts/train_cpu_only.sh

# Or specific data type
./scripts/train_cpu_only.sh voice
./scripts/train_cpu_only.sh text
```

### Scenario 5: Multi-GPU System

```bash
# Let system choose best GPU
./scripts/train_mixed.sh voice

# Or force specific GPU
python src/vega/training/train_dynamic.py --device cuda:0 --voice-mode voice
python src/vega/training/train_dynamic.py --device cuda:1 --voice-mode text
```

## 📝 Output Files

All training sessions create:

```
training/output/
├── logs/
│   └── session_YYYYMMDD_HHMMSS.json    # Detailed session log
├── voice_model/                         # Voice training output
│   ├── model.pt
│   ├── config.json
│   └── voice_profile.json
└── checkpoint-*/                        # Text training checkpoints
    ├── config.json
    ├── pytorch_model.bin
    └── training_args.bin
```

### View Session Log

```bash
# List sessions
ls -lt training/output/logs/

# View latest session
cat training/output/logs/$(ls -t training/output/logs/ | head -1) | jq .

# Search for errors
grep -i error training/output/logs/session_*.json
```

## 🔧 Troubleshooting

### GPU Not Detected

```bash
# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check nvidia-smi
nvidia-smi

# Use CPU mode as fallback
./scripts/train_cpu_only.sh
```

### Out of Memory

```bash
# Use CPU mode
./scripts/train_cpu_only.sh

# Or reduce batch size in training/config.yaml
# per_device_train_batch_size: 2
```

### No Voice Data

```bash
# Check recordings directory
ls -lh recordings/

# Create if missing
mkdir -p recordings

# Check for files
python src/vega/training/train_dynamic.py --check-voice-data

# Use text mode instead
./scripts/train_mixed.sh text
```

### Training Too Slow

```bash
# Verify GPU is being used
nvidia-smi  # Should show python process

# Force GPU mode
./scripts/train_gpu_only.sh

# Check compute mode
python src/vega/training/train_dynamic.py --show-resources
```

## ⚡ Performance Tips

1. **GPU Training**: 10-20x faster than CPU
   - Use `train_gpu_only.sh` for maximum speed
   - Requires 5-6GB VRAM for Mistral 7B
   - Check with `nvidia-smi` during training

2. **Mixed Mode**: Best reliability
   - Tries GPU first, falls back to CPU
   - Recommended for most users
   - No manual intervention needed

3. **CPU Training**: Slower but works everywhere
   - Use when GPU unavailable
   - Requires 16-32 GB RAM
   - Can run in background for days

4. **Batch Size**: Affects speed and memory
   - Larger batch = faster but more memory
   - Reduce if out of memory
   - Edit `training/config.yaml`

5. **Voice Training**: Faster than text
   - 2-4 hours for 500 samples
   - Less memory intensive
   - Can use smaller GPU

## 📚 Related Documentation

- [Full Training Guide](TRAINING_MODES.md) - Comprehensive documentation
- [Voice Training](VOICE_TRAINING_GUIDE.md) - Recording and voice setup
- [Dataset Preparation](../docs/DATASETS.md) - Text dataset creation
- [System Setup](../docs/SETUP.md) - Initial configuration

## 🆘 Getting Help

```bash
# Help text
python src/vega/training/train_dynamic.py --help

# Show resources
python src/vega/training/train_dynamic.py --show-resources

# Check voice data
python src/vega/training/train_dynamic.py --check-voice-data

# View session logs
ls training/output/logs/
```

## ✅ Quick Decision Tree

```
Do you have a GPU?
├─ Yes → GPU available?
│  ├─ Yes → Use train_mixed.sh (safest)
│  │        Or train_gpu_only.sh (fastest)
│  └─ No → Use train_cpu_only.sh
└─ No → Use train_cpu_only.sh

Do you have voice recordings?
├─ Yes → Use voice mode
│        ./scripts/train_mixed.sh voice
├─ No → Use text mode
│       ./scripts/train_mixed.sh text
└─ Not sure → Use auto mode
              ./scripts/train_mixed.sh
```

## 🎓 Training Workflow

### Voice Training Workflow

```bash
# 1. Check system
python src/vega/training/train_dynamic.py --show-resources

# 2. Record samples
mkdir -p recordings/critical_001
# ... record 50 lines from critical_session_001.txt ...

# 3. Verify recordings
python src/vega/training/train_dynamic.py --check-voice-data

# 4. Train
./scripts/train_mixed.sh voice

# 5. Check output
ls -lh training/output/voice_model/

# 6. Review logs
cat training/output/logs/$(ls -t training/output/logs/ | head -1) | jq .
```

### Text Training Workflow

```bash
# 1. Prepare dataset
python main.py cli dataset build ./datasets/samples

# 2. Configure training
vim training/config.yaml

# 3. Check system
python src/vega/training/train_dynamic.py --show-resources

# 4. Train
./scripts/train_gpu_only.sh text

# 5. Check output
ls -lh training/output/checkpoint-*/

# 6. Review logs
cat training/output/logs/$(ls -t training/output/logs/ | head -1) | jq .
```
