# Vega Training System Guide

This comprehensive guide covers all aspects of training in the Vega system, including voice training, text training, and system management.

## Quick Start

### Simple Training Commands

```bash
# Auto-detect everything (GPU/CPU, voice/text)
./scripts/training/train_mixed.sh

# GPU-only, auto-detect data
./scripts/training/train_gpu_only.sh

# CPU-only, auto-detect data
./scripts/training/train_cpu_only.sh
```

### Check System Resources

```bash
# See available GPUs, CPUs, RAM
python src/vega/training/train_dynamic.py --show-resources

# Check for voice recordings
python src/vega/training/train_dynamic.py --check-voice-data

# Verify training setup
./scripts/training/check_training_setup.sh
```

## Training System Architecture

### Core Components

- **Main Training Script**: `src/vega/training/train_dynamic.py` (650+ lines)
- **Convenience Scripts**: `scripts/training/` directory
- **Voice Dataset**: 20,040 professional voice lines across 60 categories
- **Configuration**: YAML-based training configurations

### Training Modes

1. **Compute Modes:**
   - `--mode gpu`: GPU-only, fails if unavailable
   - `--mode cpu`: CPU-only, never uses GPU
   - `--mode mixed`: Try GPU, fallback to CPU (default)

2. **Data Modes:**
   - `--voice-mode voice`: Train on voice recordings
   - `--voice-mode text`: Train on text datasets
   - `--voice-mode auto`: Automatically detect data type (default)

3. **Smart Features:**
   - Automatic GPU detection and selection
   - Memory requirement validation
   - Multi-GPU support (selects best available)
   - Session logging with JSON output
   - Progress tracking and error handling

## Voice Training

### Dataset Overview

Your voice training dataset contains **20,040 professionally-crafted lines** across **60 diverse categories** designed to train VEGA to recognize and synthesize your voice with maximum expressiveness and accuracy.

### Dataset Statistics

- **Total Lines**: 20,040
- **Categories**: 60 (from everyday speech to technical jargon)
- **Emotions**: 27 different emotional states
- **Text Length**: 3-509 characters (avg: 64 chars)
- **Balance**: 334 lines per category (uniform distribution)

### Voice Categories

#### Everyday Communication

- `everyday` - Common phrases and responses
- `agreements` / `denials_corrections` - Yes/no responses
- `greetings` - Hello, goodbye, etc.
- `hesitations_fillers` - Um, uh, like, you know

#### Technical & Professional

- `technical` - Technical terminology
- `code_readouts` - Code snippets and syntax
- `paths_cli` - File paths and terminal commands
- `error_messages` - Error handling responses
- `finance` / `legalese` / `medical` - Domain-specific vocabulary

#### Emotional Expression

- `excitement` / `enthusiasm` - High-energy responses
- `calm_reassurance` - Soothing, comforting tones
- `urgency` / `warnings` - Alert, important messages
- `confusion` / `clarification` - Questioning, uncertain responses
- `satisfaction` / `appreciation` - Positive feedback

#### Creative & Narrative

- `storytelling` - Narrative and descriptive speech
- `instructions` - Step-by-step guidance
- `explanations` - Educational content
- `creative_writing` - Imaginative, expressive text

### Voice Training Commands

#### Basic Voice Training

```bash
# Start voice training (auto-detect GPU/CPU)
python src/vega/training/train_dynamic.py --voice-mode voice

# Force GPU for voice training
python src/vega/training/train_dynamic.py --voice-mode voice --mode gpu

# CPU-only voice training
python src/vega/training/train_dynamic.py --voice-mode voice --mode cpu
```

#### Advanced Voice Training

```bash
# Custom voice training with specific parameters
python src/vega/training/train_dynamic.py \
    --voice-mode voice \
    --mode mixed \
    --epochs 10 \
    --batch-size 8 \
    --learning-rate 0.0001
```

#### Voice Data Management

```bash
# Check voice recordings status
python src/vega/training/train_dynamic.py --check-voice-data

# List available voice categories
find datasets/voice_training_prioritized -name "*.wav" | head -20

# Check voice dataset statistics
python -c "
import os
voice_dir = 'datasets/voice_training_prioritized'
categories = os.listdir(voice_dir) if os.path.exists(voice_dir) else []
print(f'Voice categories available: {len(categories)}')
for cat in sorted(categories)[:10]:
    files = len([f for f in os.listdir(os.path.join(voice_dir, cat)) if f.endswith('.wav')])
    print(f'  {cat}: {files} files')
"
```

## Text Training

### Dataset Types

- **Conversation Datasets**: Question-answer pairs
- **Knowledge Bases**: Factual information
- **Code Repositories**: Programming examples
- **Documentation**: Technical guides and references

### Text Training Commands

```bash
# Auto-detect text datasets and train
python src/vega/training/train_dynamic.py --voice-mode text

# Train on specific dataset
python src/vega/training/train_dynamic.py \
    --voice-mode text \
    --dataset-path datasets/samples/custom_dataset.jsonl

# Advanced text training
python src/vega/training/train_dynamic.py \
    --voice-mode text \
    --mode gpu \
    --epochs 5 \
    --batch-size 4 \
    --gradient-accumulation-steps 4
```

### Dataset Preparation

```bash
# Create training dataset from text files
python main.py cli dataset build ./datasets/samples

# Validate dataset format
python -c "
import json
with open('datasets/output.jsonl', 'r') as f:
    lines = f.readlines()
    print(f'Dataset contains {len(lines)} training examples')
    sample = json.loads(lines[0])
    print(f'Sample format: {list(sample.keys())}')
"
```

## Advanced Training Features

### Multi-GPU Training

```bash
# Automatic multi-GPU detection
python src/vega/training/train_dynamic.py --mode gpu --multi-gpu

# Specify GPU devices
CUDA_VISIBLE_DEVICES=0,1 python src/vega/training/train_dynamic.py --mode gpu
```

### Memory Optimization

```bash
# Enable gradient checkpointing for large models
python src/vega/training/train_dynamic.py \
    --gradient-checkpointing \
    --batch-size 2 \
    --gradient-accumulation-steps 8

# Mixed precision training
python src/vega/training/train_dynamic.py --fp16
```

### Training Monitoring

```bash
# View training progress
tail -f logs/training_session_$(date +%Y%m%d).json

# Monitor GPU usage during training
watch -n 1 nvidia-smi

# Check training logs
ls -la logs/training_*
```

## Configuration Management

### Training Configuration Files

- **Main Config**: `config/training.yaml`
- **Model Configs**: `config/models/`
- **Dataset Configs**: `config/datasets.yaml`

### Example Configuration

```yaml
# config/training.yaml
training:
  batch_size: 4
  learning_rate: 0.0001
  epochs: 10
  gradient_accumulation_steps: 4
  warmup_steps: 1000
  
model:
  name: "microsoft/DialoGPT-medium"
  max_length: 512
  
voice:
  sample_rate: 22050
  duration_limit: 10.0
  
logging:
  log_level: "INFO"
  save_steps: 500
  eval_steps: 1000
```

### Modifying Configuration

```bash
# Edit training configuration
nano config/training.yaml

# Validate configuration
python -c "
import yaml
with open('config/training.yaml', 'r') as f:
    config = yaml.safe_load(f)
    print('Configuration loaded successfully')
    print(f'Training epochs: {config[\"training\"][\"epochs\"]}')
"
```

## Troubleshooting

### Common Issues

#### Out of Memory Errors

```bash
# Reduce batch size
python src/vega/training/train_dynamic.py --batch-size 1 --gradient-accumulation-steps 8

# Enable CPU offloading
python src/vega/training/train_dynamic.py --cpu-offload

# Use CPU training as fallback
python src/vega/training/train_dynamic.py --mode cpu
```

#### CUDA Errors

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache(); print('CUDA cache cleared')"

# Force CPU mode if CUDA issues persist
python src/vega/training/train_dynamic.py --mode cpu
```

#### Voice Data Issues

```bash
# Check voice data integrity
python -c "
import os
import wave
voice_dir = 'datasets/voice_training_prioritized'
issues = []
for root, dirs, files in os.walk(voice_dir):
    for file in files:
        if file.endswith('.wav'):
            try:
                with wave.open(os.path.join(root, file), 'rb') as w:
                    frames = w.getnframes()
                    if frames == 0:
                        issues.append(file)
            except:
                issues.append(file)
print(f'Voice files with issues: {len(issues)}')
"
```

### Performance Optimization

#### GPU Optimization

```bash
# Check GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Optimize for specific GPU
python src/vega/training/train_dynamic.py \
    --batch-size 8 \
    --dataloader-num-workers 4 \
    --pin-memory

# Enable TensorFloat-32 for RTX GPUs
python src/vega/training/train_dynamic.py --tf32
```

#### CPU Optimization

```bash
# Multi-threaded CPU training
OMP_NUM_THREADS=8 python src/vega/training/train_dynamic.py --mode cpu

# Optimize data loading
python src/vega/training/train_dynamic.py \
    --mode cpu \
    --dataloader-num-workers 8 \
    --dataloader-pin-memory false
```

## Training Scripts Reference

### Convenience Scripts

All training scripts are located in `scripts/training/`:

#### `train_mixed.sh`

- Auto-detects GPU/CPU capabilities
- Falls back gracefully if GPU unavailable
- Automatically detects voice vs text data

#### `train_gpu_only.sh`

- Forces GPU-only training
- Fails fast if GPU unavailable
- Optimal for dedicated GPU machines

#### `train_cpu_only.sh`

- Forces CPU-only training
- Never attempts GPU usage
- Good for CPU-only environments

#### `check_training_setup.sh`

- Validates training environment
- Checks dependencies and data availability
- Reports system capabilities

### Advanced Training Options

```bash
# Full parameter training
python src/vega/training/train_dynamic.py \
    --mode mixed \
    --voice-mode auto \
    --epochs 10 \
    --batch-size 4 \
    --learning-rate 0.0001 \
    --warmup-steps 1000 \
    --gradient-accumulation-steps 4 \
    --gradient-checkpointing \
    --fp16 \
    --dataloader-num-workers 4 \
    --save-steps 500 \
    --eval-steps 1000 \
    --logging-steps 100 \
    --output-dir training/output \
    --run-name "vega-training-$(date +%Y%m%d-%H%M%S)"
```

## Session Management

### Training Sessions

Training sessions are automatically logged with:

- Timestamp and duration
- System configuration used
- Training parameters
- Performance metrics
- Error logs and warnings

### Session Files

```bash
# View recent training sessions
ls -la logs/training_session_*.json

# Analyze training session
python -c "
import json
import glob
sessions = sorted(glob.glob('logs/training_session_*.json'))
if sessions:
    with open(sessions[-1], 'r') as f:
        session = json.load(f)
    print(f'Last session: {session[\"timestamp\"]}')
    print(f'Duration: {session.get(\"duration\", \"unknown\")}')
    print(f'Status: {session.get(\"status\", \"unknown\")}')
"
```

### Resuming Training

```bash
# Resume from checkpoint
python src/vega/training/train_dynamic.py \
    --resume-from-checkpoint training/output/checkpoint-1000

# Auto-resume latest checkpoint
python src/vega/training/train_dynamic.py --auto-resume
```

## Integration with Main System

### API Integration

The training system integrates with the main Vega API:

```bash
# Start training via API
curl -X POST http://localhost:8000/training/start \
    -H "Content-Type: application/json" \
    -d '{"mode": "mixed", "voice_mode": "auto"}'

# Check training status
curl http://localhost:8000/training/status

# Stop training
curl -X POST http://localhost:8000/training/stop
```

### CLI Integration

```bash
# Training via main CLI
python main.py cli train --config config/training.yaml

# Quick training command
python main.py train --voice --gpu
```

## Best Practices

### Training Preparation

1. **Verify System Resources**: Check GPU memory, CPU cores, and RAM
2. **Validate Data**: Ensure voice files are valid and text datasets are formatted correctly
3. **Test Configuration**: Start with small batch sizes and short epochs
4. **Monitor Progress**: Use logging and monitoring tools during training

### Production Training

1. **Use Mixed Mode**: Allows graceful fallback from GPU to CPU
2. **Save Regularly**: Configure frequent checkpoint saves
3. **Monitor Memory**: Watch for OOM errors and adjust batch sizes
4. **Log Everything**: Enable comprehensive logging for debugging

### Voice Training Specific

1. **Quality Control**: Ensure consistent audio quality across recordings
2. **Balanced Data**: Verify even distribution across voice categories
3. **Test Samples**: Validate a subset before full training
4. **Incremental Training**: Start with smaller subsets and expand

## Performance Benchmarks

### Typical Training Times

- **Voice Training (20k samples)**:
  - GPU (RTX 4090): ~2-4 hours
  - GPU (RTX 3080): ~4-6 hours
  - CPU (16 cores): ~12-24 hours

- **Text Training (100k samples)**:
  - GPU (RTX 4090): ~1-2 hours
  - GPU (RTX 3080): ~2-4 hours
  - CPU (16 cores): ~8-16 hours

### Memory Requirements

- **Minimum**: 8GB RAM, 4GB VRAM
- **Recommended**: 32GB RAM, 12GB VRAM
- **Optimal**: 64GB RAM, 24GB VRAM

## Files Reference

### Training System Files

- **Core Script**: `src/vega/training/train_dynamic.py` (650+ lines)
- **Training Scripts**: `scripts/training/` directory
- **Configuration**: `config/training.yaml`
- **Voice Data**: `datasets/voice_training_prioritized/` (20,040 files)
- **Logs**: `logs/training_session_*.json`

### Supporting Files

- **Setup Validation**: `scripts/training/check_training_setup.sh`
- **Model Configurations**: `config/models/`
- **Dataset Tools**: `src/vega/datasets/`

---

**Last Updated**: October 29, 2025
**Version**: 2.0
**Maintainer**: Vega Development Team
