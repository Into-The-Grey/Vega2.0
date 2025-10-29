#!/bin/bash
# VEGA Training System - Getting Started Helper
# Run this script to see your current training setup status

echo "================================================================================"
echo "🚀 VEGA Training System - Setup Status Check"
echo "================================================================================"
echo ""

# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Check Python and venv
echo "📦 Python Environment:"
if [ -d ".venv" ]; then
    echo "  ✅ Virtual environment found: .venv/"
    if [ -f ".venv/bin/activate" ]; then
        echo "  ✅ Activation script available"
    else
        echo "  ❌ Activation script missing"
    fi
else
    echo "  ❌ Virtual environment not found"
    echo "     Run: python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
fi
echo ""

# Check training scripts
echo "🛠️  Training Scripts:"
if [ -f "src/vega/training/train_dynamic.py" ]; then
    echo "  ✅ Main training script: src/vega/training/train_dynamic.py"
else
    echo "  ❌ Main training script not found"
fi

if [ -f "scripts/training/train_mixed.sh" ] && [ -x "scripts/training/train_mixed.sh" ]; then
    echo "  ✅ Mixed mode script: scripts/training/train_mixed.sh (executable)"
else
    echo "  ❌ Mixed mode script missing or not executable"
fi

if [ -f "scripts/train_gpu_only.sh" ] && [ -x "scripts/train_gpu_only.sh" ]; then
    echo "  ✅ GPU-only script: scripts/train_gpu_only.sh (executable)"
else
    echo "  ❌ GPU-only script missing or not executable"
fi

if [ -f "scripts/train_cpu_only.sh" ] && [ -x "scripts/train_cpu_only.sh" ]; then
    echo "  ✅ CPU-only script: scripts/train_cpu_only.sh (executable)"
else
    echo "  ❌ CPU-only script missing or not executable"
fi
echo ""

# Check documentation
echo "📚 Documentation:"
if [ -f "TRAINING_QUICK_REFERENCE.md" ]; then
    echo "  ✅ Quick reference: TRAINING_QUICK_REFERENCE.md"
else
    echo "  ❌ Quick reference not found"
fi

if [ -f "docs/TRAINING_MODES.md" ]; then
    echo "  ✅ Full guide: docs/TRAINING_MODES.md"
else
    echo "  ❌ Full guide not found"
fi
echo ""

# Check for voice recordings
echo "🎤 Voice Training Data:"
if [ -d "recordings" ]; then
    COUNT=$(find recordings -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" -o -name "*.ogg" -o -name "*.m4a" \) 2>/dev/null | wc -l)
    if [ "$COUNT" -gt 0 ]; then
        echo "  ✅ Recordings directory: recordings/"
        echo "  ✅ Found $COUNT audio files"
    else
        echo "  ⚠️  Recordings directory exists but empty"
        echo "     Place .wav, .mp3, .flac, .ogg, or .m4a files in recordings/"
    fi
else
    echo "  ❌ No recordings directory"
    echo "     Create: mkdir -p recordings"
fi
echo ""

# Check for text training data
echo "📝 Text Training Data:"
if [ -f "datasets/output.jsonl" ]; then
    LINES=$(wc -l < datasets/output.jsonl)
    echo "  ✅ Dataset ready: datasets/output.jsonl ($LINES lines)"
else
    echo "  ⚠️  No compiled dataset found"
    echo "     Run: python main.py cli dataset build ./datasets/samples"
fi

if [ -f "training/config.yaml" ]; then
    echo "  ✅ Training config: training/config.yaml"
else
    echo "  ❌ Training config not found"
fi
echo ""

# Check system resources (if script available)
echo "💻 System Resources:"
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$GPU_COUNT" -gt 0 ]; then
        echo "  ✅ NVIDIA GPU(s) detected: $GPU_COUNT"
        nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader 2>/dev/null | while IFS=',' read -r idx name free total; do
            echo "     GPU $idx: $name"
            echo "        Memory: $free /$total"
        done
    else
        echo "  ⚠️  nvidia-smi available but no GPUs detected"
    fi
else
    echo "  ⚠️  nvidia-smi not found (CPU-only mode)"
fi

# CPU info
CPU_CORES=$(nproc 2>/dev/null || echo "unknown")
echo "  ✅ CPU cores: $CPU_CORES"

# RAM info
if command -v free &> /dev/null; then
    RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
    echo "  ✅ RAM: ${RAM_GB}GB"
fi
echo ""

# Recommendations
echo "================================================================================"
echo "💡 Quick Start Recommendations"
echo "================================================================================"
echo ""

# Check if ready for voice training
if [ -d "recordings" ] && [ "$(find recordings -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" -o -name "*.ogg" -o -name "*.m4a" \) 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "✅ You're ready for VOICE TRAINING!"
    echo ""
    echo "   Recommended command:"
    echo "   ./scripts/training/train_mixed.sh voice"
    echo ""
    echo "   This will:"
    echo "   - Try to use GPU for fastest training"
    echo "   - Fall back to CPU if GPU unavailable"
    echo "   - Process all voice files in recordings/"
    echo "   - Save trained model to training/output/voice_model/"
    echo ""
elif [ -f "datasets/output.jsonl" ]; then
    echo "✅ You're ready for TEXT TRAINING!"
    echo ""
    echo "   Recommended command:"
    echo "   ./scripts/train_mixed.sh text"
    echo ""
    echo "   This will:"
    echo "   - Try to use GPU for fastest training"
    echo "   - Fall back to CPU if GPU unavailable"
    echo "   - Train on datasets/output.jsonl"
    echo "   - Save checkpoints to training/output/"
    echo ""
else
    echo "⚠️  Setup incomplete - Choose your training type:"
    echo ""
    echo "   FOR VOICE TRAINING:"
    echo "   1. Create recordings directory: mkdir -p recordings"
    echo "   2. Record voice samples (.wav, .mp3, etc.)"
    echo "   3. Run: ./scripts/train_mixed.sh voice"
    echo ""
    echo "   FOR TEXT TRAINING:"
    echo "   1. Place .txt/.md/.json files in datasets/samples/"
    echo "   2. Build dataset: python main.py cli dataset build ./datasets/samples"
    echo "   3. Run: ./scripts/train_mixed.sh text"
    echo ""
fi

# Helpful commands
echo "📋 Helpful Commands:"
echo ""
echo "   Check system resources:"
echo "   python src/vega/training/train_dynamic.py --show-resources"
echo ""
echo "   Check for voice data:"
echo "   python src/vega/training/train_dynamic.py --check-voice-data"
echo ""
echo "   View quick reference:"
echo "   cat TRAINING_QUICK_REFERENCE.md"
echo ""
echo "   View full documentation:"
echo "   cat docs/TRAINING_MODES.md"
echo ""

echo "================================================================================"
