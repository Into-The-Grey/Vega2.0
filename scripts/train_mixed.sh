#!/bin/bash
# Mixed Training Script with GPU Fallback
# Tries GPU first, falls back to CPU if GPU unavailable or out of memory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

echo "🔄 VEGA Mixed Training (GPU with CPU Fallback)"
echo "==============================================="
echo ""

# Check if this is voice or text training
MODE="${1:-auto}"

if [ "$MODE" = "voice" ]; then
    echo "Mode: Voice Training (Mixed)"
    python src/vega/training/train_dynamic.py --mode mixed --voice-mode voice "$@"
elif [ "$MODE" = "text" ]; then
    echo "Mode: Text Training (Mixed)"
    python src/vega/training/train_dynamic.py --mode mixed --voice-mode text "$@"
else
    echo "Mode: Auto-detect (Mixed)"
    python src/vega/training/train_dynamic.py --mode mixed --voice-mode auto "$@"
fi
