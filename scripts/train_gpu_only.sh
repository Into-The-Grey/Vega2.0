#!/bin/bash
# GPU-Only Training Script
# Forces training to use GPU exclusively, fails if GPU unavailable

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

echo "🎮 VEGA GPU-Only Training"
echo "========================="
echo ""

# Check if this is voice or text training
MODE="${1:-auto}"

if [ "$MODE" = "voice" ]; then
    echo "Mode: Voice Training (GPU)"
    python src/vega/training/train_dynamic.py --mode gpu --voice-mode voice "$@"
elif [ "$MODE" = "text" ]; then
    echo "Mode: Text Training (GPU)"
    python src/vega/training/train_dynamic.py --mode gpu --voice-mode text "$@"
else
    echo "Mode: Auto-detect (GPU)"
    python src/vega/training/train_dynamic.py --mode gpu --voice-mode auto "$@"
fi
