#!/bin/bash
# CPU-Only Training Script
# Forces training to use CPU exclusively

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

echo "💻 VEGA CPU-Only Training"
echo "========================="
echo ""

# Check if this is voice or text training
MODE="${1:-auto}"

if [ "$MODE" = "voice" ]; then
    echo "Mode: Voice Training (CPU)"
    python src/vega/training/train_dynamic.py --mode cpu --voice-mode voice "$@"
elif [ "$MODE" = "text" ]; then
    echo "Mode: Text Training (CPU)"
    python src/vega/training/train_dynamic.py --mode cpu --voice-mode text "$@"
else
    echo "Mode: Auto-detect (CPU)"
    python src/vega/training/train_dynamic.py --mode cpu --voice-mode auto "$@"
fi
