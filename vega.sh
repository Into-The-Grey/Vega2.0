#!/bin/bash
# Vega2.0 Quick Launcher - No venv required
# Usage: ./vega.sh [command]

PYTHON=/usr/bin/python3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

case "$1" in
    chat)
        shift
        $PYTHON main.py cli chat "$@"
        ;;
    repl)
        $PYTHON main.py cli repl
        ;;
    server)
        echo "🚀 Starting Vega API server..."
        $PYTHON main.py server
        ;;
    history)
        $PYTHON main.py cli history --limit 20
        ;;
    *)
        echo "Vega2.0 - Quick Commands"
        echo "========================"
        echo ""
        echo "Usage: ./vega.sh [command]"
        echo ""
        echo "Commands:"
        echo "  chat \"your message\"  - Send a single message"
        echo "  repl                 - Interactive chat session"
        echo "  server               - Start API server"
        echo "  history              - View conversation history"
        echo ""
        echo "Examples:"
        echo "  ./vega.sh chat \"What's the weather like?\""
        echo "  ./vega.sh repl"
        ;;
esac
