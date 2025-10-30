#!/bin/bash
# VEGA PERSISTENT MODE - Quick Setup Script

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════"
echo "  🤖 VEGA PERSISTENT MODE - Quick Setup"
echo "════════════════════════════════════════════════════════════"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ This script must be run with sudo"
    echo "   Usage: sudo ./setup_persistent_mode.sh"
    exit 1
fi

# Get the actual user (not root)
ACTUAL_USER="${SUDO_USER:-$USER}"
ACTUAL_HOME=$(eval echo "~$ACTUAL_USER")

echo "✓ Running as: $ACTUAL_USER"
echo "✓ Project dir: $SCRIPT_DIR"
echo ""

# Step 1: Copy Vega server service
echo "📝 Step 1: Installing Vega server systemd service..."
if [ -f "systemd/vega-server.service" ]; then
    cp systemd/vega-server.service /etc/systemd/system/
    echo "✓ Service file copied to /etc/systemd/system/"
else
    echo "❌ systemd/vega-server.service not found!"
    exit 1
fi

# Step 2: Reload systemd
echo ""
echo "🔄 Step 2: Reloading systemd..."
systemctl daemon-reload
echo "✓ Systemd reloaded"

# Step 3: Enable auto-start
echo ""
echo "⚙️  Step 3: Enabling auto-start on boot..."
systemctl enable vega-server
systemctl enable vega-dashboard  # Make sure web UI is also enabled
echo "✓ Auto-start enabled"

# Step 4: Start services
echo ""
echo "🚀 Step 4: Starting services..."

# Stop if already running
systemctl stop vega-server 2>/dev/null || true
sleep 2

# Start Vega server
systemctl start vega-server
sleep 5

# Check if started successfully
if systemctl is-active --quiet vega-server; then
    echo "✓ Vega server started"
else
    echo "⚠️  Vega server failed to start"
    echo "   Check logs: sudo journalctl -u vega-server -n 50"
    exit 1
fi

# Restart web UI to pick up changes
systemctl restart vega-dashboard
sleep 2

if systemctl is-active --quiet vega-dashboard; then
    echo "✓ Web UI started"
else
    echo "⚠️  Web UI failed to start"
    echo "   Check logs: sudo journalctl -u vega-dashboard -n 50"
fi

# Step 5: Test connection
echo ""
echo "🔍 Step 5: Testing connection..."
sleep 3

if curl -s http://localhost:8000/healthz > /dev/null 2>&1; then
    echo "✓ Vega server responding on port 8000"
else
    echo "⚠️  Vega server not responding"
    echo "   Check logs: sudo journalctl -u vega-server -n 50"
fi

if curl -s http://localhost:8080 > /dev/null 2>&1; then
    echo "✓ Web UI responding on port 8080"
else
    echo "⚠️  Web UI not responding"
    echo "   Check logs: sudo journalctl -u vega-dashboard -n 50"
fi

# Step 6: Display status
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✅ PERSISTENT MODE SETUP COMPLETE!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "📊 Service Status:"
echo "   Vega Server:  $(systemctl is-active vega-server)"
echo "   Web UI:       $(systemctl is-active vega-dashboard)"
echo ""
echo "🌐 Access URLs:"
echo "   Local:   http://localhost:8080"
echo "   Network: http://$(hostname -I | awk '{print $1}'):8080"
echo ""
echo "📋 Useful Commands:"
echo "   View logs:        sudo journalctl -u vega-server -f"
echo "   Check status:     sudo systemctl status vega-server"
echo "   Restart:          sudo systemctl restart vega-server"
echo "   Stop:             sudo systemctl stop vega-server"
echo ""
echo "📖 Full Documentation:"
echo "   docs/PERSISTENT_MODE_GUIDE.md"
echo ""
echo "🤖 VEGA is now running in Persistent Mode!"
echo "   - Always available (100% uptime)"
echo "   - Never forgets (persistent memory)"
echo "   - Auto-restarts on failure"
echo "   - Proactive memory management"
echo ""
echo "════════════════════════════════════════════════════════════"
