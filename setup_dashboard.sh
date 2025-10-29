#!/bin/bash

# Vega Dashboard - Quick Start Script
# This script sets up the always-running web dashboard

set -e

VEGA_DIR="/home/ncacord/Vega2.0"
SERVICE_FILE="$VEGA_DIR/systemd/vega-dashboard.service"
SYSTEMD_DIR="/etc/systemd/system"

echo "🤖 Vega Dashboard Setup"
echo "======================="
echo ""

# Check if running as root for systemd installation
if [ "$EUID" -eq 0 ]; then
    echo "✅ Running with sudo privileges"
    
    # Install the service
    echo "📦 Installing systemd service..."
    cp "$SERVICE_FILE" "$SYSTEMD_DIR/vega-dashboard.service"
    
    # Reload systemd
    echo "🔄 Reloading systemd daemon..."
    systemctl daemon-reload
    
    # Enable the service
    echo "⚡ Enabling service to start on boot..."
    systemctl enable vega-dashboard.service
    
    # Start the service
    echo "🚀 Starting Vega Dashboard..."
    systemctl start vega-dashboard.service
    
    # Show status
    echo ""
    echo "📊 Service Status:"
    systemctl status vega-dashboard.service --no-pager
    
    echo ""
    echo "✅ Vega Dashboard is now running!"
    echo "🌐 Local access: http://localhost:8080"
    echo "🌐 Network access: http://$(hostname -I | awk '{print $1}'):8080"
    echo ""
    echo "📱 Access from ANY device on your local network:"
    echo "   Laptop:  http://$(hostname -I | awk '{print $1}'):8080"
    echo "   Mobile:  http://$(hostname -I | awk '{print $1}'):8080"
    echo "   Desktop: http://$(hostname -I | awk '{print $1}'):8080"
    echo ""
    echo "📝 Useful commands:"
    echo "   sudo systemctl status vega-dashboard   # Check status"
    echo "   sudo systemctl stop vega-dashboard     # Stop dashboard"
    echo "   sudo systemctl restart vega-dashboard  # Restart dashboard"
    echo "   sudo journalctl -u vega-dashboard -f   # View logs"
    
else
    echo "❌ This script needs sudo privileges to install the systemd service"
    echo "   Run: sudo bash $0"
    echo ""
    echo "   OR run manually without systemd:"
    echo "   cd $VEGA_DIR"
    echo "   source .venv/bin/activate"
    echo "   python tools/vega/vega_dashboard.py --host 0.0.0.0 --port 8080"
fi
