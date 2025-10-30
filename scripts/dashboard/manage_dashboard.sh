#!/bin/bash

# Vega Dashboard Manager
# Quick management script for the web dashboard

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

function show_help() {
    cat <<EOF
🤖 Vega Dashboard Manager

Usage: $0 [command]

Commands:
    start       Start dashboard manually (foreground)
    background  Start dashboard in background
    stop        Stop all dashboard instances
    restart     Restart the dashboard service
    status      Show dashboard status
    logs        View dashboard logs (real-time)
    health      Quick health check
    install     Install as systemd service (requires sudo)
    uninstall   Remove systemd service (requires sudo)
    open        Open dashboard in default browser
    url         Show network access URLs
    help        Show this help message

Examples:
    $0 start                    # Start manually for testing
    $0 background               # Start in background
    sudo $0 install             # Install as always-running service
    $0 status                   # Check if running
    $0 logs                     # View real-time logs
    $0 open                     # Open in browser

Dashboard URL: http://localhost:8080
Documentation: docs/WEB_DASHBOARD.md
Quick Start: DASHBOARD_QUICK_START.md

EOF
}

function start_dashboard() {
    echo "🚀 Starting Vega Dashboard..."
    
    # Check if virtual environment exists
    if [ ! -d ".venv" ]; then
        echo "❌ Virtual environment not found. Run: python3 -m venv .venv"
        exit 1
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Start dashboard
    echo "📊 Dashboard starting on http://localhost:8080"
    python tools/vega/vega_dashboard.py --host 0.0.0.0 --port 8080
}

function start_background() {
    echo "🚀 Starting Vega Dashboard in background..."
    
    if [ ! -d ".venv" ]; then
        echo "❌ Virtual environment not found. Run: python3 -m venv .venv"
        exit 1
    fi
    
    source .venv/bin/activate
    nohup python tools/vega/vega_dashboard.py --host 0.0.0.0 --port 8080 > /tmp/vega-dashboard.log 2>&1 &
    
    DASH_PID=$!
    echo "✅ Dashboard started with PID: $DASH_PID"
    echo "📊 Access at: http://localhost:8080"
    echo "📝 Logs: tail -f /tmp/vega-dashboard.log"
}

function stop_dashboard() {
    echo "⏹️  Stopping Vega Dashboard..."
    
    # Kill all vega_dashboard.py processes
    pkill -f "vega_dashboard.py"
    
    if [ $? -eq 0 ]; then
        echo "✅ Dashboard stopped"
    else
        echo "ℹ️  No dashboard processes found"
    fi
}

function restart_service() {
    if systemctl is-active --quiet vega-dashboard; then
        echo "🔄 Restarting Vega Dashboard service..."
        sudo systemctl restart vega-dashboard
        echo "✅ Service restarted"
        show_status
    else
        echo "❌ Service not installed or not running"
        echo "   Install with: sudo $0 install"
    fi
}

function show_status() {
    echo "📊 Vega Dashboard Status"
    echo "========================"
    echo ""
    
    # Check if running as service
    if systemctl list-unit-files | grep -q "vega-dashboard.service"; then
        echo "🔧 Service Status:"
        sudo systemctl status vega-dashboard --no-pager -l
        echo ""
    fi
    
    # Check if running manually
    if pgrep -f "vega_dashboard.py" > /dev/null; then
        echo "✅ Dashboard is running (manual mode)"
        echo ""
        echo "Processes:"
        ps aux | grep vega_dashboard.py | grep -v grep
        echo ""
    else
        echo "⏹️  No manual dashboard processes found"
        echo ""
    fi
    
    # Check port
    echo "🔌 Port Status:"
    if sudo lsof -i :8080 2>/dev/null | grep -q LISTEN; then
        sudo lsof -i :8080 | grep LISTEN
        echo ""
        echo "✅ Dashboard accessible at: http://localhost:8080"
    else
        echo "❌ Port 8080 not in use (dashboard not running)"
    fi
    echo ""
    
    # Check API
    echo "🌐 API Check:"
    if curl -s -f http://localhost:8080/api/status > /dev/null 2>&1; then
        echo "✅ API responding correctly"
        echo ""
        echo "Quick stats:"
        curl -s http://localhost:8080/api/status | python -m json.tool | grep -E "is_running|cpu_usage|memory_usage|system_health"
    else
        echo "❌ API not responding"
    fi
}

function show_logs() {
    echo "📝 Vega Dashboard Logs (Ctrl+C to exit)"
    echo "========================================"
    echo ""
    
    if systemctl is-active --quiet vega-dashboard; then
        sudo journalctl -u vega-dashboard -f
    elif [ -f /tmp/vega-dashboard.log ]; then
        tail -f /tmp/vega-dashboard.log
    else
        echo "❌ No logs found"
        echo "   Service logs: sudo journalctl -u vega-dashboard -f"
        echo "   Manual logs: tail -f /tmp/vega-dashboard.log"
    fi
}

function health_check() {
    echo "🏥 Vega Dashboard Health Check"
    echo "=============================="
    echo ""
    
    ERRORS=0
    
    # Check virtual environment
    echo -n "Virtual environment... "
    if [ -d ".venv" ]; then
        echo "✅"
    else
        echo "❌ Missing"
        ERRORS=$((ERRORS + 1))
    fi
    
    # Check dependencies
    echo -n "Dependencies... "
    source .venv/bin/activate 2>/dev/null
    if pip list 2>/dev/null | grep -q fastapi && \
       pip list 2>/dev/null | grep -q uvicorn && \
       pip list 2>/dev/null | grep -q websockets; then
        echo "✅"
    else
        echo "❌ Missing"
        ERRORS=$((ERRORS + 1))
    fi
    
    # Check dashboard file
    echo -n "Dashboard file... "
    if [ -f "tools/vega/vega_dashboard.py" ]; then
        echo "✅"
    else
        echo "❌ Missing"
        ERRORS=$((ERRORS + 1))
    fi
    
    # Check if running
    echo -n "Dashboard running... "
    if pgrep -f "vega_dashboard.py" > /dev/null || systemctl is-active --quiet vega-dashboard; then
        echo "✅"
    else
        echo "⚠️  Not running"
    fi
    
    # Check port
    echo -n "Port 8080... "
    if sudo lsof -i :8080 2>/dev/null | grep -q LISTEN; then
        echo "✅"
    else
        echo "⚠️  Not listening"
    fi
    
    # Check API
    echo -n "API responding... "
    if curl -s -f http://localhost:8080/api/status > /dev/null 2>&1; then
        echo "✅"
    else
        echo "⚠️  Not responding"
    fi
    
    # Check service installation
    echo -n "Systemd service... "
    if systemctl list-unit-files | grep -q "vega-dashboard.service"; then
        echo "✅ Installed"
    else
        echo "⚠️  Not installed (run: sudo $0 install)"
    fi
    
    echo ""
    if [ $ERRORS -eq 0 ]; then
        echo "✅ All critical checks passed!"
    else
        echo "⚠️  $ERRORS critical issue(s) found"
        echo "   See documentation: docs/WEB_DASHBOARD.md"
    fi
    
    echo ""
    echo "Dashboard URL: http://localhost:8080"
}

function install_service() {
    if [ "$EUID" -ne 0 ]; then
        echo "❌ Installation requires sudo privileges"
        echo "   Run: sudo $0 install"
        exit 1
    fi
    
    echo "📦 Installing Vega Dashboard as systemd service..."
    echo ""
    
    # Copy service file
    cp systemd/vega-dashboard.service /etc/systemd/system/
    echo "✅ Service file copied"
    
    # Reload systemd
    systemctl daemon-reload
    echo "✅ Systemd reloaded"
    
    # Enable service
    systemctl enable vega-dashboard.service
    echo "✅ Service enabled (auto-start on boot)"
    
    # Start service
    systemctl start vega-dashboard.service
    echo "✅ Service started"
    
    echo ""
    echo "🎉 Installation complete!"
    echo ""
    echo "Dashboard is now running at: http://localhost:8080"
    echo ""
    echo "Useful commands:"
    echo "  $0 status      # Check status"
    echo "  $0 logs        # View logs"
    echo "  $0 restart     # Restart service"
    echo "  $0 open        # Open in browser"
}

function uninstall_service() {
    if [ "$EUID" -ne 0 ]; then
        echo "❌ Uninstallation requires sudo privileges"
        echo "   Run: sudo $0 uninstall"
        exit 1
    fi
    
    echo "🗑️  Uninstalling Vega Dashboard service..."
    echo ""
    
    # Stop service
    systemctl stop vega-dashboard.service 2>/dev/null
    echo "✅ Service stopped"
    
    # Disable service
    systemctl disable vega-dashboard.service 2>/dev/null
    echo "✅ Service disabled"
    
    # Remove service file
    rm -f /etc/systemd/system/vega-dashboard.service
    echo "✅ Service file removed"
    
    # Reload systemd
    systemctl daemon-reload
    echo "✅ Systemd reloaded"
    
    echo ""
    echo "✅ Uninstallation complete!"
    echo ""
    echo "You can still run the dashboard manually with: $0 start"
}

function open_browser() {
    echo "🌐 Opening dashboard in browser..."
    
    # Try different browsers
    if command -v xdg-open &> /dev/null; then
        xdg-open http://localhost:8080
    elif command -v gnome-open &> /dev/null; then
        gnome-open http://localhost:8080
    elif command -v firefox &> /dev/null; then
        firefox http://localhost:8080 &
    elif command -v chromium-browser &> /dev/null; then
        chromium-browser http://localhost:8080 &
    elif command -v google-chrome &> /dev/null; then
        google-chrome http://localhost:8080 &
    else
        echo "❌ Could not detect browser"
        echo "   Open manually: http://localhost:8080"
    fi
}

function show_network_access() {
    SERVER_IP=$(hostname -I | awk '{print $1}')
    
    cat <<EOF

╔══════════════════════════════════════════════════════════════════════╗
║               🌐 VEGA DASHBOARD - NETWORK ACCESS                     ║
╚══════════════════════════════════════════════════════════════════════╝

📡 Dashboard accessible from ANY device on your local network!

🖥️  SERVER IP: $SERVER_IP
🔌 PORT: 8080

════════════════════════════════════════════════════════════════════════

📱 ACCESS URLS:
════════════════════════════════════════════════════════════════════════

From Laptop:     http://$SERVER_IP:8080
From Mobile:     http://$SERVER_IP:8080
From Desktop:    http://$SERVER_IP:8080
From Server:     http://localhost:8080

════════════════════════════════════════════════════════════════════════

📋 COPY THIS URL: http://$SERVER_IP:8080

════════════════════════════════════════════════════════════════════════

💡 TIPS:
  ✅ Bookmark on each device for easy access
  ✅ Add to mobile home screen for app-like experience
  ✅ Port 8080 - no conflict with your website (80/443)
  ✅ Local network only - no router changes needed

📖 Full guide: NETWORK_ACCESS_GUIDE.md

════════════════════════════════════════════════════════════════════════

EOF
}

# Main command handling
case "${1:-help}" in
    start)
        start_dashboard
        ;;
    background|bg)
        start_background
        ;;
    stop)
        stop_dashboard
        ;;
    restart)
        restart_service
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    health|check)
        health_check
        ;;
    install)
        install_service
        ;;
    uninstall|remove)
        uninstall_service
        ;;
    open)
        open_browser
        ;;
    url|urls|access|network)
        show_network_access
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
