#!/bin/bash

# Show Dashboard Access URLs
# Quick script to display how to access the dashboard from any device

SERVER_IP=$(hostname -I | awk '{print $1}')

cat <<EOF

╔══════════════════════════════════════════════════════════════════════╗
║               🌐 VEGA DASHBOARD - NETWORK ACCESS                     ║
╚══════════════════════════════════════════════════════════════════════╝

📡 Your Dashboard is accessible from ANY device on your local network!

🖥️  SERVER IP: $SERVER_IP
🔌 PORT: 8080

════════════════════════════════════════════════════════════════════════

📱 ACCESS FROM ANY DEVICE:
════════════════════════════════════════════════════════════════════════

From Laptop:     http://$SERVER_IP:8080
From Mobile:     http://$SERVER_IP:8080
From Desktop:    http://$SERVER_IP:8080
From Server:     http://localhost:8080

════════════════════════════════════════════════════════════════════════

📋 COPY THIS URL:
════════════════════════════════════════════════════════════════════════

http://$SERVER_IP:8080

════════════════════════════════════════════════════════════════════════

💡 TIPS:
════════════════════════════════════════════════════════════════════════

✅ Make sure all devices are on the same WiFi/network
✅ Bookmark the URL on each device for easy access
✅ On mobile: Add to home screen for app-like experience
✅ Port 8080 doesn't conflict with your website (80/443)
✅ No router configuration needed - local network only

════════════════════════════════════════════════════════════════════════

🔍 TROUBLESHOOTING:
════════════════════════════════════════════════════════════════════════

Can't access? Run these commands:

  Check status:        ./manage_dashboard.sh status
  Check if listening:  sudo lsof -i :8080
  Test locally:        curl http://localhost:8080/api/status
  View logs:           sudo journalctl -u vega-dashboard -n 20

════════════════════════════════════════════════════════════════════════

📖 FULL GUIDE: See NETWORK_ACCESS_GUIDE.md for detailed instructions

════════════════════════════════════════════════════════════════════════

EOF
