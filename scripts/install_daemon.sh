#!/bin/bash
# Vega Daemon Installation Script
# Installs and configures Vega to run as a system service with auto-management

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Vega Daemon Installation${NC}"
echo -e "${GREEN}================================================${NC}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root (use sudo)${NC}"
    exit 1
fi

# Get actual user (not root)
ACTUAL_USER=${SUDO_USER:-$(whoami)}
ACTUAL_HOME=$(eval echo "~$ACTUAL_USER")

# Get Vega directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VEGA_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${YELLOW}Configuration:${NC}"
echo "  User: $ACTUAL_USER"
echo "  Home: $ACTUAL_HOME"
echo "  Vega Dir: $VEGA_DIR"
echo ""

# Create necessary directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p "$ACTUAL_HOME/.vega"
mkdir -p "$ACTUAL_HOME/.vega/temp"
chown -R "$ACTUAL_USER":"$ACTUAL_USER" "$ACTUAL_HOME/.vega"

# Install Python dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
if [ -d "$VEGA_DIR/.venv" ]; then
    sudo -u "$ACTUAL_USER" "$VEGA_DIR/.venv/bin/pip" install schedule psutil
else
    echo -e "${RED}Virtual environment not found. Please run: python3.12 -m venv .venv${NC}"
    exit 1
fi

# Configure systemd services
echo -e "${YELLOW}Configuring systemd services...${NC}"

# Vega main service
SERVICE_FILE="$VEGA_DIR/systemd/vega.service"
if [ ! -f "$SERVICE_FILE" ]; then
    echo -e "${YELLOW}Creating vega.service...${NC}"
    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Vega AI Platform Server
Documentation=https://github.com/Into-The-Grey/Vega2.0
After=network.target

[Service]
Type=simple
User=$ACTUAL_USER
Group=$ACTUAL_USER
WorkingDirectory=$VEGA_DIR
Environment="PATH=$VEGA_DIR/.venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONPATH=$VEGA_DIR"
ExecStart=$VEGA_DIR/.venv/bin/python main.py server --host 127.0.0.1 --port 8000
Restart=always
RestartSec=10
StandardOutput=append:$ACTUAL_HOME/vega_server.log
StandardError=append:$ACTUAL_HOME/vega_server_error.log

[Install]
WantedBy=multi-user.target
EOF
fi

# Vega daemon service
DAEMON_SERVICE="$VEGA_DIR/systemd/vega-daemon.service"
sed -e "s|%USER%|$ACTUAL_USER|g" \
    -e "s|%VEGA_DIR%|$VEGA_DIR|g" \
    -e "s|%HOME_DIR%|$ACTUAL_HOME|g" \
    "$DAEMON_SERVICE" > /tmp/vega-daemon.service

# Install services
echo -e "${YELLOW}Installing systemd services...${NC}"
cp "$SERVICE_FILE" /etc/systemd/system/vega.service
cp /tmp/vega-daemon.service /etc/systemd/system/vega-daemon.service

# Configure sudoers for daemon
echo -e "${YELLOW}Configuring sudo permissions...${NC}"
SUDOERS_FILE="/etc/sudoers.d/vega-daemon"
cat > "$SUDOERS_FILE" << EOF
# Allow Vega daemon to manage system
$ACTUAL_USER ALL=(ALL) NOPASSWD: /bin/systemctl start vega
$ACTUAL_USER ALL=(ALL) NOPASSWD: /bin/systemctl stop vega
$ACTUAL_USER ALL=(ALL) NOPASSWD: /bin/systemctl restart vega
$ACTUAL_USER ALL=(ALL) NOPASSWD: /bin/systemctl status vega
$ACTUAL_USER ALL=(ALL) NOPASSWD: /usr/bin/apt update
$ACTUAL_USER ALL=(ALL) NOPASSWD: /usr/bin/apt upgrade -y
$ACTUAL_USER ALL=(ALL) NOPASSWD: /usr/bin/apt autoremove -y
$ACTUAL_USER ALL=(ALL) NOPASSWD: /usr/bin/apt clean
EOF
chmod 0440 "$SUDOERS_FILE"

# Reload systemd
echo -e "${YELLOW}Reloading systemd...${NC}"
systemctl daemon-reload

# Enable services
echo -e "${YELLOW}Enabling services...${NC}"
systemctl enable vega.service
systemctl enable vega-daemon.service

# Start services
echo -e "${YELLOW}Starting services...${NC}"
systemctl start vega.service
sleep 2
systemctl start vega-daemon.service

# Check status
echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Installation Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo -e "${YELLOW}Service Status:${NC}"
systemctl status vega.service --no-pager -l | head -20
echo ""
systemctl status vega-daemon.service --no-pager -l | head -20

echo ""
echo -e "${GREEN}Vega is now running as a system service!${NC}"
echo ""
echo -e "${YELLOW}Useful commands:${NC}"
echo "  sudo systemctl status vega           - Check server status"
echo "  sudo systemctl status vega-daemon    - Check daemon status"
echo "  sudo systemctl restart vega          - Restart server"
echo "  sudo systemctl restart vega-daemon   - Restart daemon"
echo "  sudo journalctl -u vega -f           - View server logs"
echo "  sudo journalctl -u vega-daemon -f    - View daemon logs"
echo ""
echo -e "${YELLOW}Log files:${NC}"
echo "  $ACTUAL_HOME/vega_system.log         - System manager log"
echo "  $ACTUAL_HOME/VEGA_COMMENTS.txt       - AI suggestions and comments"
echo "  $ACTUAL_HOME/vega_server.log         - Server output"
echo "  $ACTUAL_HOME/vega_daemon.log         - Daemon output"
echo ""
echo -e "${GREEN}Vega daemon will automatically:${NC}"
echo "  - Keep server running (health checks every 5 min)"
echo "  - Check for updates every 6 hours"
echo "  - Perform daily cleanup at 3 AM"
echo "  - Full system update Sunday at 2 AM"
echo "  - Generate weekly report Monday at 9 AM"
echo ""
