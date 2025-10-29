# 🚀 Vega Dashboard - Quick Reference Card

## Start Dashboard

### For Testing (Manual)

```bash
cd /home/ncacord/Vega2.0
source .venv/bin/activate
python tools/vega/vega_dashboard.py --host 0.0.0.0 --port 8080
```

**Access**: <http://localhost:8080>

### For Production (Always Running)

```bash
# One-time setup
sudo bash /home/ncacord/Vega2.0/setup_dashboard.sh

# Service will auto-start on boot!
```

## Access Points

- **Local**: <http://localhost:8080>
- **Network**: http://YOUR_IP:8080

  ```bash
  # Find your IP:
  hostname -I | awk '{print $1}'
  ```

## Service Commands

```bash
# Status
sudo systemctl status vega-dashboard

# Start/Stop/Restart
sudo systemctl start vega-dashboard
sudo systemctl stop vega-dashboard
sudo systemctl restart vega-dashboard

# Enable/Disable auto-start
sudo systemctl enable vega-dashboard
sudo systemctl disable vega-dashboard

# View logs (real-time)
sudo journalctl -u vega-dashboard -f

# View logs (last 50 lines)
sudo journalctl -u vega-dashboard -n 50
```

## Resource Usage

✅ **RAM Only**: ~100-500MB  
✅ **VRAM**: **0MB** (conserves GPU for Mistral)  
✅ **CPU**: <5% typical  
✅ **Disk**: Minimal (reads from existing databases)

## Key Features

### Real-Time Monitoring

- System status (running/stopped)
- CPU/RAM/GPU usage
- User presence detection
- Conversation history
- AI personality thoughts

### Control Panel

- Start/Stop Vega system
- Force interactions
- Refresh status
- WebSocket live updates (every 5 seconds)

### Health Monitoring

- Overall system health score
- Resource usage tracking
- Error counting
- Uptime tracking

## API Endpoints

### Status Check

```bash
curl http://localhost:8080/api/status | python -m json.tool
```

### Execute Commands

```bash
# Start system
curl -X POST http://localhost:8080/api/command/start

# Stop system
curl -X POST http://localhost:8080/api/command/stop

# Force interaction
curl -X POST http://localhost:8080/api/command/force_interaction

# Refresh status
curl -X POST http://localhost:8080/api/command/refresh
```

## Troubleshooting

### Dashboard not accessible?

```bash
# Check if running
ps aux | grep vega_dashboard

# Check port
sudo lsof -i :8080

# Check logs
sudo journalctl -u vega-dashboard -n 50
```

### WebSocket not connecting?

```bash
# Check firewall
sudo ufw status

# Allow port if needed
sudo ufw allow 8080/tcp
```

### High memory usage?

```bash
# Check current usage
ps aux | grep vega_dashboard

# Adjust limit in service file
sudo nano /etc/systemd/system/vega-dashboard.service
# Change MemoryMax=2G to lower value
sudo systemctl daemon-reload
sudo systemctl restart vega-dashboard
```

## Configuration Files

- **Service**: `/etc/systemd/system/vega-dashboard.service`
- **Dashboard**: `/home/ncacord/Vega2.0/tools/vega/vega_dashboard.py`
- **Setup Script**: `/home/ncacord/Vega2.0/setup_dashboard.sh`
- **Documentation**: `/home/ncacord/Vega2.0/docs/WEB_DASHBOARD.md`

## Port Reference

- **8080**: Vega Dashboard (this service)
- **8000**: Main Vega API (app.py)
- **8001**: OpenAPI Documentation (openapi_app.py)
- **11434**: Ollama LLM backend

## System Architecture

```
Browser → Dashboard (8080) → System Monitoring
              ↓
         CPU/RAM Only
              ↓
      [Vega Database]
      [State Files]
      [psutil/GPU monitors]
              ↓
      Mistral 7B (5GB VRAM) ← Separate from dashboard!
```

## Data Sources

Dashboard reads from:

1. **vega.db**: Conversation history
2. **~/.vega_state/** (if exists):
   - loop_state.json
   - presence_history.jsonl
   - personality_memory.jsonl
3. **System**: psutil for CPU/RAM, pynvml for GPU

## Security Notes

⚠️ **Default**: Localhost only (127.0.0.1)  
⚠️ **Network Access**: Change to 0.0.0.0 in service file  
⚠️ **No Auth**: Add nginx reverse proxy for production  
⚠️ **Firewall**: Use `ufw` to restrict access  

## Performance Tips

### Low Resource Mode

```bash
# Edit service file
sudo nano /etc/systemd/system/vega-dashboard.service

# Add slower updates
Environment="UPDATE_INTERVAL=10"  # 10 sec instead of 5

# Reduce memory
MemoryMax=512M
```

### High Performance Mode

```bash
# Allow more memory for caching
MemoryMax=4G

# Reduce update interval (faster)
Environment="UPDATE_INTERVAL=2"  # 2 seconds
```

## For Dedicated Screen

When you get a small screen:

```bash
# Option 1: Kiosk mode browser
chromium-browser --kiosk --app=http://localhost:8080

# Option 2: Full-screen browser (F11)
firefox http://localhost:8080
# Press F11 for fullscreen

# Option 3: Auto-start on boot
mkdir -p ~/.config/autostart
cat > ~/.config/autostart/vega-dashboard.desktop <<EOF
[Desktop Entry]
Type=Application
Name=Vega Dashboard
Exec=chromium-browser --kiosk http://localhost:8080
X-GNOME-Autostart-enabled=true
EOF
```

## Quick Health Check

```bash
# Single command to check everything
echo "=== Vega Dashboard Health Check ===" && \
echo "Service Status:" && sudo systemctl is-active vega-dashboard && \
echo "Port Status:" && sudo lsof -i :8080 | grep LISTEN && \
echo "API Response:" && curl -s http://localhost:8080/api/status | python -m json.tool | grep -E "is_running|cpu_usage|memory_usage" && \
echo "=== All Good! ==="
```

## Need Help?

📖 **Full Documentation**: `/home/ncacord/Vega2.0/docs/WEB_DASHBOARD.md`  
🔧 **Setup Script**: `/home/ncacord/Vega2.0/setup_dashboard.sh`  
📊 **Dashboard Code**: `/home/ncacord/Vega2.0/tools/vega/vega_dashboard.py`  
📝 **Logs**: `sudo journalctl -u vega-dashboard -f`

---

**TL;DR**: Run `sudo bash setup_dashboard.sh`, then open <http://localhost:8080> 🚀
