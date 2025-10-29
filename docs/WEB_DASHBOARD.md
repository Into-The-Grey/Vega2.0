# 🤖 Vega Web Dashboard - Complete Guide

## Overview

The Vega Web Dashboard is an always-running web interface that provides real-time monitoring and control of your Vega AI system. It's designed to use **RAM instead of VRAM** to conserve GPU memory for the Mistral model.

## Architecture

- **Framework**: FastAPI + Uvicorn (Python async web server)
- **Real-time Updates**: WebSocket for live data streaming
- **Resource Usage**: CPU/RAM only - **ZERO VRAM consumption**
- **Port**: 8080 (default, configurable)
- **Frontend**: Single-page HTML/CSS/JavaScript application

## Key Features

### 📊 Real-Time Monitoring

- System status (running/stopped)
- CPU, RAM, and GPU usage
- User presence detection
- Energy levels
- System health score

### 🧠 AI Intelligence Tracking

- Recent thoughts and personality insights
- Conversation history (last 10 interactions)
- Interaction statistics
- Error tracking

### 🎮 Control Panel

- Start/Stop Vega system
- Force chat interactions
- Refresh status
- Real-time WebSocket connection indicator

### 📈 Visualization

- Presence history chart (24 data points)
- Resource usage progress bars
- Health score with color coding
- GPU metrics per device

## Quick Start

### Option 1: Manual Start (Testing)

```bash
cd /home/ncacord/Vega2.0
source .venv/bin/activate
python tools/vega/vega_dashboard.py --host 0.0.0.0 --port 8080
```

Then open: <http://localhost:8080>

### Option 2: Always-Running Service (Production)

```bash
# Install as systemd service (requires sudo)
cd /home/ncacord/Vega2.0
sudo bash setup_dashboard.sh
```

The dashboard will:

- ✅ Start automatically on boot
- ✅ Restart on failure
- ✅ Use max 2GB RAM (conserves VRAM)
- ✅ Run in background continuously

### Access Points

- **Local**: <http://localhost:8080>
- **Network**: http://YOUR_IP:8080 (find IP with `hostname -I`)
- **Future**: Can be displayed on dedicated small screen

## System Requirements

### Dependencies (already installed)

- fastapi==0.115.0
- uvicorn==0.30.6
- websockets==15.0.1
- psutil==6.1.0
- jinja2==3.1.6
- pynvml==13.0.1 (for GPU monitoring)
- python-multipart==0.0.20

### Hardware Requirements

- **RAM**: 100-500MB typical usage (max 2GB limit)
- **CPU**: <5% typical usage
- **VRAM**: **0MB** (runs entirely on CPU)
- **Network**: Port 8080 access

## Resource Optimization

### Why RAM-Only Design?

Your Mistral 7B model uses ~5GB VRAM on the GTX 1660 SUPER. The dashboard is specifically designed to:

1. **No GPU Rendering**: Uses HTML Canvas with CPU-based rendering
2. **No VRAM Usage**: FastAPI/Uvicorn runs entirely on CPU
3. **Memory Limits**: Systemd restricts to 2GB RAM max
4. **Efficient Updates**: WebSocket minimizes network overhead

### Resource Allocation

```
╔══════════════════════════════════════════════════╗
║ System Resource Distribution                     ║
╠══════════════════════════════════════════════════╣
║ GPU (GTX 1660 SUPER 6GB)                        ║
║   ├─ Mistral 7B Model:     ~5GB VRAM ████████   ║
║   └─ Web Dashboard:         0GB VRAM            ║
║                                                  ║
║ RAM (125GB Total)                               ║
║   ├─ Mistral Inference:    ~3GB RAM  █          ║
║   ├─ Web Dashboard:        ~0.5GB    ▌          ║
║   └─ System/Other:         ~121GB    Available  ║
╚══════════════════════════════════════════════════╝
```

## Configuration

### Change Port

```bash
# Manual start
python tools/vega/vega_dashboard.py --port 9000

# For systemd service, edit:
sudo nano /etc/systemd/system/vega-dashboard.service
# Change: --port 8080 to --port 9000
sudo systemctl daemon-reload
sudo systemctl restart vega-dashboard
```

### Bind to Specific Interface

```bash
# Localhost only (default for security)
--host 127.0.0.1

# All interfaces (network access)
--host 0.0.0.0

# Specific IP
--host 192.168.1.100
```

### Resource Limits

Edit `/etc/systemd/system/vega-dashboard.service`:

```ini
# Memory limits
MemoryMax=2G        # Hard limit
MemoryHigh=1.5G     # Soft limit (warning)

# CPU limits (optional)
CPUQuota=50%        # Max 50% of one core
```

## Service Management

### Check Status

```bash
sudo systemctl status vega-dashboard
```

### View Logs

```bash
# Real-time logs
sudo journalctl -u vega-dashboard -f

# Last 100 lines
sudo journalctl -u vega-dashboard -n 100

# Since boot
sudo journalctl -u vega-dashboard -b
```

### Control Commands

```bash
# Start
sudo systemctl start vega-dashboard

# Stop
sudo systemctl stop vega-dashboard

# Restart
sudo systemctl restart vega-dashboard

# Enable auto-start on boot
sudo systemctl enable vega-dashboard

# Disable auto-start
sudo systemctl disable vega-dashboard
```

## API Endpoints

The dashboard exposes several API endpoints:

### GET `/api/status`

Returns current system status JSON

```json
{
  "is_running": true,
  "mode": "ambient",
  "uptime_seconds": 3600,
  "user_presence": "active",
  "energy_level": 0.85,
  "cpu_usage": 12.5,
  "memory_usage": 45.2,
  "gpu_usage": {
    "NVIDIA GeForce GTX 1660 SUPER": {
      "utilization": 87,
      "memory_used": 5368709120,
      "memory_total": 6442450944,
      "memory_percent": 83.3
    }
  },
  "system_health": 92.5,
  "total_conversations": 1247,
  "recent_thoughts": [...],
  "conversation_history": [...]
}
```

### POST `/api/command/{action}`

Execute system commands

**Actions**:

- `start`: Start Vega system
- `stop`: Stop Vega system
- `force_interaction`: Trigger immediate chat
- `refresh`: Refresh status data

```bash
# Example
curl -X POST http://localhost:8080/api/command/refresh
```

### WebSocket `/ws`

Real-time status updates every 5 seconds

```javascript
const ws = new WebSocket('ws://localhost:8080/ws');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Status:', data);
};
```

## Integration with Existing Systems

The dashboard integrates with your existing Vega components:

### Data Sources

1. **vega.db**: SQLite database for conversation history
2. **State Files** (if available):
   - `~/.vega_state/loop_state.json`: Main loop status
   - `~/.vega_state/presence_history.jsonl`: User presence tracking
   - `~/.vega_state/personality_memory.jsonl`: AI thoughts

3. **System Metrics**:
   - psutil: CPU/RAM usage
   - pynvml: GPU monitoring (both GTX 1660 SUPER + Quadro P1000)

### Process Detection

The dashboard automatically detects if `vega_loop.py` is running:

```python
def _check_daemon_running(self) -> bool:
    """Check if vega_loop.py is running"""
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        if "vega_loop.py" in cmdline:
            return True
    return False
```

## Troubleshooting

### Dashboard Won't Start

**Check dependencies:**

```bash
cd /home/ncacord/Vega2.0
source .venv/bin/activate
pip install -r requirements.txt
```

**Check port availability:**

```bash
sudo lsof -i :8080
# If occupied, kill process or use different port
```

**Check permissions:**

```bash
ls -la tools/vega/vega_dashboard.py
# Should be readable/executable
```

### WebSocket Connection Fails

**Check firewall:**

```bash
sudo ufw status
sudo ufw allow 8080/tcp  # If needed
```

**Check browser console:**

```
F12 → Console → Look for WebSocket errors
```

### High Memory Usage

**Check current usage:**

```bash
ps aux | grep vega_dashboard
# Shows PID, %CPU, %MEM, RSS

# Detailed memory info
pmap <PID> | tail -1
```

**Adjust limits:**

```bash
sudo systemctl edit vega-dashboard
# Add:
[Service]
MemoryMax=1G  # Lower limit
```

### GPU Metrics Not Showing

**Install NVIDIA drivers:**

```bash
nvidia-smi  # Should show GPU info

# If not found:
sudo apt install nvidia-utils-XXX  # Your driver version
```

**Check pynvml:**

```python
python -c "import pynvml; pynvml.nvmlInit(); print('OK')"
```

### Connection Refused from Network

**Check bind address:**

```bash
sudo netstat -tlnp | grep 8080
# Should show 0.0.0.0:8080 not 127.0.0.1:8080
```

**Update service:**

```bash
sudo nano /etc/systemd/system/vega-dashboard.service
# Ensure: --host 0.0.0.0
sudo systemctl daemon-reload
sudo systemctl restart vega-dashboard
```

## Performance Tuning

### For Low-Resource Systems

```ini
[Service]
# Reduce memory
MemoryMax=512M

# Reduce CPU priority
CPUWeight=50
Nice=10

# Reduce update frequency
Environment="UPDATE_INTERVAL=10"  # 10 seconds instead of 5
```

### For High-Performance Systems

```ini
[Service]
# Allow more memory for caching
MemoryMax=4G

# More worker processes (optional)
Environment="WORKERS=2"
```

## Future Enhancements

### Planned for Dedicated Screen

When you get a small dedicated screen, you can:

1. **Clone display** using `xrandr`:

   ```bash
   xrandr --output HDMI-1 --same-as eDP-1
   ```

2. **Kiosk mode** browser:

   ```bash
   chromium-browser --kiosk --app=http://localhost:8080
   ```

3. **Touch screen support**: Dashboard is mobile-responsive

4. **Auto-refresh on boot**:

   ```bash
   # Add to ~/.config/autostart/vega-dashboard-browser.desktop
   [Desktop Entry]
   Type=Application
   Name=Vega Dashboard
   Exec=chromium-browser --kiosk http://localhost:8080
   X-GNOME-Autostart-enabled=true
   ```

### Additional Features You Can Add

1. **Voice visualization**: Integrate with existing voice_visualizer.py
2. **Training progress**: Show dynamic training status
3. **Model selection**: Switch between models via UI
4. **Chat interface**: Built-in chat window
5. **Voice control**: Browser speech recognition API
6. **Mobile app**: Progressive Web App (PWA) support

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     Browser Client                       │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │  Dashboard  │  │   WebSocket  │  │  Control API  │ │
│  │   HTML/CSS  │←→│  Connection  │←→│   Endpoints   │ │
│  └─────────────┘  └──────────────┘  └───────────────┘ │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP/WS
                           ↓
┌─────────────────────────────────────────────────────────┐
│              FastAPI Application (Port 8080)             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Routes     │  │  WebSocket   │  │    Status    │ │
│  │   Handler    │→ │   Manager    │→ │   Updater    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└──────────────────────────┬──────────────────────────────┘
                           │ CPU/RAM Only
                           ↓
┌─────────────────────────────────────────────────────────┐
│                   System Integration                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   vega.db    │  │  State Files │  │  psutil/GPU  │ │
│  │  (SQLite)    │  │   (JSON/L)   │  │   Monitors   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                   Vega AI System                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Mistral 7B on GPU (5GB VRAM) ← NOT dashboard!   │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Security Considerations

### Default Security

- Binds to `127.0.0.1` (localhost only) for initial testing
- Change to `0.0.0.0` only when needed for network access
- No authentication by default (add nginx reverse proxy for production)

### Production Hardening

1. **Add nginx reverse proxy**:

```nginx
server {
    listen 443 ssl;
    server_name vega.yourdomain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

2. **Add API authentication**:

```python
# Add to vega_dashboard.py
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key")

@app.get("/api/status")
async def get_status(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    # ... rest of code
```

3. **Enable firewall**:

```bash
sudo ufw enable
sudo ufw allow from 192.168.1.0/24 to any port 8080  # LAN only
```

## Comparison with Other Vega UIs

| Feature | vega_dashboard.py | vega_ui.py | app.py |
|---------|-------------------|------------|---------|
| Type | Web GUI | Terminal UI | API Server |
| Port | 8080 | N/A (CLI) | 8000 |
| Always Running | ✅ Yes | ❌ Interactive | ✅ Yes |
| Real-time Updates | ✅ WebSocket | ✅ Live refresh | ❌ REST only |
| Resource Usage | CPU/RAM | CPU/RAM | CPU/RAM |
| Mobile Friendly | ✅ Yes | ❌ No | ⚠️ Basic HTML |
| Monitoring | ✅ Comprehensive | ✅ Detailed | ❌ Limited |
| Control Panel | ✅ Yes | ✅ Yes | ⚠️ API only |
| Best For | Dedicated screen | SSH sessions | Integrations |

**Recommendation**: Use `vega_dashboard.py` (this guide) for your always-running web visualization.

## Summary

The Vega Web Dashboard provides:

✅ **Always-running** web interface via systemd  
✅ **RAM-only** design (ZERO VRAM usage)  
✅ **Real-time** updates via WebSocket  
✅ **Comprehensive** monitoring (CPU/RAM/GPU/conversations)  
✅ **Control panel** for system management  
✅ **Mobile-responsive** design  
✅ **Production-ready** with resource limits  

**Quick Start**: `sudo bash setup_dashboard.sh`  
**Access**: <http://localhost:8080>  
**Logs**: `sudo journalctl -u vega-dashboard -f`

Perfect for displaying on a dedicated small screen when you get one! 🚀
