# 🤖 VEGA JARVIS MODE - Complete Setup Guide

## What is JARVIS Mode?

JARVIS Mode transforms Vega into a persistent, always-available AI assistant like Tony Stark's JARVIS:

- **Always Running**: 100% uptime with automatic restarts
- **Never Forgets**: Single continuous conversation that persists forever
- **Never Crashes**: Proactive memory management prevents OOM errors
- **Seamless Context**: Picks up exactly where you left off, always

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Web UI (Port 8080)                                         │
│  ├─ Chat Interface                                          │
│  ├─ Real-time Memory Stats                                  │
│  └─ Voice Input/Output                                      │
└─────────────────────────────────────────────────────────────┘
                           ↓ HTTP/WebSocket
┌─────────────────────────────────────────────────────────────┐
│  Vega Server (Port 8000)                                    │
│  ├─ FastAPI Application                                     │
│  ├─ Persistent Session Manager                              │
│  ├─ Memory Manager (Background)                             │
│  └─ LLM Backend (Ollama/etc)                                │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  SQLite Database (vega.db)                                  │
│  ├─ Single Persistent Session                               │
│  ├─ Full Conversation History                               │
│  └─ Memory Facts & Context                                  │
└─────────────────────────────────────────────────────────────┘
```

### Key Features

1. **Persistent Session Management**
   - Single session ID for entire lifetime
   - No "new chat" - continuous conversation
   - Context automatically maintained

2. **Proactive Memory Management**
   - Monitors memory every 30 seconds
   - Warning threshold: 80% system memory
   - Critical threshold: 90% system memory
   - Automatic cleanup before OOM occurs

3. **Smart Context Compression**
   - Keeps recent 50 conversations in full
   - Compresses older conversations (keeps prompts, summarizes responses)
   - Archives very old conversations (>2 hours)
   - Maintains searchable history

4. **Auto-Recovery**
   - Systemd service auto-restarts on failure
   - 10 second restart delay
   - 5 restart attempts before giving up
   - Graceful shutdown handling

## Installation Steps

### 1. Install Vega Server as Systemd Service

```bash
cd /home/ncacord/Vega2.0

# Copy service file
sudo cp systemd/vega-server.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable auto-start on boot
sudo systemctl enable vega-server

# Start the service
sudo systemctl start vega-server

# Check status
sudo systemctl status vega-server
```

### 2. Verify Web UI is Running

The web UI should already be running via `vega-dashboard` service:

```bash
# Check web UI status
sudo systemctl status vega-dashboard

# If not running, start it
sudo systemctl restart vega-dashboard
```

### 3. Access the Interface

Open your browser to:

- **Local**: <http://localhost:8080>
- **Network**: <http://192.168.1.147:8080>

You should see:

- **Status**: "ONLINE [JARVIS MODE]"
- **Memory Stats**: "MEM: XXXMBprocess_memory_mb | SYS: XX.X%"

## Usage

### Starting a Conversation

Just type and press Enter or click SEND. Vega remembers EVERYTHING from all previous conversations automatically.

**Example:**

```
You: "Hey Vega, remember my favorite color?"
Vega: "I don't have that information yet. What's your favorite color?"
You: "It's blue"
Vega: "Got it! Your favorite color is blue."

[Later, even after server restart...]

You: "What's my favorite color?"
Vega: "Your favorite color is blue!"
```

### Voice Controls

- **Microphone Button (🎤)**: Click to speak, Vega will transcribe
- **Speaker Button (🔊)**: Toggle to have Vega speak responses

### Monitoring Memory

Watch the header for real-time memory stats:

- **Blue** (normal): System memory < 80%
- **Orange** (warning): System memory 80-90%
- **Red** (critical): System memory > 90%

Memory manager will automatically clean up before hitting OOM.

## Memory Management Details

### How Memory is Managed

1. **Recent Context (Last 50 conversations)**
   - Kept in full (prompt + complete response)
   - Used for LLM context window
   - Fast access for continuity

2. **Older Context (50-100 conversations)**
   - Compressed (prompt kept, response summarized)
   - Still searchable
   - Saves significant memory

3. **Ancient Context (100+ conversations)**
   - Summarized and archived as memory facts
   - Original conversations removed from DB
   - Summary preserved in metadata

### Manual Memory Operations

If you need to manually manage memory:

```bash
# View current memory stats
curl http://localhost:8000/metrics | jq '.memory_manager'

# Check database size
ls -lh /home/ncacord/Vega2.0/vega.db*

# Manual vacuum (if needed)
sqlite3 /home/ncacord/Vega2.0/vega.db "VACUUM;"
```

### Configuration

Edit `/home/ncacord/Vega2.0/src/vega/core/memory_manager.py` to adjust:

```python
MemoryManager(
    warning_threshold_percent=80.0,   # Start gentle cleanup
    critical_threshold_percent=90.0,  # Aggressive cleanup
    check_interval_seconds=30,        # How often to check
    max_context_entries=50,           # Recent context size
    compression_trigger=100,          # When to compress
)
```

## Monitoring & Logs

### View Logs

```bash
# Real-time logs (Vega server)
sudo journalctl -u vega-server -f

# Real-time logs (Web UI)
sudo journalctl -u vega-dashboard -f

# Last 100 lines
sudo journalctl -u vega-server -n 100

# Last hour
sudo journalctl -u vega-server --since "1 hour ago"
```

### Check Service Status

```bash
# Vega server
sudo systemctl status vega-server

# Web UI
sudo systemctl status vega-dashboard

# Both at once
sudo systemctl status vega-server vega-dashboard
```

### Resource Usage

```bash
# Memory usage
free -h

# Vega process specifically
ps aux | grep python | grep main.py

# Database size
du -h /home/ncacord/Vega2.0/vega.db*
```

## Troubleshooting

### Vega Won't Start

```bash
# Check logs for errors
sudo journalctl -u vega-server -n 50

# Common issues:
# 1. Port 8000 already in use
sudo netstat -tlnp | grep 8000

# 2. Missing dependencies
source /home/ncacord/Vega2.0/.venv/bin/activate
pip install -r requirements.txt

# 3. Database locked
rm /home/ncacord/Vega2.0/vega.db-shm
rm /home/ncacord/Vega2.0/vega.db-wal
```

### Memory Issues Persist

```bash
# Check if memory manager is running
curl http://localhost:8000/metrics | jq '.memory_manager.running'

# Force aggressive cleanup
# (Stop Vega, clean DB, restart)
sudo systemctl stop vega-server
sqlite3 /home/ncacord/Vega2.0/vega.db "DELETE FROM conversations WHERE id < (SELECT MAX(id) - 50 FROM conversations);"
sqlite3 /home/ncacord/Vega2.0/vega.db "VACUUM;"
sudo systemctl start vega-server
```

### Web UI Shows OFFLINE

```bash
# 1. Check if Vega server is running
curl http://localhost:8000/healthz

# 2. Check if API key matches
grep API_KEY /home/ncacord/Vega2.0/.env
grep VEGA_API_KEY /home/ncacord/Vega2.0/tools/vega/vega_web_ui.py

# 3. Restart both services
sudo systemctl restart vega-server vega-dashboard
```

### Conversation Context Lost

This should NEVER happen in JARVIS mode, but if it does:

```bash
# Check persistent session
sqlite3 /home/ncacord/Vega2.0/vega.db "SELECT session_id, COUNT(*) FROM conversations GROUP BY session_id;"

# Should see one session with all conversations
# If multiple sessions, you may need to manually set persistent session
```

## Advanced Configuration

### Change Memory Thresholds

Edit `/home/ncacord/Vega2.0/src/vega/core/memory_manager.py`:

```python
# For systems with less RAM, be more aggressive
MemoryManager(
    warning_threshold_percent=70.0,   # Start cleanup earlier
    critical_threshold_percent=85.0,  # Be more aggressive
    max_context_entries=30,           # Keep less context
)

# For systems with lots of RAM, be more lenient
MemoryManager(
    warning_threshold_percent=85.0,   # More headroom
    critical_threshold_percent=95.0,  # Only cleanup when really needed
    max_context_entries=100,          # Keep more context
)
```

Then restart: `sudo systemctl restart vega-server`

### Change Context Window Size

Edit `/home/ncacord/Vega2.0/.env`:

```bash
# How many recent conversations to include in LLM context
CONTEXT_WINDOW_SIZE=50

# Maximum characters of context to send to LLM
CONTEXT_MAX_CHARS=8000
```

### Enable Debug Logging

Edit `/home/ncacord/Vega2.0/systemd/vega-server.service`:

```ini
[Service]
Environment="LOG_LEVEL=DEBUG"
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl restart vega-server
sudo journalctl -u vega-server -f
```

## Performance Tips

1. **SSD vs HDD**: Put `vega.db` on SSD for faster access
2. **RAM**: More RAM = more context you can keep
3. **GPU**: If using local LLM, GPU significantly speeds up responses
4. **Network**: Web UI on same network as Vega server for best performance

## Backup & Recovery

### Backup Conversation History

```bash
# Backup database
cp /home/ncacord/Vega2.0/vega.db /home/ncacord/vega-backup-$(date +%Y%m%d).db

# Export as JSON
sqlite3 /home/ncacord/Vega2.0/vega.db \
  "SELECT json_object('prompt', prompt, 'response', response, 'ts', ts) \
   FROM conversations ORDER BY id" \
  > /home/ncacord/vega-export-$(date +%Y%m%d).json
```

### Restore from Backup

```bash
# Stop Vega
sudo systemctl stop vega-server

# Restore database
cp /home/ncacord/vega-backup-20250101.db /home/ncacord/Vega2.0/vega.db

# Start Vega
sudo systemctl start vega-server
```

## System Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 16GB minimum, 32GB+ recommended for JARVIS mode
- **Storage**: 100GB+ for conversation history
- **OS**: Linux with systemd (Ubuntu 20.04+, Debian 11+, etc.)
- **Python**: 3.10+
- **LLM Backend**: Ollama, OpenAI API, or compatible

## FAQ

**Q: Will Vega really never forget?**
A: Yes! All conversations are persisted to SQLite. Even if the server restarts, context is maintained.

**Q: What happens if memory gets full?**
A: Memory manager automatically compresses old conversations before OOM occurs. You'll never lose data.

**Q: Can I have multiple conversations?**
A: JARVIS mode is designed for single continuous conversation. If you need multiple contexts, you can manually specify session IDs in API calls.

**Q: How much history can Vega remember?**
A: Effectively unlimited. Recent context is kept in full, older context is compressed, ancient context is archived. You're only limited by disk space.

**Q: Does this work with any LLM backend?**
A: Yes! Ollama, OpenAI, Anthropic, local models - anything Vega supports.

**Q: What if I want to start fresh?**
A: Stop Vega, delete `vega.db`, restart. Or use SQL to clear: `DELETE FROM conversations;`

## Support

If you have issues:

1. Check logs: `sudo journalctl -u vega-server -n 100`
2. Check status: `sudo systemctl status vega-server`
3. Check memory: `free -h && curl http://localhost:8000/metrics | jq '.memory_manager'`
4. Restart services: `sudo systemctl restart vega-server vega-dashboard`

---

**🌌 Welcome to JARVIS Mode - Your AI, Always Available, Always Aware 🌌**
