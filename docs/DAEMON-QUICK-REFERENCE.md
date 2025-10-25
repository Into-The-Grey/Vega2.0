# Vega Daemon Quick Reference

## Installation

```bash
sudo ./scripts/install_daemon.sh
```

## Service Management

| Command | Description |
|---------|-------------|
| `sudo systemctl start vega-daemon` | Start daemon |
| `sudo systemctl stop vega-daemon` | Stop daemon |
| `sudo systemctl restart vega-daemon` | Restart daemon |
| `sudo systemctl status vega-daemon` | Check status |
| `sudo systemctl enable vega-daemon` | Enable auto-start |
| `sudo systemctl disable vega-daemon` | Disable auto-start |

## CLI Commands

### Daemon Control

```bash
vega daemon start      # Start daemon service
vega daemon stop       # Stop daemon service
vega daemon restart    # Restart daemon service
vega daemon status     # Show daemon status
vega daemon logs       # Show system log (last 50 lines)
vega daemon logs -f    # Follow system log
vega daemon logs --lines 100  # Show last 100 lines
vega daemon comments   # Show AI comments (last 20 lines)
vega daemon comments -f  # Follow AI comments
```

### System Management

```bash
vega system health     # Check system health (CPU, memory, disk)
vega system update     # Check for available updates
vega system update --full  # Install all updates
vega system cleanup    # Run system cleanup
vega system server start    # Start Vega server
vega system server stop     # Stop Vega server
vega system server restart  # Restart Vega server
vega system server status   # Show server status
```

### Direct System Manager

```bash
# Run system manager directly (requires venv activation)
python -m src.vega.daemon.system_manager start
python -m src.vega.daemon.system_manager stop
python -m src.vega.daemon.system_manager restart
python -m src.vega.daemon.system_manager status
python -m src.vega.daemon.system_manager update
python -m src.vega.daemon.system_manager cleanup
python -m src.vega.daemon.system_manager health
```

## Log Files

| File | Purpose | Command to View |
|------|---------|-----------------|
| `~/vega_system.log` | System manager log | `tail -f ~/vega_system.log` |
| `~/VEGA_COMMENTS.txt` | AI suggestions | `tail -f ~/VEGA_COMMENTS.txt` |
| `~/vega_server.log` | Server output | `tail -f ~/vega_server.log` |
| `~/vega_daemon.log` | Daemon output | `tail -f ~/vega_daemon.log` |
| `~/vega_server_error.log` | Server errors | `tail -f ~/vega_server_error.log` |
| `~/vega_daemon_error.log` | Daemon errors | `tail -f ~/vega_daemon_error.log` |

## Scheduled Tasks

| Task | Frequency | Time | Purpose |
|------|-----------|------|---------|
| Health Check | Every 5 min | :00, :05, :10, ... | Monitor system, restart server if down |
| Update Check | Every 6 hours | 00:00, 06:00, 12:00, 18:00 | Check for available updates |
| Daily Cleanup | Daily | 03:00 | Remove junk, rotate logs |
| Weekly Update | Weekly | Sunday 02:00 | Full system update + restart |
| Weekly Report | Weekly | Monday 09:00 | Generate health report |

## Health Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| CPU | 80% | 90% | Log warning |
| Memory | 85% | 95% | Log warning |
| Disk | 90% | 95% | Auto-cleanup |
| Server | Down | Down | Auto-restart |

## Manual Operations

### Update Vega

```bash
# Check for updates
vega system update

# Install all updates
vega system update --full

# Or manually
cd /home/ncacord/Vega2.0
git pull
sudo systemctl restart vega
```

### Clean System

```bash
# Via CLI
vega system cleanup

# Or manually
sudo apt autoremove -y
sudo apt clean
```

### View Comments

```bash
# Recent comments
vega daemon comments

# All comments
cat ~/VEGA_COMMENTS.txt

# Search comments
grep "HEALTH" ~/VEGA_COMMENTS.txt
```

## Troubleshooting

### Daemon Not Running

```bash
# Check status
sudo systemctl status vega-daemon

# View recent errors
sudo journalctl -u vega-daemon -n 50

# Restart daemon
sudo systemctl restart vega-daemon
```

### Server Not Starting

```bash
# Check server status
vega system server status

# View server logs
tail -f ~/vega_server.log
tail -f ~/vega_server_error.log

# Manual restart
vega system server restart
```

### High Resource Usage

```bash
# Check health
vega system health

# View processes
ps aux | grep -E "(vega|python)" | grep -v grep

# Check systemd resource limits
systemctl show vega-daemon | grep -E "(Memory|CPU)"
```

### Updates Failing

```bash
# Check system log
grep "update" ~/vega_system.log

# Test manual update
sudo apt update
sudo apt upgrade -y

# Check git status
cd /home/ncacord/Vega2.0
git status
git pull
```

## Configuration Files

| File | Purpose |
|------|---------|
| `/etc/systemd/system/vega-daemon.service` | Daemon service config |
| `/etc/systemd/system/vega.service` | Server service config |
| `/etc/sudoers.d/vega-daemon` | Sudo permissions |
| `~/.vega/system_state.json` | Daemon state persistence |

## Emergency Actions

### Stop Everything

```bash
sudo systemctl stop vega-daemon
sudo systemctl stop vega
```

### Reset Logs

```bash
# Backup and clear
cp ~/vega_system.log ~/vega_system.log.backup
> ~/vega_system.log

cp ~/VEGA_COMMENTS.txt ~/VEGA_COMMENTS.backup
> ~/VEGA_COMMENTS.txt
```

### Reinstall Daemon

```bash
# Stop services
sudo systemctl stop vega-daemon
sudo systemctl disable vega-daemon

# Remove old files
sudo rm /etc/systemd/system/vega-daemon.service
sudo rm /etc/sudoers.d/vega-daemon

# Reinstall
sudo ./scripts/install_daemon.sh
```

## Python API

```python
from src.vega.daemon.system_manager import VegaSystemManager

# Initialize
manager = VegaSystemManager()

# Server control
manager.start_server()
manager.stop_server()
manager.restart_server()
status = manager.get_server_status()

# Updates
updates = manager.check_for_updates()
manager.update_system()
manager.update_python_packages()
manager.update_vega()

# Maintenance
manager.cleanup_system()
health = manager.monitor_health()

# Comments
manager.add_comment("Custom note", "GENERAL")
```

## Monitoring Dashboard

```bash
# Terminal 1: System log
tail -f ~/vega_system.log

# Terminal 2: AI comments
tail -f ~/VEGA_COMMENTS.txt

# Terminal 3: Health status (every 5 sec)
watch -n 5 'vega system health'

# Terminal 4: Service status (every 5 sec)
watch -n 5 'sudo systemctl status vega vega-daemon'
```

## Performance Tuning

### Adjust Schedule (edit src/vega/daemon/daemon.py)

```python
# Change health check frequency
schedule.every(10).minutes.do(self.health_check)  # from 5 to 10 min

# Change update check frequency
schedule.every(12).hours.do(self.check_updates)  # from 6 to 12 hours
```

### Adjust Resource Limits (edit systemd service)

```bash
sudo systemctl edit vega-daemon

# Add:
[Service]
MemoryLimit=8G      # from 4G
CPUQuota=400%       # from 200%

sudo systemctl daemon-reload
sudo systemctl restart vega-daemon
```

## Uninstall

```bash
# Stop and disable services
sudo systemctl stop vega-daemon vega
sudo systemctl disable vega-daemon vega

# Remove service files
sudo rm /etc/systemd/system/vega-daemon.service
sudo rm /etc/systemd/system/vega.service
sudo rm /etc/sudoers.d/vega-daemon

# Reload systemd
sudo systemctl daemon-reload

# Optional: Remove logs and state
rm ~/vega_*.log
rm ~/VEGA_COMMENTS.txt
rm -rf ~/.vega
```

## Support Resources

- Full Documentation: `docs/DAEMON-SYSTEM.md`
- Architecture: `docs/ARCHITECTURE.md`
- Troubleshooting: `docs/TROUBLESHOOTING.md`
- GitHub Issues: [Vega2.0 Issues](https://github.com/Into-The-Grey/Vega2.0/issues)

---

**Version:** 1.0.0  
**Last Updated:** 2025-01-17
