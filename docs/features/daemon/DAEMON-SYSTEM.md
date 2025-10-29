# Vega Daemon System

The Vega Daemon System provides autonomous operation of the Vega AI Platform with automatic updates, system maintenance, health monitoring, and AI-powered suggestions.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Architecture](#architecture)
- [Features](#features)
- [Operation](#operation)
- [Log Files](#log-files)
- [Scheduled Tasks](#scheduled-tasks)
- [Manual Commands](#manual-commands)
- [Troubleshooting](#troubleshooting)

## Overview

The daemon system enables Vega to:

- ✅ Run continuously as a system service (starts on boot)
- ✅ Monitor server health every 5 minutes
- ✅ Auto-restart server if it goes down
- ✅ Check for updates every 6 hours
- ✅ Perform daily system cleanup at 3 AM
- ✅ Execute full updates on Sundays at 2 AM
- ✅ Generate weekly health reports on Mondays at 9 AM
- ✅ Log all actions comprehensively
- ✅ Provide AI-powered suggestions and comments

## Installation

### Prerequisites

- Ubuntu/Debian Linux with systemd
- Python 3.12+ with virtual environment
- sudo access
- Vega 2.0 installed and configured

### Install Process

```bash
# From Vega directory
cd /home/ncacord/Vega2.0

# Run installer (requires sudo)
sudo ./scripts/install_daemon.sh
```

The installer will:

1. Create necessary directories (`~/.vega`, `~/.vega/temp`)
2. Install Python dependencies (schedule, psutil)
3. Configure systemd services (vega.service, vega-daemon.service)
4. Set up sudo permissions for daemon operations
5. Enable and start services
6. Display status and configuration

### Verification

```bash
# Check service status
sudo systemctl status vega
sudo systemctl status vega-daemon

# View logs
tail -f ~/vega_system.log
tail -f ~/VEGA_COMMENTS.txt
```

## Architecture

### Components

#### 1. VegaSystemManager (`src/vega/daemon/system_manager.py`)

Core system management functionality with low-level operations:

- **Server Management**: Start, stop, restart, status checks
- **Update Management**: System (apt), Python packages (pip), Vega code (git)
- **Cleanup**: Package cleanup, log rotation, temp file removal
- **Health Monitoring**: CPU, memory, disk usage with auto-suggestions
- **Logging**: Dual logging to file and Python logger
- **Comments**: AI-powered suggestions in timestamped comment file

#### 2. VegaDaemon (`src/vega/daemon/daemon.py`)

Continuous monitoring service with scheduled tasks:

- **Health Checks**: Every 5 minutes
- **Update Checks**: Every 6 hours
- **Daily Cleanup**: 3 AM daily
- **Weekly Updates**: Sunday 2 AM
- **Weekly Reports**: Monday 9 AM

#### 3. Systemd Services

**vega.service**: Main Vega API server

- Runs `main.py server --host 127.0.0.1 --port 8000`
- Auto-restarts on failure
- Logs to `~/vega_server.log`

**vega-daemon.service**: Autonomous management daemon

- Runs `src.vega.daemon.daemon`
- Always running with auto-restart
- Resource limits: 4GB memory, 200% CPU
- Security hardening enabled
- Logs to `~/vega_daemon.log`

### Data Flow

```
Daemon Loop (60s) → Scheduled Tasks → System Manager → Sudo Commands → System Actions
                                                     ↓
                                      Logging (vega_system.log + VEGA_COMMENTS.txt)
```

## Features

### Automatic Server Management

The daemon monitors server health every 5 minutes:

- Checks if server is running
- Auto-restarts if down
- Monitors CPU, memory, disk usage
- Generates alerts on resource issues

**Health Thresholds:**

- CPU: Warning at 80%, critical at 90%
- Memory: Warning at 85%, critical at 95%
- Disk: Warning at 90%, critical at 95%

**Auto-Actions:**

- Restart server if down
- Trigger cleanup if disk > 95%

### Automatic Updates

**Update Check (Every 6 Hours):**

- Checks for system package updates (apt)
- Checks for Python package updates (pip)
- Checks for Vega code updates (git)
- Logs available updates

**Weekly Update (Sunday 2 AM):**

- Updates system packages: `sudo apt update && sudo apt upgrade -y`
- Updates Python packages: `pip install --upgrade <package>`
- Updates Vega code: `git pull`
- Restarts server after updates
- Runs full cleanup

### System Cleanup

**Daily Cleanup (3 AM):**

- Removes unused packages: `sudo apt autoremove -y`
- Cleans package cache: `sudo apt clean`
- Rotates large log files (>10MB)
- Removes temp files
- Calculates and reports space freed

**Emergency Cleanup:**

- Triggered automatically if disk usage > 95%
- Can be triggered manually via CLI

### Health Monitoring

**Continuous Monitoring:**

- CPU usage (psutil)
- Memory usage (psutil)
- Disk usage (shutil)
- Server process status

**AI-Powered Suggestions:**

- High CPU: Suggests checking for runaway processes
- High memory: Suggests reviewing memory leaks
- Low disk: Suggests cleanup or disk expansion
- Server down: Auto-restart with notification

### Comprehensive Logging

All daemon activities are logged with detailed information:

**Log Format:**

```
2025-01-17 14:30:00 - VegaSystemManager - INFO - [ACTION] Server Management: Started Vega server - SUCCESS
  Details: {"status": "active", "pid": 12345}
```

**Action Types:**

- SERVER_MANAGEMENT: Start, stop, restart operations
- SYSTEM_UPDATE: apt update/upgrade operations
- PYTHON_UPDATE: pip package updates
- VEGA_UPDATE: git pull operations
- SYSTEM_CLEANUP: Cleanup and maintenance
- HEALTH_MONITORING: Health checks and monitoring
- DAEMON_START/STOP: Daemon lifecycle events

### AI Comments System

Separate from logs, the daemon generates AI-powered suggestions and reports:

**Comment Format:**

```
================================================================================
[2025-01-17 14:30:00] HEALTH_MONITORING
System health check completed. All metrics within normal range.
Current status: CPU 25%, Memory 42%, Disk 67%

================================================================================
[2025-01-17 15:00:00] SYSTEM_CLEANUP
Performed daily system cleanup.
Freed 250MB of disk space from package cache and temp files.
```

**Comment Categories:**

- GENERAL: General information and notes
- SERVER_MANAGEMENT: Server lifecycle events
- SYSTEM_UPDATE: Update operations
- PYTHON_UPDATE: Python package updates
- VEGA_UPDATE: Vega code updates
- SYSTEM_CLEANUP: Cleanup operations
- HEALTH_MONITORING: Health check results
- WEEKLY_REPORT: Comprehensive reports
- DAEMON_START/STOP: Daemon status changes

## Operation

### Service Management

```bash
# Start services
sudo systemctl start vega
sudo systemctl start vega-daemon

# Stop services
sudo systemctl stop vega
sudo systemctl stop vega-daemon

# Restart services
sudo systemctl restart vega
sudo systemctl restart vega-daemon

# Check status
sudo systemctl status vega
sudo systemctl status vega-daemon

# View logs
sudo journalctl -u vega -f
sudo journalctl -u vega-daemon -f

# Enable auto-start (already enabled by installer)
sudo systemctl enable vega
sudo systemctl enable vega-daemon
```

### Log Monitoring

```bash
# System management log
tail -f ~/vega_system.log

# AI comments and suggestions
tail -f ~/VEGA_COMMENTS.txt

# Server output
tail -f ~/vega_server.log

# Daemon output
tail -f ~/vega_daemon.log

# Error logs
tail -f ~/vega_server_error.log
tail -f ~/vega_daemon_error.log
```

## Log Files

All log files are stored in the user's home directory:

| File | Purpose | Content |
|------|---------|---------|
| `~/vega_system.log` | System manager actions | All daemon operations with timestamps, details, status |
| `~/VEGA_COMMENTS.txt` | AI suggestions | Timestamped comments, reports, suggestions by category |
| `~/vega_server.log` | Server stdout | Vega API server standard output |
| `~/vega_server_error.log` | Server stderr | Vega API server error output |
| `~/vega_daemon.log` | Daemon stdout | Daemon process standard output |
| `~/vega_daemon_error.log` | Daemon stderr | Daemon process error output |

**Log Rotation:**

- System log automatically rotated when > 10MB
- Backed up to `.log.old` before rotation
- Only one backup kept (oldest removed)

## Scheduled Tasks

### Health Check (Every 5 Minutes)

**Purpose:** Continuous server and system monitoring

**Actions:**

1. Check if server is running
2. If down, auto-restart server
3. Monitor CPU usage
4. Monitor memory usage
5. Monitor disk usage
6. Generate warnings on high usage
7. Trigger emergency cleanup if disk > 95%
8. Log health status

**Execution:** 00:00, 00:05, 00:10, 00:15, 00:20, 00:25, 00:30, 00:35, 00:40, 00:45, 00:50, 00:55 (every hour)

### Update Check (Every 6 Hours)

**Purpose:** Monitor for available updates

**Actions:**

1. Check for system package updates: `apt list --upgradable`
2. Check for Python package updates: `pip list --outdated`
3. Check for Vega code updates: `git fetch && git status`
4. Log available updates
5. Add comment with update summary

**Execution:** 00:00, 06:00, 12:00, 18:00

### Daily Cleanup (3 AM)

**Purpose:** Regular system maintenance

**Actions:**

1. Remove unused packages: `sudo apt autoremove -y`
2. Clean package cache: `sudo apt clean`
3. Rotate large log files (>10MB)
4. Remove temp files from `~/.vega/temp/`
5. Calculate space freed
6. Log cleanup results
7. Add comment with space freed

**Execution:** 03:00 daily

### Weekly Update (Sunday 2 AM)

**Purpose:** Full system update and maintenance

**Actions:**

1. Update system packages: `sudo apt update && sudo apt upgrade -y`
2. Update all Python packages: `pip install --upgrade <each package>`
3. Update Vega code: `git pull`
4. Restart Vega server
5. Run full cleanup
6. Log all update results
7. Add comprehensive comment

**Execution:** Sunday at 02:00

### Weekly Report (Monday 9 AM)

**Purpose:** Comprehensive system status report

**Actions:**

1. Gather statistics from state:
   - Total restarts this week
   - Total updates performed
   - Total cleanups performed
2. Run health check
3. Generate comprehensive report with:
   - State counters
   - Current health metrics
   - Recommendations
4. Log report
5. Add report to comments file

**Execution:** Monday at 09:00

## Manual Commands

### System Manager CLI

The system manager can be run manually for immediate operations:

```bash
# Navigate to Vega directory
cd /home/ncacord/Vega2.0

# Activate virtual environment
source .venv/bin/activate

# Start server
python -m src.vega.daemon.system_manager start

# Stop server
python -m src.vega.daemon.system_manager stop

# Restart server
python -m src.vega.daemon.system_manager restart

# Check server status
python -m src.vega.daemon.system_manager status

# Trigger update
python -m src.vega.daemon.system_manager update

# Trigger cleanup
python -m src.vega.daemon.system_manager cleanup

# Check health
python -m src.vega.daemon.system_manager health
```

### Python API

You can also use the system manager programmatically:

```python
from src.vega.daemon.system_manager import VegaSystemManager

# Initialize manager
manager = VegaSystemManager()

# Server operations
manager.start_server()
manager.stop_server()
manager.restart_server()
status = manager.get_server_status()

# Update operations
updates = manager.check_for_updates()
manager.update_system()
manager.update_python_packages()
manager.update_vega()

# Maintenance
manager.cleanup_system()
health = manager.monitor_health()

# Logging
manager.add_comment("Custom comment", "GENERAL")
```

## Troubleshooting

### Service Won't Start

**Check service status:**

```bash
sudo systemctl status vega-daemon
sudo journalctl -u vega-daemon -n 50
```

**Common causes:**

1. **Python environment not found**
   - Check `.venv` exists in Vega directory
   - Reinstall dependencies: `pip install schedule psutil`

2. **Permission issues**
   - Check log file permissions: `ls -la ~/vega_*.log`
   - Recreate directories: `mkdir -p ~/.vega/temp`

3. **Port already in use**
   - Check if Vega server is already running: `ps aux | grep python`
   - Kill existing process: `pkill -f "main.py server"`

### Sudo Commands Failing

**Check sudoers configuration:**

```bash
sudo visudo -c
sudo cat /etc/sudoers.d/vega-daemon
```

**Verify permissions:**

```bash
# Test sudo commands
sudo -n systemctl status vega
sudo -n apt update
```

**If failing, reinstall sudoers file:**

```bash
sudo rm /etc/sudoers.d/vega-daemon
sudo ./scripts/install_daemon.sh
```

### Server Not Auto-Restarting

**Check daemon logs:**

```bash
tail -f ~/vega_system.log | grep SERVER_MANAGEMENT
```

**Verify health check is running:**

```bash
sudo journalctl -u vega-daemon -f | grep health_check
```

**Manual test:**

```bash
# Stop server
sudo systemctl stop vega

# Wait 5 minutes for health check
# Check if restarted
sudo systemctl status vega
```

### Updates Not Running

**Check update check logs:**

```bash
grep "check_updates" ~/vega_system.log
```

**Verify scheduled tasks:**

```bash
sudo journalctl -u vega-daemon -f | grep schedule
```

**Manual update:**

```bash
python -m src.vega.daemon.system_manager update
```

### High Resource Usage

**Check daemon resource usage:**

```bash
systemctl status vega-daemon
ps aux | grep daemon
```

**Resource limits configured in systemd:**

- Memory: 4GB limit
- CPU: 200% quota (2 cores)
- Files: 65536 open file limit

**Adjust if needed:**

```bash
sudo systemctl edit vega-daemon
# Add override:
# [Service]
# MemoryLimit=8G
# CPUQuota=400%

sudo systemctl daemon-reload
sudo systemctl restart vega-daemon
```

### Log Files Growing Too Large

**Check log sizes:**

```bash
du -h ~/vega_*.log
du -h ~/VEGA_COMMENTS.txt
```

**Manual log rotation:**

```bash
# Backup and clear
mv ~/vega_system.log ~/vega_system.log.old
touch ~/vega_system.log

# Or truncate
truncate -s 0 ~/vega_system.log
```

**Automatic rotation:**

- System log automatically rotates at 10MB
- Configure in `system_manager.py` if needed

### Comments File Not Updating

**Check write permissions:**

```bash
ls -la ~/VEGA_COMMENTS.txt
```

**Verify add_comment is working:**

```bash
python -c "
from src.vega.daemon.system_manager import VegaSystemManager
manager = VegaSystemManager()
manager.add_comment('Test comment', 'GENERAL')
"

tail -5 ~/VEGA_COMMENTS.txt
```

### Scheduled Tasks Not Running

**Check schedule library:**

```bash
source .venv/bin/activate
pip show schedule
```

**Verify daemon main loop:**

```bash
sudo journalctl -u vega-daemon -f | grep "run_pending"
```

**Test schedule manually:**

```python
from src.vega.daemon.daemon import VegaDaemon
daemon = VegaDaemon()
daemon.schedule_tasks()
# Check schedule.jobs
import schedule
print(schedule.jobs)
```

## Security Considerations

### Sudo Permissions

The daemon requires sudo permissions for:

- `systemctl start/stop/restart vega` - Server control
- `apt update/upgrade/autoremove/clean` - System updates

**Security measures:**

- NOPASSWD only for specific commands (not full sudo)
- Sudoers file in `/etc/sudoers.d/` with 0440 permissions
- No shell access via sudo
- Commands fully qualified with paths

### Service Hardening

The systemd service includes security hardening:

- `NoNewPrivileges=true` - Cannot gain additional privileges
- `PrivateTmp=true` - Private /tmp directory
- `ProtectSystem=strict` - Read-only system directories
- `ProtectHome=read-only` - Limited home directory access
- `ReadWritePaths` - Explicit write permissions for Vega and home

### Network Security

- Vega server binds to `127.0.0.1` (localhost only)
- No external network access by default
- API key required for data endpoints

### File Permissions

- Log files owned by user, not root
- `.vega` directory user-accessible only
- Service files owned by root, 644 permissions

## Advanced Configuration

### Adjusting Schedules

Edit `src/vega/daemon/daemon.py`:

```python
def schedule_tasks(self):
    # Change health check frequency
    schedule.every(10).minutes.do(self.health_check)  # Changed from 5 to 10
    
    # Change update check frequency
    schedule.every(12).hours.do(self.check_updates)  # Changed from 6 to 12
    
    # Change cleanup time
    schedule.every().day.at("04:00").do(self.daily_cleanup)  # Changed from 3 AM to 4 AM
```

Restart daemon after changes:

```bash
sudo systemctl restart vega-daemon
```

### Custom Health Thresholds

Edit `src/vega/daemon/system_manager.py`:

```python
def monitor_health(self):
    # Adjust thresholds
    if cpu_percent > 90:  # Changed from 80
        suggestions.append("High CPU usage...")
    
    if memory_percent > 90:  # Changed from 85
        suggestions.append("High memory usage...")
    
    if disk_percent > 95:  # Changed from 90
        suggestions.append("Low disk space...")
```

### Disable Specific Features

Comment out scheduled tasks in `src/vega/daemon/daemon.py`:

```python
def schedule_tasks(self):
    schedule.every(5).minutes.do(self.health_check)
    schedule.every(6).hours.do(self.check_updates)
    # schedule.every().day.at("03:00").do(self.daily_cleanup)  # Disabled
    # schedule.every().sunday.at("02:00").do(self.weekly_update)  # Disabled
    schedule.every().monday.at("09:00").do(self.weekly_report)
```

## Monitoring Dashboard

You can monitor the daemon status in real-time:

```bash
# Watch service status
watch -n 5 'sudo systemctl status vega vega-daemon'

# Follow all logs in separate terminals
tail -f ~/vega_system.log
tail -f ~/VEGA_COMMENTS.txt
tail -f ~/vega_server.log
tail -f ~/vega_daemon.log

# Monitor resource usage
watch -n 2 'ps aux | grep -E "(vega|python)" | grep -v grep'
```

## Uninstallation

To remove the daemon system:

```bash
# Stop services
sudo systemctl stop vega-daemon
sudo systemctl stop vega

# Disable auto-start
sudo systemctl disable vega-daemon
sudo systemctl disable vega

# Remove service files
sudo rm /etc/systemd/system/vega-daemon.service
sudo rm /etc/systemd/system/vega.service

# Remove sudoers file
sudo rm /etc/sudoers.d/vega-daemon

# Reload systemd
sudo systemctl daemon-reload

# Optional: Remove logs and state
rm ~/vega_*.log
rm ~/VEGA_COMMENTS.txt
rm -rf ~/.vega
```

## Additional Resources

- [Vega Documentation](../../README.md)
- [Architecture Overview](ARCHITECTURE.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)

## Support

If you encounter issues not covered in this guide:

1. Check log files for detailed error messages
2. Review systemd journal: `sudo journalctl -u vega-daemon -n 100`
3. Test system manager manually with CLI commands
4. Verify all dependencies installed: `pip list`
5. Check file permissions in home directory
6. Report issues with full logs attached

---

**Last Updated:** 2025-01-17  
**Version:** 1.0.0  
**Status:** Production Ready
