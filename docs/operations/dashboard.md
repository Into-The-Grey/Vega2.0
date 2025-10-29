# Vega Dashboard Operations Guide

This comprehensive guide covers all aspects of the Vega Dashboard, including setup, troubleshooting, and maintenance.

## Quick Start

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
sudo bash /home/ncacord/Vega2.0/scripts/dashboard/setup_dashboard.sh

# Management commands
bash /home/ncacord/Vega2.0/scripts/dashboard/manage_dashboard.sh start
bash /home/ncacord/Vega2.0/scripts/dashboard/manage_dashboard.sh stop
bash /home/ncacord/Vega2.0/scripts/dashboard/manage_dashboard.sh status
```

## Overview

The Vega Dashboard provides a web-based interface for:

- 📊 **System Monitoring**: Real-time status of Vega services
- 🚀 **Service Management**: Start/stop/restart Vega components
- 📈 **Performance Metrics**: Memory, CPU, and resource usage
- 🔧 **Configuration**: Manage system settings
- 📱 **Mobile-Friendly**: Responsive design for all devices

## Architecture

### Main Components

- **Dashboard Application**: `tools/vega/vega_dashboard.py` (1,117+ lines)
- **Service Configuration**: `systemd/vega-dashboard.service`
- **Setup Scripts**: Located in `scripts/dashboard/`
- **Web UI**: HTML/CSS/JavaScript interface

### Key Features

✅ **Real-time Status Monitoring**
✅ **Service Control Interface** 
✅ **System Resource Tracking**
✅ **Mobile-Responsive Design**
✅ **Auto-refresh Capabilities**
✅ **Error Logging & Diagnostics**

## Installation & Setup

### Prerequisites

- Python 3.12+ with virtual environment
- systemd (for service management)
- Root/sudo access for system service installation

### Installation Steps

1. **Run Setup Script**:

   ```bash
   sudo bash /home/ncacord/Vega2.0/scripts/dashboard/setup_dashboard.sh
   ```

2. **Verify Installation**:

   ```bash
   systemctl status vega-dashboard
   ```

3. **Access Dashboard**:
   - Local: <http://localhost:8080>
   - Network: <http://YOUR_IP:8080>

### Manual Installation

If the setup script fails, follow these manual steps:

1. **Copy Service File**:

   ```bash
   sudo cp /home/ncacord/Vega2.0/systemd/vega-dashboard.service /etc/systemd/system/
   ```

2. **Reload systemd**:

   ```bash
   sudo systemctl daemon-reload
   ```

3. **Enable and Start Service**:

   ```bash
   sudo systemctl enable vega-dashboard
   sudo systemctl start vega-dashboard
   ```

## Service Management

### Using Management Script

The `manage_dashboard.sh` script provides comprehensive service control:

```bash
# Start dashboard
bash /home/ncacord/Vega2.0/scripts/dashboard/manage_dashboard.sh start

# Stop dashboard  
bash /home/ncacord/Vega2.0/scripts/dashboard/manage_dashboard.sh stop

# Restart dashboard
bash /home/ncacord/Vega2.0/scripts/dashboard/manage_dashboard.sh restart

# Check status
bash /home/ncacord/Vega2.0/scripts/dashboard/manage_dashboard.sh status

# View logs
bash /home/ncacord/Vega2.0/scripts/dashboard/manage_dashboard.sh logs
```

### Direct systemctl Commands

```bash
# Service control
sudo systemctl start vega-dashboard
sudo systemctl stop vega-dashboard
sudo systemctl restart vega-dashboard
sudo systemctl status vega-dashboard

# Enable/disable autostart
sudo systemctl enable vega-dashboard
sudo systemctl disable vega-dashboard

# View logs
journalctl -u vega-dashboard -f
```

## Troubleshooting

### Memory Limitation Issues

**Problem**: The Vega system has heavy dependencies (torch, transformers, etc.) that cause memory allocation errors when starting programmatically.

**Error**: `terminate called after throwing an instance of 'std::bad_alloc'`

**Root Cause**: 

- Python's memory management limitations
- Torch/CUDA initialization requirements  
- High system memory usage preventing contiguous allocation

**Solutions**:

1. **Increase Available Memory**:

   ```bash
   # Clear system cache
   sudo sync && sudo sysctl vm.drop_caches=3
   
   # Check memory usage
   free -h
   htop
   ```

2. **Use Alternative Start Methods**:

   ```bash
   # Manual terminal start (recommended)
   cd /home/ncacord/Vega2.0
   source .venv/bin/activate
   python main.py server --host 0.0.0.0 --port 8000
   ```

3. **Modify Memory Settings**:

   ```bash
   # Add to .env file
   echo "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512" >> .env
   ```

### Common Issues & Solutions

#### Dashboard Won't Start

1. **Check Service Status**:

   ```bash
   systemctl status vega-dashboard
   journalctl -u vega-dashboard --no-pager -l
   ```

2. **Verify Python Environment**:

   ```bash
   /home/ncacord/Vega2.0/.venv/bin/python --version
   ```

3. **Check Port Availability**:

   ```bash
   sudo netstat -tlnp | grep :8080
   ```

#### Dashboard Accessible But Features Don't Work

1. **Check Vega Service Status**: Ensure main Vega service is running
2. **Verify API Connectivity**: Test API endpoints manually
3. **Check Browser Console**: Look for JavaScript errors

#### High Memory Usage

1. **Monitor Resources**:

   ```bash
   # Check dashboard memory usage
   ps aux | grep vega_dashboard
   
   # Monitor system resources
   htop
   ```

2. **Restart Services**:

   ```bash
   sudo systemctl restart vega-dashboard
   ```

## Configuration

### Dashboard Settings

Key configuration options in `tools/vega/vega_dashboard.py`:

- **Host**: Default `0.0.0.0` (all interfaces)
- **Port**: Default `8080`
- **Auto-refresh**: 30-second intervals
- **Theme**: Dark/light mode support

### Service Configuration

The systemd service file (`systemd/vega-dashboard.service`) includes:

```ini
[Unit]
Description=Vega Dashboard Web Interface
After=network.target

[Service]
Type=simple
User=ncacord
WorkingDirectory=/home/ncacord/Vega2.0
Environment=PATH=/home/ncacord/Vega2.0/.venv/bin
ExecStart=/home/ncacord/Vega2.0/.venv/bin/python tools/vega/vega_dashboard.py --host 0.0.0.0 --port 8080
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Network Configuration

For remote access, ensure firewall allows port 8080:

```bash
# UFW (Ubuntu)
sudo ufw allow 8080

# firewalld (CentOS/RHEL)
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload
```

## Monitoring & Maintenance

### Health Checks

1. **Dashboard Endpoint**: <http://localhost:8080/health>
2. **Service Status**: `systemctl is-active vega-dashboard`
3. **Log Monitoring**: `journalctl -u vega-dashboard -f`

### Regular Maintenance

1. **Weekly**: Check log files for errors
2. **Monthly**: Review system resource usage
3. **As Needed**: Update dashboard code and restart service

### Log Management

Logs are managed by systemd and can be accessed via:

```bash
# Recent logs
journalctl -u vega-dashboard --since "1 hour ago"

# Follow logs in real-time
journalctl -u vega-dashboard -f

# Export logs
journalctl -u vega-dashboard --since "1 day ago" > dashboard.log
```

## API Integration

The dashboard integrates with Vega's main API endpoints:

- **Health Check**: `GET /healthz`
- **Service Control**: `POST /control/{action}`
- **System Stats**: `GET /stats`
- **Configuration**: `GET/POST /config`

### Adding Custom Features

To extend dashboard functionality:

1. **Modify Dashboard Code**: Edit `tools/vega/vega_dashboard.py`
2. **Add API Endpoints**: Update main Vega API in `src/vega/core/app.py`
3. **Update Frontend**: Modify HTML/CSS/JavaScript sections
4. **Restart Service**: `sudo systemctl restart vega-dashboard`

## Security Considerations

- Dashboard runs on all interfaces (`0.0.0.0`) - consider firewall rules
- No authentication by default - implement if needed for production
- Logs may contain sensitive information - review log retention policies
- Service runs as user `ncacord` - ensure proper permissions

## Performance Optimization

### Memory Management

- Monitor dashboard memory usage regularly
- Implement log rotation to prevent disk space issues
- Consider resource limits in systemd service file

### Network Optimization

- Use nginx proxy for production deployments
- Implement caching for static assets
- Consider HTTPS for secure deployments

## Backup & Recovery

### Configuration Backup

```bash
# Backup service file
cp /etc/systemd/system/vega-dashboard.service ~/vega-dashboard.service.backup

# Backup dashboard code
tar -czf dashboard-backup.tar.gz tools/vega/vega_dashboard.py
```

### Recovery Procedures

1. **Service Recovery**:

   ```bash
   sudo systemctl stop vega-dashboard
   sudo cp ~/vega-dashboard.service.backup /etc/systemd/system/vega-dashboard.service
   sudo systemctl daemon-reload
   sudo systemctl start vega-dashboard
   ```

2. **Code Recovery**:

   ```bash
   tar -xzf dashboard-backup.tar.gz
   sudo systemctl restart vega-dashboard
   ```

## Files Reference

### Created/Modified Files

- **Main Dashboard**: `tools/vega/vega_dashboard.py` (1,117+ lines)
- **Service Configuration**: `systemd/vega-dashboard.service` (59 lines)
- **Setup Script**: `scripts/dashboard/setup_dashboard.sh` (67 lines)
- **Management Script**: `scripts/dashboard/manage_dashboard.sh` (385+ lines)
- **Status Display**: `scripts/dashboard/show_dashboard_url.sh`

### Key Directories

- `/scripts/dashboard/`: All dashboard-related scripts
- `/tools/vega/`: Dashboard application code
- `/systemd/`: Service configuration files
- `/docs/operations/`: This documentation

---

**Last Updated**: October 29, 2025
**Version**: 2.0
**Maintainer**: Vega Development Team

