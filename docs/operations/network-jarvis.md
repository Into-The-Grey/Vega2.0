# Vega Operations Guide

This comprehensive guide covers network access, persistent mode (JARVIS mode), and operational configurations for the Vega system.

## Network Access Configuration

### Overview

The Vega dashboard is configured to be accessible from **any device on your local network**:

- **Port**: 8080 (doesn't conflict with other services on 80/443)
- **Binding**: 0.0.0.0 (all network interfaces)
- **No router changes needed**: Everything stays on your local network

### Finding Your Server IP

On the server, run:

```bash
hostname -I | awk '{print $1}'
```

Example output: `192.168.1.100`

### Access URLs

Replace `192.168.1.100` with your actual server IP:

**From Any Device on Network:**

```
http://192.168.1.100:8080
```

**Alternative Access Methods:**

```bash
# Using server hostname (if configured)
http://vega-server:8080

# Using localhost (server only)
http://localhost:8080

# Using loopback
http://127.0.0.1:8080
```

### Network Troubleshooting

#### Can't Access from Other Devices

1. **Check Firewall:**

   ```bash
   # Ubuntu/Debian
   sudo ufw allow 8080
   sudo ufw status
   
   # CentOS/RHEL
   sudo firewall-cmd --permanent --add-port=8080/tcp
   sudo firewall-cmd --reload
   ```

2. **Verify Service Binding:**

   ```bash
   sudo netstat -tlnp | grep :8080
   # Should show 0.0.0.0:8080, not 127.0.0.1:8080
   ```

3. **Test Network Connectivity:**

   ```bash
   # From another device
   ping 192.168.1.100
   telnet 192.168.1.100 8080
   ```

#### Service Not Responding

1. **Check Service Status:**

   ```bash
   systemctl status vega-dashboard
   journalctl -u vega-dashboard -f
   ```

2. **Restart Services:**

   ```bash
   sudo systemctl restart vega-dashboard
   ```

3. **Check Port Conflicts:**

   ```bash
   sudo lsof -i :8080
   ```

## JARVIS Mode (Persistent Mode)

### What is JARVIS Mode?

JARVIS Mode transforms Vega into a persistent, always-available AI assistant:

- **Always Running**: 100% uptime with automatic restarts
- **Never Forgets**: Single continuous conversation that persists forever
- **Never Crashes**: Proactive memory management prevents OOM errors
- **Seamless Context**: Picks up exactly where you left off, always

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
```

### JARVIS Mode Setup

#### Quick Setup

```bash
# Run the setup script
bash /home/ncacord/Vega2.0/scripts/setup/setup_persistent_mode.sh
```

#### Manual Setup

1. **Configure Persistent Session:**

   ```bash
   # Edit configuration
   nano /home/ncacord/Vega2.0/config/app.yaml
   
   # Add persistent mode settings
   jarvis_mode:
     enabled: true
     session_id: "jarvis-main"
     memory_limit: "8GB"
     auto_cleanup: true
   ```

2. **Enable Auto-Start Services:**

   ```bash
   sudo systemctl enable vega
   sudo systemctl enable vega-dashboard
   ```

3. **Start Services:**

   ```bash
   sudo systemctl start vega
   sudo systemctl start vega-dashboard
   ```

### JARVIS Mode Features

#### Persistent Memory

- Single continuous conversation thread
- Automatic context preservation across restarts
- Intelligent memory summarization for long conversations
- No conversation loss during system updates

#### Proactive Management

- Automatic memory monitoring and cleanup
- Preemptive restart before OOM conditions
- Background health checks and self-healing
- Performance optimization based on usage patterns

#### Seamless Experience

- Zero-downtime during memory management
- Instant reconnection after restarts
- Context-aware responses using full history
- Cross-device conversation synchronization

### Managing JARVIS Mode

#### Starting/Stopping JARVIS Mode

```bash
# Start JARVIS mode
bash /home/ncacord/Vega2.0/scripts/setup/setup_persistent_mode.sh start

# Stop JARVIS mode
bash /home/ncacord/Vega2.0/scripts/setup/setup_persistent_mode.sh stop

# Restart JARVIS mode
bash /home/ncacord/Vega2.0/scripts/setup/setup_persistent_mode.sh restart

# Check status
bash /home/ncacord/Vega2.0/scripts/setup/setup_persistent_mode.sh status
```

#### Monitoring JARVIS Mode

```bash
# View real-time status
curl http://localhost:8000/jarvis/status

# Check memory usage
curl http://localhost:8000/jarvis/memory

# View conversation statistics
curl http://localhost:8000/jarvis/stats
```

#### JARVIS Mode Logs

```bash
# View JARVIS logs
journalctl -u vega -f --grep="JARVIS"

# Memory management logs
tail -f logs/jarvis_memory.log

# Session persistence logs
tail -f logs/jarvis_session.log
```

### Memory Management

#### Automatic Memory Management

JARVIS mode includes sophisticated memory management:

1. **Proactive Monitoring:**
   - Continuous RAM and VRAM usage tracking
   - Predictive memory growth analysis
   - Early warning system for potential OOM

2. **Intelligent Cleanup:**
   - Automatic conversation summarization
   - Context-preserving memory reduction
   - Graceful conversation archiving

3. **Dynamic Scaling:**
   - Adaptive response length based on memory
   - Smart context window management
   - Background memory optimization

#### Manual Memory Management

```bash
# Force memory cleanup
curl -X POST http://localhost:8000/jarvis/cleanup

# Get memory statistics
curl http://localhost:8000/jarvis/memory/stats

# Archive old conversations
curl -X POST http://localhost:8000/jarvis/archive
```

### Configuration Options

#### JARVIS Mode Configuration

Edit `config/app.yaml`:

```yaml
jarvis_mode:
  enabled: true
  session_id: "jarvis-main"
  
  # Memory Management
  memory_limit: "8GB"
  cleanup_threshold: 0.8  # Cleanup at 80% memory usage
  auto_cleanup: true
  cleanup_interval: 300   # 5 minutes
  
  # Persistence
  save_interval: 60       # Save every minute
  backup_interval: 3600   # Backup every hour
  max_backups: 24         # Keep 24 hourly backups
  
  # Performance
  response_cache: true
  context_optimization: true
  background_tasks: true
  
  # Voice Integration
  voice_enabled: true
  voice_activation: "hey jarvis"
  voice_response: true
```

#### Advanced Configuration

```yaml
jarvis_mode:
  # Health Monitoring
  health_checks:
    enabled: true
    interval: 30
    memory_threshold: 0.9
    response_time_threshold: 5.0
  
  # Auto-Recovery
  auto_restart:
    enabled: true
    max_memory_usage: 0.95
    max_response_time: 30.0
    restart_cooldown: 300
  
  # Conversation Management
  conversation:
    max_context_length: 32000
    summarization_threshold: 20000
    archive_after_days: 30
```

## System Integration

### Service Dependencies

JARVIS mode requires multiple services working together:

```bash
# Check all service status
systemctl status vega vega-dashboard

# View service dependencies
systemctl list-dependencies vega
```

### Service Start Order

1. **System Services** (automatic)
2. **Vega Core** (`vega.service`)
3. **Dashboard** (`vega-dashboard.service`)
4. **JARVIS Mode** (part of Vega Core)

### Health Monitoring

#### System Health Endpoints

```bash
# Core system health
curl http://localhost:8000/healthz

# JARVIS mode health
curl http://localhost:8000/jarvis/health

# Dashboard health
curl http://localhost:8080/health
```

#### Monitoring Scripts

```bash
# Comprehensive health check
bash /home/ncacord/Vega2.0/scripts/setup/check_jarvis_health.sh

# Memory monitoring
watch -n 10 'curl -s http://localhost:8000/jarvis/memory | jq'

# Performance monitoring
bash /home/ncacord/Vega2.0/scripts/setup/monitor_jarvis.sh
```

## Troubleshooting

### Common Issues

#### JARVIS Mode Won't Start

1. **Check Dependencies:**

   ```bash
   systemctl status vega
   journalctl -u vega --no-pager -l
   ```

2. **Verify Configuration:**

   ```bash
   python -c "
   import yaml
   with open('config/app.yaml', 'r') as f:
       config = yaml.safe_load(f)
       print('JARVIS enabled:', config.get('jarvis_mode', {}).get('enabled', False))
   "
   ```

3. **Check Memory Requirements:**

   ```bash
   free -h
   # Ensure at least 8GB available RAM
   ```

#### Memory Issues

1. **Monitor Memory Usage:**

   ```bash
   # System memory
   free -h
   
   # Process memory
   ps aux | grep vega | awk '{print $6}' | awk '{sum+=$1} END {print sum/1024 " MB"}'
   
   # JARVIS memory
   curl -s http://localhost:8000/jarvis/memory
   ```

2. **Force Memory Cleanup:**

   ```bash
   curl -X POST http://localhost:8000/jarvis/cleanup
   systemctl restart vega
   ```

#### Connection Issues

1. **Check Network Connectivity:**

   ```bash
   # Test local connection
   curl http://localhost:8000/healthz
   
   # Test network connection
   curl http://192.168.1.100:8000/healthz
   ```

2. **Verify Port Configuration:**

   ```bash
   sudo netstat -tlnp | grep -E ":(8000|8080)"
   ```

### Performance Optimization

#### Memory Optimization

```bash
# Optimize Python memory usage
export PYTHONMALLOC=malloc
export MALLOC_ARENA_MAX=2

# Start with optimizations
systemctl restart vega
```

#### Network Optimization

```bash
# Increase network buffers
echo 'net.core.rmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## Backup and Recovery

### JARVIS Mode Backups

#### Automatic Backups

JARVIS mode automatically creates:

- **Conversation Backups**: Every hour
- **Configuration Backups**: Before changes
- **State Snapshots**: Every 6 hours

#### Manual Backup

```bash
# Create manual backup
curl -X POST http://localhost:8000/jarvis/backup

# Export conversation history
curl http://localhost:8000/jarvis/export > jarvis_backup_$(date +%Y%m%d).json
```

#### Restore from Backup

```bash
# List available backups
curl http://localhost:8000/jarvis/backups

# Restore from specific backup
curl -X POST http://localhost:8000/jarvis/restore -d '{"backup_id": "20251029_120000"}'
```

### Configuration Backup

```bash
# Backup entire configuration
tar -czf vega_config_backup_$(date +%Y%m%d).tar.gz config/ data/

# Restore configuration
tar -xzf vega_config_backup_20251029.tar.gz
systemctl restart vega vega-dashboard
```

## Security Considerations

### Network Security

- **Local Network Only**: Services bind to all interfaces but remain on local network
- **No External Exposure**: Firewall rules prevent external access by default
- **API Security**: All API endpoints require authentication tokens

### Data Security

- **Conversation Encryption**: All stored conversations are encrypted at rest
- **Secure Backups**: Backups include encryption and integrity verification
- **Privacy Protection**: No data sent to external services without explicit consent

### Access Control

```bash
# Restrict dashboard access to specific IPs
sudo ufw allow from 192.168.1.0/24 to any port 8080

# Enable API key authentication
echo "API_KEY_REQUIRED=true" >> config/.env
```

## Performance Metrics

### System Requirements

**Minimum Requirements:**

- RAM: 8GB
- CPU: 4 cores
- Storage: 50GB
- Network: 100Mbps

**Recommended Requirements:**

- RAM: 32GB
- CPU: 8+ cores
- Storage: 200GB SSD
- Network: 1Gbps

### Performance Benchmarks

**Response Times:**

- Simple queries: <1 second
- Complex analysis: 2-5 seconds
- Voice processing: 1-3 seconds

**Memory Usage:**

- Base system: 2-4GB
- With active conversation: 4-8GB
- Peak usage: 8-12GB

**Network Usage:**

- Dashboard: <1MB/hour
- API calls: 1-10KB per request
- Voice data: 100KB-1MB per minute

## Files Reference

### Configuration Files

- **Main Config**: `config/app.yaml`
- **Network Config**: `config/network.yaml`
- **JARVIS Config**: `config/jarvis.yaml`

### Service Files

- **Vega Service**: `systemd/vega.service`
- **Dashboard Service**: `systemd/vega-dashboard.service`

### Scripts

- **Setup Script**: `scripts/setup/setup_persistent_mode.sh`
- **Health Check**: `scripts/setup/check_jarvis_health.sh`
- **Monitor Script**: `scripts/setup/monitor_jarvis.sh`

### Log Files

- **System Logs**: `journalctl -u vega`
- **JARVIS Logs**: `logs/jarvis.log`
- **Memory Logs**: `logs/jarvis_memory.log`
- **Network Logs**: `logs/network.log`

---

**Last Updated**: October 29, 2025
**Version**: 2.0
**Maintainer**: Vega Development Team
