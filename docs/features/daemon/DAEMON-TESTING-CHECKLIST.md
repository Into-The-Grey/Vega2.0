# Vega Daemon System - Testing Checklist

## Pre-Installation Checks

- [ ] Python 3.12+ installed
- [ ] Virtual environment exists at `.venv/`
- [ ] Vega installed at `/home/ncacord/Vega2.0`
- [ ] User has sudo access
- [ ] Ubuntu/Debian system with systemd
- [ ] Sufficient disk space (>5GB free)

## Installation Testing

### Run Installer

```bash
cd /home/ncacord/Vega2.0
sudo ./scripts/install_daemon.sh
```

**Expected Output:**

- [ ] "Configuration:" section shows correct user/home/Vega paths
- [ ] "Installing Python dependencies..." succeeds (schedule, psutil)
- [ ] "Configuring systemd services..." creates vega.service
- [ ] "Installing systemd services..." copies to /etc/systemd/system/
- [ ] "Configuring sudo permissions..." creates /etc/sudoers.d/vega-daemon
- [ ] "Enabling services..." enables both services
- [ ] "Starting services..." starts both services
- [ ] Service status shows "active (running)" in green

### Verify Installation

```bash
# Check services installed
ls -la /etc/systemd/system/vega*.service
# Expected: vega.service, vega-daemon.service

# Check sudoers file
sudo cat /etc/sudoers.d/vega-daemon
# Expected: 8 NOPASSWD entries for systemctl and apt

# Check directories created
ls -la ~/.vega/
# Expected: system_state.json, temp/ directory

# Check services enabled
systemctl is-enabled vega
systemctl is-enabled vega-daemon
# Expected: "enabled" for both

# Check services running
systemctl is-active vega
systemctl is-active vega-daemon
# Expected: "active" for both
```

**Results:**

- [ ] Service files exist in /etc/systemd/system/
- [ ] Sudoers file exists with correct permissions (0440)
- [ ] ~/.vega directory created
- [ ] Both services enabled
- [ ] Both services running

## Log File Testing

### Verify Log Creation

```bash
# Check log files exist
ls -la ~/ | grep vega
# Expected files:
# - vega_system.log
# - VEGA_COMMENTS.txt
# - vega_server.log
# - vega_daemon.log
# - vega_server_error.log (may be empty)
# - vega_daemon_error.log (may be empty)

# Check log content
head -20 ~/vega_system.log
head -20 ~/VEGA_COMMENTS.txt
head -20 ~/vega_daemon.log
```

**Results:**

- [ ] vega_system.log exists and has content
- [ ] VEGA_COMMENTS.txt exists with startup comment
- [ ] vega_server.log exists (may be empty if server uses different logging)
- [ ] vega_daemon.log exists with daemon startup messages
- [ ] Error logs exist (empty is OK)

### Monitor Real-time Logging

```bash
# Terminal 1: System log
tail -f ~/vega_system.log

# Terminal 2: AI comments
tail -f ~/VEGA_COMMENTS.txt

# Terminal 3: Daemon log
tail -f ~/vega_daemon.log
```

**Expected Behavior:**

- [ ] System log updates with health check every 5 minutes
- [ ] Comments file gets health check entries
- [ ] Daemon log shows "Running scheduled tasks..." messages

## CLI Testing

### Daemon Commands

```bash
# Status
vega daemon status
# Expected: Shows systemd status, "active (running)" in green

# Logs (last 20 lines)
vega daemon logs --lines 20
# Expected: Shows last 20 lines from ~/vega_system.log

# Comments (last 10 lines)
vega daemon comments --lines 10
# Expected: Shows last 10 lines from ~/VEGA_COMMENTS.txt

# Restart
vega daemon restart
# Expected: "✅ Daemon restarted", then shows status

# Stop (test only - will disable daemon)
# vega daemon stop
# Expected: "✅ Daemon stopped"

# Start (after stop test)
# vega daemon start
# Expected: "✅ Daemon started successfully", shows status
```

**Results:**

- [ ] `vega daemon status` works
- [ ] `vega daemon logs` shows log content
- [ ] `vega daemon comments` shows comments
- [ ] `vega daemon restart` restarts service
- [ ] (Optional) Stop/start cycle works

### System Commands

```bash
# Health check
vega system health
# Expected: Shows server status, CPU/memory/disk percentages

# Server status
vega system server status
# Expected: Shows "Running: Yes", uptime, restart count

# Check for updates
vega system update
# Expected: Lists available system/Python updates, or "no updates"

# Cleanup (test only - will run cleanup)
# vega system cleanup
# Expected: "✅ Cleanup complete", shows space freed in log

# Server restart (test only - will restart server)
# vega system server restart
# Expected: "✅ Server restarted"
```

**Results:**

- [ ] `vega system health` shows metrics
- [ ] `vega system server status` shows server info
- [ ] `vega system update` checks for updates
- [ ] (Optional) `vega system cleanup` runs cleanup
- [ ] (Optional) Server restart works

## Scheduled Task Testing

### Health Check (5 minutes)

**Test:** Wait 15 minutes, observe 3 health checks

```bash
# Watch system log
tail -f ~/vega_system.log | grep "Health check"

# Or check count
grep "Health check" ~/vega_system.log | tail -10
```

**Expected:**

- [ ] Health check runs every 5 minutes (:00, :05, :10, :15, etc.)
- [ ] Each check logs CPU/memory/disk percentages
- [ ] Server running status checked
- [ ] Comments file updated with health status

**Time Window:** 15 minutes  
**Start Time:** ________  
**End Time:** ________  
**Health Checks Observed:** ___ (expected: 3)

### Update Check (6 hours)

**Test:** Monitor for update check (or modify schedule for testing)

**Option 1: Wait for natural occurrence**

```bash
# Watch for update check
grep "check_updates" ~/vega_system.log | tail -5
```

**Option 2: Modify schedule for testing**

```python
# Edit src/vega/daemon/daemon.py
# Change: schedule.every(6).hours.do(self.check_updates)
# To:     schedule.every(2).minutes.do(self.check_updates)
# Restart daemon: vega daemon restart
# Wait 2 minutes, check logs
# Restore original schedule
```

**Expected:**

- [ ] Update check runs at scheduled time
- [ ] Logs available system packages
- [ ] Logs available Python packages
- [ ] Checks Vega git status
- [ ] Comments file updated with summary

**Result:** ☐ Passed ☐ Failed ☐ Skipped

### Daily Cleanup (3 AM)

**Test:** Monitor tomorrow at 3 AM, or modify schedule

**Option 1: Wait for 3 AM**

```bash
# Next morning, check logs
grep "daily_cleanup" ~/vega_system.log | tail -1
grep "SYSTEM_CLEANUP" ~/VEGA_COMMENTS.txt | tail -20
```

**Option 2: Modify schedule for immediate test**

```python
# Edit src/vega/daemon/daemon.py
# Change: schedule.every().day.at("03:00").do(self.daily_cleanup)
# To:     schedule.every().day.at("HH:MM").do(self.daily_cleanup)
# Where HH:MM is 2 minutes from now
# Restart daemon: vega daemon restart
# Wait for scheduled time
# Restore original schedule
```

**Expected:**

- [ ] Cleanup runs at scheduled time
- [ ] apt autoremove executed
- [ ] apt clean executed
- [ ] Log rotation checked
- [ ] Temp files cleaned
- [ ] Space freed calculated and logged
- [ ] Comments file updated with cleanup results

**Result:** ☐ Passed ☐ Failed ☐ Skipped

### Weekly Tasks

**Weekly Update (Sunday 2 AM):**

- [ ] System update (apt)
- [ ] Python update (pip)
- [ ] Vega update (git)
- [ ] Server restart
- [ ] Cleanup

**Weekly Report (Monday 9 AM):**

- [ ] Restart count reported
- [ ] Update count reported
- [ ] Cleanup count reported
- [ ] Health metrics included
- [ ] Recommendations provided

**Note:** Can modify schedule for testing, or wait for natural occurrence

## Auto-Recovery Testing

### Server Auto-Restart Test

**Test:** Kill server, verify daemon restarts it within 5 minutes

```bash
# Get Vega server PID
ps aux | grep "main.py server" | grep -v grep

# Kill server
sudo systemctl stop vega

# Monitor logs for auto-restart
tail -f ~/vega_system.log | grep -E "(server|Server)"

# Or check every minute
watch -n 60 'systemctl is-active vega'
```

**Expected:**

- [ ] Daemon detects server is down in next health check (<5 min)
- [ ] Daemon logs "Server is not running, starting..."
- [ ] Server auto-starts
- [ ] Comments file updated with restart notification

**Test Start:** ________  
**Server Stopped:** ________  
**Auto-Restart Detected:** ________ (should be <5 min)  
**Result:** ☐ Passed ☐ Failed

### Emergency Cleanup Test

**Test:** Simulate high disk usage (or test with manual trigger)

**Option 1: Manual trigger**

```bash
# Trigger health check manually
python -m src.vega.daemon.system_manager health

# Check if cleanup suggested
grep "disk" ~/vega_system.log | tail -5
```

**Option 2: Fill disk (CAUTION)**

```bash
# Create large file to fill disk to >95%
# dd if=/dev/zero of=~/testfile.bin bs=1G count=X
# Where X fills disk to >95%
# Monitor logs for emergency cleanup
# tail -f ~/vega_system.log
# Delete test file after: rm ~/testfile.bin
```

**Expected (if disk >95%):**

- [ ] Health check detects high disk usage
- [ ] Emergency cleanup triggered immediately
- [ ] Cleanup runs regardless of schedule
- [ ] Space freed logged
- [ ] Disk usage drops below critical level

**Result:** ☐ Passed ☐ Failed ☐ Skipped (disk not critical)

## State Persistence Testing

### Restart Count Test

```bash
# Check current restart count
grep "restart_count" ~/.vega/system_state.json

# Restart server via daemon
vega system server restart

# Check incremented count
grep "restart_count" ~/.vega/system_state.json

# Or view via status
vega system server status
```

**Expected:**

- [ ] restart_count increments after each restart
- [ ] Count persists across daemon restarts
- [ ] Status command shows correct count

**Initial Count:** ___  
**After Restart:**___  
**Result:** ☐ Passed ☐ Failed

### State Survival Test

```bash
# Check current state
cat ~/.vega/system_state.json

# Restart daemon
vega daemon restart

# Verify state preserved
cat ~/.vega/system_state.json
```

**Expected:**

- [ ] State file exists before restart
- [ ] State file preserved after restart
- [ ] Counters not reset
- [ ] Timestamps preserved

**Result:** ☐ Passed ☐ Failed

## Update System Testing

### Check Updates

```bash
# Check for system updates
vega system update

# Expected output lists:
# - System packages (apt list --upgradable)
# - Python packages (pip list --outdated)
# - Vega updates (git status)
```

**Results:**

- [ ] System packages listed (or "none")
- [ ] Python packages listed (or "none")
- [ ] Vega status shown
- [ ] No errors

### Manual Update (Optional)

**CAUTION:** This will actually update your system

```bash
# Full update
vega system update --full

# Monitor logs
tail -f ~/vega_system.log

# Expected:
# - apt update && apt upgrade
# - pip install --upgrade for each package
# - git pull for Vega
# - Server restart
# - Cleanup
```

**Expected:**

- [ ] System packages updated
- [ ] Python packages updated
- [ ] Vega code updated
- [ ] Server restarted after updates
- [ ] Cleanup runs after updates
- [ ] All logged to vega_system.log

**Result:** ☐ Passed ☐ Failed ☐ Skipped

## Error Handling Testing

### Invalid Command Test

```bash
# Try invalid action
vega system server invalid-action
# Expected: Error message, exit code 1

# Try unknown daemon command
vega daemon unknown
# Expected: Typer help message or error
```

**Results:**

- [ ] Invalid actions rejected
- [ ] Helpful error messages shown
- [ ] Commands don't crash

### Permission Test

```bash
# Try daemon command without sudo (for systemctl commands)
# Note: daemon logs/comments don't need sudo
vega daemon logs
# Expected: Works (no sudo needed)

vega daemon restart
# Expected: Prompts for sudo password if not in sudoers
```

**Results:**

- [ ] Read commands work without sudo
- [ ] Control commands handled properly

### Missing File Test

```bash
# Rename state file
mv ~/.vega/system_state.json ~/.vega/system_state.json.bak

# Trigger operation that uses state
vega system server status

# Check if state recreated
ls ~/.vega/system_state.json

# Restore backup
mv ~/.vega/system_state.json.bak ~/.vega/system_state.json
```

**Expected:**

- [ ] Missing state file doesn't crash daemon
- [ ] New state created with default values
- [ ] Operations continue normally

**Result:** ☐ Passed ☐ Failed

## Performance Testing

### Resource Usage Check

```bash
# Check daemon memory usage
ps aux | grep daemon | grep -v grep

# Check CPU usage (run for 5 minutes)
top -b -n 60 -d 5 -p $(pgrep -f daemon) | grep daemon

# Or use systemd status
systemctl status vega-daemon | grep Memory
```

**Expected:**

- [ ] Memory usage < 200MB
- [ ] CPU usage < 5% average
- [ ] No memory leaks (stable over time)

**Memory Usage:** _____ MB  
**CPU Usage:** _____ %  
**Result:** ☐ Passed ☐ Failed

### Log File Growth

```bash
# Check log file sizes
du -h ~/vega_system.log
du -h ~/VEGA_COMMENTS.txt
du -h ~/vega_daemon.log

# Run for 24 hours, check again
# Growth should be reasonable (<10MB/day)
```

**Initial Sizes:**

- vega_system.log: _____ KB
- VEGA_COMMENTS.txt: _____ KB
- vega_daemon.log: _____ KB

**After 24 Hours:**

- vega_system.log: _____ KB
- VEGA_COMMENTS.txt: _____ KB
- vega_daemon.log: _____ KB

**Result:** ☐ Passed ☐ Failed ☐ Pending

## Stress Testing

### Rapid Server Restart Test

```bash
# Restart server 10 times rapidly
for i in {1..10}; do
  echo "Restart $i"
  vega system server restart
  sleep 2
done

# Check logs for issues
grep ERROR ~/vega_system.log | tail -20

# Verify restart count
vega system server status
```

**Expected:**

- [ ] All restarts succeed
- [ ] No errors in logs
- [ ] Restart count = 10 higher
- [ ] State file updates correctly

**Result:** ☐ Passed ☐ Failed

### Daemon Crash Recovery

```bash
# Get daemon PID
systemctl status vega-daemon | grep PID

# Kill daemon process (systemd should restart it)
sudo kill -9 <PID>

# Check if restarted
sleep 5
systemctl is-active vega-daemon

# Check logs
tail -20 ~/vega_daemon.log
```

**Expected:**

- [ ] Daemon restarts automatically (Restart=always)
- [ ] No data loss
- [ ] State preserved
- [ ] Scheduled tasks resume

**Result:** ☐ Passed ☐ Failed

## Long-term Stability Testing

### 24-Hour Test

**Monitor for 24 hours:**

```bash
# Start monitoring
echo "Test started: $(date)" >> ~/daemon_test.log

# Check every hour
watch -n 3600 '
  echo "=== $(date) ===" >> ~/daemon_test.log
  systemctl is-active vega vega-daemon >> ~/daemon_test.log
  vega system health >> ~/daemon_test.log 2>&1
  echo "" >> ~/daemon_test.log
'
```

**Checklist:**

- [ ] Daemon runs continuously for 24 hours
- [ ] No crashes or restarts (except scheduled)
- [ ] Health checks run every 5 minutes (288 total)
- [ ] Memory usage stable
- [ ] CPU usage reasonable
- [ ] Log files reasonable size
- [ ] All scheduled tasks execute

**Start:** ________  
**End:** ________  
**Issues:** ________  
**Result:** ☐ Passed ☐ Failed ☐ Pending

### 7-Day Test

**Monitor for 1 week:**

- [ ] Day 1: Installation and basic testing
- [ ] Day 2: Verify daily cleanup ran (3 AM)
- [ ] Day 3: Check health monitoring working
- [ ] Day 4: Verify update checks happening
- [ ] Day 5: Monitor resource usage trends
- [ ] Day 6: Check for weekly update (Sunday 2 AM)
- [ ] Day 7: Verify weekly report (Monday 9 AM)

**Results:**

- Total uptime: _____ hours
- Health checks: _____ (expected: ~2,016)
- Update checks: _____ (expected: 28)
- Cleanups: _____ (expected: 7)
- Updates: _____ (expected: 1)
- Reports: _____ (expected: 1)
- Issues: ________

**Result:** ☐ Passed ☐ Failed ☐ Pending

## Uninstallation Testing (Optional)

**CAUTION:** This will remove the daemon system

```bash
# Stop services
sudo systemctl stop vega-daemon vega

# Disable services
sudo systemctl disable vega-daemon vega

# Remove service files
sudo rm /etc/systemd/system/vega-daemon.service
sudo rm /etc/systemd/system/vega.service

# Remove sudoers file
sudo rm /etc/sudoers.d/vega-daemon

# Reload systemd
sudo systemctl daemon-reload

# Verify removed
systemctl status vega-daemon
# Expected: "could not be found"

# Optional: Remove logs
rm ~/vega_*.log
rm ~/VEGA_COMMENTS.txt
rm -rf ~/.vega
```

**Checklist:**

- [ ] Services stopped
- [ ] Services disabled
- [ ] Service files removed
- [ ] Sudoers file removed
- [ ] systemd reloaded
- [ ] Services no longer exist
- [ ] (Optional) Logs cleaned up

**Result:** ☐ Passed ☐ Failed ☐ Skipped

## Test Summary

### Test Execution

**Date:** ________  
**Tester:** ________  
**Vega Version:** ________  
**System:** ________

### Results

| Category | Tests | Passed | Failed | Skipped |
|----------|-------|--------|--------|---------|
| Installation | 2 | ___ | ___ | ___ |
| Log Files | 2 | ___ | ___ | ___ |
| CLI | 2 | ___ | ___ | ___ |
| Scheduled Tasks | 4 | ___ | ___ | ___ |
| Auto-Recovery | 2 | ___ | ___ | ___ |
| State Persistence | 2 | ___ | ___ | ___ |
| Updates | 2 | ___ | ___ | ___ |
| Error Handling | 3 | ___ | ___ | ___ |
| Performance | 2 | ___ | ___ | ___ |
| Stress Testing | 2 | ___ | ___ | ___ |
| Long-term | 2 | ___ | ___ | ___ |
| **TOTAL** | **25** | **___** | **___** | **___** |

### Overall Status

☐ **PASSED** - All critical tests passed  
☐ **PASSED WITH WARNINGS** - Minor issues noted  
☐ **FAILED** - Critical issues require fixes  
☐ **INCOMPLETE** - Testing in progress

### Issues Found

1. ________________________________
2. ________________________________
3. ________________________________

### Recommendations

1. ________________________________
2. ________________________________
3. ________________________________

### Sign-off

**Tested By:** ________________  
**Date:** ________________  
**Signature:** ________________

---

**Notes:**

- Mark tests with ☑ (passed), ☐ (not tested), or ☒ (failed)
- Document any issues in detail
- Keep this checklist with test results for reference
