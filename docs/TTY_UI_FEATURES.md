# TTY UI Features - Complete Enhancement Guide

## Overview

The Vega TTY Status UI provides a comprehensive, real-time monitoring interface optimized for KVM rackmount consoles and terminal environments.

## Auto-Detection Features

### KVM Console Detection

The system automatically detects KVM environments using multiple heuristics:

- **Terminal Type**: Checks `TERM` environment variable for KVM-specific values
- **SSH Detection**: Identifies SSH sessions that may indicate KVM access
- **TTY Analysis**: Examines TTY device patterns typical of KVM consoles
- **Terminal Size**: Considers screen dimensions common in rackmount environments

### Force Override

Use `VEGA_FORCE_KVM=1` to force KVM mode regardless of auto-detection.

## Enhanced Status Display

### Real-Time Metrics

- **Component Status**: Live health monitoring of all Vega ecosystem components
- **Process Information**: PID tracking, uptime calculation, process status
- **System Metrics**: CPU usage, memory consumption, temperature sensors
- **Health Summary**: Aggregated component health statistics

### Component Tracking

Each component displays:

- Health status (OK/!! indicators) 
- Process ID (PID)
- Uptime (HH:MM:SS format)
- Current status (Running/Not Running/Error)
- Full component name and type

## Interactive Controls

### Navigation Keys

- **`q`**: Quit and save preferences
- **`r`**: Force refresh now
- **`h`**: Toggle help overlay
- **`l`**: Toggle log tail display
- **`+`/`=`**: Decrease refresh rate (faster, min 1s)
- **`-`**: Increase refresh rate (slower, max 30s)

### Adjustable Refresh Rate

- Real-time adjustment from 1-30 seconds
- Displayed in status bar and help
- Saves to user preferences automatically

## Preferences System

### Persistent Settings

User preferences are saved to `vega_state/tty_ui_prefs.json`:

```json
{
    "refresh_interval": 5,
    "show_logs": false,
    "help_mode": false
}
```

### Auto-Save

Preferences are automatically saved when:

- Exiting with `q` key
- Changing refresh rate
- Toggling log display
- Modifying help mode

## Log Integration

### Live Log Tail

- Press `l` to toggle log display
- Shows recent log entries in real-time
- Automatically truncates to fit terminal
- Updates with each refresh cycle

### Log Management

- Searches `vega_logs/` directory for recent entries
- Limits to 50 most recent lines by default
- Handles multiple log files intelligently

## System Requirements

### Dependencies

- `psutil`: System and process monitoring
- `curses`: Terminal UI framework
- Standard Python 3.12+ libraries

### Terminal Compatibility

- Supports various terminal types
- Graceful fallback for incompatible terminals
- Handles small screen sizes gracefully

## Usage Examples

### Basic Status Check

```bash
python vega.py --status
```

### Force KVM Mode

```bash
VEGA_FORCE_KVM=1 python vega.py --status
```

### With Autostart

```bash
python vega.py --autostart --status
```

## Error Handling

### Graceful Fallbacks

- Falls back to plain text if curses fails
- Handles missing dependencies gracefully
- Provides error messages for troubleshooting

### Common Issues

- **"Curses UI failed"**: Terminal incompatibility, falls back to plain status
- **Missing psutil**: Install with `pip install psutil`
- **Permission errors**: Check file system permissions for log directory

## Advanced Features

### Process Detection

Uses sophisticated process matching:

- Command line argument analysis
- Process name pattern matching
- Multi-process component support
- Real-time process state tracking

### Temperature Monitoring

- Auto-detects available temperature sensors
- Displays system temperature when available
- Handles systems without temperature sensors

### Memory Management

- Efficient refresh cycles
- Minimal memory footprint
- Proper cleanup on exit

## Development Integration

### Component Registration

New components are automatically detected when added to the `components` dictionary in `vega.py`.

### Health Check Integration

The TTY UI integrates with the existing health check system for accurate status reporting.

### Extensibility

The curses UI framework supports easy addition of new display panels and metrics.

## Future Enhancements

### Planned Features

- Network status monitoring
- Disk usage display
- Process resource usage details
- Color customization
- Multiple view modes
- Configuration management UI

### Performance Optimizations

- Async refresh cycles
- Selective component monitoring
- Caching for frequently accessed data
- Background process monitoring

## Troubleshooting

### Debug Mode

Enable debug output with environment variables:

```bash
DEBUG=1 VEGA_FORCE_KVM=1 python vega.py --status
```

### Log Analysis

Check `vega_logs/` for detailed operation logs and error messages.

### Terminal Testing

Test terminal compatibility with:

```bash
python -c "import curses; curses.wrapper(lambda s: s.getch())"
```

This comprehensive TTY UI provides enterprise-grade monitoring capabilities specifically designed for rackmount KVM console environments while maintaining compatibility with standard terminal interfaces.
