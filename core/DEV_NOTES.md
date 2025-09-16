# Core Module Development Notes

## Overview

The `core/` module contains the essential infrastructure components for Vega2.0:

- **app.py**: FastAPI application with all endpoints
- **logging_setup.py**: Centralized logging infrastructure
- **config_manager.py**: YAML-based configuration management

## Logging Integration

### VegaLogger Usage

Every module should use the centralized logging system:

```python
from core.logging_setup import VegaLogger

# Get module-specific logger
logger = VegaLogger.get_logger("module_name")

# Log structured data
logger.info("User action", extra={
    "user_id": "user123",
    "action": "login",
    "duration_ms": 150
})
```

### Log Features

- **Rotating Files**: 10MB max size, 5 backup files
- **JSON Format**: Structured logs with timestamps, levels, modules
- **Module Directories**: `/logs/{module}/` for each component
- **Thread Safe**: Concurrent logging from multiple threads
- **Console Output**: Colored console logs for development

### Admin Log Access

- **GET /admin/logs**: List available log modules
- **GET /admin/logs/{module}?lines=N**: Tail N lines from module logs

## Configuration Management

### ConfigManager Usage

```python
from core.config_manager import config_manager

# Get module configuration
config = config_manager.get_config("module_name")

# Update configuration
config_manager.update_config("module_name", new_config)

# Get LLM behavior settings
behavior = config_manager.get_llm_behavior()
```

### Configuration Structure

- **YAML Files**: `config/{module}.yaml` for each component
- **Environment Overrides**: Environment variables take precedence
- **Validation**: Type checking and value validation
- **Backup**: Automatic backups on configuration changes

### Admin Config Access

- **GET /admin/config**: List available config modules
- **GET /admin/config/{module}**: Get module configuration
- **PUT /admin/config/{module}**: Update module configuration
- **GET/PUT /admin/llm/behavior**: Human-readable LLM behavior settings

## FastAPI Application Structure

### Endpoint Categories

1. **Health Checks**: `/healthz`, `/livez`, `/readyz`, `/metrics`
2. **Core Chat**: `/chat` (streaming and non-streaming)
3. **History**: `/history`, `/session/{id}`
4. **Feedback**: `/feedback` for conversation rating
5. **Admin**: `/admin/logs/*`, `/admin/config/*` for system management

### Security Features

- **API Key Authentication**: Required for all data endpoints
- **Multiple Keys**: Support for additional API keys via configuration
- **Request Validation**: Pydantic models for request/response validation
- **Error Handling**: Comprehensive error responses with proper HTTP codes

### Static File Serving

- **Static Directory**: `/static/` mounted for UI files
- **Control Panel**: Advanced web interface at `/static/index.html`

## Development Workflow

### Adding New Modules

1. **Create Logger**: `logger = VegaLogger.get_logger("new_module")`
2. **Create Config**: Add `config/new_module.yaml` with settings
3. **Import Infrastructure**: Use centralized logging and config management
4. **Add Tests**: Create comprehensive test coverage
5. **Document**: Update DEV_NOTES.md with integration details

### Testing Integration

```python
# Test with centralized infrastructure
from core.logging_setup import VegaLogger
from core.config_manager import config_manager

def test_new_feature():
    logger = VegaLogger.get_logger("test_module")
    config = config_manager.get_config("test_module")
    # Test implementation
```

### Configuration Pattern

```yaml
# config/module.yaml
module_settings:
  setting1: value1
  setting2: value2
  
nested_config:
  database:
    url: "sqlite:///module.db"
    timeout: 30
    
behavior_settings:
  enabled: true
  debug_mode: false
```

## Production Considerations

### Logging in Production

- **Log Rotation**: Automatic cleanup prevents disk space issues
- **Structured Logs**: JSON format enables log aggregation systems
- **Performance**: Async logging minimizes application impact
- **Security**: Sensitive data masking available via security.py

### Configuration Management

- **Environment Variables**: Override configs without file changes
- **Validation**: Startup fails fast on invalid configuration
- **Hot Reload**: Runtime configuration updates without restart
- **Backup Strategy**: Configuration changes are automatically backed up

### Monitoring and Observability

- **Health Checks**: Kubernetes-compatible health endpoints
- **Metrics**: Basic application metrics via `/metrics`
- **Admin APIs**: Real-time log viewing and configuration management
- **Error Tracking**: Comprehensive error logging with context

## Future Enhancements

### Planned Improvements

1. **Metrics Collection**: Prometheus-compatible metrics
2. **Distributed Tracing**: OpenTelemetry integration
3. **Config Validation**: JSON Schema validation for configurations
4. **Audit Logging**: Track all configuration changes
5. **Performance Monitoring**: Request/response time tracking

### Scalability Considerations

- **Database Connection Pooling**: Async database connections
- **Caching Layer**: Redis integration for configuration caching
- **Load Balancing**: Multi-instance deployment support
- **Message Queues**: Async task processing capabilities
