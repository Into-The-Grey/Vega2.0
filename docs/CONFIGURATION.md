# Vega2.0 Configuration Guide

## Overview

Vega2.0 uses environment variables for configuration, loaded from a `.env` file. This guide covers all configuration options and how to set up your environment for optimal performance.

## Quick Start Configuration

1. Copy the example configuration:

   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file with your settings:

   ```bash
   nano .env
   ```

3. Verify configuration:

   ```bash
   python -m cli config verify
   ```

## Core Configuration

### Required Settings

```env
# API Configuration
API_KEY=your-secure-api-key-here
HOST=127.0.0.1
PORT=8000

# LLM Configuration
MODEL_NAME=llama3.2:latest
LLM_BACKEND=ollama
OLLAMA_URL=http://127.0.0.1:11434

# Database Configuration
DATABASE_URL=sqlite:///./vega.db
```

### Security Settings

```env
# Additional API Keys (comma-separated)
API_KEYS_EXTRA=key1,key2,key3

# ECC Configuration
ECC_DEFAULT_CURVE=secp256r1
ECC_KEY_EXPIRY_DAYS=365

# Security Features
ENABLE_RATE_LIMITING=true
RATE_LIMIT_REQUESTS=60
RATE_LIMIT_WINDOW=60
```

### Background Process Configuration

```env
# Process Management
ENABLE_BACKGROUND_PROCESSES=true
PROCESS_RESTART_DELAY=5
PROCESS_MAX_RESTARTS=3
SYSTEM_MONITOR_INTERVAL=30

# Health Check Configuration
HEALTH_CHECK_INTERVAL=10
HEALTH_CHECK_TIMEOUT=30
```

### Error Handling Configuration

```env
# Error Handling
ENABLE_STRUCTURED_LOGGING=true
LOG_LEVEL=INFO
ERROR_RECOVERY_ENABLED=true
MAX_RECOVERY_ATTEMPTS=3

# Logging Configuration
LOG_FORMAT=json
LOG_FILE=vega.log
LOG_ROTATION_SIZE=10MB
LOG_RETENTION_DAYS=30
```

### Integration Configuration

```env
# Web Search Integration
SEARCH_PROVIDER=duckduckgo
SEARCH_MAX_RESULTS=10
SEARCH_TIMEOUT=30

# Slack Integration
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_SIGNING_SECRET=your-signing-secret
SLACK_APP_TOKEN=xapp-your-app-token

# OSINT Integration
OSINT_ENABLED=false
OSINT_SOURCES=shodan,virustotal
```

### Performance Settings

```env
# Cache Configuration
CACHE_TTL=300
CACHE_MAX_SIZE=1000
ENABLE_CIRCUIT_BREAKER=true

# Connection Settings
HTTP_TIMEOUT=30
HTTP_MAX_CONNECTIONS=100
HTTP_MAX_KEEPALIVE_CONNECTIONS=20

# LLM Settings
LLM_TIMEOUT=120
LLM_MAX_RETRIES=3
LLM_RETRY_DELAY=1
```

## Environment-Specific Configurations

### Development Environment

```env
# Development Settings
DEBUG=true
RELOAD=true
LOG_LEVEL=DEBUG
ENABLE_CORS=true

# Development URLs
HOST=127.0.0.1
PORT=8000
OLLAMA_URL=http://127.0.0.1:11434

# Development Database
DATABASE_URL=sqlite:///./vega_dev.db
```

### Production Environment

```env
# Production Settings
DEBUG=false
RELOAD=false
LOG_LEVEL=INFO
ENABLE_CORS=false

# Production Security
API_KEY=your-strong-production-api-key
ECC_DEFAULT_CURVE=secp384r1
ENABLE_RATE_LIMITING=true

# Production Database
DATABASE_URL=sqlite:///./vega_prod.db

# Performance Optimizations
CACHE_TTL=600
HTTP_MAX_CONNECTIONS=200
PROCESS_MAX_RESTARTS=5
```

### Testing Environment

```env
# Testing Settings
DEBUG=true
LOG_LEVEL=WARNING
ENABLE_BACKGROUND_PROCESSES=false

# Test Database
DATABASE_URL=sqlite:///./vega_test.db

# Test LLM Settings
MODEL_NAME=test-model
LLM_BACKEND=mock
LLM_TIMEOUT=5
```

## Advanced Configuration

### Multiple LLM Backends

```env
# Primary Backend
LLM_BACKEND=ollama
MODEL_NAME=llama3.2:latest
OLLAMA_URL=http://127.0.0.1:11434

# Fallback Backends
FALLBACK_BACKENDS=openai,anthropic
OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=gpt-4
ANTHROPIC_API_KEY=your-anthropic-key
ANTHROPIC_MODEL=claude-3-sonnet
```

### Custom Process Configuration

```env
# Custom Background Processes
CUSTOM_PROCESSES=data_processor,file_watcher
DATA_PROCESSOR_INTERVAL=60
FILE_WATCHER_PATH=/home/user/watch
```

### Distributed Configuration

```env
# Distributed Setup
ENABLE_CLUSTERING=true
CLUSTER_NODES=node1:8000,node2:8000,node3:8000
CLUSTER_TOKEN=your-cluster-token
LOAD_BALANCER_URL=http://lb.example.com
```

## Configuration Validation

### Built-in Validation

Vega2.0 includes configuration validation:

```python
from config import get_config, validate_config

# Validate current configuration
config = get_config()
is_valid, errors = validate_config(config)

if not is_valid:
    for error in errors:
        print(f"Configuration error: {error}")
```

### Environment Validation Script

```bash
#!/bin/bash
# validate_env.sh

echo "Validating Vega2.0 configuration..."

# Check required variables
required_vars=("API_KEY" "HOST" "PORT" "MODEL_NAME" "LLM_BACKEND")

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "ERROR: $var is not set"
        exit 1
    fi
done

# Check database connectivity
python -c "
from db import create_tables, get_db_session
try:
    create_tables()
    with get_db_session() as session:
        pass
    print('✓ Database connection successful')
except Exception as e:
    print(f'✗ Database error: {e}')
    exit(1)
"

# Check LLM connectivity
python -c "
from llm import LLMManager
try:
    llm = LLMManager()
    # Test with a simple prompt
    result = llm.generate('Hello')
    print('✓ LLM connection successful')
except Exception as e:
    print(f'✗ LLM error: {e}')
    exit(1)
"

echo "✓ Configuration validation complete"
```

## Security Configuration

### ECC Key Management

```env
# ECC Configuration
ECC_DEFAULT_CURVE=secp256r1
ECC_KEY_EXPIRY_DAYS=365
ECC_AUTO_ROTATE=true
ECC_BACKUP_ENABLED=true
ECC_BACKUP_PATH=/secure/backup/path

# Certificate Configuration
CERT_COUNTRY=US
CERT_STATE=California
CERT_CITY=San Francisco
CERT_ORG=Vega Systems
CERT_OU=Development
```

### API Security Configuration

```env
# API Security
SECURE_API_KEYS=true
API_KEY_ROTATION_DAYS=90
API_KEY_MIN_LENGTH=32
ENFORCE_HTTPS=false

# Request Security
MAX_REQUEST_SIZE=10MB
REQUEST_TIMEOUT=30
ENABLE_REQUEST_SIGNING=true
```

## Monitoring Configuration

### Metrics Collection

```env
# Metrics Configuration
ENABLE_METRICS=true
METRICS_PORT=9090
METRICS_PATH=/metrics
PROMETHEUS_ENABLED=false

# Health Check Configuration
HEALTH_CHECK_ENABLED=true
HEALTH_CHECK_INTERVAL=30
HEALTH_CHECK_TIMEOUT=10
```

### Alerting Configuration

```env
# Alert Configuration
ENABLE_ALERTS=true
ALERT_WEBHOOK_URL=https://alerts.example.com/webhook
ALERT_EMAIL=admin@example.com
ALERT_THRESHOLDS_CPU=80
ALERT_THRESHOLDS_MEMORY=85
ALERT_THRESHOLDS_DISK=90
```

## Backup and Recovery Configuration

### Database Backup

```env
# Backup Configuration
ENABLE_AUTO_BACKUP=true
BACKUP_INTERVAL_HOURS=6
BACKUP_RETENTION_DAYS=30
BACKUP_PATH=/backup/vega

# Recovery Configuration
ENABLE_AUTO_RECOVERY=true
RECOVERY_CHECK_INTERVAL=60
```

### Configuration Backup

```env
# Configuration Backup
CONFIG_BACKUP_ENABLED=true
CONFIG_BACKUP_PATH=/backup/config
CONFIG_VERSION_CONTROL=true
```

## Troubleshooting Configuration

### Debug Settings

```env
# Debug Configuration
DEBUG_MODE=true
VERBOSE_LOGGING=true
TRACE_REQUESTS=true
PROFILE_PERFORMANCE=true

# Development Tools
ENABLE_SWAGGER=true
ENABLE_REDOC=true
ENABLE_TEST_ENDPOINTS=true
```

### Performance Profiling

```env
# Profiling Configuration
ENABLE_PROFILING=true
PROFILE_OUTPUT_DIR=/tmp/profiling
PROFILE_MEMORY=true
PROFILE_CPU=true
```

## Configuration Management Best Practices

### Security Best Practices

1. **Never commit `.env` files to version control**
2. **Use strong, unique API keys**
3. **Rotate keys regularly**
4. **Use environment-specific configurations**
5. **Validate all configuration values**

### Performance Best Practices

1. **Tune cache settings for your workload**
2. **Adjust timeout values based on network conditions**
3. **Configure appropriate connection limits**
4. **Monitor and adjust based on metrics**

### Operational Best Practices

1. **Use configuration management tools**
2. **Implement configuration validation**
3. **Document all custom settings**
4. **Test configuration changes in staging**
5. **Monitor configuration drift**

## Configuration Templates

### Docker Configuration

```yaml
# docker-compose.yml
version: '3.8'
services:
  vega:
    build: .
    environment:
      - API_KEY=${API_KEY}
      - HOST=0.0.0.0
      - PORT=8000
      - DATABASE_URL=sqlite:///./data/vega.db
    volumes:
      - ./data:/app/data
    ports:
      - "8000:8000"
```

### Kubernetes Configuration

```yaml
# k8s-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: vega-config
data:
  HOST: "0.0.0.0"
  PORT: "8000"
  LOG_LEVEL: "INFO"
  ENABLE_METRICS: "true"
---
apiVersion: v1
kind: Secret
metadata:
  name: vega-secrets
type: Opaque
stringData:
  API_KEY: "your-secure-api-key"
  OLLAMA_URL: "http://ollama-service:11434"
```

## Configuration Migration

### Version Migration

When upgrading Vega2.0, some configuration options may change:

```bash
# Migration script
python -m cli config migrate --from-version 1.0 --to-version 2.0
```

### Legacy Configuration Support

Vega2.0 maintains backward compatibility with legacy configuration formats:

```env
# Legacy format (still supported)
VEGA_API_KEY=legacy-key
VEGA_MODEL=legacy-model

# New format (preferred)
API_KEY=new-key
MODEL_NAME=new-model
```

This configuration guide should help you set up Vega2.0 for any environment and use case. For additional support, refer to the troubleshooting documentation or open an issue on the project repository.
