# Vega 2.0 Admin API Documentation

## Overview

The Vega 2.0 Admin API provides administrative and advanced functionality for system management, security, backup operations, and specialized features.

**Base URL**: `http://localhost:8000` (development) | `https://api.vega2.example.com` (production)

**Authentication**: All admin endpoints require the `X-API-Key` header with administrative privileges.

## Admin & Configuration Endpoints

### GET /admin/logs

Get system logs overview

**Response**:

```json
{
  "modules": ["chat", "federated", "collaboration", "analytics", "security"],
  "total_logs": 15420,
  "recent_errors": 3,
  "log_retention_days": 30
}
```

### GET /admin/logs/{module}

Get logs for specific module

**Parameters**:

- `module` (path): Module name (chat, federated, collaboration, analytics, security)
- `limit` (query): Number of log entries (default: 100, max: 1000)
- `level` (query): Log level filter (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `since` (query): ISO timestamp to filter logs from

**Example Request**:

```bash
curl -H "X-API-Key: admin_key" \
  "http://localhost:8000/admin/logs/security?limit=50&level=ERROR&since=2025-09-20T00:00:00Z"
```

**Response**:

```json
{
  "module": "security",
  "logs": [
    {
      "timestamp": "2025-09-20T10:30:15Z",
      "level": "ERROR",
      "message": "Failed security scan: timeout after 30s",
      "details": {
        "scanner": "bandit",
        "target": "src/vega/core/",
        "error_code": "TIMEOUT"
      }
    }
  ],
  "total": 1,
  "has_more": false
}
```

### GET /admin/config

Get all configuration modules

**Response**:

```json
{
  "modules": {
    "app": {
      "last_modified": "2025-09-20T08:15:00Z",
      "size": "2.4KB",
      "valid": true
    },
    "security": {
      "last_modified": "2025-09-20T09:00:00Z", 
      "size": "5.1KB",
      "valid": true
    },
    "federated": {
      "last_modified": "2025-09-19T16:30:00Z",
      "size": "3.8KB", 
      "valid": true
    }
  }
}
```

### GET /admin/config/{module}

Get configuration for specific module

**Parameters**:

- `module` (path): Configuration module name

**Response**:

```json
{
  "module": "security",
  "config": {
    "scanner": {
      "enabled_tools": ["bandit", "safety", "semgrep"],
      "severity_threshold": "medium"
    },
    "vulnerability": {
      "auto_fix": false,
      "notify_on_critical": true
    }
  },
  "metadata": {
    "last_modified": "2025-09-20T09:00:00Z",
    "version": "1.2.3",
    "checksum": "sha256:abc123..."
  }
}
```

### PUT /admin/config/{module}

Update configuration for specific module

**Request Body**:

```json
{
  "config": {
    "scanner": {
      "enabled_tools": ["bandit", "safety", "semgrep", "trivy"],
      "severity_threshold": "high"
    }
  }
}
```

**Response**:

```json
{
  "status": "updated",
  "module": "security", 
  "changes": {
    "scanner.enabled_tools": ["bandit", "safety", "semgrep", "trivy"],
    "scanner.severity_threshold": "high"
  },
  "restart_required": false
}
```

## Process Management

### GET /admin/processes/status

Get status of all background processes

**Response**:

```json
{
  "processes": [
    {
      "id": "analytics_collector",
      "name": "Analytics Data Collector",
      "status": "running",
      "pid": 12345,
      "uptime": 3600,
      "memory_mb": 245,
      "cpu_percent": 2.3,
      "last_heartbeat": "2025-09-20T10:29:45Z"
    },
    {
      "id": "federated_coordinator", 
      "name": "Federated Learning Coordinator",
      "status": "stopped",
      "last_exit_code": 0,
      "stopped_at": "2025-09-20T09:15:22Z"
    }
  ],
  "total_processes": 2,
  "running": 1,
  "stopped": 1
}
```

### POST /admin/processes/start

Start background processes

**Request Body**:

```json
{
  "processes": ["analytics_collector", "federated_coordinator"],
  "restart_if_running": false
}
```

**Response**:

```json
{
  "started": ["analytics_collector", "federated_coordinator"],
  "failed": [],
  "already_running": [],
  "details": {
    "analytics_collector": {
      "pid": 12346,
      "started_at": "2025-09-20T10:30:00Z"
    }
  }
}
```

### POST /admin/processes/stop

Stop background processes

**Request Body**:

```json
{
  "processes": ["analytics_collector"],
  "force": false,
  "timeout": 30
}
```

### POST /admin/processes/{process_id}/restart

Restart specific process

**Parameters**:

- `process_id` (path): Process identifier

**Response**:

```json
{
  "process_id": "analytics_collector",
  "status": "restarted", 
  "old_pid": 12345,
  "new_pid": 12347,
  "restart_time": "2025-09-20T10:31:00Z"
}
```

## Security & Cryptography

### POST /admin/ecc/generate-key

Generate new ECC cryptographic key

**Request Body**:

```json
{
  "curve": "P-256",
  "key_type": "signing",
  "description": "Production API signing key",
  "expires_days": 365
}
```

**Response**:

```json
{
  "key_id": "ecc_key_abc123",
  "public_key": "-----BEGIN PUBLIC KEY-----\nMFkwEwYHKoZI...",
  "fingerprint": "SHA256:abc123def456...",
  "created_at": "2025-09-20T10:32:00Z",
  "expires_at": "2026-09-20T10:32:00Z"
}
```

### GET /admin/ecc/keys

List all ECC keys

**Response**:

```json
{
  "keys": [
    {
      "key_id": "ecc_key_abc123",
      "description": "Production API signing key",
      "curve": "P-256", 
      "key_type": "signing",
      "created_at": "2025-09-20T10:32:00Z",
      "expires_at": "2026-09-20T10:32:00Z",
      "status": "active"
    }
  ],
  "total": 1
}
```

### GET /admin/ecc/keys/{key_id}

Get specific ECC key details

**Response**:

```json
{
  "key_id": "ecc_key_abc123",
  "public_key": "-----BEGIN PUBLIC KEY-----\nMFkwEwYHKoZI...",
  "description": "Production API signing key",
  "curve": "P-256",
  "key_type": "signing", 
  "created_at": "2025-09-20T10:32:00Z",
  "expires_at": "2026-09-20T10:32:00Z",
  "usage_count": 1547,
  "last_used": "2025-09-20T10:25:00Z"
}
```

### DELETE /admin/ecc/keys/{key_id}

Delete ECC key

**Response**:

```json
{
  "key_id": "ecc_key_abc123",
  "status": "deleted",
  "deleted_at": "2025-09-20T10:35:00Z"
}
```

### POST /admin/ecc/sign

Sign data with ECC key

**Request Body**:

```json
{
  "key_id": "ecc_key_abc123",
  "data": "Hello, World!",
  "encoding": "utf-8"
}
```

**Response**:

```json
{
  "signature": "MEUCIQDXr7...",
  "algorithm": "ECDSA-SHA256",
  "key_id": "ecc_key_abc123",
  "signed_at": "2025-09-20T10:36:00Z"
}
```

### POST /admin/ecc/verify

Verify ECC signature

**Request Body**:

```json
{
  "key_id": "ecc_key_abc123",
  "data": "Hello, World!",
  "signature": "MEUCIQDXr7...",
  "encoding": "utf-8"
}
```

**Response**:

```json
{
  "valid": true,
  "key_id": "ecc_key_abc123",
  "verified_at": "2025-09-20T10:37:00Z"
}
```

## API Key Management

### POST /admin/security/generate-api-key

Generate new API key

**Request Body**:

```json
{
  "name": "Analytics Dashboard",
  "permissions": ["read:analytics", "read:metrics"],
  "expires_days": 90,
  "rate_limit": {
    "requests_per_hour": 1000
  }
}
```

**Response**:

```json
{
  "api_key": "vega_live_1234567890abcdef", 
  "key_id": "key_abc123",
  "name": "Analytics Dashboard",
  "permissions": ["read:analytics", "read:metrics"],
  "created_at": "2025-09-20T10:38:00Z",
  "expires_at": "2025-12-19T10:38:00Z",
  "rate_limit": {
    "requests_per_hour": 1000
  }
}
```

### GET /admin/security/api-keys

List all API keys

**Response**:

```json
{
  "api_keys": [
    {
      "key_id": "key_abc123",
      "name": "Analytics Dashboard", 
      "permissions": ["read:analytics", "read:metrics"],
      "created_at": "2025-09-20T10:38:00Z",
      "expires_at": "2025-12-19T10:38:00Z",
      "last_used": "2025-09-20T10:20:00Z",
      "usage_count": 2456,
      "status": "active"
    }
  ],
  "total": 1
}
```

## Error Handling & Recovery

### GET /admin/errors/stats

Get error statistics

**Response**:

```json
{
  "time_range": "24h",
  "total_errors": 23,
  "error_types": {
    "authentication": 5,
    "rate_limit": 3,
    "internal": 2,
    "validation": 13
  },
  "top_endpoints": [
    {
      "endpoint": "/chat",
      "errors": 8,
      "error_rate": 0.02
    }
  ],
  "recovery_stats": {
    "auto_recovered": 18,
    "manual_intervention": 2,
    "unresolved": 3
  }
}
```

### GET /admin/recovery/stats

Get recovery system statistics

**Response**:

```json
{
  "circuit_breakers": {
    "llm_service": {
      "state": "closed",
      "failure_count": 0,
      "last_failure": null,
      "success_rate": 0.99
    },
    "database": {
      "state": "closed", 
      "failure_count": 0,
      "success_rate": 1.0
    }
  },
  "auto_recovery": {
    "enabled": true,
    "attempts_24h": 3,
    "success_rate": 0.95
  }
}
```

### POST /admin/recovery/clear-history

Clear recovery history

**Response**:

```json
{
  "status": "cleared",
  "records_removed": 156,
  "cleared_at": "2025-09-20T10:40:00Z"
}
```

## Backup & Recovery Operations

### POST /backup/create

Create system backup

**Request Body**:

```json
{
  "name": "daily_backup_2025_09_20",
  "include": ["database", "configs", "logs", "models"],
  "compression": true,
  "encryption": true
}
```

**Response**:

```json
{
  "backup_id": "backup_abc123",
  "name": "daily_backup_2025_09_20",
  "size_bytes": 1048576000,
  "created_at": "2025-09-20T10:41:00Z",
  "checksum": "sha256:def456...",
  "location": "/backups/daily_backup_2025_09_20.tar.gz.enc"
}
```

### GET /backup/list

List all backups

**Parameters**:

- `limit` (query): Maximum backups to return (default: 20)
- `include_metadata` (query): Include detailed metadata (default: true)

**Response**:

```json
{
  "backups": [
    {
      "backup_id": "backup_abc123", 
      "name": "daily_backup_2025_09_20",
      "size_bytes": 1048576000,
      "created_at": "2025-09-20T10:41:00Z",
      "type": "full",
      "status": "completed"
    }
  ],
  "total": 1,
  "total_size_bytes": 1048576000
}
```

### POST /backup/restore

Restore from backup

**Request Body**:

```json
{
  "backup_id": "backup_abc123",
  "components": ["database", "configs"],
  "confirm": true
}
```

**Response**:

```json
{
  "restore_id": "restore_def456",
  "backup_id": "backup_abc123", 
  "status": "in_progress",
  "started_at": "2025-09-20T10:42:00Z",
  "estimated_completion": "2025-09-20T10:45:00Z"
}
```

## Voice Processing

### POST /voice/samples

Submit voice samples for profile creation

**Request Body** (multipart/form-data):

```
audio_file: <binary audio data>
duration: 30
quality: "high"
speaker_id: "user_123"
```

**Response**:

```json
{
  "sample_id": "voice_sample_abc123",
  "duration": 30.5,
  "quality_score": 0.92,
  "processed_at": "2025-09-20T10:43:00Z",
  "status": "processed"
}
```

### POST /voice/profile/update

Update voice profile

**Request Body**:

```json
{
  "speaker_id": "user_123",
  "samples": ["voice_sample_abc123", "voice_sample_def456"],
  "preferences": {
    "voice_speed": 1.0,
    "pitch_adjustment": 0.0
  }
}
```

### GET /voice/profile

Get voice profile information

**Parameters**:

- `speaker_id` (query): Speaker identifier

**Response**:

```json
{
  "speaker_id": "user_123",
  "profile_created": "2025-09-20T08:00:00Z",
  "samples_count": 5,
  "quality_score": 0.94,
  "preferences": {
    "voice_speed": 1.0,
    "pitch_adjustment": 0.0
  },
  "last_updated": "2025-09-20T10:43:00Z"
}
```

## Knowledge Base Management

### POST /kb/sites

Add website to knowledge base

**Request Body**:

```json
{
  "url": "https://example.com/docs",
  "depth": 3,
  "include_patterns": ["*/docs/*", "*/api/*"],
  "exclude_patterns": ["*/admin/*"],
  "schedule": "weekly"
}
```

**Response**:

```json
{
  "site_id": "site_abc123",
  "url": "https://example.com/docs", 
  "status": "indexing",
  "pages_discovered": 0,
  "pages_indexed": 0,
  "started_at": "2025-09-20T10:44:00Z"
}
```

### GET /kb/sites

List knowledge base sites

**Response**:

```json
{
  "sites": [
    {
      "site_id": "site_abc123",
      "url": "https://example.com/docs",
      "status": "active",
      "pages_indexed": 156,
      "last_crawl": "2025-09-20T02:00:00Z",
      "next_crawl": "2025-09-27T02:00:00Z"
    }
  ],
  "total": 1
}
```

## Finance & Investment

### POST /finance/invest

Record investment transaction

**Request Body**:

```json
{
  "symbol": "AAPL",
  "shares": 10,
  "price": 150.25,
  "transaction_type": "buy",
  "date": "2025-09-20"
}
```

**Response**:

```json
{
  "transaction_id": "txn_abc123",
  "symbol": "AAPL",
  "shares": 10,
  "price": 150.25,
  "total_cost": 1502.50,
  "transaction_type": "buy", 
  "recorded_at": "2025-09-20T10:45:00Z"
}
```

### GET /finance/portfolio

Get investment portfolio

**Response**:

```json
{
  "portfolio": [
    {
      "symbol": "AAPL",
      "shares": 50,
      "average_price": 148.75,
      "current_price": 152.30,
      "total_value": 7615.00,
      "unrealized_gain": 177.50,
      "percentage_gain": 2.38
    }
  ],
  "total_value": 7615.00,
  "total_cost": 7437.50,
  "total_gain": 177.50,
  "percentage_gain": 2.38
}
```

### GET /finance/price/{symbol}

Get current stock price

**Parameters**:

- `symbol` (path): Stock symbol (e.g., AAPL, GOOGL)

**Response**:

```json
{
  "symbol": "AAPL",
  "price": 152.30,
  "change": 1.75,
  "change_percent": 1.16,
  "volume": 45678900,
  "market_cap": 2456789000000,
  "updated_at": "2025-09-20T10:45:30Z"
}
```

## Error Responses

All endpoints may return these common error responses:

### 401 Unauthorized

```json
{
  "error": "Authentication required",
  "code": 401,
  "details": "Missing or invalid X-API-Key header"
}
```

### 403 Forbidden

```json
{
  "error": "Insufficient permissions", 
  "code": 403,
  "details": "Admin access required for this endpoint"
}
```

### 429 Rate Limited

```json
{
  "error": "Rate limit exceeded",
  "code": 429,
  "details": "Maximum 1000 requests per hour exceeded",
  "retry_after": 3600
}
```

### 500 Internal Server Error

```json
{
  "error": "Internal server error",
  "code": 500,
  "details": "An unexpected error occurred",
  "request_id": "req_abc123"
}
```
