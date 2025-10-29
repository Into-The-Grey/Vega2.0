# Vega2.0 API Documentation

## Overview

Vega2.0 provides a comprehensive REST API built with FastAPI, featuring OpenAPI 3.0 compliance, automatic documentation, and comprehensive security. The API supports both the legacy endpoints and new OpenAPI-compliant endpoints.

## Base URLs

- **Legacy API**: `http://localhost:8000`
- **OpenAPI API**: `http://localhost:8001` (when running OpenAPI server)
- **Test Suite**: `http://localhost:8002` (when running test suite)

## Authentication

### API Key Authentication

All protected endpoints require an API key provided in the `X-API-Key` header:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/admin/status
```

### ECC-Backed Secure API Keys

For enhanced security, use ECC-backed API keys generated through the security endpoints:

```bash
# Generate secure API key
curl -X POST -H "X-API-Key: admin-key" \
  -H "Content-Type: application/json" \
  -d '{"permissions": ["read", "write"], "expires_in_days": 30}' \
  http://localhost:8000/admin/security/generate-api-key
```

## Core Endpoints

### Chat API

#### POST /chat

Chat with the AI system.

**Request:**

```json
{
  "prompt": "Hello, how are you?",
  "stream": false,
  "session_id": "optional-session-id"
}
```

**Response:**

```json
{
  "response": "Hello! I'm doing well, thank you for asking.",
  "session_id": "auto-generated-or-provided-session-id"
}
```

**cURL Example:**

```bash
curl -X POST -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"prompt": "What is the weather like?", "stream": false}' \
  http://localhost:8000/chat
```

### History API

#### GET /history

Get conversation history.

**Parameters:**

- `limit` (optional): Number of conversations to return (default: 50)

**Response:**

```json
{
  "conversations": [
    {
      "id": 1,
      "prompt": "Hello",
      "response": "Hi there!",
      "timestamp": "2024-01-15T10:30:00Z",
      "session_id": "session-123"
    }
  ]
}
```

**cURL Example:**

```bash
curl -H "X-API-Key: your-api-key" \
  "http://localhost:8000/history?limit=10"
```

#### GET /session/{session_id}

Get conversation history for specific session.

**Response:**

```json
{
  "session_id": "session-123",
  "conversations": [...]
}
```

### Feedback API

#### POST /feedback

Submit feedback for a conversation.

**Request:**

```json
{
  "conversation_id": 1,
  "rating": 5,
  "comment": "Great response!"
}
```

**Response:**

```json
{
  "status": "success",
  "message": "Feedback recorded"
}
```

## Health Check Endpoints

### GET /healthz

Basic health check.

**Response:**

```json
{
  "ok": true,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### GET /livez

Liveness check (checks if service is running).

**Response:**

```json
{
  "status": "healthy",
  "checks": {
    "api": "ok",
    "database": "ok"
  }
}
```

### GET /readyz

Readiness check (checks if service is ready to serve requests).

**Response:**

```json
{
  "status": "ready",
  "checks": {
    "api": "ok",
    "database": "ok",
    "llm": "ok"
  }
}
```

### GET /metrics

Get service metrics.

**Response:**

```json
{
  "requests_total": 1250,
  "responses_total": 1248,
  "errors_total": 2,
  "degraded": false
}
```

## Administration Endpoints

All admin endpoints require API key authentication.

### Background Process Management

#### GET /admin/processes/status

Get status of all background processes.

**Response:**

```json
{
  "status": "running",
  "processes": [
    {
      "id": "process-id",
      "name": "system_monitor",
      "type": "system",
      "state": "running",
      "pid": 12345,
      "cpu_usage": 2.5,
      "memory_usage": 15.3,
      "restart_count": 0,
      "start_time": "2024-01-15T10:00:00Z",
      "uptime": 1800
    }
  ],
  "metrics": {
    "total_processes": 3,
    "running_processes": 3,
    "failed_processes": 0,
    "health_percentage": 100.0
  }
}
```

#### POST /admin/processes/start

Start all background processes.

**Response:**

```json
{
  "status": "success",
  "message": "Background processes started"
}
```

#### POST /admin/processes/stop

Stop all background processes.

**Response:**

```json
{
  "status": "success",
  "message": "Background processes stopped"
}
```

#### POST /admin/processes/{process_id}/restart

Restart a specific process.

**Response:**

```json
{
  "status": "success",
  "message": "Process restarted"
}
```

### Error Monitoring

#### GET /admin/errors/stats

Get error statistics.

**Response:**

```json
{
  "status": "success",
  "stats": {
    "total_errors": 25,
    "error_breakdown": {
      "VEGA-1201:external_service": 15,
      "VEGA-1101:validation": 10
    },
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### GET /admin/recovery/stats

Get recovery statistics.

**Response:**

```json
{
  "status": "success",
  "stats": {
    "total_recovery_attempts": 10,
    "successful_recoveries": 8,
    "success_rate": 0.8,
    "active_error_types": 2,
    "available_strategies": 8
  }
}
```

#### POST /admin/recovery/clear-history

Clear recovery history.

**Response:**

```json
{
  "status": "success",
  "message": "Recovery history cleared"
}
```

## ECC Cryptography Endpoints

### Key Management

#### POST /admin/ecc/generate-key

Generate new ECC key pair.

**Request:**

```json
{
  "curve": "secp256r1",
  "expires_in_days": 365
}
```

**Response:**

```json
{
  "status": "success",
  "key_id": "vega-ecc-abc123def456",
  "curve": "secp256r1",
  "public_key": "-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----",
  "created_at": "2024-01-15T10:30:00Z",
  "expires_at": "2025-01-15T10:30:00Z"
}
```

#### GET /admin/ecc/keys

List all ECC keys.

**Response:**

```json
{
  "status": "success",
  "keys": [
    {
      "key_id": "vega-ecc-abc123def456",
      "curve": "secp256r1",
      "created_at": "2024-01-15T10:30:00Z",
      "expires_at": "2025-01-15T10:30:00Z",
      "has_private_key": true,
      "has_certificate": false,
      "expired": false
    }
  ]
}
```

#### GET /admin/ecc/keys/{key_id}

Get specific ECC key information.

**Response:**

```json
{
  "status": "success",
  "key": {
    "key_id": "vega-ecc-abc123def456",
    "curve": "secp256r1",
    "public_key": "-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----",
    "created_at": "2024-01-15T10:30:00Z",
    "expires_at": "2025-01-15T10:30:00Z",
    "has_private_key": true,
    "has_certificate": false,
    "expired": false
  }
}
```

#### DELETE /admin/ecc/keys/{key_id}

Delete ECC key.

**Response:**

```json
{
  "status": "success",
  "message": "Key deleted"
}
```

### Digital Signatures

#### POST /admin/ecc/sign

Sign data with ECC key.

**Request:**

```json
{
  "data": "Hello, World!",
  "key_id": "vega-ecc-abc123def456"
}
```

**Response:**

```json
{
  "status": "success",
  "signature": {
    "signature": "base64-encoded-signature",
    "algorithm": "ECDSA-SHA256",
    "key_id": "vega-ecc-abc123def456",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### POST /admin/ecc/verify

Verify ECC signature.

**Request:**

```json
{
  "data": "Hello, World!",
  "signature": {
    "signature": "base64-encoded-signature",
    "algorithm": "ECDSA-SHA256",
    "key_id": "vega-ecc-abc123def456",
    "timestamp": "2024-01-15T10:30:00Z"
  },
  "public_key_pem": "-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----"
}
```

**Response:**

```json
{
  "status": "success",
  "valid": true
}
```

## Security Management Endpoints

### API Key Management

#### POST /admin/security/generate-api-key

Generate secure ECC-backed API key.

**Request:**

```json
{
  "permissions": ["read", "write"],
  "expires_in_days": 30,
  "rate_limit": 100
}
```

**Response:**

```json
{
  "status": "success",
  "api_key": "vega_ecc_...",
  "key_id": "api_12345678",
  "ecc_key_id": "vega-ecc-def789abc012",
  "permissions": ["read", "write"],
  "expires_at": "2024-02-15T10:30:00Z"
}
```

#### GET /admin/security/api-keys

List all secure API keys.

**Response:**

```json
{
  "status": "success",
  "api_keys": [
    {
      "key_id": "api_12345678",
      "ecc_key_id": "vega-ecc-def789abc012",
      "permissions": ["read", "write"],
      "expires_at": "2024-02-15T10:30:00Z",
      "rate_limit": 100,
      "created_at": "2024-01-15T10:30:00Z",
      "last_used": "2024-01-15T11:00:00Z"
    }
  ]
}
```

## Error Responses

All API endpoints return structured error responses:

### Standard Error Format

```json
{
  "error_id": "uuid-error-id",
  "code": "VEGA-1001",
  "message": "User-friendly error message",
  "recoverable": true,
  "retry_after": 30,
  "request_id": "request-uuid"
}
```

### HTTP Status Codes

- **200**: Success
- **400**: Bad Request (validation errors)
- **401**: Unauthorized (missing/invalid API key)
- **403**: Forbidden (insufficient permissions)
- **429**: Too Many Requests (rate limited)
- **500**: Internal Server Error
- **502**: Bad Gateway (external service error)
- **503**: Service Unavailable (system overloaded)

### Common Error Codes

| Code | Category | Description |
|------|----------|-------------|
| VEGA-1001 | Authentication | Invalid API key |
| VEGA-1002 | Authentication | Missing API key |
| VEGA-1101 | Validation | Invalid input |
| VEGA-1102 | Validation | Missing parameter |
| VEGA-1201 | External Service | LLM provider error |
| VEGA-1202 | External Service | Rate limit exceeded |
| VEGA-1301 | Database | Connection failed |
| VEGA-1401 | Process | Process start failed |
| VEGA-1501 | Configuration | Invalid config |
| VEGA-1601 | Network | Connection timeout |
| VEGA-1801 | Internal | Unexpected error |

## Rate Limiting

API endpoints are subject to rate limiting based on API key configuration:

- **Default Limit**: 60 requests per minute
- **Burst Limit**: 10 requests per second
- **Rate Limit Headers**: Included in responses

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1642246800
```

## WebSocket Support

### Chat Streaming

For real-time chat responses, use the streaming endpoint:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/chat');
ws.send(JSON.stringify({
  "prompt": "Tell me a story",
  "session_id": "session-123"
}));
```

## SDK Examples

### Python SDK

```python
import requests

class VegaClient:
    def __init__(self, base_url="http://localhost:8000", api_key=None):
        self.base_url = base_url
        self.headers = {"X-API-Key": api_key} if api_key else {}
    
    def chat(self, prompt, session_id=None):
        response = requests.post(
            f"{self.base_url}/chat",
            json={"prompt": prompt, "session_id": session_id},
            headers=self.headers
        )
        return response.json()
    
    def get_history(self, limit=50):
        response = requests.get(
            f"{self.base_url}/history",
            params={"limit": limit},
            headers=self.headers
        )
        return response.json()

# Usage
client = VegaClient(api_key="your-api-key")
result = client.chat("Hello, AI!")
```

### JavaScript SDK

```javascript
class VegaClient {
    constructor(baseUrl = 'http://localhost:8000', apiKey = null) {
        this.baseUrl = baseUrl;
        this.headers = apiKey ? { 'X-API-Key': apiKey } : {};
    }
    
    async chat(prompt, sessionId = null) {
        const response = await fetch(`${this.baseUrl}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...this.headers
            },
            body: JSON.stringify({ prompt, session_id: sessionId })
        });
        return response.json();
    }
    
    async getHistory(limit = 50) {
        const response = await fetch(
            `${this.baseUrl}/history?limit=${limit}`,
            { headers: this.headers }
        );
        return response.json();
    }
}

// Usage
const client = new VegaClient('http://localhost:8000', 'your-api-key');
const result = await client.chat('Hello, AI!');
```

## Interactive Documentation

When the server is running, interactive API documentation is available at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

These provide interactive interfaces for exploring and testing the API endpoints.

## Postman Collection

A Postman collection is available in the `docs/` directory with pre-configured requests for all endpoints.

## API Versioning

The API supports versioning through URL prefixes:

- **v1**: `http://localhost:8000/v1/` (current)
- **v2**: `http://localhost:8000/v2/` (future)

Version headers are also supported:

```bash
curl -H "Accept: application/vnd.vega.v1+json" http://localhost:8000/chat
```
