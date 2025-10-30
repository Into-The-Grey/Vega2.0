# Vega 2.0 API Reference

## Overview

The Vega 2.0 platform provides a comprehensive suite of APIs for advanced AI capabilities, federated learning, multi-modal processing, real-time collaboration, and intelligent analytics. This reference documentation covers all available endpoints, authentication methods, request/response formats, and integration patterns.

**Base URL**: `http://localhost:8000` (development) | `https://api.vega2.example.com` (production)

**Current API Version**: v1

## Quick Start

### Authentication

All API endpoints require authentication via the `X-API-Key` header:

```bash
curl -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     https://api.vega2.example.com/healthz
```

### Health Check

Verify API availability:

```bash
GET /healthz
```

Response:

```json
{
  "status": "healthy",
  "version": "2.0.0",
  "timestamp": "2025-09-20T10:00:00Z"
}
```

## API Documentation Structure

### Core APIs

#### [OpenAPI Specification](./openapi.yaml)

- **Purpose**: Machine-readable API specification (OpenAPI 3.0.3)
- **Contents**: Complete endpoint definitions, schemas, authentication
- **Use Cases**: API client generation, automated testing, documentation tools

#### [Admin API](./admin-api.md)

- **Purpose**: Administrative and system management endpoints
- **Key Features**: Configuration management, process control, security features
- **Access Level**: Admin/elevated privileges required

### Specialized APIs

#### [Collaboration API](./collaboration-api.md)

- **Purpose**: Real-time collaboration and workspace management
- **Key Features**: WebSocket connections, document editing, voice/video sessions
- **Technologies**: WebRTC, operational transformation, real-time messaging

#### [Analytics & Monitoring API](./analytics-api.md)

- **Purpose**: System monitoring, usage analytics, business intelligence
- **Key Features**: Performance metrics, user behavior analysis, custom reports
- **Data Types**: System health, AI performance, federated learning metrics

#### [Multi-modal Processing API](./multimodal-api.md)

- **Purpose**: Advanced AI processing across multiple media types
- **Key Features**: Audio/video analysis, image processing, document extraction
- **Capabilities**: Cross-modal search, content generation, batch processing

## API Categories

### Health & System Monitoring

| Endpoint | Description | Documentation |
|----------|-------------|---------------|
| `GET /healthz` | Basic health check | [Admin API](./admin-api.md#health-monitoring) |
| `GET /livez` | Liveness probe | [Admin API](./admin-api.md#health-monitoring) |
| `GET /readyz` | Readiness probe | [Admin API](./admin-api.md#health-monitoring) |
| `GET /analytics/system/health` | Comprehensive system metrics | [Analytics API](./analytics-api.md#system-metrics) |

### AI & Chat

| Endpoint | Description | Documentation |
|----------|-------------|---------------|
| `POST /chat` | AI chat completion | [OpenAPI Spec](./openapi.yaml#/paths/~1chat) |
| `POST /chat/stream` | Streaming chat response | [OpenAPI Spec](./openapi.yaml#/paths/~1chat~1stream) |
| `GET /chat/history` | Conversation history | [OpenAPI Spec](./openapi.yaml#/paths/~1chat~1history) |
| `POST /proactive` | Proactive AI suggestions | [OpenAPI Spec](./openapi.yaml#/paths/~1proactive) |

### Session Management

| Endpoint | Description | Documentation |
|----------|-------------|---------------|
| `POST /sessions` | Create new session | [OpenAPI Spec](./openapi.yaml#/paths/~1sessions) |
| `GET /sessions/{id}` | Get session details | [OpenAPI Spec](./openapi.yaml#/paths/~1sessions~1{id}) |
| `DELETE /sessions/{id}` | Delete session | [OpenAPI Spec](./openapi.yaml#/paths/~1sessions~1{id}) |

### Collaboration

| Endpoint | Description | Documentation |
|----------|-------------|---------------|
| `POST /collaboration/workspaces` | Create workspace | [Collaboration API](./collaboration-api.md#workspace-management) |
| `GET /collaboration/workspaces` | List workspaces | [Collaboration API](./collaboration-api.md#workspace-management) |
| `POST /collaboration/documents` | Create document | [Collaboration API](./collaboration-api.md#document-management) |
| `WebSocket /ws/collaboration/{id}` | Real-time collaboration | [Collaboration API](./collaboration-api.md#websocket-connections) |

### Multi-modal Processing

| Endpoint | Description | Documentation |
|----------|-------------|---------------|
| `POST /multimodal/upload` | Upload media files | [Multi-modal API](./multimodal-api.md#file-upload--management) |
| `POST /multimodal/audio/transcribe` | Audio transcription | [Multi-modal API](./multimodal-api.md#audio-processing) |
| `POST /multimodal/video/analyze` | Video analysis | [Multi-modal API](./multimodal-api.md#video-processing) |
| `POST /multimodal/image/analyze` | Image analysis | [Multi-modal API](./multimodal-api.md#image-processing) |

### Analytics & Monitoring

| Endpoint | Description | Documentation |
|----------|-------------|---------------|
| `GET /analytics/usage/overview` | Usage statistics | [Analytics API](./analytics-api.md#usage-analytics) |
| `GET /analytics/ai/performance` | AI model metrics | [Analytics API](./analytics-api.md#ai-model-analytics) |
| `GET /analytics/federated/training` | Federated learning metrics | [Analytics API](./analytics-api.md#federated-learning-analytics) |
| `POST /analytics/reports/custom` | Custom reports | [Analytics API](./analytics-api.md#custom-reports) |

### Administration

| Endpoint | Description | Documentation |
|----------|-------------|---------------|
| `GET /admin/config` | System configuration | [Admin API](./admin-api.md#configuration-management) |
| `POST /admin/processes/start` | Start background processes | [Admin API](./admin-api.md#process-management) |
| `GET /admin/security/scan` | Security scan results | [Admin API](./admin-api.md#security-management) |
| `POST /admin/backup/create` | Create system backup | [Admin API](./admin-api.md#backup-operations) |

## Authentication & Security

### API Key Authentication

All endpoints require a valid API key in the `X-API-Key` header:

```javascript
const response = await fetch('https://api.vega2.example.com/chat', {
  method: 'POST',
  headers: {
    'X-API-Key': 'your-api-key-here',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    prompt: 'Hello, world!'
  })
});
```

### Rate Limiting

API requests are subject to rate limiting:

- **Standard endpoints**: 1000 requests/hour
- **AI endpoints**: 100 requests/hour  
- **Analytics endpoints**: 500 requests/hour
- **Admin endpoints**: 50 requests/hour

Rate limit headers are included in responses:

```text
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1632150000
```

### Security Features

- **HTTPS**: All production endpoints use TLS 1.3
- **Input validation**: Comprehensive request validation
- **Circuit breakers**: Automatic fault tolerance
- **Audit logging**: Complete request/response logging
- **ECC cryptography**: Advanced encryption support

## Error Handling

### Standard HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Missing or invalid API key |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 422 | Unprocessable Entity | Validation failed |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |

### Error Response Format

```json
{
  "error": "Validation failed",
  "code": 422,
  "details": {
    "field": "prompt",
    "message": "Prompt cannot be empty",
    "provided": ""
  },
  "request_id": "req_abc123456",
  "timestamp": "2025-09-20T10:00:00Z"
}
```

## SDKs & Client Libraries

### Official SDKs

- **Python**: `pip install vega2-sdk`
- **JavaScript/Node.js**: `npm install vega2-sdk`
- **Java**: Maven/Gradle dependency available
- **Go**: `go get github.com/vega2/go-sdk`

### Community SDKs

- **PHP**: Available on Packagist
- **Ruby**: Available as gem
- **C#/.NET**: NuGet package
- **Rust**: Crates.io package

### Example Usage (Python)

```python
from vega2_sdk import VegaClient

client = VegaClient(
    api_key='your-api-key',
    base_url='https://api.vega2.example.com'
)

# Chat completion
response = client.chat.complete(
    prompt="Explain quantum computing",
    stream=False
)

# Upload and analyze video
file_id = client.multimodal.upload('video.mp4')
analysis = client.multimodal.video.analyze(
    file_id=file_id,
    analysis_types=['object_detection', 'transcript']
)

# Create collaborative workspace
workspace = client.collaboration.create_workspace(
    name="Project Alpha",
    members=["user1", "user2"]
)
```

## WebSocket Connections

### Real-time Endpoints

| WebSocket | Purpose | Documentation |
|-----------|---------|---------------|
| `/ws/collaboration/{workspace_id}` | Workspace collaboration | [Collaboration API](./collaboration-api.md#websocket-connections) |
| `/ws/voice/{session_id}` | Voice/video sessions | [Collaboration API](./collaboration-api.md#voicevideo-sessions) |
| `/ws/analytics/realtime` | Real-time metrics | [Analytics API](./analytics-api.md#real-time-monitoring) |

### Connection Example

```javascript
const ws = new WebSocket('wss://api.vega2.example.com/ws/collaboration/workspace_123?api_key=your-key');

ws.onopen = function() {
    console.log('Connected to workspace');
};

ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    handleMessage(message);
};

ws.onclose = function() {
    console.log('Disconnected from workspace');
};
```

## Federated Learning Integration

### Participant Setup

```python
from vega2_sdk.federated import FederatedParticipant

participant = FederatedParticipant(
    node_id='node_001',
    api_key='your-api-key',
    coordinator_url='https://api.vega2.example.com'
)

# Join training session
session = participant.join_training(
    session_id='fed_train_abc123',
    model_config={
        'architecture': 'transformer',
        'local_epochs': 5,
        'batch_size': 32
    }
)

# Start local training
results = participant.train_local_model(
    training_data=local_dataset,
    validation_data=validation_dataset
)
```

## Performance & Optimization

### Request Optimization

- **Batch requests**: Use batch endpoints when processing multiple items
- **Streaming**: Use streaming endpoints for real-time responses
- **Caching**: Implement client-side caching for static data
- **Connection pooling**: Reuse HTTP connections
- **Compression**: Enable gzip compression

### Response Formats

Most endpoints support multiple response formats via `Accept` header:

- `application/json` (default)
- `application/xml`
- `text/csv` (analytics endpoints)
- `application/octet-stream` (binary data)

## Monitoring & Observability

### Health Monitoring

```bash
# Check overall system health
curl https://api.vega2.example.com/analytics/system/health

# Monitor specific services
curl https://api.vega2.example.com/admin/processes/status

# Real-time metrics stream
wscat -c wss://api.vega2.example.com/ws/analytics/realtime
```

### Custom Metrics

```python
# Track custom application metrics
client.analytics.track_metric(
    name='custom.feature.usage',
    value=1,
    tags={'feature': 'chat', 'user_tier': 'premium'}
)

# Create custom dashboard
dashboard = client.analytics.create_dashboard(
    name='My App Dashboard',
    widgets=[
        {'type': 'line_chart', 'metric': 'custom.feature.usage'},
        {'type': 'counter', 'metric': 'api.requests.total'}
    ]
)
```

## Migration & Versioning

### API Versioning

- **Current**: v1 (default)
- **Upcoming**: v2 (beta)
- **Deprecated**: v0 (end-of-life: 2025-12-31)

Version selection via header:

```text
API-Version: v1
```

### Breaking Changes

Major version updates may include breaking changes. See [CHANGELOG.md](../CHANGELOG.md) for details.

### Migration Guide

When upgrading between major versions:

1. Review breaking changes documentation
2. Update SDK to compatible version
3. Test in staging environment
4. Gradually migrate production traffic
5. Monitor for errors and performance impact

## Support & Resources

### Documentation

- **API Reference**: This documentation
- **Tutorials**: [docs/tutorials/](../tutorials/)
- **Examples**: [examples/](../examples/)
- **SDKs**: Individual SDK documentation

### Community

- **GitHub**: [github.com/vega2/vega2.0](https://github.com/vega2/vega2.0)
- **Discord**: [Vega 2.0 Community](https://discord.gg/vega2)
- **Forum**: [community.vega2.ai](https://community.vega2.ai)

### Support Channels

- **Enterprise Support**: <support@vega2.ai>
- **Community Support**: GitHub Issues
- **Security Issues**: <security@vega2.ai>
- **Partnership Inquiries**: <partnerships@vega2.ai>

### Status Page

Monitor API status and incidents: [status.vega2.ai](https://status.vega2.ai)

## Terms & Compliance

- **Terms of Service**: [terms.vega2.ai](https://terms.vega2.ai)
- **Privacy Policy**: [privacy.vega2.ai](https://privacy.vega2.ai)
- **Data Processing Agreement**: Available for enterprise customers
- **Compliance**: GDPR, HIPAA, SOX compliant configurations available

---

*Last Updated: September 20, 2025*
*API Version: v1.0.0*
*Documentation Version: 2.0.0*
