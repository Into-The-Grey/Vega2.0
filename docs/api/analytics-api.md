# Vega 2.0 Analytics & Monitoring API Documentation

## Overview

The Vega 2.0 Analytics & Monitoring API provides comprehensive insights into system performance, user behavior, AI model metrics, federated learning progress, and operational health monitoring.

**Base URL**: `http://localhost:8000` (development) | `https://api.vega2.example.com` (production)

**Authentication**: X-API-Key header required for all endpoints

## System Metrics

### GET /analytics/system/health

Get comprehensive system health metrics

**Response**:

```json
{
  "timestamp": "2025-09-20T10:00:00Z",
  "status": "healthy",
  "uptime_seconds": 86400,
  "system": {
    "cpu_usage": 45.2,
    "memory_usage": 68.5,
    "disk_usage": 32.1,
    "load_average": [1.2, 1.5, 1.8],
    "temperature": 58.3
  },
  "services": {
    "database": {
      "status": "healthy",
      "response_time_ms": 12,
      "connections": 5,
      "pool_size": 20
    },
    "llm_backend": {
      "status": "healthy", 
      "model": "llama3.2:latest",
      "response_time_ms": 850,
      "queue_length": 2
    },
    "circuit_breakers": {
      "search_service": "closed",
      "osint_service": "half_open",
      "slack_connector": "closed"
    }
  },
  "alerts": [
    {
      "level": "warning",
      "message": "High memory usage detected",
      "threshold": 70.0,
      "current": 68.5
    }
  ]
}
```

### GET /analytics/system/performance

Get detailed performance metrics

**Parameters**:

- `timeframe` (query): Time period (1h, 6h, 24h, 7d, 30d)
- `granularity` (query): Data points (1m, 5m, 15m, 1h, 1d)

**Response**:

```json
{
  "timeframe": "24h",
  "granularity": "1h",
  "metrics": [
    {
      "timestamp": "2025-09-20T09:00:00Z",
      "cpu_usage": 42.1,
      "memory_usage": 65.3,
      "disk_io_read": 1250000,
      "disk_io_write": 890000,
      "network_in": 125000000,
      "network_out": 89000000,
      "request_count": 1250,
      "response_time_avg": 245,
      "response_time_p95": 850,
      "response_time_p99": 1200,
      "error_rate": 0.02
    }
  ],
  "summary": {
    "avg_cpu": 43.8,
    "max_memory": 72.1,
    "total_requests": 28750,
    "avg_response_time": 198,
    "total_errors": 23
  }
}
```

### GET /analytics/system/logs

Search and filter system logs

**Parameters**:

- `level` (query): Log level (debug, info, warning, error, critical)
- `component` (query): Component name (api, database, llm, federated)
- `start_time` (query): ISO timestamp
- `end_time` (query): ISO timestamp
- `limit` (query): Max entries (default: 100)
- `search` (query): Text search in log messages

**Response**:

```json
{
  "logs": [
    {
      "timestamp": "2025-09-20T10:15:23.456Z",
      "level": "error",
      "component": "llm",
      "message": "Model inference timeout after 30s",
      "context": {
        "session_id": "sess_abc123",
        "model": "llama3.2:latest",
        "prompt_length": 2048,
        "timeout_seconds": 30
      },
      "trace_id": "trace_xyz789"
    }
  ],
  "total": 1,
  "has_more": false,
  "aggregations": {
    "by_level": {
      "error": 1,
      "warning": 5,
      "info": 94
    },
    "by_component": {
      "llm": 25,
      "api": 50,
      "database": 15,
      "federated": 10
    }
  }
}
```

## Usage Analytics

### GET /analytics/usage/overview

Get high-level usage statistics

**Parameters**:

- `timeframe` (query): Time period (today, week, month, year)

**Response**:

```json
{
  "timeframe": "month",
  "period": {
    "start": "2025-09-01T00:00:00Z",
    "end": "2025-09-30T23:59:59Z"
  },
  "totals": {
    "unique_users": 1250,
    "total_sessions": 8750,
    "chat_messages": 125000,
    "ai_requests": 95000,
    "documents_created": 2500,
    "collaborations": 1850,
    "voice_sessions": 450
  },
  "growth": {
    "users_growth": 12.5,
    "sessions_growth": 18.2,
    "messages_growth": 25.8
  },
  "top_features": [
    {
      "feature": "chat",
      "usage_count": 95000,
      "percentage": 45.2
    },
    {
      "feature": "collaboration",
      "usage_count": 35000,
      "percentage": 16.7
    },
    {
      "feature": "voice_sessions",
      "usage_count": 22000,
      "percentage": 10.5
    }
  ]
}
```

### GET /analytics/usage/users

Get user activity analytics

**Parameters**:

- `timeframe` (query): Time period (day, week, month)
- `segment` (query): User segment (new, active, power, churned)

**Response**:

```json
{
  "timeframe": "week",
  "user_metrics": {
    "total_users": 1250,
    "active_users": 890,
    "new_users": 85,
    "returning_users": 805,
    "churned_users": 35
  },
  "engagement": {
    "avg_session_duration": 1850,
    "avg_sessions_per_user": 6.8,
    "avg_messages_per_session": 12.5,
    "retention_rate": {
      "day_1": 0.85,
      "day_7": 0.68,
      "day_30": 0.45
    }
  },
  "user_segments": [
    {
      "segment": "power_users",
      "count": 125,
      "percentage": 10.0,
      "avg_daily_sessions": 8.5,
      "characteristics": ["high_frequency", "long_sessions", "feature_adoption"]
    }
  ],
  "activity_patterns": {
    "peak_hours": [9, 10, 14, 15, 16],
    "peak_days": ["monday", "tuesday", "wednesday"],
    "geographic_distribution": {
      "us": 45.2,
      "eu": 32.8,
      "asia": 18.5,
      "other": 3.5
    }
  }
}
```

### GET /analytics/usage/features

Get feature usage analytics

**Response**:

```json
{
  "features": [
    {
      "name": "chat",
      "usage_count": 95000,
      "unique_users": 1200,
      "avg_per_user": 79.2,
      "adoption_rate": 0.96,
      "satisfaction_score": 4.2,
      "trends": {
        "daily_change": 2.5,
        "weekly_change": 15.8,
        "monthly_change": 45.2
      }
    },
    {
      "name": "collaboration",
      "usage_count": 35000,
      "unique_users": 650,
      "avg_per_user": 53.8,
      "adoption_rate": 0.52,
      "satisfaction_score": 4.5,
      "trends": {
        "daily_change": 5.2,
        "weekly_change": 28.5,
        "monthly_change": 125.8
      }
    }
  ],
  "feature_funnels": {
    "onboarding": {
      "signup": 1000,
      "first_chat": 850,
      "first_collaboration": 450,
      "active_user": 680
    }
  }
}
```

## AI Model Analytics

### GET /analytics/ai/performance

Get AI model performance metrics

**Parameters**:

- `model` (query): Filter by model name
- `timeframe` (query): Time period

**Response**:

```json
{
  "models": [
    {
      "name": "llama3.2:latest",
      "requests": 45000,
      "success_rate": 0.985,
      "avg_response_time": 1250,
      "p95_response_time": 2500,
      "p99_response_time": 4200,
      "timeout_rate": 0.008,
      "error_rate": 0.007,
      "quality_metrics": {
        "avg_response_length": 245,
        "coherence_score": 4.3,
        "relevance_score": 4.1,
        "user_satisfaction": 4.2
      },
      "resource_usage": {
        "avg_tokens_per_request": 185,
        "total_tokens_processed": 8325000,
        "avg_memory_mb": 2048,
        "avg_cpu_percent": 75.2
      }
    }
  ],
  "trends": {
    "request_volume": [
      {"date": "2025-09-19", "count": 1200},
      {"date": "2025-09-20", "count": 1350}
    ],
    "response_times": [
      {"date": "2025-09-19", "avg_ms": 1180},
      {"date": "2025-09-20", "avg_ms": 1250}
    ]
  }
}
```

### GET /analytics/ai/conversations

Analyze conversation patterns and quality

**Response**:

```json
{
  "conversation_metrics": {
    "total_conversations": 15000,
    "avg_length": 8.5,
    "avg_duration": 450,
    "completion_rate": 0.85,
    "satisfaction_distribution": {
      "very_satisfied": 0.45,
      "satisfied": 0.35,
      "neutral": 0.15,
      "unsatisfied": 0.04,
      "very_unsatisfied": 0.01
    }
  },
  "topic_analysis": [
    {
      "topic": "technical_support",
      "count": 3500,
      "percentage": 23.3,
      "avg_satisfaction": 4.1,
      "resolution_rate": 0.82
    },
    {
      "topic": "creative_writing",
      "count": 2800,
      "percentage": 18.7,
      "avg_satisfaction": 4.4,
      "resolution_rate": 0.95
    }
  ],
  "sentiment_analysis": {
    "positive": 0.65,
    "neutral": 0.25,
    "negative": 0.10,
    "trend": "improving"
  },
  "language_usage": {
    "english": 0.75,
    "spanish": 0.12,
    "french": 0.08,
    "other": 0.05
  }
}
```

## Federated Learning Analytics

### GET /analytics/federated/training

Get federated learning training metrics

**Response**:

```json
{
  "training_sessions": [
    {
      "session_id": "fed_train_abc123",
      "model": "llama3.2-federated",
      "status": "completed",
      "started_at": "2025-09-20T08:00:00Z",
      "completed_at": "2025-09-20T10:30:00Z",
      "duration_minutes": 150,
      "participants": {
        "total": 8,
        "active": 8,
        "dropped": 0
      },
      "rounds": 25,
      "convergence": {
        "achieved": true,
        "final_accuracy": 0.925,
        "improvement": 0.045
      },
      "resource_usage": {
        "total_compute_hours": 1200,
        "avg_bandwidth_mbps": 15.2,
        "data_transferred_gb": 125.8
      }
    }
  ],
  "aggregated_metrics": {
    "total_sessions": 45,
    "avg_participants": 6.8,
    "avg_duration_hours": 2.5,
    "success_rate": 0.89,
    "avg_improvement": 0.035
  }
}
```

### GET /analytics/federated/participants

Analyze federated learning participant performance

**Response**:

```json
{
  "participants": [
    {
      "participant_id": "node_001",
      "organization": "University A",
      "location": "US-East",
      "sessions_participated": 12,
      "avg_contribution_quality": 0.88,
      "reliability_score": 0.95,
      "data_quality_score": 0.92,
      "communication_latency_ms": 45,
      "bandwidth_mbps": 25.8,
      "uptime_percentage": 98.5
    }
  ],
  "network_topology": {
    "total_nodes": 15,
    "active_nodes": 12,
    "clusters": [
      {
        "cluster_id": "us_east",
        "nodes": 5,
        "avg_latency": 25,
        "data_locality_score": 0.85
      }
    ],
    "connectivity_matrix": {
      "avg_latency": 65,
      "max_latency": 250,
      "reliability": 0.94
    }
  }
}
```

### GET /analytics/federated/privacy

Monitor privacy preservation metrics

**Response**:

```json
{
  "privacy_metrics": {
    "differential_privacy": {
      "epsilon": 1.0,
      "delta": 1e-5,
      "noise_level": 0.1,
      "privacy_budget_used": 0.35
    },
    "secure_aggregation": {
      "enabled": true,
      "key_exchanges": 150,
      "failed_exchanges": 2,
      "encryption_overhead": 0.05
    },
    "data_minimization": {
      "avg_data_per_round": 1024,
      "compression_ratio": 0.15,
      "feature_selection_rate": 0.80
    }
  },
  "compliance": {
    "gdpr_compliant": true,
    "hipaa_compliant": true,
    "sox_compliant": false,
    "audit_trail_complete": true
  },
  "risk_assessment": {
    "membership_inference_risk": "low",
    "model_inversion_risk": "very_low",
    "property_inference_risk": "low",
    "overall_risk_score": 0.15
  }
}
```

## Business Intelligence

### GET /analytics/business/revenue

Get revenue and business metrics

**Response**:

```json
{
  "revenue": {
    "current_month": 125000,
    "previous_month": 108000,
    "growth_rate": 0.157,
    "yearly_recurring": 1380000,
    "customer_lifetime_value": 2500,
    "churn_rate": 0.05
  },
  "subscriptions": {
    "total_active": 850,
    "new_this_month": 85,
    "churned_this_month": 42,
    "net_growth": 43,
    "by_plan": {
      "basic": 450,
      "professional": 320,
      "enterprise": 80
    }
  },
  "usage_monetization": {
    "ai_requests_billable": 125000,
    "revenue_per_request": 0.02,
    "collaboration_hours": 8500,
    "revenue_per_hour": 1.50
  }
}
```

### GET /analytics/business/customers

Analyze customer behavior and segmentation

**Response**:

```json
{
  "customer_segments": [
    {
      "segment": "enterprise",
      "count": 80,
      "revenue_percentage": 55.2,
      "avg_monthly_value": 8650,
      "churn_rate": 0.02,
      "satisfaction_score": 4.6,
      "growth_rate": 0.25
    }
  ],
  "customer_health": {
    "healthy": 0.75,
    "at_risk": 0.20,
    "churning": 0.05
  },
  "usage_patterns": {
    "high_engagement": 0.35,
    "medium_engagement": 0.45,
    "low_engagement": 0.20
  },
  "support_metrics": {
    "avg_response_time_hours": 2.5,
    "resolution_rate": 0.92,
    "satisfaction_score": 4.3,
    "escalation_rate": 0.08
  }
}
```

## Custom Reports

### POST /analytics/reports/custom

Create custom analytics report

**Request Body**:

```json
{
  "name": "Weekly AI Performance Report",
  "description": "Weekly analysis of AI model performance",
  "metrics": [
    "ai.performance.response_time",
    "ai.performance.success_rate",
    "usage.ai_requests",
    "system.cpu_usage"
  ],
  "filters": {
    "timeframe": "7d",
    "models": ["llama3.2:latest"],
    "user_segments": ["enterprise", "professional"]
  },
  "groupby": ["date", "model"],
  "schedule": {
    "frequency": "weekly",
    "day": "monday",
    "time": "09:00",
    "timezone": "UTC"
  },
  "delivery": {
    "email": ["admin@example.com"],
    "webhook": "https://example.com/analytics-webhook",
    "format": "json"
  }
}
```

**Response**:

```json
{
  "report_id": "report_abc123",
  "status": "scheduled",
  "next_run": "2025-09-23T09:00:00Z",
  "created_at": "2025-09-20T10:45:00Z"
}
```

### GET /analytics/reports/{report_id}

Get report data or status

**Response**:

```json
{
  "report_id": "report_abc123",
  "name": "Weekly AI Performance Report",
  "status": "completed",
  "generated_at": "2025-09-23T09:05:23Z",
  "data": {
    "summary": {
      "timeframe": "2025-09-16 to 2025-09-22",
      "total_requests": 125000,
      "avg_response_time": 1250,
      "success_rate": 0.985
    },
    "details": [
      {
        "date": "2025-09-16",
        "model": "llama3.2:latest",
        "requests": 18500,
        "avg_response_time": 1180,
        "success_rate": 0.987
      }
    ],
    "insights": [
      "Response times improved 5% week over week",
      "Success rate maintained above 98% threshold",
      "Peak usage occurred Tuesday 2-4 PM"
    ]
  },
  "export_urls": {
    "json": "https://api.vega2.example.com/exports/report_abc123.json",
    "csv": "https://api.vega2.example.com/exports/report_abc123.csv",
    "pdf": "https://api.vega2.example.com/exports/report_abc123.pdf"
  }
}
```

## Real-time Monitoring

### WebSocket: /ws/analytics/realtime

Connect to real-time analytics stream:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/analytics/realtime?api_key=your_key');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    updateDashboard(data);
};
```

**Real-time Message Types**:

```json
{
  "type": "metric_update",
  "metric": "system.cpu_usage",
  "value": 45.2,
  "timestamp": "2025-09-20T10:50:00Z"
}
```

```json
{
  "type": "alert",
  "level": "warning",
  "metric": "ai.response_time",
  "value": 5000,
  "threshold": 3000,
  "message": "AI response time exceeded threshold"
}
```

## Error Responses

### 400 Invalid Parameters

```json
{
  "error": "Invalid timeframe parameter",
  "code": 400,
  "details": {
    "parameter": "timeframe",
    "provided": "invalid",
    "valid_options": ["1h", "6h", "24h", "7d", "30d"]
  }
}
```

### 403 Insufficient Analytics Permissions

```json
{
  "error": "Insufficient permissions for analytics access",
  "code": 403,
  "required_role": "analytics_viewer",
  "current_permissions": ["basic_user"]
}
```

### 429 Rate Limit Exceeded

```json
{
  "error": "Analytics API rate limit exceeded",
  "code": 429,
  "limit": 100,
  "window": 3600,
  "retry_after": 1800
}
```
