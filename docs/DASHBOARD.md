# Vega2.0 Real-Time Autonomous AI Dashboard

## Overview

The Vega2.0 Dashboard provides real-time monitoring and visualization of the autonomous AI improvement system. It offers comprehensive insights into system performance, autonomous improvement cycles, knowledge extraction, skill evolution, and conversation quality analysis.

## Features

### 🚀 Real-Time Monitoring

- **System Health**: Live monitoring of overall system health metrics
- **Performance Tracking**: Real-time performance baseline and improvements
- **Resource Usage**: CPU, memory, and disk usage monitoring
- **Autonomous Cycles**: Count and timing of self-improvement cycles

### 📊 Interactive Dashboard

- **Live Metrics**: Real-time updates every 5 seconds via WebSocket
- **Visual Timeline**: Recent improvement actions and their impact
- **Quality Analysis**: Conversation quality scores and trends
- **Knowledge Growth**: Tracking of knowledge graph expansion

### 🔧 Manual Controls

- **Trigger Improvements**: Manual improvement cycle activation
- **Data Refresh**: Force refresh of all dashboard data
- **Auto-Refresh Toggle**: Enable/disable automatic data updates
- **System Log**: Real-time logging of system events

### 🤖 Conversation Integration

- **Quality Analysis**: Real-time conversation quality assessment
- **Improvement Suggestions**: AI-generated improvement recommendations
- **Pattern Recognition**: Identification of conversation patterns
- **Feedback Integration**: User feedback integration with improvement system

## Quick Start

### 1. Start the Vega2.0 Server

```bash
cd /home/ncacord/Vega2.0
source .venv/bin/activate
uvicorn app:app --host 127.0.0.1 --port 8000
```

### 2. Access the Dashboard

Open your browser and navigate to:

```
http://127.0.0.1:8000/dashboard
```

### 3. Enter API Key

When prompted, enter your API key (from `app.env` file):

```
sk-test-key-123  # or your configured API key
```

## API Endpoints

### Dashboard Endpoints (require API key)

#### Get Current Metrics

```http
GET /dashboard/metrics
X-API-Key: YOUR_API_KEY
```

Returns real-time system metrics including health, performance, and activity.

#### Get Improvement Timeline

```http
GET /dashboard/timeline?hours=24
X-API-Key: YOUR_API_KEY
```

Returns improvement actions from the last N hours (default: 24).

#### Get Skill Evolution

```http
GET /dashboard/skills
X-API-Key: YOUR_API_KEY
```

Returns skill versioning and evolution data.

#### Get Knowledge Graph Stats

```http
GET /dashboard/knowledge
X-API-Key: YOUR_API_KEY
```

Returns knowledge graph statistics and growth metrics.

#### Trigger Manual Improvement

```http
POST /dashboard/trigger
X-API-Key: YOUR_API_KEY
```

Manually triggers an improvement cycle for testing.

#### WebSocket Connection

```javascript
const ws = new WebSocket('ws://127.0.0.1:8000/dashboard/ws');
```

Real-time updates for metrics and improvement events.

### Enhanced Chat Endpoint

The `/chat` endpoint now supports conversation analysis:

```http
POST /chat
X-API-Key: YOUR_API_KEY
Content-Type: application/json

{
  "prompt": "Your message here",
  "stream": false,
  "session_id": "optional-session-id",
  "include_analysis": true
}
```

When `include_analysis` is true, the response includes:

- Quality analysis scores (relevance, clarity, helpfulness, completeness)
- Improvement suggestions with confidence scores
- Areas for improvement

## Dashboard Components

### System Health Card

- **Overall Health**: 0-100% health score
- **Progress Bar**: Visual health indicator
- **Status Updates**: Real-time health changes

### Active Improvements Card

- **Count**: Number of currently active improvement processes
- **Real-time**: Updates as improvements are triggered

### Performance Metrics

- **Baseline Score**: Current performance baseline
- **Trend Tracking**: Performance improvements over time
- **Impact Assessment**: Measured improvement impact

### Knowledge Growth

- **Item Count**: Total knowledge items in graph
- **Growth Rate**: Recent knowledge extraction rate
- **Relationship Density**: Knowledge graph connectivity

### Improvement Timeline

- **Recent Actions**: Last 10 improvement actions
- **Phase Information**: Which system phase triggered improvement
- **Impact Scores**: Measured or estimated impact
- **Timestamps**: When improvements occurred

### System Controls

- **Manual Trigger**: Force an improvement cycle
- **Refresh Data**: Update all dashboard data
- **Auto-Refresh**: Toggle automatic updates
- **System Log**: Real-time event logging

## Advanced Features

### Conversation Quality Analysis

The system automatically analyzes every conversation for:

1. **Relevance** (0-1.0): How well the response addresses the prompt
2. **Clarity** (0-1.0): How clear and understandable the response is
3. **Helpfulness** (0-1.0): How useful the response is to the user
4. **Completeness** (0-1.0): How thoroughly the response addresses the question

### Improvement Suggestions

Based on conversation patterns, the system generates:

- Response length optimization suggestions
- Question answering improvement recommendations
- Clarity and structure enhancements
- Topical focus suggestions

### Auto-Improvement Triggers

The system automatically triggers improvements when:

- Conversation quality falls below thresholds (< 0.5)
- Negative user feedback is received
- Performance degradation is detected
- Pattern recognition identifies opportunities

## Monitoring & Alerts

### Quality Thresholds

- **Excellent**: > 0.85 overall score
- **Good**: 0.70 - 0.85 overall score
- **Poor**: < 0.50 overall score (triggers auto-improvement)

### Health Indicators

- **Green**: System health > 70%
- **Yellow**: System health 40-70%
- **Red**: System health < 40%

### Real-time Events

- Improvement cycles starting/completing
- Quality threshold breaches
- Knowledge extraction events
- Skill version updates
- System health changes

## Troubleshooting

### Dashboard Not Loading

1. Verify server is running: `curl http://127.0.0.1:8000/healthz`
2. Check API key configuration in `app.env`
3. Verify `static/dashboard.html` exists
4. Check browser console for JavaScript errors

### WebSocket Connection Issues

1. Ensure WebSocket endpoint is accessible
2. Check firewall settings for port 8000
3. Verify browser WebSocket support
4. Check server logs for WebSocket errors

### Missing Data

1. Verify all autonomous system phases are running
2. Check database files exist (telemetry.db, variants.db, etc.)
3. Ensure improvement cycles have been triggered
4. Check for database permission issues

### Performance Issues

1. Monitor system resource usage
2. Check database size and performance
3. Adjust cache TTL settings if needed
4. Consider reducing update frequency

## Configuration

### Environment Variables

```bash
# Required
API_KEY=your-secret-api-key

# Optional dashboard settings
DASHBOARD_UPDATE_INTERVAL=5  # seconds
DASHBOARD_CACHE_TTL=30      # seconds
DASHBOARD_MAX_TIMELINE=50   # items
```

### Dashboard Settings

```python
# In dashboard.py
cache_ttl = 30              # Cache duration in seconds
max_history_size = 50       # Maximum metrics history
update_interval = 5         # WebSocket update interval
```

## Security Considerations

- All dashboard endpoints require valid API key
- WebSocket connections are authenticated
- Dashboard only accessible on localhost by default
- No sensitive data exposed in client-side code
- API keys never logged or transmitted insecurely

## Integration Examples

### Custom Dashboard Widgets

```javascript
// Get metrics and create custom visualization
async function createCustomWidget() {
    const response = await fetch('/dashboard/metrics', {
        headers: {'X-API-Key': 'your-key'}
    });
    const metrics = await response.json();
    
    // Create custom chart or widget
    updateCustomChart(metrics);
}
```

### External Monitoring

```python
import httpx

async def monitor_vega_health():
    async with httpx.AsyncClient() as client:
        response = await client.get(
            'http://127.0.0.1:8000/dashboard/metrics',
            headers={'X-API-Key': 'your-key'}
        )
        metrics = response.json()
        
        if metrics['system_health'] < 0.5:
            send_alert("Vega system health degraded")
```

### Automation Triggers

```bash
# Trigger improvement via API
curl -X POST \
  -H "X-API-Key: your-key" \
  http://127.0.0.1:8000/dashboard/trigger
```

## Support

For issues or questions:

1. Check the system logs: `journalctl -u vega -f`
2. Review dashboard console logs in browser
3. Verify all dependencies are installed
4. Check database connectivity and permissions
5. Consult the main Vega2.0 documentation

## Future Enhancements

Planned dashboard improvements:

- Historical trend analysis with charts
- Custom alert configuration
- Export capabilities for metrics
- Advanced filtering and search
- Mobile-responsive design improvements
- Integration with external monitoring systems
