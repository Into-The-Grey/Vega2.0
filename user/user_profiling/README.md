# User Profiling Engine (UPE) for Vega2.0

## Overview

The User Profiling Engine (UPE) is a sophisticated, privacy-conscious system that creates a comprehensive understanding of users through continuous learning and contextual intelligence. It enhances Vega2.0's AI interactions by providing personalized responses, behavioral adaptation, and proactive assistance.

## 🏗️ Architecture

### Core Components

1. **Database Schema** (`database/user_profile_schema.py`)
   - Comprehensive user data modeling with 10+ interconnected tables
   - Identity, calendar, financial, educational, social, and behavioral data
   - Versioning and timestamping for data evolution tracking

2. **Intelligence Collectors** (`collectors/`)
   - `profile_intel_collector.py`: Autonomous web intelligence harvesting
   - `calendar_sync.py`: Google Calendar and CalDAV integration
   - `finance_monitor.py`: Privacy-conscious financial behavior analysis

3. **Reasoning Engines** (`engines/`)
   - `edu_predictor.py`: Academic life monitoring and prediction
   - `persona_engine.py`: Behavioral mirroring with 8 persona modes

4. **Autonomous Daemon** (`user_profile_daemon.py`)
   - 24-hour enrichment cycles
   - Daily briefing generation
   - Understanding score tracking
   - Continuous context adaptation

5. **Vega2.0 Integration** (`vega_integration.py`)
   - Contextual intelligence injection
   - Enhanced chat responses
   - Profile management APIs

## 🚀 Quick Start

### 1. Installation & Setup

```bash
# Navigate to Vega2.0 project
cd /home/ncacord/Vega2.0

# Initialize user profiling database
python -m user_profiling.cli init-database

# Check system status
python -m user_profiling.cli status
```

### 2. Start Enhanced API Server

```bash
# Start the enhanced API with user profiling
python user_profiling/enhanced_api.py
```

The enhanced API runs on `http://127.0.0.1:8001` and provides:

- Enhanced chat with contextual intelligence (`/chat/enhanced`)
- Profile summary (`/profile/summary`)
- Daily briefings (`/profile/briefing`)
- Profile scanning (`/profile/scan`)

### 3. Start Autonomous Daemon

```bash
# Start background profiling daemon
python -m user_profiling.cli daemon
```

## 📊 Features

### Persona System

8 adaptive persona modes that adjust AI behavior based on user context:

- **Default**: Balanced, general-purpose interaction
- **Focused**: Concise, task-oriented responses during high-pressure periods
- **Burnout**: Supportive, reduced-demand communication when overwhelmed
- **Social**: Engaging, conversational when in social contexts
- **Academic**: Structured, deadline-aware during study periods
- **Professional**: Formal, efficient for work contexts
- **Creative**: Inspirational, open-ended for creative projects
- **Analytical**: Detailed, logical for problem-solving tasks

### Contextual Intelligence

- **Calendar Awareness**: Understands upcoming events, deadlines, stress levels
- **Academic Tracking**: Monitors courses, assignments, exam schedules
- **Financial Insights**: Privacy-conscious spending pattern analysis
- **Social Context**: Relationship mapping and communication preferences
- **Interest Profiling**: Hobby and interest engagement tracking

### Privacy Protection

- **Amount Tier Abstraction**: Financial data stored in privacy-preserving ranges
- **Configurable Collection**: User controls what data is collected
- **Local Processing**: All profiling happens locally, no external data sharing
- **Anonymization Options**: Sensitive data can be hashed or abstracted

## 🔧 CLI Commands

### Database Management

```bash
# Initialize database
python -m user_profiling.cli init-database

# Check system status
python -m user_profiling.cli status

# Export profile data
python -m user_profiling.cli export --output-file profile_backup.json
```

### Daily Operations

```bash
# Generate daily briefing
python -m user_profiling.cli briefing --save

# Check current persona state
python -m user_profiling.cli persona --show-context

# Run manual profile scan
python -m user_profiling.cli scan --scan-type full
```

### System Management

```bash
# Start autonomous daemon
python -m user_profiling.cli daemon

# Test integration with Vega2.0
python -m user_profiling.cli test-integration
```

## 🌐 API Endpoints

### Enhanced Chat

```http
POST /chat/enhanced
Content-Type: application/json
X-API-Key: your-api-key

{
  "prompt": "What should I focus on today?",
  "session_id": "optional-session-id",
  "include_persona": true,
  "include_context": true,
  "context_categories": ["calendar", "academic", "personal"]
}
```

### Profile Summary

```http
GET /profile/summary
X-API-Key: your-api-key
```

### Daily Briefing

```http
GET /profile/briefing?date=2024-01-15
X-API-Key: your-api-key
```

### Profile Scan

```http
POST /profile/scan?scan_type=mini
X-API-Key: your-api-key
```

## 📋 Daily Briefing Example

```json
{
  "date": "2024-01-15T00:00:00",
  "generated_at": "2024-01-15T08:00:00",
  "summary": "Today's overview: 3 scheduled events (moderate-stress); 2 urgent academic deadlines approaching; Personal well-being indicators are positive.",
  "sections": {
    "calendar": {
      "today_events_count": 3,
      "predicted_stress_level": 0.6,
      "next_event": {
        "title": "Team Meeting",
        "time": "10:00",
        "type": "work"
      }
    },
    "education": {
      "active_courses": 2,
      "urgent_deadlines": [
        {
          "title": "Research Paper",
          "course": "Advanced AI",
          "days_until": 2,
          "urgency": "high"
        }
      ]
    },
    "recommendations": [
      "Focus on urgent deadline: Research Paper due in 2 days",
      "Schedule breaks between today's meetings",
      "Review tomorrow's calendar for preparation time"
    ]
  }
}
```

## 🧠 Understanding Score

The system tracks how well it understands the user across multiple dimensions:

- **Identity Understanding**: 0.8 (name, age, location, preferences)
- **Calendar Patterns**: 0.9 (routine recognition, stress prediction)
- **Academic Life**: 0.7 (course tracking, deadline awareness)
- **Social Context**: 0.6 (relationship mapping, communication style)
- **Interests**: 0.8 (hobby engagement, topic preferences)

**Overall Understanding Score**: 0.76/1.0

## 🔄 Autonomous Enrichment Cycles

The daemon runs continuous background processes:

### Full Scan (Daily at 2:00 AM)

- Web intelligence collection
- Calendar synchronization
- Financial data analysis
- Educational content parsing
- Persona state updates

### Mini Scan (Every 6 Hours)

- Calendar updates
- Persona adjustments
- Quick context refresh

### Continuous (Every 30 Minutes)

- Persona state monitoring
- Understanding score updates
- Context adaptation

## 🛡️ Privacy & Security

### Data Protection

- All data stored locally in SQLite database
- No external data transmission without explicit user consent
- Financial amounts abstracted into privacy-preserving tiers
- Personal information can be anonymized or hashed

### Access Control

- API key authentication for all endpoints
- Rate limiting on profile operations
- Configurable data collection boundaries
- User-controlled retention policies

### Compliance Features

- Data export capabilities for portability
- Granular deletion options
- Audit trail for all profile modifications
- Privacy mode settings (strict/balanced/open)

## 🔧 Configuration

### Daemon Configuration

```python
config = DaemonConfig(
    scan_interval_hours=24,
    mini_scan_interval_hours=6,
    persona_update_interval_minutes=30,
    briefing_generation_hour=8,
    enable_intelligence_collection=True,
    enable_calendar_sync=True,
    enable_financial_monitoring=True,
    enable_educational_analysis=True,
    enable_persona_tracking=True,
    max_concurrent_tasks=3
)
```

### Privacy Configuration

```python
profile_config = UserProfileConfig(
    enable_profiling=True,
    enable_contextual_responses=True,
    enable_auto_enrichment=True,
    max_context_injection_length=500,
    privacy_mode="balanced"  # strict, balanced, open
)
```

## 📈 Performance Monitoring

### System Metrics

- Database query performance
- Memory usage during scans
- API response times
- Daemon uptime and stability

### Understanding Metrics

- Profile completeness score
- Context relevance accuracy
- Persona switching frequency
- User satisfaction indicators

## 🚧 Development & Extensibility

### Adding New Data Sources

1. Create collector in `collectors/` directory
2. Implement async collection interface
3. Register with daemon scheduler
4. Add privacy controls and rate limiting

### Custom Persona Modes

1. Extend `PersonaMode` enum in `persona_engine.py`
2. Add behavior settings and triggers
3. Implement context analysis logic
4. Update response modification rules

### Integration Patterns

```python
# Custom intelligence injection
async def custom_enhancement(response, context):
    if 'custom_trigger' in context:
        response += "\n\n*Custom insight based on your profile*"
    return response

# Register with contextual intelligence engine
intelligence_engine.add_custom_enhancer(custom_enhancement)
```

## 🐛 Troubleshooting

### Common Issues

**Database Lock Errors**

```bash
# Check for zombie processes
ps aux | grep user_profiling

# Restart daemon
python -m user_profiling.cli daemon
```

**Import Errors**

```bash
# Test integration
python -m user_profiling.cli test-integration

# Check Python path
python -c "import sys; print(sys.path)"
```

**High Memory Usage**

- Reduce scan frequency in daemon config
- Implement conversation retention policies
- Monitor large data imports

### Logging

```bash
# Check daemon logs
tail -f user_profile_daemon.log

# Enable debug logging
export VEGA_LOG_LEVEL=DEBUG
```

## 🔮 Future Enhancements

### Planned Features

- Multi-modal data processing (images, documents)
- Federated learning across devices
- Real-time collaboration insights
- Advanced behavioral prediction models
- Voice interaction pattern analysis

### Scalability Roadmap

- PostgreSQL migration for multi-user scenarios
- Horizontal scaling with message queues
- Containerization and Kubernetes deployment
- Distributed training capabilities

## 📚 Documentation

- **Architecture Details**: `docs/ARCHITECTURE.md`
- **API Reference**: `docs/api/`
- **Privacy Guide**: `docs/PRIVACY.md`
- **Development Guide**: `docs/DEVELOPMENT.md`

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Follow code style guidelines
4. Add comprehensive tests
5. Update documentation
6. Submit pull request

## 📜 License

This project is part of Vega2.0 and follows the same licensing terms.

---

**🎯 The User Profiling Engine transforms Vega2.0 from a reactive chat system into a proactive, contextually-aware AI companion that grows with the user over time.**
