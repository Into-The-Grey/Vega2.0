# Observability Stack for Vega 2.0

This directory contains a comprehensive observability stack including metrics, logging, and distributed tracing.

## Components

- **Grafana**: Visualization and dashboards
- **Prometheus**: Metrics collection and storage
- **Loki**: Log aggregation and storage
- **Promtail**: Log collection agent
- **Jaeger**: Distributed tracing

## Quick Start

```bash
# Start the entire observability stack
cd observability
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f grafana
```

## Access URLs

- **Grafana**: <http://localhost:3000> (admin/admin)
- **Prometheus**: <http://localhost:9090>
- **Jaeger UI**: <http://localhost:16686>
- **Loki**: <http://localhost:3100>

## Grafana Dashboards

### Pre-configured Dashboards

1. **Vega 2.0 - System Overview**
   - API response times and error rates
   - Request volume and patterns
   - System resource usage
   - Federated learning participant status

2. **Vega 2.0 - Federated Learning**
   - Training round progress
   - Participant health and performance
   - Model accuracy metrics
   - Communication overhead
   - Cross-silo federation status

### Dashboard Features

- **Real-time monitoring** with 5-10 second refresh
- **Multi-dimensional metrics** with detailed breakdowns
- **Alerting integration** (requires Alertmanager setup)
- **Custom time ranges** and zoom capabilities
- **Export and sharing** functionality

## Metrics Collected

### API Server Metrics

- `http_requests_total`: Total HTTP requests by method, endpoint, status
- `http_request_duration_seconds`: Request duration histograms
- `process_cpu_seconds_total`: CPU usage
- `process_resident_memory_bytes`: Memory usage

### Federated Learning Metrics

- `federated_participants_active`: Active participant count
- `federated_participants_total`: Total registered participants
- `federated_round_current`: Current training round
- `federated_model_accuracy`: Global model accuracy
- `federated_aggregation_duration_seconds`: Aggregation time
- `federated_bytes_sent_total`: Communication overhead

### System Metrics

- Container resource usage
- Network I/O
- Disk usage and I/O
- Database performance

## Log Collection

### Log Sources

- **Vega API**: Application logs with request tracing
- **Federated Coordinator**: Training progress and participant logs
- **System Logs**: Container and system-level logs
- **Docker Containers**: All container stdout/stderr

### Log Structure

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "module": "federated.coordinator",
  "message": "Round 5 completed successfully",
  "participant_id": "participant-001",
  "round": 5,
  "accuracy": 0.92
}
```

## Distributed Tracing

### Jaeger Integration

- **End-to-end tracing** of federated learning workflows
- **Performance profiling** of API endpoints
- **Service dependency mapping**
- **Error tracking** and root cause analysis

### Trace Collection

- Automatic instrumentation via OpenTelemetry
- Custom spans for federated learning operations
- Correlation with logs and metrics

## Configuration

### Prometheus Targets

Edit `prometheus.yml` to add new scrape targets:

```yaml
scrape_configs:
  - job_name: 'custom-service'
    static_configs:
      - targets: ['service:port']
```

### Grafana Data Sources

Data sources are automatically provisioned:

- Prometheus (metrics)
- Loki (logs)
- Jaeger (traces)

### Log Pipeline

Customize log collection in `promtail/promtail-config.yaml`:

- Add new log sources
- Configure parsing rules
- Set up label extraction

## Alerting (Optional)

To enable alerting, add Alertmanager:

```yaml
# Add to docker-compose.yml
alertmanager:
  image: prom/alertmanager:latest
  ports:
    - "9093:9093"
  volumes:
    - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
```

### Sample Alert Rules

```yaml
groups:
  - name: vega.rules
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          
      - alert: FederatedParticipantDown
        expr: federated_participants_active < 2
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Insufficient federated participants"
```

## Backup and Retention

### Data Retention

- **Prometheus**: 15 days (configurable)
- **Loki**: 168 hours (configurable)
- **Jaeger**: In-memory (use persistent storage for production)

### Backup Strategy

```bash
# Backup Prometheus data
docker-compose exec prometheus tar -czf /prometheus/backup-$(date +%Y%m%d).tar.gz /prometheus

# Backup Grafana dashboards
docker-compose exec grafana grafana-cli admin export-dash > dashboards-backup.json
```

## Production Considerations

### High Availability

- Run multiple Prometheus instances with federation
- Use Grafana clustering
- Deploy Loki in microservices mode
- Use external storage for persistence

### Security

- Enable authentication in Grafana
- Secure Prometheus with basic auth
- Use TLS for all communications
- Implement network policies in Kubernetes

### Performance Tuning

- Adjust scrape intervals based on load
- Configure appropriate retention periods
- Use recording rules for complex queries
- Implement metric cardinality limits

## Troubleshooting

### Common Issues

**Grafana Dashboard Not Loading**

```bash
# Check data source connectivity
docker-compose logs grafana
curl http://localhost:9090/api/v1/query?query=up
```

**Missing Metrics**

```bash
# Verify Prometheus targets
curl http://localhost:9090/api/v1/targets
# Check service health
curl http://localhost:8000/metrics
```

**Log Collection Issues**

```bash
# Check Promtail logs
docker-compose logs promtail
# Verify Loki ingestion
curl http://localhost:3100/ready
```

### Debug Commands

```bash
# Test metric queries
curl 'http://localhost:9090/api/v1/query?query=http_requests_total'

# Test log queries
curl -G -s "http://localhost:3100/loki/api/v1/query" --data-urlencode 'query={job="vega-api"}'

# Check Jaeger traces
curl http://localhost:16686/api/traces?service=vega-api
```
