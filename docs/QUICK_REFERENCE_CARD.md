# Quick Reference: Phase 3 & 4 Optimizations

## 🚀 Quick Start

### Verify Installation

```bash
# Run Phase 3 verification
python tools/verify_phase3.py

# Check server startup
python main.py server --host 127.0.0.1 --port 8000
# Look for ✅ indicators in startup sequence
```

### Run Integration Tests

```bash
# All tests
pytest tests/integration/test_integrations.py -v

# Specific category
pytest tests/integration/test_integrations.py -k "slack" -v

# With coverage
pytest tests/integration/ --cov=src.vega --cov-report=html
```

---

## 📊 Monitoring Endpoints

### Health & Status

```bash
API_KEY="your_api_key_here"

# Overall system health
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/diagnostics/health-summary | jq

# Integration health checks
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/integrations/health | jq

# Full system diagnostics
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/diagnostics/system | jq
```

### Database Performance

```bash
# Query performance metrics
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/database/stats | jq

# Reset statistics
curl -X POST -H "X-API-Key: $API_KEY" http://localhost:8000/admin/database/reset-stats
```

### Advanced Performance

```bash
# Request deduplication stats
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/performance/request-coalescing | jq

# Connection pool metrics
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/performance/connection-pool | jq

# Circuit breaker status
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/performance/circuit-breakers | jq

# Response cache statistics
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/performance/cache-stats | jq

# Streaming metrics
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/performance/streaming-stats | jq

# Event loop health
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/performance/async-monitor | jq

# Memory leak detection
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/performance/memory-leaks | jq

# Batch operation stats
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/performance/batch-stats | jq
```

### Configuration

```bash
# Validate configuration
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/config/validate | jq
```

---

## 🔧 Using New Features

### Request Coalescing

```python
from src.vega.core.request_coalescing import get_llm_coalescer

coalescer = get_llm_coalescer()

# Coalesce identical requests
result = await coalescer.coalesce(
    "llm_generate",
    your_llm_function,
    prompt="Hello",
    model="llama3"
)

# Check metrics
stats = coalescer.get_stats()
print(f"Coalesce rate: {stats['coalesce_rate']:.1f}%")
print(f"Cache hit rate: {stats['cache_hit_rate']:.1f}%")
```

### Connection Pool Tracking

```python
from src.vega.core.connection_pool import get_connection_pool_manager

manager = get_connection_pool_manager()

# Register connection
manager.register_connection(conn_id=12345, host="api.example.com")

# Record request
manager.record_request(conn_id=12345, success=True)

# Check if can create new
if manager.can_create_connection("api.example.com"):
    # Create new connection
    pass

# Get stats
stats = manager.get_all_stats()
print(f"Reuse rate: {stats['metrics']['reuse_rate']:.1f}%")
```

### Streaming with Backpressure

```python
from src.vega.core.streaming_backpressure import BackpressureStream

async def data_generator():
    for i in range(100):
        yield f"chunk_{i}"

stream = BackpressureStream(
    data_generator(),
    max_buffer_size=10,
    warning_threshold=0.7
)

async for chunk in stream:
    # Process chunk
    print(chunk)

# Get metrics
metrics = stream.get_metrics()
print(f"Produced: {metrics.chunks_produced}")
print(f"Consumed: {metrics.chunks_consumed}")
print(f"Dropped: {metrics.chunks_dropped}")
```

### Async Event Loop Monitoring

```python
from src.vega.core.async_monitor import AsyncEventLoopMonitor

monitor = AsyncEventLoopMonitor(slow_callback_threshold_ms=100.0)
monitor.start_monitoring()

@monitor.wrap_callback
async def my_operation():
    # This will be monitored for slow execution
    await some_work()
    return result

# Get metrics
metrics = monitor.get_metrics()
print(f"Blocked callbacks: {metrics.blocked_callbacks}")
print(f"Max callback time: {metrics.max_callback_time:.2f}ms")

monitor.stop_monitoring()
```

### Memory Leak Detection

```python
from src.vega.core.memory_leak_detector import MemoryLeakDetector

detector = MemoryLeakDetector()

# Track objects
obj = MyObject()
detector.track(obj, "my_object")

# Get stats
stats = detector.get_stats()
print(f"Live objects: {stats['live_objects']}")
print(f"Total tracked: {stats['total_tracked']}")

# Get report
report = detector.get_detailed_report()
for leak in report:
    print(f"Potential leak: {leak['name']} (age: {leak['age_seconds']:.1f}s)")
```

### Batch Operations

```python
from src.vega.core.batch_operations import BatchWriter

async def batch_processor(items):
    # Process batch of items
    await db.insert_many(items)

writer = BatchWriter(
    batch_processor,
    max_batch_size=100,
    flush_interval=5.0
)

writer.start_auto_flush()

# Add items
await writer.add(item1)
await writer.add(item2)

# Manual flush if needed
await writer.flush()

# Stop when done
writer.stop_auto_flush()
```

---

## 📈 Key Metrics to Monitor

### Request Efficiency

- **Coalesce Rate**: Should be >20% during traffic spikes
- **Cache Hit Rate**: Should be >30% for repeated requests
- **Average Wait Time**: Should be <10ms for coalesced requests

### Connection Health

- **Reuse Rate**: Should be >50%, target 70%+
- **Total Connections**: Should stay under limits
- **Error Rate**: Should be <5%

### System Performance

- **Memory Usage**: Should be <80% of available
- **CPU Usage**: Should be <80% sustained
- **Event Loop Lag**: Should be <10ms
- **Slow Callbacks**: Should be <1% of total

### Database Performance

- **Average Query Time**: Should be <10ms for indexed queries
- **Slow Queries**: Should be <1% of total
- **Connection Pool**: Should have available connections

---

## ⚠️ Alert Thresholds

### Critical (Immediate Action)

- Memory usage >90%
- Error rate >20%
- Circuit breaker open
- Connection pool exhausted
- Event loop blocked >1s

### Warning (Monitor Closely)

- Memory usage >80%
- Error rate >10%
- Coalesce rate <10%
- Connection reuse <30%
- Slow queries >5%

### Info (Review Eventually)

- Cache hit rate <20%
- Connection reuse <50%
- Slow callbacks >1%
- Buffer overflows detected

---

## 🐛 Troubleshooting

### High Memory Usage

```bash
# Check for memory leaks
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/performance/memory-leaks | jq

# Check connection pool
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/performance/connection-pool | jq

# Full system diagnostics
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/diagnostics/system | jq '.memory'
```

### Poor Request Performance

```bash
# Check coalescing
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/performance/request-coalescing | jq

# Check database
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/database/stats | jq

# Check circuit breakers
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/performance/circuit-breakers | jq
```

### Integration Failures

```bash
# Check integration health
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/integrations/health | jq

# Check specific integration
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/integrations/health | jq '.llm'
```

### Slow Response Times

```bash
# Check event loop
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/performance/async-monitor | jq

# Check streaming
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/performance/streaming-stats | jq
```

---

## 🔄 Common Tasks

### Clear Caches

```python
from src.vega.core.request_coalescing import get_llm_coalescer

coalescer = get_llm_coalescer()
coalescer.clear_cache()
```

### Reset Metrics

```bash
# Reset database profiler
curl -X POST -H "X-API-Key: $API_KEY" http://localhost:8000/admin/database/reset-stats

# Restart monitoring systems
# (automatically resets on server restart)
```

### Force Cleanup

```python
from src.vega.core.connection_pool import get_connection_pool_manager

manager = get_connection_pool_manager()
# Cleanup happens automatically every 30s
# Manual cleanup not exposed (automatic is safer)
```

---

## 📚 Documentation Reference

### Comprehensive Guides

- **Phase 3**: `docs/PHASE3_OPTIMIZATION_SUMMARY.md`
- **Phase 4**: `docs/PHASE4_HYPEROPTIMIZATION_SUMMARY.md`
- **Complete Status**: `docs/COMPLETE_HYPEROPTIMIZATION_STATUS.md`
- **Advanced Systems**: `docs/ADVANCED_PERFORMANCE_SYSTEMS.md`

### Quick References

- **Phase 3**: `docs/PHASE3_QUICK_REFERENCE.md`
- **Phase 4**: `docs/PERFORMANCE_QUICK_REFERENCE.md`
- **This Guide**: `docs/QUICK_REFERENCE_CARD.md`

### Code Examples

- **Integration Tests**: `tests/integration/test_integrations.py`
- **Verification Script**: `tools/verify_phase3.py`

---

## 🎯 Performance Targets

### Production Benchmarks

- Request coalesce rate: **>30%**
- Connection reuse rate: **>70%**
- Cache hit rate: **>40%**
- Memory usage: **<60%**
- Response time: **<50ms p95**
- Error rate: **<1%**

### Resource Limits

- Max connections per host: **10**
- Max total connections: **100**
- Connection TTL: **5 minutes**
- Idle timeout: **1 minute**
- Request timeout: **30 seconds**

---

**Last Updated:** Phase 4 Complete
**Version:** 2.0
**Status:** Production Ready ✅
