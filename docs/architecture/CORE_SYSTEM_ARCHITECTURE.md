# Vega2.0 Core System Architecture

## System Heart (Critical Core)

These components are the **absolute foundation** - if any fail, the system cannot operate.

### Tier 0: Absolute Core (Must Always Work)

| Component | File | Purpose | Failure = System Dead |
|-----------|------|---------|----------------------|
| **Configuration** | `config.py` | Load environment, API keys, settings | ✅ Yes |
| **Logging** | `logging_setup.py` | All diagnostic output | ✅ Yes |
| **Database** | `db.py` | SQLite connection, conversation storage | ✅ Yes |
| **LLM Backend** | `llm.py` | Ollama/OpenAI communication | ✅ Yes |
| **FastAPI App** | `app.py` | HTTP server, endpoints | ✅ Yes |
| **Resilience** | `resilience.py` | Circuit breaker, caching | ✅ Yes |

### Tier 1: High Priority (System Degraded Without)

| Component | File | Purpose | Can Recover |
|-----------|------|---------|-------------|
| **Error Handler** | `error_handler.py` | Structured error handling | ✅ Yes |
| **Recovery Manager** | `recovery_manager.py` | Auto-recovery strategies | ✅ Yes |
| **Memory Manager** | `memory_manager.py` | Persistent memory operations | ✅ Yes |
| **Config Validator** | `config_validator.py` | Validate config at startup | ✅ Yes |

### Tier 2: Standard Priority (Nice to Have)

| Component | File | Purpose |
|-----------|------|---------|
| **Resource Manager** | `resource_manager.py` | HTTP pools, connection management |
| **DB Profiler** | `db_profiler.py` | Query performance monitoring |
| **Correlation** | `correlation.py` | Distributed tracing |
| **Metrics Aggregator** | `metrics_aggregator.py` | System metrics |

### Tier 3: Low Priority (Background Features)

| Component | File | Purpose |
|-----------|------|---------|
| **Process Manager** | `process_manager.py` | Background process handling |
| **Memory Leak Detector** | `memory_leak_detector.py` | Memory monitoring |
| **Batch Operations** | `batch_operations.py` | Bulk operations |
| **ECC Crypto** | `ecc_crypto.py` | Advanced encryption |

### Tier 4: Optional (Enhancements)

| Component | File | Purpose |
|-----------|------|---------|
| **API Security** | `api_security.py` | Enhanced API protection |
| **Streaming Backpressure** | `streaming_backpressure.py` | Flow control |
| **Request Coalescing** | `request_coalescing.py` | Duplicate request handling |
| **Adaptive Rate Limit** | `adaptive_rate_limit.py` | Dynamic rate limiting |

---

## Hardening Principles

### 1. Fail-Safe Defaults

Every component must have sensible defaults that allow operation without configuration.

### 2. Graceful Degradation

Non-critical failures should reduce functionality, not crash the system.

### 3. Self-Healing

The system should automatically attempt to recover from failures.

### 4. Learning from Failures

Successful recovery strategies should be remembered for future use.

### 5. Visibility

All issues, even recovered ones, should be logged and serious ones reported.

### 6. Continuous Improvement

The system should suggest improvements based on observed patterns.

---

## Core Hardening Requirements

### Every Core Function Must

1. ✅ Have try/except wrapping all operations
2. ✅ Log all errors with full context
3. ✅ Have timeout protection
4. ✅ Have fallback behavior
5. ✅ Be idempotent (safe to retry)
6. ✅ Handle None/empty inputs gracefully
7. ✅ Validate all inputs
8. ✅ Never crash on unexpected types
9. ✅ Report recovery success/failure
10. ✅ Suggest improvements when patterns detected

---

## Data Flow (Simplified)

```text
User Request
    │
    ▼
┌─────────────────┐
│   FastAPI App   │ ─── Authentication ─── API Key Validation
│    (app.py)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Input Layer   │ ─── Sanitization ─── Validation ─── Rate Limiting
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLM Backend   │ ─── Circuit Breaker ─── Retry ─── Timeout
│    (llm.py)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Database     │ ─── Connection Pool ─── Transaction ─── Logging
│     (db.py)     │
└────────┬────────┘
         │
         ▼
User Response
```

---

## Recovery Strategies

### Level 1: Immediate Retry

- Wait 100ms, retry once
- For transient failures (network blip, timeout)

### Level 2: Backoff Retry

- Exponential backoff: 1s, 2s, 4s, 8s, 16s
- For resource contention issues

### Level 3: Fallback Mode

- Use cached data or defaults
- For extended outages

### Level 4: Degraded Operation

- Disable the failing feature
- Continue with reduced functionality

### Level 5: Manual Intervention

- Log detailed diagnostic info
- Alert the operator
- Suggest fixes

---

## Improvement Suggestions Engine

The system should track:

1. **Error Frequency**: Which errors occur most often?
2. **Recovery Success**: Which strategies work best?
3. **Performance Patterns**: What's slowing down?
4. **Resource Usage**: Where are bottlenecks?

And suggest:

1. Code changes to prevent common errors
2. Configuration tweaks for better performance
3. Infrastructure changes for scalability
4. New features to handle edge cases

