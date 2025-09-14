# Copilot Instructions for Vega2.0

Vega2.0 is a local-first FastAPI service and CLI for chatting with a local LLM (primarily Ollama), logging conversations to SQLite, building datasets, and fine-tuning. All components are designed for localhost-only operation with strong security defaults.

## Core Architecture

**Main Components:**

- `app.py`: FastAPI service with `/chat`, `/history`, `/healthz` endpoints
- `cli.py`: Typer CLI mirroring API functionality (`python -m cli chat "Hello"`)
- `llm.py`: LLM integration layer with circuit breaker and TTL cache
- `db.py`: SQLAlchemy 2.0 + SQLite for conversation logging
- `config.py`: Environment config loader using python-dotenv

**Key Data Flow:** Client → FastAPI → LLM Backend (Ollama) → SQLite logging

## Critical Development Patterns

### Configuration Management

- All config via `.env` file loaded in `config.py`
- Use `get_config()` function throughout codebase
- Never hardcode secrets; use `.env.example` as template
- Keep `HOST=127.0.0.1` for localhost-only security

### Error Handling & Resilience

- Use `CircuitBreaker` and `TTLCache` from `resilience.py` for external calls
- LLM calls wrapped with retry logic and timeouts in `llm.py`
- All async functions should handle httpx exceptions gracefully

### Database Patterns

- SQLAlchemy 2.0 syntax with `select()` statements
- Session management via context managers
- All conversation logging goes through `db.log_conversation()`
- SQLite WAL mode enabled for concurrent access

### API Security

- X-API-Key header required for data endpoints (`/chat`, `/history`)
- Multiple API keys supported via `api_keys_extra` config
- Input validation with Pydantic models
- Rate limiting available via slowapi integration

## Essential Commands

**Development Setup:**

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Edit API_KEY and MODEL_NAME
```

**Running Services:**

```bash
# API server
uvicorn app:app --host $(grep ^HOST .env | cut -d= -f2) --port $(grep ^PORT .env | cut -d= -f2)

# CLI commands
python -m cli chat "Hello"          # Single prompt
python -m cli repl                  # Interactive REPL
python -m cli history --limit 10    # View conversation history
python -m cli dataset build ./datasets/samples  # Build training dataset
python -m cli train --config training/config.yaml  # Fine-tune model
```

**Production Deployment:**

```bash
sudo cp systemd/vega.service /etc/systemd/system/
sudo systemctl enable vega && sudo systemctl start vega
```

## ML/Training Workflow

**Dataset Pipeline:**

1. Place source files (.txt, .md, .json) in `datasets/samples/`
2. Run `python -m cli dataset build ./datasets/samples`
3. Outputs to `datasets/output.jsonl` with `{"prompt", "response"}` format

**Training Pipeline:**

1. Configure `training/config.yaml` with model/LoRA settings
2. Run `python -m cli train --config training/config.yaml`
3. Uses HuggingFace Transformers + Accelerate + optional PEFT/LoRA

## Integration Patterns

**Adding New Endpoints:**

- Follow FastAPI patterns in `app.py`
- Add corresponding CLI command in `cli.py`
- Use `@require_api_key` decorator for protected endpoints

**External Service Integration:**

- See `integrations/` modules for patterns (search, fetch, osint, slack)
- Use httpx for async HTTP calls with timeouts
- Implement graceful fallbacks for optional services

**Database Extensions:**

- Add columns to `Conversation` model in `db.py`
- Handle missing columns gracefully (see feedback fields pattern)
- Use SQLAlchemy migrations for schema changes

## File Organization

- `/integrations/`: External service connectors (web search, OSINT, Slack)
- `/datasets/`: Dataset preparation and loaders for different file types
- `/training/`: HuggingFace training harness with LoRA support
- `/learning/`: Conversation evaluation and curation tools
- `/docs/`: Comprehensive documentation (mirrored in `/book/` for mdBook)
- `/static/`: Simple HTML chat UI for browser testing

## Testing & Debugging

- Use `/healthz` endpoint to verify service status
- Check circuit breaker status via internal APIs
- SQLite database inspectable at `./vega.db`
- Logs via uvicorn/FastAPI default logging
- systemd logs: `journalctl -u vega -f`

## Security Considerations

- Always bind to localhost (`HOST=127.0.0.1`) unless explicitly needed
- API keys required for all data operations
- PII masking available via `security.py` utilities
- systemd service includes basic hardening (NoNewPrivileges, PrivateTmp)
- Never commit `.env` files or real API keys

## Advanced Development Patterns

### Async Programming Guidelines

- All LLM interactions use async/await pattern
- Use `httpx.AsyncClient` for HTTP calls, never blocking requests
- Handle `asyncio.TimeoutError` and `httpx.HTTPError` explicitly
- Implement proper connection pooling with client reuse
- Example pattern: `async with get_http_client() as client:`

### Error Handling Strategy

- Use structured logging with contextual information
- Implement graceful degradation for optional features
- Circuit breaker pattern prevents cascade failures
- TTL cache reduces load on external services
- Always log user-facing errors with correlation IDs

### Database Best Practices

- Use SQLAlchemy 2.0 syntax exclusively (`select()` instead of `query()`)
- Implement database migrations for schema changes
- Handle SQLite locking gracefully with retry logic
- Use WAL mode for better concurrency
- Implement soft deletes for conversation data

### Configuration Management Patterns

- Environment-specific configs via `.env` files
- Validate all required config on startup
- Use dataclasses for type-safe configuration
- Support config reloading without restart (where safe)
- Document all environment variables in `.env.example`

## Code Quality Standards

### Type Hints and Validation

- Use strict type hints throughout (`from __future__ import annotations`)
- Pydantic models for all API request/response bodies
- Validate configuration with type checking
- Use `Optional[]` for nullable fields explicitly
- Example: `async def chat(prompt: str, session_id: Optional[str] = None) -> str:`

### Testing Patterns

- Test async functions with `pytest-asyncio`
- Mock external services (Ollama, web search) in tests
- Use temporary databases for integration tests
- Test error conditions and edge cases
- Coverage should include resilience patterns

### Performance Considerations

- Use connection pooling for HTTP clients
- Implement response streaming for long outputs
- Cache frequently accessed data with TTL
- Monitor memory usage in long-running processes
- Profile slow endpoints with timing middleware

## Deployment & Operations

### Local Development

```bash
# Start Ollama (required dependency)
ollama serve

# Run with hot reload
uvicorn app:app --reload --host 127.0.0.1 --port 8000

# Monitor logs
tail -f vega.db-wal  # SQLite write-ahead log
```

### Production Deployment

```bash
# System preparation
sudo apt update && sudo apt install python3.12-venv nginx

# Service setup
sudo systemctl enable vega
sudo systemctl start vega
sudo systemctl status vega

# Log monitoring
journalctl -u vega -f --since "1 hour ago"
```

### Monitoring & Observability

- Health checks: `/healthz`, `/livez`, `/readyz`
- Circuit breaker status accessible via internal endpoints
- Database size monitoring: `du -h vega.db*`
- Memory usage: monitor FastAPI worker processes
- Response time tracking via middleware

## Troubleshooting Guide

### Common Issues

**Ollama Connection Failures:**

- Check if Ollama is running: `curl http://127.0.0.1:11434/api/tags`
- Verify model availability: `ollama list`
- Check circuit breaker status in logs

**Database Lock Errors:**

- Ensure WAL mode is enabled
- Check for long-running transactions
- Restart service to clear locks

**API Key Authentication:**

- Verify `X-API-Key` header format
- Check `api_keys_extra` configuration
- Test with curl: `curl -H "X-API-Key: YOUR_KEY" http://localhost:8000/healthz`

**Memory Issues:**

- Monitor conversation history size
- Implement retention policies
- Check for memory leaks in long-running processes

### Debugging Techniques

```bash
# Database inspection
sqlite3 vega.db ".schema conversations"
sqlite3 vega.db "SELECT COUNT(*) FROM conversations;"

# API testing
curl -X POST -H "Content-Type: application/json" -H "X-API-Key: YOUR_KEY" \
  -d '{"prompt":"test","stream":false}' http://localhost:8000/chat

# Log analysis
journalctl -u vega --since "1 hour ago" | grep ERROR
```

## Extension Patterns

### Adding New LLM Backends

1. Extend `llm.py` with new backend class
2. Update `LLM_BACKEND` config validation
3. Implement same async interface
4. Add backend-specific error handling
5. Update documentation and examples

### Custom Dataset Loaders

1. Create new loader in `datasets/loaders/`
2. Implement `load_*()` function returning `(prompt, response)` tuples
3. Register file extension in `prepare_dataset.py`
4. Add tests for new format
5. Update documentation

### Integration Development

```python
# Template for new integrations
async def my_integration(query: str) -> str:
    config = get_config()
    async with get_http_client() as client:
        try:
            # Implementation
            return result
        except Exception as e:
            logger.error(f"Integration failed: {e}")
            return "Integration unavailable"
```

## Data Management

### Conversation Data

- Schema: `conversations` table with `prompt`, `response`, `ts`, `session_id`
- Indexing: timestamp and session_id indexed for performance
- Retention: configurable via `retention_days` setting
- Export: use `sqlite3` command or API endpoints
- Backup: copy `vega.db*` files (includes WAL)

### Dataset Management

- Input formats: `.txt`, `.md`, `.json` files
- Output format: JSONL with `{"prompt", "response"}` structure
- Validation: check for empty prompts/responses
- Deduplication: implement hash-based deduplication
- Quality control: manual review workflows in `/learning/`

### Model Artifacts

- Training outputs: saved to `training/output/`
- Model checkpoints: HuggingFace format
- LoRA adapters: separate PEFT artifacts
- Evaluation metrics: tracked during training
- Model versioning: use git tags and artifact naming

## Performance Optimization

### Database Optimization

```sql
-- Useful queries for monitoring
SELECT COUNT(*) as total_conversations FROM conversations;
SELECT DATE(ts) as date, COUNT(*) as daily_count FROM conversations GROUP BY DATE(ts);
PRAGMA table_info(conversations);  -- Schema inspection
VACUUM;  -- Reclaim space after deletions
```

### API Performance

- Implement request/response middleware for timing
- Use async context managers for resource cleanup
- Monitor endpoint response times
- Implement rate limiting for resource-intensive operations
- Cache static responses where appropriate

### Memory Management

- Monitor conversation history growth
- Implement conversation pruning strategies
- Use streaming responses for large outputs
- Profile memory usage during training
- Clean up temporary files and artifacts

## Future Architecture Considerations

### Scalability Patterns

- Consider PostgreSQL migration for multi-user scenarios
- Implement horizontal scaling with load balancing
- Design for containerization (Docker/Kubernetes)
- Plan for distributed training scenarios
- Consider message queue integration for async processing

### Security Enhancements

- Implement proper JWT token authentication
- Add audit logging for all operations
- Consider encryption at rest for sensitive data
- Implement rate limiting per user/API key
- Plan for secure multi-tenant deployments

### Feature Development

- Implement conversation branching and threading
- Add multi-modal support (images, documents)
- Design plugin architecture for extensions
- Plan for federated learning scenarios
- Consider real-time collaboration features
