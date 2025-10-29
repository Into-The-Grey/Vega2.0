# Usage Guide

[Back to Docs Hub](README.md)

This guide shows how to interact with Vega2.0 via the HTTP API, CLI, and the minimal web UI.

## API

All non-health endpoints require the `X-API-Key` header. Obtain the key from your `.env`.

### Health and readiness

- GET `/healthz` -> `{ "ok": true }`
- GET `/livez` -> liveness check
- GET `/readyz` -> readiness plus degraded indicator
- GET `/metrics` -> internal counters and status

### Chat

- POST `/chat`
  - Headers: `X-API-Key: <key>`, `Content-Type: application/json`
  - Body: `{ "prompt": "Hello", "stream": false, "session_id": "optional" }`
  - Response: `{ "response": "...", "conversation_id": 123, "session_id": "abcd" }`

Example:

```bash
curl -s -H "Content-Type: application/json" \
  -H "X-API-Key: $KEY" \
  -d '{"prompt":"Hello Vega","stream":false}' \
  http://127.0.0.1:8000/chat | jq .
```

- POST `/chat/sse`
  - Same body and headers, content type `text/event-stream`
  - Use a browser EventSource or fetch reader to consume frames

### History and sessions

- GET `/history?limit=50&cursor=<id>` -> paginated recent logs
- GET `/session/{session_id}` -> last messages for a session

### Feedback

- POST `/feedback`
  - Body: `{ conversation_id, rating?, tags?, notes?, reviewed? }`

### Search and research

- POST `/search/web` -> `{ items: [{ title, href, snippet, source }] }`
- POST `/search/images` -> `{ items: [{ title, image, thumbnail, url, width, height }] }`
- POST `/research/summarize` -> `{ summary, items }`
- POST `/research/rag` -> `{ summary, sources }`

### OSINT and Networking

- POST `/osint/dns` -> `{ hostname, addresses }`
- POST `/osint/reverse_dns` -> `{ names }`
- POST `/osint/http_headers` -> `{ url, status, headers }`
- POST `/osint/ssl_cert` -> `{ host, subject, issuer, not_before, not_after }`
- POST `/osint/robots` -> `{ robots }`
- POST `/osint/whois` -> dictionary (or `{ error }`)
- POST `/osint/username` -> `{ items: [{ site, url, exists, status, note, nsfw }] }`
- POST `/net/scan` -> `{ results: [{ port, state }] }`

### Admin: generation settings

- GET `/admin/gen` -> current generation settings
- POST `/admin/gen` -> update any field (temperature, top_p, top_k, etc.)
- POST `/admin/gen/reset` -> clear overrides

## CLI

Run `python -m cli --help` for a full list.

Common commands:

```bash
# Single prompt
python -m cli chat "Hello Vega"

# REPL
python -m cli repl

# History
python -m cli history --limit 20

# Web search
python -m cli search web "rust async io"

# Image search
python -m cli search images "aurora borealis"

# Research summary
python -m cli search research "mistral vs llama"

# Slack integration test
python -m cli integrations test

# DB maintenance
python -m cli db backup
python -m cli db vacuum
python -m cli db purge 30

# Dataset build
python -m cli dataset build ./datasets/samples

# Training
python -m cli train --config training/config.yaml
```

## Static Web UI

Open the chat UI at: [http://127.0.0.1:8000/static/index.html](http://127.0.0.1:8000/static/index.html)

Steps:

- Enter your API key (X-API-Key)
- Toggle "Stream" for server-sent events mode
- Send prompts and view responses
- Try Search, Research, and Username Search panels

## Notes and Limits

- Prompts longer than `MAX_PROMPT_CHARS` are rejected with 413.
- Responses are truncated at `MAX_RESPONSE_CHARS`.
- If SlowAPI is available, rate limits apply to select endpoints.
- OSINT/network features are local-only utilities; avoid scanning beyond your own hosts.

