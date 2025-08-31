# Troubleshooting

[Back to Docs Hub](README.md)

Common issues and fixes.

## API returns 401 Unauthorized

- Ensure you include `X-API-Key` header
- Verify the key matches `API_KEY` (or an extra key) in `.env`

## API returns 503 or degraded mode

- Ollama may not be running. Start it and pull your model (e.g., `ollama run mistral`).
- Check `/metrics` to see `degraded: true`. The circuit breaker temporarily blocks requests after repeated failures.

## Timeouts

- Increase `LLM_TIMEOUT_SEC` in `.env`
- Reduce prompt size; ensure below `MAX_PROMPT_CHARS`

## ImportError: module not found

- Activate the virtualenv before installing: `source .venv/bin/activate`
- `pip install -r requirements.txt`
- For WHOIS lookups: `pip install python-whois`

## SQLite locked errors

- WAL mode is enabled; still, long-running writes can block reads briefly. Retry the operation.
- Ensure only one instance of the API writes to the same `vega.db`.

## SSL or HTTPS errors in OSINT

- Some sites block HEAD/GET from scripts. The utilities handle many cases but may return `connection error`.
- Try again or use different targets.

## Training OOM or slow

- Lower `max_seq_length` and batch size.
- Enable LoRA and/or 8-bit weights.
- Use a smaller base model.

## UI shows errors or no streaming

- Check browser console for network errors.
- Ensure the API is running on the same host/port and that you entered the API key.

## systemd service doesn’t start

- Check logs: `journalctl -u vega -n 200 --no-pager`
- Verify the venv path and permissions in the unit file
- Confirm `.env` is present and contains HOST/PORT

