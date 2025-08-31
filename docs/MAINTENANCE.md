# Maintenance

[Back to Docs Hub](README.md)

Guidelines for keeping Vega2.0 healthy over time.

## Logs and Monitoring

- Follow logs with `journalctl -u vega -f` when running under systemd
- Periodically check `/metrics` for degraded status and error counters

## Database care

- Backup: `python -m cli db backup`
- Vacuum: `python -m cli db vacuum`
- Purge old rows: `python -m cli db purge 30` (delete older than 30 days)
- Export for analysis: `python -m cli db export conversations.jsonl --limit 10000`

## Security updates

- Keep dependencies up to date within pinned ranges
- Rotate API keys periodically; update `.env`
- Keep HOST=127.0.0.1 unless placing a secure reverse proxy in front

## Scaling and performance

- Increase `CACHE_TTL_SECONDS` for repeated prompts
- Use a faster model or local GPU for generation
- Move SQLite to a faster disk; consider Postgres if you need multi-writer concurrency

## Backups and disaster recovery

- Automate database backup via cron or systemd timers
- Test restoring from backups periodically

## Extending

- Add new integrations under `integrations/` and wire to API/CLI
- Implement Hugging Face backend in `llm.py` when ready
- Add tests for critical modules and use CI to run them

