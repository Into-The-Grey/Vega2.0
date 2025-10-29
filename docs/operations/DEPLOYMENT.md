# Deployment

[Back to Docs Hub](README.md)

This guide covers running Vega2.0 under systemd and outlines a Docker option.

## Systemd

The repo includes `systemd/vega.service`.

1. Ensure a virtualenv exists at `/home/ncacord/Vega2.0/.venv` and dependencies are installed.
2. Create a `.env` in project root with HOST, PORT, API_KEY, MODEL_NAME, LLM_BACKEND, etc.
3. Copy the unit file and enable the service:

```bash
sudo cp systemd/vega.service /etc/systemd/system/vega.service
sudo systemctl daemon-reload
sudo systemctl enable vega
sudo systemctl start vega
```

Logs:

```bash
journalctl -u vega -f
```

Security hardening in the unit restricts privileges and network families. Keep HOST=127.0.0.1.

## Reverse proxy (optional)

To expose the service remotely, place an authenticating reverse proxy in front (e.g., Caddy, Nginx) and keep API key checks enabled. Use TLS.

## Docker (outline)

No Dockerfile is included. A minimal one could be:

```Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV HOST=0.0.0.0 PORT=8000 API_KEY=change_me MODEL_NAME=mistral LLM_BACKEND=ollama
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Remember to mount a volume for `vega.db` if you want persistence.

## CI/CD

No CI is configured. Consider adding:

- Linting (ruff, black, mdformat)
- Unit tests (pytest) for critical modules
- Build + container publish

## Backups

The CLI provides a simple SQLite backup:

```bash
python -m cli db backup
```

Automate with cron or systemd timers.

