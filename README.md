# Vega2.0 — Local-only Personal LLM Twin

Vega2.0 is a local-first FastAPI service and CLI for chatting with a local LLM, logging conversations to SQLite, building datasets, and fine-tuning. It integrates with Ollama by default and can be extended with Slack/Discord.

## Features

- FastAPI service with `/chat`, `/history`, `/healthz`
- Typer + Rich CLI with chat, REPL, history, dataset, train, integrations
- SQLite logging of conversations
- Dataset builder from txt/md/json -> JSONL
- Training harness using Hugging Face + Accelerate + optional LoRA/PEFT
- Minimal static chat UI
- systemd unit for service management

## Requirements

- Ubuntu 24.04 LTS
- Python 3.12
- Optional: Ollama running locally (<http://127.0.0.1:11434>)

## Setup

Create virtual environment (Python 3.12) and activate

```bash
cd /home/ncacord/Vega2.0
python3.12 -m venv .venv
source .venv/bin/activate
```

Install dependencies (pinned)

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Configure environment

```bash
cp .env.example .env
# Edit .env to set API_KEY, HOST, PORT, MODEL_NAME, LLM_BACKEND, SLACK_WEBHOOK_URL
```

- Keep HOST=127.0.0.1 for localhost-only
- Use a strong API_KEY
- MODEL_NAME should match your Ollama model name (e.g., "mistral")

Run the API locally

```bash
uvicorn app:app --host $(grep ^HOST .env | cut -d= -f2) --port $(grep ^PORT .env | cut -d= -f2)
```

Or with the config’s values via systemd unit (below).

Try the CLI

```bash
# Single shot
python -m cli chat "Hello Vega"

# REPL
python -m cli repl

# History
python -m cli history-cmd --limit 10

# Dataset build
python -m cli dataset build ./datasets/samples

# Training
python -m cli train --config training/config.yaml
```

Optional: Serve the static UI

- Open <http://127.0.0.1:8000/static/index.html> after you start the API.

## API Usage

- GET /healthz -> {"ok": true}
- POST /chat (requires X-API-Key)
  - Body: {"prompt": "...", "stream": false}
  - Returns: {"response": "..."}
- GET /history?limit=N (requires X-API-Key)

Example curl:

```bash
curl -s \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $(grep ^API_KEY .env | cut -d= -f2)" \
  -d '{"prompt":"Hello Vega","stream":false}' \
  http://127.0.0.1:8000/chat | jq .
```

## Systemd Service

Edit `.env` and ensure HOST and PORT are set.
Copy the unit file (as root):

```bash
sudo cp systemd/vega.service /etc/systemd/system/vega.service
sudo systemctl daemon-reload
sudo systemctl enable vega
sudo systemctl start vega
```

- Logs: `journalctl -u vega -f`
- Stop: `sudo systemctl stop vega`
- Status: `systemctl status vega`

Note: The unit assumes the venv at `/home/ncacord/Vega2.0/.venv` and uses `${HOST}` and `${PORT}` from environment. If systemd doesn't inherit .env automatically, you can create a drop-in or edit ExecStart to hardcode --host/--port.

## Security & Hardening

- Keep HOST=127.0.0.1; do not expose remotely.
- All non-health endpoints require `X-API-Key`.
- Optional `slowapi` for rate-limiting is included; if installed, it's used automatically.
- Consider `logrotate` for uvicorn/systemd journal (see below).

### Logrotate stub

This project uses systemd journal logs by default. If you run uvicorn directly and log to a file, add a logrotate rule:

```bash
/var/log/vega/*.log {
  daily
  rotate 14
  compress
  missingok
  notifempty
  create 0640 ncacord ncacord
}
```

## Extending Integrations

See `integrations/slack_connector.py`. Add more connectors (Discord, email) following the same interface shape (a `send_*` function returning bool). Update `cli.py integrations` to call them.

## Development Notes

- Code is type-annotated and documented inline.
- SQLite WAL mode enabled for better read concurrency.
- Training is a minimal example; adapt tokenization/formatting for your use case.

## Deactivate venv

```bash
deactivate
```

## Troubleshooting

- Missing packages? Ensure you activated the venv before installing.
- Ollama not running? Start it locally and pull your model, e.g., `ollama run mistral`.
- GPU training? Install correct CUDA wheels for transformers/bitsandbytes or disable bnb.

## Sanity Checklist

- Folder tree matches spec ✓
- Dependencies pinned in `requirements.txt` ✓
- `.env.example` covers API_KEY, HOST, PORT, MODEL_NAME, LLM_BACKEND, SLACK_WEBHOOK_URL ✓
- API endpoints and CLI command parity ✓
- Training pipeline expects `datasets/output.jsonl` ✓
- `systemd/vega.service` compatible with `systemctl` ✓
