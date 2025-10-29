# Setup Guide

[Back to Docs Hub](README.md)

This guide walks you through installing Vega2.0 on Linux with Python 3.12.

## Prerequisites

- Linux (tested on Ubuntu 24.04)
- Python 3.12
- curl, git
- Optional: Ollama running on 127.0.0.1:11434 for LLM generation.

## 1. Clone and prepare

```bash
cd /home/ncacord
# If not already cloned
# git clone https://github.com/Into-The-Grey/Vega2.0.git
cd Vega2.0
```

## 2. Create virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you plan to use whois-based lookups:

```bash
pip install python-whois
```

If you have a compatible GPU and want 8-bit optimizations:

```bash
pip install bitsandbytes
```

## 4. Configure environment

Create a `.env` file at the project root with these keys:

```dotenv
API_KEY=change_me_strong_key
HOST=127.0.0.1
PORT=8000
MODEL_NAME=mistral
LLM_BACKEND=ollama
# Optional
SLACK_WEBHOOK_URL=
LLM_TIMEOUT_SEC=60
LLM_MAX_RETRIES=2
LLM_RETRY_BACKOFF=0.5
BREAKER_FAIL_THRESHOLD=5
BREAKER_RESET_SECONDS=30
CACHE_TTL_SECONDS=60
MAX_RESPONSE_CHARS=4000
MAX_PROMPT_CHARS=4000
RETENTION_DAYS=0
PII_MASKING=false
LOG_LEVEL=INFO
```

Note: `config.py` loads `.env` automatically if present.

## 5. Initialize database (automatic)

On first run, SQLite `vega.db` is created automatically with necessary tables.

## 6. Start the API

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Visit [http://127.0.0.1:8000/](http://127.0.0.1:8000/) and [http://127.0.0.1:8000/static/index.html](http://127.0.0.1:8000/static/index.html)

## 7. Try the CLI

```bash
python -m cli chat "Hello Vega"
python -m cli repl
python -m cli history --limit 10
```

## 8. Optional: systemd service

See [DEPLOYMENT.md](DEPLOYMENT.md) for setting up `systemd/vega.service`.

## 9. Optional: training prerequisites

Ensure you have sufficient disk and RAM. For Hugging Face training, large models may require GPU.

- Install `transformers`, `accelerate` (already in requirements.txt)
- For LoRA: `peft` (in requirements.txt)
- Datasets are read from JSONL; not using the Hugging Face datasets runtime by default to avoid package name clash with local `datasets/` folder.

## 10. Environment validation

- Test health: `curl -s http://127.0.0.1:8000/healthz`
- Test chat (replace KEY):

```bash
curl -s -H "Content-Type: application/json" \
  -H "X-API-Key: KEY" \
  -d '{"prompt":"Hello Vega","stream":false}' \
  http://127.0.0.1:8000/chat | jq .
```

## 11. Next steps

- Read [USAGE.md](USAGE.md) for endpoints and CLI.
- Read [INTEGRATIONS.md](INTEGRATIONS.md) for web search and OSINT helpers.
- Read [TRAINING.md](TRAINING.md) to fine-tune a model.
