# Vega2.0 — Commands Only Cheat Sheet

## Setup

```bash
# Clone (if not already)
git clone https://github.com/Into-The-Grey/Vega2.0.git
cd Vega2.0

# Python venv
python3.12 -m venv .venv
source .venv/bin/activate

# Install deps
pip install --upgrade pip
pip install -r requirements.txt

# Environment
cp .env.example .env
# edit .env as needed
```

## Run API / Server

```bash
# Run FastAPI (module)
python -m core.app

# OR run with uvicorn directly
uvicorn core.app:app --host 127.0.0.1 --port 8000 --reload

# Health checks
curl -s http://127.0.0.1:8000/healthz | jq .
curl -s http://127.0.0.1:8000/metrics | jq .
```

## CLI — Core

```bash
# Single prompt
python -m core.cli chat "Hello Vega"

# Interactive REPL
python -m core.cli repl

# History (limit N)
python -m core.cli history --limit 20

# Train
python -m core.cli train --config training/config.yaml

# Feedback on a conversation row
python -m core.cli feedback 123 --rating 5 --tags "good,short" --notes "✅" --reviewed true
```

## CLI — Search

```bash
# Web search
python -m core.cli search web "rust async io" --max-results 5 --safesearch moderate

# Image search
python -m core.cli search images "aurora borealis" --max-results 5 --safesearch moderate

# Research: summarize top results with the LLM
python -m core.cli search research "mistral vs llama"
```

## CLI — Integrations

```bash
# Slack webhook test (if configured)
python -m core.cli integrations test
```

## CLI — Dataset

```bash
# Build dataset from directory
python -m core.cli dataset build ./datasets/samples
```

## CLI — Learning / Evaluation

```bash
# Curate dataset from feedback
python -m core.cli learn curate --min-rating 4 --reviewed-only false --out-path datasets/curated.jsonl

# Evaluate model on a JSONL file
python -m core.cli learn evaluate datasets/curated.test.jsonl

# Optimize system prompt from candidates
python -m core.cli learn optimize-prompt prompts/candidates.txt datasets/curated.test.jsonl --out-dir prompts
```

## CLI — Database

```bash
# Backup DB
python -m core.cli db backup --out vega.db.backup

# VACUUM
python -m core.cli db vacuum

# Export to JSONL
python -m core.cli db export conversations.jsonl --limit 1000

# Import from JSONL
python -m core.cli db import conversations.jsonl

# Purge rows older than N days
python -m core.cli db purge 30

# Fulltext search in DB
python -m core.cli db search "error OR timeout" --limit 20
```

## CLI — Generation Settings

```bash
# Show
python -m core.cli gen show

# Set (any subset)
python -m core.cli gen set --temperature 0.7 --top-p 0.9 --top-k 40 --repeat-penalty 1.1 --presence-penalty 0.0 --frequency-penalty 0.0

# Enable/disable dynamic generation
python -m core.cli gen dynamic true
python -m core.cli gen dynamic false

# Reset overrides
python -m core.cli gen reset
```

## CLI — Memory

```bash
# Store knowledge item
python -m core.cli memory store "ops" "How to restart the service" --metadata '{"priority":"high"}' --tags "ops,restart"

# Get item by key + topic
python -m core.cli memory get 1a2b3c4d ops

# Search
python -m core.cli memory search "restart service" --topic ops --limit 10

# Stats
python -m core.cli memory stats
```

## CLI — OSINT & Networking

```bash
# DNS / rDNS
python -m core.cli osint dns example.com
python -m core.cli osint rdns 1.1.1.1

# HTTP headers
python -m core.cli osint headers https://example.com

# SSL cert info
python -m core.cli osint ssl example.com --port 443

# robots.txt
python -m core.cli osint robots https://example.com

# WHOIS
python -m core.cli osint whois example.com

# Username search
python -m core.cli osint username johndoe --include-nsfw false --sites "github,reddit"

# TCP port scan (comma and/or ranges)
python -m core.cli net scan 127.0.0.1 22,80,8000-8010
```

## CLI — Autonomous Features

```bash
# Backups
python -m core.cli backup create --tag manual
python -m core.cli backup list
python -m core.cli backup restore backups/vega_2025-09-16_manual.tar.gz --restore-dir ./restore
python -m core.cli backup prune --keep 5

# Voice profile
python -m core.cli voice add-sample /path/to/audio.wav
python -m core.cli voice update-profile
python -m core.cli voice status

# Knowledge base
python -m core.cli kb add-site Development https://github.com
python -m core.cli kb list --category Development
python -m core.cli kb list

# Finance
python -m core.cli finance invest AAPL 10 150.00
python -m core.cli finance portfolio
python -m core.cli finance price AAPL
```

## API — Environment

```bash
# Base URL and API key
export BASE_URL="http://127.0.0.1:8000"
export KEY="vega-default-key"  # or: export KEY="$(grep ^API_KEY .env | cut -d= -f2)"
```

## API — Health & Metrics

```bash
curl -s "$BASE_URL/healthz" | jq .
curl -s "$BASE_URL/metrics" | jq .
```

## API — Chat

```bash
curl -s \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $KEY" \
  -d '{"prompt":"Hello Vega","stream":false}' \
  "$BASE_URL/chat" | jq .
```

## API — Admin

```bash
# Logs modules
curl -s -H "X-API-Key: $KEY" "$BASE_URL/admin/logs" | jq .

# Logs for module (with lines)
curl -s -H "X-API-Key: $KEY" "$BASE_URL/admin/logs/app?lines=50" | jq .

# Config modules
curl -s -H "X-API-Key: $KEY" "$BASE_URL/admin/config" | jq .

# Config for module
curl -s -H "X-API-Key: $KEY" "$BASE_URL/admin/config/app" | jq .

# LLM behavior
curl -s -H "X-API-Key: $KEY" "$BASE_URL/admin/llm/behavior" | jq .
```

## API — Backups

```bash
# Create backup
curl -s -X POST -H "X-API-Key: $KEY" "$BASE_URL/backup/create?tag=api" | jq .

# List backups
curl -s -H "X-API-Key: $KEY" "$BASE_URL/backup/list" | jq .

# Restore backup (query params)
curl -s -X POST -H "X-API-Key: $KEY" \
  "$BASE_URL/backup/restore?backup_file=/path/to/backup.tar.gz&restore_dir=/tmp/restore" | jq .
```

## API — Voice Profile

```bash
# Add sample (path on server)
curl -s -X POST -H "X-API-Key: $KEY" \
  "$BASE_URL/voice/samples?file_path=/path/to/audio.wav" | jq .

# Update profile
curl -s -X POST -H "X-API-Key: $KEY" "$BASE_URL/voice/profile/update" | jq .

# Get profile
curl -s -H "X-API-Key: $KEY" "$BASE_URL/voice/profile" | jq .
```

## API — Knowledge Base

```bash
# Add site
curl -s -X POST -H "X-API-Key: $KEY" \
  "$BASE_URL/kb/sites?category=Development&url=https://github.com" | jq .

# List sites (optional category)
curl -s -H "X-API-Key: $KEY" "$BASE_URL/kb/sites" | jq .
curl -s -H "X-API-Key: $KEY" "$BASE_URL/kb/sites?category=Development" | jq .
```

## API — Finance

```bash
# Add investment
curl -s -X POST -H "X-API-Key: $KEY" \
  "$BASE_URL/finance/invest?symbol=AAPL&shares=10&price=150" | jq .

# Portfolio
curl -s -H "X-API-Key: $KEY" "$BASE_URL/finance/portfolio" | jq .

# Price
curl -s -H "X-API-Key: $KEY" "$BASE_URL/finance/price/AAPL" | jq .
```

## Systemd

```bash
# Install unit
sudo cp systemd/vega.service /etc/systemd/system/vega.service
sudo systemctl daemon-reload

# Enable + start
sudo systemctl enable vega
sudo systemctl start vega

# Status & logs
systemctl status vega --no-pager
journalctl -u vega -f

# Restart / stop
sudo systemctl restart vega
sudo systemctl stop vega
```

## Database (SQLite) — Optional Ops

```bash
# Schema
sqlite3 vega.db ".schema"

# Count rows
sqlite3 vega.db "SELECT COUNT(*) FROM conversations;"

# Vacuum
sqlite3 vega.db "VACUUM;"
```

## SAC (System Autonomy Core) — Runners

```bash
# Core entry
python -m main

# Individual SAC modules (if needed)
python -m sac.system_probe
python -m sac.system_watchdog
python -m sac.sys_control
python -m sac.net_guard
python -m sac.economic_scanner
python -m sac.system_interface
python -m sac.self_govern
```
