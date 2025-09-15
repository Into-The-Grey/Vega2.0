# Vega2.0 — Local-only Personal LLM Twin

[![Docs](https://img.shields.io/badge/docs-mdBook-blue)](https://into-the-grey.github.io/Vega2.0/) [![Pages](https://img.shields.io/badge/GitHub%20Pages-live-brightgreen)](https://into-the-grey.github.io/Vega2.0/) [![CI](https://github.com/Into-The-Grey/Vega2.0/actions/workflows/mdbook.yml/badge.svg)](https://github.com/Into-The-Grey/Vega2.0/actions/workflows/mdbook.yml)

# Vega2.0 - Autonomous AI System

Vega2.0 is a comprehensive autonomous AI system featuring a complete System Autonomy Core (SAC) with 7 integrated phases for full system self-management. The system combines local-first FastAPI services, CLI interfaces, autonomous decision-making, and complete hardware control capabilities.

## 🤖 System Autonomy Core (SAC)

The SAC provides complete autonomous system management through 7 integrated phases:

1. **🔍 System Introspection** - Hardware enumeration and health monitoring
2. **👁️ Active Monitoring** - Real-time system watchdog with automated responses  
3. **⚙️ System Control** - Secure command execution with audit trails
4. **🛡️ Network Security** - Automated firewall and threat detection
5. **💰 Economic Intelligence** - Market analysis and upgrade recommendations
6. **🎛️ System Interface** - Unified API with web dashboard and real-time monitoring
7. **🤖 Self-Governing Operations** - ML-driven autonomous decision engine

## 📁 Project Structure

```
Vega2.0/
├── 🧠 core/              # Core application components
│   ├── app.py           # Main FastAPI application  
│   ├── cli.py           # Command-line interface
│   ├── config.py        # Configuration management
│   ├── db.py            # Database operations
│   ├── llm.py           # LLM integration layer
│   ├── memory.py        # Conversation memory
│   ├── resilience.py    # Circuit breakers and caching
│   └── security.py      # Security utilities
│
├── 🤖 sac/              # System Autonomy Core
│   ├── system_probe.py      # Phase 1: Hardware introspection
│   ├── system_watchdog.py   # Phase 2: Active monitoring
│   ├── sys_control.py       # Phase 3: System control
│   ├── net_guard.py         # Phase 4: Network security
│   ├── economic_scanner.py  # Phase 5: Economic intelligence
│   ├── system_interface.py  # Phase 6: Unified interface
│   ├── self_govern.py       # Phase 7: Autonomous operations
│   ├── config/              # SAC configuration files
│   ├── data/                # SAC databases and state
│   ├── logs/                # SAC operation logs
│   └── models/              # ML models for decision making
│
├── 🧠 intelligence/     # AI intelligence engines
│   ├── autonomous_analyzer.py
│   ├── evaluation_engine.py
│   ├── performance_engine.py
│   ├── global_self_improvement.py
│   ├── knowledge_harvesting.py
│   ├── skill_versioning.py
│   └── telemetry_system.py
│
├── 📊 analysis/         # Analysis and conversation tools
│   └── conversation_integration.py
│
├── 🎛️ ui/               # User interface components
│   ├── dashboard.py     # System dashboard
│   └── static/          # Web assets
│
├── 💾 data/             # Data storage
│   ├── *.db             # SQLite databases
│   ├── *.json           # Configuration and data files
│   └── app.env          # Environment configuration
│
├── 🔗 integrations/     # External service integrations
├── 📚 datasets/         # Dataset preparation and training data
├── 🎓 training/         # Model training and fine-tuning
├── 📖 learning/         # Learning and evaluation systems
├── 📝 docs/             # Documentation and guides
└── 📖 book/             # mdBook documentation
```

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

## Documentation

- mdBook site: [into-the-grey.github.io/Vega2.0](https://into-the-grey.github.io/Vega2.0/)
- Source docs: see `docs/` and the mdBook in `book/`.

If the site doesn't load yet:

- In GitHub, go to Settings → Pages.
- Set Source to "Deploy from a branch".
- Choose Branch: `gh-pages` and Folder: `/ (root)`.
- Save; wait ~1–2 minutes for publish.

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
