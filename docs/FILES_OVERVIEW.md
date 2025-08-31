# Files Overview

[Back to Docs Hub](README.md)

This document explains every file in the repository, its purpose, and key functions or entities.

## Root

- `app.py`: FastAPI application exposing chat, history, research, OSINT, and admin endpoints. Wires metrics, rate limiting (optional), static files, and startup/shutdown hooks.
- `cli.py`: Typer + Rich CLI mirroring API capabilities (chat, REPL, history, dataset build, training, integrations, db utilities, OSINT, net scan, generation settings).
- `config.py`: Environment-driven configuration loader using python-dotenv. Provides `Config` dataclass and `get_config()`.
- `db.py`: SQLAlchemy 2.0 models and helpers for SQLite (`vega.db`). Includes `log_conversation`, `get_history`, `get_history_page`, `get_session_history`, `set_feedback`, `search_conversations`, backup/vacuum/export/import utilities.
- `llm.py`: LLM integration with Ollama via httpx. Provides `query_llm`, `query_ollama`, resilience (CircuitBreaker, TTLCache), generation settings management, and warmup/shutdown hooks.
- `resilience.py`: Utilities implementing `CircuitBreaker` and `TTLCache`.
- `security.py`: `mask_pii` to redact emails, phones, secret-like tokens.
- `README.md`: Project-level quickstart and top-level notes.
- `requirements.txt`: Pinned package versions.
- `app.env`: Example environment variables (HOST, PORT, API_KEY). For systemd EnvironmentFile usage.
- `app.py.bak`: Temporary/backup note file; safe to ignore.
- `vega.db`, `vega.db-shm`, `vega.db-wal`: SQLite database and WAL files.
- `__init__.py`: Package marker.

## static/

- `static/index.html`: Minimal chat UI and panels for search, research, username search, and generation settings.

## systemd/

- `systemd/vega.service`: Example unit to run uvicorn with environment variables and hardening settings.

## datasets/

- `datasets/__init__.py`: Declares local package namespace to avoid conflict with PyPI `datasets`.
- `datasets/prepare_dataset.py`: Builds `datasets/output.jsonl` by traversing input directory and using loaders.
- `datasets/output.jsonl`: Output JSONL produced by the builder.
- `datasets/curated.test.jsonl`: Empty placeholder; can store curated evals/tests.
- `datasets/samples/example.txt`: Text sample file.
- `datasets/samples/example.md`: Markdown sample file.
- `datasets/samples/example.json`: JSON sample file with pairs.

### datasets/loaders/

- `loader_txt.py`: Emits `(prompt, "")` per non-empty line in text files.
- `loader_md.py`: Uses headings as prompts and following text as response.
- `loader_json.py`: Parses list or `{ data: [...] }` of `{prompt,response}` into pairs.

## integrations/

- `integrations/search.py`: DuckDuckGo search wrappers (`web_search`, `image_search`) with graceful fallback if the package is missing or network blocked.
- `integrations/fetch.py`: Async `fetch_text(url)` using httpx + BeautifulSoup(lxml) to produce readable text.
- `integrations/osint.py`: Local-only OSINT/network helpers: `dns_lookup`, `reverse_dns`, `http_headers`, `ssl_cert_info`, `robots_txt`, `whois_lookup`, `tcp_scan`, `username_search`.
- `integrations/slack_connector.py`: `send_slack_message(webhook_url, text) -> bool` for Slack Incoming Webhooks.

## learning/

- `learning/learn.py`: Curation and evaluation helpers: `curate_dataset`, `evaluate_model`, `optimize_system_prompt`. Writes `prompts/system_prompt.txt` for runtime use.
- `learning/__init__.py`: Package marker.

## training/

- `training/train.py`: Minimal fine-tuning harness for causal LM using Hugging Face `Trainer`; optional LoRA via `peft`.
- `training/config.yaml`: Training configuration parameters (model name, batch size, LoRA knobs, dataset path).
- `training/__init__.py`: Package marker.

## __pycache__/

- Python bytecode caches; can be deleted safely.

## Notes

- Hidden folder `.git/` and related files are part of version control and are not covered here.
- No Dockerfile is present; see Deployment doc for systemd usage and a suggested Dockerfile outline.

