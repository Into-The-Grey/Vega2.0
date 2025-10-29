# Dependencies

[Back to Docs Hub](README.md)

This document lists runtime and optional dependencies from `requirements.txt` and explains their roles.

## Python

- Python 3.12

## Runtime libraries

- fastapi==0.115.0: Web framework for the API service
- uvicorn[standard]==0.30.6: ASGI server to run FastAPI
- python-dotenv==1.0.1: Loads `.env` files into environment
- requests==2.32.3: Simple HTTP client (used by Slack webhook)
- httpx==0.27.2: Async HTTP client for LLM and fetch integrations
- SQLAlchemy==2.0.35: ORM/DDL for SQLite persistence
- rich==13.8.1: Pretty printing in CLI (tables, colors)
- typer==0.12.5: CLI framework
- click==8.1.7: Typer dependency

## ML and datasets

- transformers==4.44.2: Model/tokenizer utilities for training
- accelerate==0.34.2: Training acceleration utilities
- datasets==3.0.1: Included for convenience; we primarily parse JSONL manually to avoid namespace conflicts with local `datasets/`
- peft==0.13.2: Optional LoRA adapters
- bitsandbytes==0.43.1: Optional 8-bit quantization (requires compatible GPU/CUDA)

## Search and parsing

- duckduckgo-search==6.2.10: Web and image search (optional at runtime)
- beautifulsoup4==4.12.3: HTML parsing
- lxml==5.3.0: Parser backend for BeautifulSoup

## Rate limiting (optional)

- slowapi==0.1.9: If installed, enables rate limiting in API

## Optional OSINT helper

- python-whois: Not pinned in requirements; install to enable WHOIS lookups

## System packages (suggested)

- build-essential, libssl-dev, libffi-dev (compilers/libs for some wheels)
- For lxml: libxml2-dev libxslt1-dev (or use prebuilt wheels)

## Notes on compatibility

- Pinned versions reflect a Python 3.12 environment tested as of 2025-08.
- If a dependency fails to install, try `pip install --upgrade pip wheel setuptools` first.
- For GPU training, ensure CUDA toolkit and driver versions match the wheel availability.

