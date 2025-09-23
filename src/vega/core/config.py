"""
config.py - Central configuration loader for Vega2.0

- Loads environment variables from a local .env file using python-dotenv
- Provides a simple Config object with typed attributes
- Fails fast if required values are missing
- Notes:
  * Do not commit your real .env to source control
  * All modules import get_config() to read settings
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv


# Load .env from the project root if present
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)


@dataclass(frozen=True)
class Config:
    api_key: str
    host: str
    port: int
    model_name: str
    llm_backend: str  # "ollama" or "hf"
    slack_webhook_url: str | None = None
    llm_timeout_sec: float = 60.0
    llm_max_retries: int = 2
    llm_retry_backoff: float = 0.5
    breaker_fail_threshold: int = 5
    breaker_reset_seconds: float = 30.0
    cache_ttl_seconds: float = 60.0
    max_response_chars: int = 4000
    # Extra knobs
    max_prompt_chars: int = 4000
    retention_days: int = 0  # 0=disabled
    pii_masking: bool = False
    api_keys_extra: tuple[str, ...] = ()
    log_level: str = "INFO"
    # Generation defaults
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    dynamic_generation: bool = False
    # Redis cluster/distributed cache support
    redis_mode: str = "standalone"  # "standalone", "cluster", "sentinel"
    redis_cluster_nodes: tuple[str, ...] = ()  # host:port,host:port,...
    redis_username: str | None = None
    redis_password: str | None = None
    redis_db: int = 0
    redis_ssl: bool = False


class ConfigError(RuntimeError):
    pass


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise ConfigError(f"Missing required environment variable: {name}")
    return val


def get_config() -> Config:
    # Redis cluster/distributed cache settings
    redis_mode = os.getenv(
        "REDIS_MODE", "standalone"
    ).lower()  # standalone, cluster, sentinel
    redis_cluster_nodes = tuple(
        [s for s in os.getenv("REDIS_CLUSTER_NODES", "").split(",") if s.strip()]
    )
    redis_username = os.getenv("REDIS_USERNAME")
    redis_password = os.getenv("REDIS_PASSWORD")
    redis_db = int(os.getenv("REDIS_DB", "0"))
    redis_ssl = os.getenv("REDIS_SSL", "false").lower() in {"1", "true", "yes"}
    api_key = _require_env("API_KEY")
    host = _require_env("HOST")
    port_str = _require_env("PORT")
    model_name = _require_env("MODEL_NAME")
    llm_backend = _require_env("LLM_BACKEND").lower()
    if llm_backend not in {"ollama", "hf"}:
        raise ConfigError("LLM_BACKEND must be either 'ollama' or 'hf'")

    slack_url = os.getenv("SLACK_WEBHOOK_URL")
    # Optional resiliency settings
    llm_timeout_sec = float(os.getenv("LLM_TIMEOUT_SEC", "60"))
    llm_max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
    llm_retry_backoff = float(os.getenv("LLM_RETRY_BACKOFF", "0.5"))

    try:
        port = int(port_str)
    except ValueError as exc:
        raise ConfigError("PORT must be an integer") from exc

    # Extra knobs
    api_keys_extra = tuple(
        [s for s in os.getenv("API_KEYS_EXTRA", "").split(",") if s.strip()]
    )

    return Config(
        api_key=api_key,
        host=host,
        port=port,
        model_name=model_name,
        llm_backend=llm_backend,
        slack_webhook_url=slack_url,
        llm_timeout_sec=llm_timeout_sec,
        llm_max_retries=llm_max_retries,
        llm_retry_backoff=llm_retry_backoff,
        breaker_fail_threshold=int(os.getenv("BREAKER_FAIL_THRESHOLD", "5")),
        breaker_reset_seconds=float(os.getenv("BREAKER_RESET_SECONDS", "30")),
        cache_ttl_seconds=float(os.getenv("CACHE_TTL_SECONDS", "60")),
        max_response_chars=int(os.getenv("MAX_RESPONSE_CHARS", "4000")),
        max_prompt_chars=int(os.getenv("MAX_PROMPT_CHARS", "4000")),
        retention_days=int(os.getenv("RETENTION_DAYS", "0")),
        pii_masking=os.getenv("PII_MASKING", "false").lower() in {"1", "true", "yes"},
        api_keys_extra=api_keys_extra,
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        temperature=float(os.getenv("GEN_TEMPERATURE", "0.7")),
        top_p=float(os.getenv("GEN_TOP_P", "0.9")),
        top_k=int(os.getenv("GEN_TOP_K", "40")),
        repeat_penalty=float(os.getenv("GEN_REPEAT_PENALTY", "1.1")),
        presence_penalty=float(os.getenv("GEN_PRESENCE_PENALTY", "0.0")),
        frequency_penalty=float(os.getenv("GEN_FREQUENCY_PENALTY", "0.0")),
        dynamic_generation=os.getenv("GEN_DYNAMIC", "false").lower()
        in {"1", "true", "yes"},
        redis_mode=redis_mode,
        redis_cluster_nodes=redis_cluster_nodes,
        redis_username=redis_username,
        redis_password=redis_password,
        redis_db=redis_db,
        redis_ssl=redis_ssl,
    )
