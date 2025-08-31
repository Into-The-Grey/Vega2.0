"""
llm.py - LLM integration layer for Vega2.0

- Primary backend: Ollama REST API (http://127.0.0.1:11434/api/generate)
- Future backend: Hugging Face transformers (local model)
- Exposes async function query_ollama(prompt, model, stream=False) that yields or returns text
- Model name comes from config (.env)
"""

from __future__ import annotations

import asyncio
import os
import json
from typing import AsyncGenerator, Optional, Dict

import httpx

from config import get_config
from resilience import CircuitBreaker, TTLCache

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# Global breaker, cache, and pooled HTTP client
_breaker: CircuitBreaker | None = None
_cache: TTLCache | None = None
_client: httpx.AsyncClient | None = None
_gen_overrides: Dict[str, float | int | bool] = {}


class LLMBackendError(Exception):
    """Raised when the selected LLM backend is unavailable or fails."""


def _maybe_load_system_prompt() -> Optional[str]:
    path = os.path.join(os.path.dirname(__file__), "prompts", "system_prompt.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            return txt if txt else None
    except Exception:
        return None


def _apply_system_prompt(user_prompt: str) -> str:
    sp = _maybe_load_system_prompt()
    if sp:
        return f"System: {sp}\n\nUser: {user_prompt}"
    return user_prompt


def _ensure_resilience_objects():
    global _breaker, _cache, _client
    if _breaker is None or _cache is None:
        cfg = get_config()
        _breaker = CircuitBreaker(cfg.breaker_fail_threshold, cfg.breaker_reset_seconds)
        _cache = TTLCache(cfg.cache_ttl_seconds)
    if _client is None:
        _client = httpx.AsyncClient()


def _sanitize(text: str) -> str:
    cfg = get_config()
    t = text or ""
    if len(t) > cfg.max_response_chars:
        t = t[: cfg.max_response_chars] + "…"
    return t


def get_generation_settings() -> Dict:
    cfg = get_config()
    # Start from config defaults and apply overrides
    base = {
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "repeat_penalty": cfg.repeat_penalty,
        "presence_penalty": cfg.presence_penalty,
        "frequency_penalty": cfg.frequency_penalty,
        "dynamic_generation": bool(cfg.dynamic_generation),
    }
    base.update(_gen_overrides)
    return base


def set_generation_settings(**kwargs) -> Dict:
    """Set runtime generation overrides. Returns the merged settings."""
    global _gen_overrides
    for k, v in kwargs.items():
        if v is None:
            continue
        _gen_overrides[k] = v
    return get_generation_settings()


def reset_generation_settings() -> Dict:
    global _gen_overrides
    _gen_overrides = {}
    return get_generation_settings()


def _dynamic_adjust(settings: Dict, prompt: str) -> Dict:
    """Heuristic adjustments based on prompt length/feel.

    - Longer prompts -> lower temperature slightly, increase top_p a bit.
    - If prompt contains terms like 'creative', 'brainstorm', raise temperature.
    - If prompt contains 'concise', 'precise', lower temperature and top_p.
    """
    cfg = get_config()
    s = dict(settings)
    text = prompt.lower()
    L = len(prompt)
    # length-based
    if L > 2000:
        s["temperature"] = max(0.3, float(s.get("temperature", cfg.temperature)) - 0.2)
        s["top_p"] = min(0.98, float(s.get("top_p", cfg.top_p)) + 0.05)
    elif L < 200:
        s["temperature"] = min(1.2, float(s.get("temperature", cfg.temperature)) + 0.1)
    # intent words
    creative = any(w in text for w in ["creative", "brainstorm", "ideas", "story"])
    precise = any(w in text for w in ["concise", "precise", "exact", "factual"])
    if creative:
        s["temperature"] = min(1.3, float(s.get("temperature", cfg.temperature)) + 0.2)
        s["top_p"] = min(0.99, float(s.get("top_p", cfg.top_p)) + 0.05)
    if precise:
        s["temperature"] = max(0.1, float(s.get("temperature", cfg.temperature)) - 0.2)
        s["top_p"] = max(0.6, float(s.get("top_p", cfg.top_p)) - 0.1)
    return s


async def _ollama_stream(payload: dict) -> AsyncGenerator[str, None]:
    """Stream tokens from Ollama as they arrive, yielding text chunks."""
    cfg = get_config()
    attempts = max(1, int(cfg.llm_max_retries) + 1)
    backoff = float(cfg.llm_retry_backoff)
    timeout = float(cfg.llm_timeout_sec)
    for i in range(attempts):
        try:
            _ensure_resilience_objects()
            assert _client is not None
            async with _client.stream(
                "POST", OLLAMA_URL, json=payload, timeout=timeout
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "response" in data:
                        yield data["response"]
                    if data.get("done"):
                        return
        except httpx.HTTPError:
            if i < attempts - 1:
                await asyncio.sleep(backoff * (2**i))
                continue
            raise LLMBackendError("Ollama streaming request failed")


async def llm_warmup():
    """Optional one-time warmup to reduce first-token latency."""
    try:
        await query_ollama("ping", model=get_config().model_name, stream=False)
    except Exception:
        pass


async def llm_shutdown():
    """Gracefully close pooled resources."""
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        except Exception:
            pass
        _client = None


def breaker_stats() -> Dict:
    _ensure_resilience_objects()
    return _breaker.status() if _breaker else {}


def cache_stats() -> Dict:
    _ensure_resilience_objects()
    return _cache.stats() if _cache else {}


async def query_ollama(prompt: str, model: Optional[str] = None, stream: bool = False):
    """Query Ollama generate endpoint.

    Args:
        prompt: user prompt
        model: model name (defaults to config.model_name)
        stream: if True, returns an async generator of text chunks; otherwise returns a full string
    """
    cfg = get_config()
    _ensure_resilience_objects()
    model = model or cfg.model_name
    prompt = _apply_system_prompt(prompt)
    # generation settings
    settings = get_generation_settings()
    if bool(settings.get("dynamic_generation")):
        settings = _dynamic_adjust(settings, prompt)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        # Map settings to Ollama where supported
        "options": {
            "temperature": settings.get("temperature"),
            "top_p": settings.get("top_p"),
            "top_k": settings.get("top_k"),
            "repeat_penalty": settings.get("repeat_penalty"),
        },
    }

    if stream:
        if _breaker and not _breaker.allow():

            async def _degraded():
                yield "[degraded: backend temporarily unavailable]"

            return _degraded()
        return _ollama_stream(payload)
    else:
        attempts = max(1, int(cfg.llm_max_retries) + 1)
        backoff = float(cfg.llm_retry_backoff)
        timeout = float(cfg.llm_timeout_sec)
        key = f"{model}|{prompt}"
        if _cache:
            cached = _cache.get(key)
            if cached is not None:
                return _sanitize(str(cached))
        if _breaker and not _breaker.allow():
            return _sanitize("[degraded: backend temporarily unavailable]")
        for i in range(attempts):
            try:
                _ensure_resilience_objects()
                assert _client is not None
                r = await _client.post(OLLAMA_URL, json=payload, timeout=timeout)
                r.raise_for_status()
                data = r.json()
                out = _sanitize(data.get("response", ""))
                if _cache:
                    _cache.set(key, out)
                if _breaker:
                    _breaker.on_success()
                return out
            except httpx.HTTPError:
                if _breaker:
                    _breaker.on_failure()
                if i < attempts - 1:
                    await asyncio.sleep(backoff * (2**i))
                    continue
                raise LLMBackendError("Ollama request failed")


# Placeholder for the HF backend toggle in future
async def query_llm(prompt: str, stream: bool = False, model: Optional[str] = None):
    """Dispatch to selected backend. Currently supports only 'ollama'."""
    cfg = get_config()
    if cfg.llm_backend == "ollama":
        return await query_ollama(prompt, model=model, stream=stream)
    else:
        raise NotImplementedError("HF backend not implemented yet")
