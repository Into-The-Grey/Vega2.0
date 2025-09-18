"""
llm.py - Enhanced LLM integration layer for Vega2.0

- Primary backend: Ollama REST API (http://127.0.0.1:11434/api/generate)
- Multi-provider support: OpenAI, Anthropic, Azure OpenAI, Google Gemini, HuggingFace
- Intelligent routing based on provider availability and performance
- Circuit breakers and caching for resilience
- Exposes async function query_ollama(prompt, model, stream=False) that yields or returns text
- Model name comes from config (.env)
"""

from __future__ import annotations

import asyncio
import os
import json
import time
import logging
from typing import AsyncGenerator, Optional, Dict, Any, List, Union
from dataclasses import dataclass
from enum import Enum

import httpx

# Setup logging
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Available LLM providers"""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    GOOGLE_GEMINI = "google_gemini"
    HUGGINGFACE = "huggingface"


@dataclass
class LLMResponse:
    """Standardized LLM response format"""

    content: str
    provider: str
    model: str
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None
    response_time: Optional[float] = None
    cached: bool = False


@dataclass
class ProviderConfig:
    """Configuration for LLM providers"""

    name: str
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    models: Optional[List[str]] = None
    enabled: bool = True
    priority: int = 1
    cost_per_1k_tokens: float = 0.0


class LLMProviderManager:
    """Manages multiple LLM providers with intelligent routing"""

    def __init__(self):
        self.providers = {}
        self.provider_stats = {}
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize all available LLM providers based on configuration"""
        cfg = get_config()

        # Ollama (local)
        if hasattr(cfg, "ollama_enabled") and cfg.ollama_enabled:
            self.providers[LLMProvider.OLLAMA] = ProviderConfig(
                name="ollama",
                endpoint=getattr(cfg, "ollama_url", "http://127.0.0.1:11434"),
                models=getattr(cfg, "ollama_models", ["llama3"]),
                priority=1,
            )

        # OpenAI
        if hasattr(cfg, "openai_api_key") and cfg.openai_api_key:
            self.providers[LLMProvider.OPENAI] = ProviderConfig(
                name="openai",
                api_key=cfg.openai_api_key,
                endpoint="https://api.openai.com/v1",
                models=getattr(
                    cfg, "openai_models", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
                ),
                priority=2,
                cost_per_1k_tokens=0.03,
            )

        # Anthropic
        if hasattr(cfg, "anthropic_api_key") and cfg.anthropic_api_key:
            self.providers[LLMProvider.ANTHROPIC] = ProviderConfig(
                name="anthropic",
                api_key=cfg.anthropic_api_key,
                endpoint="https://api.anthropic.com",
                models=getattr(
                    cfg,
                    "anthropic_models",
                    ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
                ),
                priority=3,
                cost_per_1k_tokens=0.03,
            )

        # Azure OpenAI
        if hasattr(cfg, "azure_openai_api_key") and cfg.azure_openai_api_key:
            self.providers[LLMProvider.AZURE_OPENAI] = ProviderConfig(
                name="azure_openai",
                api_key=cfg.azure_openai_api_key,
                endpoint=getattr(cfg, "azure_openai_endpoint", ""),
                models=getattr(cfg, "azure_openai_models", ["gpt-4"]),
                priority=4,
                cost_per_1k_tokens=0.03,
            )

        # Google Gemini
        if hasattr(cfg, "google_gemini_api_key") and cfg.google_gemini_api_key:
            self.providers[LLMProvider.GOOGLE_GEMINI] = ProviderConfig(
                name="google_gemini",
                api_key=cfg.google_gemini_api_key,
                endpoint="https://generativelanguage.googleapis.com/v1beta",
                models=getattr(
                    cfg, "gemini_models", ["gemini-pro", "gemini-pro-vision"]
                ),
                priority=5,
                cost_per_1k_tokens=0.0025,
            )

    def get_best_provider(
        self, model_preference: Optional[str] = None
    ) -> Optional[LLMProvider]:
        """Select the best available provider based on priority and health"""
        available_providers = []

        for provider, config in self.providers.items():
            if not config.enabled:
                continue

            # Check if specific model is available
            if model_preference and config.models:
                if not any(model_preference in model for model in config.models):
                    continue

            # Check provider health (basic availability)
            stats = self.provider_stats.get(provider, {})
            failure_rate = stats.get("failure_rate", 0)
            if failure_rate > 0.5:  # Skip if failure rate > 50%
                continue

            available_providers.append((provider, config.priority, failure_rate))

        if not available_providers:
            return None

        # Sort by priority (lower = better) and failure rate
        available_providers.sort(key=lambda x: (x[1], x[2]))
        return available_providers[0][0]

    def update_provider_stats(
        self, provider: LLMProvider, success: bool, response_time: float
    ):
        """Update provider performance statistics"""
        if provider not in self.provider_stats:
            self.provider_stats[provider] = {
                "success_count": 0,
                "failure_count": 0,
                "total_response_time": 0,
                "request_count": 0,
            }

        stats = self.provider_stats[provider]
        stats["request_count"] += 1
        stats["total_response_time"] += response_time

        if success:
            stats["success_count"] += 1
        else:
            stats["failure_count"] += 1

        # Calculate failure rate
        total_requests = stats["success_count"] + stats["failure_count"]
        stats["failure_rate"] = (
            stats["failure_count"] / total_requests if total_requests > 0 else 0
        )
        stats["avg_response_time"] = (
            stats["total_response_time"] / stats["request_count"]
        )


# Global provider manager
_provider_manager: Optional[LLMProviderManager] = None


def get_provider_manager() -> LLMProviderManager:
    """Get or create the global provider manager"""
    global _provider_manager
    if _provider_manager is None:
        _provider_manager = LLMProviderManager()
    return _provider_manager


# Provider-specific query functions
async def _query_openai(
    prompt: str, model: str, stream: bool = False, **kwargs
) -> Union[str, AsyncGenerator[str, None]]:
    """Query OpenAI API"""
    cfg = get_config()
    manager = get_provider_manager()
    provider_config = manager.providers.get(LLMProvider.OPENAI)

    if not provider_config or not provider_config.api_key:
        raise LLMBackendError("OpenAI provider not configured")

    headers = {
        "Authorization": f"Bearer {provider_config.api_key}",
        "Content-Type": "application/json",
    }

    # Apply system prompt
    prompt = _apply_system_prompt(prompt)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "temperature": kwargs.get("temperature", 0.7),
        "max_tokens": kwargs.get("max_tokens", 1500),
    }

    if stream:

        async def _openai_stream():
            try:
                _ensure_resilience_objects()
                assert _client is not None
                async with _client.stream(
                    "POST",
                    f"{provider_config.endpoint}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=30.0,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                if "choices" in data and data["choices"]:
                                    delta = data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                logger.error(f"OpenAI streaming error: {e}")
                yield f"[Error: {str(e)}]"

        return _openai_stream()
    else:
        try:
            _ensure_resilience_objects()
            assert _client is not None
            resp = await _client.post(
                f"{provider_config.endpoint}/chat/completions",
                json=payload,
                headers=headers,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()

            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
            return ""

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise LLMBackendError(f"OpenAI request failed: {str(e)}")


async def _query_anthropic(
    prompt: str, model: str, stream: bool = False, **kwargs
) -> Union[str, AsyncGenerator[str, None]]:
    """Query Anthropic Claude API"""
    manager = get_provider_manager()
    provider_config = manager.providers.get(LLMProvider.ANTHROPIC)

    if not provider_config or not provider_config.api_key:
        raise LLMBackendError("Anthropic provider not configured")

    headers = {
        "x-api-key": provider_config.api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    # Apply system prompt
    prompt = _apply_system_prompt(prompt)

    payload = {
        "model": model,
        "max_tokens": kwargs.get("max_tokens", 1500),
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "temperature": kwargs.get("temperature", 0.7),
    }

    if stream:

        async def _anthropic_stream():
            try:
                _ensure_resilience_objects()
                assert _client is not None
                async with _client.stream(
                    "POST",
                    f"{provider_config.endpoint}/v1/messages",
                    json=payload,
                    headers=headers,
                    timeout=30.0,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                if data.get("type") == "content_block_delta":
                                    content = data.get("delta", {}).get("text", "")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                logger.error(f"Anthropic streaming error: {e}")
                yield f"[Error: {str(e)}]"

        return _anthropic_stream()
    else:
        try:
            _ensure_resilience_objects()
            assert _client is not None
            resp = await _client.post(
                f"{provider_config.endpoint}/v1/messages",
                json=payload,
                headers=headers,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()

            if "content" in data and data["content"]:
                return data["content"][0]["text"]
            return ""

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise LLMBackendError(f"Anthropic request failed: {str(e)}")


async def _query_google_gemini(
    prompt: str, model: str, stream: bool = False, **kwargs
) -> Union[str, AsyncGenerator[str, None]]:
    """Query Google Gemini API"""
    manager = get_provider_manager()
    provider_config = manager.providers.get(LLMProvider.GOOGLE_GEMINI)

    if not provider_config or not provider_config.api_key:
        raise LLMBackendError("Google Gemini provider not configured")

    # Apply system prompt
    prompt = _apply_system_prompt(prompt)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": kwargs.get("temperature", 0.7),
            "maxOutputTokens": kwargs.get("max_tokens", 1500),
        },
    }

    endpoint = f"{provider_config.endpoint}/models/{model}:generateContent"
    params = {"key": provider_config.api_key}

    if stream:

        async def _gemini_stream():
            try:
                _ensure_resilience_objects()
                assert _client is not None
                # Gemini streaming endpoint
                stream_endpoint = (
                    f"{provider_config.endpoint}/models/{model}:streamGenerateContent"
                )
                resp = await _client.post(
                    stream_endpoint, json=payload, params=params, timeout=30.0
                )
                resp.raise_for_status()

                # Process streaming response
                async for line in resp.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "candidates" in data and data["candidates"]:
                                candidate = data["candidates"][0]
                                if (
                                    "content" in candidate
                                    and "parts" in candidate["content"]
                                ):
                                    for part in candidate["content"]["parts"]:
                                        if "text" in part:
                                            yield part["text"]
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.error(f"Gemini streaming error: {e}")
                yield f"[Error: {str(e)}]"

        return _gemini_stream()
    else:
        try:
            _ensure_resilience_objects()
            assert _client is not None
            resp = await _client.post(
                endpoint, json=payload, params=params, timeout=30.0
            )
            resp.raise_for_status()
            data = resp.json()

            if "candidates" in data and data["candidates"]:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    return candidate["content"]["parts"][0].get("text", "")
            return ""

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise LLMBackendError(f"Gemini request failed: {str(e)}")


async def query_multi_provider(
    prompt: str, model: Optional[str] = None, stream: bool = False, **kwargs
) -> Union[str, AsyncGenerator[str, None]]:
    """Enhanced query function with multi-provider support and intelligent routing"""
    start_time = time.time()
    cfg = get_config()
    manager = get_provider_manager()

    # Determine the best provider
    selected_provider = manager.get_best_provider(model)

    if not selected_provider:
        # Fallback to Ollama if no other providers available
        if LLMProvider.OLLAMA in manager.providers:
            selected_provider = LLMProvider.OLLAMA
        else:
            raise LLMBackendError("No LLM providers available")

    # Route to appropriate provider
    try:
        if selected_provider == LLMProvider.OLLAMA:
            # Use existing Ollama implementation
            result = await query_ollama(prompt, model=model, stream=stream)
            manager.update_provider_stats(
                selected_provider, True, time.time() - start_time
            )
            return result

        elif selected_provider == LLMProvider.OPENAI:
            # Default to gpt-4o-mini if no model specified
            model = model or "gpt-4o-mini"
            result = await _query_openai(prompt, model, stream, **kwargs)
            manager.update_provider_stats(
                selected_provider, True, time.time() - start_time
            )
            return result

        elif selected_provider == LLMProvider.ANTHROPIC:
            # Default to claude-3-haiku-20240307 if no model specified
            model = model or "claude-3-haiku-20240307"
            result = await _query_anthropic(prompt, model, stream, **kwargs)
            manager.update_provider_stats(
                selected_provider, True, time.time() - start_time
            )
            return result

        elif selected_provider == LLMProvider.GOOGLE_GEMINI:
            # Default to gemini-pro if no model specified
            model = model or "gemini-pro"
            result = await _query_google_gemini(prompt, model, stream, **kwargs)
            manager.update_provider_stats(
                selected_provider, True, time.time() - start_time
            )
            return result

        else:
            raise LLMBackendError(f"Provider {selected_provider} not implemented yet")

    except Exception as e:
        # Update failure stats
        manager.update_provider_stats(
            selected_provider, False, time.time() - start_time
        )

        # Try fallback to Ollama if available and not already tried
        if (
            selected_provider != LLMProvider.OLLAMA
            and LLMProvider.OLLAMA in manager.providers
        ):
            logger.warning(
                f"Provider {selected_provider} failed, falling back to Ollama: {e}"
            )
            try:
                return await query_ollama(prompt, model=model, stream=stream)
            except Exception as fallback_error:
                logger.error(f"Fallback to Ollama also failed: {fallback_error}")

        raise e


try:
    from config import get_config  # type: ignore
except Exception:  # provide minimal fallback for tests

    class _CfgFallback:
        breaker_fail_threshold = 5
        breaker_reset_seconds = 30
        cache_ttl_seconds = 60
        max_response_chars = 2000
        llm_backend = "ollama"
        temperature = 0.7
        top_p = 0.9
        top_k = 40
        repeat_penalty = 1.1
        presence_penalty = 0.0
        frequency_penalty = 0.0
        dynamic_generation = True
        llm_max_retries = 1
        llm_retry_backoff = 0.2
        llm_timeout_sec = 10
        model_name = "llama3"

    def get_config():  # type: ignore
        return _CfgFallback()


try:
    from resilience import CircuitBreaker, TTLCache  # type: ignore
except Exception:

    class CircuitBreaker:
        def __init__(self, *args, **kwargs):
            pass

        def status(self):
            return {}

    class TTLCache:
        def __init__(self, *args, **kwargs):
            pass

        def stats(self):
            return {}


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


# Enhanced main query function with multi-provider support
async def query_llm(
    prompt: str, stream: bool = False, model: Optional[str] = None, **kwargs
):
    """
    Enhanced dispatch to selected backend with multi-provider support.

    Supports: ollama, openai, anthropic, azure_openai, google_gemini
    Maintains backward compatibility with existing Ollama-only interface.
    """
    cfg = get_config()

    # Check if multi-provider mode is enabled
    if hasattr(cfg, "multi_provider_enabled") and cfg.multi_provider_enabled:
        # Use intelligent provider routing
        return await query_multi_provider(prompt, model=model, stream=stream, **kwargs)

    # Backward compatibility: use specific backend or default to ollama
    backend = getattr(cfg, "llm_backend", "ollama")

    if backend == "ollama":
        return await query_ollama(prompt, model=model, stream=stream)
    elif backend == "openai":
        model = model or "gpt-4o-mini"
        return await _query_openai(prompt, model, stream, **kwargs)
    elif backend == "anthropic":
        model = model or "claude-3-haiku-20240307"
        return await _query_anthropic(prompt, model, stream, **kwargs)
    elif backend == "google_gemini":
        model = model or "gemini-pro"
        return await _query_google_gemini(prompt, model, stream, **kwargs)
    else:
        # Fallback to Ollama for unknown backends
        logger.warning(f"Unknown backend '{backend}', falling back to Ollama")
        return await query_ollama(prompt, model=model, stream=stream)


# Provider management and statistics functions
def get_provider_stats() -> Dict[str, Any]:
    """Get statistics for all LLM providers"""
    manager = get_provider_manager()
    return {
        "providers": {
            provider.value: {
                "config": {
                    "name": config.name,
                    "enabled": config.enabled,
                    "priority": config.priority,
                    "models": config.models,
                    "cost_per_1k_tokens": config.cost_per_1k_tokens,
                },
                "stats": manager.provider_stats.get(provider, {}),
            }
            for provider, config in manager.providers.items()
        },
        "circuit_breaker": breaker_stats(),
        "cache": cache_stats(),
    }


def set_provider_enabled(provider_name: str, enabled: bool) -> bool:
    """Enable or disable a specific provider"""
    try:
        provider = LLMProvider(provider_name)
        manager = get_provider_manager()
        if provider in manager.providers:
            manager.providers[provider].enabled = enabled
            return True
    except ValueError:
        pass
    return False


def get_available_models() -> Dict[str, List[str]]:
    """Get available models for each provider"""
    manager = get_provider_manager()
    models = {}

    for provider, config in manager.providers.items():
        if config.enabled and config.models:
            models[provider.value] = config.models

    return models


# Cost estimation functions
def estimate_cost(
    prompt: str, response: str, provider: str, model: str
) -> Optional[float]:
    """Estimate cost for a query based on tokens and provider pricing"""
    try:
        manager = get_provider_manager()
        provider_enum = LLMProvider(provider)

        if provider_enum not in manager.providers:
            return None

        config = manager.providers[provider_enum]

        # Rough token estimation (1 token ≈ 0.75 words)
        prompt_tokens = len(prompt.split()) * 1.33
        response_tokens = len(response.split()) * 1.33
        total_tokens = prompt_tokens + response_tokens

        # Calculate cost
        cost = (total_tokens / 1000) * config.cost_per_1k_tokens
        return round(cost, 6)

    except Exception:
        return None
