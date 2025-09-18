"""
llm.py - Enhanced LLM integration layer for Vega2.0

Consolidated version with multi-provider support:
- Primary backend: Ollama REST API (http://127.0.0.1:11434/api/generate)
- Multi-provider support: OpenAI, Anthropic, Azure OpenAI, Google Gemini, HuggingFace
- Intelligent routing based on provider availability and performance
- Circuit breakers and caching for resilience
- Unified interface with backward compatibility
"""

from __future__ import annotations

import asyncio
import os
import json
import time
import logging
import hashlib
from typing import AsyncGenerator, Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

import httpx

# Optional imports for enhanced providers
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Available LLM provider types"""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"
    GOOGLE = "google"


@dataclass
class LLMUsage:
    """Token usage information"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class LLMResponse:
    """Standardized LLM response"""

    content: str
    provider: str
    model: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    metadata: Dict[str, Any] = field(default_factory=dict)
    response_time: float = 0.0


class LLMBackendError(Exception):
    """Exception raised when LLM backend fails"""

    pass


# Configuration and resilience imports with fallbacks
try:
    from ..config import get_config  # type: ignore
except Exception:

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
    from ..resilience import CircuitBreaker, TTLCache  # type: ignore
except Exception:

    class CircuitBreaker:
        def __init__(self, failure_threshold=5, recovery_timeout=60):
            self.failure_threshold = failure_threshold
            self.recovery_timeout = recovery_timeout
            self.failure_count = 0
            self.last_failure_time = None
            self.state = "CLOSED"

        def allow(self):
            if self.state == "OPEN":
                if self._should_try_reset():
                    self.state = "HALF_OPEN"
                    return True
                return False
            return True

        def _should_try_reset(self):
            return (
                self.last_failure_time
                and time.time() - self.last_failure_time >= self.recovery_timeout
            )

        def on_success(self):
            self.failure_count = 0
            self.state = "CLOSED"

        def on_failure(self):
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"

        def status(self):
            return {
                "state": self.state,
                "failure_count": self.failure_count,
                "last_failure_time": self.last_failure_time,
            }

    class TTLCache:
        def __init__(self, ttl=300):
            self.ttl = ttl
            self.cache = {}

        def get(self, key):
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    return value
                else:
                    del self.cache[key]
            return None

        def set(self, key, value):
            self.cache[key] = (value, time.time())

        def stats(self):
            return {"size": len(self.cache), "ttl": self.ttl}


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__.replace("Provider", "").lower()

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response to the prompt"""
        pass

    @abstractmethod
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream response tokens"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available and configured"""
        pass

    @abstractmethod
    def get_models(self) -> List[str]:
        """Get list of available models"""
        pass

    def calculate_cost(self, usage: LLMUsage, model: str) -> float:
        """Calculate cost based on usage - override in subclasses"""
        return 0.0


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("ollama_host", "http://127.0.0.1:11434")
        self.default_model = config.get("model_name", "mistral")

    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            import requests

            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def get_models(self) -> List[str]:
        """Get available Ollama models"""
        if not self.is_available():
            return []
        try:
            import requests

            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except Exception:
            pass
        return [self.default_model]

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Ollama"""
        start_time = time.time()
        model = kwargs.get("model", self.default_model)

        # Apply system prompt
        prompt = self._apply_system_prompt(prompt)

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "top_k": kwargs.get("top_k", 40),
                "repeat_penalty": kwargs.get("repeat_penalty", 1.1),
            },
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/generate", json=payload
                )
                response.raise_for_status()
                result = response.json()

                response_time = time.time() - start_time

                return LLMResponse(
                    content=result.get("response", ""),
                    provider="ollama",
                    model=model,
                    response_time=response_time,
                    metadata={
                        "total_duration": result.get("total_duration", 0),
                        "load_duration": result.get("load_duration", 0),
                        "prompt_eval_count": result.get("prompt_eval_count", 0),
                        "eval_count": result.get("eval_count", 0),
                    },
                )
            except Exception as e:
                raise LLMBackendError(f"Ollama generation failed: {e}")

    async def stream_generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream tokens from Ollama"""
        model = kwargs.get("model", self.default_model)
        prompt = self._apply_system_prompt(prompt)

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "top_k": kwargs.get("top_k", 40),
                "repeat_penalty": kwargs.get("repeat_penalty", 1.1),
            },
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                async with client.stream(
                    "POST", f"{self.base_url}/api/generate", json=payload
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                chunk = json.loads(line)
                                if "response" in chunk:
                                    yield chunk["response"]
                                if chunk.get("done"):
                                    break
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                raise LLMBackendError(f"Ollama streaming failed: {e}")

    def _apply_system_prompt(self, user_prompt: str) -> str:
        """Apply system prompt if available"""
        sp = self._maybe_load_system_prompt()
        if sp:
            return f"System: {sp}\n\nUser: {user_prompt}"
        return user_prompt

    def _maybe_load_system_prompt(self) -> Optional[str]:
        """Load system prompt from file if available"""
        path = os.path.join(os.path.dirname(__file__), "prompts", "system_prompt.txt")
        try:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
                return txt if txt else None
        except Exception:
            return None


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = None
        api_key = config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if OPENAI_AVAILABLE and api_key:
            self.client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=config.get("openai_base_url", "https://api.openai.com/v1"),
            )
        self.default_model = config.get("openai_model", "gpt-3.5-turbo")

    def is_available(self) -> bool:
        """Check if OpenAI is configured"""
        return OPENAI_AVAILABLE and self.client is not None

    def get_models(self) -> List[str]:
        """Get available OpenAI models"""
        return ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"]

    def calculate_cost(self, usage: LLMUsage, model: str) -> float:
        """Calculate OpenAI API cost"""
        # Approximate pricing (update with current rates)
        rates = {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        }

        rate = rates.get(model, rates["gpt-3.5-turbo"])
        return (
            usage.prompt_tokens * rate["input"]
            + usage.completion_tokens * rate["output"]
        ) / 1000

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using OpenAI"""
        if not self.client:
            raise LLMBackendError("OpenAI client not configured")

        start_time = time.time()
        model = kwargs.get("model", self.default_model)

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", 2048),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 1.0),
            )

            response_time = time.time() - start_time
            usage = LLMUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
            usage.cost_usd = self.calculate_cost(usage, model)

            return LLMResponse(
                content=response.choices[0].message.content,
                provider="openai",
                model=model,
                usage=usage,
                response_time=response_time,
            )
        except Exception as e:
            raise LLMBackendError(f"OpenAI generation failed: {e}")

    async def stream_generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream tokens from OpenAI"""
        if not self.client:
            raise LLMBackendError("OpenAI client not configured")

        model = kwargs.get("model", self.default_model)

        try:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", 2048),
                temperature=kwargs.get("temperature", 0.7),
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise LLMBackendError(f"OpenAI streaming failed: {e}")


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = None
        api_key = config.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY")
        if ANTHROPIC_AVAILABLE and api_key:
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.default_model = config.get("anthropic_model", "claude-3-sonnet-20240229")

    def is_available(self) -> bool:
        """Check if Anthropic is configured"""
        return ANTHROPIC_AVAILABLE and self.client is not None

    def get_models(self) -> List[str]:
        """Get available Anthropic models"""
        return [
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229",
            "claude-3-opus-20240229",
            "claude-3-5-sonnet-20241022",
        ]

    def calculate_cost(self, usage: LLMUsage, model: str) -> float:
        """Calculate Anthropic API cost"""
        rates = {
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
        }

        rate = rates.get(model, rates["claude-3-sonnet-20240229"])
        return (
            usage.prompt_tokens * rate["input"]
            + usage.completion_tokens * rate["output"]
        ) / 1000

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Anthropic"""
        if not self.client:
            raise LLMBackendError("Anthropic client not configured")

        start_time = time.time()
        model = kwargs.get("model", self.default_model)

        try:
            response = await self.client.messages.create(
                model=model,
                max_tokens=kwargs.get("max_tokens", 2048),
                temperature=kwargs.get("temperature", 0.7),
                messages=[{"role": "user", "content": prompt}],
            )

            response_time = time.time() - start_time
            usage = LLMUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            )
            usage.cost_usd = self.calculate_cost(usage, model)

            return LLMResponse(
                content=response.content[0].text,
                provider="anthropic",
                model=model,
                usage=usage,
                response_time=response_time,
            )
        except Exception as e:
            raise LLMBackendError(f"Anthropic generation failed: {e}")

    async def stream_generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream tokens from Anthropic"""
        if not self.client:
            raise LLMBackendError("Anthropic client not configured")

        model = kwargs.get("model", self.default_model)

        try:
            async with self.client.messages.stream(
                model=model,
                max_tokens=kwargs.get("max_tokens", 2048),
                temperature=kwargs.get("temperature", 0.7),
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            raise LLMBackendError(f"Anthropic streaming failed: {e}")


class LLMManager:
    """Manages multiple LLM providers with intelligent routing and fallbacks"""

    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.circuit_breaker = CircuitBreaker()
        self.cache = TTLCache()
        self.usage_tracker = {}
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize all available LLM providers"""
        config = {
            # Ollama
            "ollama_host": os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            "model_name": os.getenv("MODEL_NAME", "mistral"),
            # OpenAI
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "openai_model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            "openai_base_url": os.getenv(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            ),
            # Anthropic
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "anthropic_model": os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
        }

        # Initialize providers
        self.providers["ollama"] = OllamaProvider(config)
        self.providers["openai"] = OpenAIProvider(config)
        self.providers["anthropic"] = AnthropicProvider(config)

        logger.info(f"Initialized LLM providers: {list(self.providers.keys())}")

    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return [
            name for name, provider in self.providers.items() if provider.is_available()
        ]

    def get_preferred_provider(
        self, preferred: Optional[str] = None
    ) -> BaseLLMProvider:
        """Get the preferred provider with fallbacks"""
        available = self.get_available_providers()

        if not available:
            raise LLMBackendError("No LLM providers available")

        # Try preferred provider first
        if preferred and preferred in available:
            return self.providers[preferred]

        # Default preference order
        preference_order = ["ollama", "openai", "anthropic"]
        for provider_name in preference_order:
            if provider_name in available:
                return self.providers[provider_name]

        # Fallback to any available
        return self.providers[available[0]]

    def _get_cache_key(self, prompt: str, provider: str, **kwargs) -> str:
        """Generate cache key for request"""
        key_data = f"{provider}:{prompt}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    async def generate(
        self,
        prompt: str,
        provider: Optional[str] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> LLMResponse:
        """Generate response with provider fallback"""
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(prompt, provider or "auto", **kwargs)
            cached = self.cache.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for key: {cache_key[:16]}...")
                return cached

        # Check circuit breaker
        if not self.circuit_breaker.allow():
            raise LLMBackendError(
                "Circuit breaker is open - LLM temporarily unavailable"
            )

        try:
            llm_provider = self.get_preferred_provider(provider)
            logger.info(f"Using LLM provider: {llm_provider.name}")

            response = await llm_provider.generate(prompt, **kwargs)

            # Track usage
            if llm_provider.name not in self.usage_tracker:
                self.usage_tracker[llm_provider.name] = {
                    "requests": 0,
                    "total_cost": 0.0,
                }
            self.usage_tracker[llm_provider.name]["requests"] += 1
            self.usage_tracker[llm_provider.name][
                "total_cost"
            ] += response.usage.cost_usd

            # Cache response
            if use_cache:
                self.cache.set(cache_key, response)

            # Circuit breaker success
            self.circuit_breaker.on_success()

            return response

        except Exception as e:
            self.circuit_breaker.on_failure()
            logger.error(f"LLM generation failed: {e}")
            raise LLMBackendError(f"LLM generation failed: {e}")

    async def stream_generate(
        self, prompt: str, provider: Optional[str] = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response with provider fallback"""
        # Check circuit breaker
        if not self.circuit_breaker.allow():
            yield "[Error: LLM temporarily unavailable]"
            return

        try:
            llm_provider = self.get_preferred_provider(provider)
            logger.info(f"Streaming from provider: {llm_provider.name}")

            async for token in llm_provider.stream_generate(prompt, **kwargs):
                yield token

            # Circuit breaker success
            self.circuit_breaker.on_success()

        except Exception as e:
            self.circuit_breaker.on_failure()
            logger.error(f"LLM streaming failed: {e}")
            yield f"[Error: {str(e)}]"

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics and status"""
        return {
            "available_providers": self.get_available_providers(),
            "circuit_breaker": self.circuit_breaker.status(),
            "cache": self.cache.stats(),
            "usage": self.usage_tracker,
        }


# Global LLM manager instance
_llm_manager: Optional[LLMManager] = None


def get_llm_manager() -> LLMManager:
    """Get global LLM manager instance"""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager


# Backward compatibility functions
async def query_llm(
    prompt: str,
    stream: bool = False,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    **kwargs,
) -> Union[str, AsyncGenerator[str, None]]:
    """Main LLM query function with provider dispatch"""
    manager = get_llm_manager()

    # Get backend from config if not specified
    if not provider:
        cfg = get_config()
        provider = getattr(cfg, "llm_backend", "ollama")

    if stream:
        return manager.stream_generate(prompt, provider=provider, model=model, **kwargs)
    else:
        response = await manager.generate(
            prompt, provider=provider, model=model, **kwargs
        )
        return response.content


async def query_ollama(
    prompt: str, model: Optional[str] = None, stream: bool = False, **kwargs
):
    """Legacy Ollama-specific function"""
    return await query_llm(
        prompt, stream=stream, model=model, provider="ollama", **kwargs
    )


# Utility functions
async def llm_warmup():
    """Warm up available LLM providers"""
    manager = get_llm_manager()
    for provider_name in manager.get_available_providers():
        try:
            await manager.generate("test", provider=provider_name)
            logger.info(f"Warmed up provider: {provider_name}")
        except Exception as e:
            logger.warning(f"Failed to warm up {provider_name}: {e}")


async def llm_shutdown():
    """Gracefully shutdown LLM resources"""
    global _llm_manager
    if _llm_manager:
        # Close any open connections
        for provider in _llm_manager.providers.values():
            if hasattr(provider, "client") and provider.client:
                try:
                    await provider.client.aclose()
                except Exception:
                    pass
        _llm_manager = None


def get_generation_settings() -> Dict:
    """Get current generation settings"""
    cfg = get_config()
    return {
        "temperature": getattr(cfg, "temperature", 0.7),
        "top_p": getattr(cfg, "top_p", 0.9),
        "top_k": getattr(cfg, "top_k", 40),
        "repeat_penalty": getattr(cfg, "repeat_penalty", 1.1),
        "max_tokens": 2048,
    }


def set_generation_settings(**kwargs) -> Dict:
    """Set generation settings (for backward compatibility)"""
    # This could store settings in the manager if needed
    return get_generation_settings()


def reset_generation_settings() -> Dict:
    """Reset generation settings"""
    return get_generation_settings()


def breaker_stats() -> Dict:
    """Get circuit breaker statistics"""
    manager = get_llm_manager()
    return manager.circuit_breaker.status()


def cache_stats() -> Dict:
    """Get cache statistics"""
    manager = get_llm_manager()
    return manager.cache.stats()


def get_provider_stats() -> Dict[str, Any]:
    """Get statistics for all LLM providers"""
    manager = get_llm_manager()
    return manager.get_stats()


def get_available_models() -> Dict[str, List[str]]:
    """Get available models for each provider"""
    manager = get_llm_manager()
    models = {}

    for name, provider in manager.providers.items():
        if provider.is_available():
            models[name] = provider.get_models()

    return models


def estimate_cost(
    prompt: str, response: str, provider: str, model: str
) -> Optional[float]:
    """Estimate cost for a query based on tokens and provider pricing"""
    try:
        manager = get_llm_manager()

        if provider not in manager.providers:
            return None

        provider_obj = manager.providers[provider]

        # Rough token estimation (1 token ≈ 0.75 words)
        prompt_tokens = len(prompt.split()) * 1.33
        response_tokens = len(response.split()) * 1.33

        usage = LLMUsage(
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(response_tokens),
            total_tokens=int(prompt_tokens + response_tokens),
        )

        return provider_obj.calculate_cost(usage, model)

    except Exception:
        return None
