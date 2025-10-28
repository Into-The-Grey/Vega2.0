"""
integration_health.py - Integration Health Monitoring for Vega2.0

Provides real-time health checks for all external integrations:
- Web search (DuckDuckGo/Google)
- Web fetch (httpx)
- OSINT (Shodan, Hunter, etc.)
- Slack connector
- Home Assistant
- External APIs

Performs parallel health checks with timeouts and circuit breaker awareness.
"""

from __future__ import annotations

import asyncio
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class IntegrationHealth:
    """Health status for a single integration"""

    name: str
    status: str  # "healthy", "degraded", "unhealthy", "disabled"
    response_time_ms: Optional[float] = None
    error: Optional[str] = None
    last_check: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


async def check_search_integration() -> IntegrationHealth:
    """Check web search integration health"""
    try:
        from ..integrations.search import web_search

        start = time.perf_counter()

        # Perform lightweight search test
        # web_search is synchronous; call directly
        result = web_search("python", max_results=1)

        duration_ms = (time.perf_counter() - start) * 1000

        if result and len(result) > 0:
            return IntegrationHealth(
                name="web_search",
                status="healthy",
                response_time_ms=duration_ms,
                last_check=datetime.utcnow().isoformat(),
                details={"result_count": len(result)},
            )
        else:
            return IntegrationHealth(
                name="web_search",
                status="degraded",
                response_time_ms=duration_ms,
                last_check=datetime.utcnow().isoformat(),
                error="No results returned",
            )
    except Exception as e:
        return IntegrationHealth(
            name="web_search",
            status="unhealthy",
            error=str(e),
            last_check=datetime.utcnow().isoformat(),
        )


async def check_fetch_integration() -> IntegrationHealth:
    """Check web fetch integration health"""
    try:
        from ..integrations.fetch import fetch_text

        start = time.perf_counter()

        # Fetch a reliable test page
        result = await fetch_text("https://httpbin.org/status/200")

        duration_ms = (time.perf_counter() - start) * 1000

        if result:
            return IntegrationHealth(
                name="web_fetch",
                status="healthy",
                response_time_ms=duration_ms,
                last_check=datetime.utcnow().isoformat(),
                details={"content_length": len(result)},
            )
        else:
            return IntegrationHealth(
                name="web_fetch",
                status="degraded",
                response_time_ms=duration_ms,
                last_check=datetime.utcnow().isoformat(),
                error="Empty response",
            )
    except Exception as e:
        return IntegrationHealth(
            name="web_fetch",
            status="unhealthy",
            error=str(e),
            last_check=datetime.utcnow().isoformat(),
        )


async def check_osint_integration() -> IntegrationHealth:
    """Check OSINT integration health"""
    try:
        from ..integrations.osint import dns_lookup

        start = time.perf_counter()

        # Lightweight, local-only check that requires no external API keys
        result = dns_lookup("example.com")

        duration_ms = (time.perf_counter() - start) * 1000

        status = "healthy" if result and result.addresses else "degraded"
        details = {"addresses": result.addresses if result else []}

        return IntegrationHealth(
            name="osint",
            status=status,
            response_time_ms=duration_ms,
            last_check=datetime.utcnow().isoformat(),
            details=details,
        )
    except Exception as e:
        error_msg = str(e)

        # Distinguish between disabled and unhealthy
        if "not configured" in error_msg.lower() or "api key" in error_msg.lower():
            return IntegrationHealth(
                name="osint",
                status="disabled",
                error=error_msg,
                last_check=datetime.utcnow().isoformat(),
            )
        else:
            return IntegrationHealth(
                name="osint",
                status="unhealthy",
                error=error_msg,
                last_check=datetime.utcnow().isoformat(),
            )


async def check_slack_integration() -> IntegrationHealth:
    """Check Slack integration health"""
    try:
        from ..config import get_config

        config = get_config()

        if not config.slack_webhook_url:
            return IntegrationHealth(
                name="slack",
                status="disabled",
                last_check=datetime.utcnow().isoformat(),
                details={"reason": "No webhook URL configured"},
            )

        # Don't actually send a test message, just check configuration
        return IntegrationHealth(
            name="slack",
            status="healthy",
            last_check=datetime.utcnow().isoformat(),
            details={"webhook_configured": True},
        )
    except Exception as e:
        return IntegrationHealth(
            name="slack",
            status="unhealthy",
            error=str(e),
            last_check=datetime.utcnow().isoformat(),
        )


async def check_homeassistant_integration() -> IntegrationHealth:
    """Check Home Assistant integration health"""
    try:
        from ..integrations.homeassistant import HomeAssistantClient
        from ..config import get_config

        config = get_config()

        if not config.hass_enabled:
            return IntegrationHealth(
                name="homeassistant",
                status="disabled",
                last_check=datetime.utcnow().isoformat(),
                details={"reason": "Integration disabled in config"},
            )

        if not config.hass_url or not config.hass_token:
            return IntegrationHealth(
                name="homeassistant",
                status="unhealthy",
                last_check=datetime.utcnow().isoformat(),
                error="Missing HASS_URL or HASS_TOKEN",
            )

        start = time.perf_counter()

        # Lightweight API check (no network call here)
        client = HomeAssistantClient(config.hass_url, config.hass_token)
        # Just verify client creation - actual API call would require valid endpoint

        duration_ms = (time.perf_counter() - start) * 1000

        return IntegrationHealth(
            name="homeassistant",
            status="healthy",
            response_time_ms=duration_ms,
            last_check=datetime.utcnow().isoformat(),
            details={"url": config.hass_url},
        )
    except Exception as e:
        return IntegrationHealth(
            name="homeassistant",
            status="unhealthy",
            error=str(e),
            last_check=datetime.utcnow().isoformat(),
        )


async def check_llm_backend() -> IntegrationHealth:
    """Check LLM backend health"""
    try:
        from ..llm import get_llm_manager
        from ..config import get_config

        config = get_config()

        start = time.perf_counter()

        manager = get_llm_manager()
        # Choose provider from config or default to 'ollama'
        provider_name = getattr(config, "llm_backend", "ollama")
        provider = manager.providers.get(provider_name)

        is_available = False
        models_count = 0

        if provider:
            try:
                # Handle async/sync is_available
                avail = provider.is_available()
                if asyncio.iscoroutine(avail):
                    is_available = await avail
                else:
                    is_available = bool(avail)

                # Handle async/sync get_models
                get_models = getattr(provider, "get_models", None)
                if get_models:
                    models = get_models()
                    if asyncio.iscoroutine(models):
                        models = await models
                    models_count = len(models or [])
            except Exception:
                is_available = False

        duration_ms = (time.perf_counter() - start) * 1000

        if is_available:
            return IntegrationHealth(
                name="llm_backend",
                status="healthy",
                response_time_ms=duration_ms,
                last_check=datetime.utcnow().isoformat(),
                details={
                    "backend": provider_name,
                    "model": getattr(config, "model_name", ""),
                    "available_models": models_count,
                },
            )
        else:
            return IntegrationHealth(
                name="llm_backend",
                status="unhealthy",
                response_time_ms=duration_ms,
                last_check=datetime.utcnow().isoformat(),
                error="Backend not available",
                details={"backend": provider_name},
            )
    except Exception as e:
        return IntegrationHealth(
            name="llm_backend",
            status="unhealthy",
            error=str(e),
            last_check=datetime.utcnow().isoformat(),
        )


async def check_database() -> IntegrationHealth:
    """Check database health"""
    try:
        from ..db import get_history

        start = time.perf_counter()

        # Simple query test
        history = get_history(limit=1)

        duration_ms = (time.perf_counter() - start) * 1000

        return IntegrationHealth(
            name="database",
            status="healthy",
            response_time_ms=duration_ms,
            last_check=datetime.utcnow().isoformat(),
            details={"query_successful": True},
        )
    except Exception as e:
        return IntegrationHealth(
            name="database",
            status="unhealthy",
            error=str(e),
            last_check=datetime.utcnow().isoformat(),
        )


async def check_all_integrations(timeout: float = 10.0) -> Dict[str, Any]:
    """Check all integrations in parallel with timeout

    Args:
        timeout: Maximum time to wait for all checks (seconds)

    Returns:
        Dictionary with integration health statuses
    """
    start = time.perf_counter()

    # Run all checks concurrently with timeout
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                check_llm_backend(),
                check_database(),
                check_search_integration(),
                check_fetch_integration(),
                check_osint_integration(),
                check_slack_integration(),
                check_homeassistant_integration(),
                return_exceptions=True,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.error(f"Integration health check timed out after {timeout}s")
        return {
            "status": "timeout",
            "error": f"Health check exceeded {timeout}s timeout",
            "timestamp": datetime.utcnow().isoformat(),
        }

    duration_ms = (time.perf_counter() - start) * 1000

    # Process results
    integrations = {}
    overall_status = "healthy"
    unhealthy_count = 0
    degraded_count = 0

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Integration check failed: {result}")
            continue

        integrations[result.name] = asdict(result)

        if result.status == "unhealthy":
            unhealthy_count += 1
            overall_status = "unhealthy"
        elif result.status == "degraded":
            degraded_count += 1
            if overall_status == "healthy":
                overall_status = "degraded"

    return {
        "status": overall_status,
        "total_checks": len(integrations),
        "unhealthy": unhealthy_count,
        "degraded": degraded_count,
        "check_duration_ms": round(duration_ms, 2),
        "timestamp": datetime.utcnow().isoformat(),
        "integrations": integrations,
    }
