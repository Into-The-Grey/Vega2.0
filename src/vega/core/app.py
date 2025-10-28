#!/usr/bin/env python3
"""
FastAPI application for Vega2.0 - Clean Implementation
"""

from typing import Optional, Dict, Any
import uuid
from datetime import datetime
import asyncio
import logging
import re

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

try:
    import sys
    from pathlib import Path

    tools_path = Path(__file__).parent.parent.parent.parent / "tools"
    sys.path.insert(0, str(tools_path))
    from utils.commands_generator import (
        generate_commands_markdown,
        generate_commands_html,
    )
except Exception:
    # Defer import errors to endpoint call time
    generate_commands_markdown = None  # type: ignore
    generate_commands_html = None  # type: ignore

# Background process management
try:
    from .process_manager import (
        get_process_manager,
        start_background_processes,
        stop_background_processes,
    )

    PROCESS_MANAGEMENT_AVAILABLE = True
except ImportError:
    PROCESS_MANAGEMENT_AVAILABLE = False

# Error handling system
try:
    from .error_handler import get_error_handler, log_info, log_error
    from .error_middleware import setup_error_middleware
    from .exceptions import MissingAPIKeyError, InvalidAPIKeyError, ValidationError

    ERROR_HANDLING_AVAILABLE = True
except ImportError:
    ERROR_HANDLING_AVAILABLE = False

# ECC cryptography system
try:
    from .ecc_crypto import get_ecc_manager, ECCCurve
    from .api_security import (
        get_security_manager,
        verify_ecc_api_key,
        require_permission,
    )

    ECC_AVAILABLE = True
except ImportError:
    ECC_AVAILABLE = False

# Initialize app
app = FastAPI(title="Vega2.0", version="2.0.0")

# Mount static files - use path relative to project root
import os
from pathlib import Path

static_dir = Path(__file__).parent.parent.parent.parent / "tools" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
else:
    # Create a minimal static directory if it doesn't exist
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Setup error handling middleware
if ERROR_HANDLING_AVAILABLE:
    setup_error_middleware(app)

# Add correlation ID middleware for distributed tracing
try:
    from .correlation import CorrelationIdMiddleware

    app.add_middleware(CorrelationIdMiddleware)
    CORRELATION_AVAILABLE = True
except ImportError:
    CORRELATION_AVAILABLE = False

# Lightweight compression for larger responses
try:
    # Only enable if available; safe default minimum size
    app.add_middleware(GZipMiddleware, minimum_size=500)
except Exception:
    pass

# Collaboration system integration
try:
    from ..collaboration.integration import integrate_with_main_app

    integrate_with_main_app(app)
    print("✅ Collaboration features integrated")
except ImportError as e:
    print(f"⚠️ Collaboration features not available: {e}")

# Analytics system integration
try:
    from ..analytics.collector import analytics_collector
    from ..analytics.engine import analytics_engine
    from ..analytics.visualization import create_visualization_router

    # Add analytics router
    analytics_router = create_visualization_router(
        analytics_collector, analytics_engine
    )
    app.include_router(analytics_router)
    print("✅ Analytics system integrated")
except ImportError as e:
    print(f"⚠️ Analytics system not available: {e}")

# Document intelligence integration
try:
    from ..document.api import router as document_router

    app.include_router(document_router)
    print("✅ Document intelligence system integrated")
except ImportError as e:
    print(f"⚠️ Document intelligence system not available: {e}")

# Advanced performance monitoring endpoints
try:
    from .performance_endpoints import router as performance_router

    app.include_router(performance_router)
    print("✅ Advanced performance monitoring integrated")
except ImportError as e:
    print(f"⚠️ Performance monitoring not available: {e}")

# Metrics
app.state.metrics = {
    "requests_total": 0,
    "responses_total": 0,
    "errors_total": 0,
    "degraded": False,
    # Request timing metrics (ms)
    "request_duration_ms_sum": 0.0,
    "request_duration_count": 0,
    "last_request_duration_ms": 0.0,
    # HTTP status code counts
    "status_codes": {},
}


# Lightweight extraction metrics (in-process)
_metrics = {
    "extraction_calls": 0,
    "extraction_facts_total": 0,
}


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize resources and start background processes on app startup"""

    print("=" * 80)
    print("🚀 Vega2.0 Startup Sequence")
    print("=" * 80)

    # 1. Validate configuration first (fail fast)
    try:
        from .config_validator import validate_startup_config

        print("\n📋 Validating configuration...")
        is_valid = validate_startup_config()

        if not is_valid:
            print("\n❌ CRITICAL: Configuration validation failed!")
            print("   Please fix configuration errors before starting the server.")
            print("   See error messages above for details.\n")
            # Allow server to start but log warnings
        else:
            print("✅ Configuration validation passed\n")
    except Exception as e:
        logger.error(f"Configuration validation error: {e}")
        print(f"⚠️  Configuration validation error: {e}\n")

    # 2. Configure correlation ID logging for distributed tracing
    if CORRELATION_AVAILABLE:
        try:
            from .correlation import configure_correlation_logging

            configure_correlation_logging()
            logger.info("✓ Correlation ID tracing enabled")
            print("✅ Distributed tracing enabled (correlation IDs)")
        except Exception as e:
            logger.warning(f"Failed to configure correlation logging: {e}")

    # 3. Initialize resource manager
    try:
        from .resource_manager import get_resource_manager  # type: ignore

        manager = await get_resource_manager()
        logger.info("Resource manager initialized")
        print("✅ Resource manager initialized")
    except Exception as e:
        logger.warning(f"Resource manager init failed: {e}")
        print(f"⚠️  Resource manager init failed: {e}")

    # 4. Initialize database profiler
    try:
        from .db_profiler import get_profiler

        profiler = get_profiler()
        profiler.enabled = True
        profiler.set_slow_query_threshold(100.0)  # 100ms threshold
        logger.info("Database profiler initialized")
        print("✅ Database query profiler enabled (100ms slow query threshold)")
    except Exception as e:
        logger.warning(f"Database profiler init failed: {e}")
        print(f"⚠️  Database profiler init failed: {e}")

    # 5. Background process management
    if PROCESS_MANAGEMENT_AVAILABLE:
        try:
            # Start background processes (optional, can be managed separately)
            # await start_background_processes()
            print("✅ Background process management available")
        except Exception as e:
            print(f"⚠️  Could not start background processes: {e}")

    # 6. Optional retention purge on startup
    try:
        from .config import get_config as _get_cfg  # type: ignore
        from .db import purge_old, vacuum_db  # type: ignore

        _cfg = _get_cfg()
        days = int(getattr(_cfg, "retention_days", 0) or 0)
        if days > 0:
            deleted = purge_old(days)
            # Vacuum only if we actually deleted rows to reclaim space
            if deleted > 0:
                try:
                    vacuum_db()
                except Exception:
                    pass
            logger.info(
                "Retention purge completed", extra={"days": days, "deleted": deleted}
            )
    except Exception:
        # Config/DB may not be available in some environments (e.g., tests)
        pass


@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully shutdown all resources and background processes"""
    # Shutdown LLM resources
    try:
        from .llm import llm_shutdown  # type: ignore

        await llm_shutdown()
        logger.info("LLM resources closed")
    except Exception as e:
        logger.warning(f"LLM shutdown failed: {e}")

    # Shutdown resource manager
    try:
        from .resource_manager import get_resource_manager_sync  # type: ignore

        manager = get_resource_manager_sync()
        if manager:
            await manager.shutdown()
            logger.info("Resource manager shutdown complete")
    except Exception as e:
        logger.warning(f"Resource manager shutdown failed: {e}")

    if PROCESS_MANAGEMENT_AVAILABLE:
        try:
            await stop_background_processes()
            print("Background processes stopped")
        except Exception as e:
            print(f"Warning: Error stopping background processes: {e}")


# Models
class ChatRequest(BaseModel):
    prompt: str
    stream: bool = False
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


# Health endpoints
@app.get("/healthz")
async def healthz():
    return {"ok": True, "timestamp": datetime.utcnow().isoformat()}


@app.get("/livez")
async def livez():
    """Liveness probe (legacy endpoint expected by tests)."""
    return {"alive": True}


@app.get("/readyz")
async def readyz():
    """Readiness probe. Uses get_history as a lightweight health check (patched in tests)."""
    try:
        # If get_history raises, consider the service unhealthy
        _ = get_history()
        return {"ready": True}
    except Exception:
        # Return an HTTP 503 to indicate readiness failure
        raise HTTPException(status_code=503, detail="Service not ready")


# --- Minimal config/auth and db/llm shims (for tests to patch) ---
class _Cfg:
    api_key: str = "vega-default-key"
    api_keys_extra: list[str] = []


cfg = _Cfg()

# Try to align API keys with environment config at import time
try:
    from .config import get_config as _get_cfg  # type: ignore

    _env_cfg = _get_cfg()
    # Update local cfg used by require_api_key with real values
    cfg.api_key = getattr(_env_cfg, "api_key", cfg.api_key)
    extra = getattr(_env_cfg, "api_keys_extra", ())
    if isinstance(extra, (list, tuple)):
        cfg.api_keys_extra = list(extra)
except Exception:
    # Fall back to defaults if config is unavailable at import time
    pass


def require_api_key(x_api_key: str | None):
    if not x_api_key:
        if ERROR_HANDLING_AVAILABLE:
            raise MissingAPIKeyError()
        else:
            raise HTTPException(status_code=401, detail="Missing API key")
    allowed = {cfg.api_key, *getattr(cfg, "api_keys_extra", [])}
    if x_api_key not in allowed:
        if ERROR_HANDLING_AVAILABLE:
            raise InvalidAPIKeyError()
        else:
            raise HTTPException(status_code=401, detail="Invalid API key")


# Placeholders that tests patch
# By default, we bind these to real implementations if available; tests can patch them.


def query_llm(prompt: str, stream: bool = False, **kwargs):  # patched in tests
    # Default no-op; will be rebound to real LLM at import time if available
    return f"Echo: {prompt}"


def log_conversation(
    prompt: str, response: str, session_id: str | None = None
):  # patched
    # Default no-op; will be rebound to real DB logger at import time if available
    return None


# Bind to real implementations when possible (still patchable in tests)
try:
    from .llm import query_llm as _real_query_llm  # type: ignore

    query_llm = _real_query_llm  # type: ignore
except Exception:
    pass

try:
    from .db import log_conversation as _real_log_conversation  # type: ignore

    def log_conversation(prompt: str, response: str, session_id: str | None = None):
        return _real_log_conversation(prompt, response, session_id)  # type: ignore

except Exception:
    pass


def get_history(limit: int = 50):  # patched in tests and used by readiness
    return []


def get_session_history(session_id: str, limit: int = 50):  # patched
    return []


def set_feedback(
    conversation_id: int, rating: int, comment: str | None = None
):  # patched
    return None


from fastapi import Response
from fastapi.responses import StreamingResponse
from .logging_setup import VegaLogger  # exposed for patching in tests

# VegaLogger already defined above, no need to re-assign
from .config_manager import config_manager  # exposed for patching in tests


@app.get("/docs/commands.md")
async def docs_commands_md():
    # Lazy import if earlier failed
    global generate_commands_markdown
    if generate_commands_markdown is None:
        from ...tools.utils.commands_generator import generate_commands_markdown as _gen_md  # type: ignore

        generate_commands_markdown = _gen_md
    md = generate_commands_markdown()  # type: ignore
    return Response(content=md, media_type="text/markdown")


@app.get("/docs/commands")
async def docs_commands_html():
    global generate_commands_html
    if generate_commands_html is None:
        from ...tools.utils.commands_generator import generate_commands_html as _gen_html  # type: ignore

        generate_commands_html = _gen_html
    html = generate_commands_html()  # type: ignore
    return HTMLResponse(content=html, status_code=200)


# Liveness and Readiness
@app.get("/metrics")
def metrics_endpoint():
    """Return application metrics as JSON and include lightweight extraction metrics."""
    payload = {}
    # core app metrics stored on app.state.metrics
    try:
        payload.update(app.state.metrics or {})
    except Exception:
        payload.update({})

    # include extraction metrics
    try:
        payload["extraction_calls"] = _metrics.get("extraction_calls", 0)
        payload["extraction_facts_total"] = _metrics.get("extraction_facts_total", 0)
    except Exception:
        pass

    # add derived averages where possible
    try:
        cnt = float(payload.get("request_duration_count", 0))
        total = float(payload.get("request_duration_ms_sum", 0.0))
        payload["avg_request_duration_ms"] = (total / cnt) if cnt > 0 else 0.0
    except Exception:
        payload["avg_request_duration_ms"] = 0.0

    # count global stored facts as an approximation
    try:
        from src.vega.core.db import get_memory_facts

        facts = get_memory_facts(None)
        payload["memory_facts_global_total"] = len(facts)
    except Exception:
        payload["memory_facts_global_total"] = 0

    return payload


@app.get("/metrics/prometheus")
def metrics_prometheus():
    """Expose Prometheus-formatted metrics for scraping."""
    lines = []
    # Core counters
    try:
        m = app.state.metrics or {}
    except Exception:
        m = {}
    lines.append("# HELP vega_requests_total Total number of HTTP requests")
    lines.append(f"vega_requests_total {int(m.get('requests_total', 0))}")
    lines.append("# HELP vega_responses_total Total number of HTTP responses")
    lines.append(f"vega_responses_total {int(m.get('responses_total', 0))}")
    lines.append("# HELP vega_errors_total Total number of errors")
    lines.append(f"vega_errors_total {int(m.get('errors_total', 0))}")
    # Duration summary (sum and count)
    lines.append(
        "# HELP vega_request_duration_ms_sum Cumulative request duration in ms"
    )
    lines.append(
        f"vega_request_duration_ms_sum {float(m.get('request_duration_ms_sum', 0.0))}"
    )
    lines.append("# HELP vega_request_duration_ms_count Number of measured requests")
    lines.append(
        f"vega_request_duration_ms_count {int(m.get('request_duration_count', 0))}"
    )
    # Status code distribution
    codes = m.get("status_codes", {}) if isinstance(m, dict) else {}
    if isinstance(codes, dict):
        lines.append("# HELP vega_http_responses_total HTTP responses by status code")
        for code, count in codes.items():
            try:
                lines.append(f'vega_http_responses_total{{code="{code}"}} {int(count)}')
            except Exception:
                continue
    lines.append(
        "# HELP vega_memory_extraction_calls Total number of extraction attempts"
    )
    lines.append(f"vega_memory_extraction_calls {_metrics.get('extraction_calls', 0)}")
    lines.append(
        "# HELP vega_memory_extraction_facts_total Total number of extracted facts"
    )
    lines.append(
        f"vega_memory_extraction_facts_total {_metrics.get('extraction_facts_total', 0)}"
    )
    try:
        from src.vega.core.db import get_memory_facts

        facts = get_memory_facts(None)
        total_stored = len(facts)
    except Exception:
        total_stored = 0
    lines.append(
        "# HELP vega_memory_facts_stored_total Total stored memory facts (global)"
    )
    lines.append(f"vega_memory_facts_stored_total {total_stored}")
    return PlainTextResponse("\n".join(lines), media_type="text/plain; version=0.0.4")


def _sanitize_string(s: str) -> str:
    """Sanitize a string by removing invalid UTF-8 sequences and ensuring a str return.

    This is defensive: some extreme stress tests inject surrogate or invalid bytes.
    """
    try:
        if s is None:
            return ""
        if not isinstance(s, str):
            s = str(s)
        return s.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _extract_memory_facts(text: str) -> dict[str, str]:
    facts: dict[str, str] = {}
    # Track extraction attempts (best-effort, in-memory)
    try:
        _metrics["extraction_calls"] += 1
    except Exception:
        pass
    try:
        # Sanitize input text first
        text = _sanitize_string(text)
        # Name patterns:
        # 1. "my name is [Name]"
        # 2. "I'm [Name]" (contraction)
        # Strip common titles: Dr., Mr., Ms., Mrs., Prof., etc.

        # Pattern 1: "my name is ..."
        m = re.search(
            r"(?i:\bmy name is\s+)(?:Dr\.|Mr\.|Ms\.|Mrs\.|Prof\.|Sir\.|Madam\s+)?\s*(?-i:([A-Z][A-Za-z\-'`]+(?:\s+[A-Z][A-Za-z\-'`]+)*))(?=\s+and|[,\.!?]|\s*$)",
            text,
        )
        if m:
            name = m.group(1).strip()
            facts["user_name"] = name

        # Pattern 2: "I'm [Name]" (contraction)
        if "user_name" not in facts:
            m = re.search(
                r"(?i:\bI'm\s+)(?:Dr\.|Mr\.|Ms\.|Mrs\.|Prof\.|Sir\.|Madam\s+)?\s*(?-i:([A-Z][A-Za-z\-'`]+(?:\s+[A-Z][A-Za-z\-'`]+)*))(?=\s+and|[,\.!?]|\s*$)",
                text,
            )
            if m:
                name = m.group(1).strip()
                facts["user_name"] = name

        # Pattern 3: "Call me [Name]"
        if "user_name" not in facts:
            m = re.search(
                r"(?i:\bcall me\s+)(?:Dr\.|Mr\.|Ms\.|Mrs\.|Prof\.|Sir\.|Madam\s+)?\s*(?-i:([A-Z][A-Za-z\-'`]+(?:\s+[A-Z][A-Za-z\-'`]+)*))(?=\s+and|[,\.!?]|\s*$)",
                text,
            )
            if m:
                name = m.group(1).strip()
                facts["user_name"] = name

        # Location patterns:
        # 1. "I live in [Location]"
        # 2. "I'm living in [Location]" / "living in [Location]"

        # Pattern 1: "I live in ..."
        m = re.search(
            r"\bi live in\s+([A-Z][A-Za-z][\w\s,\-]*)(?=\s+and|[,\.!?]|\s*$)",
            text,
            re.I,
        )
        if m:
            loc = m.group(1).strip().rstrip(", .!?")
            facts["user_location"] = loc

        # Pattern 2: "living in ..."
        if "user_location" not in facts:
            m = re.search(
                r"\bliving in\s+([A-Z][A-Za-z][\w\s,\-]*)(?=\s+and|[,\.!?]|\s*$)",
                text,
                re.I,
            )
            if m:
                loc = m.group(1).strip().rstrip(", .!?")
                facts["user_location"] = loc

        # Pattern 3: "based in ..." and "I'm based in ..."
        if "user_location" not in facts:
            m = re.search(
                r"\b(?:I'm\s+based in|based in)\s+([A-Z][A-Za-z][\w\s,\-]*)(?=\s+and|[,\.!?]|\s*$)",
                text,
                re.I,
            )
            if m:
                loc = m.group(1).strip().rstrip(", .!?")
                facts["user_location"] = loc

        # Timezone: "my timezone is ..."
        m = re.search(r"\bmy time ?zone is\s+([A-Za-z_\/+\-0-9]{1,64})", text, re.I)
        if m:
            tz = m.group(1).strip().rstrip(". !?")
            facts["user_timezone"] = tz
    except Exception:
        pass
    return facts


# Chat endpoint with persistent conversation memory
@app.post("/chat")
async def chat(
    request: ChatRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    app.state.metrics["requests_total"] += 1
    require_api_key(x_api_key)

    try:
        from .llm import LLMBackendError  # type: ignore
        from .db import (
            get_persistent_session_id,
            get_recent_context,
            get_conversation_summary,
            set_memory_fact,
            get_memory_facts,
        )  # type: ignore
        from .config import get_config as get_cfg  # type: ignore
    except Exception:

        class LLMBackendError(Exception):
            pass

        # Fallback implementations if imports fail

        def get_persistent_session_id():
            return str(uuid.uuid4())

        def get_recent_context(session_id=None, limit=10, max_chars=4000):
            return []

        def get_conversation_summary(
            session_id=None, older_than_id=None, max_entries=100
        ):
            return ""

        def set_memory_fact(session_id, key, value):
            return None

        def get_memory_facts(session_id=None):
            return {}

        def get_cfg():
            class _Cfg:
                context_window_size = 10
                context_max_chars = 4000

            return _Cfg()

    try:
        # Get or create persistent session ID
        # If user provided session_id, use it; otherwise get the persistent one
        if request.session_id:
            session_id = request.session_id
        else:
            session_id = get_persistent_session_id()

        # Load conversation context for continuity
        cfg = get_cfg()
        context_window = getattr(cfg, "context_window_size", 10)
        context_max = getattr(cfg, "context_max_chars", 4000)

        conversation_context = get_recent_context(
            session_id=session_id, limit=context_window, max_chars=context_max
        )

        # Extract and persist memory facts from the current prompt
        try:
            extracted = _extract_memory_facts(request.prompt)
            for k, v in extracted.items():
                set_memory_fact(session_id, k, v)
        except Exception:
            extracted = {}

        # Build system context from stored facts and a compact summary of older history
        try:
            facts = get_memory_facts(session_id)
        except Exception:
            facts = {}

        try:
            older_than_id = (
                conversation_context[0]["id"] if conversation_context else None
            )
        except Exception:
            older_than_id = None

        try:
            summary = get_conversation_summary(
                session_id=session_id, older_than_id=older_than_id, max_entries=50
            )
        except Exception:
            summary = ""

        system_parts = []
        if facts:
            mem_lines = ["[Memory Facts]"]
            for k, v in facts.items():
                mem_lines.append(f"- {k}: {v}")
            system_parts.append("\n".join(mem_lines))
        if summary:
            system_parts.append(summary)
        system_context = "\n\n".join(system_parts) if system_parts else None

        # If streaming requested, return a StreamingResponse
        if request.stream:
            # Call patchable query_llm (may return coroutine, async generator, or string)
            res = query_llm(
                request.prompt,
                stream=True,
                conversation_context=conversation_context,
                system_context=system_context,
            )

            if asyncio.iscoroutine(res):
                token_stream = await res  # type: ignore
            elif hasattr(res, "__aiter__"):
                token_stream = res  # type: ignore
            else:

                async def _single_yield():
                    yield str(res)

                token_stream = _single_yield()

            async def _stream_and_log():
                """Wrap the token stream to accumulate and log on completion."""
                buffer = []
                try:
                    async for chunk in token_stream:  # type: ignore
                        buffer.append(chunk)
                        yield chunk
                finally:
                    try:
                        full = "".join(buffer)
                        log_conversation(request.prompt, full, session_id)
                    except Exception:
                        pass

            app.state.metrics["responses_total"] += 1
            return StreamingResponse(_stream_and_log(), media_type="text/plain")

        # Non-streaming: Query LLM and return JSON
        res = query_llm(
            request.prompt,
            stream=False,
            conversation_context=conversation_context,
            system_context=system_context,
        )
        response_text = await res if asyncio.iscoroutine(res) else res  # type: ignore

        # Log this exchange
        try:
            log_conversation(request.prompt, response_text, session_id)
        except Exception:
            pass

        app.state.metrics["responses_total"] += 1
        return {
            "response": response_text,
            "session_id": session_id,
            "context_used": len(conversation_context),
        }
    except LLMBackendError as e:
        app.state.metrics["errors_total"] += 1
        raise HTTPException(status_code=503, detail="LLM backend unavailable") from e


# Home Assistant Integration endpoints
class HAWebhookPayload(BaseModel):
    """Webhook payload from Home Assistant voice commands"""

    text: str  # Transcribed speech text
    conversation_id: Optional[str] = None
    device_id: Optional[str] = None
    device_type: Optional[str] = (
        "unknown"  # ios, macos, windows, browser, smart_speaker
    )
    user_id: Optional[str] = None
    language: str = "en"


class HAVoiceResponse(BaseModel):
    """Response to send back to Home Assistant"""

    success: bool
    response: str
    tts_sent: bool = False
    session_id: Optional[str] = None


@app.post("/hass/webhook", response_model=HAVoiceResponse)
async def hass_webhook(
    payload: HAWebhookPayload,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """
    Webhook endpoint for Home Assistant voice commands

    Receives transcribed speech from HA Assist and returns response.
    Automatically sends TTS response back to Home Assistant if configured.

    Flow:
        1. HA Assist: "Hey Vega, what's the weather?"
        2. HA → POST /hass/webhook with transcribed text
        3. Vega → LLM processing with conversation context
        4. Vega → HA TTS service (send response to user's device)
        5. Return JSON response with success status
    """
    require_api_key(x_api_key)

    try:
        from .config import get_config as get_cfg
        from ..integrations.homeassistant import (
            HomeAssistantClient,
            VegaHomeAssistantBridge,
            parse_ha_webhook_payload,
            HAVoiceContext,
        )
    except ImportError:
        # Fallback if HA integration not available
        logger.warning("Home Assistant integration not available")
        return HAVoiceResponse(
            success=False,
            response="Home Assistant integration not configured",
            tts_sent=False,
        )

    cfg = get_cfg()

    # Check if HA integration is enabled
    if not getattr(cfg, "hass_enabled", False):
        return HAVoiceResponse(
            success=False,
            response="Home Assistant integration is disabled",
            tts_sent=False,
        )

    try:
        # Import HAVoiceDevice enum for type conversion
        from ..integrations.homeassistant import HAVoiceDevice

        # Convert device_type string to enum
        try:
            device_type_enum = (
                HAVoiceDevice(payload.device_type)
                if payload.device_type
                else HAVoiceDevice.UNKNOWN
            )
        except ValueError:
            device_type_enum = HAVoiceDevice.UNKNOWN

        # Parse webhook payload into context
        context = HAVoiceContext(
            text=payload.text,
            conversation_id=payload.conversation_id,
            device_id=payload.device_id,
            device_type=device_type_enum,
            user_id=payload.user_id,
            language=payload.language,
        )

        # Create async chat callback
        async def vega_chat_callback(prompt: str, session_id: Optional[str] = None):
            """Call Vega's chat endpoint internally"""
            from .llm import query_llm as llm_query
            from .db import get_persistent_session_id, get_recent_context

            if not session_id:
                session_id = get_persistent_session_id()

            # Load conversation context
            conversation_context = get_recent_context(
                session_id=session_id,
                limit=getattr(cfg, "context_window_size", 10),
                max_chars=getattr(cfg, "context_max_chars", 4000),
            )

            # Query LLM
            response_text = await llm_query(
                prompt, stream=False, conversation_context=conversation_context
            )

            # Log conversation
            log_conversation(prompt, response_text, session_id)

            return {"response": response_text, "session_id": session_id}

        # Process voice command
        hass_url = getattr(cfg, "hass_url", None)
        hass_token = getattr(cfg, "hass_token", None)

        if not hass_url or not hass_token:
            # Process without TTS response
            result = await vega_chat_callback(context.text)
            return HAVoiceResponse(
                success=True,
                response=result["response"],
                tts_sent=False,
                session_id=result["session_id"],
            )

        # Full integration with TTS response
        async with HomeAssistantClient(hass_url, hass_token) as ha_client:
            bridge = VegaHomeAssistantBridge(ha_client, vega_chat_callback)

            # Process command and get response
            ha_response = await bridge.process_voice_command(context)

            # Send TTS response back to HA
            tts_success = await bridge.send_response(ha_response)

            return HAVoiceResponse(
                success=True,
                response=ha_response.text,
                tts_sent=tts_success,
                session_id=ha_response.conversation_id,
            )

    except Exception as e:
        logger.error(f"Error processing HA webhook: {e}")
        return HAVoiceResponse(
            success=False, response=f"Error: {str(e)}", tts_sent=False
        )


@app.get("/hass/status")
async def hass_status(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Check Home Assistant integration status"""
    require_api_key(x_api_key)

    try:
        from .config import get_config as get_cfg
        from ..integrations.homeassistant import test_ha_connection
    except ImportError:
        return {
            "enabled": False,
            "available": False,
            "message": "Home Assistant integration module not found",
        }

    cfg = get_cfg()
    enabled = getattr(cfg, "hass_enabled", False)
    hass_url = getattr(cfg, "hass_url", None)
    hass_token = getattr(cfg, "hass_token", None)

    if not enabled:
        return {
            "enabled": False,
            "configured": bool(hass_url and hass_token),
            "message": "Home Assistant integration is disabled in config",
        }

    if not hass_url or not hass_token:
        return {
            "enabled": True,
            "configured": False,
            "message": "Home Assistant URL or token not configured",
        }

    # Test connection
    try:
        connected = await test_ha_connection(hass_url, hass_token)
        return {
            "enabled": True,
            "configured": True,
            "connected": connected,
            "url": hass_url,
            "webhook_endpoint": "/hass/webhook",
            "message": (
                "Home Assistant integration operational"
                if connected
                else "Cannot connect to Home Assistant"
            ),
        }
    except Exception as e:
        return {
            "enabled": True,
            "configured": True,
            "connected": False,
            "error": str(e),
            "message": "Error testing Home Assistant connection",
        }


# Proactive conversation models
class ProposeRequest(BaseModel):
    max_per_day: int = 5


class ProposeResponse(BaseModel):
    id: str | None
    session_id: str | None
    topic: str | None
    reason: str | None
    message: str | None


class SessionMessage(BaseModel):
    session_id: str
    text: str


# Proactive conversation endpoints (all require API key)
@app.post("/proactive/propose", response_model=ProposeResponse)
async def proactive_propose(
    req: ProposeRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    if not x_api_key or x_api_key != "vega-default-key":
        raise HTTPException(status_code=401, detail="Invalid API key") from None
    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.proactive_conversation import propose_initiation

        prop = propose_initiation(max_per_day=req.max_per_day)
        if not prop:
            return ProposeResponse(
                id=None, session_id=None, topic=None, reason=None, message=None
            )
        return ProposeResponse(
            id=prop.id,
            session_id=prop.session_id,
            topic=prop.topic,
            reason=prop.reason,
            message=prop.message,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/proactive/pending")
async def proactive_pending(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    if not x_api_key or x_api_key != "vega-default-key":
        raise HTTPException(status_code=401, detail="Invalid API key") from None
    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.proactive_conversation import list_pending

        return {"pending": list_pending()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/proactive/accept")
async def proactive_accept(
    proposed_id: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    if not x_api_key or x_api_key != "vega-default-key":
        raise HTTPException(status_code=401, detail="Invalid API key") from None
    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.proactive_conversation import accept_initiation

        sid = accept_initiation(proposed_id)
        if not sid:
            raise HTTPException(status_code=404, detail="Proposed initiation not found")
        return {"session_id": sid}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/proactive/decline")
async def proactive_decline(
    proposed_id: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    if not x_api_key or x_api_key != "vega-default-key":
        raise HTTPException(status_code=401, detail="Invalid API key") from None
    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.proactive_conversation import decline_initiation

        ok = decline_initiation(proposed_id)
        return {"ok": bool(ok)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/proactive/send")
async def proactive_send(
    body: SessionMessage,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    if not x_api_key or x_api_key != "vega-default-key":
        raise HTTPException(status_code=401, detail="Invalid API key") from None
    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.proactive_conversation import send_in_session

        # Support both async and sync implementations for tests
        result = send_in_session(body.session_id, body.text)
        if hasattr(result, "__await__"):
            res = await result  # type: ignore
        else:
            res = result
        return res
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/proactive/session/{session_id}")
async def proactive_get_session(
    session_id: str, x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    if not x_api_key or x_api_key != "vega-default-key":
        raise HTTPException(status_code=401, detail="Invalid API key") from None
    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.proactive_conversation import get_session

        s = get_session(session_id)
        if not s:
            raise HTTPException(status_code=404, detail="Session not found")
        return s
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# History endpoints
@app.get("/history")
async def history(
    limit: int = Query(50, ge=1, le=1000),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    require_api_key(x_api_key)
    items = get_history(limit=limit)
    return {"history": items}


@app.get("/session/{session_id}")
async def session_history(
    session_id: str,
    limit: int = Query(50, ge=1, le=1000),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    require_api_key(x_api_key)
    items = get_session_history(session_id, limit=limit)
    return {"session_id": session_id, "history": items}


class FeedbackRequest(BaseModel):
    conversation_id: int
    rating: int
    comment: Optional[str] = None


@app.post("/feedback")
async def submit_feedback(
    body: FeedbackRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    require_api_key(x_api_key)
    set_feedback(body.conversation_id, body.rating, body.comment)
    return {"status": "success"}


@app.post("/proactive/end")
async def proactive_end(
    session_id: str, x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    if not x_api_key or x_api_key != "vega-default-key":
        raise HTTPException(status_code=401, detail="Invalid API key") from None
    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.proactive_conversation import end_session

        ok = end_session(session_id)
        return {"ok": bool(ok)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# Admin endpoints
@app.get("/admin/logs")
async def admin_logs_list(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    require_api_key(x_api_key)
    from .logging_setup import VegaLogger

    return {
        "modules": (
            VegaLogger.list_modules() if hasattr(VegaLogger, "list_modules") else []
        )
    }


@app.get("/admin/logs/{module}")
async def admin_logs_tail(
    module: str,
    lines: int = 50,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    require_api_key(x_api_key)
    from .logging_setup import VegaLogger

    lines_list = VegaLogger.tail_log(module, lines)
    return {"module": module, "lines": lines_list, "total_lines": len(lines_list)}


@app.get("/admin/config")
async def admin_config_list(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    require_api_key(x_api_key)
    from .config_manager import config_manager as config_manager

    return {
        "modules": (
            config_manager.list_modules()
            if hasattr(config_manager, "list_modules")
            else []
        )
    }


@app.get("/admin/config/{module}")
async def admin_config_get(
    module: str, x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    require_api_key(x_api_key)
    # Use function to allow tests to patch
    cfg_dict = config_manager.get_config(module)
    return {"module": module, "config": cfg_dict}


@app.get("/admin/resources/health")
async def admin_resources_health(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    """Get resource manager health status"""
    require_api_key(x_api_key)

    try:
        from .resource_manager import get_resource_manager

        manager = await get_resource_manager()
        health = await manager.health_check()

        return {
            "status": "healthy" if health["healthy"] else "degraded",
            "checks": health,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        logger.error(f"Failed to get resource health: {e}")
        return {
            "status": "unavailable",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }


@app.get("/admin/resources/stats")
async def admin_resources_stats(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    """Get resource manager statistics and metrics"""
    require_api_key(x_api_key)

    stats = {}

    # Get resource manager stats
    try:
        from .resource_manager import get_resource_manager

        manager = await get_resource_manager()
        resource_stats = manager.get_stats()

        stats["resource_manager"] = {
            "http_clients_created": resource_stats.http_clients_created,
            "http_requests_made": resource_stats.http_requests_made,
            "config_cache_hits": resource_stats.config_cache_hits,
            "config_cache_misses": resource_stats.config_cache_misses,
            "config_cache_hit_rate": (
                round(
                    resource_stats.config_cache_hits
                    / (
                        resource_stats.config_cache_hits
                        + resource_stats.config_cache_misses
                    )
                    * 100,
                    2,
                )
                if (
                    resource_stats.config_cache_hits
                    + resource_stats.config_cache_misses
                )
                > 0
                else 0
            ),
            "cleanup_tasks_registered": resource_stats.cleanup_tasks_registered,
            "startup_time": resource_stats.startup_time,
            "shutdown_time": resource_stats.shutdown_time,
            "uptime_seconds": (
                (datetime.utcnow().timestamp() - resource_stats.startup_time)
                if resource_stats.startup_time
                else None
            ),
        }
    except Exception as e:
        logger.error(f"Failed to get resource manager stats: {e}")
        stats["resource_manager"] = {"error": str(e)}

    # Get config cache stats
    try:
        from .config_cache import get_cache_stats

        cache_stats = get_cache_stats()

        stats["config_cache"] = {
            "hits": cache_stats["hits"],
            "misses": cache_stats["misses"],
            "hit_rate": cache_stats["hit_rate"],
            "cache_age_seconds": cache_stats["cache_age"],
            "cached": cache_stats["hits"] > 0,
        }
    except Exception as e:
        logger.error(f"Failed to get config cache stats: {e}")
        stats["config_cache"] = {"error": str(e)}

    stats["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return stats


@app.post("/admin/resources/cache/invalidate")
async def admin_resources_invalidate_cache(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    """Invalidate configuration cache to force reload"""
    require_api_key(x_api_key)

    try:
        from .config_cache import invalidate_config_cache

        invalidate_config_cache()

        return {
            "status": "success",
            "message": "Configuration cache invalidated successfully",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        logger.error(f"Failed to invalidate config cache: {e}")
        raise HTTPException(status_code=500, detail=f"Cache invalidation failed: {e}")


@app.get("/admin/resources/pools")
async def admin_resources_pools(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    """
    Get detailed HTTP connection pool metrics.

    Provides connection pool state, active connections, and usage statistics.
    Useful for monitoring pool exhaustion and troubleshooting performance.
    """
    require_api_key(x_api_key)

    try:
        from .resource_manager import get_resource_manager

        manager = await get_resource_manager()
        pool_metrics = manager.get_pool_metrics()

        # Add timestamp
        pool_metrics["timestamp"] = datetime.utcnow().isoformat() + "Z"

        return pool_metrics
    except Exception as e:
        logger.error(f"Failed to get connection pool metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Pool metrics unavailable: {e}")


@app.get("/admin/llm/behavior")
async def admin_llm_behavior_get(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    require_api_key(x_api_key)
    behavior = config_manager.get_llm_behavior()
    return {"behavior": behavior}


@app.put("/admin/config/{module}")
async def admin_config_update(
    module: str,
    payload: Dict[str, Any],
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    require_api_key(x_api_key)
    cfg_update = payload.get("config", {}) if isinstance(payload, dict) else {}
    config_manager.update_config(module, cfg_update)
    return {"status": "success"}


@app.put("/admin/llm/behavior")
async def admin_llm_behavior_update(
    payload: Dict[str, Any],
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    require_api_key(x_api_key)
    config_manager.update_llm_behavior(payload if isinstance(payload, dict) else {})
    return {"status": "success"}


@app.get("/admin/processes/status")
async def admin_processes_status(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    """Get status of all background processes"""
    require_api_key(x_api_key)

    if not PROCESS_MANAGEMENT_AVAILABLE:
        return {"error": "Process management not available"}

    try:
        manager = get_process_manager()

        if not manager.running:
            return {
                "status": "stopped",
                "processes": [],
                "metrics": {
                    "total_processes": 0,
                    "running_processes": 0,
                    "failed_processes": 0,
                    "health_percentage": 0.0,
                },
            }

        processes = manager.list_processes()
        metrics = manager.get_process_metrics()

        process_list = []
        for process in processes:
            process_info = {
                "id": process.id,
                "name": process.name,
                "type": process.type.value,
                "state": process.state.value,
                "pid": process.pid,
                "cpu_usage": process.cpu_usage,
                "memory_usage": process.memory_usage,
                "restart_count": process.restart_count,
                "start_time": (
                    process.start_time.isoformat() if process.start_time else None
                ),
                "uptime": process.metrics.get("uptime", 0),
            }
            process_list.append(process_info)

        return {"status": "running", "processes": process_list, "metrics": metrics}

    except Exception as e:
        return {"error": f"Failed to get process status: {str(e)}"}


# ==============================================================================
# NEW DIAGNOSTIC AND MONITORING ENDPOINTS
# ==============================================================================


@app.get("/admin/database/stats")
async def admin_database_stats(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    """
    Get comprehensive database performance statistics.

    Returns:
        - Query performance metrics (avg, max, min, slow queries)
        - Connection pool stats
        - Database size and table counts
        - Recent query details
    """
    require_api_key(x_api_key)

    try:
        from .db_profiler import (
            get_profiler,
            get_connection_pool_stats,
            get_database_stats,
        )

        profiler = get_profiler()

        return {
            "query_performance": profiler.get_stats(),
            "connection_pool": get_connection_pool_stats(),
            "database": get_database_stats(),
            "recent_queries": profiler.get_recent_queries(limit=10),
            "slow_queries": profiler.get_slow_queries(limit=10),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        raise HTTPException(status_code=500, detail=f"Database stats unavailable: {e}")


@app.post("/admin/database/reset-stats")
async def admin_database_reset_stats(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    """Reset database performance statistics"""
    require_api_key(x_api_key)

    try:
        from .db_profiler import get_profiler

        profiler = get_profiler()
        profiler.reset_stats()

        return {
            "status": "success",
            "message": "Database statistics reset successfully",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        logger.error(f"Failed to reset database stats: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")


@app.get("/admin/integrations/health")
async def admin_integrations_health(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    timeout: float = Query(default=10.0, ge=1.0, le=30.0),
):
    """
    Check health of all external integrations in parallel.

    Tests:
        - LLM backend (Ollama/HuggingFace)
        - Database connectivity
        - Web search (DuckDuckGo)
        - Web fetch (httpx)
        - OSINT integrations
        - Slack connector
        - Home Assistant (if enabled)

    Query params:
        - timeout: Maximum time in seconds (default: 10)
    """
    require_api_key(x_api_key)

    try:
        from .integration_health import check_all_integrations

        results = await check_all_integrations(timeout=timeout)
        return results
    except Exception as e:
        logger.error(f"Failed to check integration health: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")


@app.get("/admin/diagnostics/system")
async def admin_diagnostics_system(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    """
    Get comprehensive system resource diagnostics.

    Returns:
        - Memory usage (process and system)
        - CPU usage and load average
        - Thread count and details
        - File descriptor usage (Unix)
        - Network connections
        - Disk usage
        - Event loop statistics
        - Process information
    """
    require_api_key(x_api_key)

    try:
        from .system_diagnostics import get_full_diagnostics

        diagnostics = await get_full_diagnostics()
        return diagnostics
    except Exception as e:
        logger.error(f"Failed to get system diagnostics: {e}")
        raise HTTPException(status_code=500, detail=f"Diagnostics unavailable: {e}")


@app.get("/admin/diagnostics/health-summary")
async def admin_diagnostics_health_summary(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    """Get overall system health status summary"""
    require_api_key(x_api_key)

    try:
        from .system_diagnostics import get_health_summary

        health_status = get_health_summary()

        return {
            "status": health_status,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        logger.error(f"Failed to get health summary: {e}")
        raise HTTPException(status_code=500, detail=f"Health summary unavailable: {e}")


@app.get("/admin/metrics/comprehensive")
async def admin_metrics_comprehensive(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    """
    Get comprehensive system metrics from all components.

    This single endpoint aggregates:
    - Resource usage (CPU, memory)
    - Request coalescing statistics
    - Connection pool metrics
    - Query cache performance
    - Rate limiter status
    - Circuit breaker health
    - Event loop monitoring
    - Database performance
    - LLM provider metrics

    Use this for unified monitoring dashboards.
    """
    require_api_key(x_api_key)

    try:
        from .metrics_aggregator import get_metrics_aggregator

        aggregator = await get_metrics_aggregator()
        snapshot = await aggregator.collect_metrics()

        return {
            "timestamp": snapshot.timestamp,
            "uptime_seconds": snapshot.uptime_seconds,
            "system": {
                "cpu_percent": snapshot.cpu_percent,
                "memory_rss_mb": snapshot.memory_rss_mb,
                "memory_percent": snapshot.memory_percent,
            },
            "performance": {
                "request_coalescing": snapshot.request_coalescing,
                "connection_pool": snapshot.connection_pool,
                "query_cache": snapshot.query_cache,
                "rate_limiter": snapshot.rate_limiter,
            },
            "health": {
                "circuit_breakers": snapshot.circuit_breakers,
                "event_loop": snapshot.event_loop,
            },
            "data": {
                "database": snapshot.database,
                "llm": snapshot.llm,
            },
        }
    except Exception as e:
        logger.error(f"Failed to collect comprehensive metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics collection failed: {e}")


@app.get("/admin/metrics/summary")
async def admin_metrics_summary(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    """Get high-level metrics summary for quick dashboard views"""
    require_api_key(x_api_key)

    try:
        from .metrics_aggregator import get_metrics_aggregator

        aggregator = await get_metrics_aggregator()
        summary = aggregator.get_metrics_summary()

        return summary
    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics summary unavailable: {e}")


@app.get("/admin/metrics/trends/{metric_path}")
async def admin_metrics_trends(
    metric_path: str,
    samples: int = Query(default=10, ge=1, le=100),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """
    Get trend data for a specific metric.

    Examples:
        - /admin/metrics/trends/cpu_percent
        - /admin/metrics/trends/query_cache.hit_rate
        - /admin/metrics/trends/connection_pool.reuse_rate
    """
    require_api_key(x_api_key)

    try:
        from .metrics_aggregator import get_metrics_aggregator

        aggregator = await get_metrics_aggregator()
        trends = aggregator.get_trends(metric_path, samples)

        return {
            "metric": metric_path,
            "samples": len(trends),
            "values": trends,
            "latest": trends[-1] if trends else None,
        }
    except Exception as e:
        logger.error(f"Failed to get metric trends: {e}")
        raise HTTPException(status_code=500, detail=f"Trend data unavailable: {e}")


@app.get("/admin/config/validate")
async def admin_config_validate(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    """
    Validate current configuration settings.

    Checks:
        - Required fields are present
        - Values are within valid ranges
        - Security best practices
        - Integration completeness
    """
    require_api_key(x_api_key)

    try:
        from .config_validator import ConfigValidator
        from .config import get_config

        config = get_config()
        validator = ConfigValidator()
        is_valid, summary = validator.validate_config(config)

        return summary
    except Exception as e:
        logger.error(f"Failed to validate config: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {e}")


@app.post("/admin/processes/start")
async def admin_processes_start(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    """Start all background processes"""
    require_api_key(x_api_key)

    if not PROCESS_MANAGEMENT_AVAILABLE:
        return {"error": "Process management not available"}

    try:
        await start_background_processes()
        return {"status": "success", "message": "Background processes started"}
    except Exception as e:
        return {"error": f"Failed to start processes: {str(e)}"}


@app.post("/admin/processes/stop")
async def admin_processes_stop(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    """Stop all background processes"""
    require_api_key(x_api_key)

    if not PROCESS_MANAGEMENT_AVAILABLE:
        return {"error": "Process management not available"}

    try:
        await stop_background_processes()
        return {"status": "success", "message": "Background processes stopped"}
    except Exception as e:
        return {"error": f"Failed to stop processes: {str(e)}"}


@app.post("/admin/processes/{process_id}/restart")
async def admin_process_restart(
    process_id: str, x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    """Restart a specific background process"""
    require_api_key(x_api_key)

    if not PROCESS_MANAGEMENT_AVAILABLE:
        return {"error": "Process management not available"}

    try:
        manager = get_process_manager()
        await manager.restart_process(process_id)
        return {"status": "success", "message": f"Process {process_id} restarted"}
    except Exception as e:
        return {"error": f"Failed to restart process: {str(e)}"}


@app.get("/admin/errors/stats")
async def admin_error_stats(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    """Get error statistics"""
    require_api_key(x_api_key)

    if not ERROR_HANDLING_AVAILABLE:
        return {"error": "Error handling not available"}

    try:
        handler = get_error_handler()
        stats = handler.get_error_stats()
        return {"status": "success", "stats": stats}
    except Exception as e:
        return {"error": f"Failed to get error stats: {str(e)}"}


@app.get("/admin/recovery/stats")
async def admin_recovery_stats(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    """Get recovery statistics"""
    require_api_key(x_api_key)

    if not ERROR_HANDLING_AVAILABLE:
        return {"error": "Error handling not available"}

    try:
        from .recovery_manager import get_recovery_manager

        recovery_manager = get_recovery_manager()
        stats = recovery_manager.get_recovery_stats()
        return {"status": "success", "stats": stats}
    except Exception as e:
        return {"error": f"Failed to get recovery stats: {str(e)}"}


@app.post("/admin/recovery/clear-history")
async def admin_recovery_clear_history(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    """Clear recovery history"""
    require_api_key(x_api_key)

    if not ERROR_HANDLING_AVAILABLE:
        return {"error": "Error handling not available"}

    try:
        from .recovery_manager import get_recovery_manager

        recovery_manager = get_recovery_manager()
        recovery_manager.clear_history()
        return {"status": "success", "message": "Recovery history cleared"}
    except Exception as e:
        return {"error": f"Failed to clear recovery history: {str(e)}"}


# ECC Cryptography endpoints
@app.post("/admin/ecc/generate-key")
async def admin_ecc_generate_key(
    curve: str = ECCCurve.SECP256R1,
    expires_in_days: Optional[int] = None,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Generate new ECC key pair"""
    require_api_key(x_api_key)

    if not ECC_AVAILABLE:
        return {"error": "ECC not available"}

    try:
        ecc_manager = get_ecc_manager()
        key_pair = ecc_manager.generate_key_pair(curve, expires_in_days=expires_in_days)

        return {
            "status": "success",
            "key_id": key_pair.key_id,
            "curve": key_pair.curve,
            "public_key": key_pair.public_key_pem,
            "created_at": key_pair.created_at.isoformat(),
            "expires_at": (
                key_pair.expires_at.isoformat() if key_pair.expires_at else None
            ),
        }
    except Exception as e:
        return {"error": f"Failed to generate key: {str(e)}"}


@app.get("/admin/ecc/keys")
async def admin_ecc_list_keys(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    """List all ECC keys"""
    require_api_key(x_api_key)

    if not ECC_AVAILABLE:
        return {"error": "ECC not available"}

    try:
        ecc_manager = get_ecc_manager()
        keys = []

        for key_id, key_pair in ecc_manager.list_keys().items():
            key_info = ecc_manager.get_key_info(key_id)
            if key_info:
                keys.append(key_info)

        return {"status": "success", "keys": keys}
    except Exception as e:
        return {"error": f"Failed to list keys: {str(e)}"}


@app.get("/admin/ecc/keys/{key_id}")
async def admin_ecc_get_key(
    key_id: str, x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    """Get ECC key information"""
    require_api_key(x_api_key)

    if not ECC_AVAILABLE:
        return {"error": "ECC not available"}

    try:
        ecc_manager = get_ecc_manager()
        key_info = ecc_manager.get_key_info(key_id)

        if not key_info:
            return {"error": "Key not found"}

        # Include public key
        key_pair = ecc_manager.get_key_pair(key_id)
        if key_pair:
            key_info["public_key"] = key_pair.public_key_pem

        return {"status": "success", "key": key_info}
    except Exception as e:
        return {"error": f"Failed to get key: {str(e)}"}


@app.delete("/admin/ecc/keys/{key_id}")
async def admin_ecc_delete_key(
    key_id: str, x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    """Delete ECC key"""
    require_api_key(x_api_key)

    if not ECC_AVAILABLE:
        return {"error": "ECC not available"}

    try:
        ecc_manager = get_ecc_manager()
        success = ecc_manager.delete_key(key_id)

        if success:
            return {"status": "success", "message": f"Key {key_id} deleted"}
        else:
            return {"error": "Key not found"}
    except Exception as e:
        return {"error": f"Failed to delete key: {str(e)}"}


@app.post("/admin/ecc/sign")
async def admin_ecc_sign_data(
    payload: Dict[str, Any],
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Sign data with ECC key"""
    require_api_key(x_api_key)

    if not ECC_AVAILABLE:
        return {"error": "ECC not available"}

    try:
        data = payload.get("data")
        key_id = payload.get("key_id")

        if not data or not key_id:
            return {"error": "data and key_id are required"}

        ecc_manager = get_ecc_manager()
        signature = ecc_manager.sign_data(data, key_id)

        return {"status": "success", "signature": signature.to_dict()}
    except Exception as e:
        return {"error": f"Failed to sign data: {str(e)}"}


@app.post("/admin/ecc/verify")
async def admin_ecc_verify_signature(
    payload: Dict[str, Any],
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Verify ECC signature"""
    require_api_key(x_api_key)

    if not ECC_AVAILABLE:
        return {"error": "ECC not available"}

    try:
        data = payload.get("data")
        signature_data = payload.get("signature")
        public_key_pem = payload.get("public_key_pem")

        if not data or not signature_data:
            return {"error": "data and signature are required"}

        from .ecc_crypto import ECCSignature
        from datetime import datetime

        signature = ECCSignature(
            signature=signature_data["signature"],
            algorithm=signature_data["algorithm"],
            key_id=signature_data["key_id"],
            timestamp=datetime.fromisoformat(signature_data["timestamp"]),
        )

        ecc_manager = get_ecc_manager()
        valid = ecc_manager.verify_signature(data, signature, public_key_pem)

        return {"status": "success", "valid": valid}
    except Exception as e:
        return {"error": f"Failed to verify signature: {str(e)}"}


@app.post("/admin/security/generate-api-key")
async def admin_security_generate_api_key(
    payload: Dict[str, Any],
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Generate secure API key with ECC backing"""
    require_api_key(x_api_key)

    if not ECC_AVAILABLE:
        return {"error": "ECC not available"}

    try:
        permissions = payload.get("permissions", ["read"])
        expires_in_days = payload.get("expires_in_days")
        rate_limit = payload.get("rate_limit")

        security_manager = get_security_manager()
        secure_key = security_manager.generate_secure_api_key(
            permissions=permissions,
            expires_in_days=expires_in_days,
            rate_limit=rate_limit,
        )

        return {
            "status": "success",
            "api_key": secure_key.api_key,
            "key_id": secure_key.key_id,
            "ecc_key_id": secure_key.ecc_key_id,
            "permissions": secure_key.permissions,
            "expires_at": (
                secure_key.expires_at.isoformat() if secure_key.expires_at else None
            ),
        }
    except Exception as e:
        return {"error": f"Failed to generate API key: {str(e)}"}


@app.get("/admin/security/api-keys")
async def admin_security_list_api_keys(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    """List all secure API keys"""
    require_api_key(x_api_key)

    if not ECC_AVAILABLE:
        return {"error": "ECC not available"}

    try:
        security_manager = get_security_manager()
        api_keys = security_manager.list_api_keys()

        return {"status": "success", "api_keys": api_keys}
    except Exception as e:
        return {"error": f"Failed to list API keys: {str(e)}"}


# Backup API endpoints
@app.post("/backup/create")
async def backup_create(
    tag: str = "api", x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    require_api_key(x_api_key)

    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.backup_manager import create_backup

        backup_path = create_backup(tag)
        return {"success": True, "backup_path": backup_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/backup/list")
async def backup_list(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    require_api_key(x_api_key)

    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.backup_manager import list_backups

        backups = list_backups()
        return {"backups": backups}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/backup/restore")
async def backup_restore(
    backup_file: str,
    restore_dir: Optional[str] = None,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    require_api_key(x_api_key)

    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.backup_manager import restore_backup

        restore_backup(backup_file, restore_dir)
        return {"success": True, "message": f"Restored from {backup_file}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# Voice Profile API endpoints
@app.post("/voice/samples")
async def voice_add_sample(
    file_path: str, x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    require_api_key(x_api_key)

    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.voice_profile_manager import add_voice_sample

        dest = add_voice_sample(file_path)
        return {"success": True, "sample_path": dest}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/voice/profile/update")
async def voice_update_profile(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    require_api_key(x_api_key)

    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.voice_profile_manager import update_voice_profile

        profile = update_voice_profile()
        return {"success": True, "profile": profile}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/voice/profile")
async def voice_get_profile(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    require_api_key(x_api_key)

    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.voice_profile_manager import get_voice_profile

        profile = get_voice_profile()
        return {"profile": profile}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# Knowledge Base API endpoints
@app.post("/kb/sites")
async def kb_add_site(
    category: str,
    url: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    require_api_key(x_api_key)

    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.web_knowledge_base import add_site

        add_site(category, url)
        return {"success": True, "message": f"Added {url} to {category}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/kb/sites")
async def kb_list_sites(
    category: Optional[str] = None,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    if not x_api_key or x_api_key != "vega-default-key":
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.web_knowledge_base import list_sites

        sites = list_sites(category)
        return {"sites": sites}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# Financial API endpoints
@app.post("/finance/invest")
async def finance_invest(
    symbol: str,
    shares: float,
    price: float,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    if not x_api_key or x_api_key != "vega-default-key":
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.finance_module import add_investment

        add_investment(symbol, shares, price)
        return {"success": True, "message": f"Added {shares} shares of {symbol}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/finance/portfolio")
async def finance_portfolio(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    if not x_api_key or x_api_key != "vega-default-key":
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.finance_module import list_investments

        investments = list_investments()
        return {"portfolio": investments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/finance/price/{symbol}")
async def finance_price(
    symbol: str, x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    if not x_api_key or x_api_key != "vega-default-key":
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        import sys

        sys.path.append("/home/ncacord/Vega2.0")
        from vega_state.finance_module import fetch_stock_price

        price = fetch_stock_price(symbol)
        return {"symbol": symbol, "price": price}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# Root endpoint
@app.get("/")
async def index():
    return HTMLResponse(
        """
    <html>
      <head><title>Vega2.0</title></head>
      <body>
        <h1>Vega2.0 Control Panel</h1>
        <p><a href="/static/index.html">Advanced Control Panel</a></p>
        <p><a href="/healthz">Health Check</a></p>
        <p><a href="/metrics">Metrics</a></p>
      </body>
    </html>
    """
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("core.app:app", host="127.0.0.1", port=8000, reload=True)
