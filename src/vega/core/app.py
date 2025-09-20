#!/usr/bin/env python3
"""
FastAPI application for Vega2.0 - Clean Implementation
"""

from typing import Optional, Dict, Any
import uuid
from datetime import datetime
import asyncio

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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

# Metrics
app.state.metrics = {
    "requests_total": 0,
    "responses_total": 0,
    "errors_total": 0,
    "degraded": False,
}


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Start background processes on app startup"""
    if PROCESS_MANAGEMENT_AVAILABLE:
        try:
            # Start background processes (optional, can be managed separately)
            # await start_background_processes()
            print("Background process management available")
        except Exception as e:
            print(f"Warning: Could not start background processes: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop background processes on app shutdown"""
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


@app.get("/metrics")
async def metrics():
    return app.state.metrics


# --- Minimal config/auth and db/llm shims (for tests to patch) ---
class _Cfg:
    api_key: str = "vega-default-key"
    api_keys_extra: list[str] = []


cfg = _Cfg()


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
def query_llm(prompt: str, stream: bool = False) -> str:  # patched in tests
    return f"Echo: {prompt}"


def log_conversation(
    prompt: str, response: str, session_id: str | None = None
):  # patched
    return None


def get_history(limit: int = 50):  # patched in tests and used by readiness
    return []


def get_session_history(session_id: str, limit: int = 50):  # patched
    return []


def set_feedback(
    conversation_id: int, rating: int, comment: str | None = None
):  # patched
    return None


from fastapi import Response
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
@app.get("/livez")
async def livez():
    return {"alive": True}


@app.get("/readyz")
async def readyz():
    try:
        # Basic readiness check calls history (tests patch this)
        _ = get_history(limit=1)
        return {"ready": True}
    except Exception as e:
        raise HTTPException(status_code=503, detail="Not ready") from e


# Chat endpoint
@app.post("/chat")
async def chat(
    request: ChatRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    app.state.metrics["requests_total"] += 1
    require_api_key(x_api_key)

    # Generate/propagate session id
    session_id = request.session_id or str(uuid.uuid4())
    try:
        from .llm import LLMBackendError  # type: ignore
    except Exception:

        class LLMBackendError(Exception):
            pass

    try:
        response_text = query_llm(request.prompt, stream=request.stream)
        log_conversation(request.prompt, response_text, session_id)
        app.state.metrics["responses_total"] += 1
        return {"response": response_text, "session_id": session_id}
    except LLMBackendError as e:
        app.state.metrics["errors_total"] += 1
        raise HTTPException(status_code=503, detail="LLM backend unavailable") from e


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


# Background Process Management endpoints
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
