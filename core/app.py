#!/usr/bin/env python3
"""
FastAPI application for Vega2.0 - Clean Implementation
"""

from typing import Optional, Dict, Any
import uuid
from datetime import datetime

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from utils.commands_generator import (
        generate_commands_markdown,
        generate_commands_html,
    )
except Exception:
    # Defer import errors to endpoint call time
    generate_commands_markdown = None  # type: ignore
    generate_commands_html = None  # type: ignore

# Initialize app
app = FastAPI(title="Vega2.0", version="2.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Metrics
app.state.metrics = {
    "requests_total": 0,
    "responses_total": 0,
    "errors_total": 0,
    "degraded": False,
}


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
        raise HTTPException(status_code=401, detail="Missing API key")
    allowed = {cfg.api_key, *getattr(cfg, "api_keys_extra", [])}
    if x_api_key not in allowed:
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
from core.logging_setup import VegaLogger  # exposed for patching in tests

VegaLogger = VegaLogger  # re-expose name explicitly
from core.config_manager import config_manager  # exposed for patching in tests


@app.get("/docs/commands.md")
async def docs_commands_md():
    # Lazy import if earlier failed
    global generate_commands_markdown
    if generate_commands_markdown is None:
        from utils.commands_generator import generate_commands_markdown as _gen_md  # type: ignore

        generate_commands_markdown = _gen_md
    md = generate_commands_markdown()  # type: ignore
    return Response(content=md, media_type="text/markdown")


@app.get("/docs/commands")
async def docs_commands_html():
    global generate_commands_html
    if generate_commands_html is None:
        from utils.commands_generator import generate_commands_html as _gen_html  # type: ignore

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
        from core.llm import LLMBackendError  # type: ignore
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
    from core.logging_setup import VegaLogger

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
    from core.logging_setup import VegaLogger

    lines_list = VegaLogger.tail_log(module, lines)
    return {"module": module, "lines": lines_list, "total_lines": len(lines_list)}


@app.get("/admin/config")
async def admin_config_list(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    require_api_key(x_api_key)
    from core.config_manager import config_manager as config_manager

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
