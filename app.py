"""
app.py - FastAPI application for Vega2.0

Endpoints:
- GET /healthz -> {"ok": true}
- GET /livez, /readyz -> process/live readiness
- POST /chat (requires X-API-Key header)
  Body: {"prompt": str, "stream": bool, "session_id": str|None}
  Returns: JSON with response or text/plain streaming
- GET /history?limit=N -> last N messages
- GET /session/{session_id} -> recent messages for session
- POST /feedback -> annotate a conversation

Security & hardening notes:
- Keep HOST=127.0.0.1 for local-only
- Requires API key for data endpoints; supports multiple keys
"""

from typing import AsyncGenerator
import asyncio
import uuid
import os

from fastapi import FastAPI, Header, HTTPException, Request, Body
from fastapi.responses import (
    JSONResponse,
    StreamingResponse,
    HTMLResponse,
    PlainTextResponse,
)
from pydantic import BaseModel

from config import get_config
from db import (
    log_conversation,
    get_history,
    get_session_history,
    set_feedback,
    get_history_page,
    purge_old,
)
from integrations.search import web_search, image_search
from integrations.fetch import fetch_text
from integrations.osint import (
    dns_lookup,
    reverse_dns,
    http_headers,
    ssl_cert_info,
    robots_txt,
    whois_lookup,
    tcp_scan,
    username_search,
)
from llm import (
    query_llm,
    LLMBackendError,
    breaker_stats,
    cache_stats,
    llm_warmup,
    llm_shutdown,
    get_generation_settings,
    set_generation_settings,
    reset_generation_settings,
)

# Optional rate limiting
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
except Exception:
    Limiter = None
    get_remote_address = None  # type: ignore


cfg = get_config()
app = FastAPI(title="Vega2.0", version="0.1.0")

if Limiter:
    limiter = Limiter(key_func=get_remote_address)  # type: ignore
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)

    @app.exception_handler(RateLimitExceeded)  # type: ignore
    async def ratelimit_handler(request: Request, exc: RateLimitExceeded):
        return PlainTextResponse(str(exc), status_code=429)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500, content={"error": "Internal error", "detail": str(exc)[:200]}
    )


# In-memory metrics
app.state.metrics = {
    "requests_total": 0,
    "responses_total": 0,
    "errors_total": 0,
    "timeouts_total": 0,
    "degraded": False,
}


@app.on_event("startup")
async def _startup():
    async def _probe_loop():
        try:
            import httpx
        except Exception:
            return
        while True:
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.post(
                        "http://127.0.0.1:11434/api/tags", timeout=3.0
                    )
                    app.state.metrics["degraded"] = (
                        False if r.status_code < 500 else True
                    )
            except Exception:
                app.state.metrics["degraded"] = True
            await asyncio.sleep(5.0)

    try:
        asyncio.create_task(_probe_loop())
    except Exception:
        pass
    try:
        await llm_warmup()
    except Exception:
        pass
    try:
        if cfg.retention_days and int(cfg.retention_days) > 0:
            purge_old(int(cfg.retention_days))
    except Exception:
        pass


@app.on_event("shutdown")
async def _shutdown():
    try:
        await llm_shutdown()
    except Exception:
        pass


@app.get("/metrics")
async def metrics():
    m = dict(app.state.metrics)
    try:
        m["breaker"] = breaker_stats()
        m["cache"] = cache_stats()
    except Exception:
        pass
    return m


@app.get("/livez")
async def livez():
    return {"ok": True}


@app.get("/readyz")
async def readyz():
    ok = not app.state.metrics.get("degraded") and os.path.exists("vega.db")
    return {"ready": bool(ok)}


class ChatRequest(BaseModel):
    prompt: str
    stream: bool = False
    session_id: str | None = None


class FeedbackRequest(BaseModel):
    conversation_id: int
    rating: int | None = None
    tags: str | None = None
    notes: str | None = None
    reviewed: bool | None = None


class SearchRequest(BaseModel):
    query: str
    max_results: int = 5
    safesearch: str = "moderate"


class ResearchRequest(BaseModel):
    query: str
    max_results: int = 5
    safesearch: str = "moderate"


class GenUpdate(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repeat_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    dynamic_generation: bool | None = None


class DNSRequest(BaseModel):
    hostname: str


class RDNSRequest(BaseModel):
    ip: str


class HeadersRequest(BaseModel):
    url: str


class SSLRequest(BaseModel):
    host: str
    port: int = 443


class RobotsRequest(BaseModel):
    url: str


class WhoisRequest(BaseModel):
    domain: str


class ScanRequest(BaseModel):
    host: str
    ports: list[int]
    timeout: float | None = 0.6
    concurrency: int | None = 200


class UsernameRequest(BaseModel):
    username: str
    include_nsfw: bool = False
    sites: list[str] | None = None


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/history")
@limiter.limit("120/minute") if Limiter else (lambda f: f)  # type: ignore
async def history(
    request: Request,
    limit: int = 50,
    cursor: int | None = None,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    if not x_api_key or x_api_key not in {cfg.api_key, *cfg.api_keys_extra}:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    data = get_history_page(limit=limit, before_id=cursor)
    next_cursor = (min([d["id"] for d in data]) - 1) if data else None
    return {"items": data, "next_cursor": next_cursor}


@app.get("/session/{session_id}")
async def session_state(
    session_id: str, x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    if not x_api_key or x_api_key not in {cfg.api_key, *cfg.api_keys_extra}:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return {"items": get_session_history(session_id)}


async def _stream_chat(prompt: str) -> AsyncGenerator[bytes, None]:
    try:
        gen = await query_llm(prompt, stream=True)
        async for chunk in gen:  # type: ignore
            yield str(chunk).encode("utf-8")
    except LLMBackendError:
        app.state.metrics["errors_total"] += 1
        yield b"[LLM backend unavailable: ensure Ollama is running on 127.0.0.1:11434]"
    except Exception as exc:
        app.state.metrics["errors_total"] += 1
        yield f"[stream error: {str(exc)[:200]}]".encode("utf-8")


@app.post("/chat")
@limiter.limit("60/minute") if Limiter else (lambda f: f)  # type: ignore
async def chat(
    request: Request,
    req: ChatRequest = Body(...),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    app.state.metrics["requests_total"] += 1
    if not x_api_key or x_api_key not in {cfg.api_key, *cfg.api_keys_extra}:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    if len(req.prompt) > cfg.max_prompt_chars:
        raise HTTPException(status_code=413, detail="Prompt too large")

    sid = req.session_id or uuid.uuid4().hex[:16]

    # include short session memory
    augmented = req.prompt
    try:
        recent = get_session_history(sid, limit=6)
        if recent:
            history_txt = "\n\n".join(
                [
                    f"User: {h['prompt']}\nAssistant: {h['response']}"
                    for h in reversed(recent)
                ]
            )
            augmented = f"Conversation so far:\n{history_txt}\n\nUser: {req.prompt}"
            if len(augmented) > cfg.max_prompt_chars:
                augmented = augmented[-cfg.max_prompt_chars :]
    except Exception:
        augmented = req.prompt

    if req.stream:
        return StreamingResponse(_stream_chat(augmented), media_type="text/plain")

    try:
        reply = await asyncio.wait_for(
            query_llm(augmented, stream=False), timeout=cfg.llm_timeout_sec
        )
    except LLMBackendError as exc:
        app.state.metrics["errors_total"] += 1
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except asyncio.TimeoutError:
        app.state.metrics["timeouts_total"] += 1
        msg = "[timeout: LLM did not respond in time]"
        try:
            cid = log_conversation(req.prompt, msg, source="api", session_id=sid)
        except Exception:
            cid = None
        return {
            "response": msg,
            "session_id": sid,
            **({"conversation_id": cid} if cid else {}),
        }
    except Exception as exc:
        app.state.metrics["errors_total"] += 1
        msg = f"[error: {str(exc)[:200]}]"
        try:
            cid = log_conversation(req.prompt, msg, source="api", session_id=sid)
        except Exception:
            cid = None
        return {
            "response": msg,
            "session_id": sid,
            **({"conversation_id": cid} if cid else {}),
        }

    if isinstance(reply, str):
        cid = log_conversation(req.prompt, reply, source="api", session_id=sid)
        app.state.metrics["responses_total"] += 1
        return {"response": reply, "conversation_id": cid, "session_id": sid}
    else:
        raise HTTPException(status_code=500, detail="Unexpected response type")


async def _sse_stream(prompt: str) -> AsyncGenerator[bytes, None]:
    """SSE formatter around the token stream."""
    # initial ping to keep connection open
    yield b": keep-alive\n\n"
    try:
        gen = await query_llm(prompt, stream=True)
        async for chunk in gen:  # type: ignore
            data = str(chunk).replace("\n", " ")
            yield f"data: {data}\n\n".encode("utf-8")
    except LLMBackendError:
        app.state.metrics["errors_total"] += 1
        yield b"event: error\ndata: LLM backend unavailable\n\n"
    except Exception as exc:
        app.state.metrics["errors_total"] += 1
        yield f"event: error\ndata: {str(exc)[:200]}\n\n".encode("utf-8")


@app.post("/chat/sse")
@limiter.limit("60/minute") if Limiter else (lambda f: f)  # type: ignore
async def chat_sse(
    request: Request,
    req: ChatRequest = Body(...),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    app.state.metrics["requests_total"] += 1
    if not x_api_key or x_api_key not in {cfg.api_key, *cfg.api_keys_extra}:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    if len(req.prompt) > cfg.max_prompt_chars:
        raise HTTPException(status_code=413, detail="Prompt too large")

    # include short session memory for SSE as well
    sid = req.session_id or uuid.uuid4().hex[:16]
    augmented = req.prompt
    try:
        recent = get_session_history(sid, limit=6)
        if recent:
            history_txt = "\n\n".join(
                [
                    f"User: {h['prompt']}\nAssistant: {h['response']}"
                    for h in reversed(recent)
                ]
            )
            augmented = f"Conversation so far:\n{history_txt}\n\nUser: {req.prompt}"
            if len(augmented) > cfg.max_prompt_chars:
                augmented = augmented[-cfg.max_prompt_chars :]
    except Exception:
        augmented = req.prompt

    return StreamingResponse(_sse_stream(augmented), media_type="text/event-stream")


@app.post("/integrations/test")
async def integrations_test(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    if not x_api_key or x_api_key not in {cfg.api_key, *cfg.api_keys_extra}:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    try:
        from integrations.slack_connector import send_slack_message
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Slack integration not available"
        ) from exc
    ok = send_slack_message(cfg.slack_webhook_url, "Vega2.0 integration test via API")
    return {"ok": bool(ok)}


@app.post("/search/web")
async def search_web(
    req: SearchRequest, x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    if not x_api_key or x_api_key not in {cfg.api_key, *cfg.api_keys_extra}:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    items = web_search(
        req.query, max_results=req.max_results, safesearch=req.safesearch
    )
    return {"items": items}


@app.post("/search/images")
async def search_images(
    req: SearchRequest, x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    if not x_api_key or x_api_key not in {cfg.api_key, *cfg.api_keys_extra}:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    items = image_search(
        req.query, max_results=req.max_results, safesearch=req.safesearch
    )
    return {"items": items}


@app.post("/research/summarize")
async def research_summarize(
    req: ResearchRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    if not x_api_key or x_api_key not in {cfg.api_key, *cfg.api_keys_extra}:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    items = web_search(
        req.query, max_results=req.max_results, safesearch=req.safesearch
    )
    if not items:
        return {"summary": "No results found or search unavailable.", "items": []}
    # Construct a compact context for LLM summarization
    context_lines = []
    for i, it in enumerate(items, start=1):
        context_lines.append(
            f"[{i}] {it.get('title') or ''} - {it.get('href') or ''}\n{it.get('snippet') or ''}"
        )
    context = "\n\n".join(context_lines)
    prompt = (
        "You are a research assistant. Given the following web results, provide a concise, unbiased summary (5-8 bullet points) and include 2-3 top links at the end.\n\n"
        + context
    )
    try:
        summary = await asyncio.wait_for(
            query_llm(prompt, stream=False), timeout=cfg.llm_timeout_sec
        )
    except Exception as exc:
        summary = f"[error summarizing: {str(exc)[:200]}]"
    return {"summary": summary, "items": items}


@app.post("/research/rag")
async def research_rag(
    req: ResearchRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    if not x_api_key or x_api_key not in {cfg.api_key, *cfg.api_keys_extra}:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    items = web_search(
        req.query, max_results=min(8, req.max_results), safesearch=req.safesearch
    )

    # Fetch a few pages concurrently
    async def _grab(it):
        url = it.get("href") or it.get("url")
        if not url:
            return None
        txt = await fetch_text(url)
        if not txt:
            return None
        return {"title": it.get("title"), "url": url, "text": txt}

    tasks = [asyncio.create_task(_grab(it)) for it in items[:6]]
    docs: list[dict] = []
    for t in tasks:
        try:
            r = await t
            if r:
                docs.append(r)
        except Exception:
            continue
    if not docs:
        return {"summary": "No fetchable documents.", "items": items}

    # Rank docs by simple keyword frequency and useful length
    def _keywords(q: str) -> list[str]:
        stop = {
            "the",
            "and",
            "or",
            "a",
            "to",
            "of",
            "in",
            "on",
            "for",
            "with",
            "by",
            "about",
            "is",
            "are",
            "what",
            "how",
            "why",
            "when",
        }
        return [w for w in q.lower().split() if w.isalpha() and w not in stop]

    kws = _keywords(req.query)

    def _score(d: dict) -> float:
        txt = (d.get("text") or "").lower()
        kw_score = sum(txt.count(k) for k in kws) if kws else 0
        len_score = min(len(txt) / 5000.0, 1.5)  # cap length contribution
        return kw_score + len_score

    docs.sort(key=_score, reverse=True)
    top_docs = docs[:3]

    # Build a prompt with trimmed doc excerpts
    parts = []
    for i, d in enumerate(top_docs, start=1):
        excerpt = d["text"][:1200]
        parts.append(f"[{i}] {d['title'] or ''} - {d['url']}\n{excerpt}")
    context = "\n\n".join(parts)
    prompt = (
        "Using the following documents, provide a concise, factual summary (5-8 bullets) with inline citations like [1], [2]. Then list top 3 links.\n\n"
        + context
    )
    try:
        summary = await asyncio.wait_for(
            query_llm(prompt, stream=False), timeout=cfg.llm_timeout_sec
        )
    except Exception as exc:
        summary = f"[error summarizing: {str(exc)[:200]}]"
    # Return source mapping for UI
    sources = [
        {"id": i + 1, "title": d["title"], "url": d["url"]}
        for i, d in enumerate(top_docs)
    ]
    return {"summary": summary, "sources": sources}


# Admin: generation settings endpoints
@app.get("/admin/gen")
async def admin_gen_get(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    if not x_api_key or x_api_key not in {cfg.api_key, *cfg.api_keys_extra}:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return get_generation_settings()


@app.post("/admin/gen")
async def admin_gen_set(
    body: GenUpdate,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    if not x_api_key or x_api_key not in {cfg.api_key, *cfg.api_keys_extra}:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    new_settings = set_generation_settings(
        temperature=body.temperature,
        top_p=body.top_p,
        top_k=body.top_k,
        repeat_penalty=body.repeat_penalty,
        presence_penalty=body.presence_penalty,
        frequency_penalty=body.frequency_penalty,
        dynamic_generation=body.dynamic_generation,
    )
    return new_settings


@app.post("/admin/gen/reset")
async def admin_gen_reset(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    if not x_api_key or x_api_key not in {cfg.api_key, *cfg.api_keys_extra}:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return reset_generation_settings()


# OSINT & Networking endpoints (local-only, API-key protected)
@app.post("/osint/dns")
async def osint_dns(
    req: DNSRequest, x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    if not x_api_key or x_api_key not in {cfg.api_key, *cfg.api_keys_extra}:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    try:
        res = dns_lookup(req.hostname)
        return {"hostname": res.hostname, "addresses": res.addresses}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/osint/reverse_dns")
async def osint_reverse_dns(
    req: RDNSRequest, x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    if not x_api_key or x_api_key not in {cfg.api_key, *cfg.api_keys_extra}:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return {"names": reverse_dns(req.ip)}


@app.post("/osint/http_headers")
async def osint_http_headers(
    req: HeadersRequest, x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    if not x_api_key or x_api_key not in {cfg.api_key, *cfg.api_keys_extra}:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    try:
        res = http_headers(req.url)
        return {"url": res.url, "status": res.status, "headers": res.headers}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/osint/ssl_cert")
async def osint_ssl_cert(
    req: SSLRequest, x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    if not x_api_key or x_api_key not in {cfg.api_key, *cfg.api_keys_extra}:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    info = ssl_cert_info(req.host, port=req.port)
    if not info:
        raise HTTPException(status_code=404, detail="No certificate info")
    return {
        "host": info.host,
        "subject": info.subject,
        "issuer": info.issuer,
        "not_before": info.not_before,
        "not_after": info.not_after,
    }


@app.post("/osint/robots")
async def osint_robots(
    req: RobotsRequest, x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    if not x_api_key or x_api_key not in {cfg.api_key, *cfg.api_keys_extra}:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    txt = robots_txt(req.url)
    return {"robots": txt}


@app.post("/osint/whois")
async def osint_whois(
    req: WhoisRequest, x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    if not x_api_key or x_api_key not in {cfg.api_key, *cfg.api_keys_extra}:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return whois_lookup(req.domain)


@app.post("/net/scan")
async def net_scan(
    req: ScanRequest, x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    if not x_api_key or x_api_key not in {cfg.api_key, *cfg.api_keys_extra}:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    # Safety caps
    if not req.ports or len(req.ports) > 1024:
        raise HTTPException(
            status_code=400, detail="Ports list required and <=1024 length"
        )
    if any(p <= 0 or p > 65535 for p in req.ports):
        raise HTTPException(status_code=400, detail="Invalid port in list")
    try:
        results = await asyncio.wait_for(
            tcp_scan(
                req.host,
                req.ports,
                timeout=(req.timeout or 0.6),
                concurrency=min(int(req.concurrency or 200), 1000),
            ),
            timeout=max(5.0, (req.timeout or 0.6) * len(req.ports) / 10.0),
        )
    except asyncio.TimeoutError:
        return {"results": [], "note": "scan timed out"}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"results": [{"port": p, "state": s} for p, s in results]}


@app.post("/osint/username")
async def osint_username(
    req: UsernameRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    if not x_api_key or x_api_key not in {cfg.api_key, *cfg.api_keys_extra}:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    # Basic sanity
    if not req.username or len(req.username) > 64:
        raise HTTPException(status_code=400, detail="Invalid username")
    # Run blocking net I/O in a thread to avoid starving the event loop
    results = await asyncio.to_thread(
        username_search,
        req.username,
        bool(req.include_nsfw),
        req.sites,
    )
    return {"items": results}


@app.get("/")
async def index():
    return HTMLResponse(
        """
    <html>
      <head><title>Vega2.0</title></head>
      <body>
        <h1>Vega2.0</h1>
        <p>See <a href="/static/index.html">Chat UI</a></p>
      </body>
    </html>
    """
    )


from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/feedback")
@limiter.limit("120/minute") if Limiter else (lambda f: f)  # type: ignore
async def feedback(
    request: Request,
    body: FeedbackRequest = Body(...),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    if not x_api_key or x_api_key not in {cfg.api_key, *cfg.api_keys_extra}:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    ok = set_feedback(
        body.conversation_id,
        rating=body.rating,
        tags=body.tags,
        notes=body.notes,
        reviewed=body.reviewed,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"ok": True}
