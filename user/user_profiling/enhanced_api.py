"""
Enhanced Chat API Wrapper
=========================

Simple wrapper around the main Vega2.0 API that adds user profiling capabilities.
Can be used alongside the main app without extensive modifications.
"""

import asyncio
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Header, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import main Vega2.0 components
try:
    from core.config import get_config
    from core.db import log_conversation, get_session_history
    from core.llm import query_llm, LLMBackendError

    VEGA_CORE_AVAILABLE = True
except ImportError:
    VEGA_CORE_AVAILABLE = False

# Import user profiling
try:
    from user.user_profiling.vega_integration import (
        startup_user_profiling,
        shutdown_user_profiling,
        get_contextual_intelligence,
        get_profile_manager,
        ContextualIntelligenceEngine,
        UserProfileManager,
        ContextualChatRequest,
        ContextualChatResponse,
        ProfileSummaryResponse,
    )

    USER_PROFILING_AVAILABLE = True
except ImportError:
    USER_PROFILING_AVAILABLE = False

logger = logging.getLogger(__name__)

# Enhanced API app
enhanced_app = FastAPI(title="Vega2.0 Enhanced", version="1.0.0")

# Add CORS middleware
enhanced_app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:*", "http://localhost:*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
cfg = None
if VEGA_CORE_AVAILABLE:
    cfg = get_config()

# Global user profiling components
contextual_intelligence: Optional[ContextualIntelligenceEngine] = None
profile_manager: Optional[UserProfileManager] = None


class SimpleResponse(BaseModel):
    """Simple response model"""

    message: str
    status: str
    timestamp: str


@enhanced_app.on_event("startup")
async def startup():
    """Initialize enhanced services"""
    global contextual_intelligence, profile_manager

    logger.info("Starting Vega2.0 Enhanced API...")

    if USER_PROFILING_AVAILABLE:
        try:
            await startup_user_profiling()

            # Get components after initialization
            contextual_intelligence = get_contextual_intelligence()
            profile_manager = get_profile_manager()

            logger.info("User profiling system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize user profiling: {e}")
    else:
        logger.warning("User profiling not available")


@enhanced_app.on_event("shutdown")
async def shutdown():
    """Shutdown enhanced services"""
    if USER_PROFILING_AVAILABLE:
        try:
            await shutdown_user_profiling()
            logger.info("User profiling system shutdown")
        except Exception as e:
            logger.error(f"Error shutting down user profiling: {e}")


@enhanced_app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Vega2.0 Enhanced API",
        "version": "1.0.0",
        "user_profiling": USER_PROFILING_AVAILABLE,
        "vega_core": VEGA_CORE_AVAILABLE,
        "timestamp": datetime.now().isoformat(),
    }


@enhanced_app.get("/healthz")
async def health_check():
    """Health check endpoint"""
    return {
        "ok": True,
        "user_profiling": USER_PROFILING_AVAILABLE,
        "vega_core": VEGA_CORE_AVAILABLE,
        "timestamp": datetime.now().isoformat(),
    }


@enhanced_app.post("/chat/enhanced")
async def enhanced_chat(
    request: Request,
    req: ContextualChatRequest = Body(...),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Enhanced chat with contextual intelligence"""

    # API key validation
    if cfg and x_api_key:
        if x_api_key not in {cfg.api_key, *cfg.api_keys_extra}:
            raise HTTPException(status_code=401, detail="Invalid API key")

    # Check required services
    if not VEGA_CORE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Vega core services not available")

    if not USER_PROFILING_AVAILABLE or not contextual_intelligence:
        raise HTTPException(
            status_code=503, detail="User profiling services not available"
        )

    if len(req.prompt) > (cfg.max_prompt_chars if cfg else 4000):
        raise HTTPException(status_code=413, detail="Prompt too large")

    sid = req.session_id or uuid.uuid4().hex[:16]

    try:
        # Build conversation context
        augmented_prompt = req.prompt
        if VEGA_CORE_AVAILABLE:
            try:
                recent = get_session_history(sid, limit=6)
                if recent:
                    history_txt = "\n\n".join(
                        [
                            f"User: {h['prompt']}\nAssistant: {h['response']}"
                            for h in reversed(recent)
                        ]
                    )
                    augmented_prompt = (
                        f"Conversation so far:\n{history_txt}\n\nUser: {req.prompt}"
                    )
                    if len(augmented_prompt) > (cfg.max_prompt_chars if cfg else 4000):
                        augmented_prompt = augmented_prompt[
                            -(cfg.max_prompt_chars if cfg else 4000) :
                        ]
            except Exception as e:
                logger.warning(f"Failed to get conversation history: {e}")
                augmented_prompt = req.prompt

        # Get base LLM response
        try:
            timeout = cfg.llm_timeout_sec if cfg else 30
            original_response = await asyncio.wait_for(
                query_llm(augmented_prompt, stream=False), timeout=timeout
            )
        except LLMBackendError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except asyncio.TimeoutError:
            original_response = "[timeout: LLM did not respond in time]"
        except Exception as exc:
            original_response = f"[error: {str(exc)[:200]}]"

        if not isinstance(original_response, str):
            raise HTTPException(status_code=500, detail="Unexpected response type")

        # Enhance with contextual intelligence
        enhanced_response, context_applied = (
            await contextual_intelligence.enhance_chat_response(
                original_response, req.prompt, sid
            )
        )

        # Log conversation
        if VEGA_CORE_AVAILABLE:
            try:
                cid = log_conversation(
                    req.prompt, enhanced_response, source="enhanced_api", session_id=sid
                )
            except Exception as e:
                logger.warning(f"Failed to log conversation: {e}")
                cid = None
        else:
            cid = None

        # Calculate understanding score
        understanding_score = 0.5  # Default fallback
        if contextual_intelligence:
            try:
                score = (
                    contextual_intelligence.understanding_calculator.calculate_understanding_score()
                )
                understanding_score = score.overall_score
            except Exception as e:
                logger.warning(f"Failed to calculate understanding score: {e}")
                # Fallback calculation based on available data
                try:
                    # Simple calculation based on conversation history length
                    if VEGA_CORE_AVAILABLE:
                        history_count = len(get_session_history(sid, limit=50))
                        understanding_score = min(0.9, 0.3 + (history_count * 0.02))
                except Exception:
                    understanding_score = 0.5

        # Generate suggestions
        suggestions = []
        if "calendar" in context_applied.get("context_categories", []):
            suggestions.append(
                "Would you like me to help you prepare for your upcoming events?"
            )
        if "academic" in context_applied.get("context_categories", []):
            suggestions.append("Need help with study planning or deadline management?")

        return ContextualChatResponse(
            response=enhanced_response,
            persona_mode=context_applied.get("persona_mode", "default"),
            context_applied=context_applied,
            understanding_score=understanding_score,
            suggestions=suggestions[:3],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@enhanced_app.get("/profile/summary")
async def get_profile_summary(
    request: Request, x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    """Get user profile summary"""

    # API key validation
    if cfg and x_api_key:
        if x_api_key not in {cfg.api_key, *cfg.api_keys_extra}:
            raise HTTPException(status_code=401, detail="Invalid API key")

    if not USER_PROFILING_AVAILABLE or not profile_manager:
        raise HTTPException(status_code=503, detail="User profiling not available")

    try:
        summary = await profile_manager.get_profile_summary()
        return summary
    except Exception as e:
        logger.error(f"Error getting profile summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@enhanced_app.post("/profile/scan")
async def trigger_profile_scan(
    request: Request,
    scan_type: str = "mini",
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Trigger profile scan"""

    # API key validation
    if cfg and x_api_key:
        if x_api_key not in {cfg.api_key, *cfg.api_keys_extra}:
            raise HTTPException(status_code=401, detail="Invalid API key")

    if not USER_PROFILING_AVAILABLE or not profile_manager:
        raise HTTPException(status_code=503, detail="User profiling not available")

    if scan_type not in ["mini", "full"]:
        raise HTTPException(status_code=400, detail="Invalid scan type")

    try:
        result = await profile_manager.trigger_profile_scan(scan_type)
        return result
    except Exception as e:
        logger.error(f"Error triggering profile scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@enhanced_app.get("/profile/briefing")
async def get_daily_briefing(
    request: Request,
    date: str | None = None,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Get daily briefing"""

    # API key validation
    if cfg and x_api_key:
        if x_api_key not in {cfg.api_key, *cfg.api_keys_extra}:
            raise HTTPException(status_code=401, detail="Invalid API key")

    if not USER_PROFILING_AVAILABLE or not profile_manager:
        raise HTTPException(status_code=503, detail="User profiling not available")

    try:
        briefing = await profile_manager.get_daily_briefing(date)
        return briefing
    except Exception as e:
        logger.error(f"Error getting daily briefing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@enhanced_app.post("/profile/settings")
async def update_profile_settings(
    request: Request,
    settings: Dict[str, Any] = Body(...),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """Update profile settings"""

    # API key validation
    if cfg and x_api_key:
        if x_api_key not in {cfg.api_key, *cfg.api_keys_extra}:
            raise HTTPException(status_code=401, detail="Invalid API key")

    if not USER_PROFILING_AVAILABLE or not profile_manager:
        raise HTTPException(status_code=503, detail="User profiling not available")

    try:
        result = await profile_manager.update_profile_settings(settings)
        return result
    except Exception as e:
        logger.error(f"Error updating profile settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Fallback endpoints for when services aren't available
@enhanced_app.get("/profile/status")
async def get_profiling_status():
    """Get profiling system status"""
    return {
        "user_profiling_available": USER_PROFILING_AVAILABLE,
        "vega_core_available": VEGA_CORE_AVAILABLE,
        "contextual_intelligence": contextual_intelligence is not None,
        "profile_manager": profile_manager is not None,
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting Vega2.0 Enhanced API...")
    print(f"User Profiling Available: {USER_PROFILING_AVAILABLE}")
    print(f"Vega Core Available: {VEGA_CORE_AVAILABLE}")

    uvicorn.run(
        enhanced_app,
        host="127.0.0.1",
        port=8001,  # Use different port from main app
        log_level="info",
    )
