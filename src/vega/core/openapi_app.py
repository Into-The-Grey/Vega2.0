"""
OpenAPI-compliant Vega2.0 API with comprehensive schemas and documentation
"""

from typing import Optional, Dict, Any, List
import uuid
from datetime import datetime
from enum import Enum

from fastapi import FastAPI, Header, HTTPException, Query, Path, Depends, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, validator


# Security
security = HTTPBearer()


# Enums
class ResponseStatus(str, Enum):
    """Response status values"""

    success = "success"
    error = "error"
    degraded = "degraded"


class FeedbackRating(int, Enum):
    """Feedback rating values"""

    very_poor = 1
    poor = 2
    fair = 3
    good = 4
    excellent = 5


class LogLevel(str, Enum):
    """Log level values"""

    debug = "debug"
    info = "info"
    warning = "warning"
    error = "error"
    critical = "critical"


# Base Models
class BaseResponse(BaseModel):
    """Base response model"""

    status: ResponseStatus = Field(..., description="Response status")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )
    message: Optional[str] = Field(None, description="Optional message")


class ErrorResponse(BaseResponse):
    """Error response model"""

    status: ResponseStatus = ResponseStatus.error
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")


# Chat Models
class ChatRequest(BaseModel):
    """Chat request model"""

    prompt: str = Field(..., description="User prompt", example="Hello, how are you?")
    stream: bool = Field(False, description="Enable streaming response")
    session_id: Optional[str] = Field(
        None, description="Session ID for conversation continuity"
    )
    model: Optional[str] = Field(None, description="Specific model to use")
    provider: Optional[str] = Field(None, description="Specific provider to use")
    temperature: Optional[float] = Field(
        None, ge=0.0, le=2.0, description="Generation temperature"
    )
    max_tokens: Optional[int] = Field(
        None, ge=1, le=4096, description="Maximum tokens to generate"
    )

    class Config:
        schema_extra = {
            "example": {
                "prompt": "Explain quantum computing",
                "stream": False,
                "session_id": "session-123",
                "temperature": 0.7,
                "max_tokens": 1000,
            }
        }


class ChatResponse(BaseModel):
    """Chat response model"""

    status: ResponseStatus = ResponseStatus.success
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    response: str = Field(..., description="LLM response")
    session_id: str = Field(..., description="Session ID")
    model_used: Optional[str] = Field(
        None, description="Model that generated the response"
    )
    provider_used: Optional[str] = Field(
        None, description="Provider that generated the response"
    )
    tokens_used: Optional[int] = Field(None, description="Number of tokens used")
    cost_estimate: Optional[float] = Field(None, description="Estimated cost in USD")
    response_time: Optional[float] = Field(None, description="Response time in seconds")


# Proactive Conversation Models
class ProposeRequest(BaseModel):
    """Proactive conversation proposal request"""

    max_per_day: int = Field(5, ge=1, le=50, description="Maximum proposals per day")
    categories: Optional[List[str]] = Field(
        None, description="Topic categories to consider"
    )


class ProposeResponse(BaseResponse):
    """Proactive conversation proposal response"""

    id: Optional[str] = Field(None, description="Proposal ID")
    session_id: Optional[str] = Field(None, description="Session ID if created")
    topic: Optional[str] = Field(None, description="Proposed topic")
    reason: Optional[str] = Field(None, description="Reason for proposal")
    message: Optional[str] = Field(None, description="Proposed message")


class SessionMessage(BaseModel):
    """Session message model"""

    session_id: str = Field(..., description="Session ID")
    text: str = Field(..., description="Message text")


# History Models
class ConversationItem(BaseModel):
    """Single conversation item"""

    id: int = Field(..., description="Conversation ID")
    prompt: str = Field(..., description="User prompt")
    response: str = Field(..., description="LLM response")
    timestamp: datetime = Field(..., description="Conversation timestamp")
    session_id: Optional[str] = Field(None, description="Session ID")
    model_used: Optional[str] = Field(None, description="Model used")
    tokens_used: Optional[int] = Field(None, description="Tokens used")


class HistoryResponse(BaseModel):
    """History response model"""

    status: ResponseStatus = ResponseStatus.success
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    history: List[ConversationItem] = Field(..., description="Conversation history")
    total_count: Optional[int] = Field(None, description="Total conversation count")


class SessionHistoryResponse(BaseModel):
    """Session history response model"""

    status: ResponseStatus = ResponseStatus.success
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: str = Field(..., description="Session ID")
    history: List[ConversationItem] = Field(
        ..., description="Session conversation history"
    )


# Feedback Models
class FeedbackRequest(BaseModel):
    """Feedback submission request"""

    conversation_id: int = Field(..., description="Conversation ID to rate")
    rating: FeedbackRating = Field(..., description="Rating (1-5)")
    comment: Optional[str] = Field(
        None, max_length=1000, description="Optional comment"
    )

    class Config:
        schema_extra = {
            "example": {
                "conversation_id": 123,
                "rating": 4,
                "comment": "Very helpful response",
            }
        }


# Admin Models
class LogsResponse(BaseResponse):
    """Logs response model"""

    modules: List[str] = Field(..., description="Available log modules")


class LogTailResponse(BaseResponse):
    """Log tail response model"""

    module: str = Field(..., description="Log module name")
    lines: List[str] = Field(..., description="Log lines")
    total_lines: int = Field(..., description="Total number of lines")
    level: Optional[LogLevel] = Field(None, description="Log level filter")


class ConfigResponse(BaseResponse):
    """Configuration response model"""

    modules: List[str] = Field(..., description="Available configuration modules")


class ConfigDetailResponse(BaseResponse):
    """Configuration detail response model"""

    module: str = Field(..., description="Configuration module name")
    config: Dict[str, Any] = Field(..., description="Configuration data")


class ConfigUpdateRequest(BaseModel):
    """Configuration update request"""

    config: Dict[str, Any] = Field(..., description="Configuration updates")


class LLMBehaviorResponse(BaseResponse):
    """LLM behavior response model"""

    behavior: Dict[str, Any] = Field(..., description="LLM behavior settings")


# Backup Models
class BackupResponse(BaseResponse):
    """Backup response model"""

    backup_path: str = Field(..., description="Path to created backup")


class BackupListResponse(BaseResponse):
    """Backup list response model"""

    backups: List[Dict[str, Any]] = Field(..., description="Available backups")


class RestoreRequest(BaseModel):
    """Backup restore request"""

    backup_file: str = Field(..., description="Backup file to restore")
    restore_dir: Optional[str] = Field(None, description="Directory to restore to")


# Voice Models
class VoiceProfileResponse(BaseResponse):
    """Voice profile response model"""

    profile: Dict[str, Any] = Field(..., description="Voice profile data")


class VoiceSampleRequest(BaseModel):
    """Voice sample request"""

    file_path: str = Field(..., description="Path to voice sample file")


class VoiceSampleResponse(BaseResponse):
    """Voice sample response model"""

    sample_path: str = Field(..., description="Path to processed voice sample")


# Knowledge Base Models
class KBSiteRequest(BaseModel):
    """Knowledge base site request"""

    category: str = Field(..., description="Site category")
    url: str = Field(..., description="Site URL")

    @validator("url")
    def validate_url(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v


class KBSitesResponse(BaseResponse):
    """Knowledge base sites response model"""

    sites: List[Dict[str, Any]] = Field(..., description="Knowledge base sites")


# Finance Models
class InvestmentRequest(BaseModel):
    """Investment request"""

    symbol: str = Field(..., description="Stock symbol")
    shares: float = Field(..., gt=0, description="Number of shares")
    price: float = Field(..., gt=0, description="Price per share")

    @validator("symbol")
    def validate_symbol(cls, v):
        return v.upper().strip()


class PortfolioResponse(BaseModel):
    """Portfolio response model"""

    status: ResponseStatus = ResponseStatus.success
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    portfolio: List[Dict[str, Any]] = Field(..., description="Investment portfolio")
    total_value: Optional[float] = Field(None, description="Total portfolio value")


class StockPriceResponse(BaseModel):
    """Stock price response model"""

    status: ResponseStatus = ResponseStatus.success
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    symbol: str = Field(..., description="Stock symbol")
    price: float = Field(..., description="Current price")
    currency: str = Field("USD", description="Price currency")
    last_updated: Optional[datetime] = Field(None, description="Last price update")


# Metrics Models
class MetricsResponse(BaseModel):
    """Metrics response model"""

    status: ResponseStatus = ResponseStatus.success
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    requests_total: int = Field(..., description="Total requests")
    responses_total: int = Field(..., description="Total responses")
    errors_total: int = Field(..., description="Total errors")
    degraded: bool = Field(..., description="System degraded status")
    uptime_seconds: Optional[float] = Field(None, description="System uptime")
    llm_stats: Optional[Dict[str, Any]] = Field(
        None, description="LLM provider statistics"
    )


# Health Models
class HealthResponse(BaseModel):
    """Health check response model"""

    status: ResponseStatus = ResponseStatus.success
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    ok: bool = Field(..., description="Health status")
    components: Optional[Dict[str, Any]] = Field(None, description="Component health")


class ReadinessResponse(BaseModel):
    """Readiness check response model"""

    status: ResponseStatus = ResponseStatus.success
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    ready: bool = Field(..., description="Readiness status")
    checks: Optional[Dict[str, Any]] = Field(None, description="Readiness checks")


# Initialize FastAPI app with comprehensive metadata
app = FastAPI(
    title="Vega2.0 API",
    version="2.0.0",
    description="""
    Vega2.0 - Advanced AI Assistant API

    ## Features
    
    * **Chat Interface**: Conversational AI with multiple LLM providers
    * **Proactive Conversations**: AI-initiated conversations
    * **Session Management**: Persistent conversation sessions
    * **Voice Processing**: Text-to-speech and speech-to-text
    * **Knowledge Base**: Web knowledge management
    * **Financial Tracking**: Investment portfolio management
    * **Admin Tools**: System monitoring and configuration
    
    ## Authentication
    
    All API endpoints require authentication using the `X-API-Key` header.
    
    ## Rate Limiting
    
    API calls are rate-limited to prevent abuse. Please implement appropriate
    retry logic with exponential backoff.
    """,
    terms_of_service="https://example.com/terms",
    contact={
        "name": "Vega2.0 Support",
        "url": "https://example.com/contact",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {
            "name": "health",
            "description": "Health and readiness checks",
        },
        {
            "name": "chat",
            "description": "Chat and conversation endpoints",
        },
        {
            "name": "proactive",
            "description": "Proactive conversation management",
        },
        {
            "name": "history",
            "description": "Conversation history and session management",
        },
        {
            "name": "feedback",
            "description": "User feedback and ratings",
        },
        {
            "name": "admin",
            "description": "Administration and monitoring endpoints",
        },
        {
            "name": "backup",
            "description": "Backup and restore operations",
        },
        {
            "name": "voice",
            "description": "Voice processing and profile management",
        },
        {
            "name": "knowledge",
            "description": "Knowledge base management",
        },
        {
            "name": "finance",
            "description": "Financial tracking and portfolio management",
        },
    ],
)

# Mount static files (use absolute path)
import os

static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# Initialize metrics
app.state.metrics = {
    "requests_total": 0,
    "responses_total": 0,
    "errors_total": 0,
    "degraded": False,
    "start_time": datetime.utcnow(),
}


# Custom OpenAPI documentation
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyHeader": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
    }

    # Apply security to all endpoints
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            if method != "get" or "admin" in path:
                openapi_schema["paths"][path][method]["security"] = [
                    {"APIKeyHeader": []}
                ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Authentication dependency
async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    """Verify API key"""
    # This will be implemented with proper config loading
    allowed_keys = {"vega-default-key"}  # Replace with config loading

    if x_api_key not in allowed_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )
    return x_api_key


# Optional API key dependency for public endpoints
async def optional_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """Optional API key verification"""
    return x_api_key


# Health endpoints
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Health check",
    description="Check if the API is running and healthy",
)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status=ResponseStatus.success,
        ok=True,
        components={"api": "healthy", "database": "healthy", "llm": "healthy"},
    )


@app.get(
    "/ready",
    response_model=ReadinessResponse,
    tags=["health"],
    summary="Readiness check",
    description="Check if the API is ready to serve requests",
)
async def ready():
    """Readiness check endpoint"""
    # Add actual readiness checks here
    return ReadinessResponse(
        status=ResponseStatus.success,
        ready=True,
        checks={"database": "ready", "llm_provider": "ready", "external_apis": "ready"},
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["health"],
    summary="System metrics",
    description="Get system performance metrics",
)
async def metrics():
    """Get system metrics"""
    current_time = datetime.utcnow()
    uptime = (current_time - app.state.metrics["start_time"]).total_seconds()

    return MetricsResponse(
        status=ResponseStatus.success,
        requests_total=app.state.metrics["requests_total"],
        responses_total=app.state.metrics["responses_total"],
        errors_total=app.state.metrics["errors_total"],
        degraded=app.state.metrics["degraded"],
        uptime_seconds=uptime,
        llm_stats={},  # Add LLM statistics here
    )


# Legacy health endpoints for backward compatibility
@app.get("/healthz", include_in_schema=False)
async def healthz():
    """Legacy health check"""
    return {"ok": True, "timestamp": datetime.utcnow().isoformat()}


@app.get("/livez", include_in_schema=False)
async def livez():
    """Legacy liveness check"""
    return {"alive": True}


@app.get("/readyz", include_in_schema=False)
async def readyz():
    """Legacy readiness check"""
    return {"ready": True}


# Root endpoint
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Root endpoint with navigation"""
    return HTMLResponse(
        """
    <html>
        <head>
            <title>Vega2.0 API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .nav { margin: 20px 0; }
                .nav a { margin-right: 20px; text-decoration: none; color: #0066cc; }
                .nav a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <h1>Vega2.0 API</h1>
            <p>Welcome to the Vega2.0 Advanced AI Assistant API</p>
            
            <div class="nav">
                <a href="/docs">API Documentation</a>
                <a href="/redoc">ReDoc Documentation</a>
                <a href="/health">Health Check</a>
                <a href="/metrics">Metrics</a>
                <a href="/static/index.html">Web Interface</a>
            </div>
            
            <h2>Quick Start</h2>
            <p>Get started by visiting the <a href="/docs">interactive API documentation</a>.</p>
            
            <h2>Authentication</h2>
            <p>All API calls require the <code>X-API-Key</code> header.</p>
        </body>
    </html>
    """
    )


# Chat endpoints
@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["chat"],
    summary="Send chat message",
    description="Send a message to the AI assistant and get a response",
    responses={
        200: {"description": "Successful chat response"},
        401: {"description": "Invalid API key"},
        503: {"description": "LLM backend unavailable"},
    },
)
async def chat(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    """Send a chat message to the AI assistant"""
    app.state.metrics["requests_total"] += 1

    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())

    try:
        # This would be replaced with actual LLM integration
        response_text = f"Echo: {request.prompt}"

        # Log conversation (placeholder)
        # log_conversation(request.prompt, response_text, session_id)

        app.state.metrics["responses_total"] += 1

        return ChatResponse(
            response=response_text,
            session_id=session_id,
            model_used=request.model or "default",
            provider_used=request.provider or "default",
            tokens_used=len(request.prompt.split()) + len(response_text.split()),
            cost_estimate=0.001,
            response_time=0.5,
        )

    except Exception as e:
        app.state.metrics["errors_total"] += 1
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM backend unavailable",
        ) from e


# Proactive conversation endpoints
@app.post(
    "/proactive/propose",
    response_model=ProposeResponse,
    tags=["proactive"],
    summary="Propose proactive conversation",
    description="Request AI to propose a proactive conversation topic",
)
async def proactive_propose(
    request: ProposeRequest, api_key: str = Depends(verify_api_key)
):
    """Propose a proactive conversation"""
    try:
        # Placeholder implementation
        proposal_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())

        return ProposeResponse(
            status=ResponseStatus.success,
            id=proposal_id,
            session_id=session_id,
            topic="Technology Trends",
            reason="User interest in AI developments",
            message="Would you like to discuss the latest AI advancements?",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e


@app.get(
    "/proactive/pending",
    response_model=List[ProposeResponse],
    tags=["proactive"],
    summary="List pending proposals",
    description="Get list of pending proactive conversation proposals",
)
async def proactive_pending(api_key: str = Depends(verify_api_key)):
    """List pending proactive conversation proposals"""
    try:
        # Placeholder implementation
        return []
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e


@app.post(
    "/proactive/accept/{proposal_id}",
    response_model=BaseResponse,
    tags=["proactive"],
    summary="Accept proposal",
    description="Accept a proactive conversation proposal",
)
async def proactive_accept(
    proposal_id: str = Path(..., description="Proposal ID to accept"),
    api_key: str = Depends(verify_api_key),
):
    """Accept a proactive conversation proposal"""
    try:
        session_id = str(uuid.uuid4())
        return BaseResponse(
            status=ResponseStatus.success,
            message=f"Accepted proposal {proposal_id}, session {session_id}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e


@app.post(
    "/proactive/decline/{proposal_id}",
    response_model=BaseResponse,
    tags=["proactive"],
    summary="Decline proposal",
    description="Decline a proactive conversation proposal",
)
async def proactive_decline(
    proposal_id: str = Path(..., description="Proposal ID to decline"),
    api_key: str = Depends(verify_api_key),
):
    """Decline a proactive conversation proposal"""
    try:
        return BaseResponse(
            status=ResponseStatus.success, message=f"Declined proposal {proposal_id}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e


@app.post(
    "/proactive/send",
    response_model=BaseResponse,
    tags=["proactive"],
    summary="Send proactive message",
    description="Send a message in a proactive conversation session",
)
async def proactive_send(
    message: SessionMessage, api_key: str = Depends(verify_api_key)
):
    """Send a message in a proactive conversation session"""
    try:
        # Placeholder implementation
        return BaseResponse(
            status=ResponseStatus.success,
            message=f"Message sent to session {message.session_id}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e


@app.get(
    "/proactive/session/{session_id}",
    response_model=SessionHistoryResponse,
    tags=["proactive"],
    summary="Get proactive session",
    description="Get details of a proactive conversation session",
)
async def proactive_get_session(
    session_id: str = Path(..., description="Session ID"),
    api_key: str = Depends(verify_api_key),
):
    """Get proactive conversation session details"""
    try:
        # Placeholder implementation
        return SessionHistoryResponse(
            status=ResponseStatus.success, session_id=session_id, history=[]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e


@app.post(
    "/proactive/end/{session_id}",
    response_model=BaseResponse,
    tags=["proactive"],
    summary="End proactive session",
    description="End a proactive conversation session",
)
async def proactive_end(
    session_id: str = Path(..., description="Session ID to end"),
    api_key: str = Depends(verify_api_key),
):
    """End a proactive conversation session"""
    try:
        return BaseResponse(
            status=ResponseStatus.success, message=f"Ended session {session_id}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e


# History endpoints
@app.get(
    "/history",
    response_model=HistoryResponse,
    tags=["history"],
    summary="Get conversation history",
    description="Get conversation history with pagination",
)
async def history(
    limit: int = Query(50, ge=1, le=1000, description="Number of items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    api_key: str = Depends(verify_api_key),
):
    """Get conversation history"""
    try:
        # Placeholder implementation
        return HistoryResponse(status=ResponseStatus.success, history=[], total_count=0)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e


@app.get(
    "/session/{session_id}/history",
    response_model=SessionHistoryResponse,
    tags=["history"],
    summary="Get session history",
    description="Get conversation history for a specific session",
)
async def session_history(
    session_id: str = Path(..., description="Session ID"),
    limit: int = Query(50, ge=1, le=1000, description="Number of items to return"),
    api_key: str = Depends(verify_api_key),
):
    """Get session conversation history"""
    try:
        # Placeholder implementation
        return SessionHistoryResponse(
            status=ResponseStatus.success, session_id=session_id, history=[]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e


# Feedback endpoints
@app.post(
    "/feedback",
    response_model=BaseResponse,
    tags=["feedback"],
    summary="Submit feedback",
    description="Submit feedback for a conversation",
)
async def submit_feedback(
    feedback: FeedbackRequest, api_key: str = Depends(verify_api_key)
):
    """Submit feedback for a conversation"""
    try:
        # Placeholder implementation
        return BaseResponse(
            status=ResponseStatus.success, message="Feedback submitted successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("openapi_app:app", host="127.0.0.1", port=8000, reload=True)
