"""
Document Intelligence API Integration
====================================

FastAPI router for document intelligence capabilities.
Provides endpoints for document processing, analysis, and batch operations.
"""

from typing import Dict, List, Optional, Any, Union
from enum import Enum
import uuid
import asyncio
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, Header
from pydantic import BaseModel, Field, validator
import httpx

# Document intelligence modules
from .understanding import DocumentUnderstandingAI
from .classification import DocumentClassificationAI
from .workflow import DocumentWorkflowAI
from .legal import LegalDocumentAI
from .technical import TechnicalDocumentationAI
from .base import ProcessingContext, ProcessingResult

# Configuration and security
try:
    from ..core.config import get_config
except ImportError:

    def get_config():
        return type("Config", (), {"api_key": "vega-default-key"})()


# Create router
router = APIRouter(prefix="/document", tags=["Document Intelligence"])


# Request/Response Models
class DocumentType(str, Enum):
    LEGAL = "legal"
    TECHNICAL = "technical"
    GENERAL = "general"
    CONTRACT = "contract"
    POLICY = "policy"
    API_DOC = "api_doc"
    CODE_DOC = "code_doc"


class ProcessingMode(str, Enum):
    ANALYSIS = "analysis"
    CLASSIFICATION = "classification"
    UNDERSTANDING = "understanding"
    WORKFLOW = "workflow"
    FULL = "full"


class DocumentProcessRequest(BaseModel):
    """Request model for document processing"""

    content: str = Field(..., description="Document content to process")
    document_type: DocumentType = Field(
        DocumentType.GENERAL, description="Type of document"
    )
    processing_mode: ProcessingMode = Field(
        ProcessingMode.ANALYSIS, description="Processing mode"
    )
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    session_id: Optional[str] = Field(None, description="Session identifier")

    @validator("content")
    def content_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Content cannot be empty")
        return v


class BatchProcessRequest(BaseModel):
    """Request model for batch document processing"""

    documents: List[DocumentProcessRequest] = Field(
        ..., description="Documents to process"
    )
    parallel: bool = Field(True, description="Process documents in parallel")
    max_concurrent: int = Field(
        5, description="Maximum concurrent processing", ge=1, le=20
    )


class DocumentProcessResponse(BaseModel):
    """Response model for document processing"""

    session_id: str
    document_type: DocumentType
    processing_mode: ProcessingMode
    result: Dict[str, Any]
    processing_time: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class BatchProcessResponse(BaseModel):
    """Response model for batch processing"""

    total_documents: int
    successful: int
    failed: int
    results: List[DocumentProcessResponse]
    errors: List[Dict[str, Any]]
    total_processing_time: float


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    components: Dict[str, Dict[str, Any]]
    timestamp: datetime


# Module instances (initialized on startup)
understanding_ai = None
classification_ai = None
workflow_ai = None
legal_ai = None
technical_ai = None


async def get_module_instance(
    document_type: DocumentType, processing_mode: ProcessingMode
):
    """Get appropriate module instance based on document type and processing mode"""
    global understanding_ai, classification_ai, workflow_ai, legal_ai, technical_ai

    # Initialize modules lazily
    if understanding_ai is None:
        understanding_ai = DocumentUnderstandingAI()
        await understanding_ai.initialize()

    if classification_ai is None:
        classification_ai = DocumentClassificationAI()
        await classification_ai.initialize()

    if workflow_ai is None:
        workflow_ai = DocumentWorkflowAI()
        await workflow_ai.initialize()

    if legal_ai is None:
        legal_ai = LegalDocumentAI()
        await legal_ai.initialize()

    if technical_ai is None:
        technical_ai = TechnicalDocumentationAI()
        await technical_ai.initialize()

    # Select appropriate module
    if document_type in [
        DocumentType.LEGAL,
        DocumentType.CONTRACT,
        DocumentType.POLICY,
    ]:
        return legal_ai
    elif document_type in [
        DocumentType.TECHNICAL,
        DocumentType.API_DOC,
        DocumentType.CODE_DOC,
    ]:
        return technical_ai
    elif processing_mode == ProcessingMode.CLASSIFICATION:
        return classification_ai
    elif processing_mode == ProcessingMode.UNDERSTANDING:
        return understanding_ai
    elif processing_mode == ProcessingMode.WORKFLOW:
        return workflow_ai
    else:
        return understanding_ai  # Default fallback


def require_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    """Validate API key for protected endpoints"""
    config = get_config()
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    allowed_keys = [config.api_key] + getattr(config, "api_keys_extra", [])
    if x_api_key not in allowed_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")


# Health and Status Endpoints
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check health of document intelligence components"""
    components = {}

    # Check each module
    modules = [
        ("understanding", understanding_ai),
        ("classification", classification_ai),
        ("workflow", workflow_ai),
        ("legal", legal_ai),
        ("technical", technical_ai),
    ]

    for name, module in modules:
        if module is None:
            components[name] = {"status": "not_initialized", "ready": False}
        else:
            try:
                health = await module.health_check()
                components[name] = {
                    "status": (
                        "healthy" if health.get("healthy", False) else "unhealthy"
                    ),
                    "ready": health.get("ready", False),
                    "details": health,
                }
            except Exception as e:
                components[name] = {"status": "error", "ready": False, "error": str(e)}

    overall_status = (
        "healthy"
        if all(comp["status"] == "healthy" for comp in components.values())
        else "degraded"
    )

    return HealthResponse(
        status=overall_status, components=components, timestamp=datetime.utcnow()
    )


@router.get("/status")
async def status():
    """Simple status check"""
    return {"status": "operational", "timestamp": datetime.utcnow()}


# Document Processing Endpoints
@router.post("/process", response_model=DocumentProcessResponse)
async def process_document(
    request: DocumentProcessRequest, x_api_key: str = Header(None, alias="X-API-Key")
):
    """Process a single document with specified analysis type"""
    require_api_key(x_api_key)

    start_time = datetime.utcnow()
    session_id = request.session_id or str(uuid.uuid4())

    try:
        # Get appropriate module
        module = await get_module_instance(
            request.document_type, request.processing_mode
        )

        # Create processing context
        context = ProcessingContext(
            document_content=request.content,
            document_type=request.document_type.value,
            processing_mode=request.processing_mode.value,
            session_id=session_id,
            metadata=request.metadata or {},
        )

        # Process document
        result = await module.process_document(context)

        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        return DocumentProcessResponse(
            session_id=session_id,
            document_type=request.document_type,
            processing_mode=request.processing_mode,
            result=result.results,
            processing_time=processing_time,
            timestamp=datetime.utcnow(),
            metadata=result.metadata,
        )

    except Exception as e:
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "processing_time": processing_time,
                "session_id": session_id,
            },
        )


@router.post("/batch", response_model=BatchProcessResponse)
async def batch_process_documents(
    request: BatchProcessRequest, x_api_key: str = Header(None, alias="X-API-Key")
):
    """Process multiple documents in batch"""
    require_api_key(x_api_key)

    start_time = datetime.utcnow()
    results = []
    errors = []
    successful = 0
    failed = 0

    if request.parallel:
        # Process documents in parallel with concurrency limit
        semaphore = asyncio.Semaphore(request.max_concurrent)

        async def process_single(doc_request: DocumentProcessRequest):
            async with semaphore:
                try:
                    response = await process_document(doc_request, x_api_key)
                    return response, None
                except Exception as e:
                    return None, {
                        "document_index": len(results) + len(errors),
                        "error": str(e),
                        "session_id": doc_request.session_id,
                    }

        # Execute all processing tasks
        tasks = [process_single(doc) for doc in request.documents]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        for result, error in task_results:
            if result:
                results.append(result)
                successful += 1
            if error:
                errors.append(error)
                failed += 1
    else:
        # Process documents sequentially
        for i, doc_request in enumerate(request.documents):
            try:
                response = await process_document(doc_request, x_api_key)
                results.append(response)
                successful += 1
            except Exception as e:
                errors.append(
                    {
                        "document_index": i,
                        "error": str(e),
                        "session_id": doc_request.session_id,
                    }
                )
                failed += 1

    total_processing_time = (datetime.utcnow() - start_time).total_seconds()

    return BatchProcessResponse(
        total_documents=len(request.documents),
        successful=successful,
        failed=failed,
        results=results,
        errors=errors,
        total_processing_time=total_processing_time,
    )


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    document_type: DocumentType = Form(DocumentType.GENERAL),
    processing_mode: ProcessingMode = Form(ProcessingMode.ANALYSIS),
    session_id: Optional[str] = Form(None),
    x_api_key: str = Header(None, alias="X-API-Key"),
):
    """Upload and process a document file"""
    require_api_key(x_api_key)

    # Validate file type
    allowed_types = {
        "text/plain",
        "text/markdown",
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, detail=f"Unsupported file type: {file.content_type}"
        )

    try:
        # Read file content
        content = await file.read()

        # For now, handle only text files
        # TODO: Add PDF, DOC, DOCX parsing
        if file.content_type in ["text/plain", "text/markdown"]:
            text_content = content.decode("utf-8")
        else:
            raise HTTPException(
                status_code=501,
                detail=f"File type {file.content_type} parsing not implemented yet",
            )

        # Create processing request
        request = DocumentProcessRequest(
            content=text_content,
            document_type=document_type,
            processing_mode=processing_mode,
            session_id=session_id,
            metadata={
                "filename": file.filename,
                "content_type": file.content_type,
                "file_size": len(content),
            },
        )

        # Process document
        return await process_document(request, x_api_key)

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400, detail="Unable to decode file content as UTF-8"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing uploaded file: {str(e)}"
        )


# Specialized Processing Endpoints
@router.post("/analyze/legal")
async def analyze_legal_document(
    content: str = Form(...),
    analysis_type: str = Form("full"),
    session_id: Optional[str] = Form(None),
    x_api_key: str = Header(None, alias="X-API-Key"),
):
    """Specialized legal document analysis"""
    require_api_key(x_api_key)

    request = DocumentProcessRequest(
        content=content,
        document_type=DocumentType.LEGAL,
        processing_mode=ProcessingMode.FULL,
        session_id=session_id,
        metadata={"analysis_type": analysis_type},
    )

    return await process_document(request, x_api_key)


@router.post("/analyze/technical")
async def analyze_technical_document(
    content: str = Form(...),
    doc_type: str = Form("general"),
    session_id: Optional[str] = Form(None),
    x_api_key: str = Header(None, alias="X-API-Key"),
):
    """Specialized technical document analysis"""
    require_api_key(x_api_key)

    document_type = DocumentType.TECHNICAL
    if doc_type == "api":
        document_type = DocumentType.API_DOC
    elif doc_type == "code":
        document_type = DocumentType.CODE_DOC

    request = DocumentProcessRequest(
        content=content,
        document_type=document_type,
        processing_mode=ProcessingMode.FULL,
        session_id=session_id,
        metadata={"doc_type": doc_type},
    )

    return await process_document(request, x_api_key)


@router.post("/classify")
async def classify_document(
    content: str = Form(...),
    confidence_threshold: float = Form(0.7),
    session_id: Optional[str] = Form(None),
    x_api_key: str = Header(None, alias="X-API-Key"),
):
    """Document classification endpoint"""
    require_api_key(x_api_key)

    request = DocumentProcessRequest(
        content=content,
        document_type=DocumentType.GENERAL,
        processing_mode=ProcessingMode.CLASSIFICATION,
        session_id=session_id,
        metadata={"confidence_threshold": confidence_threshold},
    )

    return await process_document(request, x_api_key)


# Session and History Endpoints
@router.get("/sessions")
async def list_sessions(
    limit: int = Query(50, ge=1, le=1000),
    x_api_key: str = Header(None, alias="X-API-Key"),
):
    """List document processing sessions"""
    require_api_key(x_api_key)

    # TODO: Implement session storage and retrieval
    return {
        "sessions": [],
        "total": 0,
        "limit": limit,
        "message": "Session storage not implemented yet",
    }


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str, x_api_key: str = Header(None, alias="X-API-Key")
):
    """Get specific session details"""
    require_api_key(x_api_key)

    # TODO: Implement session retrieval
    return {
        "session_id": session_id,
        "documents": [],
        "message": "Session storage not implemented yet",
    }


# Configuration and Management Endpoints
@router.get("/config")
async def get_document_config(x_api_key: str = Header(None, alias="X-API-Key")):
    """Get document intelligence configuration"""
    require_api_key(x_api_key)

    return {
        "supported_types": [dt.value for dt in DocumentType],
        "processing_modes": [pm.value for pm in ProcessingMode],
        "file_types": [
            "text/plain",
            "text/markdown",
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ],
        "max_concurrent_batch": 20,
        "modules": {
            "understanding": understanding_ai is not None,
            "classification": classification_ai is not None,
            "workflow": workflow_ai is not None,
            "legal": legal_ai is not None,
            "technical": technical_ai is not None,
        },
    }


# Cleanup and shutdown
@router.on_event("shutdown")
async def shutdown_document_modules():
    """Cleanup document intelligence modules on shutdown"""
    modules = [understanding_ai, classification_ai, workflow_ai, legal_ai, technical_ai]

    for module in modules:
        if module and hasattr(module, "cleanup"):
            try:
                await module.cleanup()
            except Exception as e:
                print(f"Error during module cleanup: {e}")
