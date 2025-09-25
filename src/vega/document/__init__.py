"""
Vega 2.0 Document Intelligence Module

Advanced document processing and analysis capabilities including:
- Document Understanding AI with layout analysis and form recognition
- Intelligent document classification and clustering
- Document workflow automation and processing pipelines
- Legal document analysis and contract processing
- Technical documentation AI and code analysis

This module provides comprehensive document intelligence tools
for the Vega2.0 AI platform.
"""

# Core Document Intelligence Classes
from .understanding import (
    DocumentUnderstandingAI,
    LayoutAnalyzer,
    TableExtractor,
    FormRecognizer,
    DocumentStructureAnalyzer,
)

from .classification import (
    DocumentClassifier,
    TopicModeler,
    DocumentClusterer,
    CategoryPredictor,
    ContentAnalyzer,
)

from .automation import (
    DocumentWorkflowManager,
    SmartRouter,
    ApprovalWorkflow,
    ProcessingPipeline,
    WorkflowOrchestrator,
)

from .legal import (
    LegalDocumentAnalyzer,
    ContractAnalyzer,
    ClauseExtractor,
    LegalEntityRecognizer,
    ComplianceChecker,
)

from .technical import (
    TechnicalDocumentationAI,
    CodeDocumentationGenerator,
    APIDocumentationAnalyzer,
    TechnicalWritingAssistant,
    DocumentationQualityAnalyzer,
)

# Configuration and utilities
from .understanding import DocumentConfig, LayoutConfig
from .classification import ClassificationConfig, TopicConfig
from .automation import WorkflowConfig, ProcessingConfig
from .legal import LegalConfig, ContractConfig
from .technical import TechnicalConfig, DocumentationConfig

__all__ = [
    # Document Understanding AI
    "DocumentUnderstandingAI",
    "LayoutAnalyzer",
    "TableExtractor",
    "FormRecognizer",
    "DocumentStructureAnalyzer",
    # Document Classification
    "DocumentClassifier",
    "TopicModeler",
    "DocumentClusterer",
    "CategoryPredictor",
    "ContentAnalyzer",
    # Document Workflow Automation
    "DocumentWorkflowManager",
    "SmartRouter",
    "ApprovalWorkflow",
    "ProcessingPipeline",
    "WorkflowOrchestrator",
    # Legal Document Analysis
    "LegalDocumentAnalyzer",
    "ContractAnalyzer",
    "ClauseExtractor",
    "LegalEntityRecognizer",
    "ComplianceChecker",
    # Technical Documentation AI
    "TechnicalDocumentationAI",
    "CodeDocumentationGenerator",
    "APIDocumentationAnalyzer",
    "TechnicalWritingAssistant",
    "DocumentationQualityAnalyzer",
    # Configuration Classes
    "DocumentConfig",
    "LayoutConfig",
    "ClassificationConfig",
    "TopicConfig",
    "WorkflowConfig",
    "ProcessingConfig",
    "LegalConfig",
    "ContractConfig",
    "TechnicalConfig",
    "DocumentationConfig",
]

__version__ = "2.0.0"
