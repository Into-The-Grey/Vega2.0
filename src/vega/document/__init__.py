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
    DocumentClassificationAI,
    ContentClassifier,
    TopicClassifier,
    HierarchicalClassifier,
    IntentClassifier,
    ClassificationConfig,
    ClassificationCategory,
    ClassificationResult,
    ClassificationProcessingResult,
    HierarchicalCategory,
)

from .workflow import (
    WorkflowManager,
    DocumentRouter,
    ApprovalManager,
    WorkflowScheduler,
    WorkflowDefinition,
    WorkflowContext,
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
    TechnicalConfig,
    DocumentationConfig,
)

from .legal_config import LegalConfig, ContractConfig

# Configuration and utilities
from .understanding import DocumentConfig, LayoutConfig

__all__ = [
    # Document Understanding AI
    "DocumentUnderstandingAI",
    "LayoutAnalyzer",
    "TableExtractor",
    "FormRecognizer",
    "DocumentStructureAnalyzer",
    # Document Classification
    "DocumentClassificationAI",
    "ContentClassifier",
    "TopicClassifier",
    "HierarchicalClassifier",
    "IntentClassifier",
    # Document Workflow Automation
    "WorkflowManager",
    "DocumentRouter",
    "ApprovalManager",
    "WorkflowScheduler",
    "WorkflowDefinition",
    "WorkflowContext",
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
    "ClassificationCategory",
    "ClassificationResult",
    "ClassificationProcessingResult",
    "HierarchicalCategory",
    "LegalConfig",
    "ContractConfig",
    "TechnicalConfig",
    "DocumentationConfig",
]

__version__ = "2.0.0"
