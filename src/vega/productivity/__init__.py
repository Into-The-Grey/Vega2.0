"""
Vega 2.0 Productivity Package

This package provides AI-powered productivity tools including:
- Smart task management with prioritization and deadline prediction
- Personal knowledge base with automated knowledge extraction
- Focus and attention tracking
- Meeting intelligence and automation
- Research assistance
- Habit and routine optimization
"""

from typing import Optional

# Smart task management
try:
    from .task_manager import (
        TaskManager,
        Task,
        TaskPriority,
        TaskStatus,
        TaskCategory,
        DeadlinePredictor,
        WorkloadOptimizer,
    )

    TASK_MANAGER_AVAILABLE = True
except ImportError as e:
    TASK_MANAGER_AVAILABLE = False
    TaskManager = Task = TaskPriority = TaskStatus = TaskCategory = None
    DeadlinePredictor = WorkloadOptimizer = None

# Personal knowledge base
try:
    from .knowledge_base import (
        KnowledgeBase,
        KnowledgeEntry,
        KnowledgeType,
        KnowledgeSource,
        Concept,
        ConceptGraph,
        KnowledgeExtractor,
        SemanticSearchEngine,
    )

    KNOWLEDGE_BASE_AVAILABLE = True
except ImportError as e:
    KNOWLEDGE_BASE_AVAILABLE = False
    KnowledgeBase = KnowledgeEntry = KnowledgeType = KnowledgeSource = None
    Concept = ConceptGraph = KnowledgeExtractor = SemanticSearchEngine = None

__all__ = [
    # Task management
    "TaskManager",
    "Task",
    "TaskPriority",
    "TaskStatus",
    "TaskCategory",
    "DeadlinePredictor",
    "WorkloadOptimizer",
    "TASK_MANAGER_AVAILABLE",
    # Knowledge base
    "KnowledgeBase",
    "KnowledgeEntry",
    "KnowledgeType",
    "KnowledgeSource",
    "Concept",
    "ConceptGraph",
    "KnowledgeExtractor",
    "SemanticSearchEngine",
    "KNOWLEDGE_BASE_AVAILABLE",
]
