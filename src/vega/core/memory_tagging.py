"""
memory_tagging.py - Lightweight Task-Based Memory Tagging System

Provides task-specific memory indexing to help Vega remember which knowledge
was useful for particular types of tasks, enabling faster retrieval without
retraining or extensive searching.

Features:
- Task type detection from prompts
- Tagging knowledge items with task contexts
- Fast retrieval of task-relevant knowledge
- Automatic relevance scoring
- Usage pattern tracking
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Any

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    select,
    update,
    func,
)
from sqlalchemy.orm import Session

from .memory import Base, memory_engine, KnowledgeItem, memory_logger

logger = logging.getLogger("vega.memory_tagging")


class TaskMemory(Base):
    """Tracks which knowledge items were useful for specific task types"""

    __tablename__ = "task_memory"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_type = Column(String(100), nullable=False, index=True)
    task_hash = Column(String(64), nullable=False, index=True)  # Hash of task context
    knowledge_item_id = Column(Integer, ForeignKey("knowledge_items.id"), nullable=False)
    relevance_score = Column(Float, default=1.0)  # How useful was this knowledge (0.0-1.0)
    used_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    success = Column(Boolean, default=True)  # Was the task successful?
    context_snippet = Column(String(500), nullable=True)  # Brief context of usage

    __table_args__ = (
        Index("idx_task_type_score", "task_type", "relevance_score"),
        Index("idx_task_hash_item", "task_hash", "knowledge_item_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<TaskMemory(id={self.id}, task_type='{self.task_type}', "
            f"knowledge_item_id={self.knowledge_item_id}, relevance={self.relevance_score})>"
        )


# Create table if it doesn't exist
Base.metadata.create_all(memory_engine)


class TaskType:
    """
    Common task types for classification.

    Use these constants to ensure consistent task type naming across the system.
    """

    CODE_REVIEW: str = "code_review"
    DEBUGGING: str = "debugging"
    EXPLANATION: str = "explanation"
    REFACTORING: str = "refactoring"
    FEATURE_IMPLEMENTATION: str = "feature_implementation"
    DOCUMENTATION: str = "documentation"
    TESTING: str = "testing"
    OPTIMIZATION: str = "optimization"
    SECURITY_ANALYSIS: str = "security_analysis"
    API_DESIGN: str = "api_design"
    DATA_ANALYSIS: str = "data_analysis"
    ARCHITECTURE_DESIGN: str = "architecture_design"
    GENERAL_QUESTION: str = "general_question"

    @classmethod
    def all_types(cls) -> List[str]:
        """Get all defined task types."""
        return [
            cls.CODE_REVIEW,
            cls.DEBUGGING,
            cls.EXPLANATION,
            cls.REFACTORING,
            cls.FEATURE_IMPLEMENTATION,
            cls.DOCUMENTATION,
            cls.TESTING,
            cls.OPTIMIZATION,
            cls.SECURITY_ANALYSIS,
            cls.API_DESIGN,
            cls.DATA_ANALYSIS,
            cls.ARCHITECTURE_DESIGN,
            cls.GENERAL_QUESTION,
        ]

    @classmethod
    def is_valid(cls, task_type: str) -> bool:
        """Check if a task type is valid."""
        return task_type in cls.all_types()

    UNKNOWN = "unknown"


# Task detection patterns (heuristic-based)
TASK_PATTERNS = {
    TaskType.CODE_REVIEW: [
        r"\breview\s+(?:this|the|my)\s+code\b",
        r"\bcheck\s+(?:this|the|my)\s+(?:code|implementation)\b",
        r"\blook\s+at\s+(?:this|the|my)\s+code\b",
        r"\bcode\s+review\b",
        r"\bis\s+this\s+code\s+(?:good|correct|right)\b",
    ],
    TaskType.DEBUGGING: [
        r"\b(?:debug|fix|solve|resolve)\b",
        r"\berror\b.*\b(?:how|why|what|in)\b",
        r"\bgetting\s+an?\s+error\b",
        r"\b(?:issue|problem|bug)\b",
        r"\bnot\s+working\b",
        r"\bfailing\b",
        r"\bexception\b",
        r"\bcrash(?:ing|ed)?\b",
    ],
    TaskType.EXPLANATION: [
        r"\bexplain\b",
        r"\bwhat\s+(?:is|are|does)\b",
        r"\bhow\s+(?:does|do)\b",
        r"\bwhy\s+(?:is|are|does)\b",
        r"\bunderstand\b",
        r"\bmeaning\s+of\b",
        r"\btell\s+me\s+about\b",
    ],
    TaskType.REFACTORING: [
        r"\brefactor\b",
        r"\bimprove\s+(?:this|the)?\s*(?:code|implementation)\b",
        r"\bclean\s+up\b",
        r"\breorganize\b",
        r"\brestructure\b",
        r"\boptimize\s+structure\b",
        r"\bhow\s+(?:can|do)\s+I\s+improve\b",
    ],
    TaskType.FEATURE_IMPLEMENTATION: [
        r"\b(?:implement|add|create)\s+(?:a|an|the)?\s*(?:new\s+)?feature\b",
        r"\b(?:build|make|develop)\s+(?:a|an)\b",
        r"\bhow\s+(?:to|can\s+I)\s+(?:implement|add|create)\b",
        r"\bneed\s+to\s+(?:implement|add|create)\b",
    ],
    TaskType.DOCUMENTATION: [
        r"\b(?:document|documentation)\b",
        r"\bwrite\s+(?:docs|documentation)\b",
        r"\badd\s+comments\b",
        r"\bdocstring\b",
        r"\bREADME\b",
    ],
    TaskType.TESTING: [
        r"\btest\b",
        r"\bunit\s+test\b",
        r"\bintegration\s+test\b",
        r"\btest\s+case\b",
        r"\bpytest\b",
        r"\bunittest\b",
    ],
    TaskType.OPTIMIZATION: [
        r"\boptimize\b",
        r"\bperformance\b",
        r"\bfaster\b",
        r"\bspeed\s+up\b",
        r"\befficiency\b",
        r"\bmemory\s+usage\b",
    ],
    TaskType.SECURITY_ANALYSIS: [
        r"\bsecurity\b",
        r"\bvulnerabilit(?:y|ies)\b",
        r"\bsecure\b",
        r"\bCVE\b",
        r"\bexploit\b",
        r"\battack\b",
    ],
    TaskType.API_DESIGN: [
        r"\bAPI\s+design\b",
        r"\bendpoint\b",
        r"\bREST\s+API\b",
        r"\bGraphQL\b",
        r"\broute\b",
    ],
    TaskType.DATA_ANALYSIS: [
        r"\banalyze\s+data\b",
        r"\bdata\s+analysis\b",
        r"\bstatistics\b",
        r"\bvisuali[sz]e\s+data\b",
        r"\bdataframe\b",
    ],
    TaskType.ARCHITECTURE_DESIGN: [
        r"\barchitecture\b",
        r"\bdesign\s+(?:pattern|system)\b",
        r"\bstructure\s+(?:of|for)\b",
        r"\bscalability\b",
        r"\bmicroservices\b",
    ],
}


def detect_task_type(prompt: str, method: str = "heuristic") -> str:
    """
    Detect the type of task from a prompt.

    Args:
        prompt: The user's prompt/question
        method: Detection method ("heuristic", "llm", "hybrid")

    Returns:
        Task type string (e.g., "debugging", "code_review")
    """
    if method == "heuristic":
        return _detect_task_heuristic(prompt)
    elif method == "llm":
        return _detect_task_llm(prompt)
    elif method == "hybrid":
        # Try heuristic first, fall back to LLM if unknown
        task = _detect_task_heuristic(prompt)
        if task == TaskType.UNKNOWN:
            task = _detect_task_llm(prompt)
        return task
    else:
        return TaskType.UNKNOWN


def _detect_task_heuristic(prompt: str) -> str:
    """Detect task type using pattern matching"""
    prompt_lower = prompt.lower()

    # Score each task type based on pattern matches
    scores: Dict[str, int] = {}

    for task_type, patterns in TASK_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                score += 1
        if score > 0:
            scores[task_type] = score

    # Return highest scoring task type
    if scores:
        best_task = max(scores, key=scores.get)
        memory_logger.debug(f"Detected task type: {best_task} (score: {scores[best_task]})")
        return best_task

    # Default to general question
    return TaskType.GENERAL_QUESTION


def _detect_task_llm(prompt: str) -> str:
    """Detect task type using LLM (placeholder for future implementation)"""
    # NOTE: LLM-based task detection not yet implemented
    # This would send a classification request to the LLM
    # For now, fall back to heuristic
    memory_logger.warning("LLM-based task detection not yet implemented, using heuristic")
    return _detect_task_heuristic(prompt)


def tag_knowledge_for_task(
    task_type: str,
    task_context: str,
    knowledge_item_ids: List[int],
    relevance_score: float = 1.0,
    success: bool = True,
) -> int:
    """
    Tag knowledge items as relevant for a specific task.

    Args:
        task_type: Type of task (e.g., "debugging", "code_review")
        task_context: Brief description of the task context
        knowledge_item_ids: List of knowledge item IDs that were useful
        relevance_score: How relevant was this knowledge (0.0-1.0)
        success: Whether the task was completed successfully

    Returns:
        Number of task memories created
    """
    if not knowledge_item_ids:
        return 0

    # Generate hash of task context for deduplication
    task_hash = hashlib.sha256(task_context.encode()).hexdigest()[:16]

    # Create context snippet (first 500 chars)
    context_snippet = task_context[:500] if len(task_context) > 500 else task_context

    created_count = 0

    with Session(memory_engine) as session:
        for item_id in knowledge_item_ids:
            # Check if this exact task-knowledge link already exists
            existing = session.execute(
                select(TaskMemory).where(
                    TaskMemory.task_hash == task_hash,
                    TaskMemory.knowledge_item_id == item_id,
                )
            ).scalar_one_or_none()

            if existing:
                # Update relevance score (weighted average)
                new_score = (existing.relevance_score + relevance_score) / 2
                session.execute(
                    update(TaskMemory)
                    .where(TaskMemory.id == existing.id)
                    .values(
                        relevance_score=new_score,
                        used_at=datetime.now(timezone.utc),
                        success=success,
                    )
                )
                memory_logger.debug(f"Updated task memory for item {item_id}, new score: {new_score:.2f}")
            else:
                # Create new task memory
                task_memory = TaskMemory(
                    task_type=task_type,
                    task_hash=task_hash,
                    knowledge_item_id=item_id,
                    relevance_score=relevance_score,
                    success=success,
                    context_snippet=context_snippet,
                )
                session.add(task_memory)
                created_count += 1
                memory_logger.debug(f"Created task memory for item {item_id}, task: {task_type}")

        session.commit()

    memory_logger.info(
        f"Tagged {len(knowledge_item_ids)} knowledge items for task type '{task_type}' "
        f"({created_count} new, {len(knowledge_item_ids) - created_count} updated)"
    )

    return created_count


def get_task_knowledge(
    task_type: str,
    limit: int = 10,
    min_relevance: float = 0.5,
    include_general: bool = True,
) -> List[Tuple[KnowledgeItem, float]]:
    """
    Get most relevant knowledge items for a task type.

    Args:
        task_type: Type of task to get knowledge for
        limit: Maximum number of items to return
        min_relevance: Minimum relevance score (0.0-1.0)
        include_general: Include general_question task type in results

    Returns:
        List of (KnowledgeItem, relevance_score) tuples, sorted by relevance
    """
    with Session(memory_engine) as session:
        # Build query to get knowledge items with their average relevance scores
        # Join TaskMemory with KnowledgeItem
        query = (
            select(KnowledgeItem, TaskMemory.relevance_score, TaskMemory.used_at)
            .join(TaskMemory, KnowledgeItem.id == TaskMemory.knowledge_item_id)
            .where(KnowledgeItem.is_active.is_(True))
            .where(TaskMemory.relevance_score >= min_relevance)
        )

        # Filter by task type
        if include_general and task_type != TaskType.GENERAL_QUESTION:
            query = query.where(TaskMemory.task_type.in_([task_type, TaskType.GENERAL_QUESTION]))
        else:
            query = query.where(TaskMemory.task_type == task_type)

        # Order by relevance score (descending) and recency
        query = query.order_by(TaskMemory.relevance_score.desc(), TaskMemory.used_at.desc())

        # Execute query
        results = session.execute(query).fetchmany(limit)

        # Convert to list of tuples
        knowledge_items = [(item, float(relevance)) for item, relevance, _ in results]

        memory_logger.info(
            f"Retrieved {len(knowledge_items)} knowledge items for task type '{task_type}' "
            f"(min_relevance={min_relevance})"
        )

        return knowledge_items


def get_task_statistics(task_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Get statistics about task memory usage.

    Args:
        task_type: Specific task type to get stats for (None = all tasks)

    Returns:
        Dictionary with statistics
    """
    with Session(memory_engine) as session:
        from sqlalchemy import func

        if task_type:
            # Stats for specific task type
            query = select(
                func.count(TaskMemory.id).label("total_memories"),
                func.avg(TaskMemory.relevance_score).label("avg_relevance"),
                func.sum(TaskMemory.success.cast(Integer)).label("success_count"),
                func.count(func.distinct(TaskMemory.knowledge_item_id)).label("unique_items"),
            ).where(TaskMemory.task_type == task_type)

            result = session.execute(query).one()

            return {
                "task_type": task_type,
                "total_memories": result.total_memories or 0,
                "avg_relevance": round(float(result.avg_relevance or 0), 3),
                "success_count": result.success_count or 0,
                "success_rate": round((result.success_count or 0) / max(result.total_memories or 1, 1), 3),
                "unique_knowledge_items": result.unique_items or 0,
            }
        else:
            # Overall stats
            overall_query = select(
                func.count(TaskMemory.id).label("total_memories"),
                func.avg(TaskMemory.relevance_score).label("avg_relevance"),
                func.count(func.distinct(TaskMemory.task_type)).label("unique_tasks"),
                func.count(func.distinct(TaskMemory.knowledge_item_id)).label("unique_items"),
            )

            overall = session.execute(overall_query).one()

            # Per-task breakdown
            task_query = (
                select(TaskMemory.task_type, func.count(TaskMemory.id).label("count"))
                .group_by(TaskMemory.task_type)
                .order_by(func.count(TaskMemory.id).desc())
            )

            task_breakdown = session.execute(task_query).fetchall()

            return {
                "total_memories": overall.total_memories or 0,
                "avg_relevance": round(float(overall.avg_relevance or 0), 3),
                "unique_tasks": overall.unique_tasks or 0,
                "unique_knowledge_items": overall.unique_items or 0,
                "task_breakdown": [{"task_type": task, "count": count} for task, count in task_breakdown],
            }


def search_by_tags(tags: List[str], task_type: Optional[str] = None, limit: int = 20) -> List[KnowledgeItem]:
    """
    Search knowledge items by tags, optionally filtered by task type.

    Args:
        tags: List of tags to search for (OR logic)
        task_type: Optional task type to filter by
        limit: Maximum number of results

    Returns:
        List of matching KnowledgeItem objects
    """
    if not tags:
        return []

    with Session(memory_engine) as session:
        # Build query
        query = select(KnowledgeItem).where(KnowledgeItem.is_active.is_(True))

        # Search in meta_data JSON field for tags
        # Note: This requires the meta_data to have a "tags" key
        from sqlalchemy import or_, func

        tag_conditions = []
        for tag in tags:
            # Search in meta_data JSON (SQLite JSON support)
            tag_conditions.append(KnowledgeItem.meta_data.like(f'%"{tag}"%'))

        if tag_conditions:
            query = query.where(or_(*tag_conditions))

        # If task_type specified, join with TaskMemory
        if task_type:
            query = (
                query.join(TaskMemory, KnowledgeItem.id == TaskMemory.knowledge_item_id)
                .where(TaskMemory.task_type == task_type)
                .order_by(TaskMemory.relevance_score.desc())
            )
        else:
            # Order by usage
            query = query.order_by(KnowledgeItem.usage_count.desc())

        # Execute query
        results = session.execute(query.distinct().limit(limit)).scalars().all()

        memory_logger.info(f"Tag search for {tags} (task_type={task_type}): {len(results)} results")

        return list(results)


# Export public API
__all__ = [
    "TaskMemory",
    "TaskType",
    "detect_task_type",
    "tag_knowledge_for_task",
    "get_task_knowledge",
    "get_task_statistics",
    "search_by_tags",
]
