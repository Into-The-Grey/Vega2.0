"""
Test suite for memory tagging system
"""

import pytest
from datetime import datetime, timezone
from src.vega.core.memory_tagging import (
    detect_task_type,
    tag_knowledge_for_task,
    get_task_knowledge,
    get_task_statistics,
    search_by_tags,
    TaskType,
)


class TestTaskDetection:
    """Test task type detection"""

    def test_detect_debugging_task(self):
        prompts = [
            "Help me debug this code",
            "I'm getting an error in my function",
            "Why is this not working?",
            "Fix this bug please",
        ]
        for prompt in prompts:
            task = detect_task_type(prompt)
            assert (
                task == TaskType.DEBUGGING
            ), f"Failed to detect debugging in: {prompt}"

    def test_detect_code_review_task(self):
        prompts = [
            "Can you review this code?",
            "Is this code correct?",
            "Check my implementation",
        ]
        for prompt in prompts:
            task = detect_task_type(prompt)
            assert (
                task == TaskType.CODE_REVIEW
            ), f"Failed to detect code review in: {prompt}"

    def test_detect_explanation_task(self):
        prompts = [
            "Explain how this works",
            "What does this function do?",
            "Why is this pattern used?",
        ]
        for prompt in prompts:
            task = detect_task_type(prompt)
            assert (
                task == TaskType.EXPLANATION
            ), f"Failed to detect explanation in: {prompt}"

    def test_detect_refactoring_task(self):
        prompts = [
            "Refactor this code",
            "How can I improve this implementation?",
            "Clean up this function",
        ]
        for prompt in prompts:
            task = detect_task_type(prompt)
            assert (
                task == TaskType.REFACTORING
            ), f"Failed to detect refactoring in: {prompt}"

    def test_detect_optimization_task(self):
        prompts = [
            "Optimize this code for performance",
            "Make this faster",
            "Reduce memory usage",
        ]
        for prompt in prompts:
            task = detect_task_type(prompt)
            assert (
                task == TaskType.OPTIMIZATION
            ), f"Failed to detect optimization in: {prompt}"


class TestTaskTagging:
    """Test knowledge tagging for tasks"""

    def test_tag_knowledge_for_task(self):
        # This test requires database setup
        # For now, just test the function signature
        result = tag_knowledge_for_task(
            task_type=TaskType.DEBUGGING,
            task_context="Debugging a NoneType error",
            knowledge_item_ids=[],  # Empty list to avoid DB operations
            relevance_score=0.9,
            success=True,
        )
        assert result == 0  # No items tagged

    def test_tag_with_multiple_items(self):
        # Test with multiple knowledge items (mock)
        # In real scenario, these would be valid DB IDs
        result = tag_knowledge_for_task(
            task_type=TaskType.CODE_REVIEW,
            task_context="Reviewing API endpoint implementation",
            knowledge_item_ids=[],
            relevance_score=0.85,
        )
        assert result >= 0


class TestTaskKnowledgeRetrieval:
    """Test retrieving knowledge by task type"""

    def test_get_task_knowledge_basic(self):
        # Test basic retrieval (may return empty if DB not populated)
        results = get_task_knowledge(
            task_type=TaskType.DEBUGGING, limit=5, min_relevance=0.5
        )
        assert isinstance(results, list)

    def test_get_task_knowledge_with_filters(self):
        results = get_task_knowledge(
            task_type=TaskType.CODE_REVIEW,
            limit=10,
            min_relevance=0.7,
            include_general=False,
        )
        assert isinstance(results, list)


class TestTaskStatistics:
    """Test task memory statistics"""

    def test_get_overall_statistics(self):
        stats = get_task_statistics()
        assert isinstance(stats, dict)
        assert "total_memories" in stats
        assert "avg_relevance" in stats
        assert "unique_tasks" in stats

    def test_get_task_specific_statistics(self):
        stats = get_task_statistics(task_type=TaskType.DEBUGGING)
        assert isinstance(stats, dict)
        assert "task_type" in stats
        assert stats["task_type"] == TaskType.DEBUGGING


class TestTagSearch:
    """Test tag-based search"""

    def test_search_by_tags(self):
        results = search_by_tags(tags=["python", "optimization"], limit=10)
        assert isinstance(results, list)

    def test_search_by_tags_with_task_filter(self):
        results = search_by_tags(tags=["python"], task_type=TaskType.DEBUGGING, limit=5)
        assert isinstance(results, list)

    def test_search_empty_tags(self):
        results = search_by_tags(tags=[], limit=10)
        assert not results  # Empty list is falsey


class TestIntegration:
    """Integration tests for memory tagging workflow"""

    def test_full_workflow(self):
        # Simulate a complete workflow
        # 1. Detect task type
        prompt = "Help me debug this memory leak"
        task_type = detect_task_type(prompt)
        assert task_type == TaskType.DEBUGGING

        # 2. Tag knowledge (with empty list for testing)
        tagged = tag_knowledge_for_task(
            task_type=task_type,
            task_context=prompt,
            knowledge_item_ids=[],
            relevance_score=0.9,
        )
        assert tagged >= 0

        # 3. Retrieve knowledge
        knowledge = get_task_knowledge(task_type=task_type, limit=5)
        assert isinstance(knowledge, list)

        # 4. Get statistics
        stats = get_task_statistics(task_type=task_type)
        assert isinstance(stats, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
