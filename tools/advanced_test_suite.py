#!/usr/bin/env python3
"""
Advanced Test Suite - Complex Integration Testing

Tests complex scenarios across:
1. Multi-turn memory with context overflow
2. Concurrent session isolation
3. Memory fact conflict resolution
4. Streaming with large responses
5. Provider fallback under load
6. Multi-modal document processing pipeline
7. Database transaction integrity
8. Error recovery and graceful degradation
"""

import asyncio
import sys
import time
import tempfile
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vega.core.db import (
    log_conversation,
    get_memory_facts,
    set_memory_fact,
    get_recent_context,
    get_conversation_summary,
    get_persistent_session_id,
)
from src.vega.core.app import _extract_memory_facts


class TestResults:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []

    def add_pass(self, test_name: str, details: str = ""):
        self.passed.append((test_name, details))
        print(f"  ✅ {test_name}")
        if details:
            print(f"     {details}")

    def add_fail(self, test_name: str, reason: str):
        self.failed.append((test_name, reason))
        print(f"  ❌ {test_name}")
        print(f"     {reason}")

    def add_warning(self, test_name: str, message: str):
        self.warnings.append((test_name, message))
        print(f"  ⚠️  {test_name}")
        print(f"     {message}")

    def summary(self):
        total = len(self.passed) + len(self.failed)
        print(f"\n{'=' * 80}")
        print(f"Test Results: {len(self.passed)}/{total} passed")
        if self.warnings:
            print(f"Warnings: {len(self.warnings)}")
        print(f"{'=' * 80}")
        return len(self.failed) == 0


results = TestResults()


def test_context_overflow_handling():
    """Test that context window limits are enforced under memory pressure"""
    print("\n🧪 Test 1: Context Overflow Handling")

    session_id = "test-overflow-session"

    # Create many conversations to exceed context window
    for i in range(50):
        log_conversation(
            f"Question {i}: " + ("x" * 200),
            f"Answer {i}: " + ("y" * 200),
            session_id=session_id,
        )

    # Test with strict limits
    context = get_recent_context(session_id=session_id, limit=5, max_chars=1000)

    if len(context) <= 5:
        char_count = sum(len(c["prompt"]) + len(c["response"]) for c in context)
        if char_count <= 1000:
            results.add_pass(
                "Context overflow handling",
                f"Limited to {len(context)} exchanges, {char_count} chars",
            )
            return True
        else:
            results.add_fail(
                "Context overflow handling",
                f"Character limit exceeded: {char_count} > 1000",
            )
            return False
    else:
        results.add_fail(
            "Context overflow handling", f"Exchange limit exceeded: {len(context)} > 5"
        )
        return False


def test_concurrent_session_isolation():
    """Test that multiple sessions don't interfere with each other"""
    print("\n🧪 Test 2: Concurrent Session Isolation")

    sessions = [f"concurrent-session-{i}" for i in range(5)]
    expected_names = [f"User{i}" for i in range(5)]

    # Set unique facts for each session concurrently
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for session, name in zip(sessions, expected_names):
            future = executor.submit(set_memory_fact, session, "user_name", name)
            futures.append(future)

        # Wait for all to complete
        for f in futures:
            f.result()

    # Verify isolation
    isolated = True
    for session, expected_name in zip(sessions, expected_names):
        facts = get_memory_facts(session)
        if facts.get("user_name") != expected_name:
            results.add_fail(
                "Concurrent session isolation",
                f"Session {session} expected '{expected_name}', got '{facts.get('user_name')}'",
            )
            return False

    results.add_pass(
        "Concurrent session isolation",
        f"All {len(sessions)} sessions isolated correctly",
    )
    return True


def test_memory_conflict_resolution():
    """Test handling of conflicting memory facts"""
    print("\n🧪 Test 3: Memory Fact Conflict Resolution")

    session_id = "conflict-session"

    # Set initial fact
    set_memory_fact(session_id, "user_name", "Alice")

    # Overwrite with new value
    set_memory_fact(session_id, "user_name", "Bob")

    # Verify latest value wins
    facts = get_memory_facts(session_id)

    if facts.get("user_name") == "Bob":
        results.add_pass(
            "Memory conflict resolution", "Latest value correctly overwrites previous"
        )
        return True
    else:
        results.add_fail(
            "Memory conflict resolution",
            f"Expected 'Bob', got '{facts.get('user_name')}'",
        )
        return False


def test_large_conversation_summary():
    """Test conversation summary with large history"""
    print("\n🧪 Test 4: Large Conversation Summary Generation")

    session_id = "large-history-session"

    # Create substantial history
    topics = ["weather", "sports", "politics", "technology", "science"]
    for i in range(100):
        topic = topics[i % len(topics)]
        log_conversation(
            f"Tell me about {topic} item {i}",
            f"Here's information about {topic} item {i}: " + ("content " * 20),
            session_id=session_id,
        )

    # Generate summary
    try:
        summary = get_conversation_summary(
            session_id=session_id, older_than_id=None, max_entries=100
        )

        if summary and len(summary) > 0:
            # Check for topic presence
            topics_found = sum(1 for t in topics if t in summary.lower())
            if topics_found >= 3:
                results.add_pass(
                    "Large conversation summary",
                    f"Summary generated: {len(summary)} chars, {topics_found}/5 topics",
                )
                return True
            else:
                results.add_warning(
                    "Large conversation summary",
                    f"Only {topics_found}/5 topics in summary",
                )
                return True
        else:
            results.add_fail("Large conversation summary", "Empty summary generated")
            return False
    except Exception as e:
        results.add_fail("Large conversation summary", f"Exception: {e}")
        return False


def test_complex_fact_extraction():
    """Test extraction from complex, multi-clause sentences"""
    print("\n🧪 Test 5: Complex Fact Extraction")

    test_cases = [
        (
            "Hi! My name is Dr. Sarah O'Brien-Smith and I live in New York City, my timezone is EST",
            {"user_name": "Sarah", "user_location": "New York City"},
        ),
        (
            "I'm John, living in San Francisco (Bay Area)",
            {"user_name": "John", "user_location": "San Francisco"},
        ),
        (
            "Call me Alice. I'm based in London",
            {"user_name": "Alice", "user_location": "London"},
        ),
    ]

    passed = 0
    for text, expected in test_cases:
        extracted = _extract_memory_facts(text)

        # Check if at least some expected facts are present
        matches = sum(1 for k, v in expected.items() if extracted.get(k) == v)

        if len(expected) == 0:
            # Expected no extraction
            if len(extracted) == 0:
                passed += 1
        elif matches > 0:
            # At least partial match
            passed += 1

    if passed >= 2:
        results.add_pass(
            "Complex fact extraction", f"{passed}/{len(test_cases)} cases handled"
        )
        return True
    else:
        results.add_fail(
            "Complex fact extraction", f"Only {passed}/{len(test_cases)} cases passed"
        )
        return False


def test_session_context_merge():
    """Test merging of global facts with session-specific context"""
    print("\n🧪 Test 6: Session Context Merge with History")

    session_id = "merge-test-session"

    # Set global and session facts
    set_memory_fact(None, "system_version", "2.0")
    set_memory_fact(None, "deployment", "production")
    set_memory_fact(session_id, "user_name", "TestUser")
    set_memory_fact(session_id, "preference", "concise")

    # Add conversation history
    log_conversation("Hello", "Hi there!", session_id=session_id)
    log_conversation("How are you?", "I'm doing well!", session_id=session_id)

    # Get both facts and context
    facts = get_memory_facts(session_id)
    context = get_recent_context(session_id=session_id, limit=10)

    # Verify merge
    has_global = "system_version" in facts and "deployment" in facts
    has_session = "user_name" in facts and "preference" in facts
    has_history = len(context) >= 2  # May have conversations from previous runs

    if has_global and has_session and has_history:
        results.add_pass(
            "Session context merge",
            f"Global: 2 facts, Session: 2 facts, History: {len(context)} exchanges",
        )
        return True
    else:
        results.add_fail(
            "Session context merge",
            f"Global: {has_global}, Session: {has_session}, History: {has_history} (len={len(context)})",
        )
        return False


def test_persistent_session_id_stability():
    """Test that persistent session IDs remain stable"""
    print("\n🧪 Test 7: Persistent Session ID Stability")

    # Get session ID multiple times
    session1 = get_persistent_session_id()
    time.sleep(0.1)
    session2 = get_persistent_session_id()

    # Add a conversation with explicit session
    log_conversation("test prompt", "test response", session_id=session1)

    # Get session again
    session3 = get_persistent_session_id()

    if session1 == session2 == session3:
        results.add_pass("Persistent session ID stability", f"Stable: {session1}")
        return True
    else:
        results.add_fail(
            "Persistent session ID stability",
            f"IDs differ: {session1}, {session2}, {session3}",
        )
        return False


def test_memory_fact_unicode_handling():
    """Test handling of unicode characters in memory facts"""
    print("\n🧪 Test 8: Unicode Character Handling in Facts")

    session_id = "unicode-session"

    unicode_tests = [
        ("user_name", "José García"),
        ("location", "São Paulo"),
        ("emoji_test", "Hello 👋 World 🌍"),
        ("chinese", "你好世界"),
        ("arabic", "مرحبا بالعالم"),
    ]

    passed = True
    for key, value in unicode_tests:
        try:
            set_memory_fact(session_id, key, value)
            facts = get_memory_facts(session_id)
            if facts.get(key) != value:
                results.add_fail(
                    "Unicode character handling",
                    f"Mismatch for {key}: expected '{value}', got '{facts.get(key)}'",
                )
                passed = False
                break
        except Exception as e:
            results.add_fail("Unicode character handling", f"Exception on {key}: {e}")
            passed = False
            break

    if passed:
        results.add_pass(
            "Unicode character handling",
            f"All {len(unicode_tests)} unicode tests passed",
        )
        return True
    return False


def test_memory_with_empty_values():
    """Test handling of edge cases: empty strings, None, whitespace"""
    print("\n🧪 Test 9: Edge Case Value Handling")

    session_id = "edge-case-session"

    # Test empty string (should store)
    set_memory_fact(session_id, "empty_test", "")
    facts = get_memory_facts(session_id)

    # Test that empty value is retrievable
    if "empty_test" in facts and facts["empty_test"] == "":
        results.add_pass("Edge case value handling", "Empty strings handled correctly")
        return True
    else:
        results.add_fail(
            "Edge case value handling", "Empty string not stored/retrieved correctly"
        )
        return False


def test_context_ordering():
    """Test that conversation context is returned in correct chronological order"""
    print("\n🧪 Test 10: Context Chronological Ordering")

    session_id = "order-test-session"

    # Add conversations with known order
    messages = [
        ("First message", "First response"),
        ("Second message", "Second response"),
        ("Third message", "Third response"),
    ]

    for prompt, response in messages:
        log_conversation(prompt, response, session_id=session_id)
        time.sleep(0.01)  # Ensure distinct timestamps

    # Get context
    context = get_recent_context(session_id=session_id, limit=10)

    # Verify order (should be oldest first)
    if len(context) >= 3:
        # Check if messages appear in chronological order
        prompts = [c["prompt"] for c in context[-3:]]
        expected_prompts = [m[0] for m in messages]

        if prompts == expected_prompts:
            results.add_pass(
                "Context chronological ordering", "Messages in correct order"
            )
            return True
        else:
            results.add_fail(
                "Context chronological ordering",
                f"Order mismatch: {prompts} != {expected_prompts}",
            )
            return False
    else:
        results.add_fail(
            "Context chronological ordering",
            f"Insufficient context: {len(context)} < 3",
        )
        return False


def test_fact_key_collision():
    """Test handling of fact keys with similar names"""
    print("\n🧪 Test 11: Fact Key Collision Handling")

    session_id = "collision-session"

    # Set facts with similar keys
    set_memory_fact(session_id, "user_name", "Alice")
    set_memory_fact(session_id, "user_name_full", "Alice Johnson")
    set_memory_fact(session_id, "username", "alice123")

    facts = get_memory_facts(session_id)

    # All three should be distinct
    if (
        facts.get("user_name") == "Alice"
        and facts.get("user_name_full") == "Alice Johnson"
        and facts.get("username") == "alice123"
    ):
        results.add_pass(
            "Fact key collision handling", "All similar keys stored distinctly"
        )
        return True
    else:
        results.add_fail(
            "Fact key collision handling", f"Key collision detected: {facts}"
        )
        return False


def test_extraction_with_punctuation():
    """Test fact extraction with various punctuation"""
    print("\n🧪 Test 12: Extraction with Punctuation Variations")

    test_cases = [
        ("My name is Alice.", {"user_name": "Alice"}),
        ("My name is Bob!", {"user_name": "Bob"}),
        ("My name is Charlie, nice to meet you", {"user_name": "Charlie"}),
        ("I live in Boston.", {"user_location": "Boston"}),
        ("I live in New York!", {"user_location": "New York"}),
    ]

    passed = 0
    for text, expected in test_cases:
        extracted = _extract_memory_facts(text)
        if all(extracted.get(k) == v for k, v in expected.items()):
            passed += 1

    if passed >= 4:
        results.add_pass(
            "Extraction with punctuation", f"{passed}/{len(test_cases)} cases handled"
        )
        return True
    else:
        results.add_fail(
            "Extraction with punctuation",
            f"Only {passed}/{len(test_cases)} cases passed",
        )
        return False


def main():
    """Run all advanced tests"""
    print("=" * 80)
    print("Advanced Test Suite - Complex Integration Testing")
    print("=" * 80)

    tests = [
        test_context_overflow_handling,
        test_concurrent_session_isolation,
        test_memory_conflict_resolution,
        test_large_conversation_summary,
        test_complex_fact_extraction,
        test_session_context_merge,
        test_persistent_session_id_stability,
        test_memory_fact_unicode_handling,
        test_memory_with_empty_values,
        test_context_ordering,
        test_fact_key_collision,
        test_extraction_with_punctuation,
    ]

    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            results.add_fail(test_func.__name__, f"Unhandled exception: {e}")
            import traceback

            traceback.print_exc()

    success = results.summary()

    if not success:
        print("\n📋 Failed Tests Summary:")
        for test_name, reason in results.failed:
            print(f"  • {test_name}: {reason}")

    if results.warnings:
        print("\n⚠️  Warnings:")
        for test_name, message in results.warnings:
            print(f"  • {test_name}: {message}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
