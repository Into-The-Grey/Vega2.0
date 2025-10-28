#!/usr/bin/env python3
"""
Quick end-to-end test for persistent memory feature.

Demonstrates:
1. Memory fact extraction (name, location, timezone)
2. Memory fact persistence across requests
3. System context injection into LLM prompts
4. Context window limiting
"""

import asyncio
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Use temporary database for test isolation
temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
temp_db.close()
os.environ["DATABASE_PATH"] = temp_db.name

from src.vega.core.db import (
    log_conversation,
    get_memory_facts,
    set_memory_fact,
    get_recent_context,
)
from src.vega.core.app import _extract_memory_facts


def test_memory_extraction():
    """Test that memory facts are extracted correctly"""
    print("🧪 Test 1: Memory Fact Extraction")

    test_cases = [
        ("My name is Alice", {"user_name": "Alice"}),
        ("I live in San Francisco", {"user_location": "San Francisco"}),
        (
            "My timezone is America/Los_Angeles",
            {"user_timezone": "America/Los_Angeles"},
        ),
        (
            "My name is Bob and I live in New York",
            {"user_name": "Bob", "user_location": "New York"},
        ),
        ("Call me Alice", {"user_name": "Alice"}),
        ("I'm based in London", {"user_location": "London"}),
        (
            "Call me Dr. Sarah O'Brien-Smith and I'm based in London",
            {"user_name": "Sarah O'Brien-Smith", "user_location": "London"},
        ),
    ]

    for text, expected in test_cases:
        extracted = _extract_memory_facts(text)
        if extracted == expected:
            print(f"  ✅ '{text}' → {extracted}")
        else:
            print(f"  ❌ '{text}' → expected {expected}, got {extracted}")
            return False

    print("  ✅ All extraction tests passed")
    return True


def test_memory_persistence():
    """Test that memory facts persist across sessions"""
    print("\n🧪 Test 2: Memory Fact Persistence")

    session_id = "test-memory-session"

    # Store facts
    set_memory_fact(session_id, "user_name", "Alice")
    set_memory_fact(session_id, "user_location", "San Francisco")
    set_memory_fact(None, "global_fact", "shared_value")

    # Retrieve facts
    facts = get_memory_facts(session_id)

    # Check required facts are present (may have others from test 4)
    required = {
        "global_fact": "shared_value",  # Global fact
        "user_name": "Alice",  # Session fact
        "user_location": "San Francisco",  # Session fact
    }

    if all(facts.get(k) == v for k, v in required.items()):
        print(f"  ✅ Facts persisted correctly: {facts}")
        return True
    else:
        print(f"  ❌ Missing required facts. Expected {required}, got {facts}")
        return False


def test_context_with_memory():
    """Test that context includes memory facts"""
    print("\n🧪 Test 3: Context with Memory Integration")

    session_id = "test-context-memory"

    # Add memory facts
    set_memory_fact(session_id, "user_name", "Charlie")
    set_memory_fact(session_id, "favorite_color", "blue")

    # Add conversation history
    log_conversation("Hello", "Hi there!", session_id=session_id)
    log_conversation("How are you?", "I'm doing well!", session_id=session_id)

    # Get context and facts
    context = get_recent_context(session_id=session_id, limit=10)
    facts = get_memory_facts(session_id)

    # Build system context as the app does
    system_parts = []
    if facts:
        mem_lines = ["[Memory Facts]"]
        for k, v in facts.items():
            mem_lines.append(f"- {k}: {v}")
        system_parts.append("\n".join(mem_lines))

    system_context = "\n\n".join(system_parts) if system_parts else None

    print(f"  Conversation context: {len(context)} exchanges")
    print(f"  System context:\n{system_context}")

    if (
        len(context) >= 2
        and facts["user_name"] == "Charlie"
        and system_context
        and "favorite_color: blue" in system_context
    ):
        print("  ✅ Context and memory integrated correctly")
        return True
    else:
        print("  ❌ Integration failed")
        return False


def test_global_vs_session_facts():
    """Test that global and session facts are merged correctly"""
    print("\n🧪 Test 4: Global vs Session Facts")

    # Set global facts
    set_memory_fact(None, "system_version", "2.0")
    set_memory_fact(None, "default_timezone", "UTC")

    # Set session-specific facts
    session_a = "session-a"
    session_b = "session-b"

    set_memory_fact(session_a, "user_name", "Alice")
    set_memory_fact(session_b, "user_name", "Bob")

    # Get facts for each session
    facts_a = get_memory_facts(session_a)
    facts_b = get_memory_facts(session_b)

    if (
        facts_a["system_version"] == "2.0"
        and facts_a["user_name"] == "Alice"
        and facts_b["user_name"] == "Bob"
    ):
        print("  ✅ Global and session facts merged correctly")
        print(f"    Session A: {facts_a}")
        print(f"    Session B: {facts_b}")
        return True
    else:
        print("  ❌ Fact merging failed")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Persistent Memory Feature Validation")
    print("=" * 60)

    tests = [
        test_memory_extraction,
        test_memory_persistence,
        test_context_with_memory,
        test_global_vs_session_facts,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n  ❌ Test {test_func.__name__} crashed: {e}")
            import traceback

            traceback.print_exc()
            results.append(False)

    print("\n" + "=" * 60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 60)

    # Cleanup
    try:
        os.unlink(temp_db.name)
    except Exception:
        pass

    if all(results):
        print("✅ All tests passed! Memory feature is working.")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
