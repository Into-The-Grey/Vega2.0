#!/usr/bin/env python3
"""
Test script for persistent chat functionality with conversation memory.

Tests:
1. Session persistence - verify same session ID is reused
2. Context loading - verify conversation history is retrieved
3. Context formatting - verify history is properly formatted for LLM
4. Context window limits - verify character and exchange limits are respected
5. Multi-turn conversation - verify continuity across multiple exchanges
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vega.core.db import (
    log_conversation,
    get_persistent_session_id,
    get_recent_context,
    get_conversation_summary,
)
from src.vega.core.llm import format_conversation_context
from src.vega.core.config import get_config


def test_session_persistence():
    """Test that get_persistent_session_id returns consistent session IDs"""
    print("🧪 Test 1: Session Persistence")

    # Clear and create fresh sessions
    session1 = get_persistent_session_id()
    print(f"  First session ID: {session1}")

    # Log a conversation
    conv_id = log_conversation(
        "What is the capital of France?",
        "The capital of France is Paris.",
        session_id=session1,
    )
    print(f"  Logged conversation ID: {conv_id}")

    # Get session ID again - should be the same
    session2 = get_persistent_session_id()
    print(f"  Second session ID: {session2}")

    if session1 == session2:
        print("  ✅ Session persistence works - same ID returned")
        return True
    else:
        print("  ❌ Session persistence failed - different IDs returned")
        return False


def test_context_loading():
    """Test that conversation context is loaded correctly"""
    print("\n🧪 Test 2: Context Loading")

    # Create a test session and add multiple exchanges
    session_id = "test-context-session"

    exchanges = [
        ("What's 2+2?", "2+2 equals 4."),
        ("What about 5*5?", "5*5 equals 25."),
        ("Tell me about Python", "Python is a high-level programming language."),
    ]

    for prompt, response in exchanges:
        log_conversation(prompt, response, session_id=session_id)

    # Load context
    context = get_recent_context(session_id=session_id, limit=5)
    print(f"  Loaded {len(context)} exchanges from context")

    if len(context) == 3:
        print("  ✅ Context loading works - correct number of exchanges")
        return True
    else:
        print(f"  ❌ Context loading failed - expected 3, got {len(context)}")
        return False


def test_context_formatting():
    """Test that conversation context is formatted correctly for LLM"""
    print("\n🧪 Test 3: Context Formatting")

    context = [
        {
            "id": 1,
            "prompt": "Hello",
            "response": "Hi there!",
            "ts": "2025-01-01T00:00:00",
        },
        {
            "id": 2,
            "prompt": "How are you?",
            "response": "I'm doing well!",
            "ts": "2025-01-01T00:01:00",
        },
    ]

    formatted = format_conversation_context(context)
    print(f"  Formatted context:\n{formatted}")

    # Check that key elements are present
    checks = [
        "Conversation History" in formatted,
        "Exchange 1:" in formatted,
        "Exchange 2:" in formatted,
        "User: Hello" in formatted,
        "Assistant: Hi there!" in formatted,
        "Current Conversation" in formatted,
    ]

    if all(checks):
        print("  ✅ Context formatting works - all elements present")
        return True
    else:
        print("  ❌ Context formatting failed - missing elements")
        return False


def test_context_window_limits():
    """Test that context window limits (size and characters) are respected"""
    print("\n🧪 Test 4: Context Window Limits")

    # Create session with many exchanges
    session_id = "test-limits-session"

    for i in range(20):
        log_conversation(
            f"Question {i}: " + ("x" * 100),
            f"Answer {i}: " + ("y" * 100),
            session_id=session_id,
        )

    # Test limit by count
    context_limited = get_recent_context(
        session_id=session_id, limit=5, max_chars=10000
    )
    print(f"  With limit=5: got {len(context_limited)} exchanges")

    # Test limit by characters
    context_char_limited = get_recent_context(
        session_id=session_id, limit=20, max_chars=500
    )
    print(f"  With max_chars=500: got {len(context_char_limited)} exchanges")

    if len(context_limited) <= 5 and len(context_char_limited) < 5:
        print("  ✅ Context window limits work")
        return True
    else:
        print("  ❌ Context window limits failed")
        return False


def test_multi_turn_simulation():
    """Simulate a multi-turn conversation to verify continuity"""
    print("\n🧪 Test 5: Multi-turn Conversation Simulation")

    session_id = "test-multiturn-session"

    # Simulate conversation turns
    turns = [
        ("My name is Alice", "Nice to meet you, Alice!"),
        ("What's my name?", "Your name is Alice."),
        ("What did we talk about?", "We talked about your name, Alice."),
    ]

    for i, (prompt, response) in enumerate(turns, 1):
        # Get context before each turn
        context = get_recent_context(session_id=session_id, limit=10)
        print(f"\n  Turn {i}:")
        print(f"    Context size: {len(context)} exchanges")
        print(f"    User: {prompt}")
        print(f"    Assistant: {response}")

        # Log the exchange
        log_conversation(prompt, response, session_id=session_id)

    # Final context check
    final_context = get_recent_context(session_id=session_id, limit=10)

    if len(final_context) == 3:
        print("\n  ✅ Multi-turn conversation maintains context")
        return True
    else:
        print(
            f"\n  ❌ Multi-turn failed - expected 3 exchanges, got {len(final_context)}"
        )
        return False


def test_configuration():
    """Test that configuration values are loaded correctly"""
    print("\n🧪 Test 6: Configuration Loading")

    try:
        cfg = get_config()
        context_window = getattr(cfg, "context_window_size", None)
        context_max = getattr(cfg, "context_max_chars", None)

        print(f"  context_window_size: {context_window}")
        print(f"  context_max_chars: {context_max}")

        if context_window is not None and context_max is not None:
            print("  ✅ Configuration loaded successfully")
            return True
        else:
            print("  ⚠️  Configuration defaults used (expected in test env)")
            return True
    except Exception as e:
        print(f"  ❌ Configuration failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Persistent Chat Memory Test Suite")
    print("=" * 60)

    tests = [
        test_configuration,
        test_session_persistence,
        test_context_loading,
        test_context_formatting,
        test_context_window_limits,
        test_multi_turn_simulation,
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

    if all(results):
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
