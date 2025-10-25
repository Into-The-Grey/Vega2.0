#!/usr/bin/env python3
"""
Demo: Persistent Chat Memory

This script demonstrates how Vega maintains conversation continuity
across multiple interactions without needing to start "new chats".
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
)
from src.vega.core.llm import format_conversation_context


def print_separator(title=""):
    """Print a visual separator"""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print("=" * 60)
    else:
        print("-" * 60)


async def demo_basic_memory():
    """Demonstrate basic conversation memory"""
    print_separator("Demo 1: Basic Conversation Memory")

    # Get persistent session
    session_id = get_persistent_session_id()
    print(f"📝 Using session: {session_id}\n")

    # Simulate a conversation about user preferences
    conversations = [
        (
            "My name is Alex and I love Python programming",
            "Nice to meet you, Alex! Python is a great language for many applications.",
        ),
        ("What's my name?", "Your name is Alex."),
        ("What language do I like?", "You mentioned that you love Python programming."),
        (
            "Can you write me a simple Python function?",
            "Sure, Alex! Here's a simple function:\n\ndef greet(name):\n    return f'Hello, {name}!'\n\nprint(greet('Alex'))",
        ),
    ]

    for i, (prompt, response) in enumerate(conversations, 1):
        print(f"🗣️  Turn {i}")
        print(f"   User: {prompt}")

        # Load context before this turn
        context = get_recent_context(session_id=session_id, limit=10)
        print(f"   💭 Context: {len(context)} previous exchanges loaded")

        # Log the conversation
        log_conversation(prompt, response, session_id=session_id)

        print(f"   🤖 Vega: {response}")
        print()


async def demo_context_formatting():
    """Show how context is formatted for the LLM"""
    print_separator("Demo 2: Context Formatting for LLM")

    # Create sample context
    context = [
        {
            "id": 1,
            "prompt": "Tell me about machine learning",
            "response": "Machine learning is a subset of AI that enables systems to learn from data.",
            "ts": "2025-01-01T10:00:00",
        },
        {
            "id": 2,
            "prompt": "What are neural networks?",
            "response": "Neural networks are computing systems inspired by biological neural networks.",
            "ts": "2025-01-01T10:05:00",
        },
    ]

    print("📋 Raw context data:")
    for entry in context:
        print(f"   Exchange {entry['id']}: {entry['prompt'][:40]}...")

    print("\n🔄 Formatted for LLM prompt:")
    print_separator()
    formatted = format_conversation_context(context)
    print(formatted)
    print_separator()

    print("\n✅ This formatted context is automatically prepended to new prompts!")


async def demo_context_limits():
    """Demonstrate context window and character limits"""
    print_separator("Demo 3: Context Window Management")

    session_id = "demo-limits-session"

    # Create a longer conversation history
    print("📝 Creating 15 conversation exchanges...")
    for i in range(1, 16):
        prompt = f"Question {i}: Tell me about topic {i}"
        response = f"Answer {i}: Here's information about topic {i}. " + ("x" * 50)
        log_conversation(prompt, response, session_id=session_id)

    print("✅ Created 15 exchanges\n")

    # Load with different limits
    print("🔍 Testing different context window sizes:\n")

    for limit in [5, 10, 15]:
        context = get_recent_context(
            session_id=session_id, limit=limit, max_chars=10000
        )
        print(f"   limit={limit:2d}: Retrieved {len(context)} exchanges")

    print("\n🔍 Testing character limits:\n")

    for max_chars in [500, 1000, 2000]:
        context = get_recent_context(
            session_id=session_id, limit=20, max_chars=max_chars
        )
        total_chars = sum(len(c["prompt"]) + len(c["response"]) for c in context)
        print(
            f"   max_chars={max_chars:4d}: Retrieved {len(context)} exchanges, {total_chars} chars"
        )

    print("\n✅ Context limits prevent memory overflow while keeping relevant history!")


async def demo_multi_session():
    """Demonstrate separate context for different sessions"""
    print_separator("Demo 4: Multiple Session Management")

    # Create contexts for different users/sessions
    sessions = {
        "user-alice": [
            ("I prefer tea", "Noted! You prefer tea."),
            ("What do I prefer?", "You prefer tea."),
        ],
        "user-bob": [
            ("I prefer coffee", "Noted! You prefer coffee."),
            ("What do I prefer?", "You prefer coffee."),
        ],
    }

    print("👥 Creating separate conversations for Alice and Bob:\n")

    for session_id, conversations in sessions.items():
        user = session_id.split("-")[1].title()
        print(f"🗣️  {user}'s conversation (session: {session_id}):")

        for prompt, response in conversations:
            log_conversation(prompt, response, session_id=session_id)
            print(f"   User: {prompt}")
            print(f"   Vega: {response}")
        print()

    print("🔍 Loading context for each user:\n")

    for session_id in sessions.keys():
        context = get_recent_context(session_id=session_id)
        user = session_id.split("-")[1].title()
        print(f"   {user}: {len(context)} exchanges in context")
        for c in context:
            print(f"      - {c['prompt']}")

    print("\n✅ Each session maintains independent conversation history!")


async def demo_real_world_scenario():
    """Simulate a real-world multi-turn conversation"""
    print_separator("Demo 5: Real-World Scenario - Project Planning")

    session_id = "project-planning-session"

    conversation = [
        (
            "I'm starting a new web project using Python",
            "Great! For web development with Python, I'd recommend FastAPI or Django. Which type of application are you building?",
        ),
        (
            "It's going to be an API service for mobile apps",
            "Perfect! FastAPI is excellent for API services. It's fast, modern, and has great documentation. Would you like help setting it up?",
        ),
        (
            "Yes, what dependencies will I need?",
            "For a FastAPI project, you'll need:\n- fastapi\n- uvicorn (ASGI server)\n- pydantic (data validation)\n- sqlalchemy (database)\nWould you like a requirements.txt file?",
        ),
        (
            "What database should I use?",
            "For your FastAPI mobile API project, I'd recommend:\n1. PostgreSQL for production (robust, scalable)\n2. SQLite for development (simple, fast)\nSince you're just starting, SQLite works great!",
        ),
        (
            "Can you remind me what we've covered?",
            "We've discussed your new Python web project - specifically a FastAPI-based API service for mobile apps. We covered the key dependencies (FastAPI, uvicorn, pydantic, SQLAlchemy) and decided on using SQLite for development with PostgreSQL as a production option.",
        ),
    ]

    print(f"📱 Scenario: Planning a mobile API backend\n")

    for i, (prompt, response) in enumerate(conversation, 1):
        # Load context
        context = get_recent_context(session_id=session_id, limit=10)

        print(f"Turn {i}:")
        print(f"  💭 Context: {len(context)} previous exchanges")
        print(f"  👤 User: {prompt}")

        # Simulate processing with context
        if context:
            print(f"  📚 Using memory of previous topics discussed")

        print(f"  🤖 Vega: {response[:100]}{'...' if len(response) > 100 else ''}")

        # Log the exchange
        log_conversation(prompt, response, session_id=session_id)
        print()

    print("✅ Vega maintained context throughout the entire planning session!")
    print("   No need to repeat information - it remembered everything!")


async def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("  🧠 Vega Persistent Chat Memory - Interactive Demo")
    print("=" * 60)

    demos = [
        ("Basic Memory", demo_basic_memory),
        ("Context Formatting", demo_context_formatting),
        ("Context Limits", demo_context_limits),
        ("Multiple Sessions", demo_multi_session),
        ("Real-World Scenario", demo_real_world_scenario),
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            await demo_func()
            if i < len(demos):
                input("\n⏸️  Press Enter to continue to next demo...")
        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted by user")
            return
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("  ✅ Demo Complete!")
    print("=" * 60)
    print("\n📖 For more information, see: docs/PERSISTENT_CHAT_MEMORY.md")
    print("🧪 Run tests with: python tests/test_persistent_chat.py\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
