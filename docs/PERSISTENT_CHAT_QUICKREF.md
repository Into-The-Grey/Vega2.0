# Persistent Chat Memory - Quick Reference

## 🚀 Quick Start

### 1. Configure (Optional)

```bash
# Add to .env (or use defaults)
CONTEXT_WINDOW_SIZE=10      # How many exchanges to remember
CONTEXT_MAX_CHARS=4000      # Max characters from history
```

### 2. Use the API

```bash
# First message
curl -X POST http://localhost:8000/chat \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "My name is Alice"}'

# Later message - automatically remembers!
curl -X POST http://localhost:8000/chat \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is my name?"}'
# Response: "Your name is Alice."
```

### 3. Use the CLI

```bash
# Chat with automatic memory
vega cli chat "I prefer Python"
vega cli chat "What do I prefer?"  # Remembers!
```

## 📋 Key Functions

### Database Functions

```python
from src.vega.core.db import (
    get_persistent_session_id,
    get_recent_context
)

# Get persistent session
session_id = get_persistent_session_id()

# Load conversation context
context = get_recent_context(
    session_id=session_id,
    limit=10,           # Max exchanges
    max_chars=4000      # Max characters
)
```

### LLM Functions

```python
from src.vega.core.llm import query_llm, format_conversation_context

# Query with context
response = await query_llm(
    prompt="Your question",
    conversation_context=context  # Automatically formatted
)

# Format context manually
formatted = format_conversation_context(context)
```

## ⚙️ Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTEXT_WINDOW_SIZE` | 10 | Number of recent exchanges to load |
| `CONTEXT_MAX_CHARS` | 4000 | Maximum characters from history |

### Tuning Profiles

**Fast & Focused (minimal context)**

```bash
CONTEXT_WINDOW_SIZE=5
CONTEXT_MAX_CHARS=2000
```

**Balanced (recommended)**

```bash
CONTEXT_WINDOW_SIZE=10
CONTEXT_MAX_CHARS=4000
```

**Comprehensive (maximum memory)**

```bash
CONTEXT_WINDOW_SIZE=20
CONTEXT_MAX_CHARS=8000
```

## 🧪 Testing

### Run Test Suite

```bash
python tests/test_persistent_chat.py
```

### Run Interactive Demo

```bash
python demos/demo_persistent_chat.py
```

## 📊 API Response Format

```json
{
  "response": "Your name is Alice.",
  "session_id": "persistent-abc-123",
  "context_used": 5
}
```

**Fields:**

- `response`: LLM's answer
- `session_id`: Session identifier (reuse for continuity)
- `context_used`: Number of previous exchanges used

## 🔧 Common Patterns

### Single Persistent Conversation

```python
# Just use the API - it handles everything automatically!
POST /chat {"prompt": "Hello"}
POST /chat {"prompt": "Remember me?"}  # Automatically continues
```

### Multiple Users/Sessions

```python
# Pass custom session_id per user
POST /chat {
  "prompt": "Hello",
  "session_id": "user-alice"
}

POST /chat {
  "prompt": "Hello", 
  "session_id": "user-bob"
}
```

### Check Context Size

```python
from src.vega.core.db import get_recent_context

context = get_recent_context(session_id="my-session")
print(f"Context has {len(context)} exchanges")
```

## ⚡ Performance

| Metric | Value |
|--------|-------|
| Context retrieval | ~1-5ms |
| Context formatting | ~1-2ms |
| Total overhead | ~2-7ms |
| Memory per session | ~10KB |

## 🆘 Troubleshooting

### Problem: Not remembering context

**Check:**

```python
from src.vega.core.db import get_recent_context
context = get_recent_context(session_id="your-session")
print(f"Found {len(context)} exchanges")
```

### Problem: Too much/little context

**Adjust config:**

```bash
# Increase
CONTEXT_WINDOW_SIZE=20
CONTEXT_MAX_CHARS=8000

# Decrease
CONTEXT_WINDOW_SIZE=5
CONTEXT_MAX_CHARS=2000
```

### Problem: Context from wrong session

**Verify session ID:**

```python
from src.vega.core.db import get_persistent_session_id
session = get_persistent_session_id()
print(f"Using session: {session}")
```

## 📖 Documentation

- **Full Guide**: `docs/PERSISTENT_CHAT_MEMORY.md`
- **Implementation**: `docs/PERSISTENT_CHAT_IMPLEMENTATION.md`
- **Tests**: `tests/test_persistent_chat.py`
- **Demo**: `demos/demo_persistent_chat.py`

## ✅ Features

- ✅ Automatic conversation continuity
- ✅ Configurable memory window
- ✅ Efficient database queries
- ✅ Backward compatible API
- ✅ Multi-session support
- ✅ Character limits prevent overflow
- ✅ Chronological ordering
- ✅ Zero breaking changes

## 🎯 Example: Complete Workflow

```python
# 1. User sends first message
POST /chat {"prompt": "My favorite color is blue"}
→ Session: persistent-abc-123
→ Context: 0 exchanges
→ Response: "Noted! Your favorite color is blue."

# 2. User sends second message
POST /chat {"prompt": "What's my favorite color?"}
→ Session: persistent-abc-123 (same!)
→ Context: 1 exchange loaded
→ Response: "Your favorite color is blue."

# 3. User sends third message
POST /chat {"prompt": "Do I prefer Python or Java?"}
→ Session: persistent-abc-123
→ Context: 2 exchanges loaded
→ Response: "I don't have information about your programming language preference yet."

# 4. User clarifies
POST /chat {"prompt": "I prefer Python"}
→ Session: persistent-abc-123
→ Context: 3 exchanges loaded
→ Response: "Got it! You prefer Python."

# 5. Later...
POST /chat {"prompt": "What do I prefer?"}
→ Session: persistent-abc-123
→ Context: 4 exchanges loaded
→ Response: "You prefer Python, and your favorite color is blue."
```

---

**Ready to use! No setup required beyond optional config tuning. 🚀**
