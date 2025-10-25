# Persistent Chat Memory System

## Overview

Vega2.0 now features a **persistent chat memory system** that maintains conversation continuity across all interactions. Unlike traditional chatbots that start fresh with each session, Vega always picks up where the conversation left off, providing a seamless and natural conversational experience.

## Key Features

### 🔄 Automatic Session Continuity

- **No "new chat" button needed** - Vega automatically continues from your last conversation
- **Single persistent session** - All conversations flow naturally as one continuous dialogue
- **Context-aware responses** - Vega remembers previous exchanges and references them

### ⚡ Efficient Memory Management

- **Sliding window approach** - Only recent exchanges are loaded into LLM context
- **Character limits** - Prevents context overflow while maximizing relevant history
- **Configurable window size** - Adjust how much history to include based on your needs

### 🎯 Smart Context Loading

- **Chronological ordering** - Conversations presented in natural time order
- **Summary generation** - Optional compressed summaries for very long histories
- **Session-based retrieval** - Load context specific to conversation threads

## Architecture

### Database Layer (`db.py`)

Three new functions manage conversation context:

#### `get_persistent_session_id() -> str`

```python
# Returns the most recent session ID or creates a new persistent one
session_id = get_persistent_session_id()
```

#### `get_recent_context(session_id, limit=10, max_chars=4000) -> List[Dict]`

```python
# Efficiently loads recent conversation history
context = get_recent_context(
    session_id="my-session",
    limit=10,           # Max number of exchanges
    max_chars=4000      # Max total characters
)
```

#### `get_conversation_summary(session_id, older_than_id, max_entries=100) -> str`

```python
# Generates compact summary of older conversations
summary = get_conversation_summary(
    session_id="my-session",
    older_than_id=100
)
```

### LLM Layer (`llm.py`)

#### `format_conversation_context(context) -> str`

Formats conversation history into structured LLM prompt:

```
[Conversation History]

Exchange 1:
User: What's 2+2?
Assistant: 2+2 equals 4.

Exchange 2:
User: What about 5*5?
Assistant: 5*5 equals 25.

[Current Conversation]
User: Now do 10*10
Assistant:
```

#### `query_llm(..., conversation_context=[])`

Updated to accept conversation context parameter:

```python
response = await query_llm(
    prompt="What did we discuss earlier?",
    conversation_context=context,  # Automatically formatted
    stream=False
)
```

### API Layer (`app.py`)

Updated `/chat` endpoint automatically:

1. Retrieves or creates persistent session ID
2. Loads recent conversation context
3. Passes context to LLM
4. Logs the new exchange
5. Returns response with context metadata

```json
POST /chat
{
  "prompt": "Tell me more about that",
  "stream": false
}

Response:
{
  "response": "Based on our earlier discussion about...",
  "session_id": "persistent-abc-123",
  "context_used": 5
}
```

## Configuration

Add to your `.env` file:

```bash
# Context window configuration
CONTEXT_WINDOW_SIZE=10      # Number of recent exchanges to include
CONTEXT_MAX_CHARS=4000      # Maximum characters from history
```

### Recommended Settings

**For short-term memory (fast, focused):**

```bash
CONTEXT_WINDOW_SIZE=5
CONTEXT_MAX_CHARS=2000
```

**For medium-term memory (balanced):**

```bash
CONTEXT_WINDOW_SIZE=10
CONTEXT_MAX_CHARS=4000
```

**For long-term memory (comprehensive):**

```bash
CONTEXT_WINDOW_SIZE=20
CONTEXT_MAX_CHARS=8000
```

## Usage Examples

### Basic Chat with Memory

```python
from src.vega.core.app import app
import httpx

async with httpx.AsyncClient() as client:
    # First message
    r1 = await client.post("http://localhost:8000/chat", 
        json={"prompt": "My favorite color is blue"},
        headers={"X-API-Key": "your-key"})
    
    # Second message - Vega remembers!
    r2 = await client.post("http://localhost:8000/chat",
        json={"prompt": "What's my favorite color?"},
        headers={"X-API-Key": "your-key"})
    
    print(r2.json()["response"])
    # Output: "Your favorite color is blue."
```

### CLI with Memory

```bash
# First interaction
$ python -m src.vega.core.cli chat "I'm working on project X"
# Vega: "I see you're working on project X..."

# Later interaction - automatic continuity
$ python -m src.vega.core.cli chat "Any updates on that?"
# Vega: "Regarding project X that you mentioned..."
```

### Programmatic Access

```python
from src.vega.core.db import get_recent_context, get_persistent_session_id
from src.vega.core.llm import query_llm, format_conversation_context

# Get persistent session
session_id = get_persistent_session_id()

# Load context
context = get_recent_context(session_id=session_id, limit=10)

# Query with context
response = await query_llm(
    "Continue our discussion",
    conversation_context=context
)
```

## Performance Characteristics

### Memory Usage

- **Database**: Minimal - indexed queries on SQLite
- **Context loading**: O(n) where n = context_window_size
- **LLM processing**: Linear with context size

### Latency

- **Context retrieval**: ~1-5ms (SQLite query)
- **Context formatting**: ~1-2ms (string operations)
- **LLM processing**: Depends on provider and context size
  - 10 exchanges (~2KB): +100-200ms
  - 20 exchanges (~4KB): +200-400ms

### Optimization Tips

1. **Adjust window size** based on your use case
   - Short conversations: smaller window (5-10)
   - Long discussions: larger window (15-20)

2. **Monitor context_used** in responses

   ```python
   if response["context_used"] > 15:
       # Consider summarization
   ```

3. **Use character limits** to prevent context overflow

   ```python
   context = get_recent_context(max_chars=3000)
   ```

## Testing

Run the comprehensive test suite:

```bash
python tests/test_persistent_chat.py
```

Tests cover:

- ✅ Session persistence
- ✅ Context loading
- ✅ Context formatting
- ✅ Window limits
- ✅ Multi-turn conversations
- ✅ Configuration loading

## Migration Guide

### From Non-Persistent Chat

**Before:**

```python
# Each chat was independent
response = await query_llm("Hello")
```

**After:**

```python
# Chats are automatically persistent
# No code changes required - happens automatically!
response = await query_llm("Hello")
```

### Existing Code Compatibility

The persistent chat system is **fully backward compatible**:

- Old endpoints continue to work
- Session IDs are optional
- No breaking changes to API

## Advanced Features

### Custom Session Management

```python
# Create a specific session for a user
user_session = f"user-{user_id}"

# Load that user's context
context = get_recent_context(session_id=user_session)

# Query with user-specific context
response = await query_llm(
    prompt,
    conversation_context=context
)

# Log to that user's session
log_conversation(prompt, response, session_id=user_session)
```

### Context Summarization

For very long conversations:

```python
from src.vega.core.db import get_conversation_summary

# Get summary of older exchanges
summary = get_conversation_summary(
    session_id=session_id,
    older_than_id=last_context_id,
    max_entries=100
)

# Combine summary with recent context
full_context = [
    {"prompt": "Earlier...", "response": summary},
    *recent_context
]
```

### Multi-User Scenarios

```python
# Each user gets their own persistent session
sessions = {}

def get_user_session(user_id):
    if user_id not in sessions:
        sessions[user_id] = f"user-{user_id}-{uuid.uuid4()}"
    return sessions[user_id]

# Use in chat endpoint
session_id = request.session_id or get_user_session(request.user_id)
```

## Troubleshooting

### Issue: Context not loading

**Check:**

1. Verify conversations are being logged: `SELECT * FROM conversations`
2. Check session_id is consistent: `SELECT DISTINCT session_id FROM conversations`
3. Verify context_window_size in config

**Fix:**

```python
# Debug context loading
context = get_recent_context(session_id=session_id)
print(f"Loaded {len(context)} exchanges")
```

### Issue: Too much/too little context

**Adjust configuration:**

```bash
# Increase context
CONTEXT_WINDOW_SIZE=20
CONTEXT_MAX_CHARS=8000

# Decrease context
CONTEXT_WINDOW_SIZE=5
CONTEXT_MAX_CHARS=2000
```

### Issue: LLM responses seem confused

**Possible causes:**

1. Context too large (exceeds LLM capacity)
2. Context includes irrelevant old exchanges
3. Multiple unrelated topics mixed

**Solutions:**

- Reduce `CONTEXT_WINDOW_SIZE`
- Implement topic-based session separation
- Use conversation summaries for old exchanges

## Future Enhancements

Planned improvements:

- 🔍 **Semantic search** in conversation history
- 🏷️ **Topic clustering** for better context selection
- 📊 **Relevance scoring** to prioritize important exchanges
- 💾 **Vector embeddings** for semantic context retrieval
- 🤖 **Automatic summarization** using LLM for long histories

## Contributing

To extend the persistent chat system:

1. **Add new context retrieval strategies** in `db.py`
2. **Implement custom formatters** in `llm.py`
3. **Add tests** in `tests/test_persistent_chat.py`
4. **Update documentation** in this file

## License

Part of Vega2.0 - see main project LICENSE.
