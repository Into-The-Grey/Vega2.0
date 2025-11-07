# Vega Memory Indexing & Tagging System

## Overview

Vega's memory indexing system provides lightweight, task-based memory tagging that helps the AI remember which pieces of information were useful for specific types of tasks. This enables Vega to quickly retrieve relevant knowledge without retraining or extensive searching.

## Key Features

✅ **Task-Based Tagging** - Automatically categorize knowledge by task type  
✅ **Smart Retrieval** - Quickly find relevant information for similar tasks  
✅ **Relevance Scoring** - Track which knowledge is most useful  
✅ **Usage Patterns** - Learn from historical task completions  
✅ **Lightweight Design** - Minimal overhead, no retraining required  
✅ **REST API** - Easy integration with existing workflows

## Architecture

### Database Schema

The memory tagging system extends Vega's existing memory infrastructure with a new `TaskMemory` table:

```python
class TaskMemory:
    - task_type: Type of task (e.g., "debugging", "code_review")
    - task_hash: Hash of task context for deduplication
    - knowledge_item_id: Link to KnowledgeItem
    - relevance_score: How useful was this knowledge (0.0-1.0)
    - used_at: When was it last used
    - success: Was the task successful?
    - context_snippet: Brief context of usage
```

### Task Types

The system recognizes 13 common task types:

1. **code_review** - Code review and validation
2. **debugging** - Bug fixing and error resolution
3. **explanation** - Explaining concepts and code
4. **refactoring** - Code improvement and restructuring
5. **feature_implementation** - Building new features
6. **documentation** - Writing docs and comments
7. **testing** - Creating and running tests
8. **optimization** - Performance and efficiency improvements
9. **security_analysis** - Security reviews and vulnerability checks
10. **api_design** - API endpoint design
11. **data_analysis** - Data processing and analysis
12. **architecture_design** - System architecture planning
13. **general_question** - General inquiries

## API Endpoints

### 1. Tag Knowledge for Task

Associate knowledge items with a task type:

```bash
POST /api/memory/tag-task
X-API-Key: your-api-key

{
  "task_type": "debugging",
  "task_context": "Debugging a NoneType error in API handler",
  "knowledge_item_ids": [123, 456, 789],
  "relevance_score": 0.9,
  "success": true
}
```

**Response:**

```json
{
  "status": "success",
  "task_type": "debugging",
  "tagged_items": 3,
  "new_tags": 2,
  "updated_tags": 1
}
```

### 2. Get Task-Relevant Knowledge

Retrieve knowledge most relevant to a task type:

```bash
GET /api/memory/task-knowledge/debugging?limit=10&min_relevance=0.5
X-API-Key: your-api-key
```

**Response:**

```json
{
  "task_type": "debugging",
  "knowledge_items": [
    {
      "id": 123,
      "key": "python-none-type-errors",
      "topic": "python-debugging",
      "content": "NoneType errors occur when...",
      "relevance_score": 0.95,
      "usage_count": 15,
      "last_used_at": "2025-11-07T10:30:00Z"
    }
  ],
  "count": 1
}
```

### 3. Search by Tags

Search knowledge items by tags:

```bash
GET /api/memory/search-by-tags?tags=python,optimization&task_type=optimization&limit=20
X-API-Key: your-api-key
```

**Response:**

```json
{
  "tags": ["python", "optimization"],
  "task_type": "optimization",
  "knowledge_items": [...],
  "count": 5
}
```

### 4. Get Task Statistics

View memory indexing usage statistics:

```bash
GET /api/memory/task-stats
X-API-Key: your-api-key
```

**Response:**

```json
{
  "total_memories": 1547,
  "avg_relevance": 0.847,
  "unique_tasks": 13,
  "unique_knowledge_items": 892,
  "task_breakdown": [
    {"task_type": "debugging", "count": 450},
    {"task_type": "code_review", "count": 320},
    {"task_type": "explanation", "count": 280}
  ]
}
```

### 5. Detect Task Type

Automatically detect task type from a prompt:

```bash
POST /api/memory/detect-task?method=heuristic
X-API-Key: your-api-key

"Help me debug this memory leak in my Python application"
```

**Response:**

```json
{
  "prompt": "Help me debug this memory leak...",
  "detected_task_type": "debugging",
  "detection_method": "heuristic"
}
```

## Python API Usage

### Basic Usage

```python
from src.vega.core.memory_tagging import (
    detect_task_type,
    tag_knowledge_for_task,
    get_task_knowledge,
    get_task_statistics,
)

# 1. Detect task type from user prompt
prompt = "Help me optimize this database query"
task_type = detect_task_type(prompt)  # Returns: "optimization"

# 2. Tag knowledge items used for this task
tag_knowledge_for_task(
    task_type=task_type,
    task_context=prompt,
    knowledge_item_ids=[101, 102, 103],
    relevance_score=0.9,
    success=True
)

# 3. Later, retrieve relevant knowledge for similar tasks
knowledge_items = get_task_knowledge(
    task_type="optimization",
    limit=10,
    min_relevance=0.7
)

for item, relevance in knowledge_items:
    print(f"Knowledge: {item.key} (relevance: {relevance:.2f})")

# 4. View statistics
stats = get_task_statistics(task_type="optimization")
print(f"Optimization tasks: {stats['total_memories']}")
print(f"Success rate: {stats['success_rate']:.1%}")
```

### Integration with Conversations

```python
from src.vega.core.db import log_conversation
from src.vega.core.memory_tagging import detect_task_type, tag_knowledge_for_task

def enhanced_conversation_handler(prompt: str, response: str, knowledge_used: list):
    """Handle a conversation with automatic memory tagging"""
    
    # Log conversation
    conv_id = log_conversation(prompt, response)
    
    # Detect task type
    task_type = detect_task_type(prompt)
    
    # Tag knowledge that was used
    if knowledge_used:
        tag_knowledge_for_task(
            task_type=task_type,
            task_context=prompt,
            knowledge_item_ids=knowledge_used,
            relevance_score=0.85,  # Can be computed based on response quality
            success=True
        )
    
    return conv_id, task_type
```

## CLI Usage

### Tag Knowledge for a Task

```bash
python main.py cli memory tag-task "debugging" \
  --context "Fixing NoneType error in API" \
  --knowledge-ids 123,456 \
  --relevance 0.9
```

### Get Task Knowledge

```bash
python main.py cli memory get-task "code_review" --limit 10
```

### Search by Tags

```bash
python main.py cli memory search --tags "python,async" --task-type "optimization"
```

### View Statistics

```bash
python main.py cli memory stats
python main.py cli memory stats --task-type "debugging"
```

## Task Detection Methods

The system supports three task detection methods:

### 1. Heuristic (Default)

Fast pattern-matching approach using regex patterns:

```python
task_type = detect_task_type(prompt, method="heuristic")
```

**Pros:** Fast, deterministic, no API calls  
**Cons:** May miss nuanced task types

### 2. LLM-Based

Uses the LLM to classify tasks (requires LLM API):

```python
task_type = detect_task_type(prompt, method="llm")
```

**Pros:** More accurate, handles complex prompts  
**Cons:** Slower, requires LLM call

### 3. Hybrid

Tries heuristic first, falls back to LLM if uncertain:

```python
task_type = detect_task_type(prompt, method="hybrid")
```

**Pros:** Balanced accuracy and speed  
**Cons:** May still require LLM calls

## Configuration

Add these options to your `.env` file:

```bash
# Enable automatic task tagging
MEMORY_TASK_TAGGING=true

# Enable automatic relevance scoring
MEMORY_AUTO_RELEVANCE=true

# Task detection method: heuristic, llm, or hybrid
MEMORY_TASK_DETECTION=heuristic

# Minimum relevance score for retrieval (0.0-1.0)
MEMORY_MIN_RELEVANCE_SCORE=0.5

# Maximum memories to track per task type
MEMORY_MAX_TASK_MEMORIES=100
```

Update `src/vega/core/config.py`:

```python
@dataclass(frozen=True)
class Config:
    # ... existing fields ...
    
    # Memory indexing settings
    memory_task_tagging: bool = True
    memory_auto_relevance: bool = True
    memory_task_detection: str = "heuristic"
    memory_min_relevance_score: float = 0.5
    memory_max_task_memories: int = 100
```

## Performance Characteristics

- **Task Detection (Heuristic):** < 1ms
- **Task Detection (LLM):** ~100-500ms (depends on LLM)
- **Knowledge Tagging:** < 10ms per tag
- **Knowledge Retrieval:** < 50ms for typical queries
- **Storage Overhead:** ~100 bytes per task memory record

## Best Practices

### 1. Tag Knowledge Immediately After Use

```python
# Right after using knowledge for a task
tag_knowledge_for_task(
    task_type=detected_task,
    task_context=original_prompt,
    knowledge_item_ids=used_knowledge,
    success=task_was_successful
)
```

### 2. Use Relevance Scoring

Track how useful knowledge was:

```python
# Knowledge that solved the problem completely
relevance_score = 1.0

# Knowledge that partially helped
relevance_score = 0.7

# Knowledge that was referenced but not critical
relevance_score = 0.5
```

### 3. Filter by Success

Retrieve knowledge from successful tasks:

```python
# This is handled automatically - failed tasks have lower weight
knowledge = get_task_knowledge(task_type="debugging", min_relevance=0.7)
```

### 4. Periodic Cleanup

Archive old task memories:

```python
# In a maintenance script
from datetime import datetime, timedelta
from src.vega.core.memory import Session, memory_engine
from src.vega.core.memory_tagging import TaskMemory

# Archive memories older than 6 months
cutoff = datetime.now() - timedelta(days=180)
with Session(memory_engine) as session:
    old_memories = session.query(TaskMemory).filter(
        TaskMemory.used_at < cutoff
    ).delete()
    session.commit()
```

## Troubleshooting

### Issue: Task detection returns "unknown" too often

**Solution:** Switch to hybrid or LLM-based detection:

```python
task_type = detect_task_type(prompt, method="hybrid")
```

Or add custom patterns to `TASK_PATTERNS` in `memory_tagging.py`.

### Issue: Retrieving too many irrelevant results

**Solution:** Increase minimum relevance threshold:

```python
knowledge = get_task_knowledge(
    task_type="debugging",
    min_relevance=0.8  # Higher threshold
)
```

### Issue: Database performance degrading

**Solution:** Check index usage and consider archiving:

```bash
# Check database size
du -h /home/ncacord/Vega2.0/data/vega_memory.db

# Run SQLite vacuum
sqlite3 /home/ncacord/Vega2.0/data/vega_memory.db "VACUUM;"
```

## Future Enhancements

🔮 **Planned Features:**

- [ ] Machine learning-based relevance prediction
- [ ] Knowledge graph integration for semantic relationships
- [ ] Multi-modal memory tagging (code, images, documents)
- [ ] Collaborative memory sharing across Vega instances
- [ ] Time-decay for relevance scores
- [ ] Auto-pruning of low-value memories
- [ ] Vector similarity for semantic task matching

## Related Documentation

- [Memory System Overview](./MEMORY_FEATURE.md)
- [Vector Database Integration](../architecture/VECTOR_DATABASE.md)
- [API Reference](../api/MEMORY_API.md)
- [Cleanup & Maintenance Plan](../operations/CLEANUP_AND_MEMORY_INDEXING_PLAN.md)

## Contributing

To add new task types:

1. Add pattern to `TASK_PATTERNS` in `memory_tagging.py`
2. Add constant to `TaskType` dataclass
3. Update tests in `tests/test_memory_tagging.py`
4. Update this documentation

## License

Part of Vega2.0 - See main LICENSE file.
