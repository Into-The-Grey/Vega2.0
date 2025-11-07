# 🚀 Quick Start: Vega Memory Indexing System

**5-Minute Setup Guide**

---

## ✅ Step 1: Verify Installation (30 seconds)

```bash
cd /home/ncacord/Vega2.0

# Check if files exist
ls src/vega/core/memory_tagging.py
ls tests/test_memory_tagging.py
ls scripts/cleanup.sh
```

**Expected:** All files exist ✓

---

## ✅ Step 2: Run Tests (2 minutes)

```bash
python -m pytest tests/test_memory_tagging.py -v
```

**Expected Output:**

```
test_memory_tagging.py::TestTaskDetection::test_detect_debugging_task PASSED
test_memory_tagging.py::TestTaskDetection::test_detect_code_review_task PASSED
...
==================== X passed in Y.YYs ====================
```

---

## ✅ Step 3: Clean Up Project (1 minute)

```bash
bash scripts/cleanup.sh
```

**What It Does:**

- Removes 717 `__pycache__` directories
- Cleans SQLite temp files
- Frees up disk space

---

## ✅ Step 4: Start Vega & Test API (2 minutes)

**Terminal 1 - Start Server:**

```bash
python main.py server --host 127.0.0.1 --port 8000
```

**Terminal 2 - Test Endpoints:**

```bash
# Test 1: Detect task type
curl -X POST http://127.0.0.1:8000/api/memory/detect-task \
  -H "Content-Type: application/json" \
  -H "X-API-Key: vega-default-key" \
  -d '"Help me debug this error"'

# Expected: {"detected_task_type": "debugging", ...}

# Test 2: Get statistics
curl http://127.0.0.1:8000/api/memory/task-stats \
  -H "X-API-Key: vega-default-key"

# Expected: {"total_memories": 0, "avg_relevance": 0.0, ...}
```

---

## 🎯 Usage Example

**In Your Python Code:**

```python
from src.vega.core.memory_tagging import (
    detect_task_type,
    tag_knowledge_for_task,
    get_task_knowledge,
)

# 1. User asks a question
user_prompt = "How do I fix this memory leak?"

# 2. Detect what they're asking about
task_type = detect_task_type(user_prompt)
# Returns: "debugging"

# 3. Get relevant knowledge you already have
knowledge = get_task_knowledge(task_type, limit=5)

# 4. Use it to help answer (your existing logic)
# ...

# 5. Tag what was useful for next time
tag_knowledge_for_task(
    task_type=task_type,
    task_context=user_prompt,
    knowledge_item_ids=[k.id for k, _ in knowledge[:2]],
    relevance_score=0.9,
    success=True
)
```

---

## 📊 Check System Status

**View Memory Statistics:**

```bash
curl http://127.0.0.1:8000/api/memory/task-stats \
  -H "X-API-Key: vega-default-key" | python -m json.tool
```

**Monitor Logs:**

```bash
tail -f logs/core/core.log | grep MEMORY
```

---

## 🎓 Task Types Available

- `debugging` - Bug fixes and error resolution
- `code_review` - Code validation and review
- `explanation` - Explaining concepts
- `optimization` - Performance improvements
- `refactoring` - Code restructuring
- `feature_implementation` - New features
- `documentation` - Writing docs
- `testing` - Test creation
- `security_analysis` - Security reviews
- `api_design` - API development
- `data_analysis` - Data processing
- `architecture_design` - System design
- `general_question` - General queries

---

## 📚 Full Documentation

- **API Reference:** `docs/features/MEMORY_TAGGING.md`
- **Cleanup Plan:** `docs/operations/CLEANUP_AND_MEMORY_INDEXING_PLAN.md`
- **Implementation Details:** `docs/operations/IMPLEMENTATION_SUMMARY.md`

---

## 🆘 Common Issues

**API returns 401 Unauthorized:**

```bash
# Make sure to include API key header
-H "X-API-Key: vega-default-key"
```

**Tests failing:**

```bash
# Install dependencies
pip install -r requirements.txt
```

**Server won't start:**

```bash
# Check if port 8000 is already in use
lsof -i :8000

# Use different port
python main.py server --host 127.0.0.1 --port 8001
```

---

## ✨ You're All Set

The memory indexing system is ready to use. Start tagging knowledge and watch Vega get smarter with every interaction! 🧠

**Next:** Read `MEMORY_TAGGING.md` for advanced features and best practices.
