# Vega 2.0 Cleanup & Memory Indexing Implementation Plan

**Date:** November 7, 2025  
**Status:** Planning Phase  
**Priority:** High

---

## Executive Summary

This document outlines a comprehensive plan to:

1. Clean up and consolidate the Vega 2.0 codebase
2. Implement a lightweight memory indexing and tagging system
3. Minimize reconfiguration needed for the new memory feature

### Key Findings

**Good News:**

- ✅ Vega already has a sophisticated memory system (`memory.py`) with `MemoryIndex` table
- ✅ Vector database infrastructure exists (`vector_database.py`)
- ✅ Multiple database schemas already support tagging and indexing
- ✅ Core architecture is well-structured and modular

**Areas Requiring Attention:**

- ⚠️ **717 `__pycache__` directories** consuming disk space
- ⚠️ **Duplicate database files** in `/data/` and `/src/vega/core/`
- ⚠️ **Multiple log directories** that could be consolidated
- ⚠️ **158 test files** scattered across different locations
- ⚠️ Memory indexing exists but needs lightweight tagging enhancements

---

## Part 1: Project Cleanup Recommendations

### A. Immediate Cleanup (Safe to Execute)

#### 1. Remove Python Cache Files

**Issue:** 717 `__pycache__` directories consuming unnecessary disk space

**Action:**

```bash
# Remove all __pycache__ directories
find /home/ncacord/Vega2.0 -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Add to .gitignore if not already present
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo "*.pyo" >> .gitignore
```

**Impact:** Reduces disk usage, improves repository cleanliness  
**Risk:** None (Python regenerates these automatically)

#### 2. Clean Up SQLite Temporary Files

**Issue:** Multiple `.db-shm` and `.db-wal` files scattered across project

**Files to Clean:**

- `/src/vega/core/vega.db-shm`, `/src/vega/core/vega.db-wal`
- `/src/vega/core/vega_memory.db-shm`, `/src/vega/core/vega_memory.db-wal`
- `/data/vega.db-wal`, `/data/vega.db-shm`

**Action:**

```bash
# Clean temporary SQLite files (safe when DB is not in use)
find /home/ncacord/Vega2.0 -type f \( -name "*.db-shm" -o -name "*.db-wal" \) -delete
```

**Impact:** Reduces clutter, these files are regenerated as needed  
**Risk:** Low (only clean when Vega is not running)

#### 3. Archive Old Log Files

**Issue:** 1.1MB of logs in `/logs/` directory

**Action:**

```bash
# Create archive directory
mkdir -p /home/ncacord/Vega2.0/logs/archive

# Move logs older than 30 days
find /home/ncacord/Vega2.0/logs -type f -name "*.log" -mtime +30 -exec mv {} /home/ncacord/Vega2.0/logs/archive/ \;

# Compress archived logs
tar -czf /home/ncacord/Vega2.0/logs/archive_$(date +%Y%m%d).tar.gz /home/ncacord/Vega2.0/logs/archive/*.log
rm /home/ncacord/Vega2.0/logs/archive/*.log
```

**Impact:** Reduces active log size while preserving history  
**Risk:** None (logs are archived, not deleted)

### B. Database Consolidation (Requires Careful Planning)

#### Issue: Duplicate Database Files

**Current State:**

```
Databases in /data/:
- evaluations.db
- knowledge_graph.db
- self_improvement.db
- skill_registry.db
- telemetry.db
- variants.db
- vega.db
- vega_memory.db

Databases in /src/vega/core/:
- vega.db (DUPLICATE)
- vega_memory.db (DUPLICATE)
```

**Recommended Action:**

1. **Consolidate Primary Databases to `/data/`:**
   - Move all database operations to reference `/data/vega.db` and `/data/vega_memory.db`
   - Remove duplicate databases from `/src/vega/core/`
   - Update `db.py` and `memory.py` to use centralized data directory

2. **Database Purpose Clarification:**

   ```
   /data/vega.db            → Main conversation/interaction storage
   /data/vega_memory.db     → Memory indexing and knowledge storage
   /data/knowledge_graph.db → Graph-based knowledge relationships
   /data/evaluations.db     → Model evaluation metrics
   /data/self_improvement.db → Self-optimization tracking
   /data/skill_registry.db  → Skill management
   /data/telemetry.db       → Performance/usage metrics
   /data/variants.db        → A/B testing and model variants
   ```

**Implementation Steps:**

```python
# Update DB_PATH in src/vega/core/db.py
DB_PATH = os.path.join(os.path.dirname(__file__), "../../../data/vega.db")

# Update MEMORY_DB_PATH in src/vega/core/memory.py
MEMORY_DB_PATH = os.path.join(os.path.dirname(__file__), "../../../data/vega_memory.db")
```

**Risk:** Medium - requires database migration and testing  
**Benefit:** Single source of truth, easier backups, clearer architecture

### C. Log Directory Consolidation

**Current State:**

```
/logs/                          (1.1MB - primary logs)
/data/logs/                     (4KB - nearly empty)
/data/vega_logs/                (4KB - nearly empty)
/src/vega/federated/logs/       (separate logs)
/tests/federated/logs/          (test logs)
/tools/autonomous_debug/logs/   (tool logs)
/tools/sac/logs/                (SAC tool logs)
```

**Recommended Structure:**

```
/logs/
  ├── core/          (main application logs)
  ├── federated/     (federated learning logs)
  ├── tests/         (test execution logs)
  ├── tools/         (tool-specific logs)
  │   ├── autonomous_debug/
  │   └── sac/
  └── archive/       (old logs)
```

**Action:**

```bash
# Consolidate empty log directories
rmdir /home/ncacord/Vega2.0/data/logs
rmdir /home/ncacord/Vega2.0/data/vega_logs

# Create unified log structure
mkdir -p /home/ncacord/Vega2.0/logs/{core,federated,tests,tools,archive}
```

**Risk:** Low  
**Benefit:** Easier log management and monitoring

### D. Test Organization

**Current State:** 158 test files scattered across:

- `/tests/` (main test suite)
- `/datasets/test_*.py` (dataset-specific tests)
- `/tests/federated/` (federated learning tests)
- `/test_scripts/` (utility test scripts)
- `/test_results/` (test outputs)

**Recommended Action:**

1. **Consolidate All Tests Under `/tests/`:**

   ```
   /tests/
     ├── unit/           (unit tests)
     ├── integration/    (integration tests)
     ├── datasets/       (dataset processing tests)
     ├── federated/      (existing federated tests)
     ├── conftest.py     (pytest configuration)
     └── README.md       (test documentation)
   ```

2. **Move Dataset Tests:**

   ```bash
   mv /home/ncacord/Vega2.0/datasets/test_*.py /home/ncacord/Vega2.0/tests/datasets/
   ```

3. **Archive Test Results:**

   ```bash
   mv /home/ncacord/Vega2.0/test_results /home/ncacord/Vega2.0/tests/results_archive
   ```

**Risk:** Low (update import paths in moved tests)  
**Benefit:** Unified testing structure, easier CI/CD integration

---

## Part 2: Memory Indexing Enhancement

### Current Memory System Analysis

**Existing Infrastructure:**

1. **`src/vega/core/memory.py`** - Comprehensive memory system with:
   - `KnowledgeItem` table (stores knowledge with metadata)
   - `MemoryIndex` table (fast lookup and search)
   - `FavoriteItem` table (usage-based favorites)
   - `AccessLog` table (comprehensive tracking)

2. **`src/vega/multimodal/vector_database.py`** - Vector database with:
   - FAISS integration for similarity search
   - Multiple index types (FLAT, IVF, HNSW)
   - Metadata filtering and hybrid search

3. **Existing Tagging Support:**
   - `KnowledgeItem.meta_data` field (JSON metadata)
   - `Conversation.tags` field in main db
   - Search terms in `MemoryIndex`

### What's Already Implemented ✅

- ✅ Database schema for knowledge storage
- ✅ Memory indexing with fast lookup
- ✅ Usage tracking and access logs
- ✅ Favorite items based on usage patterns
- ✅ Vector similarity search capabilities
- ✅ Topic-based organization
- ✅ Metadata storage (JSON)

### What Needs Enhancement 🔧

#### 1. Lightweight Task-Based Tagging System

**Problem:** Current system doesn't explicitly track which knowledge was useful for specific *tasks*.

**Solution:** Add a `TaskMemory` system that bridges tasks with knowledge items.

**New Schema Addition:**

```python
class TaskMemory(Base):
    """Tracks which knowledge items were useful for specific task types"""
    __tablename__ = "task_memory"
    
    id = Column(Integer, primary_key=True)
    task_type = Column(String(100), nullable=False, index=True)  # e.g., "code_review", "debugging", "explanation"
    task_hash = Column(String(64), nullable=False, index=True)  # Hash of task context
    knowledge_item_id = Column(Integer, ForeignKey("knowledge_items.id"))
    relevance_score = Column(Float, default=1.0)  # How useful was this knowledge?
    used_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    success = Column(Boolean, default=True)  # Was the task successful?
    
    __table_args__ = (
        Index("idx_task_type_score", "task_type", "relevance_score"),
    )
```

#### 2. Automatic Tagging During Conversations

**Integration Point:** Modify `log_conversation()` in `db.py` to auto-tag with task context.

**Implementation:**

```python
def log_conversation_with_memory(
    prompt: str, 
    response: str, 
    source: str = "api", 
    session_id: Optional[str] = None,
    task_type: Optional[str] = None,
    relevant_knowledge: Optional[List[int]] = None
) -> int:
    """
    Enhanced conversation logging that links to memory system.
    
    Args:
        task_type: Type of task being performed (auto-detected if not provided)
        relevant_knowledge: List of knowledge_item_id that were used
    """
    # Log conversation normally
    conv_id = log_conversation(prompt, response, source, session_id)
    
    # Auto-detect task type if not provided
    if task_type is None:
        task_type = detect_task_type(prompt)
    
    # Link knowledge items used in this conversation
    if relevant_knowledge:
        from .memory import link_task_to_knowledge
        task_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        link_task_to_knowledge(task_type, task_hash, relevant_knowledge)
    
    return conv_id
```

#### 3. Smart Knowledge Retrieval

**Feature:** Given a task type, quickly retrieve most relevant knowledge.

```python
def get_relevant_knowledge(
    task_type: str, 
    limit: int = 10,
    min_relevance: float = 0.5
) -> List[KnowledgeItem]:
    """
    Retrieve knowledge items most relevant to a task type.
    
    Combines:
    - Historical usage patterns (TaskMemory)
    - Recency (last_used_at)
    - Success rate (successful task completions)
    """
    # Query that joins KnowledgeItem with TaskMemory
    # Ranks by relevance_score * recency_factor * success_rate
```

#### 4. Lightweight Tag Management API

**New Endpoints in `app.py`:**

```python
@app.post("/api/memory/tag")
async def tag_knowledge_item(
    item_id: int,
    tags: List[str],
    x_api_key: str = Header(None)
):
    """Add tags to a knowledge item"""
    
@app.get("/api/memory/search-by-tags")
async def search_by_tags(
    tags: List[str],
    task_type: Optional[str] = None,
    x_api_key: str = Header(None)
):
    """Search knowledge items by tags and task type"""

@app.get("/api/memory/task-knowledge")
async def get_task_knowledge(
    task_type: str,
    limit: int = 10,
    x_api_key: str = Header(None)
):
    """Get most relevant knowledge for a task type"""
```

### Implementation Complexity Assessment

**Low Complexity (1-2 days):**

- ✅ Add `TaskMemory` table to existing schema
- ✅ Implement task type detection heuristics
- ✅ Create basic tag management functions
- ✅ Add REST API endpoints

**Medium Complexity (3-5 days):**

- ⚠️ Integrate with existing conversation flow
- ⚠️ Implement smart relevance scoring
- ⚠️ Add automatic knowledge tagging during conversations
- ⚠️ Create CLI commands for memory management

**High Complexity (5-7 days):**

- 🔴 Advanced semantic task matching
- 🔴 Machine learning-based relevance prediction
- 🔴 Integration with vector database for hybrid search
- 🔴 Knowledge graph visualization

---

## Part 3: Integration Plan

### Phase 1: Minimal Integration (Recommended First)

**Goal:** Add task-based memory tracking without disrupting existing functionality.

**Steps:**

1. **Extend Memory Schema (30 min)**

   ```bash
   # Add TaskMemory table to memory.py
   # No migration needed - SQLAlchemy creates new table automatically
   ```

2. **Create Memory Tagging Module (2 hours)**

   ```bash
   # Create src/vega/core/memory_tagging.py
   # Implement: tag_for_task(), get_task_memories(), detect_task_type()
   ```

3. **Add API Endpoints (1 hour)**

   ```python
   # Add 3 endpoints to app.py:
   # - POST /api/memory/tag-task
   # - GET /api/memory/task-knowledge
   # - GET /api/memory/search-by-tag
   ```

4. **Optional Integration with Conversations (2 hours)**

   ```python
   # Modify log_conversation() to accept optional task context
   # Add task_type parameter to /chat endpoint
   ```

5. **Testing & Documentation (1 hour)**

   ```bash
   # Create tests/test_memory_tagging.py
   # Update docs/features/MEMORY_FEATURE.md
   ```

**Total Time:** ~1 day  
**Risk:** Very Low  
**Benefit:** Lightweight memory indexing operational

### Phase 2: Enhanced Integration (Optional)

**Goal:** Full automation with smart task detection and relevance scoring.

**Steps:**

1. **Implement Task Detection AI (1 day)**
   - Use LLM to classify task types from prompts
   - Build lightweight classifier for common patterns

2. **Automatic Knowledge Tagging (1 day)**
   - Modify LLM interaction to track which knowledge was retrieved
   - Auto-link conversations with knowledge items

3. **Smart Retrieval System (2 days)**
   - Implement relevance scoring algorithm
   - Integrate with vector database for semantic search
   - Add caching layer for frequently accessed knowledge

4. **CLI Tools (1 day)**

   ```bash
   python main.py cli memory tag-task "debugging" --knowledge-id 123
   python main.py cli memory get-task "code_review"
   python main.py cli memory stats
   ```

**Total Time:** ~5 days  
**Risk:** Medium  
**Benefit:** Fully automated intelligent memory system

---

## Part 4: Reconfiguration Requirements

### Configuration Changes Needed

#### 1. Database Path Updates (Required for Cleanup)

**File:** `src/vega/core/db.py`

```python
# Change from:
DB_PATH = os.path.join(os.path.dirname(__file__), "vega.db")

# To:
DB_PATH = os.path.join(os.path.dirname(__file__), "../../../data/vega.db")
```

**File:** `src/vega/core/memory.py`

```python
# Change from:
MEMORY_DB_PATH = os.path.join(os.path.dirname(__file__), "vega_memory.db")

# To:
MEMORY_DB_PATH = os.path.join(os.path.dirname(__file__), "../../../data/vega_memory.db")
```

#### 2. New Configuration Options (Optional for Memory Enhancement)

**File:** `src/vega/core/config.py`

```python
@dataclass(frozen=True)
class Config:
    # ... existing fields ...
    
    # Memory indexing settings
    memory_task_tagging: bool = True  # Enable automatic task tagging
    memory_auto_relevance: bool = True  # Auto-calculate relevance scores
    memory_task_detection: str = "heuristic"  # "heuristic", "llm", "hybrid"
    memory_min_relevance_score: float = 0.5  # Min score for retrieval
    memory_max_task_memories: int = 100  # Max memories per task type
```

**File:** `.env`

```bash
# Add these optional settings:
MEMORY_TASK_TAGGING=true
MEMORY_AUTO_RELEVANCE=true
MEMORY_TASK_DETECTION=heuristic
MEMORY_MIN_RELEVANCE_SCORE=0.5
```

#### 3. Import Updates (If Tests Are Moved)

**Example for moved dataset tests:**

```python
# In tests/datasets/test_audio_utils.py
# Change from:
from audio_utils import process_audio

# To:
from datasets.audio_utils import process_audio
```

### Configuration Testing Checklist

- [ ] Database paths resolve correctly
- [ ] All databases accessible from unified location
- [ ] Test suite runs successfully after reorganization
- [ ] Log directories created and writable
- [ ] Memory system initializes without errors
- [ ] API endpoints respond correctly
- [ ] CLI commands execute properly

---

## Part 5: Execution Priorities

### Priority 1: Quick Wins (Execute Immediately) ⚡

1. **Clean **pycache** directories** (5 minutes)
2. **Archive old log files** (10 minutes)
3. **Clean SQLite temp files** (5 minutes)

**Impact:** Immediate cleanup, zero risk  
**Total Time:** 20 minutes

### Priority 2: Database Consolidation (Execute This Week) 📊

1. **Update database paths** in `db.py` and `memory.py`
2. **Test database access** from consolidated location
3. **Remove duplicate databases** from `/src/vega/core/`
4. **Update documentation** with new paths

**Impact:** Better architecture, easier maintenance  
**Total Time:** 2-3 hours

### Priority 3: Memory Indexing MVP (Execute Within 2 Weeks) 🧠

1. **Add TaskMemory table** to memory schema
2. **Create memory_tagging.py** module
3. **Add 3 REST API endpoints** for memory management
4. **Write tests** and documentation

**Impact:** Lightweight memory indexing operational  
**Total Time:** 1 day

### Priority 4: Log Consolidation (Execute This Month) 📁

1. **Create unified log structure**
2. **Update logging configuration** in relevant modules
3. **Migrate existing logs** to new structure

**Impact:** Better log management  
**Total Time:** 2-3 hours

### Priority 5: Test Reorganization (Execute This Month) 🧪

1. **Move dataset tests** to `/tests/datasets/`
2. **Update import paths** in moved tests
3. **Verify test suite** execution
4. **Archive old test results**

**Impact:** Better test organization  
**Total Time:** 3-4 hours

### Priority 6: Enhanced Memory Features (Execute Next Month) 🚀

1. **Implement smart task detection**
2. **Add automatic knowledge tagging**
3. **Create CLI memory tools**
4. **Integrate with vector database**

**Impact:** Fully automated intelligent memory  
**Total Time:** 5 days

---

## Part 6: Risk Assessment

### Low Risk ✅

- Removing **pycache** directories
- Archiving old logs
- Adding new database tables (non-breaking)
- Adding new API endpoints

### Medium Risk ⚠️

- Consolidating databases (requires migration)
- Moving test files (import path updates needed)
- Modifying existing conversation logging

### High Risk 🔴

- Changing core database schemas
- Modifying LLM interaction flows
- Large-scale file reorganization without backups

### Mitigation Strategies

1. **Create Backups Before Database Changes:**

   ```bash
   cp -r /home/ncacord/Vega2.0/data /home/ncacord/Vega2.0/data_backup_$(date +%Y%m%d)
   ```

2. **Test Database Migration:**

   ```bash
   # Test on copy first
   cp vega.db vega_test.db
   # Run migration
   # Verify with queries
   ```

3. **Use Git Branches:**

   ```bash
   git checkout -b cleanup-and-memory-indexing
   # Make changes
   # Test thoroughly
   # Merge when stable
   ```

4. **Incremental Rollout:**
   - Deploy memory tagging as optional feature first
   - Monitor for issues
   - Enable by default after validation period

---

## Part 7: Success Metrics

### Cleanup Success Criteria

- ✅ Disk space reduced by >100MB (**pycache** removal)
- ✅ Single source of truth for each database
- ✅ All logs in unified directory structure
- ✅ All tests passing after reorganization
- ✅ No duplicate or orphaned files

### Memory Indexing Success Criteria

- ✅ Task-based memory tagging operational
- ✅ API endpoints responding correctly
- ✅ Knowledge retrieval < 100ms for common tasks
- ✅ Relevance scoring improves over time
- ✅ CLI tools functional
- ✅ Documentation complete

### Performance Targets

- Database query time: < 50ms (95th percentile)
- Memory indexing overhead: < 10ms per conversation
- Disk space for memory index: < 10MB for 1000 knowledge items
- API response time: < 200ms for knowledge retrieval

---

## Appendix A: File Organization Summary

### Files to Remove

```
# After migration:
/src/vega/core/vega.db (duplicate)
/src/vega/core/vega_memory.db (duplicate)
/data/logs/ (empty directory)
/data/vega_logs/ (empty directory)

# Temporary files:
All *.db-shm and *.db-wal files
All __pycache__/ directories
```

### Files to Move

```
# Dataset tests:
/datasets/test_*.py → /tests/datasets/

# Test results:
/test_results/ → /tests/results_archive/
```

### Files to Create

```
/src/vega/core/memory_tagging.py (new module)
/tests/test_memory_tagging.py (new tests)
/docs/operations/CLEANUP_AND_MEMORY_INDEXING_PLAN.md (this document)
/logs/{core,federated,tests,tools,archive}/ (new directories)
```

---

## Appendix B: Quick Command Reference

### Cleanup Commands

```bash
# Remove __pycache__
find /home/ncacord/Vega2.0 -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Clean SQLite temp files (when Vega not running)
find /home/ncacord/Vega2.0 -type f \( -name "*.db-shm" -o -name "*.db-wal" \) -delete

# Archive old logs
mkdir -p /home/ncacord/Vega2.0/logs/archive
find /home/ncacord/Vega2.0/logs -type f -name "*.log" -mtime +30 -exec mv {} /home/ncacord/Vega2.0/logs/archive/ \;

# Create backup
cp -r /home/ncacord/Vega2.0/data /home/ncacord/Vega2.0/data_backup_$(date +%Y%m%d)
```

### Memory System Commands (After Implementation)

```bash
# Tag knowledge for a task
python main.py cli memory tag-task "code_review" --knowledge-id 123

# Get knowledge for task type
python main.py cli memory get-task "debugging" --limit 10

# Search by tags
python main.py cli memory search --tags "python,optimization"

# Memory stats
python main.py cli memory stats
```

---

## Conclusion

**Summary:**

1. **Cleanup:** ~2-4 hours of work to consolidate and clean up the codebase
2. **Memory Enhancement:** Already 70% implemented! Just needs task-based tagging (1 day)
3. **Reconfiguration:** Minimal - mostly path updates and optional config additions
4. **Risk:** Low for cleanup, Very Low for memory enhancement
5. **Benefit:** Cleaner codebase + intelligent memory indexing

**Recommendation:** 

Execute in this order:

1. Quick cleanup (Priority 1) - **Do today**
2. Memory indexing MVP (Priority 3) - **Do this week**  
3. Database consolidation (Priority 2) - **Do next week**
4. Log/test reorganization (Priorities 4-5) - **Do this month**

The memory indexing system is largely already built! You just need to add the lightweight task tagging layer on top of existing infrastructure. This is a very favorable situation - much less work than starting from scratch.
