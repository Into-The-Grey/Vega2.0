## Vega2.0 Mind Map (Current Session Updates)

```markmap
# Vega 2.0 AI Platform (Session Snapshot)

## Document Intelligence
- src/vega/document/
  - understanding.py
    - Orchestrator
      - async health_check() → {healthy, overall_status, initialized, components}
      - analyze_content() → aggregates
        - content_analysis
        - semantic_analysis
        - summary
        - entities
      - analyze_semantics()
      - generate_summary()
      - extract_entities()
    - Components
      - SummaryGenerator
        - Strict length enforcement
          - Abstractive: hard cap at max_length
          - Extractive: ±20% tolerance
        - Safe word-trim
      - SemanticAnalyzer
        - Theme detection: api, technical, documentation, workflow
      - Input Validation
        - Empty/whitespace → standardized error result

## Datasets
- datasets/
  - voice_lines/
    - README (CSV schema)
    - .gitkeep
  - loaders/
    - loader_voice_csv.py

## Quality Gates
- Standardized orchestrator interface
  - health_check schema unified
  - Input validation centralized pattern (incremental)

## Repository hygiene
- Decluttered caches: __pycache__ (1,663 dirs removed), .pytest_cache, .benchmarks
- Removed stray logs (audit.log, test.log)
- Removed local env folder venv_pruning/
- Removed obsolete scripts/main.py (legacy informational script)
- Verified active structure: main.py → scripts/run_*.py imports intact
- Verified vega_state/ actively used (20+ imports in app.py, cli.py)
- Kept experimental tools/ directories (vega, sac, autonomous_debug, engine) for future development
- .gitignore updated to block reintroduction

## Build & tests
- Syntax fixes: database_optimization.py, user_profiling/cli.py, tools/test_suite/app.py, tools/vega/vega_init.py
- Compile check: PASS (src, tools)
- Tests (focused):
  - tests/test_app.py: PASS
  - tests/test_document_processing.py: PASS
  - tests/test_voice.py: PASS (enhanced manager honors patched providers / missing dependencies)

## Voice Engine
- Provider resolution
  - EnhancedVoiceManager resets when patched stubs detected
  - Single-call synthesize/transcribe wrappers handle sync + async providers
- Tests
  - Piper/Vosk mock patches respected without installing heavy dependencies

## User Profiling CLI
- Lazy optional imports
  - Cached loaders for daemon components/persona engine
  - Integration test uses importlib for core modules
  - Commands exit cleanly when optional deps missing
```
