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
- Decluttered caches: __pycache__, .pytest_cache, .benchmarks
- Removed stray logs (audit.log, test.log)
- Removed local env folder venv_pruning/
- .gitignore updated to block reintroduction

## Build & tests
- Syntax fixes: database_optimization.py, user_profiling/cli.py, tools/test_suite/app.py, tools/vega/vega_init.py
- Compile check: PASS (src, tools)
- Tests (focused):
  - tests/test_app.py: PASS
  - tests/test_document_processing.py: PASS
  - tests/test_voice.py: partial (piper/vosk providers optional; failing when providers absent)
```
