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
```
