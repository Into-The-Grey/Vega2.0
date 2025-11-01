# Test Scripts

This directory contains various test scripts for validating Vega's functionality.

## Available Tests

### Ethics & Safety Tests

- **`censorship_test.py`** - Tests AI ethics and censorship behavior across categories (technical, controversial, illegal, harmful, direct harm)
- **`test_code_quality.py`** - Validates that generated code is production-ready without placeholder/dummy code

### Performance Tests

- **`stress_test.py`** - Standard load testing for API endpoints
- **`extreme_stress_test.py`** - Heavy load testing with concurrent requests

### Functional Tests

- **`simple_topic_test.py`** - Tests model responses across 22 diverse topics
- **`topic_range_test.py`** - Extended topic coverage testing

## Running Tests

```bash
cd /home/ncacord/Vega2.0

# Ethics tests
python3 test_scripts/censorship_test.py
python3 test_scripts/test_code_quality.py

# Performance tests
python3 test_scripts/stress_test.py
python3 test_scripts/extreme_stress_test.py

# Functional tests
python3 test_scripts/simple_topic_test.py
python3 test_scripts/topic_range_test.py
```

## Results

Test results are stored in `/test_results/` directory.

## Requirements

- Vega server must be running on port 8000
- API key: `devkey` (configured in `.env`)
- Python dependencies from `requirements.txt`
