# Vega 2.0 Test Suite - Quick Reference

## 🚀 Essential Commands

### Run All Tests

```bash
# Basic run
python -m pytest tests/document -v

# Quiet mode (summary only)
python -m pytest tests/document -q

# Stop on first failure
python -m pytest tests/document -x

# Parallel execution (faster)
python -m pytest tests/document -n auto
```

### Run Specific Modules

```bash
# Base tests (infrastructure)
python -m pytest tests/document/test_base.py -v

# Classification tests
python -m pytest tests/document/test_classification.py -v

# Legal tests
python -m pytest tests/document/test_legal.py -v

# Technical tests
python -m pytest tests/document/test_technical.py -v

# Understanding tests
python -m pytest tests/document/test_understanding.py -v

# Workflow tests
python -m pytest tests/document/test_workflow.py -v
```

### Run Specific Test Categories

```bash
# Error handling tests only
python -m pytest tests/document -k "test_error_handling" -v

# Integration tests only
python -m pytest tests/document -k "integration" -v

# Edge case tests only
python -m pytest tests/document -k "edge_case" -v

# Performance tests only
python -m pytest tests/document -k "performance" -v
```

### Coverage Reports

```bash
# Generate coverage report
python -m pytest tests/document --cov=src/vega/document --cov-report=term

# HTML coverage report
python -m pytest tests/document --cov=src/vega/document --cov-report=html

# View HTML report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Debugging

```bash
# Show full traceback
python -m pytest tests/document/test_legal.py -v --tb=long

# Show local variables
python -m pytest tests/document/test_legal.py -v -l

# Drop into pdb on failure
python -m pytest tests/document/test_legal.py --pdb

# Show print statements
python -m pytest tests/document/test_legal.py -v -s
```

### Performance

```bash
# Show slowest tests
python -m pytest tests/document --durations=10

# Show all test durations
python -m pytest tests/document --durations=0

# Benchmark tests
python -m pytest tests/document --benchmark-only
```

---

## 📊 Test Status Commands

### Quick Status Check

```bash
# Summary only
python -m pytest tests/document -q --tb=no 2>&1 | grep "===" | tail -1

# Count passing tests
python -m pytest tests/document -q --tb=no 2>&1 | grep "passed"

# Count failing tests
python -m pytest tests/document -q --tb=no 2>&1 | grep "failed"
```

### Module-Specific Status

```bash
# Legal module status
python -m pytest tests/document/test_legal.py -q --tb=no 2>&1 | grep "==="

# Understanding module status
python -m pytest tests/document/test_understanding.py -q --tb=no 2>&1 | grep "==="
```

---

## 🔧 Maintenance Commands

### Clean Test Artifacts

```bash
# Remove cache
rm -rf .pytest_cache __pycache__ tests/__pycache__ tests/document/__pycache__

# Remove coverage data
rm -rf .coverage htmlcov

# Remove compiled Python files
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -exec rm -rf {} +
```

### Update Dependencies

```bash
# Update requirements
pip install -r requirements.txt --upgrade

# Install test dependencies
pip install pytest pytest-asyncio pytest-cov pytest-xdist pytest-benchmark
```

### Verify Installation

```bash
# Check pytest version
python -m pytest --version

# List installed plugins
python -m pytest --version --verbose

# Verify model availability
python -c "from transformers import AutoModel; print('✅ Transformers working')"
```

---

## 📈 Continuous Monitoring

### Watch Mode

```bash
# Re-run tests on file changes (requires pytest-watch)
ptw tests/document

# Re-run with coverage
ptw tests/document -- --cov=src/vega/document
```

### Git Hooks

```bash
# Run tests before commit
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
python -m pytest tests/document -q --tb=no
if [ $? -ne 0 ]; then
    echo "❌ Tests failed. Commit aborted."
    exit 1
fi
echo "✅ Tests passed. Proceeding with commit."
EOF

chmod +x .git/hooks/pre-commit
```

---

## 🎯 Common Workflows

### Before Starting Work

```bash
# 1. Pull latest changes
git pull origin main

# 2. Run full test suite
python -m pytest tests/document -v

# 3. Check current status
python -m pytest tests/document -q --tb=no 2>&1 | tail -1
```

### While Developing

```bash
# 1. Run relevant module tests
python -m pytest tests/document/test_legal.py -v

# 2. Run specific test
python -m pytest tests/document/test_legal.py::TestLegalDocumentAI::test_contract_analysis -v

# 3. Watch for changes
ptw tests/document/test_legal.py
```

### Before Committing

```bash
# 1. Run all tests
python -m pytest tests/document -v

# 2. Check coverage
python -m pytest tests/document --cov=src/vega/document --cov-report=term

# 3. Clean up
black src/vega/document tests/document
isort src/vega/document tests/document
mypy src/vega/document
```

### After Making Changes

```bash
# 1. Run affected tests
python -m pytest tests/document -k "legal or classification" -v

# 2. Check for regressions
python -m pytest tests/document -x -v

# 3. Update documentation
# (Edit relevant docs/ files)
```

---

## 🐛 Troubleshooting

### Test Collection Errors

```bash
# Show collection details
python -m pytest tests/document --collect-only

# Verbose collection
python -m pytest tests/document --collect-only -v

# Check for syntax errors
python -m py_compile tests/document/test_*.py
```

### Import Errors

```bash
# Verify Python path
python -c "import sys; print('\n'.join(sys.path))"

# Check if module is importable
python -c "from src.vega.document import base; print('✅ Import works')"

# Install in editable mode
pip install -e .
```

### Resource Issues

```bash
# Check for resource leaks
python -m pytest tests/document -v -W error::ResourceWarning

# Monitor memory usage
python -m pytest tests/document --memray

# Check file handles
lsof -p $(pgrep -f pytest)
```

### Performance Issues

```bash
# Profile test execution
python -m pytest tests/document --profile

# Show slowest tests
python -m pytest tests/document --durations=0

# Run with profiling
python -m cProfile -o profile.stats -m pytest tests/document
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
```

---

## 📝 Test Writing Templates

### Basic Test Template

```python
@pytest.mark.asyncio
async def test_feature_name(fixture_name):
    """Test description explaining what is being tested"""
    # Arrange
    input_data = "test data"
    expected_output = "expected result"
    
    # Act
    result = await processor.process(input_data)
    
    # Assert
    assert result.success is True
    assert result.data == expected_output
```

### Error Handling Test Template

```python
@pytest.mark.asyncio
async def test_error_handling(processor):
    """Test error handling for invalid input"""
    # Arrange
    context = create_test_context(content="", processing_mode="analysis")
    
    # Act
    result = await processor.process_document(context)
    
    # Assert
    assert result.success is False
    assert "error" in result.results
    assert "Empty content" in result.results["error"]
```

### Integration Test Template

```python
@pytest.mark.asyncio
async def test_integration_feature(ai_instance, sample_data):
    """Test integration of multiple components"""
    # Arrange
    context = ProcessingContext()
    
    # Act
    result = await ai_instance.process(sample_data, context)
    
    # Assert
    assert result.success is True
    assert "component_a_result" in result.data
    assert "component_b_result" in result.data
    # Verify integration
    assert result.data["component_a_result"] != ""
    assert result.data["component_b_result"] != ""
```

---

## 📚 Additional Resources

### Documentation

- `docs/TEST_IMPROVEMENTS_SUMMARY.md` - Technical implementation details
- `docs/TEST_SUITE_STATUS.md` - Current test status and metrics
- `docs/PROJECT_IMPROVEMENT_ROADMAP.md` - Future improvement plans
- `docs/SESSION_SUMMARY.md` - Session accomplishments
- `docs/VISUAL_SUMMARY.md` - Visual progress representation

### External Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [Coverage.py](https://coverage.readthedocs.io/)

---

## 🎯 Quick Stats Checker

Create this script as `check_tests.sh`:

```bash
#!/bin/bash
echo "🧪 Vega 2.0 Test Suite Status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Run tests and capture output
OUTPUT=$(python -m pytest tests/document -q --tb=no 2>&1)

# Extract results
PASSED=$(echo "$OUTPUT" | grep -oP '\d+(?= passed)' | head -1)
FAILED=$(echo "$OUTPUT" | grep -oP '\d+(?= failed)' | head -1)
WARNINGS=$(echo "$OUTPUT" | grep -oP '\d+(?= warning)' | head -1)

# Calculate percentage
TOTAL=$((PASSED + FAILED))
PERCENTAGE=$((PASSED * 100 / TOTAL))

# Display results
echo "Tests Passed:  $PASSED / $TOTAL ($PERCENTAGE%)"
echo "Tests Failed:  $FAILED"
echo "Warnings:      $WARNINGS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Status indicator
if [ $PERCENTAGE -ge 95 ]; then
    echo "✅ Excellent! Target achieved."
elif [ $PERCENTAGE -ge 80 ]; then
    echo "✅ Good progress!"
elif [ $PERCENTAGE -ge 60 ]; then
    echo "🔄 Making progress..."
else
    echo "⚠️  Needs attention"
fi
```

Make executable: `chmod +x check_tests.sh`  
Run: `./check_tests.sh`

---

**Remember**: Always run tests before committing! 🧪✨
