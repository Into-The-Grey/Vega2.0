# Vega2.0 Directory Structure

This document describes the organization of the Vega2.0 project.

## Root Files

- **`main.py`** - Main entry point for the application (CLI, server, API)
- **`README.md`** - Project overview and documentation
- **`requirements.txt`** - Python dependencies
- **`pytest.ini`** - Test configuration
- **`.env`** - Environment variables (not in git)
- **`__init__.py`** - Python package marker

## Directory Structure

### 📦 Core Application

- **`src/`** - Source code for Vega
  - `vega/core/` - Core functionality (LLM, API, database)
  - `vega/federated/` - Federated learning implementation
  - `vega/intelligence/` - AI intelligence systems
  - `vega/integrations/` - External service integrations
  - `vega/collaboration/` - Collaboration features

### 🔧 Configuration

- **`config/`** - Configuration files
  - `app.yaml` - Application settings
  - `llm.yaml` - LLM provider settings
  - `harm_filter.yaml` - Ethics/safety configuration
  - `training.yaml` - Training parameters
  - `participants.yaml` - Federated learning nodes

### 📚 Documentation

- **`docs/`** - All project documentation
  - `features/` - Feature documentation
  - `getting-started/` - Quick start guides
  - `development/` - Development guides
  - `performance/` - Performance tuning
  - `api/` - API documentation
  - `architecture/` - System architecture

### 🧪 Testing

- **`tests/`** - Unit tests (pytest)
- **`test_scripts/`** - Integration/system test scripts
  - Ethics tests, performance tests, functional tests
- **`test_results/`** - Test outputs and logs
- **`benchmarks/`** - Performance benchmarking tools

### 🛠️ Tools & Scripts

- **`scripts/`** - Utility scripts
  - `vega.sh` - Vega management script
- **`tools/`** - Development tools
- **`demos/`** - Demo scripts
- **`examples/`** - Example code

### 📊 Data & Models

- **`data/`** - Application data
- **`datasets/`** - Dataset processing modules
- **`models/`** - Machine learning models
- **`uploads/`** - User uploaded files
- **`vega_state/`** - Application state

### 📝 Logs

- **`logs/`** - Application logs
  - Organized by component (core, federated, intelligence, etc.)

### 🚀 Deployment

- **`systemd/`** - System service files
- **`k8s/`** - Kubernetes deployment configs
- **`monitoring/`** - Monitoring configurations
- **`observability/`** - Observability tools
- **`scaling/`** - Auto-scaling configs
- **`security/`** - Security configurations
- **`alerting/`** - Alert rules and configs

### 🎨 User Interface

- **`ui/`** - Web UI components
- **`book/`** - mdBook documentation site

## Quick Navigation

### I want to

**Run the application:**

```bash
python3 main.py server --host 127.0.0.1 --port 8000
```

**Run tests:**

```bash
# Unit tests
pytest tests/

# System tests
python3 test_scripts/censorship_test.py
python3 test_scripts/test_code_quality.py
```

**Configure the system:**

```bash
# Edit configuration
nano config/harm_filter.yaml
nano config/llm.yaml
```

**Read documentation:**

```bash
# Quick reference
cat docs/getting-started/quick-reference.md

# Ethics and safety
cat docs/features/ethics-and-safety.md
```

**Check logs:**

```bash
tail -f logs/core/app.log
tail -f logs/intelligence/analysis.log
```

## File Organization Principles

1. **Source code** goes in `src/`
2. **Tests** are separated by type (unit vs integration)
3. **Documentation** is centralized in `docs/`
4. **Configuration** is in `config/` (YAML files)
5. **Results and outputs** go in appropriate result directories
6. **Deployment configs** are grouped by platform/tool

## Maintenance

To keep the directory clean:

```bash
# Clean old test results (older than 7 days)
find test_results/ -mtime +7 -delete

# Clean old logs (older than 30 days)
find logs/ -name "*.log" -mtime +30 -delete

# Remove Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
find . -name "*.pyc" -delete
```
