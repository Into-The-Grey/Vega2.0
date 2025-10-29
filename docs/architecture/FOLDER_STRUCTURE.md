# Vega 2.0 Project Structure

## Directory Tree

```text
Vega2.0/
├── config/                     # Configuration files
│   ├── app.yaml               # Application settings
│   ├── llm.yaml               # Language model configuration
│   ├── ui.yaml                # User interface settings
│   ├── voice.yaml             # Voice processing configuration
│   ├── .env.example           # Environment variables template
│   └── .env.production        # Production environment settings
├── core/                      # Core application logic
│   ├── app.py                 # Main FastAPI application
│   └── cli.py                 # Command-line interface
├── data/                      # Data storage and state
│   ├── input_data/            # User-provided input files
│   ├── logs/                  # Application log files (organized by module)
│   ├── vega_logs/             # Legacy log files
│   └── vega_state/            # Application state and backups
├── datasets/                  # Multi-modal processing modules
│   ├── audio_fingerprint.py   # Audio fingerprinting & matching
│   ├── audio_utils.py         # Audio extraction utilities
│   ├── computer_vision.py     # Computer vision models
│   ├── document_processor.py  # Document format processing
│   ├── document_structure.py  # Document structure analysis
│   ├── image_analysis.py      # Image processing & analysis
│   ├── speech_to_text.py      # Speech transcription
│   ├── video_analysis.py      # Video content analysis
│   └── test_*.py              # Test files for each module
├── docs/                      # Documentation
│   ├── devnotes/              # Developer notes
│   ├── CONFIGURATION.md       # Configuration guide
│   ├── INTEGRATIONS.md        # Integration documentation
│   └── *.md                   # Various documentation files
├── examples/                  # Usage examples and demos
├── logs/                      # Structured logging (by component)
│   ├── analysis/              # Analysis module logs
│   ├── app/                   # Application logs
│   ├── autonomous/            # Autonomous system logs
│   ├── core/                  # Core system logs
│   ├── datasets/              # Dataset processing logs
│   ├── federated/             # Federated learning logs
│   ├── integrations/          # Integration logs
│   ├── intelligence/          # AI intelligence logs
│   ├── learning/              # Learning system logs
│   ├── training/              # Model training logs
│   ├── ui/                    # User interface logs
│   └── voice/                 # Voice processing logs
├── scripts/                   # Utility and automation scripts
│   ├── autonomous_master.py   # Autonomous system controller
│   ├── run_federated_tests.py # Federated learning test runner
│   ├── run_openapi_server.py  # OpenAPI server launcher
│   └── run_processes.py       # Process management
├── src/                       # Source code modules
│   └── vega/                  # Main application package
│       ├── core/              # Core functionality
│       ├── datasets/          # Dataset processing
│       ├── federated/         # Federated learning implementation
│       ├── integrations/      # External service integrations
│       ├── intelligence/      # AI intelligence systems
│       ├── learning/          # Learning algorithms
│       ├── personality/       # AI personality framework
│       ├── training/          # Model training infrastructure
│       ├── user/              # User management & profiling
│       └── voice/             # Voice processing
├── systemd/                   # System service configuration
│   └── vega.service           # SystemD service file
├── tests/                     # Test suite
│   ├── federated/             # Federated learning tests
│   ├── test_*.py              # Individual test modules
├── tools/                     # Development and utility tools
│   ├── analysis/              # Analysis utilities
│   ├── autonomous_debug/      # Debugging tools
│   ├── network/               # Network utilities
│   ├── sac/                   # SAC (Soft Actor-Critic) implementation
│   ├── test_suite/            # Test infrastructure
│   ├── ui/                    # UI development tools
│   ├── utils/                 # General utilities
│   ├── vega/                  # Vega-specific tools
│   └── vega_integrations/     # Integration tools
├── main.py                    # Main application entry point
├── requirements.txt           # Python dependencies
├── roadmap.md                 # Development roadmap
├── README.md                  # Project documentation
└── .env                       # Environment variables (local)
```

## Folders & Purpose

### Configuration Management

- **`config/`** - Centralized configuration files for all modules and environments
- **`.env`** - Local environment variables and secrets

### Core Application

- **`core/`** - Essential application logic including FastAPI app and CLI interface
- **`main.py`** - Primary entry point for running different application components
- **`src/vega/`** - Organized source code modules with clear separation of concerns

### Data Processing & Storage

- **`datasets/`** - Multi-modal data processing (audio, video, images, documents)
- **`data/`** - Runtime data, logs, and application state storage
- **`logs/`** - Structured logging organized by functional component

### Development & Operations

- **`scripts/`** - Automation scripts for testing, deployment, and process management
- **`tools/`** - Development utilities, debugging tools, and specialized components
- **`tests/`** - Comprehensive test suite covering all modules
- **`systemd/`** - System service configuration for production deployment

### Documentation & Examples

- **`docs/`** - Project documentation, guides, and technical notes
- **`examples/`** - Usage examples and demonstration code

## Key Configuration Files

### Environment & Settings

- **`config/app.yaml`** - Main application configuration
- **`config/llm.yaml`** - Language model settings and endpoints  
- **`config/ui.yaml`** - User interface and frontend configuration
- **`config/voice.yaml`** - Voice processing and audio settings
- **`config/.env.example`** - Template for environment variables
- **`requirements.txt`** - Python package dependencies

### Service & Deployment

- **`systemd/vega.service`** - SystemD service configuration for production
- **`main.py`** - Multi-command entry point (server, cli, test, etc.)

## Tests & Data

### Test Organization

- **`tests/`** - Main test directory with modular test files
- **`tests/federated/`** - Specialized federated learning tests
- **`datasets/test_*.py`** - Unit tests co-located with implementation
- **`scripts/run_federated_tests.py`** - Automated test runner

### Data Storage

- **`data/input_data/`** - User-provided files for processing
- **`data/vega_state/`** - Application state, backups, and voice samples
- **`data/logs/`** - Historical log files organized by date and component

### Sample Data & Examples

- **`examples/`** - Usage demonstrations and sample implementations
- **`tools/test_suite/`** - Test infrastructure and utilities

## Implementation Status

### ✅ Fully Implemented

- **Multi-modal Processing** - Audio, video, image, and document processing
- **Federated Learning Core** - Privacy-preserving distributed learning
- **Security Framework** - Authentication, audit logging, anomaly detection
- **Configuration Management** - YAML-based modular configuration

### ⚠️ Needs Attention

- **TODO: Fix 46 failing tests** - Test suite requires debugging and updates
- **TODO: Missing Docker configurations** - Containerization setup incomplete
- **TODO: Incomplete API documentation** - OpenAPI specs need completion

### 🔄 In Development

- **Real-time Collaboration** - WebSocket infrastructure planned
- **Analytics Dashboard** - Data collection and visualization system
- **Production Deployment** - Kubernetes manifests and monitoring setup

## Roadmap Integration

The complete development roadmap is documented in [`roadmap.md`](./roadmap.md) with:

- Detailed phase breakdowns for each major feature
- Implementation status and progress tracking
- Critical issues and immediate priorities
- Timeline estimates and success metrics

## Notes

### Architecture Patterns

- **Modular Design** - Clear separation between core, processing, and utility modules
- **Configuration-Driven** - YAML-based configuration with environment overrides
- **Test-Driven Development** - Comprehensive test coverage across all modules
- **Security-First** - Built-in authentication, logging, and validation

### Technology Stack

- **Backend:** Python, FastAPI, SQLAlchemy
- **ML/AI:** PyTorch, TensorFlow, HuggingFace Transformers
- **Processing:** OpenCV, PyPDF2, FFmpeg, Whisper
- **Infrastructure:** Docker, Kubernetes, SystemD

### Development Workflow

1. **Entry Point:** Use `main.py` with appropriate command (server, cli, test)
2. **Configuration:** Modify YAML files in `config/` directory
3. **Testing:** Run `python main.py test` or specific test suites
4. **Development:** Work in `src/vega/` with tests in `tests/`
5. **Deployment:** Use `systemd/vega.service` for production

---

*Last Updated: September 19, 2025*  
*Related: [Development Roadmap](./roadmap.md)*
