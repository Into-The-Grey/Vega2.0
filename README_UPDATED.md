# Vega2.0 - Advanced Autonomous AI System

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Vega2.0 is a comprehensive, production-ready autonomous AI system featuring advanced security, error handling, background process management, and elliptic curve cryptography. Built with FastAPI and designed for localhost-first operation with enterprise-grade security.

## 🚀 Key Features

### 🧠 AI & LLM Integration

- **Multi-Provider Support**: Ollama, OpenAI, Anthropic with intelligent routing
- **Circuit Breaker Protection**: Automatic failover and recovery
- **Response Caching**: TTL-based caching for performance
- **Cost Tracking**: Usage monitoring and cost calculation

### 🔐 Advanced Security

- **Elliptic Curve Cryptography (ECC)**: Industry-standard encryption
- **Digital Signatures**: ECDSA for data integrity
- **Secure Key Management**: Hardware-backed key storage
- **API Authentication**: ECC-backed secure API keys
- **Message Encryption**: ECIES for end-to-end encryption

### 🛡️ Error Handling & Recovery

- **Structured Logging**: JSON-based logging with correlation IDs
- **Error Classification**: Categorized error codes and severity levels
- **Automatic Recovery**: Circuit breaker patterns and retry logic
- **User-Friendly Messages**: Technical errors translated to user messages
- **Comprehensive Monitoring**: Error statistics and health metrics

### ⚙️ Background Process Management

- **Process Lifecycle**: Start, stop, restart, monitor processes
- **Health Monitoring**: CPU, memory, and status tracking
- **Auto-Recovery**: Crashed process detection and restart
- **Resource Management**: Memory and CPU usage monitoring
- **System Integration**: Voice processing, integrations, monitoring

### 📊 API & Documentation

- **OpenAPI 3.0**: Comprehensive API documentation
- **Type Safety**: Pydantic models for all requests/responses
- **Versioning**: API version management
- **Interactive Docs**: Swagger UI and ReDoc integration
- **Request Validation**: Automatic input validation and sanitization

### 🧪 Testing & Quality

- **Dedicated Test Suite**: FastAPI-based testing interface
- **Comprehensive Coverage**: Core, integration, performance, security tests
- **Dummy Parameters**: Safe testing without real credentials
- **Automated Validation**: Error handling and recovery testing

## 📦 Installation

### Prerequisites

- Python 3.12 or higher
- Ollama (for local LLM)
- SQLite 3.35+ (included with Python)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/vega2.0.git
cd vega2.0

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up configuration
cp .env.example .env
# Edit .env with your settings

# Initialize the system
python -m core.app
```

### Production Deployment

```bash
# Install system service
sudo cp systemd/vega.service /etc/systemd/system/
sudo systemctl enable vega
sudo systemctl start vega

# Check status
sudo systemctl status vega
```

## 🔧 Configuration

### Environment Variables

```bash
# Core Configuration
HOST=127.0.0.1                    # API host (localhost-only for security)
PORT=8000                         # API port
API_KEY=your-secure-api-key       # Primary API key
API_KEYS_EXTRA=key1,key2,key3     # Additional API keys

# LLM Configuration
LLM_BACKEND=ollama                 # Primary LLM backend
MODEL_NAME=llama3.2:3b            # Default model
LLM_BASE_URL=http://127.0.0.1:11434  # Ollama base URL
LLM_TIMEOUT=30                     # Request timeout

# OpenAI Configuration (optional)
OPENAI_API_KEY=sk-...             # OpenAI API key
OPENAI_MODEL=gpt-4                # OpenAI model

# Anthropic Configuration (optional)
ANTHROPIC_API_KEY=sk-ant-...      # Anthropic API key
ANTHROPIC_MODEL=claude-3-sonnet   # Anthropic model

# Database Configuration
DATABASE_URL=sqlite:///./vega.db  # SQLite database path
DATABASE_POOL_SIZE=10             # Connection pool size

# Security Configuration
ECC_KEY_STORE=/secure/keys        # ECC key storage path
CERT_STORE=/secure/certs          # Certificate storage path
SESSION_SECRET=your-session-secret # Session encryption key

# Logging Configuration
LOG_LEVEL=INFO                     # Logging level
LOG_DIR=/var/log/vega             # Log directory
STRUCTURED_LOGS=true              # Enable structured logging

# Process Management
ENABLE_BACKGROUND_PROCESSES=true  # Enable background processes
PROCESS_MONITORING=true           # Enable process monitoring
AUTO_RESTART=true                 # Enable auto-restart
```

## 🚀 Usage

### API Server

```bash
# Start the main API server
uvicorn core.app:app --host 127.0.0.1 --port 8000

# Start with auto-reload (development)
uvicorn core.app:app --reload --host 127.0.0.1 --port 8000

# Start OpenAPI-compliant server
python run_openapi_server.py
```

### CLI Interface

```bash
# Chat with AI
python -m core.cli chat "Hello, how are you?"

# Interactive REPL
python -m core.cli repl

# View conversation history
python -m core.cli history --limit 10

# Build dataset
python -m core.cli dataset build ./datasets/samples

# Train model
python -m core.cli train --config training/config.yaml
```

### Background Processes

```bash
# Start background processes
python run_processes.py start

# Check process status
python run_processes.py status

# Stop background processes
python run_processes.py stop

# Restart specific process
python run_processes.py restart --process voice_processor
```

### Test Suite

```bash
# Run test suite UI
cd test_suite && uvicorn app:app --port 8001

# Run individual tests
python -m pytest tests/test_core.py
python -m pytest tests/test_integration.py
python -m pytest tests/test_security.py

# Run all tests
python -m pytest tests/
```

## 🔐 Security Features

### ECC Cryptography

```python
from core.ecc_crypto import get_ecc_manager, ECCCurve

# Generate key pair
ecc_manager = get_ecc_manager()
key_pair = ecc_manager.generate_key_pair(ECCCurve.SECP256R1)

# Sign data
signature = ecc_manager.sign_data("Hello World", key_pair.key_id)

# Verify signature
valid = ecc_manager.verify_signature("Hello World", signature)

# Encrypt message
encrypted = ecc_manager.encrypt_message("Secret", recipient_public_key)

# Decrypt message
decrypted = ecc_manager.decrypt_message(encrypted, recipient_key_id)
```

### Secure API Keys

```python
from core.api_security import get_security_manager

# Generate secure API key
security_manager = get_security_manager()
secure_key = security_manager.generate_secure_api_key(
    permissions=["read", "write"],
    expires_in_days=30,
    rate_limit=100
)

# Validate API key
validated = security_manager.validate_api_key(api_key)
```

### Certificate Management

```python
# Generate certificate
certificate = ecc_manager.generate_certificate(
    key_id=key_pair.key_id,
    subject_name="vega.local",
    validity_days=365
)

# Verify certificate
valid = ecc_manager.verify_certificate(certificate.certificate_pem)
```

## 🔍 Monitoring & Observability

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/healthz

# Readiness check
curl http://localhost:8000/readyz

# Liveness check
curl http://localhost:8000/livez

# Metrics
curl http://localhost:8000/metrics
```

### Error Monitoring

```bash
# Get error statistics
curl -H "X-API-Key: your-key" http://localhost:8000/admin/errors/stats

# Get recovery statistics
curl -H "X-API-Key: your-key" http://localhost:8000/admin/recovery/stats

# Process status
curl -H "X-API-Key: your-key" http://localhost:8000/admin/processes/status
```

### Logging

```bash
# View logs
tail -f /var/log/vega/errors.log
tail -f /var/log/vega/debug.log

# System logs (if using systemd)
journalctl -u vega -f
```

## 🛠️ Development

### Project Structure

```
vega2.0/
├── core/                    # Core system components
│   ├── app.py              # Main FastAPI application
│   ├── cli.py              # Command-line interface
│   ├── llm.py              # LLM integration layer
│   ├── db.py               # Database operations
│   ├── ecc_crypto.py       # ECC cryptography
│   ├── api_security.py     # API security layer
│   ├── error_handler.py    # Error handling system
│   ├── process_manager.py  # Background process management
│   ├── recovery_manager.py # Automatic recovery
│   └── openapi_app.py      # OpenAPI-compliant API
├── test_suite/             # Dedicated testing interface
├── integrations/           # External service integrations
├── datasets/               # Dataset preparation
├── training/               # Model training
├── docs/                   # Documentation
├── systemd/                # System service files
└── static/                 # Static web assets
```

### Adding New Features

1. **Create Feature Module**: Add new module in appropriate directory
2. **Add Tests**: Create tests in `test_suite/` or `tests/`
3. **Update Documentation**: Update relevant docs and API specs
4. **Add Configuration**: Add environment variables if needed
5. **Security Review**: Ensure security best practices

### Testing

```bash
# Run ECC system tests
python test_ecc_system.py

# Run error handling tests
python test_error_handling.py

# Run process manager tests
python test_process_manager.py

# Run full test suite
python -m pytest tests/ -v
```

## 📚 API Reference

### Core Endpoints

- `POST /chat` - Chat with AI
- `GET /history` - Get conversation history
- `POST /feedback` - Submit feedback
- `GET /healthz` - Health check

### Admin Endpoints

- `GET /admin/processes/status` - Background process status
- `POST /admin/processes/start` - Start background processes
- `POST /admin/processes/stop` - Stop background processes
- `GET /admin/errors/stats` - Error statistics
- `GET /admin/recovery/stats` - Recovery statistics

### ECC Endpoints

- `POST /admin/ecc/generate-key` - Generate ECC key pair
- `GET /admin/ecc/keys` - List ECC keys
- `POST /admin/ecc/sign` - Sign data with ECC
- `POST /admin/ecc/verify` - Verify ECC signature

### Security Endpoints

- `POST /admin/security/generate-api-key` - Generate secure API key
- `GET /admin/security/api-keys` - List API keys

Full API documentation available at `/docs` when server is running.

## 🔧 Troubleshooting

### Common Issues

**Ollama Connection Failed**

```bash
# Check if Ollama is running
curl http://127.0.0.1:11434/api/tags

# Start Ollama
ollama serve

# Check available models
ollama list
```

**Database Errors**

```bash
# Check database file permissions
ls -la vega.db*

# Reset database (development only)
rm vega.db* && python -c "from core.db import init_database; init_database()"
```

**Process Management Issues**

```bash
# Check process status
python run_processes.py status

# Restart all processes
python run_processes.py stop && python run_processes.py start

# Check logs
tail -f /tmp/vega_processes.log
```

**ECC Key Issues**

```bash
# List ECC keys
python test_ecc_system.py

# Check key permissions
ls -la /tmp/vega_keys/

# Regenerate keys (development only)
rm -rf /tmp/vega_keys/ && python test_ecc_system.py
```

### Performance Tuning

- **Memory Usage**: Monitor with `/admin/processes/status`
- **Database**: Use WAL mode for better concurrency
- **Caching**: Configure TTL cache sizes
- **Rate Limiting**: Adjust per-key rate limits

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass (`python -m pytest`)
5. Update documentation
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open Pull Request

### Code Standards

- Python 3.12+ with type hints
- FastAPI best practices
- Comprehensive error handling
- Security-first design
- Full test coverage

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Ollama](https://ollama.ai/) - Local LLM runtime
- [Cryptography](https://cryptography.io/) - Cryptographic library
- [StructLog](https://www.structlog.org/) - Structured logging
- [Pydantic](https://pydantic-docs.helpmanual.io/) - Data validation

## 📞 Support

- Documentation: [docs/](docs/)
- Issues: [GitHub Issues](https://github.com/your-org/vega2.0/issues)
- Discussions: [GitHub Discussions](https://github.com/your-org/vega2.0/discussions)

---

**Vega2.0** - Built with ❤️ for the future of autonomous AI systems.
