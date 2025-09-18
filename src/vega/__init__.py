"""
Vega2.0 Core Application Package
================================

This package contains the core Vega2.0 application components including
the main API server, CLI interface, LLM integration, database management,
and all essential functionality.

Modules:
--------
- core: Core application logic, APIs, and services
- integrations: External service integrations
- datasets: Dataset management and preparation
- training: Model training and fine-tuning
- learning: Learning and evaluation systems
- voice: Voice processing capabilities
- intelligence: AI analysis and intelligence systems
- personality: Personality and behavior modeling
- user: User profiling and personalization

The core module contains the main FastAPI application, CLI interface,
configuration management, database operations, LLM backends, security,
error handling, and process management.
"""

# Core components - always available
from . import core

# Data and ML components
try:
    from . import datasets
    from . import training
    from . import learning

    DATA_ML_AVAILABLE = True
except ImportError:
    DATA_ML_AVAILABLE = False

# Integration components
try:
    from . import integrations

    INTEGRATIONS_AVAILABLE = True
except ImportError:
    INTEGRATIONS_AVAILABLE = False

# Advanced components
try:
    from . import voice
    from . import intelligence
    from . import personality
    from . import user

    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

__all__ = ["core", "DATA_ML_AVAILABLE", "INTEGRATIONS_AVAILABLE", "ADVANCED_AVAILABLE"]

if DATA_ML_AVAILABLE:
    __all__.extend(["datasets", "training", "learning"])

if INTEGRATIONS_AVAILABLE:
    __all__.append("integrations")

if ADVANCED_AVAILABLE:
    __all__.extend(["voice", "intelligence", "personality", "user"])
