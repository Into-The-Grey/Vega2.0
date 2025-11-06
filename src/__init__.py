"""
Vega2.0 - AI Assistant and LLM Integration Platform
====================================================

A comprehensive AI assistant platform with local-first architecture,
featuring LLM integration, conversation logging, fine-tuning capabilities,
and extensive plugin ecosystem.

Main Components:
- vega.core: Core application logic and APIs
- vega.integrations: External service integrations
- vega.voice: Voice processing capabilities
- vega.intelligence: AI analysis and learning systems
- vega.datasets: Dataset management and preparation
- vega.training: Model training and fine-tuning
- vega.user: User profiling and personalization
"""

__version__ = "2.0.0"
__author__ = "Vega Development Team"
__email__ = "dev@vega-ai.com"
__license__ = "MIT"

# Lazy import main components to avoid loading heavy dependencies during test collection
import os

CORE_MODULES_AVAILABLE = False
OPTIONAL_MODULES_AVAILABLE = False

# Only import modules if not in test mode
if os.environ.get("VEGA_TEST_MODE") != "1":
    try:
        from .vega import core
        from .vega import integrations
        from .vega import datasets
        from .vega import training

        CORE_MODULES_AVAILABLE = True
    except ImportError:
        pass

    # Optional components
    try:
        from .vega import voice
        from .vega import intelligence
        from .vega import personality
        from .vega import user

        OPTIONAL_MODULES_AVAILABLE = True
    except ImportError:
        pass

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "CORE_MODULES_AVAILABLE",
    "OPTIONAL_MODULES_AVAILABLE",
]

if CORE_MODULES_AVAILABLE:
    __all__.extend(["core", "integrations", "datasets", "training"])

if OPTIONAL_MODULES_AVAILABLE:
    __all__.extend(["voice", "intelligence", "personality", "user"])
