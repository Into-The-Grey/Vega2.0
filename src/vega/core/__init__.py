"""Core application components for Vega2.0"""

import os

# Lazy import to avoid loading app and heavy dependencies during tests
if os.environ.get("VEGA_TEST_MODE") != "1":
    # Core modules
    from . import app
    from . import cli
    from . import config
    from . import db
    from . import llm
    from . import resilience
    from . import security

    # Error handling and recovery
    from . import error_handler
    from . import error_middleware
    from . import exceptions
    from . import recovery_manager

    # Process management
    from . import process_manager

# Cryptography and security
try:
    from . import ecc_crypto
    from . import api_security

    ECC_AVAILABLE = True
except ImportError:
    ECC_AVAILABLE = False

# Optional modules
try:
    from . import memory
    from . import config_manager
    from . import logging_setup
    from . import interaction_log

    OPTIONAL_MODULES_AVAILABLE = True
except ImportError:
    OPTIONAL_MODULES_AVAILABLE = False

__all__ = [
    "app",
    "cli",
    "config",
    "db",
    "llm",
    "resilience",
    "security",
    "error_handler",
    "error_middleware",
    "exceptions",
    "recovery_manager",
    "process_manager",
    "ECC_AVAILABLE",
    "OPTIONAL_MODULES_AVAILABLE",
]

if ECC_AVAILABLE:
    __all__.extend(["ecc_crypto", "api_security"])
