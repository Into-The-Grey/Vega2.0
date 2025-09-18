"""
Core Component Tests
===================

Single-purpose test file for core component functionality.
Tests configuration loading with dummy parameters.
"""

import asyncio
from typing import Dict, Any
import logging
import os

logger = logging.getLogger(__name__)


async def test_config_loading(config: Dict[str, Any]) -> Dict[str, Any]:
    """Test configuration loading with dummy parameters"""
    try:
        # Test environment variable loading
        original_env = os.environ.copy()

        # Set dummy environment variables
        for key, value in config["dummy_api_keys"].items():
            os.environ[key] = value

        # Test basic config access
        test_value = os.getenv("GOOGLE_CALENDAR_CREDENTIALS")
        assert test_value == "test_credentials.json"

        # Test config validation
        required_keys = [
            "GOOGLE_CALENDAR_CREDENTIALS",
            "PLAID_CLIENT_ID",
            "PLAID_SECRET",
        ]
        for key in required_keys:
            assert os.getenv(key) is not None, f"Missing required config key: {key}"

        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)

        logger.info("Config loading test completed successfully")
        return {
            "success": True,
            "message": "Configuration loading successful with dummy parameters",
        }

    except Exception as e:
        logger.error(f"Config loading test failed: {e}")
        return {"success": False, "error": str(e)}


async def test_database_operations(config: Dict[str, Any]) -> Dict[str, Any]:
    """Test database operations with dummy data"""
    try:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        from user.user_profiling.database.user_profile_schema import UserProfileDatabase

        # Create in-memory test database
        test_db = UserProfileDatabase(":memory:")

        # Test database initialization
        assert test_db is not None

        # Test session creation
        session = test_db.get_session()
        assert session is not None
        session.close()

        logger.info("Database operations test completed successfully")
        return {"success": True, "message": "Database operations successful"}

    except Exception as e:
        logger.error(f"Database operations test failed: {e}")
        return {"success": False, "error": str(e)}
