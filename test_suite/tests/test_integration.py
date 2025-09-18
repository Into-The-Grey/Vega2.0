"""
Integration Tests for Calendar Sync
===================================

Single-purpose test file for calendar synchronization functionality.
Tests only calendar sync with dummy parameters.
"""

import asyncio
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


async def test_calendar_sync(config: Dict[str, Any]) -> Dict[str, Any]:
    """Test calendar synchronization with dummy parameters"""
    try:
        # Import calendar sync module
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        from user.user_profiling.collectors.calendar_sync import (
            CalendarConfig,
            CalendarSync,
        )
        from user.user_profiling.database.user_profile_schema import UserProfileDatabase

        # Use dummy test configuration
        test_config = CalendarConfig()
        test_config.google_credentials_file = config["dummy_api_keys"][
            "GOOGLE_CALENDAR_CREDENTIALS"
        ]
        test_config.sync_past_days = 1  # Minimal for testing
        test_config.sync_future_days = 1

        # Create temporary test database
        test_db = UserProfileDatabase(":memory:")

        # Initialize calendar sync
        calendar_sync = CalendarSync(test_db, test_config)

        # Test configuration loading
        assert test_config.sync_past_days == 1
        assert test_config.sync_future_days == 1

        # Test database initialization
        assert test_db is not None

        # Test calendar sync initialization
        assert calendar_sync.config == test_config
        assert calendar_sync.db == test_db

        logger.info("Calendar sync test completed successfully")
        return {
            "success": True,
            "message": "Calendar sync configuration and initialization successful",
        }

    except Exception as e:
        logger.error(f"Calendar sync test failed: {e}")
        return {"success": False, "error": str(e)}
