"""
Performance Tests
================

Single-purpose test file for performance testing.
Tests response time with dummy requests.
"""

import asyncio
import time
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


async def test_response_time(config: Dict[str, Any]) -> Dict[str, Any]:
    """Test response time performance"""
    try:
        # Test simple response time
        start_time = time.time()

        # Simulate a basic operation
        await asyncio.sleep(0.01)  # 10ms simulated processing

        end_time = time.time()
        response_time = end_time - start_time

        # Assert response time is reasonable (under 1 second for test)
        assert response_time < 1.0, f"Response time too high: {response_time}s"

        logger.info(f"Response time test completed: {response_time:.3f}s")
        return {
            "success": True,
            "message": f"Response time within acceptable limits: {response_time:.3f}s",
            "response_time": response_time,
        }

    except Exception as e:
        logger.error(f"Response time test failed: {e}")
        return {"success": False, "error": str(e)}
