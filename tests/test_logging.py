"""
test_logging.py - Tests for VegaLogger infrastructure

Tests comprehensive logging functionality:
- Module-specific logger creation
- Rotating file handlers
- JSON structured logging
- Console output
- Log file tailing
- Thread safety
"""

import pytest
import tempfile
import shutil
import os
import json
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from tests.logging_setup import VegaLogger


class TestVegaLogger:
    """Test VegaLogger functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.original_log_dir = os.environ.get("VEGA_LOG_DIR")
        os.environ["VEGA_LOG_DIR"] = self.test_dir

        # Reset VegaLogger state
        VegaLogger._loggers = {}
        VegaLogger._setup_done = set()

    def teardown_method(self):
        """Cleanup test environment"""
        # Restore original log directory
        if self.original_log_dir:
            os.environ["VEGA_LOG_DIR"] = self.original_log_dir
        elif "VEGA_LOG_DIR" in os.environ:
            del os.environ["VEGA_LOG_DIR"]

        # Clean up test directory
        shutil.rmtree(self.test_dir, ignore_errors=True)

        # Reset VegaLogger state
        VegaLogger._loggers = {}
        VegaLogger._setup_done = set()

    def test_get_logger_creates_module_logger(self):
        """Test that get_logger creates a module-specific logger"""
        logger = VegaLogger.get_logger("test_module")

        assert logger.name == "vega.test_module"
        assert "test_module" in VegaLogger._loggers

    def test_get_logger_singleton_behavior(self):
        """Test that get_logger returns the same logger for the same module"""
        logger1 = VegaLogger.get_logger("test_module")
        logger2 = VegaLogger.get_logger("test_module")

        assert logger1 is logger2

    def test_log_directory_creation(self):
        """Test that log directories are created automatically"""
        VegaLogger.get_logger("test_module")

        module_log_dir = Path(self.test_dir) / "test_module"
        assert module_log_dir.exists()
        assert module_log_dir.is_dir()

    def test_log_file_creation(self):
        """Test that log files are created for modules"""
        logger = VegaLogger.get_logger("test_module")
        logger.info("Test message")

        log_file = Path(self.test_dir) / "test_module" / "test_module.log"
        assert log_file.exists()

    def test_json_log_format(self):
        """Test that logs are written in JSON format"""
        logger = VegaLogger.get_logger("test_module")
        logger.info("Test message", extra={"custom_field": "custom_value"})

        log_file = Path(self.test_dir) / "test_module" / "test_module.log"

        # Wait a moment for log to be written
        time.sleep(0.1)

        with open(log_file, "r") as f:
            log_line = f.readline().strip()
            log_data = json.loads(log_line)

            assert log_data["message"] == "Test message"
            assert log_data["level"] == "INFO"
            assert log_data["module"] == "test_module"
            assert log_data["custom_field"] == "custom_value"
            assert "timestamp" in log_data

    def test_different_log_levels(self):
        """Test that different log levels work correctly"""
        logger = VegaLogger.get_logger("test_module")

        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

        log_file = Path(self.test_dir) / "test_module" / "test_module.log"

        # Wait for logs to be written
        time.sleep(0.1)

        with open(log_file, "r") as f:
            lines = f.readlines()

        # Debug messages might not appear depending on log level
        assert len(lines) >= 4  # info, warning, error, critical

    def test_list_modules(self):
        """Test listing available log modules"""
        # Create some test loggers
        VegaLogger.get_logger("module1")
        VegaLogger.get_logger("module2")
        VegaLogger.get_logger("module3")

        modules = VegaLogger.list_modules()

        assert "module1" in modules
        assert "module2" in modules
        assert "module3" in modules

    def test_tail_log(self):
        """Test log file tailing functionality"""
        logger = VegaLogger.get_logger("test_module")

        # Write some test log entries
        for i in range(10):
            logger.info(f"Log message {i}")

        # Wait for logs to be written
        time.sleep(0.1)

        # Test tailing
        lines = VegaLogger.tail_log("test_module", 5)

        assert len(lines) == 5
        assert "Log message 9" in lines[-1]  # Last message
        assert "Log message 5" in lines[0]  # First of last 5

    def test_tail_log_nonexistent_module(self):
        """Test tailing logs for non-existent module"""
        with pytest.raises(FileNotFoundError):
            VegaLogger.tail_log("nonexistent_module")

    def test_thread_safety(self):
        """Test that logging is thread-safe"""
        logger = VegaLogger.get_logger("thread_test")

        def write_logs(thread_id):
            for i in range(100):
                logger.info(f"Thread {thread_id} - Message {i}")

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=write_logs, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check that all messages were written
        log_file = Path(self.test_dir) / "thread_test" / "thread_test.log"

        time.sleep(0.2)  # Wait for all logs to be written

        with open(log_file, "r") as f:
            lines = f.readlines()

        # Should have 500 log entries (5 threads * 100 messages)
        assert len(lines) == 500

    def test_rotating_file_handler(self):
        """Test that log rotation works correctly"""
        logger = VegaLogger.get_logger("rotation_test")

        # Write lots of log entries to trigger rotation
        large_message = "x" * 1000  # 1KB message
        for i in range(2000):  # Write ~2MB of logs
            logger.info(f"Large message {i}: {large_message}")

        time.sleep(0.2)  # Wait for logs to be written

        log_dir = Path(self.test_dir) / "rotation_test"
        log_files = list(log_dir.glob("*.log*"))

        # Should have multiple log files due to rotation
        assert len(log_files) > 1

    def test_convenience_functions(self):
        """Test convenience logging functions"""
        from src.vega.core.logging_setup import (
            log_info,
            log_error,
            log_warning,
            log_debug,
        )

        # These should not raise exceptions
        log_info("test_module", "Info message")
        log_error("test_module", "Error message")
        log_warning("test_module", "Warning message")
        log_debug("test_module", "Debug message")

        log_file = Path(self.test_dir) / "test_module" / "test_module.log"
        assert log_file.exists()

    def test_structured_logging_with_extra_fields(self):
        """Test structured logging with additional fields"""
        logger = VegaLogger.get_logger("structured_test")

        logger.info(
            "User action",
            extra={
                "user_id": "user123",
                "action": "login",
                "ip_address": "192.168.1.1",
                "duration_ms": 150,
            },
        )

        log_file = Path(self.test_dir) / "structured_test" / "structured_test.log"

        time.sleep(0.1)

        with open(log_file, "r") as f:
            log_line = f.readline().strip()
            log_data = json.loads(log_line)

            assert log_data["user_id"] == "user123"
            assert log_data["action"] == "login"
            assert log_data["ip_address"] == "192.168.1.1"
            assert log_data["duration_ms"] == 150

    def test_error_logging_with_exception(self):
        """Test logging errors with exception information"""
        logger = VegaLogger.get_logger("error_test")

        try:
            raise ValueError("Test exception")
        except ValueError as e:
            logger.error("An error occurred", exc_info=True)

        log_file = Path(self.test_dir) / "error_test" / "error_test.log"

        time.sleep(0.1)

        with open(log_file, "r") as f:
            log_line = f.readline().strip()
            log_data = json.loads(log_line)

            assert "exception" in log_data
            assert "ValueError" in log_data["exception"]
            assert "Test exception" in log_data["exception"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
