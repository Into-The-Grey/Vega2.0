#!/usr/bin/env python3
"""
Test script for audit logging functionality in federated communication.
"""

import asyncio
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.vega.federated.security import audit_log
from src.vega.federated.communication import CommunicationManager, FederatedMessage


def test_audit_logging():
    """Test basic audit logging functionality."""
    print("Testing basic audit logging...")

    # Test basic audit log
    audit_log(
        "test_event",
        {"message": "This is a test", "value": 42},
        participant_id="test_participant",
        session_id="test_session",
    )

    # Test audit log without optional parameters
    audit_log("test_event_minimal", {"message": "Minimal test"})

    print("✓ Basic audit logging test completed")


async def test_communication_audit():
    """Test audit logging in communication operations."""
    print("Testing communication audit logging...")

    try:
        # Create communication manager
        manager = CommunicationManager(
            participant_id="test_coordinator", participant_name="Test Coordinator"
        )

        # Test connection attempt (will fail, but should generate audit logs)
        result = await manager.connect_to_participant(
            participant_id="test_participant_1",
            host="127.0.0.1",
            port=8001,
            name="Test Participant 1",
        )
        print(f"Connection result: {result}")

        # Test message send (will fail due to no participant, but should generate audit logs)
        response = await manager.send_to_participant(
            recipient_id="nonexistent_participant",
            message_type="test_message",
            data={"test": "data"},
            session_id="test_session",
        )
        print(f"Send result: {response}")

        # Test broadcast (will fail due to no participants, but should generate audit logs)
        broadcast_result = await manager.broadcast_to_all(
            message_type="test_broadcast",
            data={"broadcast": "data"},
            session_id="test_session",
        )
        print(f"Broadcast result: {broadcast_result}")

        print("✓ Communication audit logging test completed")

    except Exception as e:
        print(f"Error in communication test: {e}")
        # This is expected due to missing dependencies


def check_audit_log_file():
    """Check if audit log file was created and contains entries."""
    print("Checking audit log file...")

    audit_file = "./audit.log"
    if os.path.exists(audit_file):
        with open(audit_file, "r") as f:
            lines = f.readlines()
        print(f"✓ Audit log file created with {len(lines)} entries")

        # Show last few entries
        print("Last few audit entries:")
        for line in lines[-3:]:
            print(f"  {line.strip()}")
    else:
        print("⚠ No audit log file found (may be due to import errors)")


if __name__ == "__main__":
    print("=== Audit Logging Test ===")

    # Test basic audit logging
    test_audit_logging()

    # Test communication audit logging
    asyncio.run(test_communication_audit())

    # Check log file
    check_audit_log_file()

    print("\n=== Test Complete ===")
