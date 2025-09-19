#!/usr/bin/env python3
"""
Simple test script for audit logging functionality.
"""

import json
import time
import sys
import os


def audit_log(
    event: str, details: dict, participant_id: str = None, session_id: str = None
):
    """
    Structured audit logging for federated operations.

    Args:
        event: Type of event (e.g., 'message_sent', 'auth_failed', 'model_update')
        details: Event-specific details
        participant_id: ID of participant involved
        session_id: Federated session ID
    """
    audit_entry = {
        "timestamp": time.time(),
        "event": event,
        "participant_id": participant_id,
        "session_id": session_id,
        "details": details,
    }

    # Log as structured JSON for easier parsing
    print(f"AUDIT: {json.dumps(audit_entry, indent=2)}")

    # Also log to a dedicated audit file if configured
    try:
        audit_file = "./audit.log"
        with open(audit_file, "a") as f:
            f.write(f"{json.dumps(audit_entry)}\n")
    except Exception as e:
        print(f"Could not write to audit file: {e}")


def test_audit_logging():
    """Test basic audit logging functionality."""
    print("=== Testing Audit Logging ===\n")

    # Test message send audit
    audit_log(
        "message_send_attempt",
        {
            "recipient_url": "http://127.0.0.1:8001",
            "message_type": "model_update",
            "message_id": "msg_12345",
            "encrypted": True,
            "has_api_key": True,
        },
        participant_id="coordinator_1",
        session_id="session_abc123",
    )

    print("\n" + "=" * 50 + "\n")

    # Test authentication failure audit
    audit_log(
        "auth_failure",
        {
            "operation": "send_to_participant",
            "recipient_id": "participant_2",
            "message_type": "model_request",
            "reason": "invalid_api_key",
        },
        participant_id="coordinator_1",
        session_id="session_abc123",
    )

    print("\n" + "=" * 50 + "\n")

    # Test participant registration audit
    audit_log(
        "participant_registered",
        {
            "participant_id": "participant_3",
            "host": "192.168.1.100",
            "port": 8002,
            "name": "Alice's Laptop",
            "capabilities": {
                "model_types": ["pytorch"],
                "encryption": True,
                "version": "1.0.0",
            },
            "has_api_key": True,
        },
    )

    print("\n" + "=" * 50 + "\n")

    # Test broadcast audit
    audit_log(
        "broadcast_complete",
        {
            "operation": "broadcast_to_all",
            "message_type": "global_model_update",
            "message_id": "broadcast_67890",
            "total_participants": 3,
            "successful_count": 2,
            "failed_count": 1,
            "successful_participants": ["participant_1", "participant_2"],
            "failed_participants": [
                {"participant_id": "participant_3", "error": "Connection timeout"}
            ],
        },
        participant_id="coordinator_1",
        session_id="session_abc123",
    )

    print("\n=== Audit Logging Test Complete ===")


def check_audit_file():
    """Check the created audit log file."""
    audit_file = "./audit.log"
    if os.path.exists(audit_file):
        print(f"\n✓ Audit log file created: {audit_file}")

        with open(audit_file, "r") as f:
            lines = f.readlines()

        print(f"✓ Contains {len(lines)} audit entries")
        print("\nLast entry in file:")
        if lines:
            last_entry = json.loads(lines[-1])
            print(json.dumps(last_entry, indent=2))
    else:
        print("\n⚠ No audit log file created")


if __name__ == "__main__":
    test_audit_logging()
    check_audit_file()
