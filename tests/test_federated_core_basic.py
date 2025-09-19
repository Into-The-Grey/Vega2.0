"""
Simple unit tests for Vega 2.0 federated learning core modules.
Basic tests to verify core functionality works correctly.
"""

import pytest
import sys
import os

# Add the src directory to the path
src_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, src_path)


def test_security_module_import():
    """Test that security module can be imported."""
    try:
        from vega.federated.security import check_api_key, audit_log

        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import security module: {e}")


def test_api_key_validation():
    """Test basic API key validation."""
    from vega.federated.security import check_api_key

    allowed_keys = {"valid_key_1", "valid_key_2"}

    # Test valid key
    assert check_api_key("valid_key_1", allowed_keys) is True

    # Test invalid key
    assert check_api_key("invalid_key", allowed_keys) is False

    # Test empty key
    assert check_api_key("", allowed_keys) is False


def test_anomaly_detection():
    """Test basic anomaly detection."""
    from vega.federated.security import is_anomalous_update

    # Test normal data (flattened format)
    normal_data = {
        "layer1_weight_0": 1.0,
        "layer1_weight_1": 2.0,
        "layer2_weight_0": 0.5,
    }

    result = is_anomalous_update(
        normal_data,
        threshold=10.0,
        participant_id="participant_1",
        session_id="session_1",
    )
    assert result["participant_id"] == "participant_1"
    assert result["session_id"] == "session_1"
    # Note: This might still trigger "unexpected_structure" but shouldn't trigger "large_values"

    # Test anomalous data (large values)
    anomalous_data = {
        "layer1_weight_0": 100.0,  # Large value exceeding threshold
        "layer1_weight_1": 2.0,
    }

    result = is_anomalous_update(
        anomalous_data,
        threshold=10.0,
        participant_id="participant_1",
        session_id="session_1",
    )
    assert result["is_anomalous"] is True
    assert "large_values" in result["anomaly_types"]


def test_model_signature():
    """Test model signature creation and verification."""
    from vega.federated.security import create_model_signature, verify_model_signature
    import json

    model_data = {"weights": [1.0, 2.0, 3.0]}
    model_bytes = json.dumps(model_data, sort_keys=True).encode("utf-8")
    secret_key = "test_secret_key"

    # Create signature
    signature = create_model_signature(model_bytes, secret_key)
    assert isinstance(signature, bytes)

    # Verify valid signature
    result = verify_model_signature(model_bytes, signature, secret_key)
    assert result["is_valid"] is True

    # Verify invalid signature
    invalid_signature = b"invalid_signature_data_here"
    result = verify_model_signature(model_bytes, invalid_signature, secret_key)
    assert result["is_valid"] is False


if __name__ == "__main__":
    # Run tests manually if needed
    print("Running basic federated core tests...")

    test_security_module_import()
    print("✓ Security module import test passed")

    test_api_key_validation()
    print("✓ API key validation test passed")

    test_anomaly_detection()
    print("✓ Anomaly detection test passed")

    test_model_signature()
    print("✓ Model signature test passed")

    print("All basic tests passed!")
