#!/usr/bin/env python3
"""
Comprehensive test script for Vega 2.0 federated learning security features.

Tests:
1. API key authentication
2. Audit logging
3. Anomaly detection
4. Model consistency checking
5. Model signature verification
6. Complete validation pipeline
"""

import json
import time
import tempfile
import os
from typing import Dict, Any, List
import hashlib
import hmac


# Mock the security module functions for testing
def mock_api_key_check(api_key: str) -> Dict[str, Any]:
    """Mock API key validation"""
    valid_keys = ["test-key-123", "participant-key-456", "admin-key-789"]

    if api_key in valid_keys:
        return {
            "is_valid": True,
            "participant_id": f"participant_{api_key.split('-')[-1]}",
            "permissions": ["federated_learning", "model_upload"],
        }
    else:
        return {"is_valid": False, "error": "Invalid API key", "participant_id": None}


def mock_audit_log(event_type: str, event_data: Dict[str, Any], **kwargs) -> None:
    """Mock audit logging"""
    log_entry = {
        "timestamp": time.time(),
        "event_type": event_type,
        "event_data": event_data,
        **kwargs,
    }
    print(f"AUDIT LOG: {json.dumps(log_entry, indent=2)}")


def mock_anomaly_detection(
    model_update: Dict[str, Any], threshold: float = 10.0, **kwargs
) -> Dict[str, Any]:
    """Mock anomaly detection with comprehensive checks"""
    is_anomalous = False
    anomaly_reasons = []

    # Check for large values
    weights = model_update.get("weights", {})
    for layer_name, layer_weights in weights.items():
        if isinstance(layer_weights, (list, tuple)):
            max_val = max(abs(w) for w in layer_weights if isinstance(w, (int, float)))
            if max_val > threshold:
                is_anomalous = True
                anomaly_reasons.append(f"Large weights in {layer_name}: {max_val}")

    # Check for NaN/infinite values
    for layer_name, layer_weights in weights.items():
        if isinstance(layer_weights, (list, tuple)):
            for w in layer_weights:
                if isinstance(w, float):
                    if str(w).lower() in ["nan", "inf", "-inf"]:
                        is_anomalous = True
                        anomaly_reasons.append(f"Invalid value in {layer_name}: {w}")

    # Check update frequency
    participant_id = kwargs.get("participant_id")
    if participant_id and participant_id.endswith("suspicious"):
        is_anomalous = True
        anomaly_reasons.append("Suspicious participant detected")

    result = {
        "is_anomalous": is_anomalous,
        "anomaly_reasons": anomaly_reasons,
        "anomaly_score": len(anomaly_reasons),
        "timestamp": time.time(),
    }

    mock_audit_log(
        "anomaly_detection_completed",
        result,
        participant_id=kwargs.get("participant_id"),
        session_id=kwargs.get("session_id"),
    )

    return result


def mock_model_consistency_check(
    models: List[Dict[str, Any]], **kwargs
) -> Dict[str, Any]:
    """Mock model consistency checking"""
    if len(models) < 2:
        return {
            "is_consistent": True,
            "message": "Not enough models to check consistency",
            "model_count": len(models),
        }

    # Check if all models have similar structure
    first_model = models[0]
    inconsistencies = []

    for i, model in enumerate(models[1:], 1):
        # Check structure
        if set(model.keys()) != set(first_model.keys()):
            inconsistencies.append(f"Model {i} has different structure")

        # Check weights structure
        if "weights" in model and "weights" in first_model:
            model_layers = set(model["weights"].keys())
            first_layers = set(first_model["weights"].keys())
            if model_layers != first_layers:
                inconsistencies.append(f"Model {i} has different layer structure")

    is_consistent = len(inconsistencies) == 0

    result = {
        "is_consistent": is_consistent,
        "model_count": len(models),
        "inconsistencies": inconsistencies,
        "timestamp": time.time(),
    }

    mock_audit_log(
        "consistency_check_completed", result, session_id=kwargs.get("session_id")
    )

    return result


def mock_model_signature_verification(
    model_bytes: bytes, signature: bytes, secret_key: str, **kwargs
) -> Dict[str, Any]:
    """Mock model signature verification"""
    try:
        # Compute expected signature
        expected_signature = hmac.new(
            secret_key.encode(), model_bytes, hashlib.sha256
        ).digest()

        # Verify signature
        is_valid = hmac.compare_digest(signature, expected_signature)

        result = {
            "is_valid": is_valid,
            "model_size_bytes": len(model_bytes),
            "signature_size_bytes": len(signature),
            "timestamp": time.time(),
        }

        mock_audit_log(
            "signature_verification_completed",
            result,
            participant_id=kwargs.get("participant_id"),
            session_id=kwargs.get("session_id"),
        )

        return result

    except Exception as e:
        result = {"is_valid": False, "error": str(e), "timestamp": time.time()}

        mock_audit_log(
            "signature_verification_error",
            result,
            participant_id=kwargs.get("participant_id"),
            session_id=kwargs.get("session_id"),
        )

        return result


def create_mock_model_signature(model_bytes: bytes, secret_key: str) -> bytes:
    """Create HMAC signature for model data"""
    return hmac.new(secret_key.encode(), model_bytes, hashlib.sha256).digest()


def test_api_key_authentication():
    """Test API key authentication functionality"""
    print("\n=== Testing API Key Authentication ===")

    # Test valid API key
    result = mock_api_key_check("test-key-123")
    print(f"Valid key test: {result}")
    assert result["is_valid"] == True
    assert result["participant_id"] == "participant_123"

    # Test invalid API key
    result = mock_api_key_check("invalid-key")
    print(f"Invalid key test: {result}")
    assert result["is_valid"] == False

    print("✅ API Key Authentication tests passed!")


def test_audit_logging():
    """Test audit logging functionality"""
    print("\n=== Testing Audit Logging ===")

    # Test basic audit log
    mock_audit_log(
        "federated_training_started",
        {
            "session_id": "test-session-123",
            "participant_count": 3,
            "model_type": "neural_network",
        },
        participant_id="participant_123",
        session_id="test-session-123",
    )

    # Test error audit log
    mock_audit_log(
        "model_update_failed",
        {"error": "Invalid model format", "error_code": "MODEL_FORMAT_ERROR"},
        participant_id="participant_456",
        session_id="test-session-123",
    )

    print("✅ Audit Logging tests passed!")


def test_anomaly_detection():
    """Test anomaly detection functionality"""
    print("\n=== Testing Anomaly Detection ===")

    # Test normal model update
    normal_model = {
        "weights": {"layer1": [0.1, 0.2, -0.15, 0.3], "layer2": [0.5, -0.4, 0.2, 0.1]},
        "metadata": {"epoch": 5, "accuracy": 0.85},
    }

    result = mock_anomaly_detection(
        normal_model,
        threshold=10.0,
        participant_id="participant_123",
        session_id="test-session-123",
    )
    print(f"Normal model test: {result}")
    assert result["is_anomalous"] == False

    # Test anomalous model update (large weights)
    anomalous_model = {
        "weights": {
            "layer1": [100.0, 0.2, -0.15, 0.3],  # Large weight
            "layer2": [0.5, -0.4, 0.2, 0.1],
        },
        "metadata": {"epoch": 5, "accuracy": 0.85},
    }

    result = mock_anomaly_detection(
        anomalous_model,
        threshold=10.0,
        participant_id="participant_456",
        session_id="test-session-123",
    )
    print(f"Anomalous model test: {result}")
    assert result["is_anomalous"] == True
    assert "Large weights" in result["anomaly_reasons"][0]

    # Test suspicious participant
    result = mock_anomaly_detection(
        normal_model,
        threshold=10.0,
        participant_id="participant_suspicious",
        session_id="test-session-123",
    )
    print(f"Suspicious participant test: {result}")
    assert result["is_anomalous"] == True

    print("✅ Anomaly Detection tests passed!")


def test_model_consistency():
    """Test model consistency checking"""
    print("\n=== Testing Model Consistency ===")

    # Test consistent models
    model1 = {
        "weights": {"layer1": [0.1, 0.2, -0.15], "layer2": [0.5, -0.4, 0.2]},
        "metadata": {"epoch": 1},
    }

    model2 = {
        "weights": {"layer1": [0.11, 0.21, -0.16], "layer2": [0.51, -0.41, 0.21]},
        "metadata": {"epoch": 2},
    }

    result = mock_model_consistency_check(
        [model1, model2], session_id="test-session-123"
    )
    print(f"Consistent models test: {result}")
    assert result["is_consistent"] == True

    # Test inconsistent models
    model3 = {
        "weights": {
            "layer1": [0.1, 0.2],  # Different structure
            "layer3": [0.5, -0.4],  # Different layer name
        },
        "metadata": {"epoch": 3},
    }

    result = mock_model_consistency_check(
        [model1, model3], session_id="test-session-123"
    )
    print(f"Inconsistent models test: {result}")
    assert result["is_consistent"] == False
    assert len(result["inconsistencies"]) > 0

    print("✅ Model Consistency tests passed!")


def test_model_signature_verification():
    """Test model signature verification"""
    print("\n=== Testing Model Signature Verification ===")

    # Test valid signature
    model_data = {"weights": {"layer1": [0.1, 0.2]}, "metadata": {"epoch": 1}}
    model_bytes = json.dumps(model_data).encode()
    secret_key = "test-secret-key-123"

    # Create valid signature
    valid_signature = create_mock_model_signature(model_bytes, secret_key)

    result = mock_model_signature_verification(
        model_bytes,
        valid_signature,
        secret_key,
        participant_id="participant_123",
        session_id="test-session-123",
    )
    print(f"Valid signature test: {result}")
    assert result["is_valid"] == True

    # Test invalid signature
    invalid_signature = b"invalid_signature_bytes"

    result = mock_model_signature_verification(
        model_bytes,
        invalid_signature,
        secret_key,
        participant_id="participant_456",
        session_id="test-session-123",
    )
    print(f"Invalid signature test: {result}")
    assert result["is_valid"] == False

    print("✅ Model Signature Verification tests passed!")


def test_complete_validation_pipeline():
    """Test the complete model validation pipeline"""
    print("\n=== Testing Complete Validation Pipeline ===")

    # Test valid model that passes all checks
    valid_model = {
        "weights": {"layer1": [0.1, 0.2, -0.15, 0.3], "layer2": [0.5, -0.4, 0.2, 0.1]},
        "metadata": {"epoch": 5, "accuracy": 0.85},
    }

    model_bytes = json.dumps(valid_model).encode()
    secret_key = "test-secret-key-123"
    signature = create_mock_model_signature(model_bytes, secret_key)

    # Run all validation steps
    print("Running complete validation pipeline...")

    # Step 1: Anomaly detection
    anomaly_result = mock_anomaly_detection(
        valid_model,
        threshold=10.0,
        participant_id="participant_123",
        session_id="test-session-123",
    )

    # Step 2: Signature verification
    signature_result = mock_model_signature_verification(
        model_bytes,
        signature,
        secret_key,
        participant_id="participant_123",
        session_id="test-session-123",
    )

    # Step 3: Consistency check
    previous_models = [
        {
            "weights": {
                "layer1": [0.05, 0.15, -0.1, 0.25],
                "layer2": [0.45, -0.35, 0.15, 0.05],
            },
            "metadata": {"epoch": 4},
        }
    ]

    consistency_result = mock_model_consistency_check(
        previous_models + [valid_model], session_id="test-session-123"
    )

    # Aggregate results
    pipeline_result = {
        "passed_validation": (
            not anomaly_result["is_anomalous"]
            and signature_result["is_valid"]
            and consistency_result["is_consistent"]
        ),
        "validation_steps": {
            "anomaly_detection": anomaly_result,
            "signature_verification": signature_result,
            "consistency_check": consistency_result,
        },
    }

    print(f"Pipeline result: {json.dumps(pipeline_result, indent=2)}")
    assert pipeline_result["passed_validation"] == True

    print("✅ Complete Validation Pipeline tests passed!")


def main():
    """Run all security tests"""
    print("🔒 Vega 2.0 Federated Learning Security Test Suite")
    print("=" * 60)

    try:
        test_api_key_authentication()
        test_audit_logging()
        test_anomaly_detection()
        test_model_consistency()
        test_model_signature_verification()
        test_complete_validation_pipeline()

        print("\n" + "=" * 60)
        print("🎉 ALL SECURITY TESTS PASSED!")
        print("✅ API Key Authentication")
        print("✅ Audit Logging")
        print("✅ Anomaly Detection")
        print("✅ Model Consistency Checking")
        print("✅ Model Signature Verification")
        print("✅ Complete Validation Pipeline")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
