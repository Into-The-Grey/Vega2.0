#!/usr/bin/env python3
"""
Test script for Vega 2.0 federated learning with integrated security features.

Tests the complete pipeline:
1. Participant initialization with security
2. Training rounds with anomaly detection
3. Weight updates with signature verification
4. Secure aggregation with validation
"""

import sys
import os
import json
import numpy as np
from typing import Dict, Any

# Add the federated module to path
sys.path.insert(0, "/home/ncacord/Vega2.0/src")


def mock_security_functions():
    """Mock the security functions for testing"""

    def mock_audit_log(event_type: str, event_data: Dict[str, Any], **kwargs):
        print(f"AUDIT: {event_type} - {json.dumps(event_data, indent=2)}")

    def mock_check_api_key(provided: str, allowed: set):
        return provided in allowed

    def mock_validate_model_update_pipeline(
        model_update: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        # Simple validation - check for reasonable values
        is_valid = True
        reasons = []

        try:
            weights = model_update.get("weights", {})
            for layer_name, layer_weights in weights.items():
                if isinstance(layer_weights, list):
                    max_val = max(
                        abs(w) for w in layer_weights if isinstance(w, (int, float))
                    )
                    if max_val > 50.0:  # Threshold for anomaly
                        is_valid = False
                        reasons.append(f"Large weights in {layer_name}: {max_val}")
        except Exception as e:
            is_valid = False
            reasons.append(f"Validation error: {str(e)}")

        return {
            "passed_validation": is_valid,
            "validation_steps": {
                "anomaly_detection": {"is_anomalous": not is_valid, "reasons": reasons}
            },
        }

    def mock_create_model_signature(model_bytes: bytes, secret_key: str) -> bytes:
        return b"mock_signature_12345"

    def mock_verify_model_signature(
        model_bytes: bytes, signature: bytes, secret_key: str, **kwargs
    ) -> Dict[str, Any]:
        return {
            "is_valid": signature == b"mock_signature_12345",
            "model_size_bytes": len(model_bytes),
            "signature_size_bytes": len(signature),
        }

    return {
        "audit_log": mock_audit_log,
        "check_api_key": mock_check_api_key,
        "validate_model_update_pipeline": mock_validate_model_update_pipeline,
        "create_model_signature": mock_create_model_signature,
        "verify_model_signature": mock_verify_model_signature,
    }


def test_federated_learning_with_security():
    """Test the federated learning pipeline with security"""

    print("🔒 Testing Vega 2.0 Federated Learning with Security")
    print("=" * 60)

    # Mock the security functions
    security_mocks = mock_security_functions()

    # Patch the security module
    import src.vega.federated.security as security_module

    for name, mock_func in security_mocks.items():
        setattr(security_module, name, mock_func)

    try:
        # Test 1: FedAvg with Security
        print("\n📊 Testing FedAvg with Security Integration")

        from src.vega.federated.fedavg import FedAvg, FedAvgConfig

        # Create FedAvg with security enabled
        config = FedAvgConfig(byzantine_tolerance=True, byzantine_method="median")
        fedavg = FedAvg(config)

        # Simulate participant model updates
        participant_weights = [
            {
                "layer1": np.array([1.0, 2.0, 3.0]),
                "layer2": np.array([0.5, 0.6]),
            },  # Normal
            {
                "layer1": np.array([1.1, 2.1, 3.1]),
                "layer2": np.array([0.51, 0.61]),
            },  # Normal
            {
                "layer1": np.array([100.0, 2.0, 3.0]),
                "layer2": np.array([0.5, 0.6]),
            },  # Anomalous (large weight)
        ]

        participant_sizes = [100, 150, 80]
        participant_ids = ["participant_1", "participant_2", "participant_suspicious"]

        # Test aggregation with security
        print("Running secure aggregation...")
        aggregated_weights, security_report = fedavg.aggregate(
            client_weights=participant_weights,
            client_sizes=participant_sizes,
            participant_ids=participant_ids,
            session_id="test_session_123",
            enable_security=True,
            anomaly_threshold=50.0,
        )

        print(f"✅ Aggregation completed!")
        print(f"Security Report: {json.dumps(security_report, indent=2)}")
        print(
            f"Aggregated weights shape: {[f'{k}: {v.shape}' for k, v in aggregated_weights.items()]}"
        )

        # Verify security filtering worked
        expected_participants_after_filtering = 2  # Should filter out the anomalous one
        assert (
            security_report["participants_after_filtering"]
            == expected_participants_after_filtering
        )
        assert len(security_report["anomalous_participants"]) == 1
        assert (
            security_report["anomalous_participants"][0]["participant_id"]
            == "participant_suspicious"
        )

        print("✅ Security filtering worked correctly!")

        # Test 2: Participant with Security
        print("\n👥 Testing Participant with Security Features")

        # Mock dependencies that aren't available
        import sys
        from unittest.mock import MagicMock

        # Mock model serialization
        mock_model_weights = MagicMock()
        mock_model_weights.to_dict.return_value = {
            "layer1": [1.0, 2.0],
            "layer2": [0.5],
        }
        mock_model_weights.from_dict.return_value = mock_model_weights

        sys.modules["src.vega.federated.model_serialization"] = MagicMock()
        sys.modules["src.vega.federated.model_serialization"].ModelWeights = MagicMock()
        sys.modules[
            "src.vega.federated.model_serialization"
        ].ModelWeights.from_dict.return_value = mock_model_weights
        sys.modules["src.vega.federated.model_serialization"].ModelSerializer = (
            MagicMock()
        )

        # Mock other dependencies
        sys.modules["src.vega.federated.communication"] = MagicMock()
        sys.modules["src.vega.federated.encryption"] = MagicMock()
        sys.modules["src.vega.federated.coordinator"] = MagicMock()

        from src.vega.federated.participant import FederatedParticipant

        # Create participant with security
        participant = FederatedParticipant(
            participant_id="test_participant",
            participant_name="Test Participant",
            api_key="test-key-123",
            secret_key="test-secret-key",
            enable_security=True,
        )

        print("✅ Participant created with security enabled!")

        # Test training round with security
        print("Testing training round with security...")

        # Mock training data
        participant.local_model = MagicMock()
        participant.training_data = [1, 2, 3]  # Mock data
        participant.active_session_id = "test_session_123"

        # Mock training results
        participant.trainer.train_pytorch_model = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.training_loss = 0.5
        mock_metrics.training_samples = 100
        mock_metrics.epochs_completed = 5
        mock_metrics.training_time = 10.0
        mock_metrics.to_dict.return_value = {
            "training_loss": 0.5,
            "training_samples": 100,
            "epochs_completed": 5,
            "training_time": 10.0,
        }
        participant.trainer.train_pytorch_model.return_value = mock_metrics
        participant.trainer.model_type = "pytorch"

        # Mock model extraction
        from src.vega.federated.model_serialization import ModelSerializer

        ModelSerializer.extract_pytorch_weights.return_value = mock_model_weights

        # Simulate training round
        training_message = {"session_id": "test_session_123", "round_number": 1}

        import asyncio

        async def test_training():
            result = await participant.handle_training_round(training_message)
            return result

        training_result = asyncio.run(test_training())

        print(f"Training result: {json.dumps(training_result, indent=2, default=str)}")

        # Verify training completed successfully
        assert training_result["status"] == "success"
        assert "security_validation" in training_result
        assert "model_signature" in training_result

        print("✅ Training round with security completed successfully!")

        # Test 3: Weight Update with Security
        print("\nTesting weight update with security...")

        weight_update_message = {
            "session_id": "test_session_123",
            "round_number": 2,
            "aggregated_weights": {"layer1": [1.1, 2.1], "layer2": [0.51]},
            "model_signature": mock_create_model_signature(
                b"test", "test-secret-key"
            ).hex(),
        }

        async def test_weight_update():
            result = await participant.handle_weight_update(weight_update_message)
            return result

        weight_update_result = asyncio.run(test_weight_update())

        print(
            f"Weight update result: {json.dumps(weight_update_result, indent=2, default=str)}"
        )

        # Verify weight update completed successfully
        assert weight_update_result["status"] == "success"
        assert "security_validation" in weight_update_result

        print("✅ Weight update with security completed successfully!")

        print("\n" + "=" * 60)
        print("🎉 ALL FEDERATED LEARNING SECURITY TESTS PASSED!")
        print("✅ Secure Aggregation with Anomaly Detection")
        print("✅ Participant Security Integration")
        print("✅ Training Round Security Validation")
        print("✅ Weight Update Security Verification")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_federated_learning_with_security()
    sys.exit(0 if success else 1)
