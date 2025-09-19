#!/usr/bin/env python3
"""
Simplified test script for federated learning security integration.

Tests the security components directly without importing the full Vega framework.
"""

import sys
import os
import json
import numpy as np
from typing import Dict, Any


def test_security_integration():
    """Test security integration with simplified mock components"""

    print("🔒 Testing Federated Learning Security Integration")
    print("=" * 60)

    try:
        # Add the path for imports
        sys.path.insert(0, "/home/ncacord/Vega2.0/src/vega/federated")

        # Test 1: Import and test security functions directly
        print("\n🔧 Testing Security Functions")

        # Import security functions
        from security import (
            audit_log,
            is_anomalous_update,
            check_model_consistency,
            validate_model_update_pipeline,
            verify_model_signature,
            create_model_signature,
        )

        print("✅ Security functions imported successfully!")

        # Test audit logging
        print("Testing audit logging...")
        audit_log(
            "test_security_integration",
            {"test": "data", "value": 123},
            participant_id="test_participant",
            session_id="test_session",
        )
        print("✅ Audit logging working!")

        # Test anomaly detection
        print("Testing anomaly detection...")

        # Normal model update
        normal_update = {
            "weights": {"layer1": [1.0, 2.0, 3.0], "layer2": [0.5, -0.3, 0.8]},
            "metadata": {"epoch": 5},
        }

        normal_result = is_anomalous_update(
            normal_update,
            threshold=10.0,
            participant_id="test_participant",
            session_id="test_session",
        )
        print(f"Normal update result: {normal_result}")
        assert not normal_result["is_anomalous"]

        # Anomalous model update
        anomalous_update = {
            "weights": {
                "layer1": [100.0, 2.0, 3.0],  # Large weight
                "layer2": [0.5, -0.3, 0.8],
            },
            "metadata": {"epoch": 5},
        }

        anomalous_result = is_anomalous_update(
            anomalous_update,
            threshold=10.0,
            participant_id="test_participant",
            session_id="test_session",
        )
        print(f"Anomalous update result: {anomalous_result}")
        assert anomalous_result["is_anomalous"]

        print("✅ Anomaly detection working correctly!")

        # Test model consistency checking
        print("Testing model consistency...")

        model1 = {"weights": {"layer1": [1.0, 2.0, 3.0], "layer2": [0.5, -0.3, 0.8]}}

        model2 = {"weights": {"layer1": [1.1, 2.1, 3.1], "layer2": [0.51, -0.31, 0.81]}}

        model3 = {
            "weights": {
                "layer1": [1.0, 2.0],  # Different structure
                "layer3": [0.5, -0.3],  # Different layer name
            }
        }

        # Test consistent models
        consistent_result = check_model_consistency(
            [model1, model2], session_id="test_session"
        )
        print(f"Consistent models result: {consistent_result}")
        assert consistent_result["is_consistent"]

        # Test inconsistent models
        inconsistent_result = check_model_consistency(
            [model1, model3], session_id="test_session"
        )
        print(f"Inconsistent models result: {inconsistent_result}")
        assert not inconsistent_result["is_consistent"]

        print("✅ Model consistency checking working!")

        # Test signature verification
        print("Testing model signature verification...")

        model_data = json.dumps(normal_update, sort_keys=True)
        model_bytes = model_data.encode()
        secret_key = "test-secret-key-123"

        # Create signature
        signature = create_model_signature(model_bytes, secret_key)
        print(f"Created signature: {signature.hex()}")

        # Verify signature
        verification_result = verify_model_signature(
            model_bytes=model_bytes,
            signature=signature,
            secret_key=secret_key,
            participant_id="test_participant",
            session_id="test_session",
        )
        print(f"Signature verification result: {verification_result}")
        assert verification_result["is_valid"]

        # Test invalid signature
        invalid_signature = b"invalid_signature_bytes_12345"
        invalid_verification = verify_model_signature(
            model_bytes=model_bytes,
            signature=invalid_signature,
            secret_key=secret_key,
            participant_id="test_participant",
            session_id="test_session",
        )
        print(f"Invalid signature verification: {invalid_verification}")
        assert not invalid_verification["is_valid"]

        print("✅ Model signature verification working!")

        # Test complete validation pipeline
        print("Testing complete validation pipeline...")

        pipeline_result = validate_model_update_pipeline(
            model_update=normal_update,
            previous_models=[model1, model2],
            participant_id="test_participant",
            session_id="test_session",
            anomaly_threshold=10.0,
        )

        print(f"Pipeline validation result: {json.dumps(pipeline_result, indent=2)}")
        assert pipeline_result["passed_validation"]

        # Test pipeline with anomalous update
        anomalous_pipeline_result = validate_model_update_pipeline(
            model_update=anomalous_update,
            previous_models=[model1, model2],
            participant_id="test_participant",
            session_id="test_session",
            anomaly_threshold=10.0,
        )

        print(
            f"Anomalous pipeline result: {json.dumps(anomalous_pipeline_result, indent=2)}"
        )
        assert not anomalous_pipeline_result["passed_validation"]

        print("✅ Complete validation pipeline working!")

        # Test 2: FedAvg Security Integration (simplified)
        print("\n📊 Testing FedAvg Security Integration")

        # Create a minimal FedAvg config and test
        config_dict = {
            "convergence_threshold": 1e-4,
            "patience": 5,
            "max_rounds": 100,
            "selection_strategy": "random",
            "byzantine_tolerance": True,
            "byzantine_method": "median",
            "async_aggregation": False,
            "seed": 42,
        }

        # Mock the FedAvg class functionality
        class MockFedAvg:
            def __init__(self, config):
                self.config = config
                self.convergence_history = []
                self.current_round = 0

            def security_aware_aggregate(
                self, client_weights, client_sizes, participant_ids, session_id
            ):
                """Simplified version of the security-aware aggregation"""

                # Security validation
                filtered_weights = []
                filtered_sizes = []
                security_report = {
                    "total_participants": len(client_weights),
                    "anomalous_participants": [],
                    "participants_after_filtering": 0,
                }

                for i, weights in enumerate(client_weights):
                    participant_id = (
                        participant_ids[i] if participant_ids else f"participant_{i}"
                    )

                    # Convert to testable format
                    weights_dict = {"weights": {}}
                    for key, array in weights.items():
                        weights_dict["weights"][key] = (
                            array.tolist() if hasattr(array, "tolist") else array
                        )

                    # Validate
                    validation_result = validate_model_update_pipeline(
                        model_update=weights_dict,
                        participant_id=participant_id,
                        session_id=session_id,
                        anomaly_threshold=10.0,
                    )

                    if validation_result["passed_validation"]:
                        filtered_weights.append(weights)
                        filtered_sizes.append(client_sizes[i])
                    else:
                        security_report["anomalous_participants"].append(
                            {
                                "participant_id": participant_id,
                                "reason": "Failed validation",
                            }
                        )

                security_report["participants_after_filtering"] = len(filtered_weights)

                if len(filtered_weights) == 0:
                    raise ValueError("No valid participants after security filtering")

                # Simple aggregation (weighted average)
                total_size = sum(filtered_sizes)
                aggregated = {}
                for key in filtered_weights[0]:
                    aggregated[key] = sum(
                        w[key] * (sz / total_size)
                        for w, sz in zip(filtered_weights, filtered_sizes)
                    )

                return aggregated, security_report

        # Test the mock FedAvg
        mock_fedavg = MockFedAvg(config_dict)

        # Test data
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
            },  # Anomalous
        ]

        participant_sizes = [100, 150, 80]
        participant_ids = ["participant_1", "participant_2", "participant_suspicious"]

        # Test aggregation
        aggregated_weights, security_report = mock_fedavg.security_aware_aggregate(
            participant_weights, participant_sizes, participant_ids, "test_session_123"
        )

        print(f"Aggregation security report: {json.dumps(security_report, indent=2)}")
        print(f"Aggregated weights keys: {list(aggregated_weights.keys())}")

        # Verify security filtering worked
        assert (
            security_report["participants_after_filtering"] == 2
        )  # Should filter out anomalous one
        assert len(security_report["anomalous_participants"]) == 1
        assert (
            security_report["anomalous_participants"][0]["participant_id"]
            == "participant_suspicious"
        )

        print("✅ FedAvg security integration working!")

        print("\n" + "=" * 60)
        print("🎉 ALL SECURITY INTEGRATION TESTS PASSED!")
        print("✅ Security Function Validation")
        print("✅ Anomaly Detection Integration")
        print("✅ Model Consistency Checking")
        print("✅ Signature Verification")
        print("✅ Complete Validation Pipeline")
        print("✅ FedAvg Security Filtering")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_security_integration()
    sys.exit(0 if success else 1)
