"""
Security utilities for Vega2.0 federated learning
- API key/token authentication
- TLS/SSL context helpers
- Audit logging stub
- Anomaly detection stub
- Model signature/verification stub
"""

import ssl
import logging
import json
import time
from typing import Optional

logger = logging.getLogger(__name__)


# --- API Key/Token Auth ---
def check_api_key(provided: str, allowed: set[str]) -> bool:
    if not provided:
        logger.warning("No API key provided")
        return False
    if provided not in allowed:
        logger.warning("Invalid API key: %s", provided)
        return False
    return True


# --- TLS/SSL Context ---
def create_ssl_context(
    certfile: str, keyfile: str, cafile: Optional[str] = None
) -> ssl.SSLContext:
    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ctx.load_cert_chain(certfile, keyfile)
    if cafile:
        ctx.load_verify_locations(cafile)
    return ctx


# --- Audit Logging ---
import json
import time


def audit_log(
    event: str,
    details: dict,
    participant_id: Optional[str] = None,
    session_id: Optional[str] = None,
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
    logger.info(f"AUDIT: {json.dumps(audit_entry)}")

    # Also log to a dedicated audit file if configured
    try:
        audit_file = "/var/log/vega/audit.log"  # Production path
        with open(audit_file, "a") as f:
            f.write(f"{json.dumps(audit_entry)}\n")
    except (FileNotFoundError, PermissionError):
        # Fallback to local audit log
        try:
            audit_file = "./audit.log"
            with open(audit_file, "a") as f:
                f.write(f"{json.dumps(audit_entry)}\n")
        except Exception:
            pass  # Continue without file logging if both fail


# --- Anomaly Detection ---
import numpy as np
from typing import Dict, List, Union, Any


def is_anomalous_update(
    update: Dict[str, Any],
    threshold: float = 10.0,
    participant_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Detect anomalous model updates based on various heuristics.

    Args:
        update: Model update dictionary (gradients, weights, etc.)
        threshold: Threshold for detecting large values
        participant_id: ID of participant sending update
        session_id: Federated session ID

    Returns:
        Dict with anomaly detection results
    """
    anomalies = []
    details = {}

    try:
        # Check for extremely large values
        large_values = []
        for key, value in update.items():
            if isinstance(value, (int, float)):
                if abs(float(value)) > threshold:
                    large_values.append(
                        {"key": key, "value": float(value), "threshold": threshold}
                    )
            elif hasattr(value, "__iter__") and not isinstance(value, str):
                # Handle arrays/lists
                try:
                    flat_values = np.array(value).flatten()
                    max_val = np.max(np.abs(flat_values))
                    if max_val > threshold:
                        large_values.append(
                            {
                                "key": key,
                                "max_abs_value": float(max_val),
                                "threshold": threshold,
                            }
                        )
                except:
                    pass  # Skip if can't convert to array

        if large_values:
            anomalies.append("large_values")
            details["large_values"] = large_values

        # Check for NaN or infinite values
        nan_inf_keys = []
        for key, value in update.items():
            try:
                if hasattr(value, "__iter__") and not isinstance(value, str):
                    arr = np.array(value)
                    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                        nan_inf_keys.append(key)
                elif isinstance(value, (int, float)):
                    if np.isnan(value) or np.isinf(value):
                        nan_inf_keys.append(key)
            except:
                pass

        if nan_inf_keys:
            anomalies.append("nan_or_inf_values")
            details["nan_or_inf_keys"] = nan_inf_keys

        # Check for suspiciously small updates (possible attack)
        tiny_updates = []
        for key, value in update.items():
            try:
                if hasattr(value, "__iter__") and not isinstance(value, str):
                    arr = np.array(value)
                    if np.all(np.abs(arr) < 1e-10):
                        tiny_updates.append(key)
                elif isinstance(value, (int, float)):
                    if abs(float(value)) < 1e-10:
                        tiny_updates.append(key)
            except:
                pass

        if len(tiny_updates) > len(update) * 0.8:  # 80% of values are tiny
            anomalies.append("suspiciously_small_update")
            details["tiny_update_keys"] = tiny_updates

        # Check update structure consistency
        expected_keys = ["weights", "gradients", "metadata"]  # Common ML update keys
        missing_expected = [k for k in expected_keys if k in update]
        if not missing_expected:
            anomalies.append("unexpected_structure")
            details["available_keys"] = list(update.keys())
            details["expected_keys"] = expected_keys

    except Exception as e:
        anomalies.append("analysis_error")
        details["error"] = str(e)

    result = {
        "is_anomalous": len(anomalies) > 0,
        "anomaly_types": anomalies,
        "details": details,
        "participant_id": participant_id,
        "session_id": session_id,
    }

    # Audit log anomaly detection result
    if anomalies:
        audit_log(
            "anomaly_detected",
            {
                "anomaly_types": anomalies,
                "details": details,
                "update_keys": list(update.keys()),
            },
            participant_id=participant_id,
            session_id=session_id,
        )

    return result


def check_model_consistency(
    models: List[Dict[str, Any]],
    tolerance: float = 0.1,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Check consistency across multiple model updates for potential Byzantine attacks.

    Args:
        models: List of model updates from different participants
        tolerance: Tolerance for considering models similar
        session_id: Federated session ID

    Returns:
        Dict with consistency analysis
    """
    if len(models) < 2:
        return {
            "is_consistent": True,
            "reason": "insufficient_models_to_compare",
            "model_count": len(models),
        }

    try:
        # Simple consistency check: compare model keys
        key_sets = [set(model.keys()) for model in models]
        common_keys = set.intersection(*key_sets)

        inconsistencies = []

        # Check if all models have same structure
        if len(set(frozenset(ks) for ks in key_sets)) > 1:
            inconsistencies.append("structure_mismatch")

        # Check value ranges for common keys
        outliers = []
        for key in common_keys:
            values = []
            for i, model in enumerate(models):
                try:
                    if isinstance(model[key], (int, float)):
                        values.append((i, float(model[key])))
                    elif hasattr(model[key], "__iter__"):
                        arr = np.array(model[key])
                        values.append((i, float(np.mean(arr))))
                except:
                    continue

            if len(values) >= 2:
                vals = [v[1] for v in values]
                mean_val = np.mean(vals)
                std_val = np.std(vals)

                # Flag outliers beyond 2 standard deviations
                for idx, val in values:
                    if std_val > 0 and abs(val - mean_val) > 2 * std_val:
                        outliers.append(
                            {
                                "model_index": idx,
                                "key": key,
                                "value": val,
                                "mean": mean_val,
                                "std": std_val,
                            }
                        )

        if outliers:
            inconsistencies.append("statistical_outliers")

        result = {
            "is_consistent": len(inconsistencies) == 0,
            "inconsistency_types": inconsistencies,
            "model_count": len(models),
            "common_keys": list(common_keys),
            "outliers": outliers[:10],  # Limit outliers
        }

        if inconsistencies:
            audit_log(
                "model_inconsistency_detected",
                {
                    "inconsistency_types": inconsistencies,
                    "model_count": len(models),
                    "outlier_count": len(outliers),
                },
                session_id=session_id,
            )

        return result

    except Exception as e:
        return {"is_consistent": False, "error": str(e), "model_count": len(models)}


# --- Model Signature/Verification ---
import hashlib
import hmac


def compute_model_hash(model_data: Dict[str, Any]) -> str:
    """
    Compute a deterministic hash of model data for integrity verification.

    Args:
        model_data: Model update data

    Returns:
        Hexadecimal hash string
    """
    try:
        # Create a deterministic string representation
        model_str = json.dumps(model_data, sort_keys=True)
        return hashlib.sha256(model_str.encode()).hexdigest()
    except Exception as e:
        # Fallback for non-serializable data
        return hashlib.sha256(str(model_data).encode()).hexdigest()


def verify_model_signature(
    model_bytes: bytes,
    signature: bytes,
    secret_key: str,
    participant_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Verify HMAC signature of model data for authenticity.

    Args:
        model_bytes: Serialized model data
        signature: HMAC signature to verify
        secret_key: Shared secret key for HMAC
        participant_id: ID of participant who signed the model
        session_id: Federated session ID

    Returns:
        Dict with verification results
    """
    try:
        # Compute expected signature
        expected_signature = hmac.new(
            secret_key.encode(), model_bytes, hashlib.sha256
        ).digest()

        # Verify signature
        is_valid = hmac.compare_digest(signature, expected_signature)

        result = {
            "is_valid": is_valid,
            "participant_id": participant_id,
            "session_id": session_id,
            "model_size_bytes": len(model_bytes),
            "signature_size_bytes": len(signature),
        }

        # Audit log verification result
        audit_log(
            "model_signature_verified" if is_valid else "model_signature_invalid",
            {
                "is_valid": is_valid,
                "model_size_bytes": len(model_bytes),
                "signature_size_bytes": len(signature),
            },
            participant_id=participant_id,
            session_id=session_id,
        )

        return result

    except Exception as e:
        result = {
            "is_valid": False,
            "error": str(e),
            "participant_id": participant_id,
            "session_id": session_id,
        }

        audit_log(
            "model_signature_verification_error",
            {"error": str(e)},
            participant_id=participant_id,
            session_id=session_id,
        )

        return result


def create_model_signature(model_bytes: bytes, secret_key: str) -> bytes:
    """
    Create HMAC signature for model data.

    Args:
        model_bytes: Serialized model data
        secret_key: Shared secret key for HMAC

    Returns:
        HMAC signature bytes
    """
    return hmac.new(secret_key.encode(), model_bytes, hashlib.sha256).digest()


def validate_model_update_pipeline(
    model_update: Dict[str, Any],
    model_bytes: Optional[bytes] = None,
    signature: Optional[bytes] = None,
    secret_key: Optional[str] = None,
    previous_models: Optional[List[Dict[str, Any]]] = None,
    participant_id: Optional[str] = None,
    session_id: Optional[str] = None,
    anomaly_threshold: float = 10.0,
) -> Dict[str, Any]:
    """
    Complete model validation pipeline combining anomaly detection and signature verification.

    Args:
        model_update: Model update to validate
        model_bytes: Serialized model for signature verification
        signature: Model signature to verify
        secret_key: Secret key for signature verification
        previous_models: Previous model updates for consistency checking
        participant_id: ID of participant sending update
        session_id: Federated session ID
        anomaly_threshold: Threshold for anomaly detection

    Returns:
        Dict with comprehensive validation results
    """
    validation_results = {
        "participant_id": participant_id,
        "session_id": session_id,
        "timestamp": time.time(),
        "passed_validation": True,
        "validation_steps": {},
    }

    # Step 1: Anomaly Detection
    try:
        anomaly_result = is_anomalous_update(
            model_update,
            threshold=anomaly_threshold,
            participant_id=participant_id,
            session_id=session_id,
        )
        validation_results["validation_steps"]["anomaly_detection"] = anomaly_result

        if anomaly_result["is_anomalous"]:
            validation_results["passed_validation"] = False
    except Exception as e:
        validation_results["validation_steps"]["anomaly_detection"] = {
            "error": str(e),
            "step_failed": True,
        }
        validation_results["passed_validation"] = False

    # Step 2: Signature Verification (if provided)
    if model_bytes and signature and secret_key:
        try:
            signature_result = verify_model_signature(
                model_bytes,
                signature,
                secret_key,
                participant_id=participant_id,
                session_id=session_id,
            )
            validation_results["validation_steps"][
                "signature_verification"
            ] = signature_result

            if not signature_result["is_valid"]:
                validation_results["passed_validation"] = False
        except Exception as e:
            validation_results["validation_steps"]["signature_verification"] = {
                "error": str(e),
                "step_failed": True,
            }
            validation_results["passed_validation"] = False

    # Step 3: Model Consistency Check (if previous models provided)
    if previous_models:
        try:
            all_models = previous_models + [model_update]
            consistency_result = check_model_consistency(
                all_models, session_id=session_id
            )
            validation_results["validation_steps"][
                "consistency_check"
            ] = consistency_result

            if not consistency_result["is_consistent"]:
                validation_results["passed_validation"] = False
        except Exception as e:
            validation_results["validation_steps"]["consistency_check"] = {
                "error": str(e),
                "step_failed": True,
            }
            validation_results["passed_validation"] = False

    # Audit log final validation result
    audit_log(
        "model_validation_complete",
        {
            "passed_validation": validation_results["passed_validation"],
            "steps_completed": list(validation_results["validation_steps"].keys()),
            "failed_steps": [
                step
                for step, result in validation_results["validation_steps"].items()
                if not result.get(
                    "is_valid",
                    result.get("is_consistent", not result.get("is_anomalous", False)),
                )
            ],
        },
        participant_id=participant_id,
        session_id=session_id,
    )

    return validation_results
