"""
Unit tests for privacy auditing and verification in DifferentialPrivacy.
"""

import numpy as np
import os
import json

try:
    from src.vega.federated.dp import DifferentialPrivacy, DifferentialPrivacyConfig
except ImportError:
    from src.vega.federated.dp import DifferentialPrivacy, DifferentialPrivacyConfig


def test_audit_log_basic():
    np.random.seed(123)
    config = DifferentialPrivacyConfig(noise_multiplier=1.0, clipping_norm=1.0)
    dp = DifferentialPrivacy(config, enable_audit=True)
    update = np.ones(10)
    dp.apply_local_dp(update)
    dp.clip_and_add_noise(update, participant_id="userA")
    log = dp.get_audit_log()
    assert log is not None
    assert len(log) == 2
    assert log[0]["event"] == "apply_local_dp"
    assert log[1]["event"] == "clip_and_add_noise"
    # Check audit details
    for entry in log:
        assert "input_norm" in entry["details"]
        assert "output_norm" in entry["details"]


def test_audit_log_export():
    np.random.seed(42)
    config = DifferentialPrivacyConfig(noise_multiplier=0.5, clipping_norm=2.0)
    dp = DifferentialPrivacy(config, enable_audit=True)
    update = np.ones(5)
    dp.apply_local_dp(update)
    path = "audit_test_log.jsonl"
    dp.export_audit_log(path)
    assert os.path.exists(path)
    with open(path) as f:
        lines = f.readlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event"] == "apply_local_dp"
    os.remove(path)
    print("Audit log export test passed.")


def test_audit_log_disabled():
    config = DifferentialPrivacyConfig()
    dp = DifferentialPrivacy(config, enable_audit=False)
    update = np.ones(3)
    dp.apply_local_dp(update)
    assert dp.get_audit_log() is None
    try:
        dp.export_audit_log("should_fail.jsonl")
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass
    print("Audit log disabled test passed.")


if __name__ == "__main__":
    test_audit_log_basic()
    test_audit_log_export()
    test_audit_log_disabled()
    print("All privacy audit tests passed.")
