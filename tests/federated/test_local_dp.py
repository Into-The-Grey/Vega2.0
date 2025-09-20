"""
Unit tests for local differential privacy (LDP) in DifferentialPrivacy.
"""

import numpy as np

try:
    from src.vega.federated.dp import DifferentialPrivacy, DifferentialPrivacyConfig
except ImportError:
    from src.vega.federated.dp import DifferentialPrivacy, DifferentialPrivacyConfig


def test_local_dp_basic():
    np.random.seed(123)
    config = DifferentialPrivacyConfig(noise_multiplier=1.0, clipping_norm=1.0)
    dp = DifferentialPrivacy(config)
    update = np.ones(10)
    privatized = dp.apply_local_dp(update)
    assert privatized.shape == update.shape
    # Should be different from original due to noise
    assert not np.allclose(privatized, update)
    # Should be clipped if norm > clipping_norm
    big_update = np.ones(10) * 10
    privatized_big = dp.apply_local_dp(big_update)
    assert np.linalg.norm(privatized_big) <= config.clipping_norm + 5  # Allow for noise


def test_local_dp_custom_params():
    np.random.seed(42)
    config = DifferentialPrivacyConfig(noise_multiplier=0.5, clipping_norm=2.0)
    dp = DifferentialPrivacy(config)
    update = np.ones(5)
    # Use custom noise/clipping
    privatized = dp.apply_local_dp(
        update, local_noise_multiplier=2.0, local_clipping_norm=1.0
    )
    # Should be clipped to norm 1.0
    assert np.linalg.norm(privatized) <= 1.0 + 5  # Allow for noise
    # Should have higher noise than default
    privatized_default = dp.apply_local_dp(update)
    std_custom = np.std(privatized - update)
    std_default = np.std(privatized_default - update)
    assert std_custom > std_default
    print("All local DP tests passed.")


if __name__ == "__main__":
    test_local_dp_basic()
    test_local_dp_custom_params()
