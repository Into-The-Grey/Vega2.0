"""
Unit tests for adaptive noise scaling in DifferentialPrivacy.
"""

import numpy as np

try:
    from vega.federated.dp import DifferentialPrivacy, DifferentialPrivacyConfig
except ImportError:
    from dp import DifferentialPrivacy, DifferentialPrivacyConfig


def test_adaptive_noise_scaling():
    np.random.seed(123)
    config = DifferentialPrivacyConfig(
        noise_multiplier=1.0, clipping_norm=1.0, adaptive_noise=True
    )
    dp = DifferentialPrivacy(config)
    update = np.ones(10)
    # Sensitivity = 2.0
    noisy1 = dp.clip_and_add_noise(update, sensitivity=2.0)
    noisy2 = dp.clip_and_add_noise(update, sensitivity=0.1)
    std1 = np.std(noisy1 - update)
    std2 = np.std(noisy2 - update)
    print(f"Std with sensitivity=2.0: {std1:.3f}")
    print(f"Std with sensitivity=0.1: {std2:.3f}")
    assert std1 > std2
    # Default sensitivity (should use L2 norm of update, but noise is per-element)
    stds = []
    for _ in range(100):
        noisy = dp.clip_and_add_noise(update)
        stds.append(np.std(noisy - update))
    mean_std = np.mean(stds)
    # The noise added is N(0, sensitivity) per element, so expected std is sensitivity
    expected_std = float(np.linalg.norm(update))
    expected_per_element_std = expected_std / np.sqrt(update.size)
    print(f"Mean std with default sensitivity (100 runs): {mean_std:.3f}")
    print(f"Expected per-element std: {expected_per_element_std:.3f}")
    assert abs(mean_std - expected_per_element_std) < 0.2
    # Edge case: zero sensitivity
    noisy4 = dp.clip_and_add_noise(update, sensitivity=0.0)
    std4 = np.std(noisy4 - update)
    print(f"Std with sensitivity=0.0: {std4:.3f}")
    assert std4 < 0.5
    print("All adaptive noise scaling tests passed.")


if __name__ == "__main__":
    test_adaptive_noise_scaling()
