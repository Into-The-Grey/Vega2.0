"""
Unit tests for SMPC secure sum using Shamir's Secret Sharing.
"""

import numpy as np

try:
    from src.vega.federated.smpc import SMPCSecureSum, SMPCConfig
except ImportError:
    from src.vega.federated.smpc import SMPCSecureSum, SMPCConfig


def test_smpc_secure_sum_basic():
    np.random.seed(123)
    cfg = SMPCConfig(threshold=2, num_participants=3)
    smpc = SMPCSecureSum(cfg)
    v1 = np.random.randn(8)
    v2 = np.random.randn(8)
    s1 = smpc.share_vector(v1)
    s2 = smpc.share_vector(v2)
    s12 = smpc.add_shared_vectors(s1, s2)
    # reconstruct from different pairs
    for pair in [(0, 1), (0, 2), (1, 2)]:
        subset = [s12[pair[0]], s12[pair[1]]]
        rec = smpc.reconstruct_sum(subset)
        expected = v1 + v2
        assert np.allclose(rec, expected, atol=1e-5)


def test_smpc_secure_sum_three_vectors():
    np.random.seed(7)
    cfg = SMPCConfig(threshold=3, num_participants=4)
    smpc = SMPCSecureSum(cfg)
    vs = [np.random.randn(6) for _ in range(3)]
    shares = [smpc.share_vector(v) for v in vs]
    total = shares[0]
    total = smpc.add_shared_vectors(total, shares[1])
    total = smpc.add_shared_vectors(total, shares[2])
    subset = [total[0], total[1], total[3]]
    rec = smpc.reconstruct_sum(subset)
    expected = sum(vs)
    assert np.allclose(rec, expected, atol=1e-5)


if __name__ == "__main__":
    test_smpc_secure_sum_basic()
    test_smpc_secure_sum_three_vectors()
    print("All SMPC tests passed.")
