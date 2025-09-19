"""
Unit tests for Shamir's Secret Sharing threshold scheme.
"""

from threshold_secret_sharing import ShamirSecretSharing
import random


def test_shamir_basic():
    sss = ShamirSecretSharing(threshold=3, num_shares=5)
    secret = random.randint(1, 2**64)
    shares = sss.split(secret)
    # Test all combinations of 3 shares
    from itertools import combinations

    for subset in combinations(shares, 3):
        recovered = sss.reconstruct(list(subset))
        assert recovered == secret
    print("All 3-of-5 reconstructions succeeded.")


def test_shamir_edge_cases():
    sss = ShamirSecretSharing(threshold=2, num_shares=2)
    secret = 42
    shares = sss.split(secret)
    recovered = sss.reconstruct(shares)
    assert recovered == secret
    print("Edge case 2-of-2 succeeded.")


if __name__ == "__main__":
    test_shamir_basic()
    test_shamir_edge_cases()
    print("All Shamir's Secret Sharing tests passed.")
