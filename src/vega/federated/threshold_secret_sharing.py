"""
Threshold Secret Sharing (Shamir's Secret Sharing) for Secure Aggregation

Implements a simple (t, n) threshold scheme for federated learning.

Dependencies: numpy
"""

import numpy as np
from typing import List, Tuple


class ShamirSecretSharing:
    def __init__(self, threshold: int, num_shares: int, prime: int = 2**127 - 1):
        assert 1 < threshold <= num_shares
        self.t = threshold
        self.n = num_shares
        self.prime = prime

    def _eval_poly(self, coeffs: List[int], x: int) -> int:
        result = 0
        for power, c in enumerate(coeffs):
            result = (result + c * pow(x, power, self.prime)) % self.prime
        return result

    def split(self, secret: int) -> List[Tuple[int, int]]:
        """
        Split secret into n shares, any t of which can reconstruct.
        Returns list of (x, y) shares.
        """
        rng = __import__("random").SystemRandom()
        coeffs = [secret] + [rng.randrange(0, self.prime) for _ in range(self.t - 1)]
        shares = [(i, self._eval_poly(coeffs, i)) for i in range(1, self.n + 1)]
        return shares

    def reconstruct(self, shares: List[Tuple[int, int]]) -> int:
        """
        Reconstruct secret from t shares using Lagrange interpolation.
        """
        assert len(shares) >= self.t
        p = self.prime
        # Fast path for two shares (common case when threshold=2)
        if len(shares) == 2:
            (x1, y1), (x2, y2) = shares
            x1 %= p
            y1 %= p
            x2 %= p
            y2 %= p
            num = (y1 * x2 - y2 * x1) % p
            den = (x2 - x1) % p
            inv_den = pow(den, p - 2, p)
            return (num * inv_den) % p
        # Normalize shares y modulo p
        norm_shares = [(int(x) % p, int(y) % p) for (x, y) in shares]
        secret = 0
        for j, (xj, yj) in enumerate(norm_shares):
            num, den = 1, 1
            for m, (xm, _) in enumerate(norm_shares):
                if m != j:
                    num = (num * ((-xm) % p)) % p
                    den = (den * ((xj - xm) % p)) % p
            # Modular inverse using Fermat's little theorem (p is prime)
            inv_den = pow(den, p - 2, p)
            lagrange_coeff = (num * inv_den) % p
            secret = (secret + (yj * lagrange_coeff)) % p
        return secret


# Example usage and test
if __name__ == "__main__":
    sss = ShamirSecretSharing(threshold=3, num_shares=5)
    secret = 12345678901234567890
    shares = sss.split(secret)
    print(f"Shares: {shares}")
    # Reconstruct from any 3 shares
    recovered = sss.reconstruct(shares[:3])
    print(f"Recovered: {recovered}")
    assert recovered == secret
    print("🎉 Shamir's Secret Sharing test passed!")
