"""
Paillier Homomorphic Encryption for Federated Aggregation

Provides simple, dependency-free Paillier cryptosystem primitives and
vector helpers to enable homomorphic addition (sum of encrypted updates).

Design goals:
- Local-first, no external network calls
- Pure Python, no external crypto packages
- Safe defaults, but optimized for test speed (configurable key size)
- Supports float vectors via fixed-point encoding

Security note:
- This implementation is for educational and small-scale federated use.
- For production-grade security, prefer battle-tested libraries and audited code.
"""

from __future__ import annotations

import math
import secrets
from dataclasses import dataclass
from typing import List, Sequence, Tuple


# ----------------------------
# Number theoretic utilities
# ----------------------------


def _egcd(a: int, b: int) -> Tuple[int, int, int]:
    if b == 0:
        return a, 1, 0
    g, x, y = _egcd(b, a % b)
    return g, y, x - (a // b) * y


def _lcm(a: int, b: int) -> int:
    return a // math.gcd(a, b) * b


def _is_probable_prime(n: int, rounds: int = 16) -> bool:
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for p in small_primes:
        if n % p == 0:
            return n == p
    # write n-1 = d * 2^s
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    rng = secrets.SystemRandom()
    for _ in range(rounds):
        a = rng.randrange(2, n - 2)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for __ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _gen_prime(bits: int) -> int:
    rng = secrets.SystemRandom()
    while True:
        # ensure top and bottom bit set
        candidate = (1 << (bits - 1)) | rng.getrandbits(bits - 1) | 1
        if _is_probable_prime(candidate):
            return candidate


# ----------------------------
# Paillier key structures
# ----------------------------


@dataclass
class PaillierPublicKey:
    n: int
    g: int

    @property
    def n_sq(self) -> int:
        return self.n * self.n


@dataclass
class PaillierPrivateKey:
    lambda_param: int  # lcm(p-1, q-1)
    mu: int
    n: int


@dataclass
class PaillierKeypair:
    public_key: PaillierPublicKey
    private_key: PaillierPrivateKey

    @staticmethod
    def generate(bits: int = 512) -> "PaillierKeypair":
        """
        Generate a Paillier keypair.

        bits: approximate size of modulus n in bits (p and q ~ bits/2)
        """
        if bits < 128:
            raise ValueError("Key size too small; must be >= 128 bits for tests")
        half = bits // 2
        while True:
            p = _gen_prime(half)
            q = _gen_prime(half)
            if p == q:
                continue
            n = p * q
            # Ensure gcd(n, (p-1)(q-1)) == 1 is always true for distinct odd primes
            lam = _lcm(p - 1, q - 1)
            g = n + 1  # simple/standard choice
            n_sq = n * n
            # Compute mu = (L(g^lambda mod n^2))^{-1} mod n
            u = pow(g, lam, n_sq)
            l_val = (u - 1) // n
            # invert l_val modulo n
            g_inv, x, _ = _egcd(l_val % n, n)
            if g_inv != 1:
                # very unlikely, regenerate
                continue
            mu = x % n
            pub = PaillierPublicKey(n=n, g=g)
            priv = PaillierPrivateKey(lambda_param=lam, mu=mu, n=n)
            return PaillierKeypair(public_key=pub, private_key=priv)


# ----------------------------
# Core Paillier operations
# ----------------------------


def encrypt_int(pub: PaillierPublicKey, m: int) -> int:
    """Encrypt integer m in [0, n) using Paillier (g=n+1)."""
    if not (0 <= m < pub.n):
        raise ValueError("Plaintext out of range for Paillier: must be 0 <= m < n")
    rng = secrets.SystemRandom()
    while True:
        r = rng.randrange(1, pub.n)
        if math.gcd(r, pub.n) == 1:
            break
    # c = g^m * r^n mod n^2
    c1 = pow(pub.g, m, pub.n_sq)
    c2 = pow(r, pub.n, pub.n_sq)
    return (c1 * c2) % pub.n_sq


def decrypt_int(pub: PaillierPublicKey, priv: PaillierPrivateKey, c: int) -> int:
    """Decrypt ciphertext to integer in [0, n)."""
    if not (0 <= c < pub.n_sq):
        raise ValueError("Ciphertext out of range")
    u = pow(c, priv.lambda_param, pub.n_sq)
    l_val = (u - 1) // pub.n
    m = (l_val * priv.mu) % pub.n
    return m


# ----------------------------
# Fixed-point helpers
# ----------------------------


def encode_fixed_point(values: Sequence[float], scale: int, n: int) -> List[int]:
    if scale <= 0:
        raise ValueError("scale must be positive integer")
    out: List[int] = []
    for x in values:
        q = int(round(x * scale))
        out.append(q % n)
    return out


def decode_fixed_point(values: Sequence[int], scale: int, n: int) -> List[float]:
    half = n // 2
    out: List[float] = []
    for v in values:
        vv = v if v <= half else v - n
        out.append(vv / float(scale))
    return out


# ----------------------------
# Vector helpers for homomorphic aggregation
# ----------------------------


def encrypt_vector(pub: PaillierPublicKey, vec: Sequence[int]) -> List[int]:
    """Encrypt an integer vector with elements already in [0, n)."""
    return [encrypt_int(pub, int(m)) for m in vec]


def aggregate_cipher_vectors(
    pub: PaillierPublicKey, encrypted_vectors: Sequence[Sequence[int]]
) -> List[int]:
    """Homomorphically sum vectors by element-wise ciphertext multiplication modulo n^2."""
    if not encrypted_vectors:
        return []
    length = len(encrypted_vectors[0])
    for ev in encrypted_vectors:
        if len(ev) != length:
            raise ValueError("All vectors must have the same length")
    agg = [1] * length
    mod = pub.n_sq
    for ev in encrypted_vectors:
        for i in range(length):
            agg[i] = (agg[i] * ev[i]) % mod
    return agg


def decrypt_vector(
    pub: PaillierPublicKey, priv: PaillierPrivateKey, enc_vec: Sequence[int]
) -> List[int]:
    return [decrypt_int(pub, priv, c) for c in enc_vec]


def homomorphic_sum_float_vectors(
    pub: PaillierPublicKey,
    priv: PaillierPrivateKey,
    float_vectors: Sequence[Sequence[float]],
    scale: int = 1_000_000,
) -> List[float]:
    """
    Encode, encrypt, aggregate and decrypt the sum of float vectors.

    Returns the decoded float sum vector.
    """
    if not float_vectors:
        return []
    n = pub.n
    encoded = [encode_fixed_point(vec, scale, n) for vec in float_vectors]
    encrypted = [encrypt_vector(pub, vec) for vec in encoded]
    agg = aggregate_cipher_vectors(pub, encrypted)
    dec_ints = decrypt_vector(pub, priv, agg)
    return decode_fixed_point(dec_ints, scale, n)


# Self-test / demo
if __name__ == "__main__":
    kp = PaillierKeypair.generate(bits=256)
    pub, priv = kp.public_key, kp.private_key
    v1 = [1.25, -2.5, 0.0, 3.14159]
    v2 = [0.75, 2.25, -1.0, -1.14159]
    summed = homomorphic_sum_float_vectors(pub, priv, [v1, v2], scale=1_000_000)
    expected = [a + b for a, b in zip(v1, v2)]
    for i, (s, e) in enumerate(zip(summed, expected)):
        assert abs(s - e) < 1e-6, f"Mismatch at {i}: {s} vs {e}"
    print("Paillier homomorphic sum demo passed.")
