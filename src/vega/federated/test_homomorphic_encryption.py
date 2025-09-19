import math
import itertools

from homomorphic_encryption import (
    PaillierKeypair,
    homomorphic_sum_float_vectors,
    encode_fixed_point,
    decode_fixed_point,
    encrypt_int,
    decrypt_int,
)


def test_round_trip_ints():
    kp = PaillierKeypair.generate(bits=256)
    pub, priv = kp.public_key, kp.private_key

    for m in [0, 1, 2, pub.n // 2, pub.n - 1]:
        c = encrypt_int(pub, m)
        d = decrypt_int(pub, priv, c)
        assert d == m


def test_fixed_point_encode_decode():
    kp = PaillierKeypair.generate(bits=256)
    n = kp.public_key.n
    scale = 1_000_000
    vals = [0.0, 1.234567, -2.5, 3.0, -0.000001]
    enc = encode_fixed_point(vals, scale, n)
    dec = decode_fixed_point(enc, scale, n)
    for a, b in zip(vals, dec):
        assert abs(a - b) < 1e-6


def test_homomorphic_sum_vectors_small():
    kp = PaillierKeypair.generate(bits=256)
    pub, priv = kp.public_key, kp.private_key
    vectors = [
        [0.5, -1.25, 3.0],
        [1.25, 1.25, -0.5],
        [-0.75, 0.0, 0.25],
    ]
    result = homomorphic_sum_float_vectors(pub, priv, vectors, scale=1_000_000)
    expected = [sum(col) for col in zip(*vectors)]
    for r, e in zip(result, expected):
        assert abs(r - e) < 1e-6


def test_homomorphic_sum_edge_cases():
    kp = PaillierKeypair.generate(bits=256)
    pub, priv = kp.public_key, kp.private_key

    # empty
    assert homomorphic_sum_float_vectors(pub, priv, [], scale=1_000_000) == []

    # zeros and negatives
    vectors = [
        [0.0, 0.0, 0.0],
        [-1.0, 2.0, -3.0],
    ]
    result = homomorphic_sum_float_vectors(pub, priv, vectors, scale=1_000_000)
    expected = [sum(col) for col in zip(*vectors)]
    for r, e in zip(result, expected):
        assert abs(r - e) < 1e-6


if __name__ == "__main__":
    test_round_trip_ints()
    test_fixed_point_encode_decode()
    test_homomorphic_sum_vectors_small()
    test_homomorphic_sum_edge_cases()
    print("All homomorphic encryption tests passed.")
