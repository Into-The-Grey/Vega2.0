"""
Simple SMPC Secure Sum using Shamir's Secret Sharing over a large prime field.

We encode float vectors as fixed-point integers, split each element into shares,
exchange shares, and reconstruct the aggregate securely from any threshold subset.
This is a demonstration suitable for personal/family-scale setups.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

try:
    # When imported as package
    from vega.federated.threshold_secret_sharing import ShamirSecretSharing
except Exception:
    try:
        # Relative import inside package
        from .threshold_secret_sharing import ShamirSecretSharing
    except Exception:
        # Direct script execution fallback
        from threshold_secret_sharing import ShamirSecretSharing

# Large prime field (>= 2**64 to avoid wrap)
PRIME = 2**127 - 1


def encode_fixed_point(vec: np.ndarray, scale: int = 1_000_000) -> np.ndarray:
    # Use integer arithmetic to avoid float modulo issues with large primes
    ints = [int(np.rint(float(x) * scale)) for x in vec.tolist()]
    return np.array([i % PRIME for i in ints], dtype=object)


def decode_fixed_point(vec: np.ndarray, scale: int = 1_000_000) -> np.ndarray:
    # Map back to [-PRIME/2, PRIME/2) then divide using Python big ints to avoid overflow
    half = PRIME // 2
    arr = vec.astype(object)
    arr = np.where(arr > half, arr - PRIME, arr).astype(object)
    return np.array([float(int(a)) / scale for a in arr.tolist()], dtype=float)


@dataclass
class SMPCConfig:
    threshold: int
    num_participants: int
    scale: int = 1_000_000
    prime: int = PRIME


class SMPCSecureSum:
    def __init__(self, config: SMPCConfig):
        assert 1 < config.threshold <= config.num_participants
        self.config = config
        self.sss = ShamirSecretSharing(
            config.threshold, config.num_participants, config.prime
        )

    def share_vector(self, vec: np.ndarray) -> List[List[Tuple[int, int]]]:
        """
        Split each element into shares; returns per-participant share lists.
        Output shape: [num_participants][len(vec)] of (x,y) shares
        """
        enc = encode_fixed_point(vec, self.config.scale)
        shares_per_participant: List[List[Tuple[int, int]]] = [
            list() for _ in range(self.config.num_participants)
        ]
        for val in enc.tolist():
            # split integer secret
            shares = self.sss.split(int(val))
            for i, share in enumerate(shares):
                shares_per_participant[i].append(share)
        return shares_per_participant

    def add_shared_vectors(
        self,
        shares_a: List[List[Tuple[int, int]]],
        shares_b: List[List[Tuple[int, int]]],
    ) -> List[List[Tuple[int, int]]]:
        """
        Add two shared vectors element-wise in the share space (same x for each position).
        """
        out = []
        for p in range(self.config.num_participants):
            combined = []
            for (x1, y1), (x2, y2) in zip(shares_a[p], shares_b[p]):
                assert x1 == x2
                combined.append((x1, (y1 + y2) % self.config.prime))
            out.append(combined)
        return out

    def reconstruct_sum(
        self, shares_from_subset: List[List[Tuple[int, int]]]
    ) -> np.ndarray:
        """
        Reconstruct the sum vector from shares of any threshold participants.
        Input shape: [threshold][len(vec)] of (x,y)
        """
        length = len(shares_from_subset[0])
        rec_vals = []
        for idx in range(length):
            point_list = [
                participant_shares[idx] for participant_shares in shares_from_subset
            ]
            secret = self.sss.reconstruct(point_list)
            rec_vals.append(secret)
        rec_array = np.array(rec_vals, dtype=object)
        return decode_fixed_point(rec_array, self.config.scale)


# Example test runner
if __name__ == "__main__":
    np.random.seed(0)
    cfg = SMPCConfig(threshold=2, num_participants=3)
    smpc = SMPCSecureSum(cfg)
    v1 = np.random.randn(5)
    v2 = np.random.randn(5)
    # Inspect encoded fixed-point first elements
    enc_v1 = encode_fixed_point(v1, cfg.scale)
    enc_v2 = encode_fixed_point(v2, cfg.scale)
    print("enc_v1 first int:", int(enc_v1[0]))
    print("enc_v2 first int:", int(enc_v2[0]))
    s1 = smpc.share_vector(v1)
    s2 = smpc.share_vector(v2)
    s12 = smpc.add_shared_vectors(s1, s2)
    # Pick any 2 participants to reconstruct
    subset = [s12[0], s12[2]]
    rec = smpc.reconstruct_sum(subset)
    rec_v1 = smpc.reconstruct_sum([s1[0], s1[2]])
    rec_v2 = smpc.reconstruct_sum([s2[0], s2[2]])
    expected = v1 + v2
    print("v1:", v1)
    print("v2:", v2)
    print("expected:", expected)
    print("reconstructed:", rec)
    # print a few shares for first element
    print("shares p0 first elem (x,y):", s1[0][0], s2[0][0], s12[0][0])
    print("shares p2 first elem (x,y):", s1[2][0], s2[2][0], s12[2][0])
    print("rec_v1:", rec_v1)
    print("rec_v2:", rec_v2)
    # Direct reconstruct of v2 first element
    first_secret_v2 = smpc.sss.reconstruct([s2[0][0], s2[2][0]])
    print("v2 first secret int:", first_secret_v2)
    print(
        "v2 first secret decoded:",
        decode_fixed_point(np.array([first_secret_v2], dtype=object))[0],
    )
    assert np.allclose(rec, expected, atol=1e-5)
    print("🎉 SMPC secure sum demo passed!")
