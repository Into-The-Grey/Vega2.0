"""
Simple Diffie-Hellman Key Exchange for Federated Learning Participants

Implements a basic DH key exchange suitable for local/family-scale secure aggregation.
"""

import secrets


class DiffieHellmanParticipant:
    def __init__(self, p: int, g: int):
        self.p = p
        self.g = g
        self.private = secrets.randbelow(p - 2) + 1  # 1 <= private < p-1
        self.public = pow(g, self.private, p)

    def compute_shared_key(self, other_public: int) -> int:
        return pow(other_public, self.private, self.p)


# Example usage and test
if __name__ == "__main__":
    # Use a small safe prime for demonstration (not secure for production)
    p = 0xE95E4A5F737059DC60DF5991D45029409E60FC09
    g = 2
    alice = DiffieHellmanParticipant(p, g)
    bob = DiffieHellmanParticipant(p, g)
    alice_shared = alice.compute_shared_key(bob.public)
    bob_shared = bob.compute_shared_key(alice.public)
    print(f"Alice's shared key: {alice_shared}")
    print(f"Bob's shared key: {bob_shared}")
    assert alice_shared == bob_shared
    print("🎉 Diffie-Hellman key exchange test passed!")
