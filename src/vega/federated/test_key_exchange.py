"""
Unit tests for Diffie-Hellman key exchange.
"""

from key_exchange import DiffieHellmanParticipant


def test_dh_basic():
    p = 0xE95E4A5F737059DC60DF5991D45029409E60FC09
    g = 2
    alice = DiffieHellmanParticipant(p, g)
    bob = DiffieHellmanParticipant(p, g)
    alice_shared = alice.compute_shared_key(bob.public)
    bob_shared = bob.compute_shared_key(alice.public)
    assert alice_shared == bob_shared
    print("Basic DH key exchange succeeded.")


def test_dh_multiple_participants():
    p = 0xE95E4A5F737059DC60DF5991D45029409E60FC09
    g = 2
    participants = [DiffieHellmanParticipant(p, g) for _ in range(4)]
    # All pairwise shared keys should be consistent
    for i in range(len(participants)):
        for j in range(i + 1, len(participants)):
            k1 = participants[i].compute_shared_key(participants[j].public)
            k2 = participants[j].compute_shared_key(participants[i].public)
            assert k1 == k2
    print("Multi-participant DH key exchange succeeded.")


if __name__ == "__main__":
    test_dh_basic()
    test_dh_multiple_participants()
    print("All Diffie-Hellman key exchange tests passed.")
