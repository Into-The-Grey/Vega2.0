#!/usr/bin/env python3
"""
ECC Cryptography System Test Script
==================================

Test script to validate the ECC cryptography system functionality.
"""

import sys
import asyncio
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from .ecc_crypto import (
    get_ecc_manager,
    ECCCurve,
    generate_key_pair,
    sign_data,
    verify_signature,
    encrypt_message,
    decrypt_message,
)
from .api_security import get_security_manager


def test_key_generation():
    """Test ECC key generation"""
    print("🔑 Testing ECC key generation...")

    ecc_manager = get_ecc_manager()

    # Test different curves
    curves = [ECCCurve.SECP256R1, ECCCurve.SECP384R1, ECCCurve.SECP521R1]

    for curve in curves:
        key_pair = ecc_manager.generate_key_pair(curve=curve)
        print(f"✅ Generated {curve} key: {key_pair.key_id}")

        # Verify key info
        key_info = ecc_manager.get_key_info(key_pair.key_id)
        assert key_info is not None
        assert key_info["curve"] == curve
        assert key_info["has_private_key"] is True
        print(f"   Key info: {key_info}")

    print(f"✅ Generated {len(curves)} key pairs")


def test_digital_signatures():
    """Test ECC digital signatures"""
    print("\n📝 Testing ECC digital signatures...")

    ecc_manager = get_ecc_manager()

    # Generate key for signing
    key_pair = ecc_manager.generate_key_pair()
    print(f"✅ Generated signing key: {key_pair.key_id}")

    # Test data
    test_messages = [
        "Hello, World!",
        "This is a test message with special chars: !@#$%^&*()",
        json.dumps({"test": "data", "number": 42, "array": [1, 2, 3]}),
        "A" * 1000,  # Large message
    ]

    for i, message in enumerate(test_messages):
        # Sign message
        signature = ecc_manager.sign_data(message, key_pair.key_id)
        print(f"✅ Signed message {i+1}: {signature.key_id}")

        # Verify signature
        valid = ecc_manager.verify_signature(message, signature)
        assert valid is True
        print(f"   Signature verified: {valid}")

        # Test invalid signature
        modified_message = message + " MODIFIED"
        invalid = ecc_manager.verify_signature(modified_message, signature)
        assert invalid is False
        print(f"   Modified message rejected: {not invalid}")

    print(f"✅ Tested {len(test_messages)} digital signatures")


def test_key_exchange():
    """Test ECDH key exchange"""
    print("\n🔄 Testing ECDH key exchange...")

    ecc_manager = get_ecc_manager()

    # Generate two key pairs (Alice and Bob)
    alice_key = ecc_manager.generate_key_pair()
    bob_key = ecc_manager.generate_key_pair()

    print(f"✅ Generated Alice's key: {alice_key.key_id}")
    print(f"✅ Generated Bob's key: {bob_key.key_id}")

    # Alice derives shared secret using Bob's public key
    alice_shared = ecc_manager.derive_shared_secret(
        alice_key.key_id, bob_key.public_key_pem
    )

    # Bob derives shared secret using Alice's public key
    bob_shared = ecc_manager.derive_shared_secret(
        bob_key.key_id, alice_key.public_key_pem
    )

    # Shared secrets should match
    assert alice_shared == bob_shared
    print(f"✅ Shared secrets match: {len(alice_shared)} bytes")
    print(f"   Shared secret hash: {alice_shared.hex()[:32]}...")


def test_message_encryption():
    """Test ECC message encryption/decryption"""
    print("\n🔒 Testing ECC message encryption...")

    ecc_manager = get_ecc_manager()

    # Generate key pairs for encryption
    sender_key = ecc_manager.generate_key_pair()
    recipient_key = ecc_manager.generate_key_pair()

    print(f"✅ Generated sender key: {sender_key.key_id}")
    print(f"✅ Generated recipient key: {recipient_key.key_id}")

    # Test messages
    test_messages = [
        "Secret message!",
        json.dumps({"secret": "data", "confidential": True}),
        "🔐 Unicode test with emojis 🚀",
        "X" * 500,  # Larger message
    ]

    for i, message in enumerate(test_messages):
        print(f"Testing message {i+1}: {message[:50]}...")

        # Encrypt message
        encrypted = ecc_manager.encrypt_message(
            message, recipient_key.public_key_pem, sender_key.key_id
        )

        print(f"✅ Encrypted message {i+1}")
        print(f"   Algorithm: {encrypted['algorithm']}")
        print(f"   Ciphertext length: {len(encrypted['ciphertext'])}")

        # Decrypt message
        decrypted = ecc_manager.decrypt_message(encrypted, recipient_key.key_id)
        decrypted_str = decrypted.decode("utf-8")

        assert decrypted_str == message
        print(f"✅ Decrypted message {i+1}: {decrypted_str[:50]}...")

    print(f"✅ Tested {len(test_messages)} encrypted messages")


def test_certificates():
    """Test ECC certificate generation and verification"""
    print("\n📜 Testing ECC certificates...")

    ecc_manager = get_ecc_manager()

    # Generate key for certificate
    key_pair = ecc_manager.generate_key_pair()
    print(f"✅ Generated key for certificate: {key_pair.key_id}")

    # Generate self-signed certificate
    certificate = ecc_manager.generate_certificate(
        key_pair.key_id, "vega.local", validity_days=30
    )

    print(f"✅ Generated certificate")
    print(f"   Subject: {certificate.subject}")
    print(f"   Issuer: {certificate.issuer}")
    print(f"   Valid until: {certificate.not_after}")

    # Verify certificate
    is_valid = ecc_manager.verify_certificate(certificate.certificate_pem)
    assert is_valid is True
    print(f"✅ Certificate verified: {is_valid}")

    # Test certificate retrieval
    retrieved_cert = ecc_manager.get_certificate(key_pair.key_id)
    assert retrieved_cert is not None
    assert retrieved_cert.subject == certificate.subject
    print(f"✅ Certificate retrieval: {retrieved_cert.subject}")


def test_api_security():
    """Test API security with ECC"""
    print("\n🛡️ Testing API security with ECC...")

    security_manager = get_security_manager()

    # Generate secure API key
    secure_key = security_manager.generate_secure_api_key(
        permissions=["read", "write"], expires_in_days=30, rate_limit=100
    )

    print(f"✅ Generated secure API key: {secure_key.key_id}")
    print(f"   API Key: {secure_key.api_key[:20]}...")
    print(f"   ECC Key: {secure_key.ecc_key_id}")
    print(f"   Permissions: {secure_key.permissions}")

    # Validate API key
    validated_key = security_manager.validate_api_key(secure_key.api_key)
    assert validated_key.key_id == secure_key.key_id
    print(f"✅ API key validation successful")

    # Test rate limiting
    rate_limit_ok = security_manager.check_rate_limit(secure_key.api_key)
    assert rate_limit_ok is True
    print(f"✅ Rate limit check: {rate_limit_ok}")

    # Test permissions
    has_read = validated_key.has_permission("read")
    has_write = validated_key.has_permission("write")
    has_admin = validated_key.has_permission("admin")

    assert has_read is True
    assert has_write is True
    assert has_admin is False

    print(
        f"✅ Permission checks: read={has_read}, write={has_write}, admin={has_admin}"
    )

    # Test secure token generation
    ecc_manager = get_ecc_manager()
    token_key = ecc_manager.generate_key_pair()

    token = security_manager.generate_secure_token(
        {"user_id": "test_user", "role": "user"},
        token_key.key_id,
        expires_in_minutes=30,
    )

    print(f"✅ Generated secure token: {token[:50]}...")

    # Verify token
    token_payload = security_manager.verify_secure_token(token)
    assert token_payload is not None
    assert token_payload["user_id"] == "test_user"
    print(f"✅ Token verification: {token_payload}")


def test_key_management():
    """Test key management operations"""
    print("\n🗂️ Testing key management...")

    ecc_manager = get_ecc_manager()

    # List initial keys
    initial_keys = ecc_manager.list_keys()
    initial_count = len(initial_keys)
    print(f"✅ Initial key count: {initial_count}")

    # Generate test keys
    test_keys = []
    for i in range(3):
        key_pair = ecc_manager.generate_key_pair(
            key_id=f"test_key_{i}", expires_in_days=1
        )
        test_keys.append(key_pair)

    # List keys after generation
    updated_keys = ecc_manager.list_keys()
    assert len(updated_keys) == initial_count + 3
    print(f"✅ Key count after generation: {len(updated_keys)}")

    # Test key export/import
    public_key_pem = ecc_manager.export_public_key(test_keys[0].key_id)
    assert public_key_pem is not None
    print(f"✅ Exported public key")

    imported_key_id = ecc_manager.import_public_key(public_key_pem, "imported_test_key")
    assert imported_key_id is not None
    print(f"✅ Imported public key: {imported_key_id}")

    # Delete test keys
    for key_pair in test_keys:
        success = ecc_manager.delete_key(key_pair.key_id)
        assert success is True

    # Clean up imported key
    ecc_manager.delete_key(imported_key_id)

    # Verify final count
    final_keys = ecc_manager.list_keys()
    assert len(final_keys) == initial_count
    print(f"✅ Final key count: {len(final_keys)}")


def test_error_handling():
    """Test error handling"""
    print("\n❌ Testing error handling...")

    ecc_manager = get_ecc_manager()

    # Test invalid key ID
    try:
        ecc_manager.sign_data("test", "invalid_key_id")
        assert False, "Should have raised exception"
    except ValueError as e:
        print(f"✅ Caught expected error: {e}")

    # Test invalid curve
    try:
        ecc_manager.generate_key_pair(curve="invalid_curve")
        assert False, "Should have raised exception"
    except ValueError as e:
        print(f"✅ Caught expected error: {e}")

    # Test invalid signature verification
    key_pair = ecc_manager.generate_key_pair()
    signature = ecc_manager.sign_data("original", key_pair.key_id)

    valid = ecc_manager.verify_signature("modified", signature)
    assert valid is False
    print(f"✅ Invalid signature correctly rejected")

    # Cleanup
    ecc_manager.delete_key(key_pair.key_id)


async def main():
    """Run all ECC tests"""
    print("🚀 Starting Vega2.0 ECC Cryptography System Tests\n")

    try:
        test_key_generation()
        test_digital_signatures()
        test_key_exchange()
        test_message_encryption()
        test_certificates()
        test_api_security()
        test_key_management()
        test_error_handling()

        print("\n✅ All ECC cryptography tests completed successfully!")
        print("🔐 ECC system is ready for production use")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
