#!/usr/bin/env python3
"""
ecc_crypto.py - ECC cryptographic utilities for tests

Provides ECC cryptographic functions and utilities used by test modules.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
from enum import Enum
import hashlib
import secrets
import base64


class ECCCurve(Enum):
    """Supported ECC curves for testing"""

    P256 = "p256"
    P384 = "p384"
    P521 = "p521"
    SECP256K1 = "secp256k1"


@dataclass
class ECCKeyPair:
    """ECC public/private key pair"""

    private_key: str
    public_key: str
    curve: ECCCurve


@dataclass
class ECCSignature:
    """ECC signature result"""

    signature: str
    algorithm: str
    curve: ECCCurve


class ECCManager:
    """Mock ECC manager for testing"""

    def __init__(self, curve: ECCCurve = ECCCurve.P256):
        self.curve = curve
        self._key_cache: Dict[str, ECCKeyPair] = {}

    def generate_key_pair(self, key_id: Optional[str] = None) -> ECCKeyPair:
        """Generate a mock ECC key pair"""
        if key_id and key_id in self._key_cache:
            return self._key_cache[key_id]

        # Mock key generation with deterministic values for testing
        private_key = base64.b64encode(secrets.token_bytes(32)).decode()
        public_key = base64.b64encode(secrets.token_bytes(64)).decode()

        key_pair = ECCKeyPair(
            private_key=private_key, public_key=public_key, curve=self.curve
        )

        if key_id:
            self._key_cache[key_id] = key_pair

        return key_pair

    def sign_data(self, data: bytes, private_key: str) -> ECCSignature:
        """Mock sign data with ECC private key"""
        # Mock signing - create deterministic signature for testing
        data_hash = hashlib.sha256(data).hexdigest()
        signature_data = f"{private_key[:8]}:{data_hash[:16]}"
        signature = base64.b64encode(signature_data.encode()).decode()

        return ECCSignature(signature=signature, algorithm="ECDSA", curve=self.curve)

    def verify_signature(
        self, data: bytes, signature: ECCSignature, public_key: str
    ) -> bool:
        """Mock verify ECC signature"""
        # Mock verification - check if signature format is valid
        try:
            decoded = base64.b64decode(signature.signature).decode()
            return ":" in decoded and len(decoded.split(":")) == 2
        except Exception:
            return False

    def encrypt_message(self, message: bytes, public_key: str) -> bytes:
        """Mock encrypt message with ECC public key"""
        # Mock encryption - simple XOR with key hash for testing
        key_hash = hashlib.sha256(public_key.encode()).digest()
        encrypted = bytearray()

        for i, byte in enumerate(message):
            encrypted.append(byte ^ key_hash[i % len(key_hash)])

        return bytes(encrypted)

    def decrypt_message(self, encrypted_data: bytes, private_key: str) -> bytes:
        """Mock decrypt message with ECC private key"""
        # Mock decryption - reverse of encryption
        key_hash = hashlib.sha256(private_key.encode()).digest()
        decrypted = bytearray()

        for i, byte in enumerate(encrypted_data):
            decrypted.append(byte ^ key_hash[i % len(key_hash)])

        return bytes(decrypted)

    def derive_shared_secret(self, private_key: str, public_key: str) -> str:
        """Mock derive shared secret from key pair"""
        # Mock ECDH - combine keys to create shared secret
        combined = private_key + public_key
        secret = hashlib.sha256(combined.encode()).hexdigest()
        return base64.b64encode(secret.encode()).decode()


# Global ECC manager instance
_ecc_manager = None


def get_ecc_manager(curve: ECCCurve = ECCCurve.P256) -> ECCManager:
    """Get global ECC manager instance"""
    global _ecc_manager
    if _ecc_manager is None or _ecc_manager.curve != curve:
        _ecc_manager = ECCManager(curve)
    return _ecc_manager


def generate_key_pair(curve: ECCCurve = ECCCurve.P256) -> ECCKeyPair:
    """Generate ECC key pair using global manager"""
    manager = get_ecc_manager(curve)
    return manager.generate_key_pair()


def sign_data(
    data: bytes, private_key: str, curve: ECCCurve = ECCCurve.P256
) -> ECCSignature:
    """Sign data using global ECC manager"""
    manager = get_ecc_manager(curve)
    return manager.sign_data(data, private_key)


def verify_signature(data: bytes, signature: ECCSignature, public_key: str) -> bool:
    """Verify signature using global ECC manager"""
    manager = get_ecc_manager(signature.curve)
    return manager.verify_signature(data, signature, public_key)


def encrypt_message(
    message: bytes, public_key: str, curve: ECCCurve = ECCCurve.P256
) -> bytes:
    """Encrypt message using global ECC manager"""
    manager = get_ecc_manager(curve)
    return manager.encrypt_message(message, public_key)


def decrypt_message(
    encrypted_data: bytes, private_key: str, curve: ECCCurve = ECCCurve.P256
) -> bytes:
    """Decrypt message using global ECC manager"""
    manager = get_ecc_manager(curve)
    return manager.decrypt_message(encrypted_data, private_key)


def derive_shared_secret(
    private_key: str, public_key: str, curve: ECCCurve = ECCCurve.P256
) -> str:
    """Derive shared secret using global ECC manager"""
    manager = get_ecc_manager(curve)
    return manager.derive_shared_secret(private_key, public_key)


def reset_ecc_manager() -> None:
    """Reset global ECC manager"""
    global _ecc_manager
    _ecc_manager = None


# Mock classes for specific test scenarios
class MockECCKeyStore:
    """Mock ECC key store for testing"""

    def __init__(self):
        self.keys: Dict[str, ECCKeyPair] = {}

    def store_key(self, key_id: str, key_pair: ECCKeyPair) -> None:
        """Store key pair"""
        self.keys[key_id] = key_pair

    def get_key(self, key_id: str) -> Optional[ECCKeyPair]:
        """Get key pair by ID"""
        return self.keys.get(key_id)

    def delete_key(self, key_id: str) -> bool:
        """Delete key pair"""
        if key_id in self.keys:
            del self.keys[key_id]
            return True
        return False

    def list_keys(self) -> list[str]:
        """List all key IDs"""
        return list(self.keys.keys())


class MockECCCertificate:
    """Mock ECC certificate for testing"""

    def __init__(self, subject: str, public_key: str, curve: ECCCurve):
        self.subject = subject
        self.public_key = public_key
        self.curve = curve
        self.serial_number = secrets.randbelow(2**64)
        self.is_valid = True

    def verify(self) -> bool:
        """Verify certificate"""
        return self.is_valid

    def get_public_key(self) -> str:
        """Get public key from certificate"""
        return self.public_key
