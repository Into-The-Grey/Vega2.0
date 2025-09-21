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
    # Aliases for backward compatibility with tests
    SECP256R1 = "p256"
    SECP384R1 = "p384"
    SECP521R1 = "p521"


@dataclass
class ECCKeyPair:
    """ECC public/private key pair"""

    private_key: str
    public_key: str
    curve: ECCCurve
    key_id: Optional[str] = None


@dataclass
class ECCSignature:
    """ECC signature result"""

    signature: str
    algorithm: str
    curve: ECCCurve
    key_id: Optional[str] = None


@dataclass
class ECCCertificate:
    """ECC certificate result"""

    certificate_pem: str
    key_id: str
    subject: str
    issuer: str
    valid_until: str


class ECCManager:
    """Mock ECC manager for testing"""

    def __init__(self, curve: ECCCurve = ECCCurve.P256):
        self.curve = curve
        self._key_cache: Dict[str, ECCKeyPair] = {}

    def generate_key_pair(
        self, key_id: Optional[str] = None, curve: Optional[ECCCurve] = None
    ) -> ECCKeyPair:
        """Generate a mock ECC key pair"""
        # Use provided curve or default
        selected_curve = curve or self.curve

        if key_id and key_id in self._key_cache:
            return self._key_cache[key_id]

        # Generate unique key_id if not provided
        if not key_id:
            key_id = f"key_{secrets.token_hex(8)}"

        # Mock key generation with deterministic values for testing
        private_key = base64.b64encode(secrets.token_bytes(32)).decode()
        public_key = base64.b64encode(secrets.token_bytes(64)).decode()

        key_pair = ECCKeyPair(
            private_key=private_key,
            public_key=public_key,
            curve=selected_curve,
            key_id=key_id,
        )

        self._key_cache[key_id] = key_pair
        return key_pair

    def get_key_info(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a stored key"""
        if key_id in self._key_cache:
            key_pair = self._key_cache[key_id]
            return {
                "key_id": key_id,
                "curve": key_pair.curve.value,
                "created_at": "2024-01-01T00:00:00Z",  # Mock timestamp
                "public_key": key_pair.public_key,
            }
        return None

    def list_keys(self) -> Dict[str, Dict[str, Any]]:
        """List all stored keys"""
        result = {}
        for key_id in self._key_cache:
            key_info = self.get_key_info(key_id)
            if key_info:
                result[key_id] = key_info
        return result

    def sign_data(self, data: str, key_id: str) -> ECCSignature:
        """Sign data using stored key"""
        if key_id not in self._key_cache:
            raise ValueError(f"Key {key_id} not found")
        
        key_pair = self._key_cache[key_id]
        # Convert string to bytes for hashing
        data_bytes = data.encode('utf-8') if isinstance(data, str) else data
        signature = self._sign_data_with_key(data_bytes, key_pair.private_key)
        signature.key_id = key_id  # Add key_id to signature
        return signature

    def _sign_data_with_key(self, data: bytes, private_key: str) -> ECCSignature:    def _sign_data_with_key(self, data: bytes, private_key: str) -> ECCSignature:
        """Mock sign data with ECC private key"""
        # Mock signing - create deterministic signature for testing
        data_hash = hashlib.sha256(data).hexdigest()
        signature_data = f"{private_key[:8]}:{data_hash[:16]}"
        signature = base64.b64encode(signature_data.encode()).decode()

        return ECCSignature(signature=signature, algorithm="ECDSA", curve=self.curve)

    def verify_signature(self, data: str, signature: ECCSignature) -> bool:
        """Verify signature for string data"""
        # Mock verification - check if signature format is valid
        try:
            decoded = base64.b64decode(signature.signature).decode()
            return ":" in decoded and len(decoded.split(":")) == 2
        except Exception:
            return False

    def encrypt_message(self, message: str, recipient_key_id: str) -> bytes:
        """Encrypt message using recipient's public key"""
        if recipient_key_id not in self._key_cache:
            raise ValueError(f"Key {recipient_key_id} not found")

        key_pair = self._key_cache[recipient_key_id]
        message_bytes = message.encode("utf-8") if isinstance(message, str) else message
        return self._encrypt_message_with_key(message_bytes, key_pair.public_key)

    def _encrypt_message_with_key(self, message: bytes, public_key: str) -> bytes:
        """Mock encrypt message with ECC public key"""
        # Mock encryption - simple XOR with key hash for testing
        key_hash = hashlib.sha256(public_key.encode()).digest()
        encrypted = bytearray()

        for i, byte in enumerate(message):
            encrypted.append(byte ^ key_hash[i % len(key_hash)])

        return bytes(encrypted)

    def decrypt_message(self, encrypted_data: bytes, key_id: str) -> str:
        """Decrypt message using stored private key"""
        if key_id not in self._key_cache:
            raise ValueError(f"Key {key_id} not found")

        key_pair = self._key_cache[key_id]
        decrypted_bytes = self._decrypt_message_with_key(
            encrypted_data, key_pair.private_key
        )
        return decrypted_bytes.decode("utf-8")

    def _decrypt_message_with_key(
        self, encrypted_data: bytes, private_key: str
    ) -> bytes:
        """Mock decrypt message with ECC private key"""
        # Mock decryption - reverse of encryption
        key_hash = hashlib.sha256(private_key.encode()).digest()
        decrypted = bytearray()

        for i, byte in enumerate(encrypted_data):
            decrypted.append(byte ^ key_hash[i % len(key_hash)])

        return bytes(decrypted)

    def generate_certificate(
        self, key_id: str, subject: str, issuer: Optional[str] = None
    ) -> ECCCertificate:
        """Generate a mock certificate for the key"""
        if key_id not in self._key_cache:
            raise ValueError(f"Key {key_id} not found")

        # Mock certificate generation
        cert_pem = f"-----BEGIN CERTIFICATE-----\nMOCK_CERT_{key_id}_{subject}\n-----END CERTIFICATE-----"
        return ECCCertificate(
            certificate_pem=cert_pem,
            key_id=key_id,
            subject=subject,
            issuer=issuer or subject,
            valid_until="2025-12-31T23:59:59Z",
        )

    def verify_certificate(self, certificate_pem: str) -> bool:
        """Verify a certificate"""
        # Mock verification - check basic format
        return certificate_pem.startswith(
            "-----BEGIN CERTIFICATE-----"
        ) and certificate_pem.endswith("-----END CERTIFICATE-----")

    def get_certificate(self, key_id: str) -> Optional[ECCCertificate]:
        """Get certificate for a key (mock - would normally be stored)"""
        if key_id in self._key_cache:
            return self.generate_certificate(key_id, f"CN=Test-{key_id}")
        return None

    def derive_shared_secret(self, private_key_id: str, public_key_id: str) -> str:
        """Mock derive shared secret from stored key IDs"""
        if (
            private_key_id not in self._key_cache
            or public_key_id not in self._key_cache
        ):
            raise ValueError("One or both keys not found")

        private_key = self._key_cache[private_key_id].private_key
        public_key = self._key_cache[public_key_id].public_key
        return self._derive_shared_secret_from_keys(private_key, public_key)

    def _derive_shared_secret_from_keys(self, private_key: str, public_key: str) -> str:
        """Mock derive shared secret from key pair"""
        # Mock ECDH - combine keys to create shared secret
        combined = private_key + public_key
        secret = hashlib.sha256(combined.encode()).hexdigest()
        return base64.b64encode(secret.encode()).decode()


# Mock SecurityManager for tests
class SecurityManager:
    """Mock security manager for testing"""

    def __init__(self):
        self.ecc_manager = get_ecc_manager()

    def generate_secure_api_key(self, key_id: str, permissions: list = None) -> str:
        """Generate a secure API key"""
        # Mock secure API key generation
        key_data = f"{key_id}:{','.join(permissions or [])}"
        return base64.b64encode(key_data.encode()).decode()


# Global ECC manager instance
_ecc_manager = None


def get_ecc_manager(curve: ECCCurve = ECCCurve.P256) -> ECCManager:
    """Get global ECC manager instance"""
    global _ecc_manager
    if _ecc_manager is None or _ecc_manager.curve != curve:
        _ecc_manager = ECCManager(curve)
    return _ecc_manager


def get_security_manager() -> SecurityManager:
    """Get global security manager instance"""
    return SecurityManager()


def generate_key_pair(curve: ECCCurve = ECCCurve.P256) -> ECCKeyPair:
    """Generate ECC key pair using global manager"""
    manager = get_ecc_manager(curve)
    return manager.generate_key_pair()


# Module-level convenience functions for backward compatibility
def sign_data(data: str, private_key: str) -> ECCSignature:
    """Sign data using private key string"""
    manager = get_ecc_manager()
    return manager._sign_data_with_key(data.encode("utf-8"), private_key)


def verify_signature(data: str, signature: ECCSignature) -> bool:
    """Verify signature for data"""
    manager = get_ecc_manager()
    return manager.verify_signature(data, signature)


def encrypt_message(message: str, public_key: str) -> bytes:
    """Encrypt message using public key string"""
    manager = get_ecc_manager()
    return manager._encrypt_message_with_key(message.encode("utf-8"), public_key)


def decrypt_message(encrypted_data: bytes, private_key: str) -> str:
    """Decrypt message using private key string"""
    manager = get_ecc_manager()
    decrypted_bytes = manager._decrypt_message_with_key(encrypted_data, private_key)
    return decrypted_bytes.decode("utf-8")
