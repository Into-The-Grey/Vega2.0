#!/usr/bin/env python3
"""
ecc_crypto.py - ECC cryptographic utilities for tests

Provides ECC cryptographic functions and utilities used by test modules.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
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

    @property
    def public_key_pem(self) -> str:
        """Alias for public_key to match test expectations"""
        return self.public_key


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

    @property
    def not_after(self) -> str:
        """Alias for valid_until to match test expectations"""
        return self.valid_until


@dataclass
class SecureAPIKey:
    """Secure API key result"""

    key_id: str
    api_key: str
    ecc_key_id: str
    permissions: List[str]
    expires_in_days: Optional[int] = None
    rate_limit: Optional[int] = None


class ECCManager:
    """Mock ECC manager for testing"""

    def __init__(self, curve: ECCCurve = ECCCurve.P256):
        self.curve = curve
        self._key_cache: Dict[str, ECCKeyPair] = {}
        self._cert_cache: Dict[str, ECCCertificate] = {}

    def generate_key_pair(
        self,
        key_id: Optional[str] = None,
        curve: Optional[ECCCurve] = None,
        expires_in_days: Optional[int] = None,
    ) -> ECCKeyPair:
        """Generate a mock ECC key pair"""
        # Use provided curve or default
        selected_curve = curve or self.curve

        # Validate curve
        if isinstance(selected_curve, str):
            # Try to find matching curve value
            for c in ECCCurve:
                if c.value == selected_curve:
                    selected_curve = c
                    break
            else:
                raise ValueError(f"Invalid curve: {selected_curve}")

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
                "curve": key_pair.curve,  # Return enum object, not value
                "created_at": "2024-01-01T00:00:00Z",  # Mock timestamp
                "public_key": key_pair.public_key,
                "has_private_key": True,  # Mock - always true for our test keys
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
        data_bytes = data.encode("utf-8") if isinstance(data, str) else data
        signature = self._sign_data_with_key(data_bytes, key_pair.private_key)
        signature.key_id = key_id  # Add key_id to signature
        return signature

    def _sign_data_with_key(self, data: bytes, private_key: str) -> ECCSignature:
        """Mock sign data with ECC private key"""
        # Mock signing - create deterministic signature for testing
        data_hash = hashlib.sha256(data).hexdigest()
        signature_data = f"{private_key[:8]}:{data_hash[:16]}"
        signature = base64.b64encode(signature_data.encode()).decode()

        return ECCSignature(signature=signature, algorithm="ECDSA", curve=self.curve)

    def verify_signature(self, data: str, signature: ECCSignature) -> bool:
        """Verify signature for string data"""
        # Mock verification - check if signature format is valid AND matches data
        try:
            decoded = base64.b64decode(signature.signature).decode()
            if ":" not in decoded or len(decoded.split(":")) != 2:
                return False

            # Extract the data hash from signature and compare with actual data hash
            _, signature_hash = decoded.split(":")
            data_bytes = data.encode("utf-8") if isinstance(data, str) else data
            actual_hash = hashlib.sha256(data_bytes).hexdigest()[:16]

            return signature_hash == actual_hash
        except Exception:
            return False

    def encrypt_message(
        self,
        message: str,
        recipient_public_key: str,
        sender_key_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Encrypt message using recipient's public key"""
        if sender_key_id is None:
            # Fall back to simple key-based encryption if no sender key
            message_bytes = (
                message.encode("utf-8") if isinstance(message, str) else message
            )
            ciphertext = self._encrypt_message_with_key(
                message_bytes, recipient_public_key
            )
        else:
            # Use ECIES-style encryption with shared secret
            if sender_key_id not in self._key_cache:
                raise ValueError(f"Sender key {sender_key_id} not found")

            message_bytes = (
                message.encode("utf-8") if isinstance(message, str) else message
            )

            # Derive shared secret between sender private and recipient public
            shared_secret = self.derive_shared_secret(
                sender_key_id, recipient_public_key
            )
            ciphertext = self._encrypt_with_shared_secret(message_bytes, shared_secret)

        return {
            "algorithm": "ECIES",
            "ciphertext": ciphertext,
            "sender_key_id": sender_key_id,
        }

    def _encrypt_message_with_key(self, message: bytes, public_key: str) -> bytes:
        """Mock encrypt message with ECC public key"""
        # Mock encryption using XOR cipher (preserves byte patterns perfectly)
        key_hash = hashlib.sha256(public_key.encode()).digest()

        # XOR encryption preserves all byte values and is reversible
        encrypted = bytearray()
        for i, byte in enumerate(message):
            key_byte = key_hash[i % len(key_hash)]
            encrypted_byte = byte ^ key_byte
            encrypted.append(encrypted_byte)

        # Encode to base64 for storage/transmission
        return base64.b64encode(bytes(encrypted))

    def _encrypt_with_shared_secret(
        self, message: bytes, shared_secret: bytes
    ) -> bytes:
        """Encrypt message using shared secret (XOR cipher)"""
        # Use the shared secret as encryption key
        encrypted = bytearray()
        for i, byte in enumerate(message):
            key_byte = shared_secret[i % len(shared_secret)]
            encrypted_byte = byte ^ key_byte
            encrypted.append(encrypted_byte)

        # Encode to base64 for storage/transmission
        return base64.b64encode(bytes(encrypted))

    def decrypt_message(self, encrypted_data: Dict[str, Any], key_id: str) -> bytes:
        """Decrypt message using stored private key"""
        if key_id not in self._key_cache:
            raise ValueError(f"Key {key_id} not found")

        key_pair = self._key_cache[key_id]

        # Extract ciphertext and sender info from dict
        if isinstance(encrypted_data, dict):
            ciphertext = encrypted_data["ciphertext"]
            sender_key_id = encrypted_data.get("sender_key_id")
        else:
            ciphertext = encrypted_data
            sender_key_id = None

        # If we have sender key info, use ECIES-style decryption with shared secret
        if sender_key_id and sender_key_id in self._key_cache:
            # Get sender's public key
            sender_public_key = self._key_cache[sender_key_id].public_key_pem
            # Derive the same shared secret that was used for encryption
            shared_secret = self.derive_shared_secret(key_id, sender_public_key)
            return self._decrypt_with_shared_secret(ciphertext, shared_secret)
        else:
            # Fall back to simple key-based decryption
            return self._decrypt_message_with_key(ciphertext, key_pair.private_key)

    def _decrypt_message_with_key(
        self, encrypted_data: bytes, private_key: str
    ) -> bytes:
        """Mock decrypt message with ECC private key"""
        # Mock decryption - reverse of XOR encryption
        # For ECC, we derive the decryption key from the private key
        key_hash = hashlib.sha256(private_key.encode()).digest()

        # First decode from base64
        try:
            decoded_data = base64.b64decode(encrypted_data)
        except Exception:
            # Fallback for raw bytes
            decoded_data = encrypted_data

        # XOR decryption (XOR is its own inverse: A ^ B ^ B = A)
        decrypted = bytearray()
        for i, byte in enumerate(decoded_data):
            key_byte = key_hash[i % len(key_hash)]
            decrypted_byte = byte ^ key_byte
            decrypted.append(decrypted_byte)

        return bytes(decrypted)

    def _decrypt_with_shared_secret(
        self, encrypted_data: bytes, shared_secret: bytes
    ) -> bytes:
        """Decrypt message using shared secret (XOR cipher)"""
        # First decode from base64
        try:
            decoded_data = base64.b64decode(encrypted_data)
        except Exception:
            # Fallback for raw bytes
            decoded_data = encrypted_data

        # XOR decryption using shared secret (XOR is its own inverse)
        decrypted = bytearray()
        for i, byte in enumerate(decoded_data):
            key_byte = shared_secret[i % len(shared_secret)]
            decrypted_byte = byte ^ key_byte
            decrypted.append(decrypted_byte)

        return bytes(decrypted)

    def generate_certificate(
        self,
        key_id: str,
        subject: str,
        issuer: Optional[str] = None,
        validity_days: Optional[int] = None,
    ) -> ECCCertificate:
        """Generate a mock certificate for the key"""
        if key_id not in self._key_cache:
            raise ValueError(f"Key {key_id} not found")

        # Mock certificate generation
        cert_pem = f"-----BEGIN CERTIFICATE-----\nMOCK_CERT_{key_id}_{subject}\n-----END CERTIFICATE-----"
        certificate = ECCCertificate(
            certificate_pem=cert_pem,
            key_id=key_id,
            subject=subject,
            issuer=issuer or subject,
            valid_until="2025-12-31T23:59:59Z",
        )

        # Store certificate for later retrieval
        self._cert_cache[key_id] = certificate
        return certificate

    def verify_certificate(self, certificate_pem: str) -> bool:
        """Verify a certificate"""
        # Mock verification - check basic format
        return certificate_pem.startswith(
            "-----BEGIN CERTIFICATE-----"
        ) and certificate_pem.endswith("-----END CERTIFICATE-----")

    def get_certificate(self, key_id: str) -> Optional[ECCCertificate]:
        """Get certificate for a key (returns stored certificate if available)"""
        if key_id in self._cert_cache:
            return self._cert_cache[key_id]
        elif key_id in self._key_cache:
            # Generate default certificate if none stored
            return self.generate_certificate(key_id, f"CN=Test-{key_id}")
        return None

    def derive_shared_secret(self, private_key_id: str, public_key: str) -> bytes:
        """Mock derive shared secret from private key ID and public key"""
        if private_key_id not in self._key_cache:
            raise ValueError(f"Key {private_key_id} not found")

        # Get my key pair info
        my_key_pair = self._key_cache[private_key_id]
        my_public_key = my_key_pair.public_key_pem

        # For ECDH simulation, we need Alice_private+Bob_public = Bob_private+Alice_public
        #
        # The insight: create a shared secret that depends on the combination of
        # BOTH public keys involved, regardless of which private key is being used
        #
        # This way:
        # - Alice calling with Bob's public key will combine Alice_public + Bob_public
        # - Bob calling with Alice's public key will combine Bob_public + Alice_public
        # - Since we sort them, both get the same combination!

        # Get both public keys involved
        key1 = my_public_key
        key2 = public_key

        # Sort the public keys to ensure deterministic order
        sorted_keys = sorted([key1, key2])

        # Create shared secret from the sorted combination
        combined_input = "|".join(sorted_keys)
        secret = hashlib.sha256(
            combined_input.encode()
        ).digest()  # Return bytes, not hex
        return secret

    def _derive_shared_secret_from_keys(self, private_key: str, public_key: str) -> str:
        """Mock derive shared secret from key pair"""
        # Mock ECDH - create deterministic shared secret
        # In real ECDH: Alice_private * Bob_public = Bob_private * Alice_public
        # We need to simulate this by finding a way to combine the keys that gives
        # the same result regardless of which private/public pair is used

        # Strategy: Extract key "fingerprints" and combine them in a commutative way
        # The insight is that we need to find some shared mathematical property

        # Get deterministic fingerprints for both keys
        private_fingerprint = self._get_key_fingerprint(private_key, "private")
        public_fingerprint = self._get_key_fingerprint(public_key, "public")

        # Combine fingerprints in a commutative way
        # Sort the fingerprints to ensure order independence
        fingerprints = sorted([private_fingerprint, public_fingerprint])
        combined = "".join(fingerprints)

        # Generate the shared secret
        secret = hashlib.sha256(combined.encode()).hexdigest()
        return base64.b64encode(secret.encode()).decode()

    def _get_key_fingerprint(self, key: str, key_type: str) -> str:
        """Extract a deterministic fingerprint from a key"""
        # Use a portion of the key hash as fingerprint
        # The fingerprint should be the same for keys that "belong together"
        full_hash = hashlib.sha256(f"{key_type}:{key}".encode()).hexdigest()
        # Take a portion that could be "shared" between related keys
        return full_hash[8:24]  # Use middle portion as fingerprint

    def export_public_key(self, key_id: str) -> str:
        """Export public key in PEM format"""
        if key_id not in self._key_cache:
            raise ValueError(f"Key {key_id} not found")

        key_pair = self._key_cache[key_id]
        # Mock PEM export
        return f"-----BEGIN PUBLIC KEY-----\n{key_pair.public_key}\n-----END PUBLIC KEY-----"

    def import_public_key(self, public_key_pem: str, key_id: str) -> str:
        """Import a public key from PEM format"""
        # Extract public key from PEM format
        if not public_key_pem.startswith("-----BEGIN PUBLIC KEY-----"):
            raise ValueError("Invalid PEM format")

        # Extract the base64 part
        lines = public_key_pem.strip().split("\n")
        if len(lines) < 3:
            raise ValueError("Invalid PEM format")

        public_key_data = lines[1]  # The base64 encoded key

        # Create a new key pair entry with just the public key
        # For mock purposes, we'll create a dummy private key
        dummy_private = base64.b64encode(b"imported_key_no_private").decode()

        key_pair = ECCKeyPair(
            private_key=dummy_private,
            public_key=public_key_data,
            curve=self.curve,
            key_id=key_id,
        )

        self._key_cache[key_id] = key_pair
        return key_id

    def delete_key(self, key_id: str) -> bool:
        """Delete a key from the manager"""
        if key_id in self._key_cache:
            del self._key_cache[key_id]
            # Also remove associated certificate if exists
            if key_id in self._cert_cache:
                del self._cert_cache[key_id]
            return True
        return False


# Mock SecurityManager for tests
class SecurityManager:
    """Mock security manager for testing"""

    def __init__(self):
        self.ecc_manager = get_ecc_manager()

    def generate_secure_api_key(
        self,
        permissions: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
        rate_limit: Optional[int] = None,
    ) -> SecureAPIKey:
        """Generate a secure API key"""
        # Generate an ECC key for the API key
        ecc_key = self.ecc_manager.generate_key_pair()

        # Mock secure API key generation
        perms = permissions or []
        key_id = f"api_key_{secrets.token_hex(8)}"
        api_key = base64.b64encode(f"{key_id}:{','.join(perms)}".encode()).decode()

        return SecureAPIKey(
            key_id=key_id,
            api_key=api_key,
            ecc_key_id=ecc_key.key_id or "unknown",
            permissions=perms,
            expires_in_days=expires_in_days,
            rate_limit=rate_limit,
        )


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
