"""
Dynamic Rotating Encryption System

Security baseline template for all Vega 2.0 modules. Implements self-rekeying
cryptographic protection suitable for trusted family environments.

Design Principles:
- Dynamic key rotation for enhanced security
- Trusted environment model (warnings vs blocking)
- Modular design for reuse across all Vega components
- Local-first operation on Ubuntu rack server
- Performance-optimized for small scale (2-3 participants)
"""

import secrets
import hashlib
import hmac
import time
import threading
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class EncryptionKey:
    """Represents an encryption key with metadata."""

    key: bytes
    created_at: float
    key_id: str
    salt: bytes

    def is_expired(self, max_age_seconds: int = 3600) -> bool:
        """Check if the key has expired."""
        return (time.time() - self.created_at) > max_age_seconds

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for storage/transmission."""
        return {
            "key": base64.b64encode(self.key).decode(),
            "created_at": str(self.created_at),
            "key_id": self.key_id,
            "salt": base64.b64encode(self.salt).decode(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "EncryptionKey":
        """Create from dictionary."""
        return cls(
            key=base64.b64decode(data["key"]),
            created_at=float(data["created_at"]),
            key_id=data["key_id"],
            salt=base64.b64decode(data["salt"]),
        )


class DynamicEncryption:
    """
    Dynamic rotating encryption system for Vega 2.0.

    Features:
    - Automatic key rotation
    - Multiple active keys for seamless transitions
    - HMAC authentication
    - Salt-based key derivation
    - Thread-safe operations
    """

    def __init__(
        self,
        master_password: Optional[str] = None,
        key_rotation_seconds: int = 3600,  # 1 hour
        max_keys: int = 3,
    ):
        """
        Initialize dynamic encryption system.

        Args:
            master_password: Master password for key derivation (auto-generated if None)
            key_rotation_seconds: How often to rotate keys
            max_keys: Maximum number of active keys to maintain
        """
        self.key_rotation_seconds = key_rotation_seconds
        self.max_keys = max_keys
        self._lock = threading.RLock()
        self._keys: Dict[str, EncryptionKey] = {}
        self._current_key_id: Optional[str] = None

        # Generate or use master password
        self.master_password = master_password or self._generate_master_password()

        # Initialize with first key
        self._rotate_key()

        # Start background key rotation
        self._rotation_thread = threading.Thread(
            target=self._background_rotation, daemon=True
        )
        self._rotation_thread.start()

        logger.info("Dynamic encryption system initialized")

    def _generate_master_password(self) -> str:
        """Generate a secure master password."""
        return base64.b64encode(secrets.token_bytes(32)).decode()

    def _derive_key(self, salt: bytes) -> bytes:
        """Derive encryption key from master password and salt."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(self.master_password.encode())

    def _generate_key_id(self) -> str:
        """Generate a unique key ID."""
        return base64.b64encode(secrets.token_bytes(8)).decode()

    def _rotate_key(self) -> str:
        """Generate and activate a new encryption key."""
        with self._lock:
            # Generate new key
            salt = secrets.token_bytes(16)
            key_material = self._derive_key(salt)
            key_id = self._generate_key_id()

            new_key = EncryptionKey(
                key=key_material, created_at=time.time(), key_id=key_id, salt=salt
            )

            # Add to key store
            self._keys[key_id] = new_key
            self._current_key_id = key_id

            # Clean up old keys
            self._cleanup_old_keys()

            logger.info(f"Rotated to new encryption key: {key_id}")
            return key_id

    def _cleanup_old_keys(self):
        """Remove excess old keys."""
        if len(self._keys) <= self.max_keys:
            return

        # Sort by creation time and keep most recent
        sorted_keys = sorted(
            self._keys.items(), key=lambda x: x[1].created_at, reverse=True
        )

        keys_to_keep = dict(sorted_keys[: self.max_keys])
        removed_count = len(self._keys) - len(keys_to_keep)

        self._keys = keys_to_keep

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old encryption keys")

    def _background_rotation(self):
        """Background thread for automatic key rotation."""
        while True:
            time.sleep(self.key_rotation_seconds)
            try:
                self._rotate_key()
            except Exception as e:
                logger.error(f"Error in background key rotation: {e}")

    def encrypt(self, data: str, key_id: Optional[str] = None) -> Dict[str, str]:
        """
        Encrypt data with specified or current key.

        Args:
            data: String data to encrypt
            key_id: Specific key to use (current key if None)

        Returns:
            Dictionary with encrypted data and metadata
        """
        with self._lock:
            # Use specified key or current key
            if key_id is None:
                key_id = self._current_key_id

            if key_id not in self._keys:
                raise ValueError(f"Key {key_id} not found")

            encryption_key = self._keys[key_id]

            # Create Fernet cipher
            fernet_key = base64.urlsafe_b64encode(encryption_key.key)
            fernet = Fernet(fernet_key)

            # Encrypt data
            encrypted_data = fernet.encrypt(data.encode())

            # Create HMAC for authentication
            hmac_key = hashlib.sha256(encryption_key.key + b"hmac").digest()
            message_hmac = hmac.new(
                hmac_key, encrypted_data, hashlib.sha256
            ).hexdigest()

            return {
                "encrypted_data": base64.b64encode(encrypted_data).decode(),
                "key_id": key_id,
                "hmac": message_hmac,
                "timestamp": str(time.time()),
            }

    def decrypt(self, encrypted_package: Dict[str, str]) -> str:
        """
        Decrypt data from encrypted package.

        Args:
            encrypted_package: Dictionary with encrypted data and metadata

        Returns:
            Decrypted string data
        """
        with self._lock:
            key_id = encrypted_package["key_id"]

            if key_id not in self._keys:
                raise ValueError(f"Key {key_id} not found or expired")

            encryption_key = self._keys[key_id]

            # Decode encrypted data
            encrypted_data = base64.b64decode(encrypted_package["encrypted_data"])

            # Verify HMAC
            hmac_key = hashlib.sha256(encryption_key.key + b"hmac").digest()
            expected_hmac = hmac.new(
                hmac_key, encrypted_data, hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(expected_hmac, encrypted_package["hmac"]):
                raise ValueError("HMAC verification failed - data may be tampered")

            # Create Fernet cipher and decrypt
            fernet_key = base64.urlsafe_b64encode(encryption_key.key)
            fernet = Fernet(fernet_key)

            decrypted_data = fernet.decrypt(encrypted_data)
            return decrypted_data.decode()

    def encrypt_json(self, data: Any, key_id: Optional[str] = None) -> Dict[str, str]:
        """Encrypt JSON-serializable data."""
        json_str = json.dumps(data, sort_keys=True)
        return self.encrypt(json_str, key_id)

    def decrypt_json(self, encrypted_package: Dict[str, str]) -> Any:
        """Decrypt and parse JSON data."""
        json_str = self.decrypt(encrypted_package)
        return json.loads(json_str)

    def get_key_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all active keys."""
        with self._lock:
            info = {}
            for key_id, key in self._keys.items():
                info[key_id] = {
                    "created_at": key.created_at,
                    "age_seconds": time.time() - key.created_at,
                    "is_current": key_id == self._current_key_id,
                    "is_expired": key.is_expired(self.key_rotation_seconds),
                }
            return info

    def force_rotation(self) -> str:
        """Force immediate key rotation."""
        return self._rotate_key()

    def export_keys(self, include_current: bool = True) -> Dict[str, Dict[str, str]]:
        """
        Export keys for backup or sharing with trusted participants.

        Args:
            include_current: Whether to include the current active key

        Returns:
            Dictionary of key data
        """
        with self._lock:
            exported = {}
            for key_id, key in self._keys.items():
                if not include_current and key_id == self._current_key_id:
                    continue
                exported[key_id] = key.to_dict()
            return exported

    def import_keys(self, key_data: Dict[str, Dict[str, str]]):
        """
        Import keys from backup or trusted participant.

        Args:
            key_data: Dictionary of key data from export_keys()
        """
        with self._lock:
            imported_count = 0
            for key_id, data in key_data.items():
                if key_id not in self._keys:
                    self._keys[key_id] = EncryptionKey.from_dict(data)
                    imported_count += 1

            self._cleanup_old_keys()
            logger.info(f"Imported {imported_count} encryption keys")

    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status and warnings."""
        with self._lock:
            current_key = self._keys.get(self._current_key_id)

            status = {
                "active_keys": len(self._keys),
                "current_key_age": (
                    time.time() - current_key.created_at if current_key else 0
                ),
                "rotation_interval": self.key_rotation_seconds,
                "warnings": [],
            }

            # Check for warnings (trusted environment - warnings only)
            if current_key and current_key.is_expired(self.key_rotation_seconds):
                status["warnings"].append("Current key is expired but still usable")

            if len(self._keys) < 2:
                status["warnings"].append("Low key count - consider manual rotation")

            if status["current_key_age"] > self.key_rotation_seconds * 2:
                status["warnings"].append("Key rotation may have failed")

            return status


class SecureChannel:
    """
    Secure communication channel using dynamic encryption.

    Provides high-level interface for encrypted communication between
    federated learning participants.
    """

    def __init__(self, participant_id: str, shared_password: Optional[str] = None):
        """
        Initialize secure channel.

        Args:
            participant_id: Unique identifier for this participant
            shared_password: Shared password for key derivation (optional)
        """
        self.participant_id = participant_id
        self.encryption = DynamicEncryption(master_password=shared_password)

    def send_message(self, data: Any, recipient_id: str) -> Dict[str, Any]:
        """
        Prepare encrypted message for transmission.

        Args:
            data: Data to send
            recipient_id: Target participant ID

        Returns:
            Encrypted message package
        """
        message = {
            "sender_id": self.participant_id,
            "recipient_id": recipient_id,
            "timestamp": time.time(),
            "data": data,
        }

        encrypted_package = self.encryption.encrypt_json(message)
        encrypted_package["sender_id"] = self.participant_id
        encrypted_package["recipient_id"] = recipient_id

        return encrypted_package

    def receive_message(self, encrypted_package: Dict[str, Any]) -> Tuple[str, Any]:
        """
        Decrypt and validate received message.

        Args:
            encrypted_package: Encrypted message package

        Returns:
            Tuple of (sender_id, decrypted_data)
        """
        # Verify recipient
        if encrypted_package.get("recipient_id") != self.participant_id:
            raise ValueError("Message not intended for this participant")

        # Decrypt message
        message = self.encryption.decrypt_json(encrypted_package)

        # Validate message structure
        required_fields = ["sender_id", "recipient_id", "timestamp", "data"]
        for field in required_fields:
            if field not in message:
                raise ValueError(f"Invalid message format: missing {field}")

        return message["sender_id"], message["data"]

    def sync_keys_with_participant(self, other_encryption: DynamicEncryption):
        """
        Synchronize encryption keys with another participant.

        Args:
            other_encryption: Another participant's encryption system
        """
        # Export our keys (excluding current)
        our_keys = self.encryption.export_keys(include_current=False)
        their_keys = other_encryption.export_keys(include_current=False)

        # Import their keys
        self.encryption.import_keys(their_keys)
        other_encryption.import_keys(our_keys)

        logger.info(f"Synchronized keys with participant")


# Example usage and testing
if __name__ == "__main__":
    # Test basic encryption/decryption
    print("Testing dynamic encryption...")
    encryptor = DynamicEncryption()

    # Test string encryption
    test_data = "Hello, federated learning!"
    encrypted = encryptor.encrypt(test_data)
    decrypted = encryptor.decrypt(encrypted)
    print(f"Original: {test_data}")
    print(f"Decrypted: {decrypted}")
    print(f"Match: {test_data == decrypted}")

    # Test JSON encryption
    test_json = {"model_weights": [1.0, 2.0, 3.0], "participant_id": "user_1"}
    encrypted_json = encryptor.encrypt_json(test_json)
    decrypted_json = encryptor.decrypt_json(encrypted_json)
    print(f"JSON match: {test_json == decrypted_json}")

    # Test secure channel
    print("\nTesting secure channel...")
    channel1 = SecureChannel("participant_1")
    channel2 = SecureChannel("participant_2")

    # Sync keys
    channel1.encryption.import_keys(channel2.encryption.export_keys())
    channel2.encryption.import_keys(channel1.encryption.export_keys())

    # Send message
    message = channel1.send_message({"test": "data"}, "participant_2")
    sender_id, received_data = channel2.receive_message(message)
    print(f"Received from {sender_id}: {received_data}")

    # Security status
    print(f"\nSecurity status: {encryptor.get_security_status()}")
