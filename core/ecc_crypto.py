"""
Vega2.0 - Elliptic Curve Cryptography (ECC) System
==================================================

Comprehensive ECC implementation for secure key generation, digital signatures,
and encryption across all sensitive operations.

Features:
- Key generation and management
- Digital signatures (ECDSA)
- Key exchange (ECDH)
- Message encryption/decryption
- Certificate management
- Secure storage
"""

import os
import json
import base64
import hashlib
from typing import Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    from cryptography import x509
    from cryptography.x509.oid import NameOID

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False


class ECCCurve:
    """Supported ECC curves"""

    SECP256R1 = "secp256r1"  # NIST P-256
    SECP384R1 = "secp384r1"  # NIST P-384
    SECP521R1 = "secp521r1"  # NIST P-521
    SECP256K1 = "secp256k1"  # Bitcoin curve


@dataclass
class ECCKeyPair:
    """ECC key pair representation"""

    private_key_pem: str
    public_key_pem: str
    curve: str
    key_id: str
    created_at: datetime
    expires_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "private_key_pem": self.private_key_pem,
            "public_key_pem": self.public_key_pem,
            "curve": self.curve,
            "key_id": self.key_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ECCKeyPair":
        """Create from dictionary"""
        return cls(
            private_key_pem=data["private_key_pem"],
            public_key_pem=data["public_key_pem"],
            curve=data["curve"],
            key_id=data["key_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=(
                datetime.fromisoformat(data["expires_at"])
                if data.get("expires_at")
                else None
            ),
        )


@dataclass
class ECCSignature:
    """ECC digital signature"""

    signature: str  # Base64 encoded
    algorithm: str
    key_id: str
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "signature": self.signature,
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ECCCertificate:
    """ECC certificate representation"""

    certificate_pem: str
    private_key_pem: str
    subject: str
    issuer: str
    serial_number: str
    not_before: datetime
    not_after: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "certificate_pem": self.certificate_pem,
            "private_key_pem": self.private_key_pem,
            "subject": self.subject,
            "issuer": self.issuer,
            "serial_number": self.serial_number,
            "not_before": self.not_before.isoformat(),
            "not_after": self.not_after.isoformat(),
        }


class ECCManager:
    """ECC key and cryptographic operations manager"""

    def __init__(self, key_store_path: str = "/tmp/vega_keys"):
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("cryptography library is required for ECC operations")

        self.key_store_path = Path(key_store_path)
        self.key_store_path.mkdir(exist_ok=True, mode=0o700)
        self.keys: Dict[str, ECCKeyPair] = {}
        self.certificates: Dict[str, ECCCertificate] = {}
        self._load_keys()

    def _get_curve_object(self, curve: str):
        """Get cryptography curve object"""
        curve_map = {
            ECCCurve.SECP256R1: ec.SECP256R1(),
            ECCCurve.SECP384R1: ec.SECP384R1(),
            ECCCurve.SECP521R1: ec.SECP521R1(),
            ECCCurve.SECP256K1: ec.SECP256K1(),
        }

        if curve not in curve_map:
            raise ValueError(f"Unsupported curve: {curve}")

        return curve_map[curve]

    def generate_key_pair(
        self,
        curve: str = ECCCurve.SECP256R1,
        key_id: Optional[str] = None,
        expires_in_days: Optional[int] = None,
    ) -> ECCKeyPair:
        """Generate a new ECC key pair"""

        if key_id is None:
            key_id = self._generate_key_id()

        # Generate private key
        curve_obj = self._get_curve_object(curve)
        private_key = ec.generate_private_key(curve_obj, default_backend())
        public_key = private_key.public_key()

        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("utf-8")

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")

        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        # Create key pair object
        key_pair = ECCKeyPair(
            private_key_pem=private_pem,
            public_key_pem=public_pem,
            curve=curve,
            key_id=key_id,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
        )

        # Store key pair
        self.keys[key_id] = key_pair
        self._save_key(key_pair)

        return key_pair

    def get_key_pair(self, key_id: str) -> Optional[ECCKeyPair]:
        """Get key pair by ID"""
        return self.keys.get(key_id)

    def list_keys(self) -> Dict[str, ECCKeyPair]:
        """List all key pairs"""
        return self.keys.copy()

    def delete_key(self, key_id: str) -> bool:
        """Delete a key pair"""
        if key_id not in self.keys:
            return False

        # Remove from memory
        del self.keys[key_id]

        # Remove from storage
        key_file = self.key_store_path / f"{key_id}.json"
        if key_file.exists():
            key_file.unlink()

        return True

    def sign_data(self, data: Union[str, bytes], key_id: str) -> ECCSignature:
        """Sign data with ECC private key"""
        key_pair = self.get_key_pair(key_id)
        if not key_pair:
            raise ValueError(f"Key not found: {key_id}")

        # Convert data to bytes
        if isinstance(data, str):
            data = data.encode("utf-8")

        # Load private key
        private_key = serialization.load_pem_private_key(
            key_pair.private_key_pem.encode("utf-8"),
            password=None,
            backend=default_backend(),
        )

        # Sign data
        signature = private_key.sign(data, ec.ECDSA(hashes.SHA256()))

        # Encode signature
        signature_b64 = base64.b64encode(signature).decode("utf-8")

        return ECCSignature(
            signature=signature_b64,
            algorithm="ECDSA-SHA256",
            key_id=key_id,
            timestamp=datetime.utcnow(),
        )

    def verify_signature(
        self,
        data: Union[str, bytes],
        signature: ECCSignature,
        public_key_pem: Optional[str] = None,
    ) -> bool:
        """Verify ECC signature"""

        # Convert data to bytes
        if isinstance(data, str):
            data = data.encode("utf-8")

        # Get public key
        if public_key_pem:
            public_key = serialization.load_pem_public_key(
                public_key_pem.encode("utf-8"), backend=default_backend()
            )
        else:
            key_pair = self.get_key_pair(signature.key_id)
            if not key_pair:
                return False

            public_key = serialization.load_pem_public_key(
                key_pair.public_key_pem.encode("utf-8"), backend=default_backend()
            )

        # Decode signature
        try:
            sig_bytes = base64.b64decode(signature.signature)
        except Exception:
            return False

        # Verify signature
        try:
            public_key.verify(sig_bytes, data, ec.ECDSA(hashes.SHA256()))
            return True
        except Exception:
            return False

    def derive_shared_secret(self, key_id: str, peer_public_key_pem: str) -> bytes:
        """Derive shared secret using ECDH"""
        key_pair = self.get_key_pair(key_id)
        if not key_pair:
            raise ValueError(f"Key not found: {key_id}")

        # Load private key
        private_key = serialization.load_pem_private_key(
            key_pair.private_key_pem.encode("utf-8"),
            password=None,
            backend=default_backend(),
        )

        # Load peer public key
        peer_public_key = serialization.load_pem_public_key(
            peer_public_key_pem.encode("utf-8"), backend=default_backend()
        )

        # Derive shared secret
        shared_key = private_key.exchange(ec.ECDH(), peer_public_key)

        return shared_key

    def encrypt_message(
        self,
        message: Union[str, bytes],
        recipient_public_key_pem: str,
        sender_key_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """Encrypt message using ECIES (Elliptic Curve Integrated Encryption Scheme)"""

        if isinstance(message, str):
            message = message.encode("utf-8")

        # Generate ephemeral key pair if no sender key specified
        if sender_key_id is None:
            ephemeral_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
            ephemeral_public_key = ephemeral_key.public_key()
        else:
            key_pair = self.get_key_pair(sender_key_id)
            if not key_pair:
                raise ValueError(f"Sender key not found: {sender_key_id}")

            ephemeral_key = serialization.load_pem_private_key(
                key_pair.private_key_pem.encode("utf-8"),
                password=None,
                backend=default_backend(),
            )
            ephemeral_public_key = ephemeral_key.public_key()

        # Load recipient public key
        recipient_public_key = serialization.load_pem_public_key(
            recipient_public_key_pem.encode("utf-8"), backend=default_backend()
        )

        # Derive shared secret
        shared_key = ephemeral_key.exchange(ec.ECDH(), recipient_public_key)

        # Derive encryption key using HKDF
        derived_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"encryption",
            backend=default_backend(),
        ).derive(shared_key)

        # Generate random IV
        iv = os.urandom(16)

        # Encrypt message
        cipher = Cipher(
            algorithms.AES(derived_key), modes.CTR(iv), backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(message) + encryptor.finalize()

        # Serialize ephemeral public key
        ephemeral_public_pem = ephemeral_public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")

        return {
            "ciphertext": base64.b64encode(ciphertext).decode("utf-8"),
            "iv": base64.b64encode(iv).decode("utf-8"),
            "ephemeral_public_key": ephemeral_public_pem,
            "algorithm": "ECIES-AES256-CTR",
        }

    def decrypt_message(
        self, encrypted_data: Dict[str, str], recipient_key_id: str
    ) -> bytes:
        """Decrypt message using ECIES"""

        key_pair = self.get_key_pair(recipient_key_id)
        if not key_pair:
            raise ValueError(f"Recipient key not found: {recipient_key_id}")

        # Load recipient private key
        private_key = serialization.load_pem_private_key(
            key_pair.private_key_pem.encode("utf-8"),
            password=None,
            backend=default_backend(),
        )

        # Load ephemeral public key
        ephemeral_public_key = serialization.load_pem_public_key(
            encrypted_data["ephemeral_public_key"].encode("utf-8"),
            backend=default_backend(),
        )

        # Derive shared secret
        shared_key = private_key.exchange(ec.ECDH(), ephemeral_public_key)

        # Derive encryption key using HKDF
        derived_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"encryption",
            backend=default_backend(),
        ).derive(shared_key)

        # Decode encrypted data
        ciphertext = base64.b64decode(encrypted_data["ciphertext"])
        iv = base64.b64decode(encrypted_data["iv"])

        # Decrypt message
        cipher = Cipher(
            algorithms.AES(derived_key), modes.CTR(iv), backend=default_backend()
        )
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()

        return plaintext

    def generate_certificate(
        self,
        key_id: str,
        subject_name: str,
        issuer_name: Optional[str] = None,
        validity_days: int = 365,
    ) -> ECCCertificate:
        """Generate X.509 certificate for key pair"""

        key_pair = self.get_key_pair(key_id)
        if not key_pair:
            raise ValueError(f"Key not found: {key_id}")

        # Load private key
        private_key = serialization.load_pem_private_key(
            key_pair.private_key_pem.encode("utf-8"),
            password=None,
            backend=default_backend(),
        )

        public_key = private_key.public_key()

        # Create certificate builder
        builder = x509.CertificateBuilder()

        # Set subject and issuer
        subject = x509.Name(
            [
                x509.NameAttribute(NameOID.COMMON_NAME, subject_name),
            ]
        )
        builder = builder.subject_name(subject)

        if issuer_name:
            issuer = x509.Name(
                [
                    x509.NameAttribute(NameOID.COMMON_NAME, issuer_name),
                ]
            )
        else:
            issuer = subject  # Self-signed

        builder = builder.issuer_name(issuer)

        # Set validity period
        not_before = datetime.utcnow()
        not_after = not_before + timedelta(days=validity_days)
        builder = builder.not_valid_before(not_before)
        builder = builder.not_valid_after(not_after)

        # Set serial number
        builder = builder.serial_number(x509.random_serial_number())

        # Set public key
        builder = builder.public_key(public_key)

        # Sign certificate
        certificate = builder.sign(private_key, hashes.SHA256(), default_backend())

        # Serialize certificate
        cert_pem = certificate.public_bytes(serialization.Encoding.PEM).decode("utf-8")

        cert_obj = ECCCertificate(
            certificate_pem=cert_pem,
            private_key_pem=key_pair.private_key_pem,
            subject=subject_name,
            issuer=issuer_name or subject_name,
            serial_number=str(certificate.serial_number),
            not_before=not_before,
            not_after=not_after,
        )

        # Store certificate
        self.certificates[key_id] = cert_obj
        self._save_certificate(key_id, cert_obj)

        return cert_obj

    def get_certificate(self, key_id: str) -> Optional[ECCCertificate]:
        """Get certificate for key"""
        return self.certificates.get(key_id)

    def verify_certificate(
        self, cert_pem: str, ca_cert_pem: Optional[str] = None
    ) -> bool:
        """Verify certificate validity"""
        try:
            cert = x509.load_pem_x509_certificate(
                cert_pem.encode("utf-8"), default_backend()
            )

            # Check expiration
            now = datetime.utcnow()
            if now < cert.not_valid_before or now > cert.not_valid_after:
                return False

            # Verify signature if CA provided
            if ca_cert_pem:
                ca_cert = x509.load_pem_x509_certificate(
                    ca_cert_pem.encode("utf-8"), default_backend()
                )
                ca_public_key = ca_cert.public_key()

                try:
                    ca_public_key.verify(
                        cert.signature,
                        cert.tbs_certificate_bytes,
                        ec.ECDSA(hashes.SHA256()),
                    )
                except Exception:
                    return False

            return True

        except Exception:
            return False

    def _generate_key_id(self) -> str:
        """Generate unique key ID"""
        import uuid

        return f"vega-ecc-{uuid.uuid4().hex[:16]}"

    def _save_key(self, key_pair: ECCKeyPair):
        """Save key pair to storage"""
        key_file = self.key_store_path / f"{key_pair.key_id}.json"
        with open(key_file, "w", mode=0o600) as f:
            json.dump(key_pair.to_dict(), f, indent=2)

    def _save_certificate(self, key_id: str, certificate: ECCCertificate):
        """Save certificate to storage"""
        cert_file = self.key_store_path / f"{key_id}_cert.json"
        with open(cert_file, "w") as f:
            json.dump(certificate.to_dict(), f, indent=2)

    def _load_keys(self):
        """Load keys from storage"""
        if not self.key_store_path.exists():
            return

        for key_file in self.key_store_path.glob("*.json"):
            if key_file.name.endswith("_cert.json"):
                continue

            try:
                with open(key_file, "r") as f:
                    key_data = json.load(f)

                key_pair = ECCKeyPair.from_dict(key_data)
                self.keys[key_pair.key_id] = key_pair

                # Load certificate if exists
                cert_file = self.key_store_path / f"{key_pair.key_id}_cert.json"
                if cert_file.exists():
                    with open(cert_file, "r") as f:
                        cert_data = json.load(f)

                    certificate = ECCCertificate(
                        certificate_pem=cert_data["certificate_pem"],
                        private_key_pem=cert_data["private_key_pem"],
                        subject=cert_data["subject"],
                        issuer=cert_data["issuer"],
                        serial_number=cert_data["serial_number"],
                        not_before=datetime.fromisoformat(cert_data["not_before"]),
                        not_after=datetime.fromisoformat(cert_data["not_after"]),
                    )
                    self.certificates[key_pair.key_id] = certificate

            except Exception as e:
                print(f"Warning: Could not load key file {key_file}: {e}")

    def export_public_key(self, key_id: str) -> Optional[str]:
        """Export public key in PEM format"""
        key_pair = self.get_key_pair(key_id)
        if key_pair:
            return key_pair.public_key_pem
        return None

    def import_public_key(
        self, public_key_pem: str, key_id: Optional[str] = None
    ) -> str:
        """Import external public key"""
        if key_id is None:
            key_id = self._generate_key_id()

        # Validate public key
        try:
            serialization.load_pem_public_key(
                public_key_pem.encode("utf-8"), backend=default_backend()
            )
        except Exception as e:
            raise ValueError(f"Invalid public key: {e}")

        # Store as key pair with only public key
        key_pair = ECCKeyPair(
            private_key_pem="",  # No private key for imported public keys
            public_key_pem=public_key_pem,
            curve="unknown",  # Could be detected from key
            key_id=key_id,
            created_at=datetime.utcnow(),
        )

        self.keys[key_id] = key_pair
        self._save_key(key_pair)

        return key_id

    def get_key_info(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed key information"""
        key_pair = self.get_key_pair(key_id)
        if not key_pair:
            return None

        info = {
            "key_id": key_pair.key_id,
            "curve": key_pair.curve,
            "created_at": key_pair.created_at.isoformat(),
            "expires_at": (
                key_pair.expires_at.isoformat() if key_pair.expires_at else None
            ),
            "has_private_key": bool(key_pair.private_key_pem),
            "has_certificate": key_id in self.certificates,
        }

        # Check if expired
        if key_pair.expires_at:
            info["expired"] = datetime.utcnow() > key_pair.expires_at
        else:
            info["expired"] = False

        return info


# Global ECC manager instance
_ecc_manager = None


def get_ecc_manager() -> ECCManager:
    """Get global ECC manager instance"""
    global _ecc_manager
    if _ecc_manager is None:
        _ecc_manager = ECCManager()
    return _ecc_manager


# Convenience functions
def generate_key_pair(
    curve: str = ECCCurve.SECP256R1, key_id: Optional[str] = None
) -> ECCKeyPair:
    """Generate ECC key pair"""
    return get_ecc_manager().generate_key_pair(curve, key_id)


def sign_data(data: Union[str, bytes], key_id: str) -> ECCSignature:
    """Sign data with ECC key"""
    return get_ecc_manager().sign_data(data, key_id)


def verify_signature(
    data: Union[str, bytes],
    signature: ECCSignature,
    public_key_pem: Optional[str] = None,
) -> bool:
    """Verify ECC signature"""
    return get_ecc_manager().verify_signature(data, signature, public_key_pem)


def encrypt_message(
    message: Union[str, bytes], recipient_public_key_pem: str
) -> Dict[str, str]:
    """Encrypt message using ECC"""
    return get_ecc_manager().encrypt_message(message, recipient_public_key_pem)


def decrypt_message(encrypted_data: Dict[str, str], recipient_key_id: str) -> bytes:
    """Decrypt message using ECC"""
    return get_ecc_manager().decrypt_message(encrypted_data, recipient_key_id)
