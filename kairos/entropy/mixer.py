"""Cryptographic mixer: SHA3-256 entropy combining and HKDF token derivation."""
import hashlib
import os
import time

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


class CryptoMixer:
    """
    Mixes entropy from multiple chaos sources using SHA3-256 + OS randomness.
    Derives tokens via HKDF-SHA256.
    """

    def mix(
        self,
        pendulum_bytes: bytes,
        lorenz_bytes: bytes,
        rd_bytes: bytes,
    ) -> bytes:
        """
        Combine chaos state bytes with OS entropy and timestamp.
        Returns a 32-byte SHA3-256 digest.
        """
        combined = (
            pendulum_bytes
            + lorenz_bytes
            + rd_bytes
            + os.urandom(32)
            + time.time_ns().to_bytes(8, "big")
        )
        return hashlib.sha3_256(combined).digest()

    def derive_token(self, pool_bytes: bytes, length: int) -> bytes:
        """
        Derive a token of `length` bytes from pool bytes using HKDF-SHA256.
        """
        if not pool_bytes:
            pool_bytes = os.urandom(32)
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=length,
            salt=os.urandom(16),
            info=b"kairos-token",
        )
        return hkdf.derive(pool_bytes)
