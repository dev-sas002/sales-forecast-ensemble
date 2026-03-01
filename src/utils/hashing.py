"""HMAC-SHA256 hashing for privacy-preserving customer IDs (salt ensures no plaintext lookup)."""

import hashlib
import hmac

from src.config import SALT


def hash_id(raw_id: str | None, salt: str = SALT) -> str | None:
    """Stable HMAC-SHA256 hash for privacy-preserving IDs. Salt as str is encoded to bytes."""
    if raw_id is None:
        return None
    return hmac.new(salt.encode(), str(raw_id).encode(), hashlib.sha256).hexdigest()
