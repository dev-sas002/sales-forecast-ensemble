"""Tests for src.utils.hashing: hash_id determinism, None input, salt change."""

from __future__ import annotations

from src.utils.hashing import hash_id


def test_hash_id_deterministic() -> None:
    a = hash_id("customer_123")
    b = hash_id("customer_123")
    assert a == b and a is not None


def test_hash_id_none_returns_none() -> None:
    assert hash_id(None) is None


def test_hash_id_different_salt_different_output() -> None:
    h1 = hash_id("customer_123", salt="salt1")
    h2 = hash_id("customer_123", salt="salt2")
    assert h1 != h2


def test_hash_id_different_input_different_output() -> None:
    h1 = hash_id("customer_123")
    h2 = hash_id("customer_456")
    assert h1 != h2
