"""Unit tests for the hashing utilities in tracking validation."""

import os
import pytest
from logic.src.tracking.validation.hashing import hash_bytes, hash_file


@pytest.mark.unit
def test_hash_bytes():
    data = b"hello world"
    # SHA-256 of "hello world" is:
    # b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9
    expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert hash_bytes(data) == expected


@pytest.mark.unit
def test_hash_file_exists(tmp_path):
    temp_file = tmp_path / "test.txt"
    temp_file.write_bytes(b"hello world")
    expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert hash_file(str(temp_file.resolve())) == expected


@pytest.mark.unit
def test_hash_file_not_exists():
    assert hash_file("non_existent_file.txt") is None
