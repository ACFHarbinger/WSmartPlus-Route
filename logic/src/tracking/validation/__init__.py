"""
Utility helpers for the WSTracker module.

Attributes:
    hash_file: Compute SHA-256 hash of a file on disk.
    hash_bytes: Compute SHA-256 hash of raw bytes.

Example:
    >>> from logic.src.tracking.validation import hash_file, hash_bytes
    >>> hash_file("data/test.pkl")
    'a1b2c3d4e5f6...'
    >>> hash_bytes(b"hello world")
    'b94d27b9934d...'
"""

from .hashing import hash_bytes, hash_file

__all__ = ["hash_file", "hash_bytes"]
