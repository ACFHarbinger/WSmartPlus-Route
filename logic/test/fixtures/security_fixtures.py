"""
Fixtures for security and cryptography tests.
"""

import base64
import os
import pytest
from unittest.mock import patch

@pytest.fixture
def mock_crypto_env():
    """Mock environment variables for crypto tests."""
    with patch.dict(os.environ, {
        "KEY_PASSWORD": "test_password",
        "SALT_STRING": base64.urlsafe_b64encode(os.urandom(16)).decode("utf-8"),
        "KEY_LENGTH": "32",
        "HASH_ITERATIONS": "1000",
        "ROOT_DIR": "/tmp/root",
    }):
        yield

@pytest.fixture
def mock_crypto_dotenv():
    """Mock dotenv_values for crypto tests."""
    with patch("logic.src.utils.security.crypto_utils.dotenv_values") as mock:
        mock.return_value = {
            "KEY_PASSWORD": "test_password",
            "SALT_SIZE": "16",
            "KEY_LENGTH": "32",
            "HASH_ITERATIONS": "1000",
        }
        yield mock
