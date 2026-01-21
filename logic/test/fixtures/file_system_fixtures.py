"""
Fixtures for file system and cryptography unit tests.
"""
import pytest


@pytest.fixture
def fs_update_dir_opts():
    """Options for updating a directory."""
    return {
        "target_entry": "some_dir",
        "update_preview": False,
        "filename_pattern": "*.json",
        "output_key": "k",
        "update_operation": "add",
        "update_value": 1,
        "input_keys": (None, None),
    }


@pytest.fixture
def fs_update_file_opts():
    """Options for updating a file."""
    return {
        "target_entry": "some_file.json",
        "update_preview": False,
        "output_key": "k",
        "update_operation": "add",
        "update_value": 1,
        "input_keys": (None, None),
    }


@pytest.fixture
def fs_delete_opts():
    """Options for deleting entries."""
    return {"wandb": True, "log": True, "log_dir": "logs", "output": False}


@pytest.fixture
def fs_crypto_gen_opts():
    """Options for generating keys."""
    return {
        "symkey_name": "testkey",
        "salt_size": 16,
        "key_length": 32,
        "hash_iterations": 1000,
        "env_file": ".env",
    }


@pytest.fixture
def fs_crypto_encrypt_opts():
    """Options for encryption."""
    return {"input_path": "file.txt", "symkey_name": "testkey", "env_file": ".env"}
