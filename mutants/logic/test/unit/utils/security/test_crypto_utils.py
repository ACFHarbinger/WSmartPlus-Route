"""Unit tests for security utils."""

import base64
import os
import pickle
from unittest.mock import MagicMock, patch, mock_open

import pytest
from cryptography.fernet import Fernet

from logic.src.utils.security import (
    generate_key, load_key, encrypt_file_data, decrypt_file_data,
    encode_data, encrypt_directory, decrypt_directory
)


def test_generate_key(mock_crypto_dotenv):
    """Test key generation."""
    with patch("logic.src.utils.security.keys.set_key") as mock_set:
        with patch("logic.src.utils.security.keys.open", mock_open()) as m:
            with patch("os.makedirs"):
                key, salt = generate_key(symkey_name="test_key")

                assert isinstance(key, bytes)
                assert isinstance(salt, bytes)
                assert len(key) > 0
                assert mock_set.call_count >= 3 # SALT, LEN, ITER


def test_load_key_from_env(mock_crypto_env):
    """Test loading key from environment."""
    with patch("logic.src.utils.security.keys.ROOT_DIR", "/tmp"):
        with patch("logic.src.utils.security.keys.load_dotenv"):
            key = load_key()
            assert isinstance(key, bytes)
            # Fernet key must be 32 base64-encoded bytes
            assert len(base64.urlsafe_b64decode(key)) == 32


def test_encryption_decryption_roundtrip():
    """Test encrypt/decrypt round trip for data bytes."""
    key = Fernet.generate_key()
    data = "Secret Message"

    enc = encrypt_file_data(key, data)
    assert enc != data

    dec = decrypt_file_data(key, enc)
    assert dec == "Secret Message"


def test_encode_data_types():
    """Test data encoding helper."""
    assert encode_data("string") == b"string"
    assert len(encode_data(12345)) > 0
    assert len(encode_data(3.14)) == 4

    test_dict = {"a": 1}
    assert encode_data(test_dict) == pickle.dumps(test_dict)


def test_encrypt_decrypt_file(tmp_path):
    """Test encrypting and decrypting a file."""
    key = Fernet.generate_key()

    input_file = tmp_path / "secret.txt"
    input_file.write_text("Hello World", encoding="utf-8")

    enc_file = tmp_path / "secret.enc"
    dec_file = tmp_path / "restored.txt"

    # Encrypt
    encrypt_file_data(key, str(input_file), str(enc_file))
    assert enc_file.exists()
    assert enc_file.read_bytes() != b"Hello World"

    # Decrypt
    res = decrypt_file_data(key, str(enc_file), str(dec_file))
    assert res == "Hello World"
    assert dec_file.read_text(encoding="utf-8") == "Hello World"


def test_encrypt_directory(tmp_path):
    """Test recursive directory encryption."""
    key = Fernet.generate_key()

    src_dir = tmp_path / "source"
    src_dir.mkdir()
    (src_dir / "f1.txt").write_text("File 1")
    (src_dir / "sub").mkdir()
    (src_dir / "sub" / "f2.txt").write_text("File 2")

    out_dir = tmp_path / "encrypted"

    encrypt_directory(key, src_dir, out_dir)

    assert (out_dir / "f1.txt.enc").exists()
    assert (out_dir / "sub" / "f2.txt.enc").exists()


def test_decrypt_directory(tmp_path):
    """Test recursive directory decryption."""
    key = Fernet.generate_key()

    # Setup encrypted structure
    enc_dir = tmp_path / "enc"
    enc_dir.mkdir()
    (enc_dir / "sub").mkdir()

    fernet = Fernet(key)
    (enc_dir / "f1.enc").write_bytes(fernet.encrypt(b"Content 1"))
    (enc_dir / "sub" / "f2.enc").write_bytes(fernet.encrypt(b"Content 2"))

    out_dir = tmp_path / "dec"
    decrypt_directory(key, enc_dir, out_dir)

    assert (out_dir / "f1").read_text() == "Content 1"
    assert (out_dir / "sub" / "f2").read_text() == "Content 2"
