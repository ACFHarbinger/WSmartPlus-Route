"""
Utilities for data encryption and decryption using Fernet (symmetric key).

Handles key generation, loading, and file encryption/decryption.
"""

import base64
import os
import pickle
import struct
from pathlib import Path
from typing import Any, List, Tuple, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from dotenv import dotenv_values, load_dotenv, set_key

from .definitions import ROOT_DIR
from .io.files import extract_zip, zip_directory


def _set_param(config, param_name, param_value=None):
    if param_value is None:
        param_value = config.get(param_name.upper())
    if not param_value:
        raise ValueError(f"{param_name} not found in .env file.")
    return param_value


def generate_key(
    salt_size: int = 16,
    key_length: int = 32,
    hash_iterations: int = 100_000,
    symkey_name: str = None,
    env_filename: str = ".env",
) -> Tuple[bytes, bytes]:
    """
    Generate a key from a password using PBKDF2HMAC.

    Args:
        salt_size (int, optional): Size of the random salt. Defaults to 16.
        key_length (int, optional): Length of derived key. Defaults to 32.
        hash_iterations (int, optional): Number of hash iterations. Defaults to 100,000.
        symkey_name (str, optional): Name to save key parameters to file (assets/keys/).
        env_filename (str, optional): Environment file to read/write config. Defaults to ".env".

    Returns:
        tuple: (key, salt) bytes.

    Raises:
        ValueError: If configuration values cannot be found or parsed.
        Exception: If directory creation fails.
    """
    env_path = Path(os.path.join(ROOT_DIR, "env", env_filename))
    print(f"Filename: {env_path}")
    config = dotenv_values(env_path)
    salt_size = int(_set_param(config, "SALT_SIZE", salt_size))
    salt = os.urandom(salt_size)
    key_length = int(_set_param(config, "KEY_LENGTH", key_length))
    hash_iterations = int(_set_param(config, "HASH_ITERATIONS", hash_iterations))
    password = _set_param(config, "KEY_PASSWORD")
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=key_length,
        salt=salt,
        iterations=hash_iterations,
        backend=default_backend(),
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    salt_str = base64.urlsafe_b64encode(salt).decode("utf-8")
    set_key(env_path, "SALT_STRING", salt_str)
    set_key(env_path, "KEY_LENGTH", str(key_length))
    set_key(env_path, "HASH_ITERATIONS", str(hash_iterations))
    if symkey_name:
        dir_path = os.path.join(ROOT_DIR, "assets", "keys")
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception:
            raise Exception("directories to save output files do not exist and could not be created")

        with open(os.path.join(dir_path, f"{symkey_name}.salt"), "wb") as salt_file:
            salt_file.write(salt)

        kh_params = (key_length, hash_iterations)
        with open(os.path.join(dir_path, f"{symkey_name}.pkl"), "wb") as khp_file:
            pickle.dump(kh_params, khp_file, protocol=pickle.HIGHEST_PROTOCOL)
    return (key, salt)


def load_key(symkey_name: str = None, env_filename: str = ".env") -> bytes:
    """
    Loads a symmetric key from environment or file parameters.

    Args:
        symkey_name (str, optional): Name of the key to load parameters from.
        env_filename (str, optional): Environment file name. Defaults to ".env".

    Returns:
        bytes: The derived encryption key.

    Raises:
        ValueError: If password is not found.
    """
    env_path = Path(os.path.join(ROOT_DIR, "env", env_filename))
    load_dotenv(dotenv_path=env_path)
    password = os.getenv("KEY_PASSWORD")
    if not password:
        raise ValueError("Password not found in .env file.")

    if symkey_name:
        print(f"Loading salt from {symkey_name}.salt and key params from {symkey_name}.pkl")
        input_path = os.path.join(ROOT_DIR, "assets", "keys", symkey_name)
        with open(f"{input_path}.salt", "rb") as salt_file:
            salt = salt_file.read()

        with open(f"{input_path}.pkl", "rb") as khp_file:
            key_length, hash_iterations = pickle.load(khp_file)
    else:
        print(f"Loading salt and key params from environment variables in {env_filename}")
        salt = base64.urlsafe_b64decode(os.getenv("SALT_STRING").encode("utf-8"))
        key_length = int(os.getenv("KEY_LENGTH"))
        hash_iterations = int(os.getenv("HASH_ITERATIONS"))

    # Derive the key using the password and salt
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=key_length,
        salt=salt,
        iterations=hash_iterations,
        backend=default_backend(),
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key


def encode_data(data: Any) -> bytes:
    """
    Encodes various data types into bytes for encryption.

    Args:
        data (Any): Data to encode (str, int, float, list, dict, etc.).

    Returns:
        bytes: Encoded byte representation.
    """
    if isinstance(data, str):
        return data.encode("utf-8")
    elif isinstance(data, int):
        return data.to_bytes((data.bit_length() + 7) // 8, byteorder="big")
    elif isinstance(data, float):
        return struct.pack("!f", data)
    else:  # elif isinstance(data, list) or isinstance(data, dict):
        return pickle.dumps(data)


def encrypt_file_data(key: bytes, input: Union[os.PathLike, Any], output_file: os.PathLike = None) -> bytes:
    """
    Encrypt a file or data object using Fernet symmetric encryption.

    Args:
        key (bytes): The encryption key.
        input (Union[os.PathLike, Any]): Path to file OR data object to encrypt.
        output_file (os.PathLike, optional): Path to save encrypted data.

    Returns:
        bytes: The encrypted data.
    """
    fernet = Fernet(key)
    if os.path.isfile(input):
        with open(input, "rb") as f:
            original_data = f.read()
    else:
        original_data = encode_data(input)

    encrypted_data = fernet.encrypt(original_data)
    if output_file:
        with open(output_file, "wb") as f:
            f.write(encrypted_data)
    return encrypted_data


def decrypt_file_data(key: bytes, input: Union[os.PathLike, Any], output_file: os.PathLike = None) -> str:
    """
    Decrypt a file or data bytes using Fernet symmetric encryption.

    Args:
        key (bytes): The encryption key.
        input (Union[os.PathLike, Any]): Path to encrypted file OR encrypted bytes.
        output_file (os.PathLike, optional): Path to save decrypted content.

    Returns:
        str: The decrypted data (decoded as utf-8 string).
    """
    fernet = Fernet(key)
    if os.path.isfile(input):
        with open(input, "rb") as f:
            encrypted_data = f.read()
    else:
        encrypted_data = input

    decrypted_data = fernet.decrypt(encrypted_data).decode("utf-8")
    if output_file:
        with open(output_file, "w") as f:
            f.write(decrypted_data)
    return decrypted_data


def encrypt_directory(key: bytes, input_dir: os.PathLike, output_dir: os.PathLike = None) -> List[bytes]:
    """
    Encrypt all files in a directory recursively.

    Args:
        key (bytes): The encryption key.
        input_dir (os.PathLike): Directory to encrypt.
        output_dir (os.PathLike, optional): Output directory. Defaults to input_dir.

    Returns:
        list: List of encrypted data bytes for each file.

    Raises:
        Exception: If directory creation fails.
    """
    if output_dir is None:
        output_dir = input_dir
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        raise Exception("directories to save output files do not exist and could not be created")

    # Recursively process all files in the input directory
    encdata_ls = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            input_file = os.path.join(root, file)
            relative_path = os.path.relpath(input_file, input_dir)
            output_file = os.path.join(output_dir, relative_path + ".enc")

            # Create subdirectories in the output directory if they don't exist
            try:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
            except Exception:
                raise Exception("subdirectories to save output files do not exist and could not be created")
            encdata_ls.append(encrypt_file_data(input_file, output_file, key))
    return encdata_ls


def decrypt_directory(key: bytes, input_dir: os.PathLike, output_dir: os.PathLike = None) -> List[str]:
    """
    Decrypt all .enc files in a directory recursively.

    Args:
        key (bytes): The encryption key.
        input_dir (os.PathLike): Directory to decrypt.
        output_dir (os.PathLike, optional): Output directory. Defaults to input_dir.

    Returns:
        list: List of decrypted string data for each file.

    Raises:
        Exception: If directory creation fails.
    """
    if output_dir is None:
        output_dir = input_dir
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        raise Exception("directories to save output files do not exist and could not be created")

    # Recursively process all files in the input directory
    decdata_ls = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            input_file = os.path.join(root, file)
            file_path, file_ext = os.path.splitext(input_file)
            if file_ext == ".enc":
                relative_path = os.path.relpath(file_path, input_dir)
                output_file = os.path.join(output_dir, relative_path)

                # Create subdirectories in the output directory if they don't exist
                try:
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                except Exception:
                    raise Exception("subdirectories to save output files do not exist and could not be created")
                decdata_ls.append(decrypt_file_data(input_file, output_file, key))
    return decdata_ls


def encrypt_zip_directory(key: bytes, input_dir: os.PathLike, output_enczip: os.PathLike = None) -> bytes:
    """
    Zip a directory and then encrypt the resulting zip file.

    Args:
        key (bytes): The encryption key.
        input_dir (os.PathLike): Directory to zip and encrypt.
        output_enczip (os.PathLike, optional): Output path for the encrypted zip.

    Returns:
        bytes: The encrypted zip data.
    """
    if output_enczip is None:
        norm_path = os.path.normpath(input_dir)
        output_enczip = os.path.join(os.path.dirname(norm_path), f"{os.path.basename(input_dir)}.zip")
    tmp_zip = "{}.tmp.zip".format(output_enczip)
    zip_directory(input_dir, tmp_zip)
    encrypted_data = encrypt_file_data(tmp_zip, output_enczip, key)
    os.remove(tmp_zip)
    return encrypted_data


def decrypt_zip(key: bytes, input_enczip: os.PathLike, output_dir: os.PathLike = None) -> str:
    """
    Decrypt a zip file and extract its contents.

    Args:
        key (bytes): The encryption key.
        input_enczip (os.PathLike): Path to the encrypted zip file.
        output_dir (os.PathLike, optional): Directory to extract to.

    Returns:
        str: The decrypted data (raw string content of the zip file).
    """
    if output_dir is None:
        norm_path = os.path.normpath(input_enczip)
        output_dir, _ = os.path.splitext(norm_path)
    tmp_zip = input_enczip + ".tmp.zip"
    decrypted_data = decrypt_file_data(input_enczip, tmp_zip, key)
    extract_zip(tmp_zip, output_dir)
    os.remove(tmp_zip)
    return decrypted_data


if __name__ == "__main__":
    env_filename = "vars.env"
    input_file = "hexaly.dat"
    output_file = "hexaly.dat.enc"
    salt_size = 16
    key_length = 32
    hash_iterations = 100_000
    symkey_name = "skey"
    # _, salt = generate_key(salt_size, key_length, hash_iterations, symkey_name)
    key = load_key(symkey_name, env_filename)
    inpath = os.path.join(ROOT_DIR, "assets", "api", input_file)
    outpath = os.path.join(ROOT_DIR, "assets", "api", output_file)
    enc_data = encrypt_file_data(key, inpath, outpath)
    dec_data = decrypt_file_data(key, outpath)
    with open(inpath, "r") as gp_file:
        data = gp_file.read()
    assert dec_data == data
