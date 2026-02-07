"""
Unit tests for file system and cryptographic operations.
"""

from unittest.mock import mock_open, patch

from logic.src.file_system import (
    delete_file_system_entries,
    perform_cryptographic_operations,
    update_file_system_entries,
)


class TestFileSystem:
    """Test suite for file system management logic."""

    @patch("logic.src.file_system.process_pattern_files")
    @patch("logic.src.file_system.process_pattern_files_statistics")
    @patch("logic.src.file_system.preview_changes")
    @patch("logic.src.file_system.confirm_proceed", return_value=True)
    @patch("os.path.isdir", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_update_directory(
        self,
        mock_exists,
        mock_isdir,
        mock_confirm,
        mock_preview,
        mock_stats,
        mock_process,
        fs_update_dir_opts,
    ):
        """Test directory-level file system updates."""
        res = update_file_system_entries(fs_update_dir_opts)
        assert res == 1
        # mock_process is not name-bound here if not in signature, but we removed it to fix F841
        # Let's check the signature.

    @patch("logic.src.file_system.process_file")
    @patch("logic.src.file_system.process_file_statistics")
    @patch("logic.src.file_system.preview_file_changes")
    @patch("logic.src.file_system.confirm_proceed", return_value=True)
    @patch("os.path.isdir", return_value=False)
    @patch("os.path.isfile", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_update_file(
        self,
        mock_exists,
        mock_isfile,
        mock_isdir,
        mock_confirm,
        mock_preview,
        mock_stats,
        mock_process,
        fs_update_file_opts,
    ):
        """Test single-file file system updates."""
        res = update_file_system_entries(fs_update_file_opts)
        assert res == 1
        mock_process.assert_called_once()

    @patch("shutil.rmtree")
    @patch("os.path.exists", return_value=True)
    @patch("logic.src.file_system.confirm_proceed", return_value=True)
    def test_delete_entries(self, mock_confirm, mock_exists, mock_rmtree, fs_delete_opts):
        """Test deletion of file system entries (wandb, logs)."""
        delete_file_system_entries(fs_delete_opts)
        # Should delete wandb and logs
        assert mock_rmtree.call_count == 2

    @patch("logic.src.file_system.generate_key")
    def test_crypto_gen_key(self, mock_gen, fs_crypto_gen_opts):
        """Test cryptographic key generation."""
        mock_gen.return_value = (None, None)
        perform_cryptographic_operations(fs_crypto_gen_opts)
        mock_gen.assert_called_once()

    @patch("logic.src.file_system.load_key")
    @patch("logic.src.file_system.encrypt_file_data")
    @patch("logic.src.file_system.decrypt_file_data")
    @patch("builtins.open", new_callable=mock_open, read_data="data")
    def test_crypto_encrypt(self, mock_open, mock_decrypt, mock_encrypt, mock_load, fs_crypto_encrypt_opts):
        """Test file encryption and decryption operations."""
        # mock decrypt to return same data as read from file
        mock_decrypt.return_value = "data"

        perform_cryptographic_operations(fs_crypto_encrypt_opts)
        mock_encrypt.assert_called_once()
        mock_decrypt.assert_called_once()

    def test_update_directory_preview(self):
        """Test preview functionality for directory updates."""
        opts = {
            "target_entry": "test_dir",
            "update_preview": True,
            "filename_pattern": "*.txt",
            "output_key": "out",
            "update_operation": "add",
            "update_value": "test",
            "input_keys": [],
            "stats_function": None,
        }
        with (
            patch("logic.src.file_system.os.path.isdir", return_value=True),
            patch("logic.src.file_system.preview_pattern_files_statistics"),
            patch("logic.src.file_system.preview_changes") as mock_preview,
            patch("logic.src.file_system.confirm_proceed", return_value=True),
            patch("logic.src.file_system.process_pattern_files"),
        ):
            res = update_file_system_entries(opts)
            assert res == 1
            mock_preview.assert_called()

    def test_update_directory_stats(self):
        """Test statistics processing for directory updates."""
        opts = {
            "target_entry": "test_dir",
            "update_preview": False,
            "filename_pattern": "*.txt",
            "output_key": "out",
            "output_filename": "stats.json",
            "stats_function": lambda x: x,
        }
        with (
            patch("logic.src.file_system.os.path.isdir", return_value=True),
            patch("logic.src.file_system.process_pattern_files_statistics") as mock_process,
        ):
            res = update_file_system_entries(opts)
            assert res == 1
            mock_process.assert_called()
