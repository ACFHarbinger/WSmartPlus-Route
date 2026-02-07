"""Unit tests for files.py."""

import os
import json
import zipfile
import threading
import signal
import pytest
from unittest.mock import patch, MagicMock
from logic.src.utils.io.files import (
    read_json,
    zip_directory,
    extract_zip,
    confirm_proceed,
    compose_dirpath
)

def test_read_json(tmp_path):
    """Test safe JSON reading."""
    path = str(tmp_path / "test.json")
    data = {"hello": "world"}
    with open(path, "w") as f:
        json.dump(data, f)

    loaded = read_json(path)
    assert loaded == data

    # Test with lock
    lock = threading.Lock()
    loaded_locked = read_json(path, lock=lock)
    assert loaded_locked == data

def test_zip_and_extract(tmp_path):
    """Test zipping and extracting directories."""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    f1 = in_dir / "file1.txt"
    f1.write_text("content1")

    zip_path = str(tmp_path / "test.zip")
    zip_directory(str(in_dir), zip_path)
    assert os.path.exists(zip_path)

    out_dir = tmp_path / "out"
    extract_zip(zip_path, str(out_dir))
    assert (out_dir / "file1.txt").read_text() == "content1"

@patch("builtins.input")
def test_confirm_proceed(mock_input):
    """Test confirmation prompt."""
    mock_input.return_value = "y"
    assert confirm_proceed() is True

    mock_input.return_value = "n"
    assert confirm_proceed() is False

    mock_input.return_value = "" # default
    assert confirm_proceed(default_no=True) is False
    assert confirm_proceed(default_no=False) is True

def test_compose_dirpath():
    """Test the directory path composition decorator."""
    @compose_dirpath
    def mock_fun(dir_path, extra=None):
        return dir_path, extra

    # Single nbins
    res, extra = mock_fun("home", 30, 50, "out", "area", extra="test")
    assert "home/assets/out/30_days/area_50" in res
    assert extra == "test"

    # Multi nbins (list)
    res_list, _ = mock_fun("home", 30, [20, 50], "out", "area")
    assert isinstance(res_list, list)
    assert len(res_list) == 2
    assert "area_20" in res_list[0]
    assert "area_50" in res_list[1]
