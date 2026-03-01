"""Unit tests for files.py."""

import json
import os
import threading
from typing import Any, cast
from unittest.mock import patch

from logic.src.utils.io.files import compose_dirpath, confirm_proceed, extract_zip, read_json, zip_directory


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
    res, extra_val = cast(Any, mock_fun)("home", 30, 50, "out", "area", "test")
    assert "home/assets/out/30_days/area_50" in res
    assert extra_val == "test"

    # Multi nbins (list)
    res_list, _ = cast(Any, mock_fun)("home", 30, [20, 50], "out", "area")
    assert isinstance(res_list, list)
    assert len(res_list) == 2
    assert "area_20" in res_list[0]
    assert "area_50" in res_list[1]
